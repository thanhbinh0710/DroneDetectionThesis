"""Export misclassified source audio files for manual review.

This script loads the processed dataset, reproduces the group-based test split,
runs the trained model on the test set, identifies misclassified samples, and
copies the corresponding source .wav files to an output folder.

Usage:
    python scripts/export_misclassified.py

Optional:
    python scripts/export_misclassified.py --output-dir data/review/misclassified --max-files 16
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf

# Ensure project imports work when running the script directly from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load data function from your pipeline
from src.training.train import load_processed_data


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    return pd.read_csv(metadata_path, sep=';', encoding='utf-8-sig')


def build_model_path(project_root: Path) -> Path:
    return project_root / 'models' / 'drone_model_v1.keras'


def export_files(file_paths: list[Path], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_paths = []

    for source_path in file_paths:
        destination_path = output_dir / source_path.name
        shutil.copy2(source_path, destination_path)
        exported_paths.append(destination_path)

    return exported_paths


def main() -> int:
    parser = argparse.ArgumentParser(description='Export misclassified source audio files for review.')
    parser.add_argument(
        '--output-dir',
        default='data/review/misclassified',
        help='Directory to copy misclassified audio files into.',
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=16,
        help='Maximum number of unique misclassified source files to export.',
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    processed_dir = project_root / 'data' / 'processed'
    raw_dir = project_root / 'data' / 'raw'
    metadata_path = project_root / 'data' / 'metadata.csv'
    model_path = build_model_path(project_root)
    output_dir = project_root / args.output_dir

    print('=' * 60)
    print('EXPORT MISCLASSIFIED AUDIO')
    print('=' * 60)
    print(f'Project root: {project_root}')
    print(f'Model: {model_path}')
    print(f'Processed data: {processed_dir}')
    print(f'Raw audio: {raw_dir}')
    print(f'Output dir: {output_dir}')

    if not model_path.exists():
        print(f'Error: model not found: {model_path}')
        return 1

    # 1. Load toàn bộ dữ liệu đã qua xử lý
    X, y, groups = load_processed_data(str(processed_dir))
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    # 2. Tái tạo quá trình chia Group Split chính xác như trong file train.py
    # 2.1: Tách Train (70%) và Temp (30%)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss1.split(X, y, groups))

    # 2.2: Tách Temp (30%) thành Val (15%) và Test (15%) -> Tức là chia đôi Temp (test_size=0.5)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx_relative, test_idx_relative = next(gss2.split(X[temp_idx], y[temp_idx], groups[temp_idx]))

    # Lấy index tuyệt đối của tập Test để map ngược lại mảng gốc
    absolute_test_idx = temp_idx[test_idx_relative]

    # Đây là 3 mảng hoàn toàn đồng nhất với nhau
    X_test = X[absolute_test_idx]
    y_test = y[absolute_test_idx]
    groups_test = groups[absolute_test_idx]

    # 3. Chạy model dự đoán
    model = tf.keras.models.load_model(model_path)
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Tìm index của các đoạn audio (segment) bị đoán sai
    misclassified_mask = y_pred != y_test
    test_sample_indices = np.where(misclassified_mask)[0]

    if len(test_sample_indices) == 0:
        print('No misclassified samples found in the test split.')
        return 0

    metadata = load_metadata(metadata_path)
    raw_source_paths = []
    source_rows = []
    unique_misclassified_groups = []

    # 4. Truy vết ngược (Traceback) an toàn tuyệt đối
    for sample_idx in test_sample_indices:
        # Nhờ mảng groups_test, ta biết chính xác đoạn audio này lấy từ file gốc nào
        group_id = int(groups_test[sample_idx])
        if group_id not in unique_misclassified_groups:
            unique_misclassified_groups.append(group_id)

    # 5. Lấy file gốc tương ứng từ metadata
    for group_id in unique_misclassified_groups:
        if group_id < 0 or group_id >= len(metadata):
            continue
            
        filename = metadata.iloc[group_id]['filename']
        source_path = raw_dir / filename
        
        if source_path.exists():
            raw_source_paths.append(source_path)
            source_rows.append((group_id, filename))

    if not raw_source_paths:
        print('No source audio files were resolved from misclassified samples.')
        return 1

    # 6. Giới hạn số lượng file cần xuất (nếu cần)
    if args.max_files > 0:
        raw_source_paths = raw_source_paths[: args.max_files]
        source_rows = source_rows[: args.max_files]

    # 7. Copy file sang thư mục review
    exported_paths = export_files(raw_source_paths, output_dir)

    list_path = output_dir / 'misclassified_files.txt'
    with list_path.open('w', encoding='utf-8') as f:
        for group_id, filename in source_rows:
            f.write(f'{group_id};{filename}\n')

    print('\nExport summary:')
    print(f'  - Misclassified unique source files: {len(source_rows)}')
    print(f'  - Exported files: {len(exported_paths)}')
    print(f'  - List file: {list_path}')
    for path in exported_paths:
        print(f'    * {path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())