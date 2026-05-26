import pandas as pd

df = pd.read_csv('data/metadata.csv', sep=';', encoding='utf-8-sig')
print(f'Total rows: {len(df)}')
print(f'Unique filenames: {df["filename"].nunique()}')

duplicates = df[df.duplicated(subset=['filename'], keep=False)].sort_values('filename')
print(f'\nDuplicate entries found: {len(duplicates)}')

if len(duplicates) > 0:
    print('\nDuplicate filenames:')
    for fname in sorted(duplicates['filename'].unique()):
        count = len(df[df['filename'] == fname])
        print(f'  {fname}: {count} entries')
        # Show all rows for this filename
        rows = df[df['filename'] == fname]
        for idx, row in rows.iterrows():
            print(f'    Row {idx}: {row.to_dict()}')
else:
    print('\n✓ No duplicates found!')
