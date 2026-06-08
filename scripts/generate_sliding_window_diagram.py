"""
Generate Sliding Window illustration figure for thesis/report.
Panel (a): Conceptual diagram with synthetic signal
Panel (b): Real waveform from dataset with segment overlay
Panel (c): Zoom detail of overlap region between two segments

Usage: python -m scripts.generate_sliding_window_diagram
Output: assets/Figure_Sliding_Window.png
"""

import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

sr = 16000
segment_duration = 1.0
overlap = 0.5
hop = segment_duration * (1 - overlap)
segment_samples = int(segment_duration * sr)
hop_samples = int(segment_samples * (1 - overlap))

segment_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
                  '#937860', '#DA8BC3', '#8C8C8C']

audio_path = os.path.join(project_root, 'data', 'raw', 'drone', 'DRONE_001.wav')
display_duration = 3.0

y, _ = librosa.load(audio_path, sr=sr, duration=display_duration)


def segment_audio(audio, sr=16000, segment_duration=1.0, overlap=0.5):
    seg_samples = int(segment_duration * sr)
    hop_samps = int(seg_samples * (1 - overlap))
    segments = []
    start = 0
    while start + seg_samples <= len(audio):
        segments.append(audio[start:start + seg_samples])
        start += hop_samps
    return segments


segments = segment_audio(y, sr=sr, segment_duration=segment_duration, overlap=overlap)
n_segments = min(5, len(segments))

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(3, 1, height_ratios=[1.0, 1.2, 0.9], hspace=0.35)

# ──────────────────────────────────────────────
# (a) Conceptual diagram
# ──────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_title('(a) Sliding Window Concept — Overlapping 1-second Segments',
              fontsize=13, fontweight='bold', pad=8)

t_total = 3.0
t_synth = np.linspace(0, t_total, int(t_total * 1000))
synth = (0.5 * np.sin(2 * np.pi * 5 * t_synth)
         + 0.3 * np.sin(2 * np.pi * 13 * t_synth)
         + 0.2 * np.random.randn(len(t_synth)) * 0.1)

ax1.plot(t_synth, synth, color='gray', linewidth=0.8, alpha=0.6, zorder=1)

y_min, y_max = -1.1, 1.1
block_height = y_max - y_min

for i in range(n_segments):
    st = i * hop
    et = st + segment_duration
    c = segment_colors[i % len(segment_colors)]
    rect = mpatches.Rectangle((st, y_min), segment_duration, block_height,
                              facecolor=c, alpha=0.20, edgecolor=c,
                              linewidth=1.5, zorder=2)
    ax1.add_patch(rect)
    ax1.text(st + segment_duration / 2, y_max + 0.08, f'S{i+1}',
             ha='center', va='bottom', fontsize=9, fontweight='bold', color=c)

# Hop arrows
for i in range(n_segments - 1):
    sp = i * hop
    ep = (i + 1) * hop
    arrow_y = y_min - 0.15
    ax1.annotate('', xy=(ep, arrow_y), xytext=(sp, arrow_y),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax1.text(hop / 2, y_min - 0.30, 'hop = 0.5s',
         ha='center', va='top', fontsize=8, fontstyle='italic')

# Segment duration callout
ax1.annotate('segment = 1.0s', xy=(segment_duration / 2, y_max),
             xytext=(segment_duration / 2, y_max + 0.45),
             ha='center', fontsize=8, fontstyle='italic',
             arrowprops=dict(arrowstyle='->', color='gray', lw=1))

# Overlap callout
ov_mid = (hop + segment_duration) / 2
ax1.annotate('50% overlap', xy=(ov_mid, y_max),
             xytext=(ov_mid + 0.3, y_max + 0.75),
             ha='center', fontsize=8, fontstyle='italic', color='red',
             arrowprops=dict(arrowstyle='->', color='red', lw=1))

xlim_max = min(t_total, n_segments * hop + segment_duration)
ax1.set_xlim(0, xlim_max)
ax1.set_ylim(y_min - 0.5, y_max + 0.9)
ax1.set_xlabel('Time (seconds)', fontsize=10)
ax1.set_ylabel('Amplitude', fontsize=10)
ax1.grid(True, alpha=0.3, linestyle=':')

legend_patches = [
    mpatches.Patch(facecolor=segment_colors[i], alpha=0.4,
                   edgecolor=segment_colors[i], label=f'Segment {i+1}')
    for i in range(min(3, n_segments))
]
ax1.legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=3)

# ──────────────────────────────────────────────
# (b) Real waveform
# ──────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_title('(b) Actual Audio Waveform with Sliding Window Segmentation',
              fontsize=13, fontweight='bold', pad=8)

time = np.arange(len(y)) / sr
ax2.plot(time, y, color='black', linewidth=0.5, alpha=0.8, zorder=1)

for i in range(n_segments):
    st = i * hop
    et = st + segment_duration
    c = segment_colors[i % len(segment_colors)]
    ax2.axvspan(st, et, alpha=0.15, color=c, zorder=2)
    ax2.axvline(x=st, color=c, linestyle='--', linewidth=1.0, alpha=0.7, zorder=3)
    ax2.axvline(x=et, color=c, linestyle='--', linewidth=1.0, alpha=0.7, zorder=3)
    ax2.text(st + segment_duration / 2, 0.85, f'S{i+1}',
             ha='center', fontsize=9, fontweight='bold', color=c,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                       edgecolor=c, alpha=0.8))

for i in range(n_segments + 1):
    ax2.axvline(x=i * hop, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

ax2.set_xlim(0, min(display_duration, n_segments * hop + segment_duration))
ax2.set_ylim(-1.1, 1.1)
ax2.set_xlabel('Time (seconds)', fontsize=10)
ax2.set_ylabel('Amplitude', fontsize=10)
ax2.grid(True, alpha=0.3, linestyle=':')

# ──────────────────────────────────────────────
# (c) Zoom on overlap
# ──────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.set_title('(c) Detail: 50% Overlap Between Consecutive Segments',
              fontsize=13, fontweight='bold', pad=8)

zoom_start, zoom_end = 0.3, 1.7
zmask = (time >= zoom_start) & (time <= zoom_end)
ax3.plot(time[zmask], y[zmask], color='black', linewidth=0.8, alpha=0.6, zorder=1)

s1_start, s1_end = 0.0, 1.0
ax3.axvspan(max(zoom_start, s1_start), min(zoom_end, s1_end),
            alpha=0.25, color=segment_colors[0], zorder=2,
            label=f'Segment 1: {s1_start:.1f}s–{s1_end:.1f}s')

s2_start, s2_end = 0.5, 1.5
ax3.axvspan(max(zoom_start, s2_start), min(zoom_end, s2_end),
            alpha=0.25, color=segment_colors[1], zorder=2,
            label=f'Segment 2: {s2_start:.1f}s–{s2_end:.1f}s')

# Overlap bracket
ov_s, ov_e = 0.5, 1.0
ov_mid = (ov_s + ov_e) / 2
ax3.annotate('', xy=(ov_s, -0.95), xytext=(ov_e, -0.95),
             arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
ax3.text(ov_mid, -1.1, '50% Overlap (0.5s)',
         ha='center', va='top', fontsize=8.5, color='red', fontweight='bold')

# Boundary lines
for t_pos, lbl in [(0.5, 'S2 start'), (1.0, 'S3 start')]:
    if zoom_start <= t_pos <= zoom_end:
        ax3.axvline(x=t_pos, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

ax3.set_xlim(zoom_start, zoom_end)
ax3.set_ylim(-1.3, 1.2)
ax3.set_xlabel('Time (seconds)', fontsize=10)
ax3.set_ylabel('Amplitude', fontsize=10)
ax3.grid(True, alpha=0.3, linestyle=':')
ax3.legend(loc='upper right', fontsize=8)

# ──────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────
fig.suptitle('Sliding Window Segmentation: 1.0s Segments with 50% Overlap',
             fontsize=14, fontweight='bold', y=0.98)

plt.subplots_adjust(top=0.935)
output_dir = os.path.join(project_root, 'assets')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'Figure_Sliding_Window.png')
plt.savefig(output_path, dpi=300)
print(f'Saved: {output_path}')
