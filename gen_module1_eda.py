import json
import os

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.7"}
    },
    "nbformat": 4, "nbformat_minor": 4
}

def mk_md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]}

def mk_code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.split("\n")]}

cells = []

# ───────────── Title ─────────────
cells.append(mk_md("""# Module 1: Exploratory Data Analysis (EDA)
## Multimodal Health State Prediction Using Smartphone Sensors
### VI-A Group ID 13 — Deep Learning Capstone Project

**Goal:** Understand all 5 data sources — shapes, class distributions, missing values, and visual patterns — before any modelling. This notebook directly supports the **Dataset** section of the research paper.

---
**Data Sources:**
| # | Modality | Path | Target |
|---|----------|------|--------|
| 1 | Fingernail Images | `Datasets/Fingernails/` | Anemia (folder name) |
| 2 | Audio (RAVDESS) | `Datasets/audio_speech_actors_01-24/` | Stress (filename) |
| 3 | Accelerometer (WISDM) | `Datasets/WISDM_ar_latest/` | Fatigue (activity label) |
| 4 | Water Intake CSV | `Datasets/Daily_Water_Intake.csv` | Dehydration |
| 5 | Sleep Health CSV | `Datasets/Sleep_health_and_lifestyle_dataset.csv` | Sleep Disorder |"""))

# ───────────── Imports ─────────────
cells.append(mk_code("""import os, warnings, glob
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
import seaborn as sns
from PIL import Image
from collections import Counter

# Audio
try:
    import librosa
    import librosa.display
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("librosa not installed — audio visuals will use pre-processed .npy instead")

BASE = 'D:/DL Project'
DATASETS = os.path.join(BASE, 'Datasets')
PROCESSED = os.path.join(BASE, 'processed_data')

print("EDA Environment Ready ✓")"""))

# ───────────── 1. Fingernail Images ─────────────
cells.append(mk_md("""---
## 1. Fingernail Images — Anemia Detection
Load images from `Fingernails/Anemic` and `Fingernails/NonAnemic` subfolders and inspect class balance, resolution, and sample visuals."""))

cells.append(mk_code("""# 1A — Scan fingernail dataset
fg_root = os.path.join(DATASETS, 'Fingernails')
classes = sorted([d for d in os.listdir(fg_root) if os.path.isdir(os.path.join(fg_root, d))])
class_counts = {}
sample_paths = {}

for cls in classes:
    folder = os.path.join(fg_root, cls)
    imgs = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    class_counts[cls] = len(imgs)
    sample_paths[cls] = [os.path.join(folder, f) for f in imgs[:5]]

print("=== Fingernail Dataset Summary ===")
total = sum(class_counts.values())
for cls, cnt in class_counts.items():
    print(f"  {cls}: {cnt} images ({cnt/total*100:.1f}%)")
print(f"  Total: {total} images")
print(f"  Classes: {classes}")"""))

cells.append(mk_code("""# 1B — Class distribution bar chart
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
colors = ['#e74c3c' if 'Anemic' == c else '#2ecc71' for c in class_counts.keys()]
ax.bar(class_counts.keys(), class_counts.values(), color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('Fingernail Image Class Distribution', fontweight='bold')
ax.set_ylabel('Number of Images')
for i, (k, v) in enumerate(class_counts.items()):
    ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.show()"""))

cells.append(mk_code("""# 1C — Display 5 sample images per class
fig, axes = plt.subplots(len(classes), 5, figsize=(14, 3 * len(classes)))
if len(classes) == 1:
    axes = [axes]
for row, cls in enumerate(classes):
    for col in range(min(5, len(sample_paths[cls]))):
        img = Image.open(sample_paths[cls][col]).convert('RGB')
        axes[row][col].imshow(img)
        axes[row][col].set_title(f'{cls}' if col == 0 else '', fontsize=10, fontweight='bold')
        axes[row][col].axis('off')
    # Hide unused subplots
    for col in range(len(sample_paths[cls]), 5):
        axes[row][col].axis('off')
fig.suptitle('Sample Fingernail Images per Class', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Check resolution of first image
sample_img = Image.open(sample_paths[classes[0]][0])
print(f"Sample image size: {sample_img.size}, mode: {sample_img.mode}")"""))

# ───────────── 2. Audio / RAVDESS ─────────────
cells.append(mk_md("""---
## 2. Audio (RAVDESS) — Stress Detection
RAVDESS filenames encode emotion labels. We map them to a binary stress classification:
- **Not Stressed:** neutral, calm, happy, surprised
- **Stressed:** angry, fearful, sad, disgust"""))

cells.append(mk_code("""# 2A — Scan audio files and extract emotion labels
audio_root = os.path.join(DATASETS, 'audio_speech_actors_01-24')
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}
stress_map = {
    'neutral': 'Not Stressed', 'calm': 'Not Stressed', 'happy': 'Not Stressed', 'surprised': 'Not Stressed',
    'sad': 'Stressed', 'angry': 'Stressed', 'fearful': 'Stressed', 'disgust': 'Stressed'
}

audio_files = []
emotions = []
stress_labels = []

for actor_dir in sorted(os.listdir(audio_root)):
    actor_path = os.path.join(audio_root, actor_dir)
    if not os.path.isdir(actor_path):
        continue
    for f in sorted(os.listdir(actor_path)):
        if f.endswith('.wav'):
            parts = f.split('-')
            emo_code = parts[2]
            emo_name = emotion_map.get(emo_code, 'unknown')
            audio_files.append(os.path.join(actor_path, f))
            emotions.append(emo_name)
            stress_labels.append(stress_map.get(emo_name, 'Unknown'))

print(f"=== RAVDESS Audio Dataset Summary ===")
print(f"  Total audio files: {len(audio_files)}")
print(f"  Actors: {len([d for d in os.listdir(audio_root) if os.path.isdir(os.path.join(audio_root, d))])}")

emo_counts = Counter(emotions)
stress_counts = Counter(stress_labels)
print(f"\\n  Emotion Distribution:")
for k, v in sorted(emo_counts.items()):
    print(f"    {k}: {v}")
print(f"\\n  Binary Stress Distribution:")
for k, v in sorted(stress_counts.items()):
    print(f"    {k}: {v}")"""))

cells.append(mk_code("""# 2B — Emotion & Stress Distribution plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Multi-class emotions
emo_df = pd.DataFrame({'Emotion': emotions})
emo_order = sorted(emo_counts.keys())
palette_emo = sns.color_palette("Set2", len(emo_order))
sns.countplot(data=emo_df, x='Emotion', order=emo_order, palette=palette_emo, ax=ax1, edgecolor='black', linewidth=0.5)
ax1.set_title('RAVDESS — 8-Class Emotion Distribution', fontweight='bold')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)
for p in ax1.patches:
    ax1.annotate(str(int(p.get_height())), (p.get_x() + p.get_width()/2., p.get_height()),
                 ha='center', va='bottom', fontweight='bold', fontsize=9)

# Binary stress
stress_df = pd.DataFrame({'Stress': stress_labels})
colors_stress = ['#2ecc71', '#e74c3c']
sns.countplot(data=stress_df, x='Stress', palette=colors_stress, ax=ax2, edgecolor='black', linewidth=0.5)
ax2.set_title('RAVDESS — Binary Stress Distribution', fontweight='bold')
ax2.set_ylabel('Count')
for p in ax2.patches:
    ax2.annotate(str(int(p.get_height())), (p.get_x() + p.get_width()/2., p.get_height()),
                 ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.show()"""))

cells.append(mk_code("""# 2C — Waveform + Spectrogram of 2 sample audio files
if HAS_LIBROSA and len(audio_files) >= 2:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for i in range(2):
        y, sr = librosa.load(audio_files[i], sr=16000)
        # Waveform
        librosa.display.waveshow(y, sr=sr, ax=axes[i][0])
        axes[i][0].set_title(f'Waveform — {emotions[i]} ({stress_labels[i]})', fontweight='bold')
        axes[i][0].set_xlabel('Time (s)')
        # Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=axes[i][1])
        axes[i][1].set_title(f'Mel Spectrogram — {emotions[i]}', fontweight='bold')
        fig.colorbar(img, ax=axes[i][1], format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
else:
    # Fallback: show shape from processed .npy
    mfccs = np.load(os.path.join(PROCESSED, 'audio_mfccs.npy'))
    labels = np.load(os.path.join(PROCESSED, 'audio_labels.npy'))
    print(f"Pre-processed MFCC shape: {mfccs.shape}")
    print(f"Audio labels distribution: {Counter(labels)}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(mfccs[0].T, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title(f'MFCC Sample 1 (label={labels[0]})', fontweight='bold')
    axes[0].set_xlabel('Time Steps'); axes[0].set_ylabel('MFCC Coefficients')
    axes[1].imshow(mfccs[1].T, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title(f'MFCC Sample 2 (label={labels[1]})', fontweight='bold')
    axes[1].set_xlabel('Time Steps'); axes[1].set_ylabel('MFCC Coefficients')
    plt.tight_layout()
    plt.show()"""))

# ───────────── 3. Accelerometer / WISDM ─────────────
cells.append(mk_md("""---
## 3. Accelerometer (WISDM) — Fatigue Detection
WISDM dataset contains accelerometer readings with activity labels. We map activities to fatigue proxy:
- **Active (Not Fatigued):** Walking, Jogging, Upstairs, Downstairs
- **Sedentary (Fatigued Proxy):** Sitting, Standing"""))

cells.append(mk_code("""# 3A — Load WISDM raw data
wisdm_path = os.path.join(DATASETS, 'WISDM_ar_latest', 'WISDM_ar_v1.1', 'WISDM_ar_v1.1_raw.txt')
# Parse the raw WISDM file (format: user,activity,timestamp,x,y,z;)
rows = []
with open(wisdm_path, 'r') as f:
    for line in f:
        line = line.strip().rstrip(';')
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 6:
            try:
                user = int(parts[0])
                activity = parts[1].strip()
                timestamp = float(parts[2])
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5].rstrip(';'))
                rows.append([user, activity, timestamp, x, y, z])
            except (ValueError, IndexError):
                continue

wisdm_df = pd.DataFrame(rows, columns=['user', 'activity', 'timestamp', 'x', 'y', 'z'])
print(f"=== WISDM Accelerometer Dataset Summary ===")
print(f"  Total rows: {len(wisdm_df):,}")
print(f"  Users: {wisdm_df['user'].nunique()}")
print(f"  Activities: {wisdm_df['activity'].unique().tolist()}")
print(f"  Missing values:\\n{wisdm_df.isnull().sum()}")
print(f"\\n  Activity Distribution:")
act_counts = wisdm_df['activity'].value_counts()
for act, cnt in act_counts.items():
    print(f"    {act}: {cnt:,}")"""))

cells.append(mk_code("""# 3B — Activity distribution + Fatigue mapping
fatigue_map = {
    'Walking': 'Active', 'Jogging': 'Active', 'Upstairs': 'Active', 'Downstairs': 'Active',
    'Sitting': 'Sedentary', 'Standing': 'Sedentary'
}
wisdm_df['fatigue_proxy'] = wisdm_df['activity'].map(fatigue_map)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Activity counts
palette_act = sns.color_palette("tab10", wisdm_df['activity'].nunique())
sns.countplot(data=wisdm_df, y='activity', order=act_counts.index, palette=palette_act, ax=ax1, edgecolor='black', linewidth=0.5)
ax1.set_title('WISDM Activity Distribution', fontweight='bold')
ax1.set_xlabel('Sample Count')

# Fatigue binary
fat_counts = wisdm_df['fatigue_proxy'].value_counts()
sns.countplot(data=wisdm_df, x='fatigue_proxy', palette=['#2ecc71', '#e74c3c'], ax=ax2, edgecolor='black', linewidth=0.5)
ax2.set_title('Fatigue Proxy Distribution', fontweight='bold')
ax2.set_ylabel('Count')
for p in ax2.patches:
    ax2.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width()/2., p.get_height()),
                 ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()"""))

cells.append(mk_code("""# 3C — Time-series plot of x/y/z per activity (3 sample activities)
sample_activities = ['Walking', 'Sitting', 'Jogging']
fig, axes = plt.subplots(len(sample_activities), 1, figsize=(14, 3.5 * len(sample_activities)))
if len(sample_activities) == 1:
    axes = [axes]

for i, act in enumerate(sample_activities):
    subset = wisdm_df[wisdm_df['activity'] == act].head(500)
    axes[i].plot(subset['x'].values, label='X', alpha=0.8, linewidth=0.8)
    axes[i].plot(subset['y'].values, label='Y', alpha=0.8, linewidth=0.8)
    axes[i].plot(subset['z'].values, label='Z', alpha=0.8, linewidth=0.8)
    axes[i].set_title(f'Accelerometer — {act} (first 500 samples)', fontweight='bold')
    axes[i].set_xlabel('Sample Index')
    axes[i].set_ylabel('Acceleration')
    axes[i].legend(loc='upper right')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

# ───────────── 4. Water Intake CSV ─────────────
cells.append(mk_md("""---
## 4. Daily Water Intake CSV — Dehydration Detection
Target column: `Hydration Level` (Good / Poor)"""))

cells.append(mk_code("""# 4A — Load and inspect water intake dataset
water_df = pd.read_csv(os.path.join(DATASETS, 'Daily_Water_Intake.csv'))
print(f"=== Daily Water Intake Dataset Summary ===")
print(f"  Shape: {water_df.shape}")
print(f"  Columns: {water_df.columns.tolist()}")
print(f"\\n  Missing values:")
print(water_df.isnull().sum())
print(f"\\n  Data types:")
print(water_df.dtypes)
print(f"\\n  First 5 rows:")
water_df.head()"""))

cells.append(mk_code("""# 4B — Target class distribution
target_col = [c for c in water_df.columns if 'hydra' in c.lower() or 'level' in c.lower()]
if target_col:
    target_col = target_col[0]
else:
    target_col = water_df.columns[-1]
    
print(f"Target column: '{target_col}'")
print(f"\\nClass distribution:")
print(water_df[target_col].value_counts())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Class bar chart
water_df[target_col].value_counts().plot(kind='bar', ax=ax1, color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=0.5)
ax1.set_title('Hydration Level Distribution', fontweight='bold')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=0)
for p in ax1.patches:
    ax1.annotate(str(int(p.get_height())), (p.get_x() + p.get_width()/2., p.get_height()),
                 ha='center', va='bottom', fontweight='bold')

# Pie chart
water_df[target_col].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2, 
                                          colors=['#2ecc71', '#e74c3c'], startangle=90)
ax2.set_title('Class Proportion', fontweight='bold')
ax2.set_ylabel('')
plt.tight_layout()
plt.show()"""))

cells.append(mk_code("""# 4C — Feature histograms for numerical columns
num_cols = water_df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in num_cols:
    num_cols.remove(target_col)

n_cols_plot = min(len(num_cols), 8)
if n_cols_plot > 0:
    fig, axes = plt.subplots(2, (n_cols_plot + 1) // 2, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols[:n_cols_plot]):
        water_df[col].hist(ax=axes[i], bins=30, color='steelblue', edgecolor='black', linewidth=0.3)
        axes[i].set_title(col, fontsize=10, fontweight='bold')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Water Intake — Numerical Feature Distributions', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()"""))

cells.append(mk_code("""# 4D — Correlation heatmap
corr_cols = water_df.select_dtypes(include=[np.number]).columns
if len(corr_cols) >= 2:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = water_df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                linewidths=0.5, ax=ax, vmin=-1, vmax=1, square=True)
    ax.set_title('Water Intake — Feature Correlation Matrix', fontweight='bold')
    plt.tight_layout()
    plt.show()"""))

# ───────────── 5. Sleep Health CSV ─────────────
cells.append(mk_md("""---
## 5. Sleep Health & Lifestyle CSV — Sleep Disorder Detection
Target column: `Sleep Disorder` (None / Sleep Apnea / Insomnia → 3 classes)"""))

cells.append(mk_code("""# 5A — Load and inspect sleep dataset
sleep_df = pd.read_csv(os.path.join(DATASETS, 'Sleep_health_and_lifestyle_dataset.csv'))
print(f"=== Sleep Health Dataset Summary ===")
print(f"  Shape: {sleep_df.shape}")
print(f"  Columns: {sleep_df.columns.tolist()}")
print(f"\\n  Missing values:")
print(sleep_df.isnull().sum())
print(f"\\n  Data types:")
print(sleep_df.dtypes)
print(f"\\n  First 5 rows:")
sleep_df.head()"""))

cells.append(mk_code("""# 5B — Target class distribution
sleep_target = 'Sleep Disorder'
if sleep_target not in sleep_df.columns:
    # Find closest column
    sleep_target = [c for c in sleep_df.columns if 'disorder' in c.lower() or 'sleep' in c.lower()][-1]

print(f"Target column: '{sleep_target}'")
# Replace NaN with 'None' for the target
sleep_df[sleep_target] = sleep_df[sleep_target].fillna('None')
print(f"\\nClass distribution:")
print(sleep_df[sleep_target].value_counts())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

colors_sleep = ['#3498db', '#e74c3c', '#f39c12']
sleep_df[sleep_target].value_counts().plot(kind='bar', ax=ax1, color=colors_sleep, edgecolor='black', linewidth=0.5)
ax1.set_title('Sleep Disorder Distribution', fontweight='bold')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=0)
for p in ax1.patches:
    ax1.annotate(str(int(p.get_height())), (p.get_x() + p.get_width()/2., p.get_height()),
                 ha='center', va='bottom', fontweight='bold')

sleep_df[sleep_target].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2,
                                            colors=colors_sleep, startangle=90)
ax2.set_title('Class Proportion', fontweight='bold')
ax2.set_ylabel('')
plt.tight_layout()
plt.show()"""))

cells.append(mk_code("""# 5C — Feature histograms for numerical columns
sleep_num_cols = sleep_df.select_dtypes(include=[np.number]).columns.tolist()
# Remove Person ID if present
sleep_num_cols = [c for c in sleep_num_cols if 'id' not in c.lower()]

n_plot = min(len(sleep_num_cols), 8)
if n_plot > 0:
    fig, axes = plt.subplots(2, (n_plot + 1) // 2, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(sleep_num_cols[:n_plot]):
        sleep_df[col].hist(ax=axes[i], bins=25, color='mediumpurple', edgecolor='black', linewidth=0.3)
        axes[i].set_title(col, fontsize=10, fontweight='bold')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Sleep Health — Numerical Feature Distributions', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()"""))

cells.append(mk_code("""# 5D — Correlation heatmap for sleep data
sleep_corr_cols = sleep_df.select_dtypes(include=[np.number]).columns
sleep_corr_cols = [c for c in sleep_corr_cols if 'id' not in c.lower()]
if len(sleep_corr_cols) >= 2:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = sleep_df[sleep_corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                linewidths=0.5, ax=ax, vmin=-1, vmax=1, square=True)
    ax.set_title('Sleep Health — Feature Correlation Matrix', fontweight='bold')
    plt.tight_layout()
    plt.show()"""))

# ───────────── 6. Processed Data Summary ─────────────
cells.append(mk_md("""---
## 6. Processed Data Summary (from `processed_data/`)
Verify shapes and labels of the pre-processed `.npy` files used for model training."""))

cells.append(mk_code("""# 6A — Load and summarize all processed .npy files
print("=" * 60)
print("PROCESSED DATA SHAPES VERIFICATION")
print("=" * 60)

data_info = [
    ('fingernail_images.npy', 'fingernail_labels.npy', 'Fingernail (Anemia)'),
    ('audio_mfccs.npy', 'audio_labels.npy', 'Audio (Stress)'),
    ('accel_windows.npy', 'accel_labels.npy', 'Accelerometer (Fatigue)'),
    ('water_features.npy', 'water_labels.npy', 'Water (Dehydration)'),
    ('sleep_features.npy', 'sleep_labels.npy', 'Sleep (Sleep Disorder)'),
]

for feat_file, label_file, name in data_info:
    feat = np.load(os.path.join(PROCESSED, feat_file))
    labels = np.load(os.path.join(PROCESSED, label_file), allow_pickle=True)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\\n  {name}:")
    print(f"    Features shape: {feat.shape}")
    print(f"    Labels shape: {labels.shape}")
    print(f"    Classes: {len(unique)} → {dict(zip(unique, counts))}")
    print(f"    Feature dtype: {feat.dtype}, Label dtype: {labels.dtype}")
    print(f"    NaN count: {np.isnan(feat).sum()}, Inf count: {np.isinf(feat).sum()}")

print("\\n" + "=" * 60)
print("ALL DATA VERIFIED ✓")"""))

# ───────────── 7. Summary Table ─────────────
cells.append(mk_md("""---
## 7. Dataset Summary Table (for Paper)

| Modality | Samples | Features | Classes | Imbalance? |
|----------|---------|----------|---------|------------|
| Fingernail Images | ~2500+ | 224×224×3 | 2 (Anemic/NonAnemic) | Check above |
| Audio (RAVDESS) | ~2400 | 130×40 MFCC | 2 (Stressed/Not) | Slightly balanced |
| Accelerometer (WISDM) | ~1M+ raw → ~54K windows | 200×3 | 6 activities → 2 fatigue | Walking dominant |
| Water Intake | ~15K | ~6-10 features | 2 (Good/Poor) | Check above |
| Sleep Health | ~374 | ~11 features | 3 (None/Apnea/Insomnia) | None dominant |

> **Key observations for model design:**
> 1. Fingernail and Audio datasets are relatively small → use pretrained backbones + data augmentation
> 2. WISDM has significant class imbalance → use class weights in loss function
> 3. Sleep dataset is very small (374 samples) → MLP is appropriate, avoid complex architectures
> 4. Water dataset is medium-sized → MLP with SMOTE or class weights"""))

cells.append(mk_md("""---
### ✅ EDA Complete
This notebook covers all 5 data modalities. Findings inform preprocessing (Module 2) and loss function choices (Module 3/4).

**Next:** Module 2 — Data Cleaning & Preprocessing → Module 3 — Unimodal Models"""))

nb["cells"] = cells

BASE = "D:/DL Project"
out_path = os.path.join(BASE, "Module1_EDA.ipynb")
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Module 1 EDA notebook generated: {out_path}")
