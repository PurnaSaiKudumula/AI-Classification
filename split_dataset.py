import os
import json
import shutil
import random
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split

# Path to your dataset
DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILES = [f for f in os.listdir(DATASET_DIR) if f.endswith('.json')]

# Extract class label from each file (using first label in 'shapes')
def get_label(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        shapes = data.get('shapes', [])
        if shapes:
            return shapes[0].get('label', 'unknown')
        return 'unknown'

file_labels = [(fname, get_label(os.path.join(DATASET_DIR, fname))) for fname in JSON_FILES]

# Helper to create split folders
def create_split_folders(base_dir):
    for split in ['random', 'stratified']:
        for subset in ['train', 'test', 'validate']:
            os.makedirs(os.path.join(base_dir, split, subset), exist_ok=True)

create_split_folders(DATASET_DIR)

# Split ratios
train_ratio = 0.7
test_ratio = 0.15
val_ratio = 0.15

# --- RANDOM SPLIT ---
random.shuffle(file_labels)
N = len(file_labels)
train_end = int(N * train_ratio)
test_end = train_end + int(N * test_ratio)
train_files = file_labels[:train_end]
test_files = file_labels[train_end:test_end]
val_files = file_labels[test_end:]

for subset, files in zip(['train', 'test', 'validate'], [train_files, test_files, val_files]):
    for fname, _ in files:
        shutil.copy2(os.path.join(DATASET_DIR, fname), os.path.join(DATASET_DIR, 'random', subset, fname))

# --- STRATIFIED SPLIT ---
labels = [label for _, label in file_labels]
files = [fname for fname, _ in file_labels]
train_files, temp_files, train_labels, temp_labels = train_test_split(files, labels, stratify=labels, test_size=(1-train_ratio), random_state=42)
test_files, val_files, test_labels, val_labels = train_test_split(temp_files, temp_labels, stratify=temp_labels, test_size=val_ratio/(val_ratio+test_ratio), random_state=42)

for subset, files in zip(['train', 'test', 'validate'], [train_files, test_files, val_files]):
    for fname in files:
        shutil.copy2(os.path.join(DATASET_DIR, fname), os.path.join(DATASET_DIR, 'stratified', subset, fname))

print('Splitting complete!')
