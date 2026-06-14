import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name

def load_file(filepath):
    with h5py.File(filepath, 'r') as f:
        dataset_name = get_dataset_name(filepath)
        matrix = f.get(dataset_name)[()]
    return matrix

# --- Load one example file ---
filepath = "Intra/train/rest_105923_1.h5"
matrix = load_file(filepath)
print(f"Shape: {matrix.shape}")  # (248, 35624)

# --- Plot a few sensors over time ---
fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(matrix[i * 60], linewidth=0.5)
    ax.set_ylabel(f"Sensor {i*60}")
plt.suptitle("Raw MEG Signal - Resting State")
plt.xlabel("Time steps")
plt.tight_layout()
plt.show()

# --- Plot a heatmap of all sensors ---
plt.figure(figsize=(14, 5))
plt.imshow(matrix, aspect='auto', cmap='RdBu_r',
           vmin=np.percentile(matrix, 1), vmax=np.percentile(matrix, 99))
plt.colorbar(label='fT')
plt.xlabel("Time steps")
plt.ylabel("Sensor (0–247)")
plt.title("All 248 Sensors - Resting State")
plt.tight_layout()
plt.show()

def preprocess(matrix, downsample_factor=4):
    # Downsample along time axis (2034 Hz -> ~500 Hz)
    matrix = matrix[:, ::downsample_factor]
    
    # Z-score normalization per time step (column-wise)
    mean = matrix.mean(axis=0, keepdims=True)
    std  = matrix.std(axis=0, keepdims=True) + 1e-8
    matrix = (matrix - mean) / std
    
    return matrix

matrix_preprocessed = preprocess(matrix)
print(f"After preprocessing: {matrix_preprocessed.shape}")  # (248, ~8906)

LABEL_MAP = {
    "rest": 0,
    "task_motor": 1,
    "task_story_math": 2,
    "task_working_memory": 3
}

def load_folder(folder_path):
    X, y = [], []
    for fname in os.listdir(folder_path):
        if not fname.endswith('.h5'):
            continue
        task_type = "_".join(fname.split('_')[:-2])  # e.g. "task_motor"
        if task_type not in LABEL_MAP:
            continue
        filepath = os.path.join(folder_path, fname)
        matrix = load_file(filepath.replace("\\", "/"))
        matrix = preprocess(matrix)
        X.append(matrix)
        y.append(LABEL_MAP[task_type])
    return np.array(X), np.array(y)

X_train, y_train = load_folder("Intra/train")
X_test,  y_test  = load_folder("Intra/test")
print(f"Train: {X_train.shape}, Labels: {y_train.shape}")