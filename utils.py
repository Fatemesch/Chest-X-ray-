import numpy as np
import math
import torch
from datasets import load_from_disk, load_dataset

biomarkers = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']

def filter_nans(olives):
    valid_indices = []
    for i in range(len(olives)):
        has_valid_biomarker = any(olives[i][bm] is not None for bm in biomarkers)
        if has_valid_biomarker:
            valid_indices.append(i)
    filtered_olives = olives.select(valid_indices)
    print(f"Original dataset size: {len(olives)}")
    print(f"Filtered dataset size: {len(filtered_olives)}")
    print(f"Removed {len(olives) - len(filtered_olives)} samples where all biomarkers were None")
    print(f"Percentage of data retained: {(len(filtered_olives)/len(olives))*100:.2f}%")
    return filtered_olives

def compute_clinical_mean_std(filtered_olives):
    n = 0
    clinical_mean = np.zeros(2, dtype=np.float64)
    M2 = np.zeros(2, dtype=np.float64)
    for i in range(len(filtered_olives)):
        data = filtered_olives[i]
        vals = [data['BCVA'], data['CST']]
        if any([v is None or (isinstance(v, float) and math.isnan(v)) for v in vals]):
            continue
        vals = np.array(vals, dtype=np.float64)
        n += 1
        delta = vals - clinical_mean
        clinical_mean += delta / n
        delta2 = vals - clinical_mean
        M2 += delta * delta2
    if n > 1:
        clinical_std = np.sqrt(M2 / (n - 1))
    else:
        clinical_std = np.zeros_like(clinical_mean)
    return clinical_mean, clinical_std, n

def compute_image_mean_std(filtered_olives, img_key='Image'):
    n = 0
    img_mean = 0.0
    M2 = 0.0
    for i in range(len(filtered_olives)):
        data = filtered_olives[i]
        img = data[img_key]
        if isinstance(img, torch.Tensor):
            arr = img.numpy()
        else:
            arr = np.array(img)
        if arr.ndim == 3 and arr.shape[0] == 1:  # (1, H, W)
            arr = arr[0]
        vals = arr.flatten().astype(np.float64)
        n_vals = len(vals)
        if n_vals == 0:
            continue
        old_mean = img_mean
        img_mean = (img_mean * n + vals.sum()) / (n + n_vals)
        M2 += ((vals - old_mean) * (vals - img_mean)).sum()
        n += n_vals
    if n > 1:
        img_std = np.sqrt(M2 / (n - 1))
    else:
        img_std = 0.0
    return img_mean, img_std

if __name__ == "__main__":
    raw_train_path = "./"
    olives = load_from_disk(raw_train_path)
    filtered_olives = filter_nans(olives)
    filtered_olives.save_to_disk("removed_nans_filtered_olives_dataset")
    
    olives_test  = load_dataset('gOLIVES/OLIVES_Dataset', 'biomarker_detection', split = 'test')
    olives_test.save_to_disk("olving_test_dataset")

    clinical_mean, clinical_std, n_clin = compute_clinical_mean_std(filtered_olives)
    print(f"Clinical means: {clinical_mean}, stds: {clinical_std}, n = {n_clin}")

    # img_mean, img_std = compute_image_mean_std(filtered_olives)
    # print(f"Image mean: {img_mean}, std: {img_std}")
