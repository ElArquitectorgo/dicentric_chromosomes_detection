from utils import *
from pathlib import Path
import numpy as np

import pandas as pd
from collections import Counter
import random
from sklearn.model_selection import KFold
import shutil
import cv2

def get_folds():
    dataset_path = Path("dataset/train/")
    labels = sorted(dataset_path.rglob("labels/*.txt")) 

    classes = {0: "n_d", 1: "d"}
    cls_idx = sorted(classes.keys())

    index = [label.stem for label in labels]  # uses base filename as ID (no extension)
    labels_df = pd.DataFrame([], columns=cls_idx, index=index)

    for label in labels:
        lbl_counter = Counter()

        with open(label) as lf:
            lines = lf.readlines()

        for line in lines:
            # classes for YOLO label uses integer at first position of each line
            lbl_counter[int(line.split(" ", 1)[0])] += 1

        labels_df.loc[label.stem] = [lbl_counter.get(cls, 0) for cls in cls_idx]

    random.seed(0) 
    ksplit = 5
    kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)

    kfolds = list(kf.split(labels_df))

    folds = [f"split_{n}" for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=index, columns=folds)

    for i, (train, val) in enumerate(kfolds, start=1):
        folds_df.loc[labels_df.iloc[train].index, f"split_{i}"] = "train"
        folds_df.loc[labels_df.iloc[val].index, f"split_{i}"] = "val"
    
    return folds_df

def save(img, method, output_dir, img_name, folds_df):
    for i in range(5):
        split = folds_df.loc[img_name, f"split_{i+1}"]
        path = output_dir / method / f"split_{i+1}" / split

        np.save(path / f"images/{img_name}.npy", img)
        shutil.copy(f"dataset/train/labels/{img_name}.txt", path / f"labels/{img_name}.txt")

def main():
    input_dir = Path(f"dataset/train/images")
    output_dir = Path(f"folds/train")
    folds_df = get_folds()

    for img_path in input_dir.glob("*"):
        img = cv2.imread(str(img_path), 0)
        img_name = img_path.stem
        print(f"Processing: {img_path}, dtype: {img.dtype}")

        # Base
        save(img, "raw", output_dir, img_name, folds_df)

        d = {}
        # Textures
        d["mean"] = fast_glcm_mean(img).astype(np.uint8)
        d["max"] = fast_glcm_max(img).astype(np.uint8)
        d["std"] = fast_glcm_std(img).astype(np.uint8)
        d["contrast"] = fast_glcm_contrast(img).astype(np.uint8)
        d["dissimilarity"] = fast_glcm_dissimilarity(img).astype(np.uint8)
        glcm_asm, glcm_energy = fast_glcm_ASM(img)
        d["asm"] = glcm_asm.astype(np.uint8)
        d["energy"] = glcm_energy.astype(np.uint8)
        d["homogeneity"] = fast_glcm_homogeneity(img).astype(np.uint8)
        d["entropy"] = fast_glcm_entropy(img).astype(np.uint8)
        d["lbp"] = lbp(img)

        # Enhancement
        d["he"] = he(img)
        d["clahe"] = clahe(img)
        d["lt"] = lt(img)
        d["gc"] = gc(img)

        # Bmo
        d["no2"] = otsu(img, "open", 2)
        d["nc5"] = otsu(img, "close", 5)

        # Top categories
        d["raw_clahe_lt"] = cv2.merge([img, d["clahe"], d["lt"]])
        d["raw_clahe_std"] = cv2.merge([img, d["clahe"], d["std"]])
        d["raw_lt_std"] = cv2.merge([img, d["lt"], d["std"]])
        d["raw_max_std"] = cv2.merge([img, d["max"], d["std"]])
        d["raw_no2_nc5"] = cv2.merge([img, d["no2"], d["nc5"]])
        d["lbp_max_std"] = cv2.merge([d["lbp"], d["max"], d["std"]])
        d["lt_clahe_he"] = cv2.merge([d["lt"], d["clahe"], d["he"]])
        d["lt_clahe_std"] = cv2.merge([d["lt"], d["clahe"], d["std"]])
        d["lbp_mean_std"] = cv2.merge([d["lbp"], d["mean"], d["std"]])
        d["raw_lbp_std"] = cv2.merge([img, d["lbp"], d["std"]])

        for k, v in d.items():
            save(v, k, output_dir, img_name, folds_df)

if __name__ == "__main__":
    main()