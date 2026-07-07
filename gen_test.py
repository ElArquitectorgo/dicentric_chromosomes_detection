from utils import *
from pathlib import Path
import cv2
import numpy as np

def save(img, method, output_dir, img_name):
    np.save(output_dir / method / "val" / "images" / f"{img_name}.npy", img)

def main():
    input_dir = Path(f"dataset/test/images")
    output_dir = Path(f"folds/test")

    for img_path in input_dir.glob("*"):
        img = cv2.imread(str(img_path), 0)
        img_name = img_path.stem
        print(f"Processing: {img_path}, dtype: {img.dtype}")

        # Original
        save(img, "raw", output_dir, img_name)
        
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

        # top categories
        d["raw_clahe_lt"] = cv2.merge([img, d["clahe"], d["lt"]])
        d["raw_max_std"] = cv2.merge([img, d["max"], d["std"]])
        d["raw_clahe_std"] = cv2.merge([img, d["clahe"], d["std"]])
        d["raw_lt_std"] = cv2.merge([img, d["lt"], d["std"]])
        d["raw_lbp_std"] = cv2.merge([img, d["lbp"], d["std"]])
        d["raw_no2_nc5"] = cv2.merge([img, d["no2"], d["nc5"]])
        d["lbp_max_std"] = cv2.merge([d["lbp"], d["max"], d["std"]])
        d["lt_clahe_he"] = cv2.merge([d["lt"], d["clahe"], d["he"]])
        d["lt_clahe_std"] = cv2.merge([d["lt"], d["clahe"], d["std"]])
        d["lbp_mean_std"] = cv2.merge([d["lbp"], d["mean"], d["std"]])

        for k, v in d.items():
            save(v, k, output_dir, img_name)

if __name__ == '__main__':
    main()
