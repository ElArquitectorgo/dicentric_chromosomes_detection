import os
import shutil
import argparse
from ultralytics import YOLO
from pathlib import Path

def test(dataset, project_name, overwrite):
    results = {}
    project = f"runs/detect/{project_name}"
    
    for k in range(5):
        if not overwrite and os.path.isfile(f"{project}/fold_{k+1}_val/test_confusion_matrix.csv"):
            print(f"Skipping testing from {project_name}, fold: {k+1}")
            continue

        model = YOLO(f"{project}/fold_{k+1}_train/weights/best.pt")

        results[k] = model.val(
            data=f"{dataset}/data.yaml",
            project=project_name,
            name=f"fold_{k+1}_val",
            iou=0.5,
            max_det=60,
            save_txt=True,
            save_conf=True,
            device=-1)
        
        with open(f"{project}/fold_{k+1}_val/test_validation_results.csv", "w") as f:
            f.write(results[k].to_csv())
        with open(f"{project}/fold_{k+1}_val/test_confusion_matrix.csv", "w") as f:
            f.write(results[k].confusion_matrix.to_csv())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="path to images dataset folder", required=True)
    parser.add_argument("-n", "--name", help="project name (the results will be saved here)", required=True)
    parser.add_argument("-o", "--overwrite", help="overwrite existing results", required=False, default=False)

    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        parser.error(f"Dataset path does not exist or is not a directory: {args.dataset}")
    
    test(args.dataset, args.name, args.overwrite)

if __name__ == "__main__":
    main()

