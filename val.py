import os
import shutil
import argparse
from ultralytics import YOLO
from pathlib import Path

def get_dataset(dataset_path):
    ds_yamls = []
    ksplit = 5
    save_path = Path(dataset_path)
    folds = [f"split_{n}" for n in range(1, ksplit + 1)]

    for split in folds:
        split_dir = save_path / split
        dataset_yaml = split_dir / f"{split}_dataset.yaml"
        ds_yamls.append(dataset_yaml)

    return ds_yamls

def val(dataset, project_name):
    results = {}
    project = f"runs/detect/{project_name}"

    for k, dataset_yaml in enumerate(dataset):
        model = YOLO(f"{project}/fold_{k+1}_train/weights/best.pt")
        results[k] = model.val(
            data=dataset_yaml,
            project=project_name,
            name=f"fold_{k+1}_val",
            iou=0.5,
            max_det=60,
            save_txt=True,
            save_conf=True,
            device=-1)
        
        with open(f"{project}/fold_{k+1}_val/validation_results.csv", "w") as f:
            f.write(results[k].to_csv())
        with open(f"{project}/fold_{k+1}_val/confusion_matrix.csv", "w") as f:
            f.write(results[k].confusion_matrix.to_csv())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="path to images dataset folder", required=True)
    parser.add_argument("-n", "--name", help="project name (the results will be saved here)", required=True)

    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        parser.error(f"Dataset path does not exist or is not a directory: {args.dataset}")
    
    ds_yamls = get_dataset(args.dataset)
    val(ds_yamls, args.name)

if __name__ == "__main__":
    main()

