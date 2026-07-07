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

def train_val(dataset, config_file, model_version, gpu_id, project_name):
    results = {}
    project = f"runs/detect/{project_name}"

    for k, dataset_yaml in enumerate(dataset):
        if os.path.isdir(f"{project}/fold_{k+1}_val"):
            print(f"Skipping training from {project_name}, fold: {k+1}")
            continue

        if os.path.isdir(f"{project}/fold_{k+1}_train"):
            print(f"Training {project_name} again, fold: {k+1}")
            shutil.rmtree(f"{project}/fold_{k+1}_train", ignore_errors=True)

        model = YOLO(f"models/yolo26{model_version}.pt")
        results[k] = model.train(
            data=dataset_yaml,
            project=project_name,
            name=f"fold_{k+1}_train",
            batch=4,
            workers=0,
            cfg=config_file,
            freeze=10,
            device=[gpu_id]
            )
        
        results[k] = model.val(
            project=project_name,
            name=f"fold_{k+1}_val",
            iou=0.5,
            max_det=60,
            save_txt=True,
            save_conf=True)
        
        with open(f"{project}/fold_{k+1}_val/validation_results.csv", "w") as f:
            f.write(results[k].to_csv())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="path to images dataset folder", required=True)
    parser.add_argument("-c", "--config", help="path to yaml configuration file", required=True)
    parser.add_argument("-m", "--model", help="model size: n s m l x", required=True)
    parser.add_argument("-g", "--gpu_id", help="gpu id (default=0)", required=False, type=int, default=0)
    parser.add_argument("-n", "--name", help="project name (the results will be saved here)", required=True)

    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        parser.error(f"Dataset path does not exist or is not a directory: {args.dataset}")

    if not os.path.isfile(args.config):
        parser.error(f"Config file does not exist or is not a file: {args.config}")
    
    if args.model not in "nsmlx":
        parser.error(f"Incorrect model size: {args.model}. Should be n, s, m, l or x")

    if args.gpu_id < 0 and args.gpu_id != -1:
        parser.error("GPU id must be 0, -1 or positive integer")
    
    ds_yamls = get_dataset(args.dataset)
    train_val(ds_yamls, args.config, args.model, args.gpu_id, args.name)

if __name__ == "__main__":
    main()

