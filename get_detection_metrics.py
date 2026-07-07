from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import os

def get_detection_results(project: Path, split: str, compute_mean: bool, target: int):
    detection_results = []
    file_name = "validation_results.csv" if split == "val" else "test_validation_results.csv"

    for folder in project.glob("*val"):
        dt = pd.read_csv(project / folder.name / file_name).values
        dt = dt[:, 3:] # Remove class name, image number and instances cols
        
        dt = np.mean(dt, axis=0) if compute_mean else dt[target] # If not, return specified class values
        detection_results.append(dt)

    return detection_results

def aggregate_metrics(metrics):
    df = pd.DataFrame(metrics, columns=["Box-P", "Box-R", "Box-F1", "mAP50", "mAP50-95"])
    out = {}

    for col in df.columns:
        out[col] = df[col].mean()
        out[f"{col}_std"] = df[col].std()
    
    return out

def get_group(name: str):
    name = name.split("_x_")[0][3:]

    if name == "raw":
        group = "base"
    elif name in ["lbp", "asm", "contrast", "dissimilarity", "energy", "entropy", "homogeneity", "max", "mean", "std"]:
        group = "texture"
    elif name in ["he", "clahe", "gc", "lt"]:
        group = "enhancement"
    elif name in ["nc5", "no2"]:
        group = "bmo"
    elif name in ["raw_clahe_lt", "raw_max_std", "raw_no2_nc5", "raw_clahe_std", "raw_lt_std", "raw_lbp_std", "lt_clahe_he", "lt_clahe_std", "lbp_mean_std", "lbp_max_std"]:
        group = "multi"
    else:
        group = "consensus"

    return group

def get_experiment_metrics(input_dir: Path, split: str, compute_mean: bool, target: int=1):
    rows = []
    
    for project in input_dir.glob("*"):
        group = get_group(project.name)
        if split == "val" and group == "consensus":
            continue

        split_metrics = get_detection_results(project, split, compute_mean, target)
        agg = aggregate_metrics(split_metrics)
        rows.append({
            "method": project.name.split("_base")[0],
            "method_group": group,
            **agg
        })


    df = pd.DataFrame(rows)
    return df.sort_values("Box-F1", ascending=False)

def main():
    input_dir = Path("runs/detect")
    output_dir = Path("metrics")
    for split in ["val", "test"]:
        df = get_experiment_metrics(input_dir, split, True).round(3)
        df.to_csv(f"{output_dir}/{split}_detection_metrics.csv", index=False)

        for target in zip(["non_dicentric", "dicentric"], [0, 1]):
            df = get_experiment_metrics(input_dir, split, False, target[1]).round(3)
            df.to_csv(f"{output_dir}/{split}_{target[0]}_detection_metrics.csv", index=False)

if __name__ == "__main__":
    main()
