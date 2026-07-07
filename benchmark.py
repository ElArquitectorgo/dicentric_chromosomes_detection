from ultralytics import YOLO
from pathlib import Path
import numpy as np
import torch
import time
import statistics
import csv
from typing import Callable, List, Union, Optional, Dict, Any
from ensemble_boxes import nms, weighted_boxes_fusion
from utils import *
from pynvml import *

nvmlInit()
_HANDLE = nvmlDeviceGetHandleByIndex(0)
_NVML_AVAILABLE = True


def get_gpu_stats() -> Dict[str, float]:
    util = nvmlDeviceGetUtilizationRates(_HANDLE)
    mem = nvmlDeviceGetMemoryInfo(_HANDLE)
    power = nvmlDeviceGetPowerUsage(_HANDLE) / 1000.0
    temp = nvmlDeviceGetTemperature(_HANDLE, NVML_TEMPERATURE_GPU)
    return {
        "gpu_util": util.gpu,
        "mem_util": util.memory,
        "vram_used_gb": mem.used / 1024**3,
        "vram_total_gb": mem.total / 1024**3,
        "power_w": power,
        "temp_c": temp,
    }

def load_images(img_dir) -> List[np.ndarray]:
    img_dir = Path(img_dir)
    npy_files = sorted(img_dir.glob("*.npy"))
    return [np.load(f) for f in npy_files]

def make_preprocessor(fn_spec) -> Callable[[np.ndarray], np.ndarray]:
    if fn_spec is None:
        return lambda x: x
    if callable(fn_spec):
        return fn_spec
    if isinstance(fn_spec, list):
        def preprocessor(img: np.ndarray) -> np.ndarray:
            channels = [fn(img) for fn in fn_spec]
            return np.stack(channels, axis=-1)
        return preprocessor
    raise ValueError("fn_spec must be None, Callable o List[Callable]")


def _run_benchmark(inference_fn: Callable[[], None], repeats: int, warmup_repeats: int = 3) -> Dict[str, Any]:
    print("Warming up...")
    for _ in range(warmup_repeats):
        inference_fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    gpu_utils = []
    mem_used = []
    powers = []
    peak_memory = 0.0

    print("Running benchmark...")
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()

        inference_fn()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        stats = get_gpu_stats()
        gpu_utils.append(stats["gpu_util"])
        mem_used.append(stats["vram_used_gb"])
        powers.append(stats["power_w"])

        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / 1024**3)

    return {
        "times": times,
        "gpu_utils": gpu_utils,
        "mem_used": mem_used,
        "powers": powers,
        "peak_memory_gb": peak_memory,
    }

def _collect_results(experiment_name: str, fold: int, repeats: int, num_images: int, results: Dict[str, Any]) -> Dict[str, Any]:
    times = results["times"]
    return {
        "method": experiment_name,
        "fold": fold,
        "repeats": repeats,
        "num_images": num_images,
        "mean_latency_ms": statistics.mean(times) * 1000,
        "median_latency_ms": statistics.median(times) * 1000,
        "min_latency_ms": min(times) * 1000,
        "max_latency_ms": max(times) * 1000,
        "fps": num_images / statistics.mean(times),
        "gpu_util_avg_percent": statistics.mean(results["gpu_utils"]),
        "vram_avg_gb": statistics.mean(results["mem_used"]),
        "vram_peak_gb": results["peak_memory_gb"],
        "power_avg_w": statistics.mean(results["powers"]),
    }

# ----------------------------------------------------------------------
def benchmark(
    project_name: str,
    fns = None,
    fold: int = 1,
    repeats: int = 50,
    img_dir = "folds/test/raw/val/images/",
    experiment_name: str = None,
) -> Dict[str, Any]:
    if experiment_name is None:
        experiment_name = project_name

    model = YOLO(f"runs/detect/{project_name}/fold_{fold}_train/weights/best.pt")
    preprocess = make_preprocessor(fns)

    images = load_images(img_dir)
    processed_images = [preprocess(img) for img in images]

    def inference_iteration():
        for img in processed_images:
            model(img, max_det=60, save=False, verbose=False)

    raw_results = _run_benchmark(inference_iteration, repeats)
    return _collect_results(experiment_name, fold, repeats, len(images), raw_results)

def benchmark_consensus(
    project_names: List[str],
    fns,
    consensus: str = "nms",
    fold: int = 1,
    repeats: int = 50,
    img_dir = "folds/test/raw/val/images/",
    iou_thr: float = 0.7,
    skip_box_thr: float = 0.1,
    experiment_name: str = None,
) -> Dict[str, Any]:
    if experiment_name is None:
        experiment_name = f"ensemble_{'_'.join(project_names)}_{consensus}"

    num_models = len(project_names)
    models = []
    preprocessors = []
    for name, fn_spec in zip(project_names, fns):
        models.append(YOLO(f"runs/detect/{name}/fold_{fold}_train/weights/best.pt"))
        preprocessors.append(make_preprocessor(fn_spec))

    raw_images = load_images(img_dir)
    processed_images_per_model = []
    for preproc in preprocessors:
        proc_list = [preproc(img) for img in raw_images]
        processed_images_per_model.append(proc_list)

    if consensus == "nms":
        def fuse_boxes(boxes_list, scores_list, labels_list):
            return nms(boxes_list, scores_list, labels_list,
                       weights=[1.0] * num_models, iou_thr=iou_thr)
    elif consensus == "wbf":
        def fuse_boxes(boxes_list, scores_list, labels_list):
            return weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                                         weights=[1.0] * num_models,
                                         iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    else:
        raise ValueError("consensus must be 'nms' o 'wbf'")

    def inference_iteration():
        for img_idx in range(len(raw_images)):
            results_per_model = []
            for model_idx, model in enumerate(models):
                img = processed_images_per_model[model_idx][img_idx]
                res = model(img, max_det=60, save=False, verbose=False)[0]
                results_per_model.append(res)

            boxes_list = [r.boxes.xyxyn.cpu() for r in results_per_model]
            scores_list = [r.boxes.conf.cpu() for r in results_per_model]
            labels_list = [r.boxes.cls.cpu() for r in results_per_model]
            _ = fuse_boxes(boxes_list, scores_list, labels_list)

    raw_results = _run_benchmark(inference_iteration, repeats)
    return _collect_results(experiment_name, fold, repeats, len(raw_images), raw_results)

def save_results_to_csv(results_list: List[Dict[str, Any]], csv_path: str = "benchmark_results.csv"):
    if not results_list:
        return
    fieldnames = list(results_list[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_list)
    print(f"\nSaved in {csv_path}")

if __name__ == "__main__":
    all_results = []

    single_benchmarks = [
        ("fr_raw_x_auto", raw),
        ("fr_std_x_auto", fast_glcm_std),
        ("fr_max_x_auto", fast_glcm_max),
        ("fr_clahe_x_auto", clahe),
        ("fr_lt_x_auto", lt),
        ("fr_lbp_x_auto", lbp),
        ("fr_lbp_max_std_x_auto", [lbp, fast_glcm_max, fast_glcm_std]),
        ("fr_raw_max_std_x_auto", [raw, fast_glcm_max, fast_glcm_std]),
        ("fr_raw_clahe_std_x_auto", [raw, clahe, fast_glcm_std]),
        ("fr_raw_lt_std_x_auto", [raw, lt, fast_glcm_std]),
        ("fr_raw_clahe_lt_x_auto", [raw, clahe, lt]),
        ("fr_raw_lbp_std_x_auto", [raw, lbp, fast_glcm_std]),
        ("fr_raw_no2_nc5_x_auto", [raw, lambda x: otsu(x, "open", 2), lambda x: otsu(x, "close", 5)]),
        ("fr_lt_clahe_he_x_auto", [lt, clahe, he]),
        ("fr_lt_clahe_std_x_auto", [lt, clahe, fast_glcm_std]),
        ("fr_lbp_mean_std_x_auto", [lbp, fast_glcm_mean, fast_glcm_std]),
    ]
    for proj, fns in single_benchmarks:
        print(f"\n>>> Benchmarking {proj}")
        res = benchmark(proj, fns=fns, fold=1, repeats=100)
        all_results.append(res)

    simple_consensus = [
        (["fr_raw_x_auto", "fr_clahe_x_auto", "fr_lt_x_auto"], [raw, clahe, lt], "nms", "c_raw_clahe_lt_nms"),
        (["fr_raw_x_auto", "fr_clahe_x_auto", "fr_lt_x_auto"], [raw, clahe, lt], "wbf", "c_raw_clahe_lt_wbf"),
        (["fr_raw_x_auto", "fr_lt_x_auto", "fr_std_x_auto"], [raw, lt, fast_glcm_std], "wbf", "c_raw_lt_std_wbf"),
        (["fr_raw_x_auto", "fr_lt_x_auto", "fr_std_x_auto"], [raw, lt, fast_glcm_std], "nms", "c_raw_lt_std_nms"),
        (["fr_raw_x_auto", "fr_clahe_x_auto", "fr_std_x_auto"], [raw, clahe, fast_glcm_std], "wbf", "c_raw_clahe_std_wbf"),
        (["fr_raw_x_auto", "fr_clahe_x_auto", "fr_std_x_auto"], [raw, clahe, fast_glcm_std], "nms", "c_raw_clahe_std_nms"),
        (["fr_raw_x_auto", "fr_no2_x_auto", "fr_nc5_x_auto"],
         [raw, lambda x: otsu(x, "open", 2), lambda x: otsu(x, "close", 5)], "wbf", "c_raw_no2_nc5_wbf"),
        (["fr_raw_x_auto", "fr_no2_x_auto", "fr_nc5_x_auto"],
         [raw, lambda x: otsu(x, "open", 2), lambda x: otsu(x, "close", 5)], "nms", "c_raw_no2_nc5_nms"),
        (["fr_lt_x_auto", "fr_clahe_x_auto", "fr_std_x_auto"], [lt, clahe, fast_glcm_std], "nms", "c_lt_clahe_std_nms"),
        (["fr_lt_x_auto", "fr_clahe_x_auto", "fr_std_x_auto"], [lt, clahe, fast_glcm_std], "wbf", "c_lt_clahe_std_wbf"),
        (["fr_lt_x_auto", "fr_clahe_x_auto", "fr_he_x_auto"], [lt, clahe, he], "nms", "c_lt_clahe_he_nms"),
    ]
    for proj_names, fns_list, cons, exp_name in simple_consensus:
        print(f"\n>>> Benchmarking consensus {exp_name}")
        res = benchmark_consensus(proj_names, fns_list, consensus=cons, fold=1, repeats=100, experiment_name=exp_name)
        all_results.append(res)

    special_consensus = [
        (["fr_raw_clahe_lt_x_auto", "fr_raw_x_auto", "fr_lt_clahe_std_x_auto"],
         [[raw, clahe, lt], raw, [lt, clahe, fast_glcm_std]], "nms", "top_non_dic_nms"),
        (["fr_raw_clahe_lt_x_auto", "fr_raw_x_auto", "fr_lt_clahe_std_x_auto"],
         [[raw, clahe, lt], raw, [lt, clahe, fast_glcm_std]], "wbf", "top_non_dic_wbf"),
        # top aa
        (["fr_raw_lt_std_x_auto", "fr_raw_clahe_lt_x_auto", "fr_lt_clahe_std_x_auto"],
         [[raw, lt, fast_glcm_std], [raw, clahe, lt], [lt, clahe, fast_glcm_std]], "wbf", "top_aa_wbf"),
        (["fr_raw_lt_std_x_auto", "fr_raw_clahe_lt_x_auto", "fr_lt_clahe_std_x_auto"],
         [[raw, lt, fast_glcm_std], [raw, clahe, lt], [lt, clahe, fast_glcm_std]], "nms", "top_aa_nms"),
        # top dic
        (["fr_raw_lt_std_x_auto", "fr_raw_clahe_lt_x_auto", "fr_raw_max_std_x_auto"],
         [[raw, lt, fast_glcm_std], [raw, clahe, lt], [raw, fast_glcm_std, fast_glcm_max]], "wbf", "top_dic_wbf"),
        (["fr_raw_lt_std_x_auto", "fr_raw_clahe_lt_x_auto", "fr_raw_max_std_x_auto"],
         [[raw, lt, fast_glcm_std], [raw, clahe, lt], [raw, fast_glcm_std, fast_glcm_max]], "nms", "top_dic_nms"),
    ]
    for proj_names, fns_list, cons, exp_name in special_consensus:
        print(f"\n>>> Benchmarking consensus {exp_name}")
        res = benchmark_consensus(proj_names, fns_list, consensus=cons, fold=1, repeats=100, experiment_name=exp_name)
        all_results.append(res)

    save_results_to_csv(all_results, "benchmark_results.csv")