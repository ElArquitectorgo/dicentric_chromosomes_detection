import torch
from typing import Callable

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, RANK, TQDM, colorstr
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import select_device, unwrap_model
from ultralytics.utils.checks import check_imgsz
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.autobackend import AutoBackend

from ensemble_boxes import weighted_boxes_fusion, nms
import numpy as np
import cv2

from utils import *

def apply_three_channel_fn(img_tensor: torch.Tensor, fns: list[Callable[[np.ndarray], np.ndarray] | None]):
    """img_tensor: (B, 1 o 3, H, W) float32, valores [0,1]"""
    if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.repeat(1, len(fns), 1, 1)  # (B, 1, H, W) -> (B, 3, H, W)

    imgs = (img_tensor.cpu().numpy() * 255).astype(np.uint8)  # (B, 3, H, W)
    batch = []
    for img in imgs:                                           # img: (3, H, W)
        channels = [
            img[i] if fn is None else fn(img[i])
            for i, fn in enumerate(fns)
        ]
        batch.append(np.stack(channels))                       # (3, H, W)
    return torch.from_numpy(np.stack(batch)).float().to(img_tensor.device) / 255.0

def apply_fn(img_tensor: torch.Tensor, fn: Callable[[np.ndarray], np.ndarray]):
    """img_tensor: (B, 1, H, W) float32, valores [0,1]"""
    imgs = (img_tensor.cpu().numpy() * 255).astype(np.uint8) # (B, 1, H, W)
    batch = [fn(img[0])[np.newaxis] for img in imgs]
    return torch.from_numpy(np.stack(batch)).float().to(img_tensor.device) / 255.0

def apply_lbp(img_tensor: torch.Tensor):
    return apply_fn(img_tensor, lbp)

def apply_dissimilarity(img_tensor: torch.Tensor):
    return apply_fn(img_tensor, fast_glcm_dissimilarity)

def apply_max(img_tensor: torch.Tensor):
    return apply_fn(img_tensor, fast_glcm_max)

def apply_mean(img_tensor: torch.Tensor):
    return apply_fn(img_tensor, fast_glcm_mean)

def apply_std(img_tensor: torch.Tensor):
    return apply_fn(img_tensor, fast_glcm_std)

def apply_he(img_tensor: torch.Tensor):
    return apply_fn(img_tensor, he)

def apply_clahe(img_tensor: torch.Tensor):
    return apply_fn(img_tensor, clahe)

def apply_lt(img_tensor: torch.Tensor):
    return apply_fn(img_tensor, lt)

def apply_no2(img_tensor: torch.Tensor):
    return apply_fn(img_tensor, fn=lambda img: otsu(img, "open", 2))

def apply_nc5(img_tensor: torch.Tensor):
    return apply_fn(img_tensor, fn=lambda img: otsu(img, "close", 5))

def apply_raw_clahe_lt(img_tensor: torch.Tensor):
    return apply_three_channel_fn(img_tensor, fns=[None, clahe, lt])

def apply_raw_lt_std(img_tensor: torch.Tensor):
    return apply_three_channel_fn(img_tensor, fns=[None, lt, fast_glcm_std])

def apply_raw_max_std(img_tensor: torch.Tensor):
    return apply_three_channel_fn(img_tensor, fns=[None, fast_glcm_max, fast_glcm_std])

def apply_lt_clahe_std(img_tensor: torch.Tensor):
    return apply_three_channel_fn(img_tensor, fns=[lt, clahe, fast_glcm_std])

def val(paths, transforms, project_name, consensus="wbf", box_thr=0.1):
    for k in range(5):

        model_paths = [
            f"runs/detect/{p}/fold_{k+1}_train/weights/best.pt" for p in paths
        ]

        models = []

        args = dict(data="folds/test/raw/data.yaml", project=project_name, mode="val", name=f"fold_{k+1}_val",)
        validator = DetectionValidator(args=args)
        validator.training = False

        for model_path in model_paths:
            model = AutoBackend(
                        model=model_path,
                        device=select_device(validator.args.device) if RANK == -1 else torch.device("cuda", RANK),
                        dnn=validator.args.dnn,
                        data=validator.args.data,
                        fp16=validator.args.half,
                    )

            if model.model.end2end:
                validator.end2end = True
                model.model.set_head_attr(max_det=60, agnostic_nms=False)

            models.append(model)

        validator.device = select_device(validator.args.device) if RANK == -1 else torch.device("cuda", RANK)

        validator.data = check_det_dataset(validator.args.data)

        validator.stride = 32 # used in get_dataloader() for padding
        validator.args.rect = True # In training this is false
        validator.dataloader = validator.dataloader or validator.get_dataloader(validator.data.get(validator.args.split), validator.args.batch)

        for model in models:
            model.eval()

        dt = (
            Profile(device=validator.device),
            Profile(device=validator.device),
            Profile(device=validator.device),
            Profile(device=validator.device),
        )

        bar = TQDM(validator.dataloader, desc=validator.get_desc(), total=len(validator.dataloader))
        validator.init_metrics(unwrap_model(models[0])) # We are not compiling models yet
        validator.jdict = []  # empty before each val

        for batch_i, batch in enumerate(bar):
            validator.batch_i = batch_i
            with dt[0]:
                batch = validator.preprocess(batch)

            with dt[1]:
                #preds = [model(batch["img"]) for model in models]
                preds = []
                for model, transform in zip(models, transforms):
                    img = transform(batch["img"]) if transform is not None else batch["img"]
                    preds.append(model(img))

            with dt[3]:
                preds = [validator.postprocess(pred) for pred in preds]

            batch_size = batch["img"].shape[0]
            img_h, img_w = batch["img"].shape[2], batch["img"].shape[3]
            divisor = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=validator.device)

            result = []
            for img_i in range(batch_size):
                labels_i = [pred[img_i]['cls'].cpu().numpy() for pred in preds]
                scores_i = [pred[img_i]['conf'].cpu().numpy() for pred in preds]
                boxes_i  = [pred[img_i]['bboxes'].cpu().numpy() / np.array([img_w, img_h, img_w, img_h]) for pred in preds]
                boxes_i  = [b.clip(0, 1) for b in boxes_i]

                if consensus == "wbf":
                    boxes_f, scores_f, labels_f = weighted_boxes_fusion(
                        boxes_i, scores_i, labels_i,
                        weights=[1 for _ in models],
                        iou_thr=0.7,
                        skip_box_thr=box_thr
                    )
                elif consensus == "nms":
                    boxes_f, scores_f, labels_f = nms(
                        boxes_i, scores_i, labels_i,
                        weights=[1 for _ in models],
                        iou_thr=0.7,
                    )

                result.append({
                    'bboxes': torch.from_numpy(boxes_f).to(validator.device) * divisor,
                    'conf':   torch.from_numpy(scores_f).to(validator.device),
                    'cls':    torch.from_numpy(labels_f).to(validator.device).float(),
                    'extra':  torch.zeros((len(boxes_f), 0), device=validator.device)
                })

            validator.update_metrics(result, batch)
            if validator.args.plots and batch_i < 3 and RANK in {-1, 0}:
                validator.plot_val_samples(batch, batch_i)
                validator.plot_predictions(batch, result, batch_i)


        stats = {}
        validator.gather_stats()
        if RANK in {-1, 0}:
            stats = validator.get_stats()
            validator.speed = dict(zip(validator.speed.keys(), (x.t / len(validator.dataloader.dataset) * 1e3 for x in dt)))
            validator.finalize_metrics()
            validator.print_results()

        if RANK > 0:
            print(stats)
            exit()

        LOGGER.info(
            "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                *tuple(validator.speed.values())
            )
        )
        if validator.args.save_json and validator.jdict:
            with open(str(validator.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
                LOGGER.info(f"Saving {f.name}...")
                json.dump(validator.jdict, f)  # flatten and save
            stats = validator.eval_json(stats)  # update stats
        if validator.args.plots or validator.args.save_json:
            LOGGER.info(f"Results saved to {colorstr('bold', validator.save_dir)}")
        print(stats)

        with open(f"runs/detect/{project_name}/fold_{k+1}_val/test_validation_results.csv", "w") as f:
            f.write(validator.metrics.to_csv())


val(paths=["fr_raw_x_auto", "fr_lt_x_auto", "fr_std_x_auto"], transforms=[None, apply_lt, apply_std], project_name="fr_mix_raw_lt_std_wbf01", consensus="wbf")
val(paths=["fr_raw_x_auto", "fr_clahe_x_auto", "fr_lt_x_auto"], transforms=[None, apply_clahe, apply_lt], project_name="fr_mix_raw_clahe_lt_wbf01", consensus="wbf")
val(paths=["fr_raw_x_auto", "fr_max_x_auto", "fr_std_x_auto"], transforms=[None, apply_max, apply_std], project_name="fr_mix_raw_max_std_wbf01", consensus="wbf")
val(paths=["fr_lt_x_auto", "fr_clahe_x_auto", "fr_std_x_auto"], transforms=[apply_lt, apply_clahe, apply_std], project_name="fr_mix_lt_clahe_std_wbf01", consensus="wbf")
val(paths=["fr_raw_x_auto", "fr_clahe_x_auto", "fr_std_x_auto"], transforms=[None, apply_clahe, apply_std], project_name="fr_mix_raw_clahe_std_wbf01", consensus="wbf")
val(paths=["fr_raw_x_auto", "fr_no2_x_auto", "fr_nc5_x_auto"], transforms=[None, apply_no2, apply_nc5], project_name="fr_mix_raw_no2_nc5_wbf01", consensus="wbf")
val(paths=["fr_raw_x_auto", "fr_lbp_x_auto", "fr_std_x_auto"], transforms=[None, apply_lbp, apply_std], project_name="fr_mix_raw_lbp_std_wbf01", consensus="wbf")
val(paths=["fr_lt_x_auto", "fr_clahe_x_auto", "fr_he_x_auto"], transforms=[apply_lt, apply_clahe, apply_he], project_name="fr_mix_lt_clahe_he_wbf01", consensus="wbf")
val(paths=["fr_lbp_x_auto", "fr_max_x_auto", "fr_std_x_auto"], transforms=[apply_lbp, apply_max, apply_std], project_name="fr_mix_lbp_max_std_wbf01", consensus="wbf")
val(paths=["fr_lbp_x_auto", "fr_mean_x_auto", "fr_std_x_auto"], transforms=[apply_lbp, apply_mean, apply_std], project_name="fr_mix_lbp_mean_std_wbf01", consensus="wbf")

val(paths=["fr_raw_lt_std_x_auto", "fr_raw_clahe_lt_x_auto", "fr_lt_clahe_std_x_auto"], transforms=[apply_raw_lt_std, apply_raw_clahe_lt, apply_lt_clahe_std], project_name="fr_mix_topaa_wbf01", consensus="wbf")
val(paths=["fr_raw_lt_std_x_auto", "fr_raw_clahe_lt_x_auto", "fr_raw_max_std_x_auto"], transforms=[apply_raw_lt_std, apply_raw_clahe_lt, apply_raw_max_std], project_name="fr_mix_topdic_wbf01", consensus="wbf")
val(paths=["fr_lt_clahe_std_x_auto", "fr_raw_x_auto", "fr_raw_clahe_lt_x_auto"], transforms=[apply_lt_clahe_std, None, apply_raw_clahe_lt], project_name="fr_mix_topnondic_wbf01", consensus="wbf")




val(paths=["fr_raw_x_auto", "fr_lt_x_auto", "fr_std_x_auto"], transforms=[None, apply_lt, apply_std], project_name="fr_mix_raw_lt_std_nms", consensus="nms")
val(paths=["fr_raw_x_auto", "fr_clahe_x_auto", "fr_lt_x_auto"], transforms=[None, apply_clahe, apply_lt], project_name="fr_mix_raw_clahe_lt_nms", consensus="nms")
val(paths=["fr_raw_x_auto", "fr_max_x_auto", "fr_std_x_auto"], transforms=[None, apply_max, apply_std], project_name="fr_mix_raw_max_std_nms", consensus="nms")
val(paths=["fr_lt_x_auto", "fr_clahe_x_auto", "fr_std_x_auto"], transforms=[apply_lt, apply_clahe, apply_std], project_name="fr_mix_lt_clahe_std_nms", consensus="nms")
val(paths=["fr_raw_x_auto", "fr_clahe_x_auto", "fr_std_x_auto"], transforms=[None, apply_clahe, apply_std], project_name="fr_mix_raw_clahe_std_nms", consensus="nms")
val(paths=["fr_raw_x_auto", "fr_no2_x_auto", "fr_nc5_x_auto"], transforms=[None, apply_no2, apply_nc5], project_name="fr_mix_raw_no2_nc5_nms", consensus="nms")
val(paths=["fr_raw_x_auto", "fr_lbp_x_auto", "fr_std_x_auto"], transforms=[None, apply_lbp, apply_std], project_name="fr_mix_raw_lbp_std_nms", consensus="nms")
val(paths=["fr_lt_x_auto", "fr_clahe_x_auto", "fr_he_x_auto"], transforms=[apply_lt, apply_clahe, apply_he], project_name="fr_mix_lt_clahe_he_nms", consensus="nms")
val(paths=["fr_lbp_x_auto", "fr_max_x_auto", "fr_std_x_auto"], transforms=[apply_lbp, apply_max, apply_std], project_name="fr_mix_lbp_max_std_nms", consensus="nms")
val(paths=["fr_lbp_x_auto", "fr_mean_x_auto", "fr_std_x_auto"], transforms=[apply_lbp, apply_mean, apply_std], project_name="fr_mix_lbp_mean_std_nms", consensus="nms")

val(paths=["fr_raw_lt_std_x_auto", "fr_raw_clahe_lt_x_auto", "fr_lt_clahe_std_x_auto"], transforms=[apply_raw_lt_std, apply_raw_clahe_lt, apply_lt_clahe_std], project_name="fr_mix_topaa_nms", consensus="nms")
val(paths=["fr_raw_lt_std_x_auto", "fr_raw_clahe_lt_x_auto", "fr_raw_max_std_x_auto"], transforms=[apply_raw_lt_std, apply_raw_clahe_lt, apply_raw_max_std], project_name="fr_mix_topdic_nms", consensus="nms")
val(paths=["fr_lt_clahe_std_x_auto", "fr_raw_x_auto", "fr_raw_clahe_lt_x_auto"], transforms=[apply_lt_clahe_std, None, apply_raw_clahe_lt], project_name="fr_mix_topnondic_nms", consensus="nms")