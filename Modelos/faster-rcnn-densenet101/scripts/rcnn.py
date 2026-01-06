import os
import json
import cv2
import torch
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

IMG_DIR   = "/home/messyas/ml/data/tccdata/images"
ANN_FILE  = "/home/messyas/ml/data/tccdata/annotations/4_class_splits/annotations_test.json"
RCNN_DIR  = "/home/messyas/ml/tcc/models/combinacao/rcnn"
CFG_FILE  = os.path.join(RCNN_DIR, "config.yaml")
OUT_JSON  = "frcnn_gpu_fallback_preds.json"


def main() -> None:
    print("Carregando anotações de teste…")
    coco_gt = COCO(ANN_FILE)

    print("Carregando configuração e pesos…")
    cfg = get_cfg()
    cfg.merge_from_file(CFG_FILE)      
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

    try:
        cfg.MODEL.DEVICE = "cuda"
        predictor = DefaultPredictor(cfg)
        print("Inferência na GPU habilitada.")
    except RuntimeError as exc:
        print("GPU falhou – mudando para CPU:", exc)
        cfg.MODEL.DEVICE = "cpu"
        predictor = DefaultPredictor(cfg)

    results: list[dict] = []
    total_boxes = 0

    print("Rodando inferência…")
    for img_info in tqdm(coco_gt.loadImgs(coco_gt.getImgIds()), desc="FRCNN"):
        img_id   = img_info["id"]
        img_path = os.path.join(IMG_DIR, img_info["file_name"])
        img_bgr  = cv2.imread(img_path)
        if img_bgr is None:
            continue

        with torch.no_grad():
            outputs = predictor(img_bgr)["instances"]

        inst = outputs.to("cpu")
        total_boxes += len(inst)

        for box, score, cls in zip(
            inst.pred_boxes.tensor.numpy(),
            inst.scores.numpy(),
            inst.pred_classes.numpy(),
        ):
            x1, y1, x2, y2 = box
            results.append(
                {
                    "image_id":    img_id,
                    "category_id": int(cls) + 1,     
                    "bbox":        [float(x1), float(y1),
                                     float(x2 - x1), float(y2 - y1)],
                    "score":       float(score),
                }
            )
        del outputs, inst
        if cfg.MODEL.DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"Total de caixas geradas: {total_boxes}")
    if total_boxes == 0:
        print("Nenhuma detecção – verifique config.yaml, pesos ou thresholds.")
        return

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print("Predições salvas em:", OUT_JSON)

    print("Executando COCOeval…")
    coco_dt   = coco_gt.loadRes(OUT_JSON)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    main()
