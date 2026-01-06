import os
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import random
import csv
as
import matplotlib
matplotlib.use('Agg') 

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode, Boxes, pairwise_iou
from detectron2.utils.visualizer import Visualizer, ColorMode

from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score


IMG_DIR = "/home/messyas/ml/data/tccdata/images"
ANN_TEST = "/home/messyas/ml/data/tccdata/annotations/4_class_splits/annotations_test.json"
MODEL_PATH = "/home/messyas/ml/pesos/retinanet/retinanet.pth"
OUTPUT_DIR = "./output_test_retinanet"
DATASET_TEST_NAME = "my_dataset_test_retinanet" 
NUM_CLASSES = 4
IOU_THRESHOLD = 0.5  
SCORE_THRESHOLD = 0.5 

try:
    DatasetCatalog.remove(DATASET_TEST_NAME)
    MetadataCatalog.remove(DATASET_TEST_NAME)
except KeyError:
    pass 

register_coco_instances(DATASET_TEST_NAME, {}, ANN_TEST, IMG_DIR)
test_metadata = MetadataCatalog.get(DATASET_TEST_NAME)
dataset_dicts = DatasetCatalog.get(DATASET_TEST_NAME)
class_names = test_metadata.thing_classes
print(f"Dataset de teste '{DATASET_TEST_NAME}' registrado com {len(dataset_dicts)} imagens.")
print(f"Classes: {class_names}")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))

cfg.MODEL.WEIGHTS = MODEL_PATH
cfg.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD 
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = SCORE_THRESHOLD

cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.DATASETS.TEST = (DATASET_TEST_NAME,)
cfg.OUTPUT_DIR = OUTPUT_DIR
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
torch.cuda.empty_cache()

predictor = DefaultPredictor(cfg)
print(f"Preditor criado. Usando dispositivo: {cfg.MODEL.DEVICE}")
print("Iniciando avaliacao.")
evaluator = COCOEvaluator(DATASET_TEST_NAME, cfg, False, output_dir=OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, DATASET_TEST_NAME)
coco_results = inference_on_dataset(predictor.model, val_loader, evaluator)
print("\nResultados:")
print(coco_results)

map_coco = coco_results['bbox']['AP']
map50_coco = coco_results['bbox']['AP50']
map75_coco = coco_results['bbox']['AP75']

def get_gt_and_preds_for_cm(predictor, dataset_dicts, num_classes, iou_thresh, score_thresh):
    y_true = []
    y_pred = []
    bg_class_id = num_classes 

    for d in dataset_dicts:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)["instances"].to("cpu")

        gt_boxes = Boxes([BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in d["annotations"]])
        gt_classes = torch.tensor([obj["category_id"] for obj in d["annotations"]])
        gt_matched = [False] * len(gt_boxes)

        pred_boxes = outputs.pred_boxes
        pred_scores = outputs.scores
        pred_classes = outputs.pred_classes

        keep = pred_scores >= score_thresh
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_classes = pred_classes[keep]
        
        sorted_idx = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_idx]
        pred_classes = pred_classes[sorted_idx]

        if len(gt_boxes) == 0:
            for p_cls in pred_classes.tolist():
                y_true.append(bg_class_id)
                y_pred.append(p_cls)
            continue

        if len(pred_boxes) == 0:
            for g_cls in gt_classes.tolist():
                y_true.append(g_cls)
                y_pred.append(bg_class_id)
            continue

        iou_matrix = pairwise_iou(pred_boxes, gt_boxes)

        for p_idx in range(len(pred_boxes)):
            p_cls = pred_classes[p_idx].item()
            ious = iou_matrix[p_idx, :]
            best_gt_idx = torch.argmax(ious).item()
            best_iou = ious[best_gt_idx].item()

            if best_iou >= iou_thresh:
                g_cls = gt_classes[best_gt_idx].item()
                if not gt_matched[best_gt_idx]:
                    y_true.append(g_cls)
                    y_pred.append(p_cls)
                    gt_matched[best_gt_idx] = True
                else:
                    y_true.append(bg_class_id)
                    y_pred.append(p_cls)
            else:
                y_true.append(bg_class_id)
                y_pred.append(p_cls)

        for gt_idx, matched in enumerate(gt_matched):
            if not matched:
                y_true.append(gt_classes[gt_idx].item())
                y_pred.append(bg_class_id)

    return y_true, y_pred

y_true, y_pred = get_gt_and_preds_for_cm(predictor, dataset_dicts, NUM_CLASSES, IOU_THRESHOLD, SCORE_THRESHOLD)

labels = list(range(NUM_CLASSES)) + [NUM_CLASSES]
cm_names = class_names + ['Background']
cm = confusion_matrix(y_true, y_pred, labels=labels)

f1_macro = f1_score(y_true, y_pred, labels=labels[:-1], average='macro', zero_division=0)
recall_macro = recall_score(y_true, y_pred, labels=labels[:-1], average='macro', zero_division=0)
precision_macro = precision_score(y_true, y_pred, labels=labels[:-1], average='macro', zero_division=0)

f1_per_class = f1_score(y_true, y_pred, labels=labels[:-1], average=None, zero_division=0)
recall_per_class = recall_score(y_true, y_pred, labels=labels[:-1], average=None, zero_division=0)
precision_per_class = precision_score(y_true, y_pred, labels=labels[:-1], average=None, zero_division=0)

print(f"F1-Score: {f1_macro:.4f}")
print(f"Recall: {recall_macro:.4f}")
print(f"Precision: {precision_macro:.4f}")
print("Plotando Matriz de Confusao")
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_names, yticklabels=cm_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix (RetinaNet - IoU={IOU_THRESHOLD}, Score={SCORE_THRESHOLD})')
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_retinanet.png") # <-- Novo nome
plt.savefig(cm_path)
print(f"Matriz Confusao salva em: {cm_path}")
plt.close() 

print("Salvando metricas em CSV...")
csv_path = os.path.join(OUTPUT_DIR, "evaluation_metrics_retinanet.csv") 
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['mAP', map_coco])
    writer.writerow(['mAP50', map50_coco])
    writer.writerow(['mAP75', map75_coco])
    writer.writerow(['---', '---'])
    writer.writerow(['F1-Score', f1_macro])
    writer.writerow(['Recall', recall_macro])
    writer.writerow(['Precision', precision_macro])
    writer.writerow(['---', '---'])
    for i, name in enumerate(class_names):
        writer.writerow([f'F1 ({name})', f1_per_class[i]])
        writer.writerow([f'Recall ({name})', recall_per_class[i]])
        writer.writerow([f'Precision ({name})', precision_per_class[i]])

print(f"Métricas salvas em: {csv_path}")
print("\nVisualizando algumas predições...")
visualize_count = 5
output_viz_dir = os.path.join(OUTPUT_DIR, "visualizations_retinanet")
os.makedirs(output_viz_dir, exist_ok=True)

for i, d in enumerate(random.sample(dataset_dicts, visualize_count)):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=test_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img_path = os.path.join(output_viz_dir, f"prediction_retinanet_{i}.jpg")
    cv2.imwrite(img_path, out.get_image()[:, :, ::-1])
    print(f"Visualização salva em: {img_path}")

print("\nScript de teste (RetinaNet) concluído!")
