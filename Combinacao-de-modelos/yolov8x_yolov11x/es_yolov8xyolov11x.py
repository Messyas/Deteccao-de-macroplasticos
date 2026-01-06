import os
import json
import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import csv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_DIR = "/home/messyas/ml/data/tccdata/images"
ANN_FILE = "/home/messyas/ml/data/tccdata/annotations/4_class_splits/annotations_test.json"

# Modelo 1: YOLOv8x
YOLOv8_WEIGHTS = "/home/messyas/ml/tcc/models/combinacao/yolo/best.pt"
YOLOv8_CONF_THR = 0.005 

# Modelo 2: YOLOv11x
YOLOv11_WEIGHTS = "/home/messyas/ml/jetson/runs/detect/yolov11xtrain2/weights/best.pt"
YOLOv11_CONF_THR = 0.005 

# Ensemble
WBF_IOU_THR = 0.45
WBF_SKIP_THR = 0.001
WBF_WEIGHTS = [1.0, 1.0] 

FINAL_SCORE_THRESHOLD = 0.35

OUT_JSON_FILENAME = "ensemble_preds_yolov8x_yolov11x_gpu.json"
OUTPUT_IMG_DIR = "predicted_images_yolov8x_yolov11x_gpu"
CSV_METRICS_FILENAME = "ensemble_metrics_yolov8x_yolov11x_gpu.csv"
NUM_IMAGES_TO_DEBUG = 3 
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

def predict_yolo(model: YOLO, img_rgb, conf_thr, iou_thr):
    out = model(img_rgb, conf=conf_thr, iou=iou_thr)[0]
    h, w = img_rgb.shape[:2]
    boxes = (out.boxes.xyxy.cpu().numpy() / [w, h, w, h]).tolist() if hasattr(out.boxes, 'xyxy') and out.boxes.xyxy is not None else []
    scores = out.boxes.conf.cpu().tolist() if hasattr(out.boxes, 'conf') and out.boxes.conf is not None else []
    labels = out.boxes.cls.cpu().tolist() if hasattr(out.boxes, 'cls') and out.boxes.cls is not None else []
    return boxes, scores, labels

def fuse_predictions(yolov8_res, yolov11_res):
    all_boxes_lists = []
    all_scores_lists = []
    all_labels_lists = []

    if yolov8_res[0]:
        all_boxes_lists.append(yolov8_res[0])
        all_scores_lists.append(yolov8_res[1])
        all_labels_lists.append(yolov8_res[2])

    if yolov11_res[0]:
        all_boxes_lists.append(yolov11_res[0])
        all_scores_lists.append(yolov11_res[1])
        all_labels_lists.append(yolov11_res[2])

    if not all_boxes_lists:
        return [], [], []

    wbf_weights_to_use = []
    if yolov8_res[0] and yolov11_res[0]:
        wbf_weights_to_use = WBF_WEIGHTS
    elif yolov8_res[0]:
        wbf_weights_to_use = [WBF_WEIGHTS[0]]
    elif yolov11_res[0]:
        wbf_weights_to_use = [WBF_WEIGHTS[1]]
    else:
        wbf_weights_to_use = None
        
    try:
        boxes, scores, labels = weighted_boxes_fusion(
            all_boxes_lists, all_scores_lists, all_labels_lists,
            iou_thr=WBF_IOU_THR, weights=wbf_weights_to_use, skip_box_thr=WBF_SKIP_THR
        )
        return boxes.tolist(), scores.tolist(), labels.tolist()
    except Exception:
        return [], [], []

def plot_and_save_predictions(img_rgb, filename, predictions_for_plot, class_names):
    img_to_draw = img_rgb.copy()
    box_colors_cycle = [(0, 200, 0), (255, 100, 0), (0, 100, 255), (200, 0, 200)]
    text_color_on_bg = (255, 255, 255)
    thickness = 2
    font_scale = 0.5
    font_thickness = 1

    for i, (box_coords, class_idx, score_str) in enumerate(predictions_for_plot):
        x1, y1, x2, y2 = map(int, box_coords)
        class_name = class_names.get(int(class_idx), f'ID {int(class_idx)}')
        label_text = f"{class_name}: {score_str}"
        current_box_color = box_colors_cycle[int(class_idx) % len(box_colors_cycle)]
        cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), current_box_color, thickness)
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_bg_y1 = y1 - text_height - baseline - 5
        text_y_cv = y1 - baseline - 3
        if text_bg_y1 < 0:
            text_bg_y1 = y2 + baseline + 5
            text_y_cv = y2 + text_height + baseline + 3
        cv2.rectangle(img_to_draw, (x1, text_bg_y1), (x1 + text_width, text_bg_y1 + text_height + baseline), current_box_color, cv2.FILLED)
        cv2.putText(img_to_draw, label_text, (x1, text_y_cv), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color_on_bg, font_thickness, cv2.LINE_AA)

    output_path = os.path.join(OUTPUT_IMG_DIR, filename)
    cv2.imwrite(output_path, cv2.cvtColor(img_to_draw, cv2.COLOR_RGB2BGR))

def debug_and_plot_all_steps(img_rgb, filename, class_names, yolov8_preds, yolov11_preds, fused_preds, img_w, img_h):
    preds_v8_to_plot = []
    for box, score, label in zip(yolov8_preds[0], yolov8_preds[1], yolov8_preds[2]):
        x1, y1, x2, y2 = box[0]*img_w, box[1]*img_h, box[2]*img_w, box[3]*img_h
        preds_v8_to_plot.append(((x1, y1, x2, y2), int(label), f"{score:.1%}"))
    plot_and_save_predictions(img_rgb, f"DEBUG_01_YOLOv8_{filename}", preds_v8_to_plot, class_names)
    
    preds_v11_to_plot = []
    for box, score, label in zip(yolov11_preds[0], yolov11_preds[1], yolov11_preds[2]):
        x1, y1, x2, y2 = box[0]*img_w, box[1]*img_h, box[2]*img_h, box[3]*img_h
        preds_v11_to_plot.append(((x1, y1, x2, y2), int(label), f"{score:.1%}"))
    plot_and_save_predictions(img_rgb, f"DEBUG_02_YOLOv11_{filename}", preds_v11_to_plot, class_names)
    
    fused_to_plot = []
    for box, score, label in zip(fused_preds[0], fused_preds[1], fused_preds[2]):
         if score >= FINAL_SCORE_THRESHOLD:
            x1, y1, x2, y2 = box[0]*img_w, box[1]*img_h, box[2]*img_w, box[3]*img_h
            fused_to_plot.append(((x1, y1, x2, y2), int(label), f"{score:.1%}"))
    plot_and_save_predictions(img_rgb, f"DEBUG_03_FUSED_{filename}", fused_to_plot, class_names)


def save_coco_metrics_to_csv(coco_eval_instance, csv_filename):
    stats = coco_eval_instance.stats
    metric_names = [
        "mAP @[ IoU=0.50:0.95 | area=all ]", "AP @[ IoU=0.50 ]",
        "AP @[ IoU=0.75 ]", "AP @[ area=small ]",
        "AP @[ area=medium ]", "AP @[ area=large ]",
        "AR @[ maxDets=1 ]", "AR @[ maxDets=10 ]",
        "AR @[ maxDets=100 ]", "AR @[ area=small ]",
        "AR @[ area=medium ]", "AR @[ area=large ]"
    ]
    metrics_to_save = {name: stat for name, stat in zip(metric_names, stats)}
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metrica', 'Valor'])
            for key, value in metrics_to_save.items():
                writer.writerow([key, f"{value:.4f}"])
        print(f"\nMétricas salvas com sucesso em: {csv_filename}")
    except IOError as e:
        print(f"Erro ao salvar o arquivo CSV de métricas: {e}")

def run_ensemble_evaluation():
    print(f"Dispositivo selecionado: {str(DEVICE).upper()}")

    try:
        coco_ground_truth = COCO(ANN_FILE)
        category_map_coco = {cat_info['id']: cat_info for cat_info in coco_ground_truth.loadCats(coco_ground_truth.getCatIds())}
    except Exception as e:
        print(f"Erro ao carregar o arquivo COCO de ground truth {ANN_FILE}: {e}")
        return

    print("Carregando modelos YOLO...")
    try:
        yolo_v8_model = YOLO(YOLOv8_WEIGHTS).to(DEVICE)
        print("Modelo YOLOv8x carregado.")
        yolo_v11_model = YOLO(YOLOv11_WEIGHTS).to(DEVICE)
        print("Modelo YOLOv11x carregado.")
        class_names = yolo_v8_model.names
    except Exception as e:
        print(f"Falha ao carregar um dos modelos YOLO. Verifique os caminhos dos pesos. Erro: {e}")
        return

    coco_results_for_eval = []
    image_id_list = coco_ground_truth.getImgIds()
    
    print(f"\nIniciando predições em {len(image_id_list)} imagens no dispositivo '{str(DEVICE).upper()}'...")
    print(f">>> Filtro de confiança para visualização e avaliação: > {FINAL_SCORE_THRESHOLD*100:.0f}% <<<")

    for i, current_img_id in enumerate(tqdm(image_id_list, desc=f"Ensemble on {str(DEVICE).upper()}")):
        img_metadata = coco_ground_truth.loadImgs(current_img_id)[0]
        img_file_path = os.path.join(IMG_DIR, img_metadata["file_name"])
        img_bgr = cv2.imread(img_file_path)
        if img_bgr is None:
            print(f"⚠️ Aviso: Não foi possível ler a imagem {img_file_path}. Pulando...")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = img_rgb.shape

        with torch.no_grad():
            yolov8_preds = predict_yolo(yolo_v8_model, img_rgb, YOLOv8_CONF_THR, WBF_IOU_THR)
            yolov11_preds = predict_yolo(yolo_v11_model, img_rgb, YOLOv11_CONF_THR, WBF_IOU_THR)
        
        if i < NUM_IMAGES_TO_DEBUG or (i % 100 == 0): 
            print(f"\n[Imagem: {img_metadata['file_name']}] -> YOLOv8: {len(yolov8_preds[0])} caixas | YOLOv11: {len(yolov11_preds[0])} caixas")
        
        fused_boxes_norm, fused_scores, fused_labels_zero_indexed = fuse_predictions(yolov8_preds, yolov11_preds)
        
        if i < NUM_IMAGES_TO_DEBUG:
            debug_and_plot_all_steps(img_rgb, img_metadata["file_name"], class_names, 
                                     yolov8_preds, yolov11_preds, 
                                     (fused_boxes_norm, fused_scores, fused_labels_zero_indexed),
                                     w_img, h_img)
        if fused_boxes_norm:
            for box_norm, score_val, label_idx in zip(fused_boxes_norm, fused_scores, fused_labels_zero_indexed):
                if score_val < FINAL_SCORE_THRESHOLD:
                    continue
                
                x1, y1, x2, y2 = box_norm[0] * w_img, box_norm[1] * h_img, box_norm[2] * w_img, box_norm[3] * h_img
                width_box, height_box = x2 - x1, y2 - y1
                class_name_pred = class_names[int(label_idx)]
                current_coco_category_id = next((cat_id for cat_id, cat_info in category_map_coco.items() if cat_info['name'] == class_name_pred), None)

                if width_box > 0 and height_box > 0 and current_coco_category_id is not None:
                    coco_results_for_eval.append({
                        "image_id": current_img_id,
                        "category_id": current_coco_category_id,
                        "bbox": [float(x1), float(y1), float(width_box), float(height_box)],
                        "score": float(score_val),
                    })
    
    if not coco_results_for_eval:
        print("\nNenhuma predição final foi gerada pelo ensemble. Avaliação COCO não será executada.")
        return

    try:
        with open(OUT_JSON_FILENAME, "w", encoding="utf-8") as f:
            json.dump(coco_results_for_eval, f, indent=2)
        print(f"\nPredições em formato COCO salvas em: {OUT_JSON_FILENAME}")
    except IOError as e:
        print(f"Erro ao salvar o arquivo JSON de predições: {e}")
        return

    print("\nExecutando COCOeval...")
    try:
        coco_detected_results = coco_ground_truth.loadRes(OUT_JSON_FILENAME)
        coco_evaluator = COCOeval(coco_ground_truth, coco_detected_results, iouType="bbox")
        coco_evaluator.evaluate()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        save_coco_metrics_to_csv(coco_evaluator, CSV_METRICS_FILENAME)
    except Exception as e:
        print(f"Erro durante a avaliação COCO: {e}")
    print("\nExecução do ensemble concluída com sucesso.")


if __name__ == "__main__":
    run_ensemble_evaluation()
