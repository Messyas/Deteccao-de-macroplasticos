import os
import json
import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from ensemble_boxes import weighted_boxes_fusion
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import csv

IMG_DIR = "/home/messyas/ml/data/tccdata/images"
ANN_FILE = "/home/messyas/ml/data/tccdata/annotations/4_class_splits/annotations_test.json"

# Modelo 1: YOLOv8x
YOLO_WEIGHTS = "/home/messyas/ml/tcc/models/combinacao/yolo/best.pt"
YOLO_CONF_THR = 0.3
YOLO_IOU_NMS_THR = 0.5

# Modelo 2: Cascade R-CNN
CASCADE_RCNN_DIR = "/home/messyas/ml/detectron/detectron2/rcascate-resnet50/pesos"
CASCADE_RCNN_WEIGHTS = os.path.join(CASCADE_RCNN_DIR, "best.pth")
CASCADE_RCNN_CFG_FILE = os.path.join(CASCADE_RCNN_DIR, "config_cascade.yaml")

FG_CLASSES = 4

# Ensemble
WBF_IOU_THR = 0.55
WBF_SKIP_THR = 0.01
WBF_WEIGHTS = [2.0, 1.0]

FINAL_SCORE_THRESHOLD = 0.40

# Saídas
OUT_JSON_FILENAME = "ensemble_preds_yolov8x_cascade_cpu.json"
OUTPUT_IMG_DIR = "predicted_images_yolov8x_cascade_cpu"
CSV_METRICS_FILENAME = "ensemble_metrics_yolov8x_cascade_cpu.csv"
NUM_GRAYSCALE_IMAGES_TO_SAVE = 5

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
# ───────────────────────────────────────────────────────────────────

def build_detectron2_predictor(cfg_file: str, weights_file: str, model_name: str) -> DefaultPredictor:
    cfg = get_cfg()
    if not os.path.exists(cfg_file):
        print(f"Arquivo de configuração não encontrado para {model_name}: {cfg_file}")
        return None
    if not os.path.exists(weights_file):
        print(f"Arquivo de pesos não encontrado para {model_name}: {weights_file}")
        return None

    cfg.merge_from_file(cfg_file)
    cfg.MODEL.WEIGHTS = weights_file
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = FG_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.DEVICE = "cpu"
    try:
        predictor = DefaultPredictor(cfg)
        print(f"Modelo {model_name} configurado para usar CPU.")
        return predictor
    except Exception as e:
        print(f"ERRO ao construir o preditor para {model_name}: {e}")
        return None

def predict_yolo(model: YOLO, img_rgb):
    out = model(img_rgb, conf=YOLO_CONF_THR, iou=YOLO_IOU_NMS_THR, device='cpu')[0]
    h, w = img_rgb.shape[:2]
    boxes = (out.boxes.xyxy.cpu().numpy() / [w, h, w, h]).tolist() if hasattr(out.boxes, 'xyxy') and out.boxes.xyxy is not None else []
    scores = out.boxes.conf.cpu().tolist() if hasattr(out.boxes, 'conf') and out.boxes.conf is not None else []
    labels = out.boxes.cls.cpu().tolist() if hasattr(out.boxes, 'cls') and out.boxes.cls is not None else []
    return boxes, scores, labels

def predict_detectron2_model(predictor: DefaultPredictor, img_rgb):
    if predictor is None:
        return [], [], []
    outputs = predictor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    instances = outputs["instances"].to("cpu")
    h, w = img_rgb.shape[:2]
    
    if not instances.has("pred_boxes") or not instances.has("scores") or not instances.has("pred_classes"):
        return [], [], []
    if len(instances.pred_boxes) == 0: 
        return [], [], []

    boxes = (instances.pred_boxes.tensor.numpy() / [w, h, w, h]).tolist()
    scores = instances.scores.numpy().tolist()
    labels = instances.pred_classes.numpy().tolist()
    return boxes, scores, labels

def fuse_predictions(yolo_res, cascade_res):
    all_boxes_lists = []
    all_scores_lists = []
    all_labels_lists = []
    current_wbf_weights = []

    if yolo_res[0] and yolo_res[1] and yolo_res[2]:
        all_boxes_lists.append(yolo_res[0])
        all_scores_lists.append(yolo_res[1])
        all_labels_lists.append(yolo_res[2])
        if WBF_WEIGHTS: current_wbf_weights.append(WBF_WEIGHTS[0])
    
    if cascade_res[0] and cascade_res[1] and cascade_res[2]:
        all_boxes_lists.append(cascade_res[0])
        all_scores_lists.append(cascade_res[1])
        all_labels_lists.append(cascade_res[2])
        if WBF_WEIGHTS and len(WBF_WEIGHTS) > 1: current_wbf_weights.append(WBF_WEIGHTS[1])
        elif WBF_WEIGHTS and not yolo_res[0]: current_wbf_weights.append(WBF_WEIGHTS[0])


    if not all_boxes_lists:
        return [], [], []
    
    if not current_wbf_weights: 
        wbf_weights_to_use = None
    elif len(current_wbf_weights) != len(all_boxes_lists): 
        wbf_weights_to_use = [1.0] * len(all_boxes_lists)
    else:
        wbf_weights_to_use = current_wbf_weights

    try:
        boxes, scores, labels = weighted_boxes_fusion(
            all_boxes_lists,
            all_scores_lists,
            all_labels_lists,
            iou_thr=WBF_IOU_THR,
            weights=wbf_weights_to_use,
            skip_box_thr=WBF_SKIP_THR,
        )
        return boxes.tolist(), scores.tolist(), labels.tolist()
    except Exception as e:
        return [], [], []


def plot_and_save_predictions(img_rgb, filename, predictions_for_plot, categories_dict, is_grayscale):
    img_to_draw = img_rgb.copy()
    if is_grayscale:
        img_gray = cv2.cvtColor(img_to_draw, cv2.COLOR_RGB2GRAY)
        img_to_draw = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    box_colors_cycle = [(0, 200, 0), (255, 100, 0), (0, 100, 255), (200, 0, 200)]
    default_error_color = (0, 0, 255)
    text_color_on_bg = (255, 255, 255) 
    thickness = 2 
    font_scale = 0.5
    font_thickness = 1
    img_h_draw, img_w_draw = img_to_draw.shape[:2]

    for i, prediction_data in enumerate(predictions_for_plot):
        if not (isinstance(prediction_data, tuple) and len(prediction_data) == 3):
            continue
        box_coords, category_id_coco, score_str = prediction_data

        if not (isinstance(box_coords, (tuple, list)) and len(box_coords) == 4):
            continue
            
        x1, y1, x2, y2 = map(int, box_coords)
        category_info = categories_dict.get(category_id_coco)
        label_text = f"ID {category_id_coco}: {score_str}"
        current_box_color = default_error_color

        if category_info and isinstance(category_info, dict):
            label_text = f"{category_info.get('name', f'ID {category_id_coco}')}: {score_str}"
            retrieved_color = category_info.get('color')
            if isinstance(retrieved_color, tuple) and len(retrieved_color) == 3:
                current_box_color = retrieved_color
            else:
                current_box_color = box_colors_cycle[i % len(box_colors_cycle)]
        else:
            current_box_color = box_colors_cycle[i % len(box_colors_cycle)]

        if not (isinstance(current_box_color, tuple) and len(current_box_color) == 3):
            current_box_color = default_error_color

        cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), current_box_color, thickness)
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_bg_y1 = y1 - text_height - baseline - 5 
        text_y_cv = y1 - baseline - 3 

        if text_bg_y1 < 0:
            text_bg_y1 = y2 + baseline + 5
            text_y_cv = y2 + text_height + baseline + 3
        
        text_bg_y2 = text_bg_y1 + text_height + baseline
        
        bg_x1_c = max(0, x1)
        bg_y1_c = max(0, text_bg_y1)
        bg_x2_c = min(img_w_draw - 1, x1 + text_width)
        bg_y2_c = min(img_h_draw - 1, text_bg_y2)
        
        text_x_c = max(0, x1)
        text_y_c = max(text_height, text_y_cv)

        if bg_x1_c < bg_x2_c and bg_y1_c < bg_y2_c:
             cv2.rectangle(img_to_draw, (bg_x1_c, bg_y1_c), (bg_x2_c, bg_y2_c), current_box_color, cv2.FILLED)
        cv2.putText(img_to_draw, label_text, (text_x_c, text_y_c), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color_on_bg, font_thickness, cv2.LINE_AA)

    output_path = os.path.join(OUTPUT_IMG_DIR, f"pred_{os.path.basename(filename)}")
    cv2.imwrite(output_path, cv2.cvtColor(img_to_draw, cv2.COLOR_RGB2BGR))

def save_coco_metrics_to_csv(coco_eval_instance, csv_filename):
    stats = coco_eval_instance.stats
    metric_names = [
        "mAP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]", "AP @[ IoU=0.50      | area=   all | maxDets=100 ]",
        "AP @[ IoU=0.75      | area=   all | maxDets=100 ]", "AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
        "AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]", "AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
        "AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]", "AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]",
        "AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]", "AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
        "AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]", "AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]"
    ]
    metrics_to_save = {name: stat for name, stat in zip(metric_names, stats)}

    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metrica', 'Valor'])
            for key, value in metrics_to_save.items():
                writer.writerow([key, f"{value:.4f}"])
        print(f"Métricas salvas com sucesso em: {csv_filename}")
    except IOError as e:
        print(f"Erro ao salvar o arquivo CSV de métricas: {e}")

def run_ensemble_evaluation():
    try:
        coco_ground_truth = COCO(ANN_FILE)
        category_map = {cat_info['id']: cat_info for cat_info in coco_ground_truth.loadCats(coco_ground_truth.getCatIds())}
        base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0,255,255), (255,0,255)]
        for i, cat_id_key in enumerate(category_map.keys()):
            if 'color' not in category_map[cat_id_key]:
                bgr_color = base_colors[i % len(base_colors)]
                if len(bgr_color) == 3:
                    category_map[cat_id_key]['color'] = bgr_color 
                else: 
                    category_map[cat_id_key]['color'] = (0,255,0)


    except Exception as e:
        print(f"Erro ao carregar o arquivo COCO de ground truth {ANN_FILE}: {e}")
        return

    print("Carregando modelo YOLOv8x")
    yolo_model = YOLO(YOLO_WEIGHTS)
    print("Carregando modelo Cascade R-CNN")
    cascade_predictor = build_detectron2_predictor(CASCADE_RCNN_CFG_FILE, CASCADE_RCNN_WEIGHTS, "Cascade R-CNN")

    if cascade_predictor is None:
        print("Falha ao carregar o modelo Cascade R-CNN.")
        return

    coco_results_for_eval = []
    total_fused_boxes_before_filter = 0
    image_id_list = coco_ground_truth.getImgIds()
    
    print(f"Iniciando predições em {len(image_id_list)} imagens")
    print(f">>> Aplicando filtro final de confiança para visualização: > {FINAL_SCORE_THRESHOLD*100:.0f}% <<<")

    for i, current_img_id in enumerate(tqdm(image_id_list, desc="Ensemble (YOLOv8x + Cascade R-CNN)")):
        img_metadata = coco_ground_truth.loadImgs(current_img_id)[0]
        img_file_path = os.path.join(IMG_DIR, img_metadata["file_name"])
        
        img_bgr_format = cv2.imread(img_file_path)
        if img_bgr_format is None:
            print(f"Aviso: Não foi possível ler a imagem {img_file_path}. Pulando...")
            continue
        img_rgb_format = cv2.cvtColor(img_bgr_format, cv2.COLOR_BGR2RGB)
        
        h_img, w_img = img_rgb_format.shape[:2]

        with torch.no_grad():
            yolo_predictions = predict_yolo(yolo_model, img_rgb_format)
            cascade_predictions = predict_detectron2_model(cascade_predictor, img_rgb_format)
        
        fused_boxes_norm, fused_scores, fused_labels_zero_indexed = fuse_predictions(yolo_predictions, cascade_predictions)
        
        predictions_to_plot_on_image = []
        if fused_boxes_norm: # Verifica se há caixas fundidas
            total_fused_boxes_before_filter += len(fused_boxes_norm)
            for box_norm, score_val, label_idx in zip(fused_boxes_norm, fused_scores, fused_labels_zero_indexed):
                x1_abs_f, y1_abs_f, x2_abs_f, y2_abs_f = box_norm[0] * w_img, box_norm[1] * h_img, box_norm[2] * w_img, box_norm[3] * h_img
                
                x1_abs, y1_abs, x2_abs, y2_abs = int(x1_abs_f), int(y1_abs_f), int(x2_abs_f), int(y2_abs_f)
                
                x1_abs = max(0, x1_abs)
                y1_abs = max(0, y1_abs)
                x2_abs = min(w_img, x2_abs)
                y2_abs = min(h_img, y2_abs)

                width_box, height_box = x2_abs - x1_abs, y2_abs - y1_abs
                
                current_coco_category_id = int(label_idx) + 1

                if width_box > 0 and height_box > 0 :
                    coco_results_for_eval.append({
                        "image_id": current_img_id,
                        "category_id": current_coco_category_id,
                        "bbox": [float(x1_abs), float(y1_abs), float(width_box), float(height_box)],
                        "score": float(score_val),
                    })

                if score_val >= FINAL_SCORE_THRESHOLD:
                    score_display_str = f"{score_val:.1%}"
                    predictions_to_plot_on_image.append(
                        ((x1_abs, y1_abs, x2_abs, y2_abs), current_coco_category_id, score_display_str)
                    )
        
        save_as_grayscale = i < NUM_GRAYSCALE_IMAGES_TO_SAVE
        if predictions_to_plot_on_image:
            plot_and_save_predictions(img_rgb_format, img_metadata["file_name"], predictions_to_plot_on_image, category_map, save_as_grayscale)
        else:
            if save_as_grayscale: 
                img_to_save_no_det = cv2.cvtColor(img_rgb_format, cv2.COLOR_RGB2GRAY)
                img_to_save_no_det = cv2.cvtColor(img_to_save_no_det, cv2.COLOR_GRAY2BGR)
            else: 
                img_to_save_no_det = cv2.cvtColor(img_rgb_format, cv2.COLOR_RGB2BGR)
            
            output_path_no_det = os.path.join(OUTPUT_IMG_DIR, f"pred_nodetect_{img_metadata['file_name']}")
            cv2.imwrite(output_path_no_det, img_to_save_no_det)


    print(f"\nTotal de caixas geradas pelo ensemble (antes do filtro de visualização): {total_fused_boxes_before_filter}")

    if not coco_results_for_eval:
        print("Nenhuma predição foi gerada pelo ensemble. Avaliação COCO não será executada.")
        return

    try:
        with open(OUT_JSON_FILENAME, "w", encoding="utf-8") as f:
            json.dump(coco_results_for_eval, f, indent=2)
        print(f"Predições em formato COCO salvas em: {OUT_JSON_FILENAME}")
    except IOError as e:
        print(f"Erro ao salvar o arquivo JSON de predições: {e}")
        return

    print("\nExecutando COCOeval.")
    try:
        coco_detected_results = coco_ground_truth.loadRes(coco_results_for_eval)
        coco_evaluator = COCOeval(coco_ground_truth, coco_detected_results, iouType="bbox")
        coco_evaluator.evaluate()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        save_coco_metrics_to_csv(coco_evaluator, CSV_METRICS_FILENAME)
    except Exception as e:
        print(f"Erro durante a avaliação COCO: {e}")

    print("\nExecução do ensemble em CPU concluída.")


if __name__ == "__main__":
    main_function_name = "run_ensemble_evaluation"
    
    if torch.cuda.is_available():
        print("CUDA disponível")
    else:
        print("CUDA não disponível")
        
    globals()[main_function_name]()
