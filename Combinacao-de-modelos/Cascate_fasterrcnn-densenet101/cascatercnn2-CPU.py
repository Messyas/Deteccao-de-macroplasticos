import os
import json
import cv2
import torch
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from ensemble_boxes import weighted_boxes_fusion
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import csv

IMG_DIR = "/home/messyas/ml/data/tccdata/images"
ANN_FILE = "/home/messyas/ml/data/tccdata/annotations/4_class_splits/annotations_test.json"

# Modelo 1: Faster R-CNN
RCNN_DIR = "/home/messyas/ml/tcc/models/combinacao/rcnn"
RCNN_WEIGHTS = os.path.join(RCNN_DIR, "rcnn.pth")
RCNN_CFG_FILE = os.path.join(RCNN_DIR, "config.yaml")

# Modelo 2: Cascade R-CNN
CASCADE_RCNN_DIR = "/home/messyas/ml/detectron/detectron2/rcascate-resnet50/pesos"

CASCADE_RCNN_WEIGHTS = os.path.join(CASCADE_RCNN_DIR, "best.pth")
CASCADE_RCNN_CFG_FILE = os.path.join(CASCADE_RCNN_DIR, "config_cascade.yaml") 

FG_CLASSES = 4 

# Ensemble 
WBF_IOU_THR = 0.55
WBF_SKIP_THR = 0.01 
WBF_WEIGHTS = [1.0, 1.0]  

OUT_JSON = "ensemble_preds_frcnn_cascade_cpu.json"
OUTPUT_IMG_DIR = "predicted_images_frcnn_cascade_cpu"
CSV_METRICS_FILE = "ensemble_metrics_frcnn_cascade_cpu.csv"
NUM_GRAYSCALE_IMAGES = 5

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

def build_predictor(cfg_file: str, weights_file: str, model_name: str) -> DefaultPredictor:
    cfg = get_cfg()
    if not os.path.exists(cfg_file):
        print(f"Arquivo de configuração não encontrado: {cfg_file}")
        return None
    if not os.path.exists(weights_file):
        print(f"Arquivo de pesos não encontrado: {weights_file}")
        return None

    cfg.merge_from_file(cfg_file)
    cfg.MODEL.WEIGHTS = weights_file
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = FG_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  

    #Usa a CPU
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)
    print(f"{model_name} configurado para usar CPU.")
    return predictor

def predict_detectron2(predictor: DefaultPredictor, img_rgb):
    out = predictor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))["instances"].to("cpu")
    h, w = img_rgb.shape[:2]
    boxes = (out.pred_boxes.tensor.numpy() / [w, h, w, h]).tolist()
    scores = out.scores.numpy().tolist()
    labels = out.pred_classes.numpy().tolist()
    return boxes, scores, labels

def fuse(frcnn_res, cascade_res):
    boxes, scores, labels = weighted_boxes_fusion(
        [frcnn_res[0], cascade_res[0]],
        [frcnn_res[1], cascade_res[1]],
        [frcnn_res[2], cascade_res[2]],
        iou_thr=WBF_IOU_THR,
        weights=WBF_WEIGHTS,
        skip_box_thr=WBF_SKIP_THR,
    )
    return boxes, scores, labels

def plot_predictions(img, filename, predictions, categories, thickness=4, font_scale=0.7, font_thickness=2, gray_scale=False):
    img_draw = img.copy()

    if gray_scale:
        img_gray = cv2.cvtColor(img_draw, cv2.COLOR_RGB2GRAY)
        img_draw = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    for (x1, y1, x2, y2), category_id, score_percentage in predictions:
        try:
            category_info = categories[category_id]
            label = f"{category_info['name']}: {score_percentage}"
            color = category_info.get('color', (0, 255, 0))
        except KeyError:
            print(f"Aviso: category_id {category_id} não encontrado. Usando 'Desconhecido'.")
            label = f"Desconhecido ({category_id}): {score_percentage}"
            color = (0, 0, 255)

        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
        cv2.rectangle(img_draw, (x1, text_y - text_height - 5), (x1 + text_width, text_y + 5), color, cv2.FILLED)
        cv2.putText(img_draw, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    output_path = os.path.join(OUTPUT_IMG_DIR, f"pred_{filename}")
    cv2.imwrite(output_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))

def save_metrics_to_csv(coco_eval, filename="ensemble_metrics.csv"):
    """Salva as métricas do COCOeval em um arquivo CSV."""
    stats = coco_eval.stats
    metrics = {
        "mAP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]": stats[0],
        "AP @[ IoU=0.50      | area=   all | maxDets=100 ]": stats[1],
        "AP @[ IoU=0.75      | area=   all | maxDets=100 ]": stats[2],
        "AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]": stats[3],
        "AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]": stats[4],
        "AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]": stats[5],
        "AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]": stats[6],
        "AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]": stats[7],
        "AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]": stats[8],
        "AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]": stats[9],
        "AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]": stats[10],
        "AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]": stats[11],
    }
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metrica', 'Valor'])
            for key, value in metrics.items():
                writer.writerow([key, f"{value:.4f}"])
        print(f"Métricas salvas com sucesso em: {filename}")
    except IOError as e:
        print(f"Erro ao salvar o arquivo CSV: {e}")

def main() -> None:
    coco_gt = COCO(ANN_FILE)
    categories = {cat['id']: cat for cat in coco_gt.loadCats(coco_gt.getCatIds())}
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    for i, cat_id in enumerate(categories):
        categories[cat_id]['color'] = colors[i % len(colors)]

    print("Carregando modelo Faster R-CNN (CPU)...")
    frcnn_pred = build_predictor(RCNN_CFG_FILE, RCNN_WEIGHTS, "Faster R-CNN")
    print("Carregando modelo Cascade R-CNN (CPU)...")
    cascade_pred = build_predictor(CASCADE_RCNN_CFG_FILE, CASCADE_RCNN_WEIGHTS, "Cascade R-CNN")

    if frcnn_pred is None or cascade_pred is None:
        print("Erro ao carregar um ou mais modelos.")
        return

    results = []
    total_boxes = 0

    image_ids = coco_gt.getImgIds()
    print(f"Iniciando predições em {len(image_ids)} imagens")

    for i, img_id in enumerate(tqdm(image_ids, desc="Ensemble")):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(IMG_DIR, img_info["file_name"])
        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            print(f"Aviso: Não foi possível ler a imagem {img_path}.")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        with torch.no_grad():
            frcnn_out = predict_detectron2(frcnn_pred, img_rgb)
            cascade_out = predict_detectron2(cascade_pred, img_rgb)

        boxes, scores, labels = fuse(frcnn_out, cascade_out)

        image_predictions_for_plot = []
        for b, s, l in zip(boxes, scores, labels):
            x1 = int(b[0] * w)
            y1 = int(b[1] * h)
            x2 = int(b[2] * w)
            y2 = int(b[3] * h)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            category_id = int(l) + 1
            score_percentage = f"{s:.2%}"

            image_predictions_for_plot.append(((x1, y1, x2, y2), category_id, score_percentage))

            results.append({
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(s),
            })

        total_boxes += len(boxes)
        use_grayscale = i < NUM_GRAYSCALE_IMAGES
        plot_predictions(img_rgb, img_info["file_name"], image_predictions_for_plot, categories, gray_scale=use_grayscale)

    print(f"\nTotal de caixas geradas pelo ensemble: {total_boxes}")

    try:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"Predições salvas em: {OUT_JSON}")
    except IOError as e:
        print(f"Erro ao salvar o arquivo JSON: {e}")
        return

    print("\nExecutando COCOeval…")
    try:
        coco_dt = coco_gt.loadRes(results)
        eval_ = COCOeval(coco_gt, coco_dt, iouType="bbox")
        eval_.evaluate()
        eval_.accumulate()
        eval_.summarize()
        save_metrics_to_csv(eval_, CSV_METRICS_FILE)
    except Exception as e:
        print(f"Erro durante a avaliação COCO: {e}")

    print("\nExecução concluída.")

if __name__ == "__main__":
    main()
