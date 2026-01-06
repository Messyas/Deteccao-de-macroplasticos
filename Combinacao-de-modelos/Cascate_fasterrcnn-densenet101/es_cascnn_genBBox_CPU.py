import os
import cv2
import torch
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from ensemble_boxes import weighted_boxes_fusion
from pycocotools.coco import COCO 

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

FINAL_SCORE_THRESHOLD = 0.40

OUTPUT_IMG_DIR = "predicted_images_visuals_filtered"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

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
    cfg.MODEL.DEVICE = "cpu" # Força CPU
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

def draw_enhanced_boxes(img_rgb, filename, predictions, categories):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_draw = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    img_bgr = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)

    box_color_blue = (255, 100, 0)
    box_color_green = (0, 200, 0)
    text_color = (255, 255, 255)
    text_bg_color = (0, 0, 0)
    thickness = 4
    font_scale = 0.8
    font_thickness = 2

    color_idx = 0

    for (x1, y1, x2, y2), category_id, score_percentage in predictions:
        try:
            cat_name = categories.get(category_id, {'name': f'ID {category_id}'})['name']
            label = f"{cat_name}: {score_percentage}"
        except Exception:
             label = f"ID {category_id}: {score_percentage}"

        current_box_color = box_color_green if color_idx % 2 == 0 else box_color_blue
        color_idx += 1

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), current_box_color, thickness)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_y_base = y1 - 10
        if text_y_base - text_height - 10 < 0:
            text_y_base = y2 + text_height + 15

        cv2.rectangle(img_bgr, (x1, text_y_base - text_height - 5), (x1 + text_width, text_y_base + 5), text_bg_color, cv2.FILLED)
        cv2.putText(img_bgr, label, (x1, text_y_base), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)

    output_path = os.path.join(OUTPUT_IMG_DIR, f"pred_{filename}")
    cv2.imwrite(output_path, img_bgr)

def main() -> None:
    try:
        coco_gt = COCO(ANN_FILE)
        categories = {cat['id']: cat for cat in coco_gt.loadCats(coco_gt.getCatIds())}
        image_ids = coco_gt.getImgIds()
    except Exception as e:
        print(f"Erro ao carregar o arquivo COCO {ANN_FILE}: {e}")
        return

    print("Carregando modelo Faster R-CNN")
    frcnn_pred = build_predictor(RCNN_CFG_FILE, RCNN_WEIGHTS, "Faster R-CNN")
    print("Carregando modelo Cascade R-CNN")
    cascade_pred = build_predictor(CASCADE_RCNN_CFG_FILE, CASCADE_RCNN_WEIGHTS, "Cascade R-CNN")

    if frcnn_pred is None or cascade_pred is None:
        print("Erro ao carregar um ou mais modelos. Abortando.")
        return

    print(f"Iniciando predições e desenho em {len(image_ids)} imagens...")
    print(f">>> Aplicando filtro final de confiança: > {FINAL_SCORE_THRESHOLD*100:.0f}% <<<")

    for img_id in tqdm(image_ids, desc="Gerando Predições Visuais"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(IMG_DIR, img_info["file_name"])
        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            print(f"Não foi possível ler a imagem {img_path}. Pulando...")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        with torch.no_grad():
            frcnn_out = predict_detectron2(frcnn_pred, img_rgb)
            cascade_out = predict_detectron2(cascade_pred, img_rgb)

        boxes, scores, labels = fuse(frcnn_out, cascade_out)

        image_predictions_for_plot = []
     
        for b, s, l in zip(boxes, scores, labels):
            if s >= FINAL_SCORE_THRESHOLD:
                x1, y1, x2, y2 = [int(coord) for coord in (b[0] * w, b[1] * h, b[2] * w, b[3] * h)]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                category_id = int(l) + 1
                score_percentage = f"{s:.1%}"
                image_predictions_for_plot.append(((x1, y1, x2, y2), category_id, score_percentage))

        if image_predictions_for_plot:
            draw_enhanced_boxes(img_rgb, img_info["file_name"], image_predictions_for_plot, categories)
        else:
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            output_path = os.path.join(OUTPUT_IMG_DIR, f"pred_nodetect_{img_info['file_name']}")
            cv2.imwrite(output_path, img_bgr)

    print(f"\nImagens com predições visuais filtradas salvas em: {OUTPUT_IMG_DIR}")

if __name__ == "__main__":
    main()
