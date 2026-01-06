import os
import cv2
import torch
import argparse 
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from ensemble_boxes import weighted_boxes_fusion

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

CATEGORIES_MANUAL = {
    1: 'Plastic',
    2: 'Pile',
    3: 'Face mask',
    4: 'Trash bin',
}
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
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)
    print(f"{model_name} configurado para usar CPU.")
    return predictor

def predict_detectron2(predictor: DefaultPredictor, img_rgb):
    outputs = predictor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)) 
    instances = outputs["instances"].to("cpu")
    h, w = img_rgb.shape[:2]
    boxes = (instances.pred_boxes.tensor.numpy() / [w, h, w, h]).tolist()
    scores = instances.scores.numpy().tolist()
    labels = instances.pred_classes.numpy().tolist()
    return boxes, scores, labels

def fuse(frcnn_res, cascade_res):
    boxes_list = [frcnn_res[0], cascade_res[0]]
    scores_list = [frcnn_res[1], cascade_res[1]]
    labels_list = [frcnn_res[2], cascade_res[2]]

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        iou_thr=WBF_IOU_THR,
        weights=WBF_WEIGHTS,
        skip_box_thr=WBF_SKIP_THR, 
    )
    return boxes, scores, labels

def draw_enhanced_boxes(img_rgb, filename, predictions, categories, output_dir, grayscale=True):
    img_draw = img_rgb.copy()
    if grayscale:
        img_gray = cv2.cvtColor(img_draw, cv2.COLOR_RGB2GRAY)
        img_draw = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    img_bgr = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)

    # --- Estilos 
    box_color_blue = (255, 100, 0) 
    text_color = (255, 255, 255)   # Branco
    text_bg_color = (0, 0, 0)      # Preto
    thickness = 4                  # Espessura da caixa
    font_scale = 0.8               # Tamanho da fonte
    font_thickness = 2             # Espessura da fonte
    # ---

    for (x1, y1, x2, y2), category_id, score_percentage in predictions:
        cat_name = categories.get(category_id, f'ID {category_id}')
        label = f"{cat_name}: {score_percentage}"
        current_box_color = box_color_blue
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), current_box_color, thickness)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_y_base = y1 - 10
        if text_y_base - text_height - 10 < 0: 
            text_y_base = y2 + text_height + 15
            
        cv2.rectangle(img_bgr, (x1, text_y_base - text_height - 5), (x1 + text_width, text_y_base + 5), text_bg_color, cv2.FILLED)
        cv2.putText(img_bgr, label, (x1, text_y_base), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)

    output_filename = f"pred_{os.path.basename(filename)}"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, img_bgr)
    print(f"Imagem processada salva em: {output_path}")

def main(args):
    final_score_threshold = args.threshold
    output_dir = args.output_dir
    img_path = args.image_path
    use_grayscale = args.grayscale

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(img_path):
        print(f"Imagem de entrada não encontrada: {img_path}")
        return

    print("Carregando modelos")
    frcnn_pred = build_predictor(RCNN_CFG_FILE, RCNN_WEIGHTS, "Faster R-CNN")
    cascade_pred = build_predictor(CASCADE_RCNN_CFG_FILE, CASCADE_RCNN_WEIGHTS, "Cascade R-CNN")

    if frcnn_pred is None or cascade_pred is None:
        print("Erro ao carregar modelos.")
        return

    print(f"Processando imagem: {img_path}")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Não foi possível ler a imagem {img_path}.")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    print("Executando predições")
    with torch.no_grad():
        frcnn_out = predict_detectron2(frcnn_pred, img_rgb)
        cascade_out = predict_detectron2(cascade_pred, img_rgb)

    print("Aplicando Weighted Boxes Fusion")
    boxes, scores, labels = fuse(frcnn_out, cascade_out)

    print(f"Filtrando resultados (limiar = {final_score_threshold*100:.0f}%)")
    image_predictions_for_plot = []
    for b, s, l in zip(boxes, scores, labels):
        if s >= final_score_threshold:
            x1, y1, x2, y2 = [int(coord) for coord in (b[0] * w, b[1] * h, b[2] * w, b[3] * h)]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            category_id = int(l) + 1 
            score_percentage = f"{s:.1%}"
            image_predictions_for_plot.append(((x1, y1, x2, y2), category_id, score_percentage))

    if not image_predictions_for_plot:
        print("Nenhuma detecção encontrada acima do limiar")
        return

    print(f"Desenhando {len(image_predictions_for_plot)} caixas delimitadoras")
    draw_enhanced_boxes(img_rgb, img_path, image_predictions_for_plot, CATEGORIES_MANUAL, output_dir, use_grayscale)

    print("Processo concluído.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa uma única imagem com ensemble de modelos e WBF, desenhando caixas azuis.")
    parser.add_argument("image_path")
    parser.add_argument("-o", "--output_dir", default="single_image_output")
    parser.add_argument("-t", "--threshold", type=float, default=0.50)
    parser.add_argument("-g", "--grayscale", action="store_true")

    parsed_args = parser.parse_args()
    main(parsed_args)
