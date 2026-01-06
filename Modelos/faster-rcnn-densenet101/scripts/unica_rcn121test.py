import os
import cv2
import torch
import argparse  
from detectron2 import model_zoo 
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger

setup_logger()

MODEL_PATH = "/home/messyas/ml/tcc/models/combinacao/rcnn/rcnn.pth"
CONFIG_FILE = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
NUM_CLASSES = 4
MODEL_SCORE_THR = 0.05 

CATEGORIES_MANUAL = {
    1: 'Plastic',
    2: 'Pile',
    3: 'Face mask',
    4: 'Trash bin',
}
# ─────────────────────────────────────────────────────────────────────────────

def build_predictor(model_path: str, num_classes: int) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))

    if not os.path.exists(model_path):
        print(f"ERRO: Arquivo de pesos não encontrado: {model_path}")
        return None

    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = MODEL_SCORE_THR
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = DefaultPredictor(cfg)
    print(f"Preditor Faster R-CNN X101 configurado para usar {cfg.MODEL.DEVICE}.")
    return predictor

def predict_rcnn(predictor: DefaultPredictor, img_rgb):
    outputs = predictor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)) 
    instances = outputs["instances"].to("cpu")
    h, w = img_rgb.shape[:2]
    boxes = (instances.pred_boxes.tensor.numpy() / [w, h, w, h]).tolist()
    scores = instances.scores.numpy().tolist()
    labels = instances.pred_classes.numpy().tolist()
    return boxes, scores, labels

def draw_enhanced_boxes(img_rgb, filename, predictions, categories, output_dir, grayscale=True):
    img_draw = img_rgb.copy()
    if grayscale:
        img_gray = cv2.cvtColor(img_draw, cv2.COLOR_RGB2GRAY)
        img_draw = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    img_bgr = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)

    # --- Estilos 
    box_color_blue = (255, 100, 0) # Azul
    text_color = (255, 255, 255)   # Branco
    text_bg_color = (0, 0, 0)      # Preto
    thickness = 4                  # Espessura da caixa
    font_scale = 0.8               # Tamanho da fonte
    font_thickness = 2             # Espessura da fonte
    # ----

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

    output_filename = f"pred_faster101_{os.path.basename(filename)}" # Nome indica o modelo
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, img_bgr)
    print(f"Imagem processada salva em: {output_path}")

def main(args):
    """Função principal para processar uma única imagem com Faster R-CNN X101."""
    final_score_threshold = args.threshold
    output_dir = args.output_dir
    img_path = args.image_path
    use_grayscale = args.grayscale

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(img_path):
        print(f"ERRO: Imagem de entrada não encontrada: {img_path}")
        return

    print("Carregando modelo Faster R-CNN X101...")
    predictor = build_predictor(MODEL_PATH, NUM_CLASSES)

    if predictor is None:
        print("Erro ao carregar o modelo. Abortando.")
        return

    print(f"Processando imagem: {img_path}")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"ERRO: Não foi possível ler a imagem {img_path}.")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    print("Executando predição...")
    with torch.no_grad():
        boxes, scores, labels = predict_rcnn(predictor, img_rgb)

    print(f"Filtrando resultados (limiar = {final_score_threshold*100:.0f}%)...")
    image_predictions_for_plot = []
    for b, s, l in zip(boxes, scores, labels):
        if s >= final_score_threshold:
            x1, y1, x2, y2 = [int(coord) for coord in (b[0] * w, b[1] * h, b[2] * w, b[3] * h)]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            category_id = int(l) + 1 
            score_percentage = f"{s:.1%}"
            image_predictions_for_plot.append(((x1, y1, x2, y2), category_id, score_percentage))

    if not image_predictions_for_plot:
        print("Nenhuma detecção encontrada acima do limiar.")
        return

    print(f"Desenhando {len(image_predictions_for_plot)} caixas delimitadoras...")
    draw_enhanced_boxes(img_rgb, img_path, image_predictions_for_plot, CATEGORIES_MANUAL, output_dir, use_grayscale)

    print("Processo concluído.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa uma única imagem com Faster R-CNN X101.")
    parser.add_argument("image_path", help="Caminho para a imagem de entrada.")
    parser.add_argument("-o", "--output_dir", default="single_image_output_faster101", help="Diretório para salvar a imagem processada.")
    parser.add_argument("-t", "--threshold", type=float, default=0.50, help="Limiar de confiança final para exibir caixas (0.0 a 1.0). Padrão: 0.50")
    parser.add_argument("-g", "--grayscale", action="store_true", help="Converter a imagem de saída para escala de cinza (P&B).")

    parsed_args = parser.parse_args()
    main(parsed_args)
