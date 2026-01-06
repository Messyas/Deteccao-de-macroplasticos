import os
import cv2
import torch
import argparse  # Para ler argumentos da linha de comando
from ultralytics import YOLO  # Importa a biblioteca YOLO
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from ensemble_boxes import weighted_boxes_fusion

# ───────────────────────── CONFIGURAÇÕES ESSENCIAIS ──────────────────────────
# Modelo 1: YOLOv8
YOLO_WEIGHTS = "/home/messyas/ml/tcc/models/combinacao/yolo/best.pt"
YOLO_CONF_THR = 0.05 # Limiar baixo para YOLO (WBF fará o trabalho)

# Modelo 2: Faster R-CNN
RCNN_DIR = "/home/messyas/ml/tcc/models/combinacao/rcnn"
RCNN_WEIGHTS = os.path.join(RCNN_DIR, "rcnn.pth")
RCNN_CFG_FILE = os.path.join(RCNN_DIR, "config.yaml")
FG_CLASSES = 4 # Número de classes (deve ser o mesmo para ambos)
RCNN_SCORE_THR = 0.05 # Limiar baixo para R-CNN

# Ensemble (YOLO + Faster R-CNN)
WBF_IOU_THR = 0.5
WBF_SKIP_THR = 0.3  # Limiar para WBF (conforme script base)
WBF_WEIGHTS = [2.0, 1.0]  # [YOLO, FRCNN] - Mais peso para YOLO

# <<< CATEGORIAS (Manter as mesmas do script anterior) >>>
CATEGORIES_MANUAL = {
    1: 'Plastic',
    2: 'Pile',
    3: 'Face mask',
    4: 'Trash bin',
}
# ─────────────────────────────────────────────────────────────────────────────

def build_frcnn_predictor() -> DefaultPredictor:
    """Configura e constrói o preditor do Faster R-CNN para rodar em CPU."""
    cfg = get_cfg()
    if not os.path.exists(RCNN_CFG_FILE):
        print(f"ERRO: Arquivo de config R-CNN não encontrado: {RCNN_CFG_FILE}")
        return None
    if not os.path.exists(RCNN_WEIGHTS):
        print(f"ERRO: Arquivo de pesos R-CNN não encontrado: {RCNN_WEIGHTS}")
        return None

    cfg.merge_from_file(RCNN_CFG_FILE)
    cfg.MODEL.WEIGHTS = RCNN_WEIGHTS
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = FG_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = RCNN_SCORE_THR
    cfg.MODEL.DEVICE = "cpu" # Força CPU
    predictor = DefaultPredictor(cfg)
    print("Faster R-CNN configurado para usar CPU.")
    return predictor

def build_yolo_model() -> YOLO:
    """Carrega o modelo YOLO."""
    if not os.path.exists(YOLO_WEIGHTS):
        print(f"ERRO: Arquivo de pesos YOLO não encontrado: {YOLO_WEIGHTS}")
        return None
    try:
        model = YOLO(YOLO_WEIGHTS)
        print("Modelo YOLO carregado.")
        return model
    except Exception as e:
        print(f"ERRO ao carregar modelo YOLO: {e}")
        return None


def predict_yolo(model: YOLO, img_rgb):
    """Executa a predição com o modelo YOLO em CPU."""
    # O device='cpu' força a inferência na CPU
    out = model(img_rgb, conf=YOLO_CONF_THR, device='cpu')[0]
    h, w = img_rgb.shape[:2]
    # Normaliza as coordenadas para [0, 1] e move para CPU (já estará)
    boxes = (out.boxes.xyxy.cpu().numpy() / [w, h, w, h]).tolist()
    scores = out.boxes.conf.cpu().tolist()
    labels = out.boxes.cls.cpu().tolist()
    return boxes, scores, labels


def predict_frcnn(predictor: DefaultPredictor, img_rgb):
    """Executa a predição com o modelo Faster R-CNN (já configurado para CPU)."""
    out = predictor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))["instances"].to("cpu")
    h, w = img_rgb.shape[:2]
    # Normaliza as coordenadas para [0, 1]
    boxes = (out.pred_boxes.tensor.numpy() / [w, h, w, h]).tolist()
    scores = out.scores.numpy().tolist()
    labels = out.pred_classes.numpy().tolist()
    return boxes, scores, labels


def fuse(yolo_res, frcnn_res):
    """Aplica o Weighted Boxes Fusion (WBF) nas predições."""
    boxes_list = [yolo_res[0], frcnn_res[0]]
    scores_list = [yolo_res[1], frcnn_res[1]]
    labels_list = [yolo_res[2], frcnn_res[2]]

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
    """Desenha caixas delimitadoras aprimoradas (AZUIS) e salva a imagem."""
    img_draw = img_rgb.copy()

    if grayscale:
        img_gray = cv2.cvtColor(img_draw, cv2.COLOR_RGB2GRAY)
        img_draw = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    img_bgr = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)

    box_color_blue = (255, 100, 0) # Azul (BGR)
    text_color = (255, 255, 255)
    text_bg_color = (0, 0, 0)
    thickness = 4
    font_scale = 0.8
    font_thickness = 2

    for (x1, y1, x2, y2), category_id, score_percentage in predictions:
        cat_name = categories.get(category_id, f'ID {category_id}')
        label = f"{cat_name}: {score_percentage}"
        current_box_color = box_color_blue # Usa sempre azul

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), current_box_color, thickness)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_y_base = y1 - 10
        if text_y_base - text_height - 10 < 0:
            text_y_base = y2 + text_height + 15
        cv2.rectangle(img_bgr, (x1, text_y_base - text_height - 5), (x1 + text_width, text_y_base + 5), text_bg_color, cv2.FILLED)
        cv2.putText(img_bgr, label, (x1, text_y_base), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)

    output_filename = f"pred_yolo_frcnn_{os.path.basename(filename)}" # Nome indica os modelos
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, img_bgr)
    print(f"Imagem processada salva em: {output_path}")

def main(args):
    """Função principal para processar uma única imagem com YOLO e Faster R-CNN."""
    final_score_threshold = args.threshold
    output_dir = args.output_dir
    img_path = args.image_path
    use_grayscale = args.grayscale

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(img_path):
        print(f"ERRO: Imagem de entrada não encontrada: {img_path}")
        return

    print("Carregando modelos...")
    yolo_model = build_yolo_model()
    frcnn_predictor = build_frcnn_predictor()

    if yolo_model is None or frcnn_predictor is None:
        print("Erro ao carregar um ou mais modelos. Abortando.")
        return

    print(f"Processando imagem: {img_path}")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"ERRO: Não foi possível ler a imagem {img_path}.")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    print("Executando predições...")
    with torch.no_grad():
        yolo_out = predict_yolo(yolo_model, img_rgb)
        frcnn_out = predict_frcnn(frcnn_predictor, img_rgb)

    print("Aplicando Weighted Boxes Fusion...")
    boxes, scores, labels = fuse(yolo_out, frcnn_out)

    print(f"Filtrando resultados (limiar = {final_score_threshold*100:.0f}%)...")
    image_predictions_for_plot = []
    for b, s, l in zip(boxes, scores, labels):
        # Aplica o filtro final de 70%
        if s >= final_score_threshold:
            x1, y1, x2, y2 = [int(coord) for coord in (b[0] * w, b[1] * h, b[2] * w, b[3] * h)]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            category_id = int(l) + 1 # +1 pois modelos dão 0-N
            score_percentage = f"{s:.1%}" # Formato de porcentagem
            image_predictions_for_plot.append(((x1, y1, x2, y2), category_id, score_percentage))

    if not image_predictions_for_plot:
        print("Nenhuma detecção encontrada acima do limiar.")
        return

    print(f"Desenhando {len(image_predictions_for_plot)} caixas delimitadoras...")
    draw_enhanced_boxes(img_rgb, img_path, image_predictions_for_plot, CATEGORIES_MANUAL, output_dir, use_grayscale)

    print("Processo concluído.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa uma única imagem com ensemble YOLO + Faster R-CNN e WBF.")
    parser.add_argument("image_path", help="Caminho para a imagem de entrada.")
    parser.add_argument("-o", "--output_dir", default="single_image_output_yolo_frcnn", help="Diretório para salvar a imagem processada.")
    parser.add_argument("-t", "--threshold", type=float, default=0.70, help="Limiar de confiança final para exibir caixas (0.0 a 1.0). Padrão: 0.70")
    parser.add_argument("-g", "--grayscale", action="store_true", help="Converter a imagem de saída para escala de cinza antes de desenhar.")

    parsed_args = parser.parse_args()
    main(parsed_args)
