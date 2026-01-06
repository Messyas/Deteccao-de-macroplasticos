import logging
import sys
from ultralytics import YOLO
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

for pkg, imp in [('ultralytics','ultralytics'), 
                 ('torch','torch'), 
                 ('torchvision','torchvision'),
                 ('detectron2','detectron2'),
                 ('ensemble_boxes','ensemble_boxes'),
                 ('PIL','PIL.Image'),
                 ('cv2','cv2'),
                 ('numpy','numpy')]:
    try:
        __import__(imp)
    except ImportError:
        logger.error(f"Pacote faltando: {pkg}. Rode: pip install {pkg}")
        sys.exit(1)

YOLO_WEIGHTS       = '/home/messyas/ml/tcc/models/combinacao/yolo/best.pt'
DETECTRON2_WEIGHTS = '/home/messyas/ml/tcc/models/combinacao/rcnn/rcnn.pth'
FG_CLASSES         = 4            
IOU_THR            = 0.5
WBF_WEIGHTS        = [2.0, 1.0] 
IMG_PATH           = '/home/messyas/ml/detectron/detectron2/foia.jpg'
OUT_PATH           = './fused_result.jpg'

logger.info(f"YOLO pesos: {YOLO_WEIGHTS}")
logger.info(f"Detectron2 pesos: {DETECTRON2_WEIGHTS}")
logger.info(f"Classes (sem background): {FG_CLASSES}")

logger.info("Carregando YOLOv8")
yolo = YOLO(YOLO_WEIGHTS)
logger.info("YOLOv8 pronto.")

cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = DETECTRON2_WEIGHTS
cfg.MODEL.ROI_HEADS.NUM_CLASSES = FG_CLASSES
cfg.MODEL.DEVICE = "cpu"   # ou "cuda"
predictor = DefaultPredictor(cfg)
logger.info("Detectron2 pronto.")

# — 5) Funções de predição —
def predict_yolo(img: np.ndarray):
    res = yolo(img)[0]
    h, w = img.shape[:2]
    boxes  = (res.boxes.xyxy.cpu().numpy() / [w, h, w, h]).tolist()
    scores = res.boxes.conf.cpu().tolist()
    labels = res.boxes.cls.cpu().tolist()
    logger.info(f"YOLO detections: {len(boxes)}")
    return boxes, scores, labels

def predict_det2(img: np.ndarray):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    inst = predictor(img_bgr)["instances"].to("cpu")
    boxes  = inst.pred_boxes.tensor.numpy()     
    scores = inst.scores.numpy().tolist()
    labels = inst.pred_classes.numpy().tolist()
    h, w = img.shape[:2]
    boxes = (boxes / [w, h, w, h]).tolist()
    logger.info(f"Detectron2 detections: {len(boxes)}")
    return boxes, scores, labels

def fuse(results, iou_thr, weights):
    logger.info(f"Executando WBF (IoU={iou_thr}, pesos={weights})")
    b, s, l = zip(*results)
    fb, fs, fl = weighted_boxes_fusion(
        b, s, l,
        iou_thr=iou_thr,
        weights=weights,
        skip_box_thr=0.0
    )
    logger.info(f"WBF resultou em {len(fb)} caixas")
    return fb, fs, fl

if __name__ == "__main__":
    logger.info(f"Abrindo imagem: {IMG_PATH}")
    try:
        img = np.array(Image.open(IMG_PATH).convert("RGB"))
    except Exception as e:
        logger.error(f"Falha ao abrir imagem: {e}")
        sys.exit(1)

    yolo_out = predict_yolo(img)
    det2_out = predict_det2(img)
    fb, fs, fl = fuse([yolo_out, det2_out], IOU_THR, WBF_WEIGHTS)
    h, w = img.shape[:2]
    boxes_px = [[int(x1*w), int(y1*h), int(x2*w), int(y2*h)] for x1,y1,x2,y2 in fb]
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    for (x1,y1,x2,y2), score, label in zip(boxes_px, fs, fl):
        draw.rectangle([x1,y1,x2,y2], outline="red", width=2)
        draw.text((x1, y1-10), f"{label}:{score:.2f}", font=font, fill="red")

    im.save(OUT_PATH)
    logger.info(f"Imagem fundida salva em: {OUT_PATH}")
    im.show()
