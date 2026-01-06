from detectron2 import model_zoo
from detectron2.config import get_cfg
import os

BASE_CONFIG_FILE = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml" 

CASCADE_RCNN_DIR = "/home/messyas/ml/detectron/detectron2/rcascate-resnet50/pesos"
CASCADE_WEIGHTS_FILE = "best.pth"
CASCADE_WEIGHTS_PATH = os.path.join(CASCADE_RCNN_DIR, CASCADE_WEIGHTS_FILE)


NUM_CLASSES = 4 
IMS_PER_BATCH = 16 
BASE_LR = 0.0004 
MAX_ITER = 20000 
STEPS = (12000, 16000) 
GAMMA = 0.1

cfg = get_cfg()

try:
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG_FILE))
    print(f"Arquivo base '{BASE_CONFIG_FILE}' carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o arquivo base '{BASE_CONFIG_FILE}': {e}")
    exit() 

cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
cfg.SOLVER.BASE_LR = BASE_LR
cfg.SOLVER.MAX_ITER = MAX_ITER
cfg.SOLVER.STEPS = STEPS
cfg.SOLVER.GAMMA = GAMMA
cfg.MODEL.WEIGHTS = CASCADE_WEIGHTS_PATH 
os.makedirs(CASCADE_RCNN_DIR, exist_ok=True)
cfg_path = os.path.join(CASCADE_RCNN_DIR, "config_cascade.yaml") 

with open(cfg_path, "w") as f:
    f.write(cfg.dump())

print(f"\nconfig_cascade.yaml salvo em: {cfg_path}")

