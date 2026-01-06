from detectron2 import model_zoo
from detectron2.config import get_cfg
import os

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.SOLVER.IMS_PER_BATCH        = 16
cfg.SOLVER.BASE_LR              = 0.0004
cfg.SOLVER.MAX_ITER             = 20000
cfg.SOLVER.STEPS                = (12000, 16000)
cfg.SOLVER.GAMMA                = 0.1
cfg.MODEL.WEIGHTS = "/home/messyas/ml/tcc/models/combinacao/rcnn/rcnn.pth"

out_dir  = "/home/messyas/ml/tcc/models/combinacao/rcnn"
os.makedirs(out_dir, exist_ok=True)
cfg_path = os.path.join(out_dir, "config.yaml")

with open(cfg_path, "w") as f:
    f.write(cfg.dump())

print(f" config.yaml salvo em: {cfg_path}")
