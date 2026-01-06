# -----------------------------------------------------------------------------
# 0) Imports e setup
# -----------------------------------------------------------------------------
import os
import copy
import random
import torch
import numpy as np

from detectron2.utils.logger import setup_logger
setup_logger()

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.data import build_detection_train_loader, detection_utils as utils, transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
from detectron2.evaluation import COCOEvaluator

# -----------------------------------------------------------------------------
# 1) Paths do seu dataset
# -----------------------------------------------------------------------------
IMG_DIR   = "/home/messyas/data/images"
ANN_TRAIN = "/home/messyas/data/annotations/4_class_splits/annotations_train.json"
ANN_VAL   = "/home/messyas/data/annotations/4_class_splits/annotations_val.json"
NUM_CLASSES = 4

# -----------------------------------------------------------------------------
# 2) Registro COCO no Detectron2
# -----------------------------------------------------------------------------
register_coco_instances("my_dataset_train", {}, ANN_TRAIN, IMG_DIR)
register_coco_instances("my_dataset_val",   {}, ANN_VAL,   IMG_DIR)

# -----------------------------------------------------------------------------
# 3) custom_mapper com filtro de caixas inválidas e augmentations
# -----------------------------------------------------------------------------
def custom_mapper(dataset_dict):
    d = copy.deepcopy(dataset_dict)
    img = utils.read_image(d["file_name"], format="BGR")
    aug = T.AugmentationList([
        T.ResizeShortestEdge(short_edge_length=(512, 600), max_size=1024, sample_style="choice"),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomRotation(angle=[-15, 15]),
        T.RandomBrightness(0.8, 1.2),
        T.RandomContrast(0.8, 1.2),
        T.RandomSaturation(0.8, 1.2),
    ])
    aug_input = T.AugInput(img)
    transforms = aug(aug_input)
    img = aug_input.image

    annos = []
    for obj in d.pop("annotations"):
        x, y, w, h = obj.get("bbox", [0, 0, 0, 0])
        if w <= 0 or h <= 0:
            continue  # descarta caixas sem área
        annos.append(utils.transform_instance_annotations(obj, transforms, img.shape[:2]))

    d["image"] = torch.as_tensor(img.transpose(2, 0, 1).astype("float32"))
    d["instances"] = utils.annotations_to_instances(annos, img.shape[:2])
    return d

# -----------------------------------------------------------------------------
# 4) Hook de early stopping
# -----------------------------------------------------------------------------
class LossEarlyStopping(HookBase):
    def __init__(self, patience, eval_period):
        self.patience = patience
        self.eval_period = eval_period
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def after_step(self):
        nxt = self.trainer.iter + 1
        if nxt % self.eval_period != 0:
            return

        hist = self.trainer.storage.history("total_loss")
        curr = hist.latest()
        if curr is None:
            return

        if curr < self.best_loss:
            self.best_loss = curr
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                print(f"Early stopping em iter {nxt} — sem melhora em {self.patience} avaliações.")
                self.trainer.max_iter = nxt

# -----------------------------------------------------------------------------
# 5) Trainer customizado
# -----------------------------------------------------------------------------
class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

    def build_hooks(self):
        hooks = super().build_hooks()
        eval_period = self.cfg.TEST.EVAL_PERIOD if self.cfg.TEST.EVAL_PERIOD > 0 else 1
        hooks.insert(-1, LossEarlyStopping(patience=10, eval_period=eval_period))
        return hooks

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
            os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

# -----------------------------------------------------------------------------
# 6) Configuração do cfg
# -----------------------------------------------------------------------------
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))

# Pesos pré-treinados e dispositivo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

# Mixed precision
cfg.SOLVER.AMP.ENABLED = True

# Datasets
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST  = ("my_dataset_val",)

# Dataloader
cfg.DATALOADER.NUM_WORKERS = max(1, os.cpu_count() - 1)

# Solver / Hiperparâmetros
cfg.SOLVER.IMS_PER_BATCH    = 16
cfg.SOLVER.BASE_LR          = 0.004
cfg.SOLVER.WARMUP_ITERS     = 1000
cfg.SOLVER.WARMUP_FACTOR    = 0.001

# Clipping de gradiente
cfg.SOLVER.CLIP_GRADIENTS.ENABLED    = True
cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE  = "value"
cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0

# Desativa avisos de STEPS > MAX_ITER
cfg.SOLVER.STEPS = []

# Cálculo dinâmico de iterações e período de avaliação (~100 épocas)
train_len = len(DatasetCatalog.get("my_dataset_train"))
iters_per_epoch = max(train_len // cfg.SOLVER.IMS_PER_BATCH, 1)
cfg.SOLVER.MAX_ITER    = iters_per_epoch * 100
cfg.TEST.EVAL_PERIOD   = iters_per_epoch

# Número de classes
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

# Diretório de saída
cfg.OUTPUT_DIR = "./output_my_dataset"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# 7) Inicia o treinamento
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
