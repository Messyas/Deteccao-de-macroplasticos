import os
import copy
import random
import torch
import numpy as np

from detectron2.utils.logger import setup_logger
setup_logger()

from torch.optim.lr_scheduler import ReduceLROnPlateau
from detectron2.engine import HookBase, DefaultTrainer
from detectron2.data import build_detection_train_loader, detection_utils as utils, transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog

class LRPlateauHook(HookBase):
    def __init__(self, optimizer, patience=2, factor=0.1, mode="max"):
        self.scheduler = ReduceLROnPlateau(
            optimizer, mode=mode, patience=patience,
            factor=factor, verbose=True
        )
    def after_step(self):
        it = self.trainer.iter + 1
        if it % self.trainer.cfg.TEST.EVAL_PERIOD != 0:
            return
        map_val = self.trainer.storage.history("bbox/AP").latest()
        if map_val is not None:
            self.scheduler.step(map_val)

class MapEarlyStopping(HookBase):
    def __init__(self, patience=5):
        self.patience = patience
        self.best_map = 0.0
        self.bad_runs = 0
        self.eval_period = None

    def before_train(self):
        self.eval_period = self.trainer.cfg.TEST.EVAL_PERIOD

    def after_step(self):
        it = self.trainer.iter + 1
        if self.eval_period is None or it % self.eval_period != 0:
            return
        curr_map = self.trainer.storage.history("bbox/AP").latest()
        if curr_map is None:
            return
        if curr_map > self.best_map:
            self.best_map = curr_map
            self.bad_runs = 0
        else:
            self.bad_runs += 1
            if self.bad_runs >= self.patience:
                print(f"[EarlyStop mAP] parada em iter {it} (sem melhoria em {self.patience} ciclos).")
                self.trainer.max_iter = it

def custom_mapper(dataset_dict):
    d = copy.deepcopy(dataset_dict)
    img = utils.read_image(d["file_name"], format="BGR")
    aug = T.AugmentationList([
         T.ResizeShortestEdge(short_edge_length=600, max_size=1000), 
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
    for obj in d.get("annotations", []):
        x, y, w, h = obj.get("bbox", [0, 0, 0, 0])
        if w <= 0 or h <= 0:
            continue
        try:
            ann = utils.transform_instance_annotations(obj, transforms, img.shape[:2])
            annos.append(ann)
        except Exception:
            continue

    d["image"] = torch.as_tensor(img.transpose(2, 0, 1).astype("float32"))
    d["instances"] = utils.annotations_to_instances(annos, img.shape[:2])
    return d

class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
            os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        ep = self.cfg.TEST.EVAL_PERIOD
        hooks.insert(-1, MapEarlyStopping(patience=5))
        hooks.insert(-1, LRPlateauHook(self.optimizer, patience=2, factor=0.1))
        return hooks

if __name__ == "__main__":
  
    dataset_dir = "/home/messyas/data"
    register_coco_instances(
        "my_dataset_train", {},
        f"{dataset_dir}/annotations/4_class_splits/annotations_train.json",
        f"{dataset_dir}/images"
    )
    register_coco_instances(
        "my_dataset_val", {},
        f"{dataset_dir}/annotations/4_class_splits/annotations_val.json",
        f"{dataset_dir}/images"
    )

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg.SOLVER.AMP.ENABLED = True
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = max(1, os.cpu_count() - 1)

    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.SOLVER.BASE_LR = 0.0004
    cfg.SOLVER.WARMUP_ITERS = 2000
    cfg.SOLVER.WARMUP_FACTOR = 0.01
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0

    train_len = len(DatasetCatalog.get("my_dataset_train"))
    its_per_epoch = max(train_len // cfg.SOLVER.IMS_PER_BATCH, 1)
    total_iters = its_per_epoch * 100
    cfg.SOLVER.MAX_ITER = total_iters
    cfg.TEST.EVAL_PERIOD = its_per_epoch
    cfg.SOLVER.STEPS = (int(0.6 * total_iters), int(0.8 * total_iters))
    cfg.SOLVER.GAMMA = 0.1

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.OUTPUT_DIR = "./output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

