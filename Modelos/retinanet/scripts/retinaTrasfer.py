import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader, detection_utils as d2_utils, transforms as d2_T
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator

import torch
import copy
import numpy as np
import os
import random

IMG_DIR   = "/home/messyas/ml/data/tccdata/images"
ANN_TRAIN = "/home/messyas/ml/data/tccdata/annotations/4_class_splits/annotations_train.json"
ANN_VAL   = "/home/messyas/ml/data/tccdata/annotations/4_class_splits/annotations_val.json"
NUM_CLASSES = 4
CLASSES_LIST = ["Plastic", "Pile", "Face mask", "Trash bin"]
SEED = 42

def register_datasets_for_retinanet_r101():
    dataset_train_name = "my_dataset_train_retinanet_r101"
    dataset_val_name = "my_dataset_val_retinanet_r101"

    if not os.path.exists(ANN_TRAIN):
        raise FileNotFoundError(f"Arquivo de anotação de treino não encontrado: {ANN_TRAIN}")
    if not os.path.exists(IMG_DIR):
        print(f"AVISO CRÍTICO: Diretório de imagens '{IMG_DIR}' não encontrado.")

    try:
        register_coco_instances(dataset_train_name, {}, ANN_TRAIN, IMG_DIR)
        MetadataCatalog.get(dataset_train_name).set(thing_classes=CLASSES_LIST)
    except AssertionError:
        if not hasattr(MetadataCatalog.get(dataset_train_name), 'thing_classes') or \
           not MetadataCatalog.get(dataset_train_name).thing_classes:
             MetadataCatalog.get(dataset_train_name).set(thing_classes=CLASSES_LIST)

    val_name_to_return = None
    if os.path.exists(ANN_VAL):
        try:
            register_coco_instances(dataset_val_name, {}, ANN_VAL, IMG_DIR)
            MetadataCatalog.get(dataset_val_name).set(thing_classes=CLASSES_LIST)
            val_name_to_return = dataset_val_name
        except AssertionError:
            val_name_to_return = dataset_val_name
            if not hasattr(MetadataCatalog.get(dataset_val_name), 'thing_classes') or \
               not MetadataCatalog.get(dataset_val_name).thing_classes:
                MetadataCatalog.get(dataset_val_name).set(thing_classes=CLASSES_LIST)
    else:
        pass
    
    return dataset_train_name, val_name_to_return

def custom_mapper_retinanet(dataset_dict, model_cfg):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = d2_utils.read_image(dataset_dict["file_name"], format="BGR")
    d2_utils.check_image_size(dataset_dict, image)

    augmentations = d2_T.AugmentationList([
        d2_T.ResizeShortestEdge(
            short_edge_length=model_cfg.INPUT.MIN_SIZE_TRAIN,
            max_size=model_cfg.INPUT.MAX_SIZE_TRAIN,
            sample_style=model_cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        ),
        d2_T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        d2_T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        d2_T.RandomRotation(angle=[-15, 15]),
        d2_T.RandomBrightness(0.8, 1.2),
        d2_T.RandomContrast(0.8, 1.2),
        d2_T.RandomSaturation(0.8, 1.2),
    ])

    aug_input = d2_T.AugInput(image)
    transforms_applied = augmentations(aug_input)
    image_transformed = aug_input.image

    dataset_dict["image"] = torch.as_tensor(image_transformed.transpose(2, 0, 1).astype("float32"))

    annotations_original = dataset_dict.pop("annotations", [])
    annos_transformed = []
    for obj_annotation in annotations_original:
        if obj_annotation.get("iscrowd", 0) == 1:
            continue
        x, y, w, h = obj_annotation.get("bbox", [0,0,0,0])
        if w <=0 or h <= 0:
            continue
        annos_transformed.append(d2_utils.transform_instance_annotations(obj_annotation, transforms_applied, image_transformed.shape[:2]))
    
    instances = d2_utils.annotations_to_instances(annos_transformed, image_transformed.shape[:2])
    dataset_dict["instances"] = d2_utils.filter_empty_instances(instances)
    
    if not dataset_dict["instances"].has("gt_boxes") or len(dataset_dict["instances"].gt_boxes) == 0:
        return None
        
    return dataset_dict

class RetinaNetTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=lambda dd: custom_mapper_retinanet(dd, cfg))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
        os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, distributed=False, output_dir=output_folder)

def main_retinanet_r101_feature_extraction(args):
    train_dataset_name, val_dataset_name = register_datasets_for_retinanet_r101()

    if not train_dataset_name:
        raise ValueError("O nome do dataset de treino não foi retornado ou é inválido.")

    cfg = get_cfg()
    
    CONFIG_FILE_BASE = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file_path)
    
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    if val_dataset_name:
        cfg.DATASETS.TEST = (val_dataset_name,)
    else:
        cfg.DATASETS.TEST = ()

    cfg.DATALOADER.NUM_WORKERS = min(max(1, os.cpu_count() -1), 2)
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True

    cfg.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
    
    cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 600) 
    cfg.INPUT.MAX_SIZE_TRAIN = 1024 
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"

    cfg.SOLVER.IMS_PER_BATCH = 24 
    cfg.SOLVER.BASE_LEARNING_RATE = 0.0005
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.AMP.ENABLED = True

    num_epochs_for_feature_extraction = 30
    
    iters_per_epoch = 1 
    try:
        dataset_size = len(DatasetCatalog.get(train_dataset_name))
        if dataset_size > 0 and cfg.SOLVER.IMS_PER_BATCH > 0:
            iters_per_epoch = max(dataset_size // cfg.SOLVER.IMS_PER_BATCH, 1)
        cfg.SOLVER.MAX_ITER = iters_per_epoch * num_epochs_for_feature_extraction
    except Exception:
        cfg.SOLVER.MAX_ITER = 15000 
        if num_epochs_for_feature_extraction > 0 :
             iters_per_epoch = cfg.SOLVER.MAX_ITER // num_epochs_for_feature_extraction
    
    if iters_per_epoch <= 0: iters_per_epoch = 1

    cfg.SOLVER.STEPS = (int(0.7 * cfg.SOLVER.MAX_ITER), int(0.9 * cfg.SOLVER.MAX_ITER))
    if not (cfg.SOLVER.STEPS and \
            0 < cfg.SOLVER.STEPS[0] < cfg.SOLVER.MAX_ITER and \
            cfg.SOLVER.STEPS[0] < cfg.SOLVER.STEPS[1] < cfg.SOLVER.MAX_ITER):
        cfg.SOLVER.STEPS = [] 

    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = max(iters_per_epoch * 1, 500)
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.WARMUP_FACTOR = 0.001

    cfg.SOLVER.CHECKPOINT_PERIOD = iters_per_epoch * 5 
    if cfg.DATASETS.TEST:
        cfg.TEST.EVAL_PERIOD = iters_per_epoch * 1 
    else:
        cfg.TEST.EVAL_PERIOD = 0 
    
    cfg.OUTPUT_DIR = "./output_retinanet_R101_feat_extract_corrected_paths"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    cfg.SEED = SEED

    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    if args.eval_only:
        if not cfg.DATASETS.TEST:
            return {}
        model = RetinaNetTrainer.build_model(cfg)
        detectron2.checkpoint.DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = RetinaNetTrainer.test(cfg, model)
        return res

    trainer = RetinaNetTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    frozen_param_count = 0
    if hasattr(trainer.model, 'backbone'):
        for param_name, parameter_obj in trainer.model.backbone.named_parameters():
            parameter_obj.requires_grad = False
            frozen_param_count += 1
    
    try:
        return trainer.train()
    except Exception as e:
        pass
    finally:
        pass

if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    
    launch(
        main_retinanet_r101_feature_extraction,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
