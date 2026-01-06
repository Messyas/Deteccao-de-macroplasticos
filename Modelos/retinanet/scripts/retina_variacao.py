
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger() 

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator

import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
import copy
import torch
import numpy as np
import os

def register_my_datasets():
    img_dir = "/home/messyas/data/images"
    ann_train = "/home/messyas/data/annotations/4_class_splits/annotations_train.json"
    ann_val = "/home/messyas/data/annotations/4_class_splits/annotations_val.json"

    dataset_train_name = "my_waste_dataset_train" 
    dataset_val_name = "my_waste_dataset_val"  
    
    if not os.path.exists(ann_train):
        raise FileNotFoundError(f"Arquivo de anotacoes de treino nao encontrado: {ann_train}")
        
    register_coco_instances(
        name=dataset_train_name,
        metadata={}, 
        json_file=ann_train,
        image_root=img_dir
    )
    
   
    classes = ["Plastic", "Pile", "Face mask", "Trash bin"] 
    MetadataCatalog.get(dataset_train_name).set(thing_classes=classes)
    
    print(f"Dataset de treino '{dataset_train_name}' registrado.")
    print(f"  Arquivo JSON: {ann_train}")
    print(f"  Raiz das Imagens: {img_dir}")
    print(f"  Classes: {classes}")

    val_name_to_return = None 
    if os.path.exists(ann_val):
        register_coco_instances(
            name=dataset_val_name,
            metadata={},
            json_file=ann_val,
            image_root=img_dir
        )
        MetadataCatalog.get(dataset_val_name).set(thing_classes=classes)
        val_name_to_return = dataset_val_name 
        print(f"Dataset de validação '{dataset_val_name}' registrado.")
        print(f"  Arquivo JSON: {ann_val}")
        print(f"  Raiz das Imagens: {img_dir}")
    else:
        print(f"Arquivo de anotacao de validacao nao encontrado: {ann_val}.")

    return classes, dataset_train_name, val_name_to_return

def custom_train_mapper(dataset_dict, cfg_model):
    dataset_dict = copy.deepcopy(dataset_dict) 
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    utils.check_image_size(dataset_dict, image) 

    augmentations = [
        T.ResizeShortestEdge(
            short_edge_length=cfg_model.INPUT.MIN_SIZE_TRAIN[0],
            max_size=cfg_model.INPUT.MAX_SIZE_TRAIN,
            sample_style="choice"
        ),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False), 
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomRotation(angle=[-15, 15]),
        T.RandomBrightness(0.8, 1.2), 
        T.RandomContrast(0.8, 1.2),   
        T.RandomSaturation(0.8, 1.2), 
    ]
    
    image_transformed, transforms = T.apply_transform_gens(augmentations, image)
    dataset_dict["image"] = torch.as_tensor(image_transformed.transpose(2, 0, 1).astype("float32"))
    annos = [
        utils.transform_instance_annotations(obj, transforms, image_transformed.shape[:2])
        for obj in dataset_dict.pop("annotations") 
        if obj.get("iscrowd", 0) == 0  
    ]
    instances = utils.annotations_to_instances(annos, image_transformed.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    
    return dataset_dict

class RetinaNetTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=lambda dataset_dict: custom_train_mapper(dataset_dict, cfg))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
        os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, distributed=False, output_dir=output_folder)

def setup_cfg(args, num_classes, train_dataset_name, val_dataset_name):
    cfg = get_cfg() 
    try:
        config_file_path = "RetinaNet/retinanet_R_101_FPN_3x.yaml"
        cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
        print(f"Config base carregada em: {config_file_path}")
    except Exception as e:
        print(f"Erro ao carregar config base '{config_file_path}': {e}.")

    cfg.DATASETS.TRAIN = (train_dataset_name,)
    if val_dataset_name:
        cfg.DATASETS.TEST = (val_dataset_name,)
    else:
        cfg.DATASETS.TEST = ()
 
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_101_FPN_3x/190397697/model_final_971ab9.pkl"
    cfg.DATALOADER.NUM_WORKERS = 2 
    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes 
    cfg.INPUT.MIN_SIZE_TRAIN = (800,) 
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.SOLVER.IMS_PER_BATCH = 8 
    cfg.SOLVER.BASE_LR = 0.001  
    cfg.SOLVER.MAX_ITER = 15000 
    cfg.SOLVER.STEPS = (10000, 13000) 
    cfg.SOLVER.GAMMA = 0.1   
    cfg.SOLVER.CHECKPOINT_PERIOD = 2500
    
    if cfg.DATASETS.TEST: 
        cfg.TEST.EVAL_PERIOD = cfg.SOLVER.CHECKPOINT_PERIOD 
    else:
        cfg.TEST.EVAL_PERIOD = 0

    cfg.OUTPUT_DIR = "./output_retinanet" 
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.merge_from_list(args.opts)
    cfg.freeze() 
    default_setup(cfg, args) 
    return cfg


def main(args):
    classes_list, train_dataset_name, val_dataset_name = register_my_datasets()
    num_classes = len(classes_list)
    cfg = setup_cfg(args, num_classes, train_dataset_name, val_dataset_name)
    if args.eval_only:
        if not cfg.DATASETS.TEST: 
            print("Nenhum dataset de teste foi configurado.")
            return {}
        
        model = RetinaNetTrainer.build_model(cfg)
        detectron2.checkpoint.DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        print(f"Rodando avaliacao no dataset: {cfg.DATASETS.TEST[0]}")
        res = RetinaNetTrainer.test(cfg, model)
        return res

    trainer = RetinaNetTrainer(cfg)
    trainer.resume_or_load(resume=args.resume) 
    return trainer.train()

if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    
    print("Argumentos de Linha de Comando Recebidos:", args)
    launch(
        main, 
        args.num_gpus, 
        num_machines=args.num_machines, 
        machine_rank=args.machine_rank, 
        dist_url=args.dist_url,         
        args=(args,),                   
    )
