import os
import copy
import torch
import logging  
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
from detectron2.evaluation import COCOEvaluator

IMG_DIR = "/home/messyas/data/images"
ANN_TRAIN = "/home/messyas/data/annotations/4_class_splits/annotations_train.json"
ANN_VAL = "/home/messyas/data/annotations/4_class_splits/annotations_val.json"

register_coco_instances("my_train", {}, ANN_TRAIN, IMG_DIR)
register_coco_instances("my_val", {}, ANN_VAL, IMG_DIR)

processed_count = 0
skipped_read_error = 0
skipped_none_img = 0
skipped_no_annos = 0

def custom_mapper(dataset_dict):
    global processed_count, skipped_read_error, skipped_none_img, skipped_no_annos

    d = copy.deepcopy(dataset_dict)
    try:
        img = utils.read_image(d["file_name"], format="BGR")
    except Exception as e:
        print(f"Erro ao ler a imagem {d['file_name']}: {e}")
        skipped_read_error += 1
        return None 

    if img is None:
        print(f"Imagem {d['file_name']} npo pode ser carregada.")
        skipped_none_img += 1
        return None

    aug = T.AugmentationList([
        T.ResizeShortestEdge(short_edge_length=1024, max_size=1333),
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
    original_annos_count = len(d.get("annotations", []))
    filtered_annos_count = 0

    for obj in d.pop("annotations"):
        x, y, w, h = obj.get("bbox", [0, 0, 0, 0])
        if w <= 0 or h <= 0:
            filtered_annos_count += 1
            continue

        anno = utils.transform_instance_annotations(obj, transforms, img.shape[:2])
        _x, _y, _w, _h = anno["bbox"]
        x1, y1, x2, y2 = anno["bbox"]
        _w = x2 - x1
        _h = y2 - y1

        if _w > 1 and _h > 1: 
            annos.append(anno)
        else:
            filtered_annos_count += 1


    if not annos:
        print(f"DEBUG MAPPER: Imagem {d['file_name']} sem anotacoes validas ({filtered_annos_count}/{original_annos_count} filtradas), pulando.")
        skipped_no_annos += 1
        return None

    processed_count += 1
    if processed_count % 100 == 0:
        print(f"DEBUG MAPPER: Processadas={processed_count}, Erro Leitura={skipped_read_error}, Img Nula={skipped_none_img}, Sem Anno={skipped_no_annos}")

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
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
            os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

if __name__ == "__main__":
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.SOLVER.AMP.ENABLED = True
    cfg.DATASETS.TRAIN = ("my_train",)
    cfg.DATASETS.TEST = ("my_val",)
    cfg.DATALOADER.NUM_WORKERS = max(1, os.cpu_count() - 1)
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True

    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.00001

    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_FACTOR = 0.001

    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
 
    train_dataset = DatasetCatalog.get("my_train")
    train_len = len(train_dataset)
    print(f"Numero de imagens de treino registradas: {train_len}")
    if train_len == 0:
        print("Nenhuma imagem encontrada no dataset de treino")

    iters_per_epoch = max(train_len // cfg.SOLVER.IMS_PER_BATCH, 1)
    cfg.SOLVER.MAX_ITER = iters_per_epoch * 100
    cfg.SOLVER.STEPS = (int(cfg.SOLVER.MAX_ITER * 0.67), int(cfg.SOLVER.MAX_ITER * 0.9))
    cfg.TEST.EVAL_PERIOD = iters_per_epoch
    cfg.MODEL.RETINANET.NUM_CLASSES = 4
    
   
    cfg.OUTPUT_DIR = "./output_retinanet_R101" 
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    log_file_path = os.path.join(cfg.OUTPUT_DIR, "log.txt")
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=0, name="detectron2")
    logger = logging.getLogger("detectron2")
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    print(f"--- Configuração Finalizada ---")
    print(f"Output Dir: {cfg.OUTPUT_DIR}")
    print(f"Log File: {log_file_path}")
    print(f"Max Iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"Batch Size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"Base LR: {cfg.SOLVER.BASE_LR}")
    print(f"---------------------------------")

    torch.cuda.empty_cache()
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    
    print("--- Iniciando Treinamento ---")
    trainer.train()
    print("--- Treinamento Concluido ---")
    print(f"Total Processadas: {processed_count}")
    print(f"Erro Leitura: {skipped_read_error}")
    print(f"Img Nula): {skipped_none_img}")
    print(f"-------------------------")
