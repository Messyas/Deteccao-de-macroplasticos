#!/usr/bin/env python
# coding: utf-8

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from PIL import Image
import cv2
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1) SEED E CONFIGURAÇÃO BÁSICA
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# 2) CAMINHOS E PARÂMETROS DE AVALIAÇÃO
# -----------------------------------------------------------------------------

IMG_DIR = "/home/messyas/data/images" 
ANN_TEST_FILE_PATH = "/home/messyas/data/annotations/4_class_splits/annotations_test.json" 
MODEL_WEIGHTS_PATH = "/home/messyas/detectron/rersnet50/pesos/best.pth"

NUM_CLASSES_MODEL = 4  # Total (10 foreground + 1 background) que o FastRCNNPredictor foi configurado

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !! IMPORTANTE: SUBSTITUA OS VALORES EXEMPLO (1, 2, 3, 4) ABAIXO
# !!             PELOS 'category_id' NUMÉRICOS REAIS DO SEU ARQUIVO JSON COCO
# !!             PARA CADA CLASSE, NA ORDEM CORRESPONDENTE AOS LABELS DE SAÍDA DO MODELO 1, 2, 3, 4.
# !!             Ex: Se "Plastic" é ID 10 no JSON e é a 1ª classe do modelo, use 1: 10.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MODEL_FG_LABEL_TO_TRUE_COCO_ID_MAP = {
    # ===== INÍCIO DA SEÇÃO QUE VOCÊ PRECISA EDITAR =====
    1: 1,  # Exemplo: Label 1 do modelo -> "Plastic" (assumindo COCO ID 1 no seu JSON)
    2: 2,  # Exemplo: Label 2 do modelo -> "Pile" (assumindo COCO ID 2 no seu JSON)
    3: 3,  # Exemplo: Label 3 do modelo -> "Face mask" (assumindo COCO ID 3 no seu JSON)
    4: 4,  # Exemplo: Label 4 do modelo -> "Trash bin" (assumindo COCO ID 4 no seu JSON)
    # ===== FIM DA SEÇÃO QUE VOCÊ PRECISA EDITAR =====
}

if not MODEL_FG_LABEL_TO_TRUE_COCO_ID_MAP or \
   not all(isinstance(k, int) and k > 0 and isinstance(v, int) for k, v in MODEL_FG_LABEL_TO_TRUE_COCO_ID_MAP.items()):
    print("ERRO CRÍTICO: MODEL_FG_LABEL_TO_TRUE_COCO_ID_MAP não foi definido corretamente.")
    exit()
if len(MODEL_FG_LABEL_TO_TRUE_COCO_ID_MAP) > (NUM_CLASSES_MODEL - 1 ):
    print(f"AVISO: MODEL_FG_LABEL_TO_TRUE_COCO_ID_MAP tem {len(MODEL_FG_LABEL_TO_TRUE_COCO_ID_MAP)} entradas, "
          f"mas o modelo foi configurado para no máximo {NUM_CLASSES_MODEL-1} classes de foreground. Verifique.")

# ATIVAR GPU SE DISPONÍVEL
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {DEVICE}")

OUTPUT_DIR = 'pytorch_fasterrcnn_evaluation_outputs_gpu' # Nome do diretório de saída para clareza
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_PRED_COCO_JSON = os.path.join(OUTPUT_DIR, 'pytorch_predictions_coco_format.json')
METRICS_FILE = os.path.join(OUTPUT_DIR, 'pytorch_evaluation_metrics.txt')
CONFUSION_MATRIX_FILE = os.path.join(OUTPUT_DIR, 'pytorch_confusion_matrix.png')
NUM_EXAMPLE_IMAGES = 4

PRED_SCORE_THRESHOLD_COCO = 0.05
PRED_SCORE_THRESHOLD_VIS = 0.5
IOU_THRESHOLD_CM = 0.5

# -----------------------------------------------------------------------------
# 3) DATASET E DATALOADER PARA TESTE
# -----------------------------------------------------------------------------
test_val_transforms = T.Compose([T.ToTensor()])

class CocoLikeDataset(Dataset):
    def __init__(self, img_dir_root, ann_file_path, transforms=None, return_img_id_bool=False):
        from torchvision.datasets import CocoDetection
        self.coco_api_instance = COCO(ann_file_path)
        self.img_dir_root = img_dir_root
        self.transforms = transforms
        self.return_img_id_bool = return_img_id_bool
        self.ids = list(sorted(self.coco_api_instance.imgs.keys()))

    def __getitem__(self, idx):
        img_coco_id = self.ids[idx]
        img_info_dict = self.coco_api_instance.loadImgs(img_coco_id)[0]
        img_file_path = os.path.join(self.img_dir_root, img_info_dict['file_name'])

        try:
            img_pil = Image.open(img_file_path).convert("RGB")
        except FileNotFoundError:
            print(f"ERRO: Imagem não encontrada em {img_file_path} para img_id {img_coco_id}")
            if self.return_img_id_bool: return None, None, img_coco_id
            return None, None

        ann_ids_list = self.coco_api_instance.getAnnIds(imgIds=img_coco_id)
        anns_list = self.coco_api_instance.loadAnns(ann_ids_list)

        boxes_list, labels_list, areas_list, iscrowd_list = [], [], [], []
        for obj_ann in anns_list:
            x, y, w, h = obj_ann['bbox']
            if w <= 0 or h <= 0 or obj_ann.get('ignore', 0) == 1:
                continue
            
            boxes_list.append([x, y, x + w, y + h])
            labels_list.append(obj_ann['category_id'])
            areas_list.append(obj_ann['area'])
            iscrowd_list.append(obj_ann.get('iscrowd', 0))

        target_dict = {}
        target_dict['boxes'] = torch.tensor(boxes_list, dtype=torch.float32) if boxes_list else torch.zeros((0, 4), dtype=torch.float32)
        target_dict['labels'] = torch.tensor(labels_list, dtype=torch.int64) if labels_list else torch.zeros(0, dtype=torch.int64)
        target_dict['image_id'] = torch.tensor([img_coco_id])
        target_dict['area'] = torch.tensor(areas_list, dtype=torch.float32) if areas_list else torch.zeros(0, dtype=torch.float32)
        target_dict['iscrowd'] = torch.tensor(iscrowd_list, dtype=torch.int64) if iscrowd_list else torch.zeros(0, dtype=torch.int64)

        if self.transforms:
            img_pil = self.transforms(img_pil)
        
        if self.return_img_id_bool: return img_pil, target_dict, img_coco_id
        return img_pil, target_dict

    def __len__(self):
        return len(self.ids)

def collate_fn_eval_batch(batch_data):
    batch_data_filtered = [b for b in batch_data if b[0] is not None]
    if not batch_data_filtered: return None, None, None
    images_b, targets_b, img_ids_b = zip(*batch_data_filtered)
    return list(images_b), list(targets_b), list(img_ids_b)

print(f"Carregando dataset de teste de: {ANN_TEST_FILE_PATH}")
try:
    test_dataset = CocoLikeDataset(IMG_DIR, ANN_TEST_FILE_PATH, transforms=test_val_transforms, return_img_id_bool=True)
    if len(test_dataset) == 0:
        print(f"ERRO CRÍTICO: Nenhum dado carregado do dataset de teste em {ANN_TEST_FILE_PATH}. Verifique o arquivo.")
        exit()
except Exception as e:
    print(f"ERRO CRÍTICO ao carregar o dataset de teste: {e}")
    exit()

test_loader = DataLoader(
    test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_eval_batch, num_workers=2
)

# -----------------------------------------------------------------------------
# 4) CARREGAR MODELO E PESOS
# -----------------------------------------------------------------------------
print(f"Carregando modelo Faster R-CNN (ResNet-50 FPN) com {NUM_CLASSES_MODEL} classes (para o predictor)...")
model = fasterrcnn_resnet50_fpn(weights=None, progress=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES_MODEL)

print(f"Carregando pesos treinados de: {MODEL_WEIGHTS_PATH}")
try:
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
except FileNotFoundError:
    print(f"ERRO CRÍTICO: Arquivo de pesos não encontrado em {MODEL_WEIGHTS_PATH}")
    exit()
except RuntimeError as e:
    print(f"ERRO CRÍTICO: Falha ao carregar os pesos. Verifique se NUM_CLASSES_MODEL ({NUM_CLASSES_MODEL}) "
          f"corresponde ao modelo que gerou os pesos. Erro: {e}")
    exit()
    
model.to(DEVICE)
model.eval()
print("Modelo carregado e em modo de avaliação.")

# -----------------------------------------------------------------------------
# 5) LOOP DE AVALIAÇÃO E COLETA DE PREDIÇÕES
# -----------------------------------------------------------------------------
coco_predictions_list = []
all_preds_for_cm_vis = []

try:
    coco_gt_api = COCO(ANN_TEST_FILE_PATH)
except Exception as e:
    print(f"ERRO CRÍTICO: Falha ao carregar o arquivo de anotações de teste COCO '{ANN_TEST_FILE_PATH}': {e}")
    exit()

gt_categories_info_dict = coco_gt_api.loadCats(coco_gt_api.getCatIds())
gt_id_to_name_map = {cat['id']: cat['name'] for cat in gt_categories_info_dict}
cm_gt_coco_ids_list = sorted(list(gt_id_to_name_map.keys()))
cm_class_names_list = [gt_id_to_name_map[id_val] for id_val in cm_gt_coco_ids_list]
cm_num_display_classes = len(cm_gt_coco_ids_list)
cm_coco_id_to_matrix_idx_map = {id_val: i for i, id_val in enumerate(cm_gt_coco_ids_list)}

print(f"Classes consideradas para Matriz de Confusão (do GT de teste): {cm_class_names_list}")

print(f"\nIniciando avaliação no conjunto de teste (Dispositivo: {DEVICE})...")
for images_batch, _, image_ids_in_batch in tqdm(test_loader, desc=f"Avaliando ({DEVICE})"):
    if images_batch is None: continue
    
    images_on_device = list(img.to(DEVICE) for img in images_batch)
    
    with torch.no_grad():
        model_predictions_batch = model(images_on_device)

    for i_img_in_batch, single_img_prediction in enumerate(model_predictions_batch):
        current_coco_img_id = image_ids_in_batch[i_img_in_batch]
        
        pred_boxes_abs = single_img_prediction['boxes'].cpu().numpy()
        pred_model_labels = single_img_prediction['labels'].cpu().numpy()
        pred_scores_val = single_img_prediction['scores'].cpu().numpy()

        for i_box in range(len(pred_boxes_abs)):
            current_score = pred_scores_val[i_box]
            current_model_label = pred_model_labels[i_box]
            
            true_coco_id_for_this_pred = MODEL_FG_LABEL_TO_TRUE_COCO_ID_MAP.get(current_model_label)
            if true_coco_id_for_this_pred is None: continue 

            if current_score >= PRED_SCORE_THRESHOLD_VIS:
                if true_coco_id_for_this_pred in gt_id_to_name_map:
                    all_preds_for_cm_vis.append({
                        'image_id': current_coco_img_id,
                        'category_id': true_coco_id_for_this_pred,
                        'bbox_abs_xyxy': pred_boxes_abs[i_box].tolist(),
                        'score': float(current_score)
                    })
            
            if current_score >= PRED_SCORE_THRESHOLD_COCO:
                x1, y1, x2, y2 = pred_boxes_abs[i_box]
                coco_predictions_list.append({
                    "image_id": current_coco_img_id,
                    "category_id": true_coco_id_for_this_pred,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(current_score)
                })
    
    # Otimização: Deletar tensores da GPU e limpar cache
    del images_on_device
    del model_predictions_batch
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# 6) CÁLCULO DE MÉTRICAS COCO
# (O restante do script permanece o mesmo)
# -----------------------------------------------------------------------------
if not coco_predictions_list:
    print("Nenhuma predição (acima do limiar para COCOeval) foi feita. Métricas COCO não podem ser calculadas.")
else:
    print(f"\nSalvando {len(coco_predictions_list)} predições no formato COCO em {OUT_PRED_COCO_JSON}")
    with open(OUT_PRED_COCO_JSON, 'w') as f: json.dump(coco_predictions_list, f, indent=4)

    print("Calculando métricas COCO com pycocotools...")
    try:
        coco_dt_results = coco_gt_api.loadRes(OUT_PRED_COCO_JSON)
    except Exception as e:
        print(f"ERRO ao carregar predições para COCOeval: {e}")
        print("Verifique o formato do arquivo JSON de predições.")
        coco_dt_results = None

    if coco_dt_results:
        coco_evaluator = COCOeval(coco_gt_api, coco_dt_results, iouType='bbox')
        
        coco_evaluator.evaluate()
        coco_evaluator.accumulate()
        print("\n--- Resumo das Métricas COCO (PyTorch Faster R-CNN) ---")
        coco_evaluator.summarize()
        print("-------------------------------------------------------")

        with open(METRICS_FILE, 'w') as f:
            f.write(f"PyTorch Faster R-CNN - COCO Evaluation Metrics (Device: {DEVICE}):\n")
            metric_names_list = [
                "AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]", "AP @[ IoU=0.50      | area=   all | maxDets=100 ] (mAP@.50)",
                "AP @[ IoU=0.75      | area=   all | maxDets=100 ]", "AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
                "AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]", "AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
                "AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]", "AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]",
                "AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]", "AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
                "AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]", "AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]"
            ]
            for i_stat_idx, stat_value in enumerate(coco_evaluator.stats):
                f.write(f"{metric_names_list[i_stat_idx]}: {stat_value:.4f}\n")
            f.write("\nNota: As métricas de Precisão, Recall e F1-score por classe para um ponto de operação específico "
                    "não são diretamente fornecidas pelo COCOeval. Estas são as métricas AP/AR padrão.\n")
        print(f"Métricas COCO salvas em {METRICS_FILE}")

# -----------------------------------------------------------------------------
# 7) MATRIZ DE CONFUSÃO
# -----------------------------------------------------------------------------
def calculate_iou_for_cm(box1_xyxy_coords, box2_xyxy_coords):
    x1_intersect = max(box1_xyxy_coords[0], box2_xyxy_coords[0])
    y1_intersect = max(box1_xyxy_coords[1], box2_xyxy_coords[1])
    x2_intersect = min(box1_xyxy_coords[2], box2_xyxy_coords[2])
    y2_intersect = min(box1_xyxy_coords[3], box2_xyxy_coords[3])
    inter_width_val = max(0, x2_intersect - x1_intersect); inter_height_val = max(0, y2_intersect - y1_intersect)
    area_intersect = inter_width_val * inter_height_val
    area_box1 = (box1_xyxy_coords[2] - box1_xyxy_coords[0]) * (box1_xyxy_coords[3] - box1_xyxy_coords[1])
    area_box2 = (box2_xyxy_coords[2] - box2_xyxy_coords[0]) * (box2_xyxy_coords[3] - box2_xyxy_coords[1])
    area_union = area_box1 + area_box2 - area_intersect
    return area_intersect / area_union if area_union > 0 else 0

if cm_num_display_classes > 0 and all_preds_for_cm_vis:
    print("\nCalculando Matriz de Confusão...")
    confusion_matrix_plot_data = np.zeros((cm_num_display_classes, cm_num_display_classes), dtype=np.int32)
    
    preds_for_cm_by_img_id_dict = {}
    for pred_cm_item in all_preds_for_cm_vis:
        img_id_cm_item = pred_cm_item['image_id']
        if img_id_cm_item not in preds_for_cm_by_img_id_dict: preds_for_cm_by_img_id_dict[img_id_cm_item] = []
        preds_for_cm_by_img_id_dict[img_id_cm_item].append(pred_cm_item)

    for current_img_id_cm in tqdm(coco_gt_api.getImgIds(), desc=f"Processando CM ({DEVICE})"):
        gt_ann_ids_for_cm = coco_gt_api.getAnnIds(imgIds=current_img_id_cm, iscrowd=False)
        gts_in_current_img = coco_gt_api.loadAnns(gt_ann_ids_for_cm)
        
        preds_in_current_img = sorted(preds_for_cm_by_img_id_dict.get(current_img_id_cm, []), key=lambda p_item:p_item['score'], reverse=True)
        
        gt_already_matched_flags = [False] * len(gts_in_current_img)

        for pred_item_for_cm in preds_in_current_img:
            pred_bbox_for_cm = pred_item_for_cm['bbox_abs_xyxy']
            pred_true_coco_id = pred_item_for_cm['category_id']
            
            pred_matrix_col_idx = cm_coco_id_to_matrix_idx_map.get(pred_true_coco_id)
            if pred_matrix_col_idx is None: continue

            iou_best_match = 0
            gt_best_match_idx = -1

            for i_gt_item_cm, gt_ann_for_cm in enumerate(gts_in_current_img):
                if gt_already_matched_flags[i_gt_item_cm]: continue
                if gt_ann_for_cm['category_id'] not in cm_coco_id_to_matrix_idx_map: continue
                
                gt_bbox_coco_format = gt_ann_for_cm['bbox']
                gt_bbox_xyxy_format = [gt_bbox_coco_format[0], gt_bbox_coco_format[1], 
                                       gt_bbox_coco_format[0] + gt_bbox_coco_format[2], 
                                       gt_bbox_coco_format[1] + gt_bbox_coco_format[3]]
                
                current_iou = calculate_iou_for_cm(pred_bbox_for_cm, gt_bbox_xyxy_format)
                if current_iou > iou_best_match:
                    iou_best_match = current_iou
                    gt_best_match_idx = i_gt_item_cm
            
            if iou_best_match >= IOU_THRESHOLD_CM:
                matched_gt_annotation = gts_in_current_img[gt_best_match_idx]
                true_gt_coco_id = matched_gt_annotation['category_id']
                true_matrix_row_idx = cm_coco_id_to_matrix_idx_map.get(true_gt_coco_id)

                if true_matrix_row_idx is not None:
                    confusion_matrix_plot_data[true_matrix_row_idx, pred_matrix_col_idx] += 1
                    gt_already_matched_flags[gt_best_match_idx] = True
    
    plt.figure(figsize=(max(8, cm_num_display_classes * 1.0), max(6, cm_num_display_classes * 0.9)))
    sns.heatmap(confusion_matrix_plot_data, annot=True, fmt='d', cmap='Blues',
                xticklabels=cm_class_names_list, yticklabels=cm_class_names_list)
    plt.ylabel('Classe Verdadeira (Ground Truth)')
    plt.xlabel('Classe Predita pelo Modelo')
    plt.title(f'Matriz de Confusão (PyTorch Faster R-CNN {DEVICE})\nIoU Thresh: {IOU_THRESHOLD_CM}, Score Thresh: {PRED_SCORE_THRESHOLD_VIS}')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE)
    plt.close()
    print(f"Matriz de confusão salva em {CONFUSION_MATRIX_FILE}")
else:
    print("Pulando Matriz de Confusão: sem classes de GT no teste ou sem predições filtradas.")

# -----------------------------------------------------------------------------
# 8) PLOTAR EXEMPLOS DE IMAGENS
# -----------------------------------------------------------------------------
if all_preds_for_cm_vis :
    print("\nPlotando imagens de exemplo...")
    test_img_ids_for_vis = coco_gt_api.getImgIds()
    if test_img_ids_for_vis:
        selected_img_ids_for_vis = np.random.choice(test_img_ids_for_vis, 
                                                size=min(NUM_EXAMPLE_IMAGES, len(test_img_ids_for_vis)), 
                                                replace=False)
        
        color_gt_example = (0, 255, 0)
        color_pred_example = (0, 0, 255)

        for i_example_img, current_img_id_example in enumerate(tqdm(selected_img_ids_for_vis, desc=f"Plotando exemplos ({DEVICE})")):
            img_info_example = coco_gt_api.loadImgs(current_img_id_example)[0]
            img_file_path_example = os.path.join(IMG_DIR, img_info_example['file_name'])
            image_bgr_to_draw = cv2.imread(img_file_path_example)
            if image_bgr_to_draw is None: continue

            # Desenhar GT
            gt_ann_ids_example = coco_gt_api.getAnnIds(imgIds=current_img_id_example, iscrowd=False)
            gts_in_img_example = coco_gt_api.loadAnns(gt_ann_ids_example)
            for ann_item_example in gts_in_img_example:
                x_gt, y_gt, w_gt, h_gt = [int(val) for val in ann_item_example['bbox']]
                cat_id_gt = ann_item_example['category_id']
                cat_name_gt = gt_id_to_name_map.get(cat_id_gt, f"ID:{cat_id_gt}")
                cv2.rectangle(image_bgr_to_draw, (x_gt,y_gt), (x_gt+w_gt, y_gt+h_gt), color_gt_example, 2)
                cv2.putText(image_bgr_to_draw, f"GT: {cat_name_gt}", (x_gt, y_gt - 10 if y_gt > 20 else y_gt + h_gt + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_gt_example, 1)
            
            # Desenhar Predições
            current_img_preds_example = [p_ex for p_ex in all_preds_for_cm_vis if p_ex['image_id'] == current_img_id_example]
            for pred_item_example in current_img_preds_example:
                x1p_ex, y1p_ex, x2p_ex, y2p_ex = [int(coord) for coord in pred_item_example['bbox_abs_xyxy']]
                pred_true_coco_id_example = pred_item_example['category_id']
                score_example = pred_item_example['score']
                pred_cat_name_example = gt_id_to_name_map.get(pred_true_coco_id_example, f"ID:{pred_true_coco_id_example}")

                cv2.rectangle(image_bgr_to_draw, (x1p_ex, y1p_ex), (x2p_ex, y2p_ex), color_pred_example, 2)
                text_y_pos = y1p_ex - 10 if y1p_ex > 30 else y2p_ex + 15 
                cv2.putText(image_bgr_to_draw, f"P: {pred_cat_name_example} ({score_example:.2f})",
                            (x1p_ex, text_y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_pred_example, 1)
            
            out_img_file_path_example = os.path.join(OUTPUT_DIR, f"example_pytorch_{DEVICE}_{i_example_img+1}_{img_info_example['file_name']}")
            cv2.imwrite(out_img_file_path_example, image_bgr_to_draw)
        print(f"Imagens de exemplo salvas em {OUTPUT_DIR}")
else:
    print("Pulando plotagem de imagens de exemplo: sem predições (filtradas para CM/Vis).")

print(f"\nAvaliação e geração de plots (PyTorch Faster R-CNN - {DEVICE}) concluídos. Resultados em: {os.path.abspath(OUTPUT_DIR)}")
