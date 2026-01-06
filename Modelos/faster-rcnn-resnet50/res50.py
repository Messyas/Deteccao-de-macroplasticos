#!/usr/bin/env python3
import os
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog # Para nomes de classes do modelo
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# --- CONFIGURAÇÕES ---
IMG_DIR = "/home/messyas/data/images"
ANN_FILE = "/home/messyas/data/annotations/4_class_splits/annotations_test.json"

# Caminho para o arquivo de configuração do Detectron2 (usado no treinamento do best.pth)
CFG_FILE_PATH = "/home/messyas/detectron/rersnet50/config.yaml" # Verifique se este é o config correto para os pesos abaixo
# Caminho para os pesos do modelo Detectron2
MODEL_WEIGHTS_PATH = "/home/messyas/detectron/rersnet50/best.pth"

# Limiar de score para considerar uma detecção (afeta o que o predictor retorna)
# Se muito baixo, pode gerar muitas predições de baixa confiança.
# O script original usava 0.05.
MODEL_SCORE_THRESH = 0.3 # Ajuste conforme necessário para visualização e CM
# Limiar de IoU para parear predições com GT (para CM e exemplos)
IOU_THR_MATCHING = 0.5

OUTPUT_DIR = 'detectron2_evaluation_outputs'
OUT_PRED_JSON = os.path.join(OUTPUT_DIR, 'detectron2_preds.json')
METRICS_FILE = os.path.join(OUTPUT_DIR, 'detectron2_evaluation_metrics.txt')
CONFUSION_MATRIX_FILE = os.path.join(OUTPUT_DIR, 'detectron2_confusion_matrix.png')
NUM_EXAMPLE_IMAGES = 4

# Criar diretório de saída
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_iou(box1_xyxy, box2_xyxy):
    """ Calcula o IoU entre duas caixas delimitadoras no formato [x1, y1, x2, y2]. """
    x1_i = max(box1_xyxy[0], box2_xyxy[0])
    y1_i = max(box1_xyxy[1], box2_xyxy[1])
    x2_i = min(box1_xyxy[2], box2_xyxy[2])
    y2_i = min(box1_xyxy[3], box2_xyxy[3])

    inter_width = max(0, x2_i - x1_i)
    inter_height = max(0, y2_i - y1_i)
    inter_area = inter_width * inter_height

    box1_area = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    box2_area = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def main():
    print("Carregando anotações de ground truth (GT)...")
    coco_gt = COCO(ANN_FILE)

    print("Carregando configuração e pesos do modelo Detectron2...")
    cfg = get_cfg()
    try:
        cfg.merge_from_file(CFG_FILE_PATH)
    except FileNotFoundError:
        print(f"ERRO CRÍTICO: Arquivo de configuração Detectron2 não encontrado em {CFG_FILE_PATH}")
        return
    except Exception as e:
        print(f"ERRO CRÍTICO: Falha ao carregar o arquivo de configuração Detectron2: {e}")
        return
        
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = MODEL_SCORE_THRESH # Limiar para o que o modelo retorna

    # Tenta GPU, senão CPU
    try:
        cfg.MODEL.DEVICE = "cuda"
        predictor = DefaultPredictor(cfg)
        print("Inferência na GPU habilitada.")
    except RuntimeError as exc:
        print(f"GPU falhou – mudando para CPU: {exc}")
        cfg.MODEL.DEVICE = "cpu"
        predictor = DefaultPredictor(cfg)
    except Exception as e:
        print(f"ERRO CRÍTICO: Falha ao inicializar o predictor Detectron2: {e}")
        return

    # --- Mapeamento de Classes ---
    model_thing_classes = []
    try:
        # cfg.DATASETS.TEST precisa estar definido no config.yaml e o dataset registrado
        if cfg.DATASETS.TEST and len(cfg.DATASETS.TEST) > 0:
            dataset_test_name = cfg.DATASETS.TEST[0]
            metadata = MetadataCatalog.get(dataset_test_name)
            model_thing_classes = metadata.thing_classes # Nomes das classes que o modelo prevê
            print(f"Classes do modelo (Detectron2 Metadata - {dataset_test_name}): {model_thing_classes}")
        else:
            print("AVISO: cfg.DATASETS.TEST não está definido no config.yaml. Tentando fallback para mapeamento de classes.")
    except Exception as e:
        print(f"AVISO: Falha ao obter 'thing_classes' do MetadataCatalog: {e}. Tentando fallback.")

    coco_categories_gt = coco_gt.loadCats(coco_gt.getCatIds())
    coco_gt_name_to_id = {cat['name']: cat['id'] for cat in coco_categories_gt}
    coco_gt_id_to_name = {cat['id']: cat['name'] for cat in coco_categories_gt}

    model_cls_idx_to_coco_cat_id = {}
    if model_thing_classes: # Se conseguimos os nomes das classes do modelo via Metadata
        for i, model_class_name in enumerate(model_thing_classes):
            if model_class_name in coco_gt_name_to_id:
                model_cls_idx_to_coco_cat_id[i] = coco_gt_name_to_id[model_class_name]
            else:
                print(f"AVISO: Classe do modelo '{model_class_name}' não encontrada nos nomes do COCO GT.")
    else: # Fallback: se não temos os nomes do modelo, assumimos que o número de classes e a ordem correspondem
        model_num_classes_cfg = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        sorted_coco_gt_cats = sorted(coco_categories_gt, key=lambda c: c['id']) # Ordenar por ID do GT
        if model_num_classes_cfg == len(sorted_coco_gt_cats):
            print(f"AVISO: Usando fallback para mapeamento de classes: índice do modelo (0-{model_num_classes_cfg-1}) para ID de categoria COCO GT ordenado.")
            for i in range(model_num_classes_cfg):
                model_cls_idx_to_coco_cat_id[i] = sorted_coco_gt_cats[i]['id']
        else:
            print(f"ERRO CRÍTICO DE MAPEAMENTO DE CLASSES: cfg.MODEL.ROI_HEADS.NUM_CLASSES ({model_num_classes_cfg}) "
                  f"não corresponde ao número de categorias no COCO GT ({len(sorted_coco_gt_cats)}), "
                  "e nomes de classes do modelo não puderam ser obtidos via metadata. Avaliação será imprecisa.")
            # Como último recurso perigoso, poderia usar int(cls)+1 do script original,
            # mas isso é muito propenso a erros se os IDs do COCO GT não forem 1,2,3...
            # return # Melhor parar aqui se o mapeamento for incerto.

    if not model_cls_idx_to_coco_cat_id:
        print("ERRO CRÍTICO: Mapeamento de classes do modelo para COCO GT não pôde ser estabelecido. Abortando.")
        return

    print(f"Mapeamento de índice de classe do modelo para ID COCO GT: {model_cls_idx_to_coco_cat_id}")

    # Classes para Matriz de Confusão (baseado nas classes do GT)
    gt_cat_ids_for_cm = sorted(coco_gt.getCatIds())
    class_names_for_cm_plot = [coco_gt_id_to_name.get(cat_id, f"ID:{cat_id}") for cat_id in gt_cat_ids_for_cm]
    num_classes_cm = len(gt_cat_ids_for_cm)
    coco_id_to_matrix_idx = {cat_id: i for i, cat_id in enumerate(gt_cat_ids_for_cm)}

    results_coco_format = []
    all_predictions_for_plotting_cm = [] # Para nossa CM e exemplos
    total_boxes_predicted = 0

    print("Rodando inferência do Detectron2...")
    for img_info in tqdm(coco_gt.loadImgs(coco_gt.getImgIds()), desc="Detectron2 Inference"):
        img_id = img_info["id"]
        img_path = os.path.join(IMG_DIR, img_info["file_name"])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"AVISO: Imagem não encontrada {img_path}, pulando.")
            continue

        with torch.no_grad():
            outputs = predictor(img_bgr)["instances"]

        instances = outputs.to("cpu")
        total_boxes_predicted += len(instances)

        pred_boxes_tensor = instances.pred_boxes.tensor.numpy()
        pred_scores_tensor = instances.scores.numpy()
        pred_classes_tensor = instances.pred_classes.numpy() # Índices 0 a N-1

        for box, score, model_cls_idx in zip(pred_boxes_tensor, pred_scores_tensor, pred_classes_tensor):
            # Mapear model_cls_idx para o category_id do COCO GT
            target_coco_category_id = model_cls_idx_to_coco_cat_id.get(int(model_cls_idx))
            
            if target_coco_category_id is None: # Classe predita pelo modelo não tem mapeamento para o GT
                # print(f"AVISO: Classe do modelo com índice {model_cls_idx} não tem mapeamento para COCO ID, pulando predição.")
                continue

            x1, y1, x2, y2 = box
            coco_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

            results_coco_format.append({
                "image_id": img_id,
                "category_id": target_coco_category_id,
                "bbox": coco_bbox,
                "score": float(score),
            })
            all_predictions_for_plotting_cm.append({
                'image_id': img_id,
                'category_id': target_coco_category_id,
                'bbox_abs_xyxy': [float(x1), float(y1), float(x2), float(y2)],
                'score': float(score)
            })
        
        if cfg.MODEL.DEVICE == "cuda":
            torch.cuda.empty_cache()


    print(f"Total de caixas delimitadoras preditas: {total_boxes_predicted}")
    if not results_coco_format:
        print("Nenhuma predição válida foi gerada ou mapeada. Verifique configuração, pesos, limiares ou mapeamento de classes.")
        with open(METRICS_FILE, 'w') as f:
            f.write("Nenhuma predição válida para avaliação.\n")
        return

    print(f"Salvando predições em {OUT_PRED_JSON}...")
    with open(OUT_PRED_JSON, "w", encoding="utf-8") as f:
        json.dump(results_coco_format, f)

    print("Executando COCOeval...")
    coco_dt = coco_gt.loadRes(OUT_PRED_JSON)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    print("\nResumo da Avaliação COCO (Detectron2):")
    coco_eval.summarize()

    with open(METRICS_FILE, 'w') as f:
        f.write("Detectron2 COCO Evaluation Metrics:\n")
        metric_names = [
            "AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]", "AP @[ IoU=0.50      | area=   all | maxDets=100 ] (mAP@.50)",
            "AP @[ IoU=0.75      | area=   all | maxDets=100 ]", "AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
            "AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]", "AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
            "AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]", "AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]",
            "AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]", "AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
            "AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]", "AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]"
        ]
        for i_stat, stat_val in enumerate(coco_eval.stats):
            f.write(f"{metric_names[i_stat]}: {stat_val:.4f}\n")
        f.write("\nNota: COCOeval não expõe diretamente P, R, F1 por classe para um único limiar de confiança/IoU.\n")
        f.write("As métricas acima são os AP/AR padrão da avaliação COCO.\n")
    print(f"Métricas salvas em {METRICS_FILE}")


    # --- Matriz de Confusão ---
    if num_classes_cm > 0 and all_predictions_for_plotting_cm:
        print("Calculando Matriz de Confusão...")
        confusion_matrix_values = np.zeros((num_classes_cm, num_classes_cm), dtype=np.int32)
        
        preds_by_img_id = {}
        for pred in all_predictions_for_plotting_cm:
            img_id = pred['image_id']
            if img_id not in preds_by_img_id:
                preds_by_img_id[img_id] = []
            preds_by_img_id[img_id].append(pred)

        for img_info_loop in tqdm(coco_gt.loadImgs(coco_gt.getImgIds()), desc="Processando CM"):
            current_img_id_loop = img_info_loop['id']
            gt_ann_ids = coco_gt.getAnnIds(imgIds=current_img_id_loop, iscrowd=False)
            gts_in_img_loop = coco_gt.loadAnns(gt_ann_ids)
            
            # Filtrar predições apenas por score, já que o modelo já aplicou SCORE_THRESH_TEST
            preds_in_img_loop = sorted(preds_by_img_id.get(current_img_id_loop, []), key=lambda p: p['score'], reverse=True)
            
            gt_used_flags = [False] * len(gts_in_img_loop)

            for pred_item in preds_in_img_loop:
                # Não precisa filtrar por score aqui se MODEL_SCORE_THRESH já foi aplicado
                # e se queremos considerar todas as predições retornadas para a CM.
                # Se quiser um limiar diferente para a CM, aplique aqui:
                # if pred_item['score'] < YOUR_CM_CONF_THRESHOLD:
                # continue

                pred_bbox_xyxy = pred_item['bbox_abs_xyxy']
                pred_coco_cat_id = pred_item['category_id']
                # O pred_coco_cat_id já é o ID do GT COCO, não o índice do modelo
                
                pred_matrix_idx = coco_id_to_matrix_idx.get(pred_coco_cat_id)
                if pred_matrix_idx is None: continue # Classe predita não está no nosso conjunto de CM (improvável se mapeamento ok)

                best_iou_val = 0
                best_gt_idx_match = -1

                for i_gt, gt_ann_item in enumerate(gts_in_img_loop):
                    if gt_used_flags[i_gt]: continue
                    
                    gt_bbox_coco_fmt = gt_ann_item['bbox']
                    gt_bbox_xyxy_fmt = [gt_bbox_coco_fmt[0], gt_bbox_coco_fmt[1], 
                                     gt_bbox_coco_fmt[0] + gt_bbox_coco_fmt[2], 
                                     gt_bbox_coco_fmt[1] + gt_bbox_coco_fmt[3]]
                    
                    iou_val = calculate_iou(pred_bbox_xyxy, gt_bbox_xyxy_fmt)
                    if iou_val > best_iou_val:
                        best_iou_val = iou_val
                        best_gt_idx_match = i_gt
                
                if best_iou_val >= IOU_THR_MATCHING:
                    matched_gt_ann = gts_in_img_loop[best_gt_idx_match]
                    true_coco_cat_id = matched_gt_ann['category_id']
                    true_matrix_idx = coco_id_to_matrix_idx.get(true_coco_cat_id)

                    if true_matrix_idx is not None:
                        confusion_matrix_values[true_matrix_idx, pred_matrix_idx] += 1
                        gt_used_flags[best_gt_idx_match] = True
        
        plt.figure(figsize=(max(8, num_classes_cm * 0.9), max(6, num_classes_cm * 0.8)))
        sns.heatmap(confusion_matrix_values, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names_for_cm_plot, yticklabels=class_names_for_cm_plot)
        plt.ylabel('Classe Verdadeira (GT)')
        plt.xlabel('Classe Predita')
        plt.title(f'Matriz de Confusão (Detectron2, IoU>{IOU_THR_MATCHING}, Score>{MODEL_SCORE_THRESH})')
        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_FILE)
        plt.close()
        print(f"Matriz de confusão salva em {CONFUSION_MATRIX_FILE}")
    else:
        print("Pulando geração da matriz de confusão: sem classes válidas ou sem predições.")


    # --- Plotar Exemplos de Imagens ---
    if all_predictions_for_plotting_cm:
        print("Plotando imagens de exemplo...")
        img_ids_all_gt = coco_gt.getImgIds()
        selected_img_ids = np.random.choice(img_ids_all_gt, 
                                            size=min(NUM_EXAMPLE_IMAGES, len(img_ids_all_gt)), 
                                            replace=False)
        
        color_gt_draw = (0, 255, 0)
        color_pred_draw = (0, 0, 255)

        for i_ex, current_img_id_ex in enumerate(tqdm(selected_img_ids, desc="Plotando exemplos")):
            img_info_ex = coco_gt.loadImgs(current_img_id_ex)[0]
            img_path_ex = os.path.join(IMG_DIR, img_info_ex['file_name'])
            image_to_draw = cv2.imread(img_path_ex)
            if image_to_draw is None: continue

            # Desenhar GT
            gt_ann_ids_ex = coco_gt.getAnnIds(imgIds=current_img_id_ex, iscrowd=False)
            gts_in_img_ex = coco_gt.loadAnns(gt_ann_ids_ex)
            for ann_ex in gts_in_img_ex:
                x, y, w, h = [int(v) for v in ann_ex['bbox']]
                cat_id_ex = ann_ex['category_id']
                cat_name_ex = coco_gt_id_to_name.get(cat_id_ex, f"ID:{cat_id_ex}")
                cv2.rectangle(image_to_draw, (x,y), (x+w, y+h), color_gt_draw, 2)
                cv2.putText(image_to_draw, f"GT: {cat_name_ex}", (x, y - 10 if y > 20 else y + h + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_gt_draw, 2)
            
            # Desenhar Predições
            current_img_preds_ex = [p for p in all_predictions_for_plotting_cm if p['image_id'] == current_img_id_ex]
            for pred_ex in current_img_preds_ex:
                # Considerar apenas predições acima de um limiar para visualização se MODEL_SCORE_THRESH for muito baixo
                # if pred_ex['score'] < VISUALIZATION_SCORE_THRESHOLD: 
                # continue
                x1p, y1p, x2p, y2p = [int(c) for c in pred_ex['bbox_abs_xyxy']]
                pred_cat_id_ex = pred_ex['category_id']
                score_ex = pred_ex['score']
                pred_cat_name_ex = coco_gt_id_to_name.get(pred_cat_id_ex, f"ID:{pred_cat_id_ex}")

                cv2.rectangle(image_to_draw, (x1p, y1p), (x2p, y2p), color_pred_draw, 2)
                cv2.putText(image_to_draw, f"P: {pred_cat_name_ex} ({score_ex:.2f})",
                            (x1p, y1p - 25 if y1p > 40 else y1p + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_pred_draw, 2)
            
            out_img_path_ex = os.path.join(OUTPUT_DIR, f"example_detectron2_{i_ex+1}_{img_info_ex['file_name']}")
            cv2.imwrite(out_img_path_ex, image_to_draw)
        print(f"Imagens de exemplo salvas em {OUTPUT_DIR}")
    else:
        print("Pulando plotagem de imagens de exemplo: sem predições.")

    print(f"\nAvaliação e geração de plots (Detectron2) concluídos. Resultados em: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
