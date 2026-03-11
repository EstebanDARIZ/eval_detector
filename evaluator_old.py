import os
import argparse
import cv2 
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt

from config import PATH_DATA_TEST 

"""
How to use : 
python evaluator.py --model-name EfficientDet --path-pred /home/esteban-dreau-darizcuren/doctorat/code/detector/sam3/output/dataset_test_2.0_pred

python evaluator.py --model-name EfficientDet --path-pred /home/esteban-dreau-darizcuren/doctorat/code/detector/sam3/output/dataset_test_2.0_pred --iou-tab .1 .2 .3 .4 .5 .6 .7 .8 .9
"""

def xywnh2xyxyc_gt(label_path, img_w, img_h):
    """
    Retourne une liste de GT boxes au format [x1, y1, x2, y2, class_id]
    """
    boxes = []

    if not os.path.exists(label_path):
        # print(f"No GT in {label_path}")
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            cls, xc, yc, w, h = map(float, line.strip().split())

            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h

            boxes.append([x1, y1, x2, y2, int(cls)])

    return boxes

def xywnh2xyxyc_pred(label_path, img_w, img_h):
    """
    Retourne une liste de GT boxes au format [x1, y1, x2, y2, class_id, conf]
    """
    boxes = []

    if not os.path.exists(label_path):
        # print(f"No detection in {label_path}")
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            cls, xc, yc, w, h, conf = map(float, line.strip().split())

            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h

            boxes.append([x1, y1, x2, y2, int(cls), conf])

    return boxes

def compute_iou(boxA, boxB):
    """
    boxA, boxB : [x, y, w, h]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union = boxAArea + boxBArea - interArea

    if union == 0:
        return 0.0

    return interArea / union

def draw_detections(img, boxes):
    # img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for box in boxes:
        x1, y1, x2, y2, cls, conf = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)

    return img

def print_res(tp, fp, gt, total_pred, all_ious, iou):
    fn = gt - total_pred
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0
    print(f"# For iou : {iou} #")
    print(f"TP : {tp}")
    print(f"FP : {fp}")
    print(f"FN : {fn}")
    print(f"Precision : {precision}")
    print(f"Recall : {recall}")
    print(f"Mean iou : {mean_iou}")

def compute_precision_recall_old(results, iou, total_gt):
    tp_add = []
    fp_add = []
    fp_count = 0
    tp_count  =0
    results.sort(key=lambda x: x["conf"], reverse=True)    # sort of prediction by confidence score 
    for result in results:
        if result["tp"] == 1:
            tp_count +=1
        else:
            fp_count +=1
        tp_add.append(tp_count)
        fp_add.append(fp_count)
    
    tp_add = np.array(tp_add)
    fp_add = np.array(fp_add)

    recall = tp_add / total_gt
    precision = tp_add / (tp_add+fp_add + 1e-6)
    
    return results, recall, precision

def compute_precision_recall(preds, total_gt):
    tp_add = []
    fp_add = []
    fp_count = 0
    tp_count  =0
    preds.sort(key=lambda x : x['conf'], reverse= True)
    for result in preds:
        if result["tp"] == 1:
            tp_count +=1
        else:
            fp_count +=1
        tp_add.append(tp_count)
        fp_add.append(fp_count)
    
    tp_add = np.array(tp_add)
    fp_add = np.array(fp_add)

    recall = tp_add / 85
    precision = tp_add / (tp_add+fp_add + 1e-6)
    
    return recall, precision

def plot_pr_curves(results, total_gt):
    """
    """
    plt.figure(figsize=(10, 7))
    
    for iou, preds in results.items():
        # On utilise ta fonction pour récupérer les points
        recalls, precisions = compute_precision_recall(preds, total_gt)
        
        # On ajoute (0, 1) et (1, 0) pour que la courbe soit bien cadrée
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([1.0], precisions, [0.0]))
        
        # Lissage (interpolation) pour le look "standard"
        precisions = np.maximum.accumulate(precisions[::-1])[::-1]

        recalls = np.array(recalls)
        precisions = np.array(precisions)

        indices = np.argsort(recalls)
        recalls = recalls[indices]
        precisions = precisions[indices]
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        plt.plot(recalls, precisions, label=f'IoU {iou}')
        # plt.fill_between(recalls, precisions, alpha=0.1) # Optionnel pour voir l'aire


    plt.xlabel('Recall (Rappel)')
    plt.ylabel('Precision (Précision)')
    plt.title('Courbe Precision-Recall par seuil IoU')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.05])
    plt.legend()
    plt.grid(True)
    plt.show() # Ou plt.savefig("ma_courbe.png")

def ap(recalls, precisions):
    """
    Calcule l'AP à partir de listes de Rappel et Précision (déjà interpolées).
    """
    # 1. On s'assure d'avoir les bornes (0,0) au début du Rappel si nécessaire
    # Le rappel doit commencer à 0 pour bien calculer la première largeur
    mrec = np.concatenate(([0.0], recalls))
    mpre = np.concatenate(([precisions[0]], precisions))

    # 2. On calcule les "largeurs" des rectangles (différences entre rappels successifs)
    # np.diff([0.0, 0.1, 0.3]) -> [0.1, 0.2]
    widths = np.diff(mrec)

    # 3. L'AP est la somme des surfaces (Largeur * Hauteur)
    # On multiplie chaque largeur par la précision correspondante
    ap = np.sum(widths * mpre[1:])
    
    return ap

def compute_mAP(results, total_gt):
    all_aps = []
    for iou, preds in results.items():
        r, p = compute_precision_recall(preds, total_gt)

        # On ajoute (0, 1) et (1, 0) pour que la courbe soit bien cadrée
        r = np.concatenate(([0.0], r, [1.0]))
        p = np.concatenate(([1.0], p, [0.0]))
        
        # Lissage (interpolation) pour le look "standard"
        p = np.maximum.accumulate(p[::-1])[::-1]

        r = np.array(r)
        p = np.array(p)

        indices = np.argsort(r)
        r = r[indices]
        p = p[indices]

        # 3. Calcul de l'AP pour cet IoU précis
        current_ap = ap(r, p)
        all_aps.append(current_ap)

        print(f"AP à IoU {iou}: {current_ap:.4f}")

    # 4. Le mAP final
    mAP = np.mean(all_aps)
    print(f"\n--- mAP Global: {mAP:.4f} ---")


def main():
    print("Loading ...")
    parser = argparse.ArgumentParser(description="Evaluation pipeline : IoU, mAP and F1-score")
    parser.add_argument("--dataset-test", type=str, default=PATH_DATA_TEST, help="Path to the dataset_test_2.0. by default")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model under evaluation")
    parser.add_argument("--pred-format", type=str, default="xywhn")
    parser.add_argument("--path-pred", type = str, required=True, help="Path to the predictions of the model")
    parser.add_argument("--conf-thresh", type=float, default=0.5)
    parser.add_argument("--iou-tab", default=[0.25, 0.5, 0.75], type=float, nargs="+", help="Tab of iou threshold to compute AP and mAP")
    parser.add_argument("--classes", default=[0, 1, 2, 3, 4,], type=int, nargs="+", help="Classes name of the predictions. SHould be integer"  )
    parser.add_argument("--show-boxes", type=bool, default=False)
    parser.add_argument("--show-PRC", type=bool, default=False) 
    parser.add_argument("--show-pr", default=False, type=bool)

    args = parser.parse_args()

    path_test_folder = args.dataset_test
    path_pred_folder = args.path_pred
    path_test_labels = os.path.join(path_test_folder, "labels")
    path_test_images = os.path.join(path_test_folder, "images")

    pred_format = args.pred_format

    total_gt = 0
    total_pred = 0
    all_ious = {iou: [] for iou in args.iou_tab}
    tp_dict = {iou: 0 for iou in args.iou_tab}
    fp_dict = {iou: 0 for iou in args.iou_tab}
    results = {iou: [] for iou in args.iou_tab}



    for img_name in os.listdir(path_test_images):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            print(f"{img_name} is not an image  ##############################################################")
            continue
        label_name = os.path.splitext(img_name)[0]+".txt"
        gt_path = os.path.join(path_test_labels, label_name)
        pred_path = os.path.join(path_pred_folder, label_name)

        img = cv2.imread(os.path.join(path_test_images, img_name))
        img_h, img_w, _ = img.shape

        ## If pred are in xywhn format 
        if pred_format.lower() == "xywhn":
            pred_boxes = xywnh2xyxyc_pred(pred_path, img_w, img_h)
            pred_boxes.sort(key=lambda x: x[5], reverse=True)    # sort of prediction by confidence score 

        gt_boxes = xywnh2xyxyc_gt(gt_path, img_w, img_h)
        total_gt += len(gt_boxes)

        matched_gt = {iou : set() for iou in args.iou_tab}

        # -----------------------------
        # Matching predictions ↔ GT
        # -----------------------------
        for pred in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            if not pred[5]>= args.conf_thresh:
                    continue
            total_pred +=1
            for i, gt in enumerate(gt_boxes):
                if pred[4] != gt[4]:  # class mismatch
                    continue

                iou = compute_iou(pred[:4], gt[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            for iou_thresh in args.iou_tab:
                if best_iou >= iou_thresh and best_gt_idx not in matched_gt[iou_thresh] : 
                    tp_dict[iou_thresh] += 1
                    all_ious[iou_thresh].append(best_iou)
                    matched_gt[iou_thresh].add(best_gt_idx)
                    results[iou_thresh].append({"conf": pred[5], "tp": 1})
                else:
                    fp_dict[iou_thresh] += 1
                    results[iou_thresh].append({"conf": pred[5], "tp": 0})



        # -----------------------------
        # Optional image saving
        # -----------------------------
        if args.show_boxes:
            img_res = draw_detections(img, pred_boxes)
            cv2.imshow("Prediction", img_res)
            cv2.waitKey(0)
        
        
    
    # -----------------------------
    # Final metrics
    # -----------------------------
    if args.show_pr: 
        print("\n================= RESULTS =================")
        print(f"Total GT : {total_gt}")
        for iou in args.iou_tab:
            print_res(tp_dict[iou], fp_dict[iou], total_gt, total_pred, all_ious[iou], iou)
        print("===========================================\n")  
    
    compute_mAP(results, total_gt)
    if args.show_PRC:
        plot_pr_curves(results, total_gt)

if __name__ == "__main__":
    main()
