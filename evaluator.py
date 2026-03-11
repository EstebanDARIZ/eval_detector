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

def compute_iou(boxA, boxB, img):
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

def print_pr(tp, fp, gt, total_pred, all_ious, iou):
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

    recall = tp_add / total_gt
    precision = tp_add / (tp_add+fp_add + 1e-6)
    
    return recall, precision


def ap(recalls, precisions):
    """
    Compute Average Precision (AP) using the trapezoidal rule.
    This function assumes that recalls and precisions are sorted by recall in ascending order.
    """
    mrec = np.concatenate(([0.0], recalls))
    mpre = np.concatenate(([precisions[0]], precisions))
    widths = np.diff(mrec)
    ap = np.sum(widths * mpre[1:])
    
    return ap

def compute_mAP(results, all_aps, total_gt, classes):
    for iou, preds in results.items():
        for cls in classes:
            rs, ps = compute_precision_recall(preds[cls], total_gt[cls])
            if len(rs) ==0 or np.all(ps == 0):
                r = np.array([0.0, 1.0])
                p = np.array([0.0, 0.0])
                current_ap = 0.0                
            else:
                # Add (0, 1) and (1, 0) to ensure the curve is well framed (square)
                r = np.concatenate(([0.0], rs, [1.0]))
                p = np.concatenate(([1.0], ps, [0.0]))
                # Smoothing (interpolation) for the "standard" look (monotone decreasing precision, escalier step) 
                p = np.maximum.accumulate(p[::-1])[::-1] # [::-1] reverse the array, np.maximum.accumulate computes the cumulative maximum, then we reverse back to get the interpolated precision values
                r = np.array(r)
                p = np.array(p)
                indices = np.argsort(r) # sort by recall to ensure the curve is well formed (increasing recall)
                r = r[indices]
                p = p[indices]

                # 3. Calcul de l'AP pour cet IoU précis

                current_ap = ap(r, p)
            all_aps[iou][cls].append(current_ap)
            plt.plot(r, p, label=f'Cls {cls}, AP {current_ap:.4f}')
        plt.xlabel('Recall (Rappel)')
        plt.ylabel('Precision (Précision)')
        plt.title(f"Courbe Precision-Recall @ IoU : {iou}")
        plt.xlim([0, 1.0])
        plt.ylim([0, 1.05])
        plt.legend()
        plt.grid(True)
        plt.show()
    return all_aps

def main():
    print("Loading ...")
    parser = argparse.ArgumentParser(description="Evaluation pipeline : IoU, mAP and F1-score")
    parser.add_argument("--dataset-test", type=str, default=PATH_DATA_TEST, help="Path to the dataset_test_2.0. by default")
    parser.add_argument("--pred-format", type=str, default="xywhn")
    parser.add_argument("--path-pred", type = str, required=True, help="Path to the predictions of the model")
    parser.add_argument("--conf-thresh", type=float, default=0.5)
    parser.add_argument("--iou-tab", default=[0.25, 0.5, 0.75], type=float, nargs="+", help="Tab of iou threshold to compute AP and mAP")
    parser.add_argument("--classes", default=[0, 1, 2, 3, 4,], type=int, nargs="+", help="Classes name of the predictions. SHould be integer"  )
    parser.add_argument("--show-boxes", type=bool, default=False)
    parser.add_argument("--show-pr", default=False, type=bool)

    args = parser.parse_args()

    path_test_folder = args.dataset_test
    path_pred_folder = args.path_pred
    path_test_labels = os.path.join(path_test_folder, "labels")
    path_test_images = os.path.join(path_test_folder, "images")

    pred_format = args.pred_format

    total_pred = 0
    total_gt = 0
    gt_cls = {cls: 0 for cls in args.classes}
    all_ious = {iou: [] for iou in args.iou_tab}
    tp_dict = {iou: 0 for iou in args.iou_tab}
    fp_dict = {iou: 0 for iou in args.iou_tab}
    results = {iou: {cls: [] for cls in args.classes} for iou in args.iou_tab}



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

        for gt in gt_boxes:
            gt_cls[gt[4]] += 1
            total_gt +=1

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

                iou = compute_iou(pred[:4], gt[:4], img)
                

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            for iou_thresh in args.iou_tab:
                if best_iou >= iou_thresh and best_gt_idx not in matched_gt[iou_thresh] : 
                    tp_dict[iou_thresh] += 1
                    all_ious[iou_thresh].append(best_iou)
                    matched_gt[iou_thresh].add(best_gt_idx)
                    results[iou_thresh][pred[4]].append({"conf": pred[5], "tp": 1})
                else:
                    fp_dict[iou_thresh] += 1
                    results[iou_thresh][pred[4]].append({"conf": pred[5], "tp": 0})



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
    print("\n================= RESULTS =================")
    # print(f"Total GT : {gt_cls}")
    if args.show_pr: 
        print("-" * 70)
        print(f"{'IoU':<6} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Prec.':<8} | {'Rec.':<8} | {'mIoU':<8}")
        print("-" * 70)
        for iou in args.iou_tab:
            tp = tp_dict[iou]
            fp = fp_dict[iou]
            fn = total_gt - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / total_gt if total_gt > 0 else 0
            val_miou = all_ious[iou]
            if isinstance(val_miou, list) and len(val_miou) > 0:
                miou_display = np.mean(val_miou)
            elif isinstance(val_miou, (int, float, np.float64)):
                miou_display = val_miou
            else:
                miou_display = 0.0
            print(f"{iou:<6} | {tp:<5} | {fp:<5} | {fn:<5} | {prec:<8.4f} | {rec:<8.4f} | {miou_display:<8.4f}")
            print("-" * 70)
        print(" ")
    
    all_aps = {iou: {cls:[] for cls in args.classes} for iou in args.iou_tab}
    all_aps = compute_mAP(results, all_aps, gt_cls, args.classes)
    
    
    for iou in args.iou_tab:
        ap_iou = 0
        for cls in args.classes:
            ap_val = all_aps[iou][cls]
            if isinstance(ap_val, (list, np.ndarray)) and len(ap_val) > 0:
                ap_val = ap_val[0]
                ap_iou += ap_val
            print(f"AP for classe {cls} @ IoU {iou} : {ap_val:.4f}")
        if args.classes != 0: 
            ap_iou = ap_iou / len(args.classes) 
            print(f"mAp @ IoU {iou} : {ap_iou:.4f}")
    
    print("===========================================\n")

if __name__ == "__main__":
    main()
