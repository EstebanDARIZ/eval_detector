import os
import argparse
import cv2 
from PIL import Image 
import numpy as np

from config import PATH_DATA_TEST 

"""
How to use : 
python evaluator.py \\
    --model-name EfficientDet \\ 
    --path-pred /home/esteban-dreau-darizcuren/doctorat/code/detector/sam3/output/dataset_test_2.0_pred
"""

def xywnh2xyxyc(label_path, img_w, img_h):
    """
    Retourne une liste de GT boxes au format [x1, y1, x2, y2, class_id]
    """
    boxes = []

    if not os.path.exists(label_path):
        print(f"No detection in {label_path}")
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


def match_predictions_to_gt(pred_boxes, gt_boxes, iou_thresh=0.5):
    matches = []

    for pb in pred_boxes:
        best_iou = 0
        best_gt = None

        for gt in gt_boxes:
            if pb[4] != gt[4]:  # classe différente
                continue

            iou = compute_iou(pb[:4], gt[:4])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        matches.append({
            "pred": pb,
            "best_iou": best_iou,
            "is_tp": best_iou >= iou_thresh
        })

    return matches

def draw_detections(img, boxes):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for box in boxes:
        x1, y1, x2, y2, cls = box
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img_cv



def main():
    print("Pipeline start")
    parser = argparse.ArgumentParser(description="Evaluation pipeline : IoU, mAP and F1-score")
    parser.add_argument("--dataset-test", type=str, default=PATH_DATA_TEST, help="Path to the dataset_test_2.0. by default")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model under evaluation")
    parser.add_argument("--path-pred", type = str, required=True, help="Path to the predictions of the model")
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--show", type=bool, default=False)

    args = parser.parse_args()

    path_test_folder = args.dataset_test
    path_pred_folder = args.path_pred
    path_test_labels = os.path.join(path_test_folder, "labels")
    path_test_images = os.path.join(path_test_folder, "images")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious = []

    for img_name in os.listdir(path_test_images):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        print("###########NEW IMAGE##############")

        img = cv2.imread(os.path.join(path_test_images, img_name))
        img_h, img_w, _ = img.shape
        label_name = os.path.splitext(img_name)[0]+".txt"
        gt_path = os.path.join(path_test_labels, label_name)
        pred_path = os.path.join(path_pred_folder, label_name)

        pred_boxes = xywnh2xyxyc(pred_path, img_w, img_h)
        print(f"Pred boxes : {pred_boxes}")
        gt_boxes = xywnh2xyxyc(gt_path, img_w, img_h)
        print(f"Gt boxes : {gt_boxes}")

        matched_gt = set()

        # -----------------------------
        # Matching predictions ↔ GT
        # -----------------------------
        for pb in pred_boxes:
            best_iou = 0
            best_gt_idx = -1

            for i, gt in enumerate(gt_boxes):
                if pb[4] != gt[4]:  # class mismatch
                    continue

                iou = compute_iou(pb[:4], gt[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= args.iou_thresh:
                total_tp += 1
                all_ious.append(best_iou)
                matched_gt.add(best_gt_idx)
            else:
                total_fp += 1

        # False negatives
        total_fn += len(gt_boxes) - len(matched_gt)

        # -----------------------------
        # Optional image saving
        # -----------------------------
        if args.show:
            result, _ = draw_detections(img, pred_boxes)
        #     out_path = os.path.join(args.output_dir, img_name)
        #     cv2.imwrite(out_path, result)

    # -----------------------------
    # Final metrics
    # -----------------------------
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0

    print("\n================= RESULTS =================")
    print(f"TP : {total_tp}")
    print(f"FP : {total_fp}")
    print(f"FN : {total_fn}")
    # print(f"Precision : {precision:.4f}")
    # print(f"Recall    : {recall:.4f}")
    print(f"Mean IoU  : {mean_iou:.4f}")
    print("===========================================\n")





        


    # for txt_file in os.listdir(path_test_labels):
    #     if not txt_file.lower().endswith(".txt"):
    #         continue
    #     path_gt = os.path.join(path_test_labels, txt_file)

    #     with open(path_gt, "r", encoding="utf-8") as gts:
    #         print("##########NEW IMAGE########")
    #         for gt in gts:
    #             gt = gt.strip().split()
    #             if len(gt) == 5:
    #                 cls, x_c, y_c, h, w  = gt
                    
    #                 print(f"GT : {cls}, {x_c}, {y_c}, {h}, {w}")

    #     path_pred = os.path.join(path_pred_folder, txt_file)
    #     if not os.path.exists(path_pred):
    #         print(f"The file {path_pred} does not exist.")
    #         continue
    #     with open(path_pred, "r", encoding="utf-8") as preds:
    #         print("Pred")
    #         for pred in preds:
    #             pred = pred.strip().split()
    #             if len(pred) == 5:
    #                 cls_p, x_c_p, y_c_p, h_p, w_p = pred
    #                 cls_p = int(cls_p) + 1
    #             print(f"Predictions : {cls_p}, {x_c_p}, {y_c_p}, {h_p}, {w_p}")
                                
        



if __name__ == "__main__":
    main()
