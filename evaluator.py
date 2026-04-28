import os
import argparse
import cv2 
from PIL import Image 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from config import PATH_DATA_TEST 

"""
How to use : 
python evaluator.py --path-pred /path/to/predictions --show-curves False
python evaluator.py --path-pred /path/to/predictions --iou-tab .1 .2 .3 .4 .5 .6 .7 .8 .9
"""

# ─────────────────────────────────────────────
#  Parsing / IO helpers
# ─────────────────────────────────────────────

def xywnh2xyxyc_gt(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
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
    boxes = []
    if not os.path.exists(label_path):
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

def compute_iou(boxA, boxB, img=None):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0

def draw_detections(img, boxes):
    for box in boxes:
        x1, y1, x2, y2, cls, conf = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
    return img


# ─────────────────────────────────────────────
#  Metrics computation
# ─────────────────────────────────────────────

def compute_precision_recall(preds, total_gt):
    preds.sort(key=lambda x: x['conf'], reverse=True)
    tp_count = fp_count = 0
    tp_add, fp_add = [], []
    for result in preds:
        if result["tp"] == 1:
            tp_count += 1
        else:
            fp_count += 1
        tp_add.append(tp_count)
        fp_add.append(fp_count)
    tp_add = np.array(tp_add)
    fp_add = np.array(fp_add)
    recall    = tp_add / total_gt
    precision = tp_add / (tp_add + fp_add + 1e-6)
    return recall, precision

def ap_score(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls))
    mpre = np.concatenate(([precisions[0]], precisions))
    widths = np.diff(mrec)
    return np.sum(widths * mpre[1:])

def compute_mAP(results, total_gt, classes, iou_tab, show, path_pred_folder):
    all_aps = {iou: {cls: [] for cls in classes} for iou in iou_tab}
    for iou, preds in results.items():
        plt.figure()
        for cls in classes:
            rs, ps = compute_precision_recall(preds[cls], total_gt[cls])
            if len(rs) == 0 or np.all(ps == 0):
                r, p, current_ap = np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0.0
            else:
                r = np.concatenate(([0.0], rs, [1.0]))
                p = np.concatenate(([1.0], ps, [0.0]))
                p = np.maximum.accumulate(p[::-1])[::-1]
                indices = np.argsort(r)
                r, p = r[indices], p[indices]
                current_ap = ap_score(r, p)
            all_aps[iou][cls].append(current_ap)
            plt.plot(r, p, label=f'Cls {cls}, AP {current_ap:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f"Courbe Precision-Recall @ IoU : {iou}")
        plt.xlim([0, 1.0]); plt.ylim([0, 1.05])
        plt.legend(); plt.grid(True)
        plot_path = os.path.join(path_pred_folder, f"PR_curve_iou{iou}.png")
        os.makedirs(path_pred_folder, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé : {plot_path}")
        if show:
            plt.show()
        plt.close()
    return all_aps

def compute_f1(results, gt_cls, classes, iou_tab):
    f1 = {iou: {cls: 0 for cls in classes} for iou in iou_tab}
    for iou in iou_tab:
        for cls in classes:
            tp = fp = 0
            for pred in results[iou][cls]:
                if pred['tp'] == 1: tp += 1
                else: fp += 1
            fn = gt_cls[cls] - tp
            precision = tp / (tp + fp + 1e-6)
            recall    = tp / (tp + fn + 1e-6)
            f1[iou][cls] = 2 * recall * precision / (recall + precision + 1e-6)
    return f1


# ─────────────────────────────────────────────
#  Reporting helpers
# ─────────────────────────────────────────────

SEP  = "─" * 80
SEP2 = "═" * 80
SEP3 = "━" * 80

def _row(*cols, widths):
    parts = [str(c).ljust(w) for c, w in zip(cols, widths)]
    return "│ " + " │ ".join(parts) + " │"

def _header(*cols, widths):
    row  = _row(*cols, widths=widths)
    bar  = "├─" + "─┼─".join("─" * w for w in widths) + "─┤"
    top  = "┌─" + "─┬─".join("─" * w for w in widths) + "─┐"
    return top + "\n" + row + "\n" + bar

def _footer(widths):
    return "└─" + "─┴─".join("─" * w for w in widths) + "─┘"

def _divider(widths):
    return "├─" + "─┼─".join("─" * w for w in widths) + "─┤"


def format_global_table(tp_dict, fp_dict, total_gt, all_ious, iou_tab):
    """Tableau global : IoU | TP | FP | FN | Prec | Rec | mIoU"""
    W = [6, 5, 5, 5, 8, 8, 8]
    lines = []
    lines.append(_header("IoU", "TP", "FP", "FN", "Prec.", "Rec.", "mIoU", widths=W))
    for i, iou in enumerate(iou_tab):
        tp = tp_dict[iou]; fp = fp_dict[iou]; fn = total_gt - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / total_gt  if total_gt > 0  else 0
        val  = all_ious[iou]
        miou = np.mean(val) if isinstance(val, list) and len(val) > 0 else float(val) if isinstance(val, (int, float, np.float64)) else 0.0
        lines.append(_row(f"{iou}", tp, fp, fn, f"{prec:.4f}", f"{rec:.4f}", f"{miou:.4f}", widths=W))
        if i < len(iou_tab) - 1:
            lines.append(_divider(W))
    lines.append(_footer(W))
    return "\n".join(lines)


def format_map_f1_table(all_aps, f1_cls, gt_cls, total_gt, classes, iou_tab):
    """Tableau mAP / F1 global"""
    W = [6, 10, 14, 14]
    lines = []
    lines.append(_header("IoU", "mAP", "F1 (Macro)", "F1 (Pondéré)", widths=W))
    for i, iou in enumerate(iou_tab):
        ap_iou = f1_macro = f1_pond = 0
        for cls in classes:
            ap_val = all_aps[iou][cls]
            f1_val = f1_cls[iou][cls]
            f1_macro += f1_val
            f1_pond  += f1_val * gt_cls[cls]
            if isinstance(ap_val, (list, np.ndarray)) and len(ap_val) > 0:
                ap_iou += ap_val[0]
        if classes:
            ap_iou   /= len(classes)
            f1_macro /= len(classes)
            f1_pond  /= total_gt
        lines.append(_row(f"{iou}", f"{ap_iou:.4f}", f"{f1_macro:.4f}", f"{f1_pond:.4f}", widths=W))
        if i < len(iou_tab) - 1:
            lines.append(_divider(W))
    lines.append(_footer(W))
    return "\n".join(lines)


def format_per_class_table(results, gt_cls, all_aps, f1_cls, total_gt, classes, iou_tab, all_ious_cls):
    """Tableau détaillé par classe"""
    W = [6, 5, 5, 5, 8, 8, 8, 8, 14, 14]
    lines = []
    lines.append(_header(
        "IoU", "TP", "FP", "FN", "Prec.", "Rec.", "mIoU", "AP", "F1 (Macro)", "F1 (Pondéré)",
        widths=W
    ))

    for cls in classes:
        gt_count = gt_cls[cls]
        # séparateur de classe
        cls_label = f" Classe {cls}  (GT : {gt_count} objets) "
        pad = (sum(W) + 3 * (len(W) - 1) + 4 - len(cls_label)) // 2
        lines.append("│" + " " * pad + cls_label + " " * (sum(W) + 3 * (len(W) - 1) + 4 - pad - len(cls_label) - 1) + "│")
        lines.append(_divider(W))

        for i, iou in enumerate(iou_tab):
            tp = fp = 0
            for pred in results[iou][cls]:
                if pred['tp'] == 1: tp += 1
                else: fp += 1
            fn   = gt_count - tp
            prec = tp / (tp + fp + 1e-6)
            rec  = tp / (tp + fn + 1e-6)
            # mIoU par classe
            iou_vals = all_ious_cls[iou][cls]
            miou = np.mean(iou_vals) if len(iou_vals) > 0 else 0.0
            ap_val = all_aps[iou][cls]
            ap_v   = ap_val[0] if isinstance(ap_val, (list, np.ndarray)) and len(ap_val) > 0 else float(ap_val)
            f1_v   = f1_cls[iou][cls]
            # F1 pondéré par classe = identique au F1 (une seule classe)
            lines.append(_row(
                f"{iou}", tp, fp, fn,
                f"{prec:.4f}", f"{rec:.4f}", f"{miou:.4f}", f"{ap_v:.4f}",
                f"{f1_v:.4f}", f"{f1_v:.4f}",
                widths=W
            ))
            if i < len(iou_tab) - 1:
                lines.append(_divider(W))

        lines.append(_divider(W))  # séparateur entre classes

    lines[-1] = _footer(W)  # remplace le dernier divider par un footer
    return "\n".join(lines)


def parse_path_info(path_pred):
    """
    Extrait modèle, dataset entraînement et dataset test depuis :
      .../results/detector/{model}/{train_dataset}/res_{test_dataset}/...
    """
    parts = os.path.normpath(path_pred).split(os.sep)
    try:
        idx = parts.index("detector")
        model         = parts[idx + 1]
        train_dataset = parts[idx + 2]
        res_part      = parts[idx + 3]
        test_dataset  = res_part[len("res_"):] if res_part.startswith("res_") else res_part
        return model, train_dataset, test_dataset
    except (ValueError, IndexError):
        return None, None, None


def write_results(path, tp_dict, fp_dict, total_gt, all_ious, all_ious_cls,
                  all_aps, f1_cls, gt_cls, total_pred, classes, iou_tab,
                  path_pred, path_test, conf_thresh):
    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    model, train_ds, test_ds = parse_path_info(path_pred)

    with open(path, "w", encoding="utf-8") as f:
        def w(s=""): f.write(s + "\n")

        w(SEP2)
        w(f"  RAPPORT D'ÉVALUATION")
        w(f"  Généré le : {now}")
        w(SEP2)
        w(f"  Modèle               : {model    or 'N/A'}")
        w(f"  Dataset entraînement : {train_ds  or 'N/A'}")
        w(f"  Dataset test         : {test_ds   or 'N/A'}")
        w(SEP2)
        w(f"  Chemin prédictions   : {path_pred}")
        w(f"  Chemin dataset test  : {path_test}")
        w(f"  Seuil conf.          : {conf_thresh}")
        w(f"  Classes              : {classes}")
        w(f"  Seuils IoU           : {iou_tab}")
        w(SEP2)
        w()

        # ── Résumé ──────────────────────────────────────────────
        w(SEP3)
        w("  RÉSUMÉ")
        w(SEP3)
        w(f"  Total GT          : {total_gt}")
        w(f"  Total prédictions : {total_pred}")
        w(f"  Répartition GT par classe :")
        for cls in classes:
            pct = gt_cls[cls] / total_gt * 100 if total_gt > 0 else 0
            w(f"    Classe {cls} : {gt_cls[cls]:>5} objets  ({pct:.1f}%)")
        w()

        # ── Résultats globaux ────────────────────────────────────
        w(SEP3)
        w("  RÉSULTATS GLOBAUX")
        w(SEP3)
        w()
        w("  Détection (TP / FP / FN / Précision / Rappel / mIoU)")
        w()
        w(format_global_table(tp_dict, fp_dict, total_gt, all_ious, iou_tab))
        w()
        w("  mAP et F1")
        w()
        w(format_map_f1_table(all_aps, f1_cls, gt_cls, total_gt, classes, iou_tab))
        w()

        # ── Résultats par classe ─────────────────────────────────
        w(SEP3)
        w("  RÉSULTATS PAR CLASSE")
        w(SEP3)
        w()
        w(format_per_class_table(
            results_global, gt_cls, all_aps, f1_cls, total_gt, classes, iou_tab, all_ious_cls
        ))
        w()
        w(SEP2)
        w("  FIN DU RAPPORT")
        w(SEP2)

    print(f"\nRapport sauvegardé : {path}")


# ─────────────────────────────────────────────
#  Legacy print (console)
# ─────────────────────────────────────────────

def print_pr(tp_dict, fp_dict, total_gt, all_ious, iou_tab, save_res_path=None):
    print(format_global_table(tp_dict, fp_dict, total_gt, all_ious, iou_tab))

def print_mAp_F1(all_aps, f1_cls, gt_cls, total_gt, classes, iou_tab, save_res_path=None):
    print(format_map_f1_table(all_aps, f1_cls, gt_cls, total_gt, classes, iou_tab))


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

results_global = None  # référence globale pour write_results

def main():
    global results_global

    print("Loading ...")
    parser = argparse.ArgumentParser(description="Evaluation pipeline : IoU, mAP and F1-score")
    parser.add_argument("--dataset-test",  type=str,   default=PATH_DATA_TEST)
    parser.add_argument("--pred-format",   type=str,   default="xywhn")
    parser.add_argument("--path-pred",     type=str,   required=True)
    parser.add_argument("--conf-thresh",   type=float, default=0.1)
    parser.add_argument("--iou-tab",       default=[0.25, 0.5, 0.75], type=float, nargs="+")
    parser.add_argument("--classes",       default=[0, 1, 2, 3, 4], type=int, nargs="+")
    parser.add_argument("--show-boxes",    type=bool,  default=False)
    parser.add_argument("--show-curves",   type=bool,  default=False)
    args = parser.parse_args()

    path_test_folder = args.dataset_test
    path_pred_folder = args.path_pred
    path_test_labels = os.path.join(path_test_folder, "labels")
    path_test_images = os.path.join(path_test_folder, "images")

    total_pred = 0
    total_gt   = 0
    gt_cls     = {cls: 0 for cls in args.classes}
    all_ious   = {iou: [] for iou in args.iou_tab}
    # mIoU par classe
    all_ious_cls = {iou: {cls: [] for cls in args.classes} for iou in args.iou_tab}
    tp_dict    = {iou: 0 for iou in args.iou_tab}
    fp_dict    = {iou: 0 for iou in args.iou_tab}
    results    = {iou: {cls: [] for cls in args.classes} for iou in args.iou_tab}
    results_global = results

    for img_name in sorted(os.listdir(path_test_images)):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        label_name = os.path.splitext(img_name)[0] + ".txt"
        gt_path    = os.path.join(path_test_labels, label_name)
        pred_path  = os.path.join(path_pred_folder, label_name)

        img     = cv2.imread(os.path.join(path_test_images, img_name))
        img_h, img_w, _ = img.shape

        pred_boxes = xywnh2xyxyc_pred(pred_path, img_w, img_h) if args.pred_format.lower() == "xywhn" else []
        pred_boxes.sort(key=lambda x: x[5], reverse=True)
        gt_boxes = xywnh2xyxyc_gt(gt_path, img_w, img_h)

        for gt in gt_boxes:
            if gt[4] not in args.classes:
                continue
            gt_cls[gt[4]] += 1
            total_gt += 1

        matched_gt = {iou: set() for iou in args.iou_tab}

        for pred in pred_boxes:
            if pred[4] not in args.classes:
                continue
            if pred[5] < args.conf_thresh:
                continue
            total_pred += 1

            best_iou = 0; best_gt_idx = -1
            for i, gt in enumerate(gt_boxes):
                if pred[4] != gt[4]:
                    continue
                iou_val = compute_iou(pred[:4], gt[:4])
                if iou_val > best_iou:
                    best_iou = iou_val; best_gt_idx = i

            for iou_thresh in args.iou_tab:
                if best_gt_idx >= 0 and best_iou >= iou_thresh and best_gt_idx not in matched_gt[iou_thresh]:
                    tp_dict[iou_thresh] += 1
                    all_ious[iou_thresh].append(best_iou)
                    all_ious_cls[iou_thresh][pred[4]].append(best_iou)
                    matched_gt[iou_thresh].add(best_gt_idx)
                    results[iou_thresh][pred[4]].append({"conf": pred[5], "tp": 1})
                else:
                    fp_dict[iou_thresh] += 1
                    results[iou_thresh][pred[4]].append({"conf": pred[5], "tp": 0})

        if args.show_boxes:
            cv2.imshow("Prediction", draw_detections(img, pred_boxes))
            cv2.waitKey(0)

    all_aps = compute_mAP(results, gt_cls, args.classes, args.iou_tab, args.show_curves, path_pred_folder)
    f1_cls  = compute_f1(results, gt_cls, args.classes, args.iou_tab)

    print("\n" + SEP2)
    print("  RÉSULTATS GLOBAUX")
    print(SEP2)
    print_pr(tp_dict, fp_dict, total_gt, all_ious, args.iou_tab)
    print()
    print_mAp_F1(all_aps, f1_cls, gt_cls, total_gt, args.classes, args.iou_tab)

    path_save_res = os.path.join(path_pred_folder, "results.txt")
    write_results(
        path_save_res,
        tp_dict, fp_dict, total_gt, all_ious, all_ious_cls,
        all_aps, f1_cls, gt_cls, total_pred,
        args.classes, args.iou_tab,
        path_pred_folder, path_test_folder, args.conf_thresh
    )


if __name__ == "__main__":
    main()