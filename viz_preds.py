import os
import glob
import argparse
import numpy as np


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Palette tab10 en RGB entiers
COLORS = [
    (31,  119, 180),
    (255, 127,  14),
    ( 44, 160,  44),
    (214,  39,  40),
    (148, 103, 189),
    (140,  86,  75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189,  34),
    ( 23, 190, 207),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise les prédictions YOLO sur les images correspondantes"
    )
    parser.add_argument("labels", help="Dossier contenant les fichiers .txt YOLO")
    parser.add_argument("images", help="Dossier contenant les images")
    parser.add_argument(
        "--score-thresh", type=float, default=0.0,
        help="Seuil de score minimum pour afficher une boîte (default: 0.0)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Sauvegarder les images annotées au lieu de les afficher"
    )
    parser.add_argument(
        "--out-dir", default="viz_out",
        help="Dossier de sortie si --save (default: viz_out)"
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Nombre maximum d'images à traiter"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Désactiver plt.show()"
    )
    return parser.parse_args()


def find_image(images_dir, stem):
    for ext in IMAGE_EXTS:
        for variant in (ext, ext.upper()):
            path = os.path.join(images_dir, stem + variant)
            if os.path.isfile(path):
                return path
    return None


def load_boxes(label_path, score_thresh):
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                score = float(parts[5]) if len(parts) >= 6 else None
            except ValueError:
                continue
            if score is not None and score < score_thresh:
                continue
            boxes.append((cls, cx, cy, w, h, score))
    return boxes


def draw_boxes(ax, img, boxes, img_name):
    import matplotlib.patches as patches

    H, W = img.shape[:2]
    ax.imshow(img)
    ax.set_title(img_name, fontsize=8)
    ax.axis("off")

    for cls, cx, cy, bw, bh, score in boxes:
        x0 = (cx - bw / 2) * W
        y0 = (cy - bh / 2) * H
        pw = bw * W
        ph = bh * H

        color = tuple(c / 255 for c in COLORS[cls % len(COLORS)])

        rect = patches.Rectangle(
            (x0, y0), pw, ph,
            linewidth=1.5, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        label = f"cls{cls}" if score is None else f"cls{cls} {score:.2f}"
        ax.text(
            x0, y0 - 3, label,
            fontsize=6, color="white", fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor="none"),
            va="bottom"
        )


def main():
    args = parse_args()

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    show = has_display and not args.no_display

    if not has_display or args.save:
        import matplotlib
        matplotlib.use("Agg")
        if not has_display:
            print("Info : aucun display détecté, backend Agg activé.")

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    label_files = sorted(glob.glob(os.path.join(args.labels, "*.txt")))
    if not label_files:
        print(f"Aucun fichier .txt trouvé dans : {args.labels}")
        return

    if args.max_images:
        label_files = label_files[: args.max_images]

    if args.save:
        os.makedirs(args.out_dir, exist_ok=True)

    print(f"{len(label_files)} fichier(s) labels à traiter...")
    missing = 0

    for label_path in label_files:
        stem = os.path.splitext(os.path.basename(label_path))[0]
        img_path = find_image(args.images, stem)

        if img_path is None:
            missing += 1
            continue

        boxes = load_boxes(label_path, args.score_thresh)
        img = mpimg.imread(img_path)

        fig, ax = plt.subplots(figsize=(10, 7))
        draw_boxes(ax, img, boxes, os.path.basename(img_path))
        fig.tight_layout(pad=0.5)

        if args.save:
            out_path = os.path.join(args.out_dir, stem + ".jpg")
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            print(f"  Sauvegardé : {out_path}")
        elif show:
            plt.show()

        plt.close(fig)

    if missing:
        print(f"Warning : {missing} label(s) sans image correspondante ignoré(s).")
    print("Terminé.")


if __name__ == "__main__":
    main()
