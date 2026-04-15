import os
import glob
import argparse
import numpy as np
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Plot YOLO score distributions par classe")
    parser.add_argument("folder", help="Dossier contenant les fichiers .txt de labels YOLO")
    parser.add_argument("--bin-size", type=float, default=0.05, help="Taille des bins (default: 0.05)")
    parser.add_argument("--save", action="store_true", help="Sauvegarder les graphiques en PNG")
    parser.add_argument("--out-dir", default=".", help="Dossier de sortie pour les PNG (default: .)")
    parser.add_argument("--no-display", action="store_true", help="Désactiver plt.show()")
    return parser.parse_args()


def load_labels(folder):
    scores_by_class = defaultdict(list)
    txt_files = glob.glob(os.path.join(folder, "*.txt"))

    if not txt_files:
        print(f"Aucun fichier .txt trouvé dans : {folder}")
        return scores_by_class

    print(f"{len(txt_files)} fichier(s) trouvé(s)...")

    for path in txt_files:
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                try:
                    cls = int(parts[0])
                    score = float(parts[-1])
                    scores_by_class[cls].append(score)
                except ValueError:
                    continue

    return scores_by_class


def plot_hist(ax, scores, title, color, bin_size):
    import matplotlib.ticker as ticker

    scores = np.array(scores)
    total = len(scores)

    n_bins = round(1.0 / bin_size)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    counts, edges = np.histogram(scores, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    bar_width = (edges[1] - edges[0]) * 0.8

    ax.bar(centers, counts, width=bar_width, color=color, alpha=0.82,
           edgecolor=color, linewidth=0.5)

    ax.set_yscale("log")
    ax.set_ylim(bottom=0.9)

    # Zones < 0.5 / >= 0.5
    n_below = int(np.sum(scores < 0.5))
    n_above = int(np.sum(scores >= 0.5))
    pct_below = n_below / total * 100
    pct_above = n_above / total * 100

    ax.axvspan(0, 0.5, alpha=0.06, color="red", zorder=0)
    ax.axvspan(0.5, 1.0, alpha=0.06, color="green", zorder=0)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)

    ymax = ax.get_ylim()[1]
    ax.text(0.25, ymax * 0.6, f"{pct_below:.1f}%\n< 0.5",
            ha="center", va="top", fontsize=9, color="#cc3333", fontweight="bold")
    ax.text(0.75, ymax * 0.6, f"{pct_above:.1f}%\n≥ 0.5",
            ha="center", va="top", fontsize=9, color="#2e7d32", fontweight="bold")

    ax.set_title(f"{title}  ({total:,} prédictions)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Score", fontsize=9)
    ax.set_ylabel("Nb prédictions (log)", fontsize=9)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.tick_params(axis="x", labelsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle="--", alpha=0.3, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure(title, class_list, scores_by_class, colors, bin_size, save, out_dir, filename, show):
    import matplotlib.pyplot as plt

    n = len(class_list)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.5 * ncols, 4 * nrows),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.38})

    axes = np.array(axes).reshape(nrows, ncols)

    for i, cls in enumerate(class_list):
        row, col = divmod(i, ncols)
        color = colors[i % len(colors)]
        lbl = "Global" if cls == "global" else f"Classe {cls}"
        plot_hist(axes[row, col], scores_by_class[cls], lbl, color, bin_size)

    # Masquer les axes vides
    for j in range(len(class_list), nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.005)

    if save:
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Sauvegardé : {path}")

    if show:
        plt.show()

    plt.close(fig)


def main():
    args = parse_args()

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    show = has_display and not args.no_display

    if not has_display:
        import matplotlib
        matplotlib.use("Agg")
        print("Info : aucun display détecté, backend Agg activé.")

    import matplotlib.pyplot as plt

    scores_by_class = load_labels(args.folder)
    if not scores_by_class:
        return

    classes = sorted(scores_by_class.keys())
    all_scores = np.array([s for cls in classes for s in scores_by_class[cls]])

    print(f"\nClasses détectées : {classes}")
    print(f"Total prédictions : {len(all_scores):,}")
    for cls in classes:
        s = np.array(scores_by_class[cls])
        print(f"  Classe {cls} : {len(s):,} préd.  |  "
              f"{np.sum(s < 0.5) / len(s) * 100:.1f}% < 0.5  |  "
              f"{np.sum(s >= 0.5) / len(s) * 100:.1f}% >= 0.5")

    colors = list(plt.cm.tab10.colors)
    os.makedirs(args.out_dir, exist_ok=True)

    scores_by_class["global"] = list(all_scores)

    make_figure(
        title="Toutes les classes (global)",
        class_list=["global"],
        scores_by_class=scores_by_class,
        colors=["#2878B5"],
        bin_size=args.bin_size,
        save=args.save,
        out_dir=args.out_dir,
        filename="scores_global.png",
        show=show,
    )

    make_figure(
        title="Distribution des scores par classe",
        class_list=classes,
        scores_by_class=scores_by_class,
        colors=colors,
        bin_size=args.bin_size,
        save=args.save,
        out_dir=args.out_dir,
        filename="scores_par_classe.png",
        show=show,
    )

    print("Terminé.")


if __name__ == "__main__":
    main()