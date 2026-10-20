"""Create reproducible exploratory data analysis figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from heart_mlops.data import FEATURES, load_or_download


def main() -> None:
    output = Path("artifacts/eda")
    output.mkdir(parents=True, exist_ok=True)
    frame = load_or_download("data/raw/processed.cleveland.data")
    sns.set_theme(style="whitegrid", palette="colorblind")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(data=frame, x="age", hue="target", bins=18, element="step", ax=axes[0])
    axes[0].set_title("Age distribution by target")
    frame["target"].value_counts().sort_index().rename(index={0: "absent", 1: "present"}).plot.bar(
        ax=axes[1], color=["#287271", "#e76f51"]
    )
    axes[1].set_title("Class balance")
    axes[1].set_xlabel("Diagnosis")
    fig.tight_layout()
    fig.savefig(output / "distributions.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 8))
    sns.heatmap(frame[FEATURES + ["target"]].corr(numeric_only=True), cmap="vlag", center=0, ax=axis)
    axis.set_title("Feature correlation (Pearson)")
    fig.tight_layout()
    fig.savefig(output / "correlation.png", dpi=160)
    plt.close(fig)

    print(f"EDA charts saved in {output}")


if __name__ == "__main__":
    main()