import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import (
    PLOT_STYLE,
    get_combined_global_df,
    get_model_style,
    load_data,
    save_figure,
    setup_style,
)


def plot_convergence_profile(combined_df, output_name="convergence_profile.png"):
    setup_style()
    fig, ax = plt.subplots(
        figsize=PLOT_STYLE["figsize_single_col"], constrained_layout=True
    )

    # 0% to 100% tolerance thresholds
    thresholds = np.linspace(0, 100, 501)
    models = combined_df["model"].unique()
    styles = get_model_style(models)

    legend_elements = {}

    for model in models:
        subset = combined_df[combined_df["model"] == model].dropna(subset=["fog"])
        if len(subset) == 0:
            continue

        gaps = subset["fog"].values
        # Convergence Rate: Proportion of samples where Gap <= tau
        convergence_rates = [np.mean(gaps <= t) for t in thresholds]

        short_name = model.split("-")[0]
        style = styles[model]

        (line,) = ax.plot(
            thresholds,
            convergence_rates,
            color=style["color"],
            linewidth=2.5,
            alpha=0.9,
        )

        if short_name not in legend_elements:
            line.set_label(short_name)
            legend_elements[short_name] = line

    # NeurIPS / Engineering Standard Labels
    ax.set_xlabel(r"Optimality Tolerance $\tau$ (\%)")
    ax.set_ylabel(r"Convergence Rate ($Gap \leq \tau$)")

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, 100)

    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(
        loc="lower right", frameon=True, fontsize=PLOT_STYLE["font_sizes"]["legend"]
    )

    sns.despine()
    save_figure(fig, output_name)


def main():
    data = load_data()
    df = get_combined_global_df(data)
    if df is not None:
        # Saving as PDF for high-quality LaTeX inclusion
        plot_convergence_profile(df, "convergence_profile.png")


if __name__ == "__main__":
    main()
