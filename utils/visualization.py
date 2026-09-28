import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_performance_plots(chunks_data):
    """Generates visual evaluation diagnostics across multiple data chunks.

    Parameters:
    -----------
    chunks_data : dict
        Nested dictionary containing ground truth and prediction arrays per
        chunk: {
            "Train": {"y_true": y_train, "y_pred": train_preds},
            "Validation": {"y_true": y_val, "y_pred": val_preds}
        }

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Rendered figure containing Actual vs Predicted and Residual plots.
    """
    num_chunks = len(chunks_data)
    fig, axes = plt.subplots(
        nrows=num_chunks,
        ncols=2,
        figsize=(12, 5 * num_chunks),
        tight_layout=True,
    )

    # Ensure axes matrix handles single-chunk cases safely
    if num_chunks == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (chunk_name, data) in enumerate(chunks_data.items()):
        y_true = np.array(data["y_true"])
        y_pred = np.array(data["y_pred"])
        residuals = y_true - y_pred

        # Subplot 1: Actual vs. Predicted
        ax_act = axes[idx, 0]
        ax_act.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", linewidth=0.5)

        # Perfect prediction reference line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax_act.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="red",
            linestyle="--",
            label="Ideal Fit",
        )

        ax_act.set_title(f"{chunk_name}: Actual vs Predicted Price")
        ax_act.set_xlabel("Actual Price")
        ax_act.set_ylabel("Predicted Price")
        ax_act.legend()
        ax_act.grid(True, linestyle=":", alpha=0.6)

        # Subplot 2: Residual Distribution
        ax_res = axes[idx, 1]
        sns.histplot(
            residuals,
            kde=True,
            ax=ax_res,
            color="teal",
            line_kws={"linewidth": 2},
        )
        ax_res.axvline(
            x=0, color="red", linestyle="--", label="Zero Residual Line"
        )

        ax_res.set_title(f"{chunk_name}: Residual Error Distribution")
        ax_res.set_xlabel("Residual (Actual - Predicted)")
        ax_res.set_ylabel("Frequency")
        ax_res.legend()
        ax_res.grid(True, linestyle=":", alpha=0.6)

    return fig
