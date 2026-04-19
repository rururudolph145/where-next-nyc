"""Reusable visualization helpers for NYC Taxi Demand Forecasting."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_time_series(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    *,
    group_col: str | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple[int, int] = (14, 4),
    marker: str = "o",
    markersize: int = 4,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot one or more time series lines.

    When *group_col* is provided each unique value gets its own coloured line,
    which is useful for comparing demand across days, zones, or models.

    Args:
        df: DataFrame containing the data to plot.
        time_col: Column to use for the x-axis (e.g. ``"hour"``).
        value_col: Column to use for the y-axis (e.g. ``"trip_count"``).
        group_col: Optional column to split into separate lines.
        title: Plot title.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        figsize: Figure size as ``(width, height)``.
        marker: Matplotlib marker style.
        markersize: Size of the markers.
        ax: Optional pre-existing Axes to draw on.

    Returns:
        The matplotlib Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if group_col is not None:
        for label, group in df.groupby(group_col):
            ax.plot(
                group[time_col],
                group[value_col],
                marker=marker,
                markersize=markersize,
                linewidth=1.8,
                label=str(label),
            )
        ax.legend()
    else:
        ax.plot(
            df[time_col],
            df[value_col],
            marker=marker,
            markersize=markersize,
            linewidth=1.8,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return ax


def plot_bar_comparison(
    labels: list[str],
    metrics: dict[str, list[float]],
    *,
    title: str = "",
    ylabel: str = "",
    figsize: tuple[int, int] = (12, 4),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Draw grouped bar chart comparing multiple metrics across categories.

    Useful for comparing model performance (MAE, RMSE, etc.) side by side.

    Args:
        labels: Category labels (e.g. model names).
        metrics: Mapping of ``{metric_name: [values_per_label]}``.
        title: Plot title.
        ylabel: Label for the y-axis.
        figsize: Figure size as ``(width, height)``.
        ax: Optional pre-existing Axes to draw on.

    Returns:
        The matplotlib Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(labels))
    n_metrics = len(metrics)
    width = 0.8 / max(n_metrics, 1)

    for i, (name, values) in enumerate(metrics.items()):
        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    *,
    kind: str = "kde",
    title: str = "",
    xlabel: str = "",
    figsize: tuple[int, int] = (8, 4),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the distribution of a single numeric column.

    Args:
        df: DataFrame containing the data.
        column: Column whose distribution to plot.
        kind: One of ``"kde"``, ``"hist"``, or ``"box"``.
        title: Plot title.
        xlabel: Label for the x-axis.
        figsize: Figure size as ``(width, height)``.
        ax: Optional pre-existing Axes to draw on.

    Returns:
        The matplotlib Axes object.

    Raises:
        ValueError: If *kind* is not one of kde, hist, box.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if kind == "kde":
        sns.kdeplot(df[column], ax=ax, fill=True)
    elif kind == "hist":
        ax.hist(df[column].dropna(), bins=50, edgecolor="white")
    elif kind == "box":
        sns.boxplot(y=df[column], ax=ax)
    else:
        raise ValueError(f"Unknown kind {kind!r}. Choose from: kde, hist, box.")

    ax.set_title(title or f"Distribution of {column}")
    ax.set_xlabel(xlabel or column)
    return ax


def plot_splits(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    splits: dict[str, tuple[Any, Any]],
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple[int, int] = (16, 4),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a time series with shaded regions for train / val / test splits.

    Args:
        df: DataFrame containing the full time series.
        time_col: Column for the x-axis.
        value_col: Column for the y-axis.
        splits: Mapping of ``{split_name: (start, end)}`` defining shaded
            regions.  E.g. ``{"Train": (0, 700), "Val": (700, 750),
            "Test": (750, 800)}``.
        title: Plot title.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        figsize: Figure size as ``(width, height)``.
        ax: Optional pre-existing Axes to draw on.

    Returns:
        The matplotlib Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    split_colors = ["#4488ff", "#ff9900", "#ff4444", "#44bb44"]

    ax.plot(df[time_col], df[value_col], linewidth=0.8, color="steelblue")

    for i, (name, (start, end)) in enumerate(splits.items()):
        color = split_colors[i % len(split_colors)]
        ax.axvspan(start, end, alpha=0.12, color=color, label=name)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_actual_vs_predicted(
    actual: pd.Series | np.ndarray,
    predictions: dict[str, pd.Series | np.ndarray],
    *,
    time_index: pd.Series | np.ndarray | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple[int, int] = (14, 4),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlay actual values against one or more model predictions.

    Args:
        actual: Ground-truth values.
        predictions: Mapping of ``{model_name: predicted_values}``.
        time_index: Optional x-axis values. Defaults to integer index.
        title: Plot title.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        figsize: Figure size as ``(width, height)``.
        ax: Optional pre-existing Axes to draw on.

    Returns:
        The matplotlib Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    x = time_index if time_index is not None else np.arange(len(actual))

    ax.plot(x, actual, color="black", linewidth=1.5, label="Actual", zorder=5)
    for name, preds in predictions.items():
        ax.plot(x, preds, linewidth=1.2, alpha=0.8, linestyle="--", label=name)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax
