import json
import math
import os
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from PIL import ImageColor
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler  # z-scoring
from umap import UMAP

import wandb
from src.global_vars import BASE_DATA_DIR

disease_colors = {
    "cardiometabolic disease": "#E41A1C",  # Red
    "immune disease": "#377EB8",  # Blue
    "cancer": "#4DAF4A",  # Green
    "mental disease": "#984EA3",  # Purple
    "infectious disease": "#FF7F00",  # Orange
    "digestive disease": "#FFFF33",  # Yellow
    "nerve disease": "#A65628",  # Brown
    "endocrine disease": "#F781BF",  # Pink
    "liver disease": "#999999",  # Gray
    "kidney disease": "#66C2A5",  # Teal
    "Unknown": "#000000",  # Black
}

method_colors = {
    "Balanced RF classifier": "#559ccf",  # blue
    "RF with balanced data": "#1a9fff",  # blue
    "k-shot RF": "#003a64",
    "simple baseline RF": "#56b8ff",
    "k-shot XGB": "#858500",
    "simple baseline XGB": "#ffff34",
    "k-shot MLP": "#4e0096",
    "simple baseline MLP": "#992dff",
    "Protonet with embeddings": "#ff7f0e",
    "Protonet without embeddings": "#d62728",
    "Dummy": "gray",
    "RF": "#1f77b4",  # blue
    "Protonet": "#ff7f0e",  # orange
    "XGB": "#bcbd22",  # green
    "XGBoost": "#bcbd22",  # green
    "MLP": "#9467bd",  # green
    "Random": "gray",
}


def get_method_color(method_name):
    for key in method_colors:
        if key in method_name:
            return method_colors[key]
    return "gray"  # default


color_adjust = {"base": 1.0, "pca": 0.8, "pca_feat": 0.5}


def get_color_adjustment(method_name):
    if "base" in method_name.lower():
        return color_adjust["base"]
    elif "pca feat" in method_name.lower():
        return color_adjust["pca_feat"]
    elif "pca" in method_name.lower():
        return color_adjust["pca"]
    else:
        return 1.0


# Line width based on shot number
width_map = {2: 1, 5: 3, 10: 5, 25: 7}


def get_line_width(method_name):
    if "25" in method_name:
        return width_map[25]
    elif "5" in method_name:
        return width_map[5]
    elif "10" in method_name:
        return width_map[10]
    elif "2" in method_name:
        return width_map[2]
    else:
        return 5  # Default line width


colorway = [
    "#ff7f0e",  # orange
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


def print_full_df(x):
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    print(x)
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")
    pd.reset_option("display.width")
    pd.reset_option("display.float_format")
    pd.reset_option("display.max_colwidth")


def get_data_df_from_wandb(entity_name, project_name, tags):
    # Initialize wandb API
    api = wandb.Api()

    # Get runs from W&B with the specified tag
    runs = api.runs(f"{entity_name}/{project_name}", filters={"tags": {"$all": tags}})
    print(len(runs))

    # Create an empty DataFrame to store all data
    full_df = pd.DataFrame()

    # Extract data from Test Metrics Summary table
    for run in runs:
        table_file = None
        for f in run.files():
            if (
                f.name.endswith(".table.json")
                and "Test Metrics Summary table" in f.name
            ):
                table_file = f
                break

        if table_file is not None:
            # Download and load the JSON file
            local_path = table_file.download(replace=True)
            with open(table_file.name, "r") as fp:
                table_json = json.load(fp)

            # Create DataFrame and add run name
            df = pd.DataFrame(table_json["data"], columns=table_json["columns"])
            df["run_name"] = run.name
            full_df = pd.concat([full_df, df])

    return full_df


def get_cross_val_results_from_wandb(
    project_name,
    tags,
    column_wanted="Outer fold.test/f1",
    notes_contains=None,
    studies_to_ignore=[],
):
    api = wandb.Api()

    # get run names interested in
    runs = api.runs(f"shayan000/{project_name}", filters={"tags": {"$all": tags}})

    # Download all data points for the graph "Outer fold.test/f1" from wandb
    data_wanted_dict = {}
    for run in runs:
        # if run.state != "finished":
        #     print(f"Run {run.name} is not finished. Skipping...")
        #     continue

        # Check if notes contain the specified text
        if notes_contains is not None:
            if (
                not hasattr(run, "notes")
                or run.notes is None
                or notes_contains not in str(run.notes)
            ):
                print(
                    f"Run {run.name} doesn't have required text in notes. Skipping..."
                )
                continue

        # Download the data points)
        run_data = run.scan_history(keys=[column_wanted])

        data_wanted = [
            row[column_wanted]
            for row in run_data
            if column_wanted in row and row[column_wanted] is not None
        ]
        if not data_wanted:
            print(f"No data found for run {run.name}. Skipping...")
            continue

        # data_wanted_dict[run.name] = [max(data_wanted)]
        data_wanted_dict[run.name] = data_wanted
    # Create a DataFrame from the dictionary
    try:
        full_df = pd.DataFrame(data_wanted_dict)
    except Exception as e:
        print(len(data_wanted_dict))
        print([v for v in data_wanted_dict.keys()])
        print([len(v) for v in data_wanted_dict.values()])
        raise e

    return full_df


def visualize_metrics_by_study_groups(
    full_dfs,
    study_groups,
    legend_name,
    group_names,
    metrics=None,
    metric_display_names=None,
    title_prefix="Average Mean of Test Metrics by Metric and Study Size",
):
    """
    Generate a grouped bar chart showing average metrics across different study groups.

    Parameters:
    -----------
    study_groups : list of lists
        Each inner list contains study names belonging to a specific group
    group_names : list
        Names for each study group (e.g., "<100", "100-200")
    entity_name : str
        Weights & Biases entity name
    project_name : str
        Weights & Biases project name
    tag : str
        Tag to filter runs by
    metrics : list, optional
        List of metric names to analyze (default metrics provided if None)
    metric_display_names : list, optional
        Display names for metrics (defaults to same as metrics if None)
    title_prefix : str, optional
        Prefix for the chart title

    Returns:
    --------
    tuple
        (plotly figure, processed dataframe with metrics, raw dataframe from W&B)
    """
    # Set default metrics if not provided
    if metrics is None:
        metrics = [
            "test/accuracy",
            "test/f1",
            "test/roc_auc(_macro)",
            "test/precision(_binary)",
            "test/recall(_binary)",
        ]

    # Set default metric display names if not provided
    if metric_display_names is None:
        metric_display_names = ["Accuracy", "F1", "ROC AUC", "Precision", "Recall"]

    # Calculate average metrics for each study group
    avg_metrics = {metric: [] for metric in metrics}

    for study_list in study_groups:
        # Filter DataFrame for runs containing any study name in the current group
        study_df = full_df[full_df["run_name"].str.contains("|".join(study_list))]

        # Calculate average for each metric
        for metric in metrics:
            avg_metrics[metric].append(
                study_df[study_df["Metric"] == metric]["Mean"].mean()
            )

    # Build a tidy DataFrame for plotting
    plot_data = []
    for m, m_disp in zip(metrics, metric_display_names):
        for i, group_name in enumerate(group_names):
            plot_data.append(
                {
                    "Metric": m_disp,
                    legend_name: group_name,
                    "Average Mean": avg_metrics[m][i],
                }
            )

    df_plot = pd.DataFrame(plot_data)

    # Create an interactive grouped bar chart
    fig = px.bar(
        df_plot,
        x="Metric",
        y="Average Mean",
        color=legend_name,
        barmode="group",
        title=f"{title_prefix}",
        template="plotly_white",
        hover_data={"Average Mean": ":.3f"},
    )

    # Style the figure
    fig.update_layout(
        height=700,
        title=dict(text=f"{title_prefix}", font=dict(weight="bold", size=22), x=0.5),
    )

    # Larger font size for axes and labels
    fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=14))

    return fig, df_plot, full_df


def compare_studies_within_group(
    study_list,
    full_df,
    title_addition,
    metadata_file=None,
    metrics=None,
    metric_display_names=None,
    width=1700,
    height=700,
    barmode="group",
    sort_by_metric=None,
    sort_ascending=False,
    tickangle=10,
    orientation="v",
):
    """
    Compare metrics between individual studies within a study group, with additional metadata on x-axis.

    Parameters:
    -----------
    study_list : list
        List of study names to compare
    full_df : pandas.DataFrame
        The complete dataframe containing all metrics data from W&B
    title_addition : str
        Text to add to the title (e.g., "<100 samples")
    metadata_file : str, optional
        Path to CSV file containing study metadata
    metrics : list, optional
        List of metric names to analyze (default metrics provided if None)
    metric_display_names : list, optional
        Display names for metrics (defaults to same as metrics if None)
    height : int, optional
        Height of the plot in pixels
    barmode : str, optional
        Bar mode for plotly ('group', 'stack', 'relative', etc.)
    sort_by_metric : str, optional
        Metric to sort studies by (None for no sorting)
    sort_ascending : bool, optional
        Sort order (if sort_by_metric is provided)

    Returns:
    --------
    tuple
        (plotly figure, processed dataframe with metrics)
    """
    # Set default metrics if not provided
    if metrics is None:
        metrics = [
            "test/accuracy",
            "test/f1",
            "test/roc_auc(_macro)",
            "test/precision(_binary)",
            "test/recall(_binary)",
        ]

    # Set default metric display names if not provided
    if metric_display_names is None:
        metric_display_names = ["Accuracy", "F1", "ROC AUC", "Precision", "Recall"]

    # Create mapping between internal metric names and display names
    metric_to_display = dict(zip(metrics, metric_display_names))
    display_to_metric = dict(zip(metric_display_names, metrics))

    # Load metadata if provided
    metadata_df = None
    if metadata_file:
        metadata_df = pd.read_csv(metadata_file, header=0, index_col=0)

    # Prepare data for plotting
    data = []
    study_info = {}  # Dictionary to store additional info for each study

    for study in study_list:
        # Extract metadata information if available
        if metadata_df is not None:
            study_metadata = metadata_df[metadata_df["Project_1"] == study]

            if not study_metadata.empty:
                # Count label distribution (Control vs Disease)
                label_counts = study_metadata["Group"].value_counts().to_dict()

                # Count disease distribution
                disease_counts = study_metadata["group"].value_counts().to_dict()

                # Store this information
                study_info[study] = {
                    "label_counts": label_counts,
                    "disease_counts": disease_counts,
                    "total_samples": len(study_metadata),
                }

        # Filter DataFrame for runs containing the current study
        study_df = full_df[full_df["run_name"].str.contains(study)]

        for m, m_disp in zip(metrics, metric_display_names):
            # Calculate the mean metric value for this study
            metric_mean = study_df[study_df["Metric"] == m]["Mean"].mean()

            data.append(
                {
                    "Study": study,
                    "Metric": m_disp,  # Use display name for the plot
                    "Mean": metric_mean,
                }
            )

    df_plot = pd.DataFrame(data)

    # Create custom x-axis labels with metadata
    if metadata_df is not None:
        # Create a mapping of study name to custom label
        study_labels = {}
        for study in study_list:
            info = study_info.get(study, {})

            if info:
                # Format label counts
                label_str = ", ".join(
                    [f"{k}: {v}" for k, v in info.get("label_counts", {}).items()]
                )

                # Format disease counts
                disease_str = ", ".join(
                    [f"{k}: {v}" for k, v in info.get("disease_counts", {}).items()]
                )

                # Create the custom label
                custom_label = f"{study}<br>Total: {info.get('total_samples', 'N/A')}<br>Labels: {label_str}<br>Diseases: {disease_str}"
                study_labels[study] = custom_label
            else:
                study_labels[study] = study

        # Add custom label column to the dataframe
        df_plot["StudyLabel"] = df_plot["Study"].map(study_labels)
    else:
        # No metadata, use study name as is
        df_plot["StudyLabel"] = df_plot["Study"]

    # Sort studies by a specific metric if requested
    if sort_by_metric is not None:
        # Get means for the specified metric for each study
        if sort_by_metric in metric_display_names:
            # If display name was provided
            sort_metric = sort_by_metric
        elif sort_by_metric in metrics:
            # If internal metric name was provided
            sort_metric = metric_to_display.get(sort_by_metric, sort_by_metric)
        else:
            # Default to first metric if not found
            sort_metric = metric_display_names[0]

        # Filter dataframe to get only the sorting metric
        sort_df = df_plot[df_plot["Metric"] == sort_metric].copy()

        # Get the ordered list of studies
        ordered_studies = sort_df.sort_values("Mean", ascending=sort_ascending)[
            "Study"
        ].tolist()

        # Create ordered list of study labels based on the ordered studies
        ordered_labels = (
            [study_labels.get(study, study) for study in ordered_studies]
            if metadata_df is not None
            else ordered_studies
        )

        # Convert Study column to categorical with the ordered categories
        df_plot["Study"] = pd.Categorical(
            df_plot["Study"], categories=ordered_studies, ordered=True
        )

        # Sort the dataframe
        df_plot = df_plot.sort_values("Study")

        # Decide on plotting orientation
    if orientation == "h":
        # Horizontal bars: x = numeric, y = category
        x_col = "Mean"
        y_col = "StudyLabel"
        orientation_plotly = "h"
    else:
        # Vertical bars: x = category, y = numeric
        x_col = "StudyLabel"
        y_col = "Mean"
        orientation_plotly = "v"

    # Create figure
    fig = px.bar(
        df_plot,
        x=x_col,
        y=y_col,
        color="Metric",
        orientation=orientation_plotly,
        barmode=barmode,
        template="plotly_white",
        hover_data={"Mean": ":.3f", "Study": True},
        labels={"StudyLabel": "Study"},
        height=height,
        width=width,
    )

    # Style the figure
    fig.update_layout(
        height=height,
        width=width,
        title=dict(
            text=f"Mean of Test Metrics for studies with {title_addition} samples",
            font=dict(weight="bold", size=22),
            x=0.5,
        ),
    )

    # Update layout (title, grid, axis fonts)
    fig.update_layout(
        title=dict(
            text=f"Mean of Test Metrics for studies with {title_addition} samples",
            font=dict(weight="bold", size=22),
            x=0.5,
        )
    )
    fig.update_xaxes(title_font=dict(size=32), tickfont=dict(size=30))
    fig.update_yaxes(title_font=dict(size=32), tickfont=dict(size=30))

    # Optionally rotate tick labels if vertical
    if orientation == "v":
        fig.update_xaxes(tickangle=tickangle)
    else:
        fig.update_yaxes(tickangle=tickangle)

    return fig, df_plot


# Example usage:
# fig, df_comparison = compare_studies_within_group(
#     study_list=studies_less_than_100,
#     full_df=full_df,
#     metadata_file="path/to/metadata.csv",
#     title_addition="<100",
#     sort_by_metric="Accuracy",
#     sort_ascending=False
# )
#
# fig.show()


def plot_cross_val_results_grouped(
    group_df_lists,
    group_titles,
    models,
    title=None,
    save_path: str = "",
    height=1000,
    width=1400,
    legend=True,
    group_title=None,
    legend_title=None,
):
    # Create figure
    fig = go.Figure()

    show_annotations = False if len(models) > 1 else True

    # Track overall min/max for y-axis scaling
    all_means = []
    all_errors = []

    # Position tracking for bars
    group_width = 0.8  # Width allocated for each group
    model_width = group_width / len(models)  # Width for each model's bar
    # remove empty groups:
    empty_groups = []
    for i, gt in enumerate(group_titles):
        if sum([df.shape[1] for df in group_df_lists[i]]) == 0:
            empty_groups.append(gt)
            continue
    print("Empty groups: ", empty_groups)

    group_df_lists = [
        df_list
        for i, df_list in enumerate(group_df_lists)
        if group_titles[i] not in empty_groups
    ]
    group_titles = [g for g in group_titles if g not in empty_groups]

    dummy_median = None
    if "Dummy" in group_titles:
        dummy_idx = group_titles.index("Dummy")
        dummy_df = group_df_lists[dummy_idx][0]
        dummy_values = dummy_df.values.flatten()
        dummy_values = dummy_values[dummy_values != 0]
        if len(dummy_values) > 0:
            dummy_median = np.median(dummy_values)

    # For each group
    for group_idx, (df_list, gt) in enumerate(zip(group_df_lists, group_titles)):
        # print("group title  ", gt)
        # print([df.shape for df in df_list])

        vals_per_df = [
            df.values.flatten()[df.values.flatten() != 0] for df in df_list
        ]  # Filter out zeros
        print(vals_per_df)
        means = np.array(
            [np.mean(vals) if len(vals) > 0 else 0 for vals in vals_per_df]
        )
        stds = np.array([np.std(vals) if len(vals) > 0 else 0 for vals in vals_per_df])
        all_means.extend(means)
        all_errors.extend(stds)

        # Add bars for each model in this group
        for model_idx, (model, df) in enumerate(zip(models, df_list)):
            mean = means[model_idx]
            std = stds[model_idx]
            # Calculate position for this bar
            position = group_idx + (model_idx - len(models) / 2 + 0.5) * model_width
            values = df.values.flatten()
            non_zero_values = values[values != 0]
            color_by = model if len(models) > 1 else gt

            # Get styling
            line_width = get_line_width(color_by)
            opacity = get_color_adjustment(color_by)

            # Add bar
            fig.add_trace(
                go.Box(
                    x=[position] * len(non_zero_values),
                    y=non_zero_values,
                    # error_y=dict(type="data", array=[std]),
                    name=model,
                    legendgroup=model if legend else None,
                    marker_color=get_method_color(color_by),
                    opacity=opacity,
                    width=model_width * 0.9,  # Slightly narrower than allocated space
                    line=dict(
                        width=line_width,
                    ),
                    showlegend=group_idx == 0
                    if legend
                    else False,  # Only show in legend for first group
                    # text=f"{mean:.2f}<br>±<br>{std:.2f}",
                    # textposition="",
                )
            )

            if show_annotations and len(non_zero_values) > 0:
                fig.add_annotation(
                    x=position,
                    y=max(non_zero_values) + 0.01,  # Position above the box
                    text=f"{mean:.3f}<br>±<br>{std:.2f}",
                    showarrow=False,
                    font=dict(size=20, weight="bold"),
                    xanchor="center",
                    yanchor="bottom",
                )

            base_position = (
                group_idx + (model_idx - len(models) / 2 + 0.5) * model_width
            )
            # if show_annotations:
            #     fig.add_annotation(
            #         x=base_position,
            #         y=0,
            #         text=f"{mean:.2f}<br>±<br>{std:.2f}",
            #         showarrow=False,
            #         font=dict(size=10, weight="bold", color="black"),
            #     )

    if dummy_median is not None:
        fig.add_hline(
            y=dummy_median,
            line_dash="dash",
            line_color="gray",
        )

    all_values = []
    for df_list in group_df_lists:
        for df in df_list:
            all_values.extend(df.values.flatten())
    all_values = [v for v in all_values if v != 0]  # Remove zeros
    y_min = min(all_values) if all_values else 0
    y_max = max(all_values) if all_values else 1
    y_min = math.floor(y_min * 10) / 10  # Round down to nearest 0.1
    y_max = math.ceil(y_max * 10) / 10  # Round up to nearest 0.1

    # Update layout
    group_positions = np.arange(len(group_titles))
    fig.update_layout(
        title=dict(text=title, font=dict(weight="bold", size=24), x=0.5, y=0.99)
        if title
        else None,
        xaxis=dict(
            tickvals=group_positions,
            ticktext=group_titles,
            title=str(group_title),
            title_font=dict(size=32),
            tickfont=dict(size=26),
            tickangle=65,
            automargin=True,
        ),
        yaxis=dict(
            title="F1 Score",
            title_font=dict(size=32),
            tickfont=dict(size=30),
            automargin=True,
            range=[0, 1.3],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(size=18),
            title=dict(text=legend_title, font=dict(size=20)),
        )
        if legend
        else None,
        height=height,
        width=width,
        margin=dict(b=0, t=150, l=100),
    )

    fig.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the figure as an image
        fig.write_image(save_path, scale=2)


def plot_combined_cross_val_results(
    unbalanced_df_lists,
    balanced_df_lists,
    group_titles,
    models,
    title=None,
    save_path: str = "",
):
    # Create figure
    fig = go.Figure()

    # Define color palette
    colors = px.colors.qualitative.Plotly

    # Track overall min/max for y-axis scaling
    all_means = []
    all_errors = []

    # Position tracking for bars
    group_width = 0.8  # Width allocated for each group
    model_pair_width = group_width / len(models)  # Width for each model pair
    bar_width = model_pair_width * 0.4  # Width for individual bars
    group_positions = np.arange(len(group_titles))

    # First create a hidden trace for each model to build the model section of the legend
    for model_idx, model in enumerate(models):
        fig.add_trace(
            go.Bar(
                x=[None],
                y=[None],
                name=model,
                legendgroup="models",
                legendgrouptitle_text="Models",
                marker_color=colors[model_idx % len(colors)],
                showlegend=True,
            )
        )

    # Create hidden traces for the data types section of the legend
    # Unbalanced - with pattern
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            name="Unbalanced",
            legendgroup="data_types",
            legendgrouptitle_text="Data Types",
            marker=dict(
                color="lightgray",
                pattern_shape="/",  # Pattern for unbalanced
            ),
            showlegend=True,
        )
    )

    # Balanced - without pattern
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            name="Balanced",
            legendgroup="data_types",
            marker_color="lightgray",  # Solid gray for balanced
            showlegend=True,
        )
    )

    # For each group
    for group_idx, group_title in enumerate(group_titles):
        # Process each model
        for model_idx, model in enumerate(models):
            # Calculate position for this model's bars
            base_position = (
                group_idx + (model_idx - len(models) / 2 + 0.5) * model_pair_width
            )

            # Process unbalanced data first (left side)
            unbalanced_vals = unbalanced_df_lists[group_idx][model_idx].values.flatten()
            unbalanced_mean = np.mean(unbalanced_vals)
            unbalanced_std = np.std(unbalanced_vals)

            # Process balanced data second (right side)
            balanced_vals = balanced_df_lists[group_idx][model_idx].values.flatten()
            balanced_mean = np.mean(balanced_vals)
            balanced_std = np.std(balanced_vals)

            # Track statistics for y-axis scaling
            all_means.extend([balanced_mean, unbalanced_mean])
            all_errors.extend([balanced_std, unbalanced_std])

            # Add unbalanced data bar with pattern (on left)
            fig.add_trace(
                go.Bar(
                    x=[base_position - bar_width / 2],
                    y=[unbalanced_mean],
                    error_y=dict(type="data", array=[unbalanced_std]),
                    marker_color=colors[model_idx % len(colors)],
                    marker_pattern_shape="/",  # Same pattern for all unbalanced
                    width=bar_width * 0.9,
                    showlegend=False,
                    hovertemplate=f"{model} (Unbalanced): {unbalanced_mean:.2f}±{unbalanced_std:.2f}<extra></extra>",
                )
            )

            # Add balanced data bar (on right)
            fig.add_trace(
                go.Bar(
                    x=[base_position + bar_width / 2],
                    y=[balanced_mean],
                    error_y=dict(type="data", array=[balanced_std]),
                    marker_color=colors[model_idx % len(colors)],
                    width=bar_width * 0.9,
                    showlegend=False,
                    hovertemplate=f"{model} (Balanced): {balanced_mean:.2f}±{balanced_std:.2f}<extra></extra>",
                )
            )

            # Add text annotations for means and stds
            fig.add_annotation(
                x=base_position - bar_width / 2,
                y=unbalanced_mean + unbalanced_std + 0.06,
                text=f"{unbalanced_mean:.2f}<br>±<br>{unbalanced_std:.2f}",
                showarrow=False,
                font=dict(size=10),
            )

            fig.add_annotation(
                x=base_position + bar_width / 2,
                y=balanced_mean + balanced_std + 0.06,
                text=f"{balanced_mean:.2f}<br>±<br>{balanced_std:.2f}",
                showarrow=False,
                font=dict(size=10),
            )

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(weight="bold", size=16),
            x=0.5,
        )
        if title
        else None,
        xaxis=dict(
            tickvals=group_positions,
            ticktext=group_titles,
            title="Group",
            title_font=dict(size=32),
            tickfont=dict(size=30),
            tickangle=55,
        ),
        yaxis=dict(
            title="F1 Score",
            title_font=dict(size=32),
            tickfont=dict(size=30),
            range=[
                min(all_means) - max(all_errors) - 0.05,
                max(all_means) + max(all_errors) + 0.15,
            ],
        ),
        legend=dict(
            orientation="v",  # vertical orientation
            yanchor="top",  # anchor to middle of y axis
            # y=0.0,  # center vertically
            xanchor="center",  # anchor to left of the legend
            x=0.0,  # position slightly to the right of the plot
            groupclick="togglegroup",
            tracegroupgap=10,  # space between groups
            font=dict(size=23),
        ),
        height=1200,
        width=1500,
        margin=dict(b=150),
    )

    fig.show()

    if save_path:
        fig.write_image(save_path, scale=2)


# @cache
def get_desired_cross_val_results_for_models(
    project, models, tags, metric, studies_to_ignore, notes_contains=None
):
    dfs = []
    for model in models:
        df = get_cross_val_results_from_wandb(
            project, [model] + list(tags), metric, notes_contains, studies_to_ignore
        )
        # print_full_df(df)
        print(f"Model: {model}, shape: {df.shape}")
        # print(len(set(map(lambda x: x.split("_")[3], df.columns.tolist()))))  # not always 3
        if studies_to_ignore:
            df = df.loc[:, ~df.columns.str.contains("|".join(studies_to_ignore))]
            print(f"After removing studies: {df.shape}")
        dfs.append(df)
    return dfs


def keep_some_studies_in_dfs(dfs, studies_to_keep):
    new_dfs = []
    for i, df in enumerate(dfs):
        new_dfs.append(df.loc[:, df.columns.str.contains("|".join(studies_to_keep))])
    return new_dfs


def plot_cross_val_results_by_study(
    model_dfs,
    model_names,
    extract_study_func=None,
    title=None,
    save_path="",
    color_palette=None,
    y_axis_title="F1 Score",
    y_range=None,
    height=1000,
    width=1800,
    show_annotations=True,
    annotation_offset=0.05,
    bar_width_factor=0.7,
    study_sizes: dict = None,
):
    """Generate a grouped bar plot showing model performance across different studies.

    Parameters:
    -----------
    model_dfs : list of pandas.DataFrame
        List of dataframes for different models, where columns are cross-val results for each study
    model_names : list of str
        Names of the models corresponding to the dataframes
    title : str, optional
        Title for the plot
    save_path : str, optional
        Path to save the figure (if empty, figure is not saved)
    color_palette : list, optional
        Custom color palette for the models
    y_axis_title : str, optional
        Title for the y-axis
    y_range : list, optional
        Custom range for y-axis [min, max]
    height, width : int, optional
        Height and width of the figure in pixels
    extract_study_func : function, optional
        Function to extract study name from column name
    show_annotations : bool, optional
        Whether to show mean±std annotations on bars
    annotation_offset : float, optional
        Vertical offset for annotations
    bar_width_factor : float, optional
        Factor to determine bar width (0-1)

    Returns:
    --------
    plotly.graph_objects.Figure
        The generated plot
    dict
        Dictionary containing computed statistics (means and stds) for each model and study
    """
    # Check input validity
    if len(model_dfs) != len(model_names):
        raise ValueError("Number of dataframes must match number of model names")

    if not model_dfs:
        raise ValueError("At least one dataframe must be provided")

    # Create figure
    fig = go.Figure()

    # Process all dataframes to get all unique studies
    all_studies = set()
    for df in model_dfs:
        # Extract study names from column names
        if extract_study_func:
            studies = [extract_study_func(col) for col in df.columns]
        else:
            studies = df.columns.tolist()
        all_studies.update(studies)

    # Sort studies alphabetically for consistent ordering
    all_studies = sorted(list(all_studies))
    # sort by study_sizes if given
    if study_sizes:
        all_studies = sorted(all_studies, key=lambda x: study_sizes.get(x, 0))

    # Calculate statistics for each model and study
    stats = {}
    for model_idx, (df, model_name) in enumerate(zip(model_dfs, model_names)):
        stats[model_name] = {}

        for col in df.columns:
            if extract_study_func:
                study = extract_study_func(col)
            else:
                study = col

            # Skip if column doesn't match our pattern
            if not study:
                continue

            # Get values for this model-study combination
            values = df[col].values

            # # Calculate mean and standard deviation
            # mean = np.mean(values)
            # std = np.std(values)

            # Store statistics
            if study not in stats[model_name]:
                stats[model_name][study] = {
                    # 'means': [mean],
                    # 'stds': [std],
                    "values": [values]
                }
            else:
                raise ValueError(
                    f"Duplicate study '{study}' found for model '{model_name}'. Please ensure unique study names."
                )
                # Append if multiple columns map to the same study
                # stats[model_name][study]['means'].append(mean)
                # stats[model_name][study]['stds'].append(std)
                stats[model_name][study]["values"].append(values)

    # # For each study, get or compute final statistics (average if multiple entries)
    # for model_name in stats:
    #     for study in stats[model_name]:
    #         if len(stats[model_name][study]['means']) > 1:
    #             raise ValueError(
    #                 f"Multiple entries found for study '{study}' in model '{model_name}'. Please ensure unique study names."
    #             )
    #             # Average if multiple columns for same study
    #             stats[model_name][study]['mean'] = np.mean(stats[model_name][study]['means'])
    #             stats[model_name][study]['std'] = np.mean(stats[model_name][study]['stds'])
    #         else:
    #             # Use single value
    #             stats[model_name][study]['mean'] = stats[model_name][study]['means'][0]
    #             stats[model_name][study]['std'] = stats[model_name][study]['stds'][0]

    # Position tracking for bars
    group_width = 0.8  # Width allocated for each study group
    model_width = group_width / len(
        model_names
    )  # Width for each model's bar within group
    study_positions = np.arange(len(all_studies))

    # Track min/max for y-axis scaling if not provided
    all_means = []
    all_errors = []

    # For each model, add bars for all studies
    for model_idx, model_name in enumerate(model_names):
        # means = []
        # stds = []

        # Collect data points for this model across all studies
        for study_idx, study in enumerate(all_studies):
            # Calculate position for this box
            position = (
                study_positions[study_idx]
                + (model_idx - len(model_names) / 2 + 0.5) * model_width
            )

            # Get values if study exists for this model
            if study in stats[model_name]:
                # For boxplots, we need the raw values
                values = np.concatenate(stats[model_name][study]["values"])
                values = values[values != 0]

                line_width = get_line_width(model_name)
                opacity = get_color_adjustment(model_name)

                # Add box plot
                fig.add_trace(
                    go.Box(
                        x=[position] * len(values),
                        y=values,
                        name=model_name,
                        legendgroup=model_name,
                        marker_color=get_method_color(model_name),
                        opacity={1: 0.4, 3: 0.6, 5: 0.8, 7: 1.0}[line_width],
                        # line=dict(width=line_width),
                        width=model_width * bar_width_factor,
                        showlegend=study_idx == 19,  # Only show in legend for one study
                        hovertemplate=f"{model_name}<br>{study}<extra></extra>",
                    )
                )
        #     if study in stats[model_name]:
        #         mean = stats[model_name][study]['mean']
        #         std = stats[model_name][study]['std']
        #     else:
        #         # Study not found for this model - use NaN
        #         mean = np.nan
        #         std = np.nan

        #     means.append(mean)
        #     stds.append(std)

        #     # Track for y-axis scaling
        #     if not np.isnan(mean):
        #         all_means.append(mean)
        #         all_errors.append(std)

        # # Calculate positions for this model's bars across all studies
        # positions = study_positions + (model_idx - len(model_names) / 2 + 0.5) * model_width

        # # Add bars for this model
        # fig.add_trace(
        #     go.Bar(
        #         x=positions,
        #         y=means,
        #         error_y=dict(type="data", array=stds, visible=True),
        #         name=model_name,
        #         legendgroup=model_name,
        #         marker_color=color_palette[model_idx % len(color_palette)],
        #         width=model_width * bar_width_factor,  # Slightly narrower than allocated space
        #         hovertemplate=f"{model_name}<br>%{{x}}<br>Mean: %{{y:.3f}}<br>Std: %{{error_y.array:.3f}}<extra></extra>",
        #     )
        # )

        # # Add annotations for mean ± std if enabled
        # if show_annotations:
        #     for i, (pos, mean, std) in enumerate(zip(positions, means, stds)):
        #         if not np.isnan(mean):
        #             fig.add_annotation(
        #                 x=pos,
        #                 y=mean + std + annotation_offset,
        #                 text=f"{mean:.2f}<br>±<br>{std:.2f}",
        #                 showarrow=False,
        #                 font=dict(size=10, color="black"),
        #             )

    # Set y-axis range if not specified
    if y_range is None and all_means:
        y_min = min(0, min(all_means) - max(all_errors) - 0.05)
        y_max = max(all_means) + max(all_errors) + 0.15
        y_range = [y_min, y_max]

    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(weight="bold", size=16), x=0.5, y=0.99)
        if title
        else None,
        xaxis=dict(
            tickvals=study_positions,
            ticktext=all_studies,
            title="Study",
            title_font=dict(size=32),
            tickfont=dict(size=30),
            tickangle=45,
        ),
        yaxis=dict(
            title=y_axis_title,
            title_font=dict(size=32),
            tickfont=dict(size=30),
            range=y_range,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=18),
            title=dict(text="Model", font=dict(size=20)),
            bordercolor="White",
        ),
        height=height,
        width=width,
        margin=dict(b=200, t=150, l=100, r=50),
        # template="plotly_white",
    )

    # Save if path provided
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path, scale=2)

    # Show the figure
    fig.show()

    return fig, stats


def plot_method_comparison_scatter(
    df1,
    df2,
    method1_name,
    method2_name,
    study_sizes_dict,
    study_disease_dict,
    study_balance_dict=None,
    title=None,
    save_path=None,
    height=1000,
    width=1000,
    legend_title=None,
    show_legend=True,
):
    """
    Create scatter plot comparing two methods across studies.

    Parameters:
    -----------
    df1, df2 : pandas.DataFrame
        DataFrames with test_study_name as columns, CV loop scores as rows
    method1_name, method2_name : str
        Names of the two methods (for axis labels)
    study_sizes_dict : dict
        {study_name: size} - determines point sizes
    study_disease_dict : dict
        {disease_name: [study_list]} - determines point colors
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
    """

    # replace zero values with NaN to avoid plotting them
    df1 = df1.replace(0, np.nan)
    df2 = df2.replace(0, np.nan)

    # Calculate mean scores for each study (across CV folds)
    means1 = df1.mean(axis=0)
    means2 = df2.mean(axis=0)

    # Convert disease_dict from {disease: [studies]} to {study: disease}
    study_to_disease = {}
    for disease, studies in study_disease_dict.items():
        for study in studies:
            study_to_disease[study] = disease

    # Get common studies between both dataframes
    common_studies = set(means1.index) & set(means2.index)
    # check size of common studies with each df
    print(f"Common studies: {len(common_studies)}")
    print(f"Method 1 studies: {len(means1.index)}")
    print(f"Method 2 studies: {len(means2.index)}")

    # Prepare data for plotting
    plot_data = []
    max_balance = max(study_balance_dict.values()) if study_balance_dict else 1.0
    for study in common_studies:
        opacity_value = (
            min(1, study_balance_dict[study]) * 0.9 if study_balance_dict else 0.8
        )
        border_width = (
            ((study_balance_dict[study] - 1) / (max_balance - 1)) * 100
            if study_balance_dict and study_balance_dict[study] > 1
            else 0
        )
        plot_data.append(
            {
                "study": study,
                "method1_score": means1[study],
                "method2_score": means2[study],
                "size": study_sizes_dict[study],  # Default size if not found
                "disease": study_to_disease[study] if study_to_disease else "",
                "balance": study_balance_dict[study] if study_balance_dict else None,
                "opacity": 0.6,
                "border_width": border_width,
            }
        )

    # Convert to DataFrame and sort by size (largest first for proper layering)
    plot_df = pd.DataFrame(plot_data)
    plot_df = plot_df.sort_values("size", ascending=False)

    # Create scatter plot
    fig = px.scatter(
        plot_df,
        x="method1_score",
        y="method2_score",
        size="size",
        color="disease" if study_to_disease else None,
        hover_name="study",
        hover_data={"size": True, "disease": True, "balance": True},
        title=title,
        labels={"method1_score": method1_name, "method2_score": method2_name},
        size_max=20,  # Maximum point size
        # custom_data=["study", "size", "disease", "balance", "opacity", "border_width"],
    )

    # Add diagonal equality line
    min_val = 0
    max_val = 1
    min_val = min(plot_df["method1_score"].min(), plot_df["method2_score"].min())
    max_val = max(plot_df["method1_score"].max(), plot_df["method2_score"].max())

    # Extend line slightly beyond data range
    line_min = min_val - 0.05  # min_val - 0.05 * (max_val - min_val)
    line_max = max_val + 0.05  # max_val + 0.05 * (max_val - min_val)
    # round to first decimal
    line_min = math.floor(line_min * 10) / 10
    line_max = math.ceil(line_max * 10) / 10

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray", width=2),
            name="Equal performance",
            showlegend=show_legend,
        )
    )

    # Update traces for opacity and better visibility
    # opacities = []
    # line_widths = []
    # for data in fig.data:
    #     print(data.customdata)
    #     if data.customdata is None:
    #         continue
    #     opacities.extend(list(data.customdata[:, -2]))
    #     line_widths.extend(list(data.customdata[:, -1]))

    # fig.update_traces(
    #     selector=dict(mode="markers"),
    #     marker=dict(
    #         opacity=opacities,  # Use custom_data
    #         line=dict(width=line_widths, color="black"),
    #     ),
    # )

    # Update traces for opacity and better visibility
    # fig.update_traces(
    #     marker=dict(
    #         opacity=plot_df["balance"].tolist(),  # Set opacity here
    #         line=dict(width=0.5, color="white"),
    #     )
    # )
    # fig.update_traces(
    #     selector=dict(mode='markers'),
    #     opacity=0.98,
    #     marker=dict(
    #         line=dict(width=0.5, color='white')  # White border around points
    #     )
    # )

    # Update layout
    fig.update_layout(
        width=width,
        height=height,
        showlegend=show_legend,
        title=dict(
            text=title,
            font=dict(size=24, weight="bold"),
            y=0.99,  # Position title at the top
        )
        if title
        else None,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,  # Inside plot area
            bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent background
            font=dict(size=23),
            title=dict(text=legend_title, font=dict(size=20)),
        )
        if show_legend
        else None,
        margin=dict(l=100, r=100, t=100, b=100),  # Extra margin for legend
        xaxis=dict(
            title=dict(text=method1_name, font=dict(size=32), standoff=25),
            tickfont=dict(size=30),
            range=[0, 1],  # Set range to [0, 1] for better comparison
        ),
        yaxis=dict(
            title=dict(text=method2_name, font=dict(size=32), standoff=5),
            tickfont=dict(size=30),
            # scaleanchor="x",  # Ensure equal scaling on both axes
            # scaleratio=1, # Keep aspect ratio square
            range=[0, 1],  # Set range to [0, 1] for better comparison
        ),
    )

    # Ensure square aspect ratio for better comparison
    # fig.update_yaxis(scaleanchor="x", scaleratio=1)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
    fig.show()

    # scale_factor = 0.3
    # display_fig = go.Figure(fig)
    # display_fig.update_layout(
    #     width=int(width * scale_factor),
    #     height=int(height * scale_factor),
    #     font=dict(size=int(20 * scale_factor)),
    #     xaxis=dict(
    #         title=dict(font=dict(size=int(26 * scale_factor))),
    #         tickfont=dict(size=int(23 * scale_factor)),
    #     ),
    #     yaxis=dict(
    #         title=dict(font=dict(size=int(26 * scale_factor))),
    #         tickfont=dict(size=int(23 * scale_factor)),
    #     ),
    #     showlegend=False,
    #     title=dict(
    #         font=dict(size=int(18 * scale_factor), weight="bold"),
    #     )
    #     if title
    #     else None,
    # )
    # display_fig.update_traces(
    #     marker=dict(size=[s * scale_factor * 0.2 for s in plot_df["size"]])
    # )
    # display_fig.show()

    return fig


def cluster_metrics(
    support: np.ndarray,
    query: np.ndarray,
    support_y: np.ndarray,
    query_y: np.ndarray,
    metric: str = "euclidean",
) -> dict[str, float | np.ndarray]:
    """Silhouette, dispersion ratio and prototype→query distance quartiles."""
    X = np.vstack([support, query])
    y = np.concatenate([support_y, query_y])
    k = len(np.unique(y))

    sil = silhouette_score(X, y, metric=metric) if k > 1 else np.nan

    # dispersion ratio: mean intra-class ÷ mean inter-class distance
    intra, inter = [], []
    for lbl in np.unique(y):
        idx = y == lbl
        d = pairwise_distances(X[idx], metric=metric)
        intra.extend(d[np.triu_indices_from(d, k=1)])
    for i, lbl1 in enumerate(np.unique(y)):
        for lbl2 in np.unique(y)[i + 1 :]:
            inter.extend(
                pairwise_distances(X[y == lbl1], X[y == lbl2], metric=metric).ravel()
            )
    disp = np.mean(intra) / np.mean(inter) if inter else np.nan

    # prototype → query distance percentiles (25/50/75)
    protos = {
        lbl: support[support_y == lbl].mean(axis=0) for lbl in np.unique(support_y)
    }
    d_qp = [
        pairwise_distances(
            query[i : i + 1], protos[q_lbl].reshape(1, -1), metric=metric
        )[0, 0]
        for i, q_lbl in enumerate(query_y)
    ]
    pct_25, pct_50, pct_75 = np.percentile(d_qp, [25, 50, 75])

    # get rounded int
    return {
        "silhouette": sil.round(3).item(),
        "dispersion_ratio": disp.round(3).item(),
        "proto-query_p25": pct_25.round(3).item(),
        "proto-query_p50": pct_50.round(3).item(),
        "proto-query_p75": pct_75.round(3).item(),
    }


def tw_cont(
    X_high: np.ndarray,
    X_low: np.ndarray,
    n_neighbors: int = 10,
    metric: str = "euclidean",
) -> tuple[float, float]:
    """
    Returns (trustworthiness, continuity) for a single embedding.
    Trustworthiness uses sklearn; continuity is implemented per Venna & Kaski (2001).
    """
    # --- trustworthiness (false-positives) ---
    tw = trustworthiness(X_high, X_low, n_neighbors=n_neighbors, metric=metric)

    # --- continuity (false-negatives) ---
    n = X_high.shape[0]
    nn_high = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric).fit(X_high)
    nn_low = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric).fit(X_low)
    high_idx = nn_high.kneighbors(return_distance=False)[:, 1:]
    low_idx = nn_low.kneighbors(return_distance=False)[:, 1:]

    ranks_high = np.argsort(
        np.argsort(pairwise_distances(X_high, metric=metric), axis=1), axis=1
    )  # rank matrix

    c_sum = 0.0
    for i in range(n):
        for j in low_idx[i]:
            r = ranks_high[i, j]
            if r >= n_neighbors:
                c_sum += r - n_neighbors + 1
    denom = n * n_neighbors * (2 * n - 3 * n_neighbors - 1)
    cont = 1 - (2 * c_sum) / denom
    return dict(
        trustworthiness=tw.round(3).item(),
        continuity=cont.round(3).item(),
        lcmc=((tw + cont) / 2).round(3).item(),
    )  # simple LCMC variant


def plot_umap_interactive(
    support_data,
    query_data,
    support_labels,
    query_labels,
    title,
    n_components,
    unique_labels,
    colors,
    metric,
    study,
    embedding_or_original,
    f1_score,
    disease,
    n_neighbors: int | None = None,
    factor: int = 1,
    min_dist: float = 0.1,
):
    print(f1_score)
    cluster_metrics_dict = cluster_metrics(
        support_data,
        query_data,
        support_labels,
        query_labels,
    )
    print(cluster_metrics_dict)

    data = np.concatenate([support_data, query_data], axis=0)
    labels = np.concatenate([support_labels, query_labels], axis=0)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    if n_neighbors is None:
        n_neighbors = int(np.ceil(np.sqrt(data.shape[0])) * factor)
        # print(f"n_neighbors set to {n_neighbors} based on data size {data.shape[0]}")

    umapper = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    transformed = umapper.fit_transform(data)

    trustworthiness_continuity_res = tw_cont(
        data, transformed, n_neighbors=n_neighbors, metric=metric
    )
    print(trustworthiness_continuity_res)
    title += f"| {study} | {disease} | F1: {f1_score:.3f}<br>{cluster_metrics_dict.__str__()}<br>{trustworthiness_continuity_res.__str__()}"

    n_support = len(support_data)
    support_transformed = transformed[:n_support]
    query_transformed = transformed[n_support:]

    # Create plotly figure
    fig = go.Figure()

    for i, label in enumerate(unique_labels):
        # indices = np.where(labels == label)
        # coords = transformed[indices]

        # Support points (circles)
        support_indices = np.where(support_labels == label)
        support_coords = support_transformed[support_indices]
        # class prototype
        support_prototype = support_coords.mean(axis=0, keepdims=True)

        # Query points (diamonds)
        query_indices = np.where(query_labels == label)
        query_coords = query_transformed[query_indices]

        if n_components == 3:
            # Support points
            fig.add_trace(
                go.Scatter3d(
                    x=support_coords[:, 0],
                    y=support_coords[:, 1],
                    z=support_coords[:, 2],
                    mode="markers",
                    marker=dict(size=20, color=colors[i], opacity=0.6, symbol="circle"),
                    name=f"Support - {label}",
                    text=[
                        f"Support - Label: {label}" for _ in range(len(support_coords))
                    ],
                    hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
                )
            )

            # Query points
            fig.add_trace(
                go.Scatter3d(
                    x=query_coords[:, 0],
                    y=query_coords[:, 1],
                    z=query_coords[:, 2],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=colors[i],
                        opacity=0.7,
                        symbol="diamond",
                        line=dict(width=1, color="black"),
                    ),
                    name=f"Query - {label}",
                    text=[f"Query - Label: {label}" for _ in range(len(query_coords))],
                    hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=support_coords[:, 0],
                    y=support_coords[:, 1],
                    mode="markers",
                    marker=dict(
                        size=20,
                        color=colors[i],
                        opacity=0.6,
                        symbol="circle",
                        line=dict(width=2, color="black"),
                    ),
                    name=f"Support - {label}",
                    text=[
                        f"Support - Label: {label}" for _ in range(len(support_coords))
                    ],
                    hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=query_coords[:, 0],
                    y=query_coords[:, 1],
                    mode="markers",
                    marker=dict(
                        size=13,
                        color=colors[i],
                        opacity=0.7,
                        symbol="diamond",
                    ),
                    name=f"Query - {label}",
                    text=[f"Query - Label: {label}" for _ in range(len(query_coords))],
                    hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                )
            )

            # add class prototypes as stars
            fig.add_trace(
                go.Scatter(
                    x=support_prototype[:, 0],
                    y=support_prototype[:, 1],
                    mode="markers",
                    marker=dict(
                        size=18,
                        color=colors[i],
                        symbol="star",
                        line=dict(width=2, color="black"),
                    ),
                    name=f"Prototype - {label}",
                    text=[
                        f"Prototype - Label: {label}"
                        for _ in range(len(support_prototype))
                    ],
                    hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                )
            )

    # Update layout
    if n_components == 3:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=13, weight="bold"),
                y=0.99,  # Position title at the top
            )
            if title
            else None,
            legend=dict(
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.02,  # Inside plot area
                bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent background
                font=dict(size=15),
            ),
            scene=dict(
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            width=1000,
            height=1000,
            margin=dict(l=100, r=100, t=100, b=100),  # Extra margin for legend
            xaxis=dict(
                title=dict(text="UMAP 1", font=dict(size=32), standoff=25),
                tickfont=dict(size=30),
            ),
            yaxis=dict(
                title=dict(text="UMAP 2", font=dict(size=32), standoff=5),
                tickfont=dict(size=30),
            ),
            zaxis=dict(
                title=dict(text="UMAP 3", font=dict(size=32), standoff=5),
                tickfont=dict(size=30),
            ),
            showlegend=False,
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    showticklabels=False,
                    ticks="",
                    showgrid=False,
                    zeroline=False,
                    showline=True,
                ),
                yaxis=dict(
                    showticklabels=False,
                    ticks="",
                    showgrid=False,
                    zeroline=False,
                    showline=True,
                ),
                zaxis=dict(
                    showticklabels=False,
                    ticks="",
                    showgrid=False,
                    zeroline=False,
                    showline=True,
                ),
            )
        )
    else:
        fig.update_layout(
            width=1000,
            height=1000,
            showlegend=False,
            title=dict(
                text=title,
                font=dict(size=13, weight="bold"),
                y=0.99,  # Position title at the top
            )
            if title
            else None,
            legend=dict(
                yanchor="top",
                y=0.90,
                xanchor="left",
                x=0.02,  # Inside plot area
                bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent background
                font=dict(size=18),
            ),
            margin=dict(l=100, r=100, t=100, b=100),  # Extra margin for legend
            xaxis=dict(
                title=dict(text="UMAP 1", font=dict(size=32), standoff=25),
                tickfont=dict(size=30),
            ),
            yaxis=dict(
                title=dict(text="UMAP 2", font=dict(size=32), standoff=5),
                tickfont=dict(size=30),
            ),
        )

        fig.update_xaxes(
            showticklabels=False,
            ticks="",
            showgrid=False,
            zeroline=False,
            showline=True,
            title=None,
        )
        fig.update_yaxes(
            showticklabels=False,
            ticks="",
            showgrid=False,
            zeroline=False,
            showline=True,
            title=None,
        )

        metrics_dict = {
            "F1 Score": f1_score,
            "Disease": disease,
            **cluster_metrics_dict,
            **trustworthiness_continuity_res,
        }

        save_path = f"./all_studies_umaps/{factor}_{min_dist}/{study}_support_query_{embedding_or_original}_umap.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path:
            fig.write_image(save_path)
            with open(f"./all_studies_umaps/{factor}_{min_dist}/metrics.txt", "a") as f:
                f.write(f"{study} | {embedding_or_original} | {metrics_dict}\n")
        print(f"UMAP saved for {study}")
    return fig


# Function to plot support and query embeddings with different markers
def plot_umap_support_query_separate(
    support_data,
    query_data,
    support_labels,
    query_labels,
    title,
    n_components,
    unique_labels,
    metric,
    n_neighbors: int | None = None,
    min_dist: float = 0.05,
    colors=None,
):
    # ─── NEW: z-score on support then reuse for query ───
    scaler = StandardScaler()
    support_data = scaler.fit_transform(support_data)
    query_data = scaler.transform(query_data)

    # ─── NEW: √N based on total points if not given ───
    if n_neighbors is None:
        total_n = support_data.shape[0] + query_data.shape[0]
        n_neighbors = int(np.ceil(np.sqrt(total_n)))
        print(f"n_neighbors set to {n_neighbors} based on total data size {total_n}")

    fig = go.Figure()

    umapper = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )

    # Transform support embeddings
    support_transformed = umapper.fit_transform(support_data)
    # Transform query embeddings using the same fitted UMAP
    query_transformed = umapper.transform(query_data)
    # but also fit_transform to ensure same scale
    umapper_separate = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    query_transformed_separate = umapper_separate.fit_transform(query_data)

    colors = colors or px.colors.qualitative.Set1[: len(unique_labels)]

    # Add support points
    for j, label in enumerate(unique_labels):
        support_indices = np.where(support_labels == label)
        support_coords = support_transformed[support_indices]

        if n_components == 3:
            fig.add_trace(
                go.Scatter3d(
                    x=support_coords[:, 0],
                    y=support_coords[:, 1],
                    z=support_coords[:, 2],
                    mode="markers",
                    marker=dict(size=6, color=colors[j], symbol="circle", opacity=0.8),
                    name=f"Support - {label}",
                    text=[
                        f"Support - Label: {label}" for _ in range(len(support_coords))
                    ],
                    hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=support_coords[:, 0],
                    y=support_coords[:, 1],
                    mode="markers",
                    marker=dict(size=8, color=colors[j], symbol="circle", opacity=0.8),
                    name=f"Support - {label}",
                    text=[
                        f"Support - Label: {label}" for _ in range(len(support_coords))
                    ],
                    hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                )
            )

    # Add query points
    for j, label in enumerate(unique_labels):
        query_indices = np.where(query_labels == label)
        query_coords = query_transformed[query_indices]

        if n_components == 3:
            fig.add_trace(
                go.Scatter3d(
                    x=query_coords[:, 0],
                    y=query_coords[:, 1],
                    z=query_coords[:, 2],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=colors[j],
                        symbol="diamond",
                        opacity=0.8,
                        line=dict(width=2, color="black"),
                    ),
                    name=f"Query - {label}",
                    text=[f"Query - Label: {label}" for _ in range(len(query_coords))],
                    hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=query_coords[:, 0],
                    y=query_coords[:, 1],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=colors[j],
                        symbol="diamond",
                        opacity=0.8,
                        line=dict(width=2, color="black"),
                    ),
                    name=f"Query - {label}",
                    text=[f"Query - Label: {label}" for _ in range(len(query_coords))],
                    hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                )
            )

    # Create separate figure for query points with different markers
    fig_query = go.Figure()
    for j, label in enumerate(unique_labels):
        query_indices = np.where(query_labels == label)
        query_coords = query_transformed_separate[query_indices]

        if n_components == 3:
            fig_query.add_trace(
                go.Scatter3d(
                    x=query_coords[:, 0],
                    y=query_coords[:, 1],
                    z=query_coords[:, 2],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=colors[j],
                        symbol="diamond",
                        opacity=0.8,
                        line=dict(width=2, color="black"),
                    ),
                    name=f"Query - {label}",
                    text=[f"Query - Label: {label}" for _ in range(len(query_coords))],
                    hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
                )
            )
        else:
            fig_query.add_trace(
                go.Scatter(
                    x=query_coords[:, 0],
                    y=query_coords[:, 1],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=colors[j],
                        symbol="diamond",
                        opacity=0.8,
                        line=dict(width=2, color="black"),
                    ),
                    name=f"Query - {label}",
                    text=[f"Query - Label: {label}" for _ in range(len(query_coords))],
                    hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                )
            )

    # Update layout
    if n_components == 3:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=24, weight="bold"),
                y=0.99,  # Position title at the top
            )
            if title
            else None,
            legend=dict(
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.02,  # Inside plot area
                bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent background
                font=dict(size=18),
            ),
            scene=dict(
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            width=1000,
            height=1000,
            margin=dict(l=100, r=100, t=100, b=100),  # Extra margin for legend
            xaxis=dict(
                title=dict(text="UMAP 1", font=dict(size=32), standoff=25),
                tickfont=dict(size=30),
            ),
            yaxis=dict(
                title=dict(text="UMAP 2", font=dict(size=32), standoff=5),
                tickfont=dict(size=30),
            ),
            zaxis=dict(
                title=dict(text="UMAP 3", font=dict(size=32), standoff=5),
                tickfont=dict(size=30),
            ),
            showlegend=True,
        )
        fig_query.update_layout(
            title=dict(
                text=title,
                font=dict(size=24, weight="bold"),
                y=0.99,  # Position title at the top
            )
            if title
            else None,
            legend=dict(
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.02,  # Inside plot area
                bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent background
                font=dict(size=18),
            ),
            scene=dict(
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            width=1000,
            height=1000,
            margin=dict(l=100, r=100, t=100, b=100),  # Extra margin for legend
            xaxis=dict(
                title=dict(text="UMAP 1", font=dict(size=32), standoff=25),
                tickfont=dict(size=30),
            ),
            yaxis=dict(
                title=dict(text="UMAP 2", font=dict(size=32), standoff=5),
                tickfont=dict(size=30),
            ),
            zaxis=dict(
                title=dict(text="UMAP 3", font=dict(size=32), standoff=5),
                tickfont=dict(size=30),
            ),
            showlegend=True,
        )

    else:
        fig.update_layout(
            width=1000,
            height=1000,
            showlegend=True,
            title=dict(
                text=title,
                font=dict(size=18, weight="bold"),
                y=0.99,  # Position title at the top
            )
            if title
            else None,
            legend=dict(
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.02,  # Inside plot area
                bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent background
                font=dict(size=18),
            ),
            margin=dict(l=100, r=100, t=100, b=100),  # Extra margin for legend
            xaxis=dict(
                title=dict(text="UMAP 1", font=dict(size=32), standoff=25),
                tickfont=dict(size=30),
            ),
            yaxis=dict(
                title=dict(text="UMAP 2", font=dict(size=32), standoff=5),
                tickfont=dict(size=30),
            ),
        )
        fig_query.update_layout(
            width=1000,
            height=1000,
            showlegend=True,
            title=dict(
                text=title,
                font=dict(size=18, weight="bold"),
                y=0.99,  # Position title at the top
            )
            if title
            else None,
            legend=dict(
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.02,  # Inside plot area
                bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent background
                font=dict(size=18),
            ),
            margin=dict(l=100, r=100, t=100, b=100),  # Extra margin for legend
            xaxis=dict(
                title=dict(text="UMAP 1", font=dict(size=32), standoff=25),
                tickfont=dict(size=30),
            ),
            yaxis=dict(
                title=dict(text="UMAP 2", font=dict(size=32), standoff=5),
                tickfont=dict(size=30),
            ),
        )

    return fig, fig_query


# def plot_study_agnostic_features(
#     all_studies_embeddings,
#     all_studies_labels,
#     study_identifiers,
#     title="Study-Agnostic Feature Analysis",
#     n_components=3,
#     metric="euclidean",
#     n_neighbors=120,
#     min_dist=0.05,
#     alpha=0.6,
# ):
#     """
#     Plot embeddings colored by study ID to assess study-agnostic learning.

#     Parameters:
#     - all_studies_embeddings: List of embedding arrays, one per study
#     - all_studies_labels: List of label arrays, one per study
#     - study_identifiers: List of study names/IDs
#     - alpha: Transparency level (0-1) to highlight density
#     """
#     fig = go.Figure()

#     # Combine all embeddings and create study labels
#     combined_embeddings = []
#     combined_study_ids = []

#     for i, (embeddings, study_id) in enumerate(
#         zip(all_studies_embeddings, study_identifiers)
#     ):
#         combined_embeddings.append(embeddings)
#         combined_study_ids.extend([study_id] * len(embeddings))

#     combined_embeddings = np.vstack(combined_embeddings)
#     combined_study_ids = np.array(combined_study_ids)

#     # Fit UMAP on combined data (ignoring class labels)
#     umapper = UMAP(
#         n_components=n_components,
#         n_neighbors=n_neighbors,
#         min_dist=min_dist,
#         metric=metric,
#         random_state=42,
#     )

#     transformed_embeddings = umapper.fit_transform(combined_embeddings)

#     # Get unique studies and assign colors
#     unique_studies = np.unique(combined_study_ids)
#     colors = px.colors.qualitative.Set1[: len(unique_studies)]

#     # Plot points colored by study
#     for i, study in enumerate(unique_studies):
#         study_indices = np.where(combined_study_ids == study)
#         study_coords = transformed_embeddings[study_indices]

#         if n_components == 3:
#             fig.add_trace(
#                 go.Scatter3d(
#                     x=study_coords[:, 0],
#                     y=study_coords[:, 1],
#                     z=study_coords[:, 2],
#                     mode="markers",
#                     marker=dict(
#                         size=4,
#                         color=colors[i],
#                         opacity=alpha,
#                         line=dict(width=0.5, color="white"),
#                     ),
#                     name=f"Study: {study}",
#                     text=[f"Study: {study}" for _ in range(len(study_coords))],
#                     hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
#                 )
#             )
#         else:
#             fig.add_trace(
#                 go.Scatter(
#                     x=study_coords[:, 0],
#                     y=study_coords[:, 1],
#                     mode="markers",
#                     marker=dict(
#                         size=6,
#                         color=colors[i],
#                         opacity=alpha,
#                         line=dict(width=0.5, color="white"),
#                     ),
#                     name=f"Study: {study}",
#                     text=[f"Study: {study}" for _ in range(len(study_coords))],
#                     hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
#                 )
#             )

#     # Update layout
#     if n_components == 3:
#         fig.update_layout(
#             title=f"{title}<br><sub>Well-mixed colors = study-agnostic features; Isolated clusters = batch effects</sub>",
#             scene=dict(
#                 xaxis_title="UMAP 1",
#                 yaxis_title="UMAP 2",
#                 zaxis_title="UMAP 3",
#                 camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
#             ),
#             width=900,
#             height=700,
#             showlegend=True,
#         )
#     else:
#         fig.update_layout(
#             title=f"{title}<br><sub>Well-mixed colors = study-agnostic features; Isolated clusters = batch effects</sub>",
#             xaxis_title="UMAP 1",
#             yaxis_title="UMAP 2",
#             width=900,
#             height=700,
#             showlegend=True,
#         )

#     return fig


def f1_vs_jaccard_sim_scatter_plot(jaccard_df, f1_df):
    merged_df = jaccard_df.merge(
        f1_df, left_index=True, right_index=True, suffixes=("_jaccard", "_f1")
    )
    merged_df = merged_df[["Jaccard Similarity Against Union", "F1 Score"]].dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged_df["Jaccard Similarity Against Union"],
            y=merged_df["F1 Score"],
            mode="markers",
            marker=dict(size=10, color="blue", opacity=0.7),
        )
    )
    fig.update_layout(
        title="F1 Score vs Jaccard Similarity",
        xaxis_title="Jaccard Similarity",
        yaxis_title="F1 Score",
        showlegend=False,
    )
    return fig


def create_jaccard_heatmap_with_f1_table(
    jaccard_df, df1, df2, RF_df, jaccard_union_df, f1_column="F1 Score"
):
    """
    Create a heatmap with F1 scores displayed in a clean table format.
    """
    # Get unique studies
    studies = sorted(
        set(jaccard_df["Study1"].unique()) | set(jaccard_df["Study2"].unique())
    )
    n_studies = len(studies)

    # Create similarity matrix
    similarity_matrix = pd.DataFrame(
        np.zeros((n_studies, n_studies)), index=studies, columns=studies
    )

    for _, row in jaccard_df.iterrows():
        study1, study2 = row["Study1"], row["Study2"]
        similarity = row["Jaccard Similarity"]
        similarity_matrix.loc[study1, study2] = similarity
        similarity_matrix.loc[study2, study1] = similarity

    np.fill_diagonal(similarity_matrix.values, 1.0)

    # # sort based on union values
    studies_sorted = jaccard_union_df.sort_values(
        by="Jaccard Similarity Against Union", ascending=False
    ).index.tolist()

    # Calculate sum of similarities for each study and sort
    # similarity_sums = similarity_matrix.sum(axis=1)
    # studies_sorted = similarity_sums.sort_values(ascending=False).index.tolist()

    # Reorder the similarity matrix
    similarity_matrix = similarity_matrix.loc[studies_sorted, studies_sorted]

    # Prepare F1 data and colors (using sorted order)
    f1_df1_vals = []
    f1_df2_vals = []
    f1_rf_vals = []
    f1_df1_colors = []
    f1_df2_colors = []
    f1_rf_colors = []
    diseases = []

    for study in studies_sorted:
        f1_1 = df1.loc[study, f1_column] if study in df1.index else np.nan
        f1_2 = df2.loc[study, f1_column] if study in df2.index else np.nan
        f1_rf = RF_df.loc[study, f1_column] if study in RF_df.index else np.nan
        f1_df1_vals.append(f1_1)
        f1_df2_vals.append(f1_2)
        f1_rf_vals.append(f1_rf)
        diseases.append(df2.loc[study, "Disease"] if study in df2.index else "N/A")

        # Determine colors based on which value is higher
        if np.isnan(f1_1) or np.isnan(f1_2):
            f1_df1_colors.append("lightgray")
            f1_df2_colors.append("lightgray")
        elif f1_1 > f1_2:
            f1_df1_colors.append("#FFFFCC")  # Light green
            f1_df2_colors.append("#FFFFCC")  # Light red
        elif f1_2 > f1_1:
            f1_df1_colors.append("#FFFFCC")  # Light red
            f1_df2_colors.append("#FFFFCC")  # Light green
        else:
            f1_df1_colors.append("#FFFFCC")  # Light yellow for equal
            f1_df2_colors.append("#FFFFCC")
        f1_rf_colors.append("#FFFFCC")

    from plotly.subplots import make_subplots

    # Create figure with heatmap and table
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.64, 0.353],
        specs=[[{"type": "heatmap"}, {"type": "table"}]],
        horizontal_spacing=0.07,
    )

    # Add heatmap with better colorscale for small differences
    fig.add_trace(
        go.Heatmap(
            z=similarity_matrix.values,
            x=studies_sorted,
            y=studies_sorted,
            colorscale="Greys",
            zmin=similarity_matrix.values[similarity_matrix.values < 1].min() - 0.05,
            zmax=1.0,
            colorbar=dict(title="Similarity", x=1.03, len=0.8, y=0.5),
            hovertemplate="Study1: %{y}<br>"
            + "Study2: %{x}<br>"
            + "Jaccard: %{z:.3f}<br>"
            + "<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add table for F1 scores
    fig.add_trace(
        go.Table(
            columnwidth=[50, 50, 50, 200],
            header=dict(values=["", ""], height=0),
            cells=dict(
                values=[
                    [f"{v:.3f}" if not np.isnan(v) else "N/A" for v in f1_df1_vals],
                    [f"{v:.3f}" if not np.isnan(v) else "N/A" for v in f1_df2_vals],
                    [f"{v:.3f}" if not np.isnan(v) else "N/A" for v in f1_rf_vals],
                    [f"{disease}" for disease in diseases],
                ],
                align="center",
                font=dict(size=18),
                fill_color=[f1_df1_colors, f1_df2_colors, f1_rf_colors, "#FFFFCC"],
                line_color="darkgray",
                height=(610 + len(studies_sorted) * 15 - 100) / n_studies,
            ),
        ),
        row=1,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title={
            "text": "Jaccard Similarity Between Studies with F1 Scores",
            "font": {"size": 22},
            "x": 0.5,
            "xanchor": "center",
        },
        width=2030,
        height=800 + len(studies_sorted) * 15,
        showlegend=False,
        margin=dict(l=190, r=50, t=100, b=190),
    )

    # Update axes
    # fig.update_xaxes(, row=1, col=1)
    fig.update_xaxes(
        tickangle=-45,
        row=1,
        col=1,
        scaleanchor="y",
        scaleratio=1,
    )

    # Add sum values as secondary y-axis
    fig.update_yaxes(
        ticktext=[f"{study}" for study in studies_sorted],
        tickvals=list(range(len(studies_sorted))),
        row=1,
        col=1,
    )

    for i, study in enumerate(studies_sorted):
        # sum_val = similarity_sums[study]
        union_val = jaccard_union_df.loc[study, "Jaccard Similarity Against Union"]
        # Add annotation for sum value
        fig.add_annotation(
            text=round(union_val, 3),
            xref="x domain",
            yref="y",
            x=1.005,
            y=i,
            showarrow=False,
            font=dict(size=20),
            xanchor="left",
        )

    fig.update_xaxes(title_font=dict(size=19), tickfont=dict(size=24))
    fig.update_yaxes(title_font=dict(size=19), tickfont=dict(size=24))

    # fig.update_layout(
    #     xaxis=dict(
    #         # title_font=dict(size=32),
    #         tickfont=dict(size=12),
    #         tickangle=55,
    #         automargin=True,
    #     ),
    #     yaxis=dict(
    #         # title_font=dict(size=14),
    #         tickfont=dict(size=12),
    #         automargin=True,
    #         # range=[0, 1],
    #     )
    # )

    fig.update()

    return fig


def plot_f1_vs_jaccard_similarity(
    jaccard_df, df1, df2, RF_df, f1_column="F1 Score", save_path=None
):
    """Plot F1 scores against Jaccard similarity sums for each study."""

    # Get unique studies and create similarity matrix (same as heatmap function)
    studies = sorted(
        set(jaccard_df["Study1"].unique()) | set(jaccard_df["Study2"].unique())
    )
    n_studies = len(studies)

    similarity_matrix = pd.DataFrame(
        np.zeros((n_studies, n_studies)), index=studies, columns=studies
    )

    for _, row in jaccard_df.iterrows():
        study1, study2 = row["Study1"], row["Study2"]
        similarity = row["Jaccard Similarity"]
        similarity_matrix.loc[study1, study2] = similarity
        similarity_matrix.loc[study2, study1] = similarity

    np.fill_diagonal(similarity_matrix.values, 1.0)

    # Calculate sum of similarities for each study
    similarity_sums = similarity_matrix.sum(axis=1)

    fig = go.Figure()

    # Define colors and names for each dataset
    datasets = [
        (df1, "Pronet without embeddings", "#1f77b4"),
        (df2, "Protonet with embeddings", "#ff7f0e"),
        (RF_df, "Random Forest", "#2ca02c"),
    ]

    # Plot each dataset
    for df, name, color in datasets:
        # Get studies that exist in both the df and similarity_sums
        common_studies = [s for s in studies if s in df.index]

        if len(common_studies) > 0:
            f1_scores = [df.loc[s, f1_column] for s in common_studies]
            jaccard_sims = [similarity_sums[s] for s in common_studies]

            # Remove NaN values
            valid_pairs = [
                (f1, js, s)
                for f1, js, s in zip(f1_scores, jaccard_sims, common_studies)
                if not np.isnan(f1)
            ]

            if valid_pairs:
                f1_scores = [p[0] for p in valid_pairs]
                jaccard_sims = [p[1] for p in valid_pairs]
                study_labels = [p[2] for p in valid_pairs]

                # Add scatter trace
                fig.add_trace(
                    go.Scatter(
                        x=jaccard_sims,
                        y=f1_scores,
                        mode="markers",
                        name=name,
                        marker=dict(size=12, color=color, opacity=0.6),
                        customdata=study_labels,
                        hovertemplate="Study: %{customdata}<br>"
                        + "Jaccard Sum: %{x:.3f}<br>"
                        + "F1 Score: %{y:.3f}<br>"
                        + "<extra></extra>",
                    )
                )

                # # Add trend line if enough points
                # if len(f1_scores) > 1:
                #     z = np.polyfit(jaccard_sims, f1_scores, 1)
                #     p = np.poly1d(z)
                #     x_trend = np.linspace(min(jaccard_sims), max(jaccard_sims), 100)

                #     fig.add_trace(go.Scatter(
                #         x=x_trend,
                #         y=p(x_trend),
                #         mode='lines',
                #         name=f"{name} trend",
                #         line=dict(color=color, dash='dash'),
                #         showlegend=False
                #     ))

    # Update layout
    fig.update_layout(
        title=dict(
            text="F1 Scores vs Jaccard Similarity Sum",
            x=0.5, y=0.99
        ),
        xaxis_title="Sum of Jaccard Similarities",
        yaxis_title="F1 Score",
        width=1000,
        height=800,
        hovermode="closest",
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        xaxis=dict(tickfont=dict(size=30), title_font=dict(size=32)),
        yaxis=dict(tickfont=dict(size=30), title_font=dict(size=32)),
        margin=dict(l=100, r=100, t=100, b=100),  # Extra margin for legend
    )

    if save_path:
        fig.write_image(save_path, scale=1)

    return fig


def create_jaccard_clustermap_with_f1(
    jaccard_df, df1, df2, RF_df, f1_column="F1 Score"
):
    """Create clustermap with both disease groups and F1 scores as annotations."""
    # Get unique studies
    studies = sorted(
        set(jaccard_df["Study1"].unique()) | set(jaccard_df["Study2"].unique())
    )

    # Create similarity matrix
    similarity_matrix = pd.DataFrame(
        np.zeros((len(studies), len(studies))), index=studies, columns=studies
    )

    for _, row in jaccard_df.iterrows():
        study1, study2 = row["Study1"], row["Study2"]
        similarity = row["Jaccard Similarity"]
        similarity_matrix.loc[study1, study2] = similarity
        similarity_matrix.loc[study2, study1] = similarity

    np.fill_diagonal(similarity_matrix.values, 1.0)

    distance_matrix = 1 - similarity_matrix.values
    condensed_dist = squareform(distance_matrix)

    # Create annotation DataFrame
    annotations = pd.DataFrame(index=studies)

    # Add disease groups
    annotations["Disease"] = [
        df2.loc[study, "Disease"] if study in df2.index else "Unknown"
        for study in studies
    ]

    # Add F1 scores
    annotations["F1_df1"] = [
        df1.loc[study, f1_column] if study in df1.index else np.nan for study in studies
    ]
    annotations["F1_df2"] = [
        df2.loc[study, f1_column] if study in df2.index else np.nan for study in studies
    ]
    annotations["F1_RF"] = [
        RF_df.loc[study, f1_column] if study in RF_df.index else np.nan
        for study in studies
    ]

    # Create color mappings
    from matplotlib.colors import to_rgba

    disease_colors_rgba = {
        disease: to_rgba(color) for disease, color in disease_colors.items()
    }
    disease_lut = {
        disease: disease_colors.get(disease, (0, 0, 0, 1))
        for disease in annotations["Disease"].unique()
    }

    # Create F1 color maps (red-white-green)
    f1_norm = plt.Normalize(vmin=0, vmax=1)
    f1_cmap = plt.cm.RdYlGn

    # Create row colors DataFrame
    row_colors_df = pd.DataFrame(index=studies)
    row_colors_df["Disease"] = annotations["Disease"].map(disease_lut)
    # row_colors_df["F1_df1"] = [
    #     f1_cmap(f1_norm(val)) if not np.isnan(val) else "lightgray"
    #     for val in annotations["F1_df1"]
    # ]
    # row_colors_df["F1_df2"] = [
    #     f1_cmap(f1_norm(val)) if not np.isnan(val) else "lightgray"
    #     for val in annotations["F1_df2"]
    # ]
    # row_colors_df["F1_RF"] = [
    #     f1_cmap(f1_norm(val)) if not np.isnan(val) else "lightgray"
    #     for val in annotations["F1_RF"]
    # ]

    # Create clustermap
    g = sns.clustermap(
        similarity_matrix,
        method="average",
        metric="precomputed",
        row_linkage=linkage(condensed_dist, method="average"),
        col_linkage=linkage(condensed_dist, method="average"),
        cmap="Greys",
        vmin=similarity_matrix.values[similarity_matrix.values < 1].min() - 0.05,
        vmax=1.0,
        row_colors=row_colors_df,
        col_colors=None,  # Remove column color bar
        figsize=(16, 14),
        cbar_pos=(1.05, 0.5, 0.03, 0.15),
        dendrogram_ratio=(0.12, 0.12),
        linewidths=0.3,
        xticklabels=True,
        yticklabels=True,
    )

    for label in g.ax_row_colors.xaxis.get_majorticklabels():
        label.set_rotation(45)
        label.set_ha('right')

    # Hide the column dendrogram
    g.ax_col_dendrogram.set_visible(False)

    # g.ax_heatmap.invert_yaxis()
    # g.ax_heatmap.invert_xaxis()

    g.figure.suptitle(
        "Jaccard Similarity with Disease Groups and F1 Scores", fontsize=18, y=0.99
    )

    # Rotate labels
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

    from matplotlib.patches import Patch
    disease_patches = [
        Patch(color=disease_colors.get(disease, 'black'), label=disease)
        for disease in sorted(annotations["Disease"].unique())
    ]

    # Add legend to the figure
    g.ax_heatmap.legend(
        handles=disease_patches,
        title="Disease",
        bbox_to_anchor=(1.15, 1),
        loc='upper left',
        frameon=True
    )

    plt.tight_layout()
    plt.show()
    return g
