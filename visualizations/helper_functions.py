import json

import pandas as pd
import plotly.express as px

import wandb
from src.global_vars import BASE_DATA_DIR


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

def visualize_metrics_by_study_groups(
    full_df,
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
    fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=13))
    fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=13))

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
