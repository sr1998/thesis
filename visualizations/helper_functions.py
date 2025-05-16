import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import wandb
from src.global_vars import BASE_DATA_DIR


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
    project_name, tags, column_wanted="Outer fold.test/f1", notes_contains=None, studies_to_ignore=[]
):
    api = wandb.Api()

    # get run names interested in
    runs = api.runs(f"shayan000/{project_name}", filters={"tags": {"$all": tags}})

    # Download all data points for the graph "Outer fold.test/f1" from wandb
    data_wanted_dict = {}
    for run in runs:
        if run.state != "finished":
            print(f"Run {run.name} is not finished. Skipping...")
            continue

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

        data_wanted_dict[run.name] = data_wanted
    # Create a DataFrame from the dictionary
    full_df = pd.DataFrame(data_wanted_dict)

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


def plot_cross_val_results_grouped(
    group_df_lists, group_titles, models, title=None, save_path: str = "", show_annotations=True, height=1000, width=1200, legend=True, group_title=None
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
    model_width = group_width / len(models)  # Width for each model's bar
    # remove empty groups:
    empty_groups = []
    for i, group_title in enumerate(group_titles):
        if sum([df.shape[1] for df in group_df_lists[i]]) == 0:
            empty_groups.append(group_title)
            continue
    print("Empty groups: ", empty_groups)
    group_df_lists = [
        df_list for i, df_list in enumerate(group_df_lists) if group_titles[i] not in empty_groups
    ]
    group_titles = [g for g in group_titles if g not in empty_groups]


    # For each group
    for group_idx, (df_list, group_title) in enumerate(
        zip(group_df_lists, group_titles)
    ):
        print("group title  ", group_title)
        print([df.shape for df in df_list])


        # Calculate statistics for this group
        # vals_per_df = [df.values.flatten() for df in df_list]
        # means = np.array([np.mean(vals) for vals in vals_per_df])
        # stds = np.array([np.std(vals) for vals in vals_per_df])
        # all_means.extend(means)
        # all_errors.extend(stds)

        # Add bars for each model in this group
        for model_idx, (model, df) in enumerate(zip(models, df_list)):
            # Calculate position for this bar
            position = group_idx + (model_idx - len(models) / 2 + 0.5) * model_width
            values = df.values.flatten()
            non_zero_values = values[values != 0]
            # Add bar
            fig.add_trace(
                go.Box(
                    x=[position] * len(non_zero_values),
                    y=non_zero_values,
                    # error_y=dict(type="data", array=[std]),
                    name=model,
                    legendgroup=model if legend else None,
                    marker_color=colors[model_idx % len(colors)],
                    width=model_width * 0.9,  # Slightly narrower than allocated space
                    showlegend=group_idx == 0 if legend else False,  # Only show in legend for first group
                    # text=f"{mean:.2f}<br>±<br>{std:.2f}",
                    # textposition="",
                )
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

    # Update layout
    group_positions = np.arange(len(group_titles))
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(weight="bold", size=16),
            x=0.5,
        ) if title else None,
        xaxis=dict(
            tickvals=group_positions,
            ticktext=group_titles,
            title=group_title or "Group",
            title_font=dict(size=20),
            tickfont=dict(size=18),
            tickangle=55,
        ),
        yaxis=dict(
            title="F1 Score",
            title_font=dict(size=20),
            tickfont=dict(size=18),
            range=[-.03, 1],
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.4,
            xanchor="center",
            x=0.5,
            title="Method",
        ) if legend else None,
        height=height,
        width=width,
        margin=dict(b=100),
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
            title_font=dict(size=20),
            tickfont=dict(size=18),
            tickangle=55,
        ),
        yaxis=dict(
            title="F1 Score",
            title_font=dict(size=20),
            tickfont=dict(size=18),
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
        ),
        height=1200,
        width=1500,
        margin=dict(b=150),
    )

    fig.show()

    if save_path:
        fig.write_image(save_path, scale=2)


def get_desired_cross_val_results_for_models(
    project, models, tags, metric, studies_to_ignore, notes_contains=None
):
    dfs = []
    for model in models:
        df = get_cross_val_results_from_wandb(
            project, [model] + tags, metric, notes_contains, studies_to_ignore
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
        new_dfs.append(df.loc[:,    df.columns.str.contains("|".join(studies_to_keep))])
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
    width=1900,
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
    
    # Use default Plotly colors if not specified
    if color_palette is None:
        color_palette = px.colors.qualitative.Plotly
    
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
                    'values': [values]
                }
            else:
                raise ValueError(
                    f"Duplicate study '{study}' found for model '{model_name}'. Please ensure unique study names."
                )
                # Append if multiple columns map to the same study
                # stats[model_name][study]['means'].append(mean)
                # stats[model_name][study]['stds'].append(std)
                stats[model_name][study]['values'].append(values)
    
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
    group_width = 0.7  # Width allocated for each study group
    model_width = group_width / len(model_names)  # Width for each model's bar within group
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
            position = study_positions[study_idx] + (model_idx - len(model_names) / 2 + 0.5) * model_width
            
            # Get values if study exists for this model
            if study in stats[model_name]:
                # For boxplots, we need the raw values
                values = np.concatenate(stats[model_name][study]['values'])
                values = values[values != 0]
                
                # Add box plot
                fig.add_trace(
                    go.Box(
                        x=[position] * len(values),
                        y=values,
                        name=model_name,
                        legendgroup=model_name,
                        marker_color=color_palette[model_idx % len(color_palette)],
                        width=model_width * bar_width_factor,
                        showlegend=study_idx == 19,  # Only show in legend for first study
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
        title=dict(
            text=title,
            font=dict(weight="bold", size=16),
            x=0.5,
        ) if title else None,
        xaxis=dict(
            tickvals=study_positions,
            ticktext=all_studies,
            title="Study",
            title_font=dict(size=20),
            tickfont=dict(size=14),
            tickangle=45,
        ),
        yaxis=dict(
            title=y_axis_title,
            title_font=dict(size=20),
            tickfont=dict(size=14),
            range=y_range,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            title="Model",
        ),
        height=height,
        width=width,
        margin=dict(b=150, t=150),
        template="plotly_white",
    )
    
    # Show the figure
    fig.show()
    
    # Save if path provided
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path, scale=2)
    
    return fig, stats