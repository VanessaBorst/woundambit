import os
from typing import List, Dict, Union, Optional

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from medseg.data.split_type import SplitType
from medseg.evaluation.metrics import EvalMetric


@click.command()
@click.option("--dataframe_paths", type=str, multiple=True, help="Paths to the metrics dataframes (pickle files).")
@click.option("--model_names", type=str, multiple=True, help="Names of the models to use in the plot.")
@click.option("--split", type=click.Choice([split.value for split in SplitType]), help="The dataset split to plot.")
@click.option("--metrics_to_plot", type=click.Choice([metric.value for metric in EvalMetric] + ['mean_loss']),
              multiple=True, help="List of metrics to plot.")
@click.option("--output_folder", type=str, default="output", help="Output folder for the plots.")
@click.option("--single_plot", is_flag=True, help="Set this flag to create a single plot with all metrics.")
def main(dataframe_paths: List[str], model_names: List[str], split: str, metrics_to_plot: List[str], output_folder: str,
         single_plot: bool):
    if len(dataframe_paths) != len(model_names):
        raise ValueError("The number of dataframe_paths and model_names should be the same.")

    dfs = {}
    for path, name in zip(dataframe_paths, model_names):
        df = pd.read_pickle(path)
        dfs[name] = df

    split = SplitType(split)
    metrics_to_plot = [EvalMetric(metric) if metric != 'mean_loss' else metric for metric in metrics_to_plot]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plot_metrics_model_comparison(dfs, split, metrics_to_plot, output_folder, single_plot)


def plot_metrics(df: pd.DataFrame, split: SplitType, metrics_to_plot: List[Union[EvalMetric, str]], output_folder: str,
                 single_plot: bool = False):
    sns.set(style="darkgrid", palette="bright")

    metrics_to_plot = [metric.value if isinstance(metric, EvalMetric) else metric for metric in metrics_to_plot]

    df = df[(df['split'] == split.value) & (df['metric'].isin(metrics_to_plot))]
    os.makedirs(output_folder, exist_ok=True)
    if single_plot:
        plt.figure(figsize=(12, 6))
        ax = sns.lineplot(data=df, x='epoch', y='value', hue='metric', style='metric', markers=True, dashes=False)
        ax.set(xlabel='Epoch', ylabel='Value')
        plt.legend(title="Metric")
        plt.title(f"{split.value.capitalize()} Metrics")
        sns.despine()
        plt.savefig(os.path.join(output_folder, f"{split.value}.svg"), format='svg')
        plt.close()
    else:
        for metric in metrics_to_plot:
            plt.figure()
            sns.lineplot(data=df[df['metric'] == metric], x='epoch', y='value', markers=True, dashes=False)
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.title(f"{split.value.capitalize()} {metric}")
            sns.despine()
            plt.savefig(os.path.join(output_folder, f"{split.value}_{metric}.svg"), format='svg')
            plt.close()


def plot_metrics_model_comparison(dfs: Dict[str, pd.DataFrame], split: SplitType,
                                  metrics_to_plot: List[Union[EvalMetric, str]], output_folder: str,
                                  single_plot: bool = False):
    sns.set(style="darkgrid", palette="bright")
    # TODO: unused code

    if 'mean_loss' in metrics_to_plot:
        metrics_to_plot.remove('mean_loss')
        metrics_to_plot = [metric.value for metric in metrics_to_plot] + ['mean_loss']
    else:
        metrics_to_plot = [metric.value for metric in metrics_to_plot]

    combined_df = pd.DataFrame()

    for model_name, df in dfs.items():
        df = df[(df['split'] == split.value) & (df['metric'].isin(metrics_to_plot))].copy()
        df['model'] = model_name
        combined_df = combined_df.append(df)

    if single_plot:
        plt.figure(figsize=(12, 6))
        ax = sns.lineplot(data=combined_df, x='epoch', y='value', hue='model', style='metric', markers=True,
                          dashes=False)
        ax.set(xlabel='Epoch', ylabel='Value')
        plt.legend(title="Model")
        plt.title(f"{split.value.capitalize()} Metrics - Model Comparison")
        sns.despine()
        plt.savefig(os.path.join(output_folder, f"{split.value}_all_metrics_model_comparison.svg"), format='svg')
        plt.close()
    else:
        for metric in metrics_to_plot:
            plt.figure()
            sns.lineplot(data=combined_df[combined_df['metric'] == metric], x='epoch', y='value', hue='model',
                         markers=True, dashes=False)
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.title(f"{split.value.capitalize()} {metric} - Model Comparison")
            sns.despine()
            plt.savefig(os.path.join(output_folder, f"{split.value}_{metric}_model_comparison.svg"), format='svg')
            plt.close()


def save_boxplot(metrics: Dict[str, List[float]],
                 save_path: str,
                 ylabel: str = "IoU",
                 title: Optional[str] = None):
    """
    Plots boxplots for each provided set of metric values and saves the plot to a file.

    The function generates a boxplot for each set of metric values. Each set of metric values corresponds
    to a specific neural network. The metric values are intended to be image-wise metrics, in order to provide a
    plot that shows the distribution of the metric values and makes it possible to compare models in terms of
    their performance on outlier images.

    Args:
        metrics (Dict[str, List[float]]): A dictionary with neural network names as keys and lists of metric
                                          values as values. Each list represents a series of measurements
                                          (e.g., iou, dice) of the respective neural network.
        save_path (str): The file path where the generated plot should be saved.

    """

    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.boxplot(data=list(metrics.values()))
    plt.xticks(range(len(metrics)), metrics.keys())
    if title is not None:
        plt.title(title)
    plt.xlabel('Models')
    plt.ylabel(ylabel)
    plt.savefig(save_path)
