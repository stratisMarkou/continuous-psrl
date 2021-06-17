""" Plotting script to visualize RL performance. """

import os
import re
import glob
import json
import argparse
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dir", type=str, default="./results", help="Path to results directory"
)
parser.add_argument(
    "--env",
    type=str,
    choices=["MountainCar", "CartPole"],
    default="MountainCar",
    help="RL environment",
)
parser.add_argument(
    "--format", type=str, default="png", help="File format, e.g. png, pdf"
)


color_defaults = [
    "#1f77b4",  # muted blue
    "#d62728",  # brick red
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]


def get_results_paths(dir: str) -> List[str]:
    """Returns a list of file names from the given results directory."""
    return glob.glob(os.path.join(dir, "*.txt"))


def parse_json_lines_file(path: str) -> List[dict]:
    """Parses a file consisting of lines of json expressions."""
    dicts = []
    with open(path, mode="r") as f:
        for line in f:
            dicts.append(json.loads(line))
    return dicts


def parse_results_filename(filename: str) -> dict:
    """Parses the file name of a results file."""
    regex = re.compile(
        r"(?P<agent>.*?)_(?P<env>.*?)_(" + "?P<seed>\d+)\.txt"
    )
    match = regex.match(filename)
    if not match:
        raise RuntimeError(f"Cannot parse filename: {filename}")
    return {
        "Agent": match.group("agent"),
        "Env": match.group("env"),
        "Seed": int(match.group("seed")),
    }


def get_data(dir: str) -> pd.DataFrame:
    """Loads data from a directory."""
    paths = get_results_paths(dir)
    assert len(paths) > 0

    frames = []
    for path in paths:
        data = parse_json_lines_file(path)
        df = pd.DataFrame(data)

        info = parse_results_filename(os.path.basename(path))
        for k, v in info.items():
            df[k] = v
        frames.append(df)

    data = pd.concat(frames)

    # Compute average and std over seeds
    grp = ["Agent", "Env", "Episode"]
    data = data.groupby(grp).agg([np.mean, np.std]).reset_index()
    return data


def plot(
    df: pd.DataFrame,
    dir: str,
    **plt_kwargs
):
    plt.figure()
    plt.clf()
    prop = "Return"
    for i, (label, group) in enumerate(df.groupby("Agent")):
        x = group["Episode"].to_numpy()
        y_mean = group[prop]["mean"].to_numpy()
        y_std = group[prop]["std"].to_numpy()
        color = color_defaults[i]

        plt.plot(x, y_mean, label=label, linewidth=2, color=color)
        plt.fill_between(
            x, y_mean + 2 * y_std, y_mean - 2 * y_std, alpha=0.2, color=color
        )

        final_mean, final_std = y_mean[-1], y_std[-1]
        print("{}: {:.2f} (+-{:.4f})".format(label, final_mean, final_std))

    plt.legend(loc="lower right")
    plt.grid()
    plt.xlabel("Episode")
    plt.ylabel(prop)
    plt.gca().set(**plt_kwargs)
    plt.savefig(dir, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    import matplotlib
    matplotlib.rcParams.update({"font.size": 16})

    args = parser.parse_args()
    save_dir = os.path.join(args.dir, f"{args.env}.{args.format}")
    plt_kwargs = {"title": "{}".format(args.env)}

    df = get_data(args.dir)
    df_data = df[df["Env"] == args.env]
    plot(df_data, save_dir, **plt_kwargs)
