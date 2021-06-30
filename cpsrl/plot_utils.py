import os
import re
import glob
import json
from typing import List

import numpy as np
import pandas as pd


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
