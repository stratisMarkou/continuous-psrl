""" Plotting script to visualize RL performance. """

import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from cpsrl.plot_utils import color_defaults, get_data

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
