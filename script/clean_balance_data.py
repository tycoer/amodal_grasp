from pathlib import Path
import argparse

import numpy as np
import os
from utils.io import *
from utils.perception import *
from utils.transform import Rotation, Transform
import logging


def main(args):
    root = args.root
    logging.basicConfig(filename=root / 'clean_balance_data.log',
                        filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        )


    # logging.info
    df = read_df(root)
    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]

    logging.info("Before clean and balance:")
    logging.info(f"Number of samples: {len(df.index)}")
    logging.info(f"Number of positives: {len(positives.index)}")
    logging.info(f"Number of negatives: {len(negatives.index)}")

    # clean
    df = read_df(root)
    df.drop(df[df["x"] < 0.02].index, inplace=True)
    df.drop(df[df["y"] < 0.02].index, inplace=True)
    df.drop(df[df["z"] < 0.02].index, inplace=True)
    df.drop(df[df["x"] > 0.28].index, inplace=True)
    df.drop(df[df["y"] > 0.28].index, inplace=True)
    df.drop(df[df["z"] > 0.28].index, inplace=True)
    # write_df(df, root)

    # balance
    # df = read_df(root)
    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]
    i = np.random.choice(negatives.index, len(negatives.index) - len(positives.index), replace=False)
    df = df.drop(i)
    write_df(df, root)

    # remove unreferenced scenes.
    # df = read_df(root)
    scenes = df["scene_id"].values
    for f in (root / "scenes").iterdir():
        if f.suffix == ".npz" and f.stem not in scenes:
            logging.info(f"Removed {f}")
            f.unlink()

    # tycoer
    if os.path.exists(root / "mesh_pose_list"):
        for f in (root / "mesh_pose_list").iterdir():
            if (f.suffix == ".npz" or f.suffix == '.npy') and f.stem not in scenes:
                logging.info(f"Removed {f}")
                f.unlink()


    # logging.info
    df = read_df(root)
    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]

    logging.info("After clean and balance:")
    logging.info(f"Number of samples: {len(df.index)}")
    logging.info(f"Number of positives: {len(positives.index)}")
    logging.info(f"Number of negatives: {len(negatives.index)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    main(args)