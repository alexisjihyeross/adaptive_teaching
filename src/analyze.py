import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb
import argparse
import os

from src.utils import *

F_KNOWER = "b-learner"
G_KNOWER = "f-learner"

MULTIPLY_GENERALIZER = "multiply_generalizer"
ADD_GENERALIZER = "add_generalizer"
CROSS_MULTIPLIER = "cross_multiplier"


def download_wandb_runs(
    entity,
    project,
    gold_prob_key="gold_total_prob",
    extra_keys=[],
):
    wandb.login()

    api = wandb.Api(timeout=100)

    runs = api.runs(f"{entity}/{project}")

    mega_df = pd.DataFrame()

    all_data = []

    for run_idx, run in tqdm(enumerate(runs), total=len(runs)):

        # add all the keys that are logged that you want to download
        keys = [
            "data/iter",
            "iter",
            f"student_learning/{gold_prob_key}",
            "student_learning/iter",
            "preds/iter",
            "preds/pred_is_correct",
            "preds/pred",
            "data/x",
            "data/y",
            "end_stats/aucs/dataset_acc_MAX",
            "end_stats/aucs/gold_prob",
            "gpt_teacher_info/guess_is_correct",
        ]

        keys.extend(extra_keys)

        # filter ones that were killed
        if run.state != "finished":
            print("filtering run: ", run.path)
            continue

        history = run.scan_history()

        history_df = pd.DataFrame(history)

        # map the data/y to "undefined" if iter is not None; otherwise, keep the value
        history_df["data/y"] = history_df.apply(
            lambda row: (
                "undefined"
                if (
                    not pd.isnull(row["data/iter"])
                    and not pd.isnull(row["data/x"])
                    and pd.isnull(row["data/y"])
                )
                else row["data/y"]
            ),
            axis=1,
        )

        history_df["preds/pred"] = history_df.apply(
            lambda row: (
                "undefined"
                if (not pd.isnull(row["preds/iter"]) and pd.isnull(row["preds/pred"]))
                else row["data/y"]
            ),
            axis=1,
        )

        config = {k: v for k, v in run.config.items() if not k.startswith("_")}

        exp = {}

        # if key doesn't exist, set to nan (though this shouldn't happen after filtering empty runs)
        exp.update(
            {
                f"{col}": (
                    history_df[~history_df[col].isnull()][col].values
                    if col in history_df.columns
                    else np.nan
                )
                for col in keys
            }
        )

        # expand the config
        for key, val in config.items():
            if key in [
                "prog_concept_params",
                "student_concept_params",
                "teaching_params",
                "student_params",
            ]:
                for sub_key, sub_val in val.items():
                    exp.update({f"config/{key}/{sub_key}": sub_val})
            else:
                exp.update({f"config/{key}": val for key, val in config.items()})
        exp.update({"config": run.config})
        exp.update({f"strategy": run.config["strategy"]})
        exp.update({f"tags": run.tags})
        exp.update({"name": run.name})

        exp["run_id"] = run.id
        all_data.append(exp)

    mega_df = pd.DataFrame(all_data)

    return mega_df


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entity", type=str, help="Name of wandb entity.", required=True
    )
    parser.add_argument(
        "--project", type=str, help="Name of wandb project.", required=True
    )
    parser.add_argument(
        "--gold_prob_key",
        type=str,
        default="gold_total_prob",
        help="Key for gold probability.",
    )
    parser.add_argument("--out_file", default=None, help="Path to save results to.")

    args = parser.parse_args()

    if args.out_file is None:
        out_dir = "results/wandb"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_file = os.path.join(out_dir, f"{args.project}.csv")
        print(f"No out_file specified, saving to {args.out_file}")

    df = download_wandb_runs(
        args.entity,
        args.project,
        gold_prob_key=args.gold_prob_key,
    )

    # Print mean gold_prob for each strategy in reverse order, and number of runs
    print("Mean gold_prob by teacher:")
    sorted = (
        df.groupby("name")["end_stats/aucs/gold_prob"]
        .mean()
        .sort_values(ascending=False)
    )

    for name, mean in sorted.items():
        # print lined up (name up to 20 characters)
        print(f"{name:30} {mean:.3f}  ({df[df['name'] == name].shape[0]} runs)")

    # Write df to file
    df.to_csv(args.out_file, index=False)
    print(f"Saved results to {args.out_file}")
