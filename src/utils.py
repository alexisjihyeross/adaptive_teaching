import json
import textwrap
import random
import numpy as np
import torch
import pickle
import wandb

from src.exp_configs import *
from src.eval import get_auc


# TODO: move to src.programs.utils? (for now, leaving here bc of import issues)
def end_teacher_log(
    teacher,
    teaching_params,
    global_results,
    config,
    gpt_helper,
    gpt_args,
    auc_keys=[
        ("gold_prob", "gold_total_probs"),
        ("dataset_acc_MAX", "dataset_accs_max"),
        ("dataset_acc_SAMPLE", "dataset_accs_sample"),
    ],
):
    """Helper function to log end teacher information:
    - Logs end information for GPT teachers (student guess, cost, prompts)
    - Logs end accuracies/aucs
    - Logs reason for finish
    """

    strategy = teaching_params["strategy"]
    if strategy in GPT_ONLY_STRATEGIES:
        # if not assume_known_prior, then log student guess
        if not teaching_params["assume_known_prior"]:
            teacher.get_student_type()

            # TODO: hacky, but only use end prompt if not assume_known_prior
            write_file(teacher.end_prompt, "teacher_end_prompt.txt")
            wandb.save("teacher_end_prompt.txt")

            gpt_student_type_results = gpt_helper.eval_student_type(
                teacher, config, gpt_args
            )
            wandb.log(
                {
                    f"gpt_teacher_info/{k}": v
                    for k, v in gpt_student_type_results.items()
                }
            )

        wandb.log(
            {
                "gpt_teacher_info/generated_invalid_input": int(
                    teacher.generated_invalid_input
                ),
                "gpt_teacher_info/generated_student_no_learning_response": int(
                    teacher.generated_student_no_learning_response
                ),
            }
        )

        y_acc = sum(global_results["y_correct_lst"]) / len(
            global_results["y_correct_lst"]
        )
        wandb.log({"gpt_teacher_info/gen_y_acc": y_acc})

    if strategy in GPT_STRATEGIES:
        # Write cache
        teacher.write_cache()

        # Write prompts and messages
        with open("teacher_messages.json", "w") as fout:
            json.dump(teacher.messages, fout)
        wandb.save("teacher_messages.json")
        with open("parsed_teacher_messages.json", "w") as fout:
            json.dump(teacher.parsed_messages, fout)
        wandb.save("parsed_teacher_messages.json")
        write_file(teacher.base_prompt, "teacher_base_prompt.txt")
        wandb.save("teacher_base_prompt.txt")

        # Log teacher cost
        teacher_cost = teacher.get_cost()
        print("teacher cost: ", teacher_cost)
        wandb.log({"gpt_teacher_info/teacher_cost": teacher_cost})

    end_stats = {}
    target_length = config["num_iterations"]
    for name, vals in auc_keys:
        auc = get_auc(global_results[vals], target_length=target_length)
        end_stats[f"aucs/{name}"] = auc

    wandb.log({f"end_stats/{k}": v for k, v in end_stats.items()})

    # Log reason for finish (if early finish)
    wandb.log(
        {
            "end_stats/reason_for_finish": global_results["reason_for_finish"],
            "end_stats/early_finish=True": int(
                global_results["reason_for_finish"] != ""
            ),
        }
    )

    # If early finish, print reason
    if global_results["reason_for_finish"] != "":
        print(f'Early finish because "{global_results["reason_for_finish"]}"')


def set_random_seeds(random_state):
    random.seed(random_state)
    np.random.seed(random_state)
    torch.random.manual_seed(random_state)
    torch.manual_seed(random_state)


def wrap_text(s, width=50):
    s_lst = textwrap.wrap(s, width)
    return "\n".join(s_lst)


def print_dict(d):
    print(json.dumps(d, sort_keys=True, indent=4))


def write_dict(d, file_name):
    print(f"Writing dict to {file_name}")
    # TODO: what if folder doesn't exist
    with open(file_name, "w") as file:
        file.write(json.dumps(d, sort_keys=True, indent=4))


def write_pickle(s, file_path):
    print(f"Writing to: {file_path}")
    with open(file_path, "wb") as file:
        pickle.dump(s, file)


def write_file(s, file_path):
    print(f"Writing to: {file_path}")
    with open(file_path, "w") as file:
        file.write(s)


def read_file(file_path):
    try:
        with open(file_path, "r") as file:
            txt = file.read()
        return txt
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None


def read_json(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data
