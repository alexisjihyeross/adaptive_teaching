from tqdm import tqdm
import pandas as pd
import pickle
import json
import time
import os
import argparse
import random
import wandb
from sklearn.metrics import accuracy_score

from src.verbs.utils import *
from src.verbs.gpt_utils import *
from src.verbs.teacher import *
from src.verbs.dataset import *

from src.utils import set_random_seeds, print_dict, end_teacher_log
from src.eval import *
from src.exp_configs import *
from src.verbs.exp_configs import *

# Used for logging results
Y_MAP = {
    "+d": 0,
    "+ed": 1,
    "y_to_ied": 2,
    "+consonant+ed": 3,
}


def initialize_gpt_args(strategy, gpt_helper, teaching_params, student_params):
    gpt_args = {}
    if strategy in GPT_STRATEGIES:
        base_prompt = gpt_helper.get_teacher_base_prompt(
            student_params, assume_known_prior=teaching_params["assume_known_prior"]
        )
        end_prompt = gpt_helper.get_teacher_end_prompt()
        gpt_args["base_prompt"] = base_prompt
        gpt_args["end_prompt"] = end_prompt

    return gpt_args


def log_teacher_results(teacher_results):
    print_teacher_results(teacher_results)
    wandb.log(
        {
            f"teacher_results/{k}": v
            for k, v in teacher_results.items()
            if "acc" in k or "label_likelihood" in k
        }
    )


def log_results(
    global_results,
    i,
    x,
    y,
    gold_y,
    pred,
    x_is_valid,
    y_is_valid,
    ex_is_valid,
    teacher_results,
    teacher,
    student,
    dataset,
    unknown_concept,
):

    # only consider y validity for now in deciding whether to learn from example
    y_int = Y_MAP[y] if ex_is_valid else None

    pred_is_correct = int(pred == y)

    results = {
        #                "iter": i,
        "data/iter": i,
        "data/x": x,
        "data/y_int": y_int,
        "data/y=unknown": int(y == unknown_concept),
        "data/y": y,
        "data_validity/iter": i,
        "data_validity/y=gold_y": int(
            y == gold_y
        ),  # TODO: should this be in data instead of data_validity?
        "data_validity/y=valid": int(y_is_valid),
        "data_validity/x=valid": int(x_is_valid),
        "data_validity/ex=valid": int(ex_is_valid),
        "preds/iter": i,
        "preds/pred": pred,
        "preds/pred_is_correct": pred_is_correct,
        # mean prob(true label) across full dataset
    }

    log_prob = student.get_posterior_log_prob(*teacher_results["posterior_mean"]).item()
    post_kl_div = kl_divergence(student, teacher_results["nb"])

    prob_preds = student.predict_proba(dataset.inputs)
    label_likelihood_mean = get_label_likelihoods(student, dataset, prob_preds)
    # TODO: also accuracy with sampling predictions?
    max_preds = student.predict(dataset.inputs, proba=prob_preds, do_sample=False)
    max_acc = accuracy_score(dataset.outputs, max_preds)

    results.update(
        {
            "student_learning/iter": i,
            "student_learning/gold_prob": log_prob,
            "student_learning/posterior_kl_div": post_kl_div,
            "student_learning/acc_on_dataset": max_acc,
            "student_learning/label_likelihoods": label_likelihood_mean,
        }
    )

    if isinstance(teacher, ApproximateProbabilisticVerbsTeacher):
        # posterior kl div between teacher guess of student and actual student
        guess_post_kl_div = kl_divergence(teacher.student_guess, student)
        pop_is_correct = int(teacher.student_pop_idx == 0)
    else:
        guess_post_kl_div = None
        pop_is_correct = None

    results.update(
        {
            "student_model/iter": i,
            "student_model/posterior_kl_div": guess_post_kl_div,
            # TODO: hacky, but assumes true is always 0
            "student_model/pop_is_correct": pop_is_correct,
        }
    )

    # get accuracy and preds breakdowns for full dataset
    df = dataset.to_dataframe()
    df["pred"] = max_preds

    results["student_preds_breakdown/iter"] = i
    results["student_acc_breakdown/iter"] = i
    unique_labels = dataset.get_unique_outputs()
    for lab in unique_labels:
        temp = df[df["output"] == lab]
        temp_acc = accuracy_score(temp["output"].values, temp["pred"].values)
        results[f"student_preds_breakdown/{lab}_{Y_MAP[lab]}"] = len(
            df[df["pred"] == lab]
        )
        results[f"student_acc_breakdown/{lab}_{Y_MAP[lab]}"] = temp_acc

    global_results["y_correct_lst"].append(y == gold_y)
    global_results["log_PDFs"].append(log_prob)
    # TODO: max/sample
    global_results["dataset_acc_MAX"].append(max_acc)
    global_results["gold_prob"].append(log_prob)

    wandb.log(results)

    return results, global_results


def save_config(run_config):
    with open("config.json", "w") as fout:
        json.dump(run_config, fout)
    wandb.save("config.json")


def get_label_likelihoods(nb, dataset, pred_probs):
    label_indices = [nb.get_label_idx(y) for y in dataset.outputs]
    likelihoods = [probs[idx] for probs, idx in zip(pred_probs, label_indices)]
    return np.mean(likelihoods)


def load_teacher_model(dataset, seed):
    """Loads and fits a teacher model to the dataset"""
    # TODO: make sure this defaults to betas = ones
    teacher_nb = sample_bnb(dataset, seed, alphas=torch.ones(len(set(dataset.outputs))))
    teacher_nb.transform_and_fit(dataset.inputs, dataset.outputs)
    return teacher_nb


def get_populations(
    student,
    teacher_nb,
    seed,
    teaching_params,
    unknown_label_idx,
    prior_initialize_params,
):
    # first population is real student, can't just use student because will modify object when updating (will increment counts twice?)
    populations = [
        sample_bnb_with_unknown(
            dataset, seed, teacher_nb, unknown_label_idx, prior_initialize_params
        )
    ]
    student_pop_idx = 0
    assert torch.equal(
        populations[0].alpha_prior_parameters, student.alpha_prior_parameters
    )
    for pop in populations:
        assert all(
            [a == b for (a, b) in zip(student.get_label_names(), pop.get_label_names())]
        )
    populations.extend(
        [
            sample_bnb_with_unknown(
                dataset, seed, teacher_nb, lab_idx, prior_initialize_params
            )
            for lab_idx in range(len(student.get_label_names()))
            if lab_idx != unknown_label_idx
        ]
    )

    # assume known prior
    if (
        "assume_known_prior" in teaching_params
        and teaching_params["assume_known_prior"]
    ):
        student_pop_idx = 0
    # if don't assume known prior, randomly sample one of the wrong populations
    else:
        print("Sampling a student population...")
        # student_pop_idx = random.sample(range(1, len(populations)), 1)[0]
        # sample from all populations, including the true student
        student_pop_idx = random.sample(range(len(populations)), 1)[0]
        print(f"Student population idx: {student_pop_idx}")
        print("Alpha parameters of student guess:")
        print(populations[student_pop_idx].alpha_post_parameters)
        print(populations[student_pop_idx].get_label_names())
    return populations, student_pop_idx


def validate_config(config):
    prior_initial_params = config["student_params"]["prior_initialize_params"]
    # prior initial params must be either floats or 'MAP'
    for v in prior_initial_params.values():
        assert (
            isinstance(v, float) or isinstance(v, int) or v in ["MAP", "MAP_inverse"]
        ), f"{v} has invalid type ({type(v)})"


def load_and_eval_teacher(dataset, seed):
    teacher_nb = load_teacher_model(dataset, seed)
    teacher_pred_probs = teacher_nb.predict_proba(dataset.inputs)
    # TODO: use argmax? (i.e. no sample?)
    teacher_preds = teacher_nb.predict(
        dataset.inputs, proba=teacher_pred_probs, do_sample=False
    )
    teacher_label_likelihood = get_label_likelihoods(
        teacher_nb, dataset, teacher_pred_probs
    )
    teacher_acc = accuracy_score(dataset.outputs, teacher_preds)

    teacher_post_mean = teacher_nb.get_posterior_mean()
    teacher_results = {
        "nb": teacher_nb,
        "preds": teacher_preds,
        "acc": teacher_acc,
        "posterior_mean": teacher_post_mean,
        "label_likelihood": teacher_label_likelihood,
    }

    teacher_df = dataset.to_dataframe()
    teacher_df["pred"] = teacher_preds

    unique_labels = dataset.get_unique_outputs()
    for lab in unique_labels:
        temp = teacher_df[teacher_df["output"] == lab]
        temp_acc = accuracy_score(temp["output"].values, temp["pred"].values)
        teacher_results[f"{lab}_{Y_MAP[lab]}_acc"] = temp_acc
    return teacher_results


def print_teacher_results(teacher_results):
    teacher_acc = teacher_results["acc"]
    teacher_nb = teacher_results["nb"]
    teacher_label_likelihood = teacher_results["label_likelihood"]

    print("teacher acc: ", teacher_acc)
    print("teacher posterior alpha params: ", teacher_nb.alpha_post_parameters)
    print("teacher posterior beta params: ", teacher_nb.beta_post_parameters)
    print("teacher label likelihood mean: ", teacher_label_likelihood)


def get_teaching_args(
    config,
    teaching_params,
    teacher_results,
    student_pop_idx,
    populations,
    gpt_helper,
    student_params,
):
    teaching_kwargs = teaching_params.copy()
    teaching_kwargs.pop("strategy")
    teaching_kwargs.pop("id")

    strategy = teaching_params["strategy"]
    if strategy in PROBABILISTIC_RANKING_STRATEGIES:
        teaching_args = (
            strategy,
            dataset,
            teacher_results["nb"],
            student_pop_idx,
            populations,
        )
    else:
        teaching_args = (strategy, dataset)

    if strategy in NON_RANDOM_STRATEGIES:
        teaching_kwargs.pop("assume_known_prior")
    teaching_kwargs["seed"] = config["seed"]

    if strategy in GPT_STRATEGIES:
        gpt_args = initialize_gpt_args(
            strategy,
            gpt_helper,
            teaching_params,
            student_params,
        )
        teaching_kwargs.update(gpt_args)

    return teaching_args, teaching_kwargs


def run_exp(config, dataset, wandb_project="pedagogy_lists", exp_notes=None, tag=None):

    validate_config(config)

    seed = config["seed"]
    set_random_seeds(seed)

    num_iterations = config["num_iterations"]
    teaching_params_lst = config["strategies"]
    student_params = config["student_params"]
    unknown_concept = student_params["unknown_concept"]
    prior_initialize_params = student_params["prior_initialize_params"]

    # add this for easy access (eg in gpt_helper.parse_student_type)
    config["unknown_concept"] = unknown_concept

    if tag is None:
        tags = []
    else:
        tags = tag.split(",")

    teacher_results = load_and_eval_teacher(dataset, seed)
    gpt_helper = VerbsGPTHelper(dataset)

    unknown_label_idx = teacher_results["nb"].get_label_idx(unknown_concept)
    for t_idx, teaching_params in enumerate(teaching_params_lst):
        run_config = config.copy()
        run_config.update({"teaching_params": teaching_params})
        run_config.update({"unknown_concept_idx": unknown_label_idx})

        strategy = teaching_params["strategy"]

        print("======================================================")
        print(f"STRATEGY: {strategy} ({t_idx+1}/{len(teaching_params_lst)})")

        set_random_seeds(seed)
        # wandb_name = get_wandb_name(teaching_params)
        wandb_name = teaching_params["id"]

        run = wandb.init(
            config=run_config,
            name=wandb_name,
            project=wandb_project,
            reinit=True,
            notes=exp_notes,
            tags=tags,
        )
        wandb.config.update({"strategy": strategy})

        log_teacher_results(teacher_results)

        save_config(run_config)

        print(f"unknown concept: {unknown_concept}")
        print(f"unknown concept idx: {unknown_label_idx}")

        student = sample_bnb_with_unknown(
            dataset,
            seed,
            teacher_results["nb"],
            unknown_label_idx,
            prior_initialize_params,
        )

        populations, student_pop_idx = get_populations(
            student,
            teacher_results["nb"],
            seed,
            teaching_params,
            unknown_label_idx,
            prior_initialize_params,
        )

        student = student
        print("Num features:", student.n_features)
        print("Num alpha parameters:", student.alpha_post_parameters.shape)
        print("Alpha parameters of true student:")
        print(student.alpha_post_parameters)
        print(student.get_label_names())

        print("Beta parameters of true student:")
        print(student.beta_post_parameters)
        print(student.get_label_names())
        print("Num beta parameters:", student.beta_post_parameters.shape)

        teaching_args, teaching_kwargs = get_teaching_args(
            config,
            teaching_params,
            teacher_results,
            student_pop_idx,
            populations,
            gpt_helper,
            student_params,
        )
        teacher = initialize_teacher(
            *teaching_args,
            **teaching_kwargs,
        )

        global_results = {
            "gold_prob": [],
            "log_PDFs": [],
            "dataset_acc_MAX": [],
            "dataset_acc_SAMPLE": [],
            "y_correct_lst": [],
            "reason_for_finish": "",  # used if teacher finishes early
        }

        for i in tqdm(range(num_iterations)):
            if isinstance(teacher, GPTVerbsTeacher):
                x = teacher.select()
                # defaults to sampling
                pred = student.predict([x])[0]
                assert pred is not None, f"pred is None for inp {x}"
                y = teacher.update_predictions(x, pred)

            else:
                x, y = teacher.select()

                pred = student.predict([x])[0]
                teacher.update_predictions(x, pred, y)

            try:
                gold_y = get_verb_category(x, dataset)
            # TODO: need to account for the fact that verb might not be in dataset; this will set to None if input is not found in dataset
            # except KeyError:
            except VerbCategoryError:
                gold_y = None

            y_is_valid = dataset.check_output_validity(y)
            x_is_valid = dataset.check_input_validity(x)
            # only consider y validity for now in deciding whether to learn from example
            ex_is_valid = y_is_valid
            # only update student if example is valid
            student_update_bool = ex_is_valid

            teacher.store_student_response(student_update_bool)

            # if example is invalid, do not learn from it, and skip logging updates
            # TODO: should we send a message to teacher if GPT? (currently not, because what to do if using generated label and it's an invalid category? could try just doing it for gold labels and overwriting and saying 'I don't understand that verb type')
            if not student_update_bool:
                # TODO: update GPT message for next prediction
                print("Example is invalid, so skipping...")
                continue

            log_results(
                global_results,
                i,
                x,
                y,
                gold_y,
                pred,
                x_is_valid,
                y_is_valid,
                ex_is_valid,
                teacher_results,
                teacher,
                student,
                dataset,
                unknown_concept,
            )

            print(f"iter:\t{i}")
            print(f"datapoint:\t{x} -> {y}")
            print(f"pred:\t{pred}")

            student.transform_and_fit([x], [y])

        end_teacher_log(
            teacher,
            teaching_params,
            global_results,
            config,
            gpt_helper,
            {},
            auc_keys=[
                ("gold_prob", "gold_prob"),
                ("dataset_acc_MAX", "dataset_acc_MAX"),
                # ("dataset_acc_SAMPLE", "dataset_accs_sample"),
            ],
        )

        run.finish()


def get_configs(dataset):
    """Helper function to get all experimental configurations"""
    config = {
        "seed": 0,
        "strategies": STRATEGIES,
        "student_params": student_params,
        "num_iterations": 50,
        "num_test": 20,
    }

    configs = []

    set_random_seeds(0)
    unique_outputs = dataset.get_unique_outputs()

    for seed in [0, 1, 2]:
        for concept in unique_outputs:
            temp = config.copy()
            temp["seed"] = seed
            # need to copy dict so that it's not overriden later
            temp_student_params = temp["student_params"].copy()
            temp_student_params.update({"unknown_concept": concept})
            temp_student_params.update(
                {"prior_initialize_params": both_initialize_params}
            )
            temp["student_params"] = temp_student_params
            configs.append(temp)

    return configs


def load_dataset(args):
    if args.dataset_path is not None and os.path.exists(args.dataset_path):
        print(f"Reading dataset from path: {args.dataset_path}")
        dataset = pickle.load(open(args.dataset_path, "rb"))
    else:
        dataset = load_dataset()
        print(f"Writing dataset to path: {args.dataset_path}")
        with open(args.dataset_path, "wb") as f:
            pickle.dump(dataset, f)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, default="verbs")
    parser.add_argument("--exp_id", type=str, default=None)
    parser.add_argument("--exp_notes", type=str, default=None)
    parser.add_argument("--exp_tag", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)

    args = parser.parse_args()

    dataset = load_dataset(args)
    configs = get_configs(dataset)

    for exp_idx, config in enumerate(configs):
        print(f"Starting exp {exp_idx+1}/{len(configs)}")
        print_dict(config)
        t0 = time.time()

        run_exp(
            config,
            dataset,
            wandb_project=args.wandb_project,
            exp_notes=args.exp_notes,
            tag=args.exp_tag,
        )
        t1 = time.time()
        print(f"Finished exp {exp_idx+1}/{len(configs)}")
        print(f"Took {round((t1-t0)/60, 2)} minutes")
