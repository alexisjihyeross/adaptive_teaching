import pandas as pd
from time import sleep
import time
import argparse
import random
import wandb

from src.eval import *
from src.programs.interpreter import Interpreter
from src.programs.synthesizer import *
from src.programs.bps import *
from src.programs.utils import get_prog_reps, print_beliefs
from src.programs.prior import *

from src.programs.exp_configs import TEACHING_PARAMS
from src.programs.concepts import *
from src.programs.prog_configs import *
from src.programs.utils import run_teaching_exp

from src.programs.fractions.dataset import *
from src.programs.fractions.teacher import *

from run_functions import (
    run_exp,
    run_teaching_exp,
    get_outputs_by_inp,
    save_initial_files,
)

from src.utils import set_random_seeds, print_dict


def initialize_gpt_args(config, gpt_helper, ordered_student_params):
    population_params = config["student_populations"]
    student_concept_params = config["student_concept_params"]
    teaching_params = config["teaching_params"]

    base_prompt = gpt_helper.get_teacher_base_prompt(
        student_concept_params,
        population_params,
        assume_known_prior=teaching_params["assume_known_prior"],
    )

    end_prompt = gpt_helper.get_teacher_end_prompt(
        population_params,
    )

    gpt_args = {
        "base_prompt": base_prompt,
        "end_prompt": end_prompt,
        # TODO: hacky, but store true_student_idx and student_concept_params for parsing gpt student type later (gpt_helper.parse_gpt_student_type())
        "true_student_idx": population_params.index(student_concept_params),
        "population_params": population_params,
    }

    return gpt_args


def get_configs():
    """Helper function to get all experimental configurations for fractions"""
    strategies = [
        TEACHING_PARAMS["ranking_known"],
        TEACHING_PARAMS["ranking_unknown"],
        TEACHING_PARAMS["non-adaptive_known"],
        TEACHING_PARAMS["non-adaptive"],
        TEACHING_PARAMS["atom"],
        TEACHING_PARAMS["gpt4"],
        TEACHING_PARAMS["gpt4_known"],
        TEACHING_PARAMS["random"],
    ]
    base_config = {
        "seed": 0,
        "env_name": "fraction",
        "strategies": strategies,
        "num_iterations": 40,  # TODO: not comparable direcly with 50
        "num_test": 50,
        "min_denom": 2,
        "max_denom": 10,
        "min_num": 1,
        "max_num": 5,
        "target_prog": "(_f_ (_h_ x_))",
    }

    student_params = {
        "add_generalizer": {
            "student_noise": 0.8,
            "student_concept_params": {
                "id": "add_generalizer",
                "single_description": "student who performs addition correctly, but tends to incorrectly multiply only numerators when multiplying fractions, especially when the denominators are equal; if the denominators are not equal, the student sometimes makes common denominators and then multiplies the numerators",
                "plural_description": "Students who perform addition correctly, but tend to incorrectly multiply only numerators when multiplying fractions, especially when the denominators are equal; if the denominators are not equal, the student sometimes makes common denominators and then multiplies the numerators",
                "concept_probs": {
                    "make_common_denoms_and_multiply_nums": 1e5,
                    "make_common_denoms_and_add_nums": 1e5,
                    "multiply_nums_if_common_else_both": 1e5,
                    "DEFAULT": 1,
                },
            },
        },
        "multiply_generalizer": {
            "student_noise": 0.8,
            "student_concept_params": {
                "id": "multiply_generalizer",
                "single_description": "student who performs multiplication correctly, but tends to incorrectly add both numerators and denominators when adding fractions, especially when denominators are different",
                "plural_description": "Students who perform multiplication correctly, but tend to incorrectly add both numerators and denominators when adding fractions, especially when denominators are different",
                "concept_probs": {
                    "add_nums_and_denoms": 1e5,
                    "add_both_if_diff_denoms_else_nums": 1e5,
                    "multiply_nums_and_denoms": 1e5,
                    "DEFAULT": 1,
                },
            },
        },
    }

    configs = []

    set_random_seeds(0)

    for seed in [
        0,
        1,
        2,
    ]:
        students = ["add_generalizer", "multiply_generalizer"]
        student_population_params = [
            student_params[s]["student_concept_params"] for s in students
        ]
        # random shuffle so that the first student is not always the same
        random.shuffle(student_population_params)
        for s in students:
            temp_student_params = student_params[s]
            temp = base_config.copy()
            temp["seed"] = seed
            # need to copy dict so that it's not overriden later
            temp.update(temp_student_params.copy())
            population_params = student_population_params.copy()
            temp["student_populations"] = population_params

            temp2 = temp.copy()
            temp2["teaching_params"] = strategies

            configs.append(temp2)

    return configs


def get_progs():
    add_prog = "(ADD x_)"
    multiply_prog = "(MULTIPLY x_)"

    base_prog = f"(if (is_add x_) {add_prog} else {multiply_prog})"

    add_prog_1 = "make_common_denoms_and_add_nums"  # target
    add_prog_2 = "add_nums_and_denoms"
    add_prog_3 = "add_both_if_diff_denoms_else_nums"

    mult_prog_1 = "make_common_denoms_and_multiply_nums"
    mult_prog_2 = "multiply_nums_and_denoms"  # target
    mult_prog_3 = "multiply_nums_if_common_else_both"

    target_prog = base_prog.replace("ADD", add_prog_1).replace("MULTIPLY", mult_prog_2)

    progs = []

    for f in [add_prog_1, add_prog_2, add_prog_3]:
        temp = base_prog.replace("ADD", f)

        for g in [mult_prog_1, mult_prog_2, mult_prog_3]:
            temp2 = temp.replace("MULTIPLY", g)

            progs.append(temp2)

    return progs, target_prog


def run_exp(config, wandb_project="pedagogy_lists", exp_notes=None, tag=None):
    # TODO: currently hacky, setting it manually here and in bps.py
    # Better to give as an argument and use that to set both
    torch.set_default_dtype(torch.float64)

    seed = config["seed"]

    # TODO: also create the concepts data here?

    progs, target_prog = get_progs()
    str_progs = progs

    config["target_prog"] = target_prog

    env_name = config["env_name"]

    min_denom = config["min_denom"]
    max_denom = config["max_denom"]
    min_num = config["min_num"]
    max_num = config["max_num"]

    target_prog = config["target_prog"]
    print(f"Gold program:\t{target_prog}")

    # TODO: pass functions
    interp = Interpreter(env_name=env_name)

    dataset = enumerate_fraction_dataset(
        interp,
        target_prog,
        min_denom=min_denom,
        max_denom=max_denom,
        min_num=min_num,
        max_num=max_num,
    )

    # get dataset of addition examples
    add_dataset = enumerate_fraction_dataset(
        interp,
        target_prog,
        min_denom=min_denom,
        max_denom=max_denom,
        min_num=min_num,
        max_num=max_num,
        ops=["+"],
    )

    # get dataset of multiplication examples
    multiply_dataset = enumerate_fraction_dataset(
        interp,
        target_prog,
        min_denom=min_denom,
        max_denom=max_denom,
        min_num=min_num,
        max_num=max_num,
        ops=["*"],
    )

    # add dataset with common denoms
    add_dataset_common_denoms = enumerate_fraction_dataset(
        interp,
        target_prog,
        min_denom=min_denom,
        max_denom=max_denom,
        min_num=min_num,
        max_num=max_num,
        ops=["+"],
        filter_criterion="common_denoms",
    )

    # multiply dataset with common denoms
    multiply_dataset_common_denoms = enumerate_fraction_dataset(
        interp,
        target_prog,
        min_denom=min_denom,
        max_denom=max_denom,
        min_num=min_num,
        max_num=max_num,
        ops=["*"],
        filter_criterion="common_denoms",
    )

    special_datasets = {
        "add": add_dataset,
        "multiply": multiply_dataset,
        "add_common_denoms": add_dataset_common_denoms,
        "multiply_common_denoms": multiply_dataset_common_denoms,
    }

    # Get number unique possible inputs
    num_possible_inputs = dataset.get_len()
    config["num_possible_inputs"] = num_possible_inputs
    assert num_possible_inputs != 0, f"{num_possible_inputs} should not be 0"

    teaching_params_lst = config["strategies"]

    # TODO: make into general 'student_params'; what is student params?
    student_concept_params = config["student_concept_params"]

    env_name = config["env_name"]

    outputs_by_inp, tuple_progs_to_outputs_by_inp = get_outputs_by_inp(str_progs, args)
    tuple_progs = tuple(str_progs)

    set_random_seeds(seed)

    assert (
        dataset.get_len() == num_possible_inputs
    ), f"should be equal, but len(dataset): {len(dataset)}; num_possible_inputs: {num_possible_inputs}"

    if outputs_by_inp is not None:
        assert dataset.get_len() == len(
            outputs_by_inp.keys()
        ), f"should be equal, but len(dataset): {len(dataset)}; len(outputs_by_inp.keys()): {len(outputs_by_inp.keys())}"

    if tag is None:
        tags = []
    else:
        tags = [tag]

    # TODO: cache canonicalized
    inps = dataset.inputs
    canonicalized_to_prog, prog_to_canonicalized = canonicalize_progs(
        inps, interp, str_progs
    )

    # TODO: rename to canonicalized_target_progs?
    target_progs = get_all_canonicalized_progs(
        target_prog, canonicalized_to_prog, prog_to_canonicalized
    )
    # make sure only one target prog
    assert len(target_progs) == 1
    assert target_progs[0] == target_prog, f"{target_progs[0]} != {target_prog}"

    # Load list from file (if not already loaded)
    concepts = ConceptLibrary(interp)

    progs_reps = get_prog_reps(interp, str_progs, concepts)

    for t_idx, teaching_params in enumerate(teaching_params_lst):
        run_config = config.copy()
        run_config.update({"teaching_params": teaching_params})

        strategy = teaching_params["strategy"]

        sleep(1)
        print("======================================================")
        print(f"Strategy: {strategy} ({t_idx+1}/{len(teaching_params_lst)})")

        set_random_seeds(seed)

        student_concepts = get_concept_library_from_params(
            interp, student_concept_params
        )

        student = BayesianProgramSynthesizer(
            str_progs,
            interp,
            student_concepts,
            dataset,
            progs_reps=progs_reps,
            prog_to_canonicalized=prog_to_canonicalized,
            canonicalized_to_prog=canonicalized_to_prog,
            noise=config["student_noise"],
            outputs_by_inp=outputs_by_inp,
        )
        if tuple_progs not in tuple_progs_to_outputs_by_inp:
            tuple_progs_to_outputs_by_inp[tuple_progs] = student.outputs_by_inp.copy()
        write_outputs_by_inp(
            tuple_progs_to_outputs_by_inp, file_name=args.outputs_by_inp_file
        )

        wandb_name = teaching_params["id"]

        run = wandb.init(
            config=run_config,
            project=wandb_project,
            name=wandb_name,
            reinit=True,
            notes=exp_notes,
            tags=tags,
        )

        print_beliefs(student, target_prog, str_progs)

        wandb.config.update({"strategy": strategy})

        save_initial_files(
            run_config,
            student,
        )

        progs_to_eval_on_inputs = [
            ("(is_multiply x_)", "x_is_multiply"),
            ("(is_add x_)", "x_is_add"),
            ("(is_common_denoms x_)", "x_is_common_denoms"),
        ]

        gpt_helper = FractionGPTHelper()

        run_teaching_exp(
            student,
            teaching_params,
            dataset,
            target_prog,
            run_config,
            gpt_helper,
            initialize_teacher,
            initialize_gpt_args,
            progs_to_eval_on_inputs=progs_to_eval_on_inputs,
            map_inputs_to_str=True,
            special_datasets=special_datasets,
        )

        run.finish()


def parse_gpt_student_type(teacher, student_concept_params, population_params):
    # TODO: confirm, assumes that 1 is true_student_type bc populations are ordered
    # student_concept_params/population_params passed to gpt teacher; populations listed in order of student type by get_base_prompt and get_end_prompt
    true_pop_idx = population_params.index(student_concept_params)
    teacher_pop_idx = teacher.student_type - 1

    guess_id = [pop["id"] for pop in population_params][teacher.student_type - 1]
    guess_is_add_generalizer = int(guess_id == "add_generalizer")
    guess_is_correct = int(true_pop_idx == teacher_pop_idx)
    results = {
        "student_guess": teacher.student_type,
        "guess_id": guess_id,
        "guess_is_add_generalizer": guess_is_add_generalizer,
        "guess_is_correct": guess_is_correct,
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, default="pedagogy_lists")
    parser.add_argument("--exp_notes", type=str, default=None)
    parser.add_argument("--exp_tag", type=str, default=None)

    parser.add_argument(
        "--outputs_by_inp_file",
        default="tuple_progs_to_outputs_by_inp.pkl",
        type=str,
        help="File to read/write outputs_by_inp to/from",
    )

    args = parser.parse_args()

    configs = get_configs()
    print(f"Number of experiments: {len(configs)}")

    for exp_idx, config in enumerate(configs):
        print(f"Starting exp {exp_idx+1}/{len(configs)}")
        print_dict(config)
        t0 = time.time()

        run_exp(
            config,
            wandb_project=args.wandb_project,
            exp_notes=args.exp_notes,
            tag=args.exp_tag,
        )
        t1 = time.time()
        print(
            f"Finished exp {exp_idx+1}/{len(configs)} ({round((t1-t0)/60, 2)} minutes)"
        )
