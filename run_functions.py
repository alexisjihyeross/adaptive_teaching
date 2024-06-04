import itertools
from time import sleep
import time
import argparse
import wandb

from src.eval import *
from src.programs.interpreter import Interpreter
from src.programs.synthesizer import *
from src.programs.bps import *
from src.programs.utils import *
from src.programs.prior import *
from src.programs.teacher import (
    GPTProgramTeacher,
)
from src.programs.functions.teacher import initialize_teacher
from src.programs.functions.dataset import *
from src.programs.utils import run_teaching_exp
from src.programs.functions.exp_configs import *
from src.programs.concepts import *
from src.programs.prog_configs import *
from src.programs.functions.gpt_utils import *
from src.utils import set_random_seeds, print_dict, write_file


def initialize_gpt_args(config, gpt_helper, ordered_student_params):
    teaching_params = config["teaching_params"]

    base_prompt = gpt_helper.get_teacher_base_prompt(
        config["prog_concept_params"],
        config["student_concept_params"],
        assume_known_prior=teaching_params["assume_known_prior"],
    )

    end_prompt = gpt_helper.get_teacher_end_prompt(
        config["prog_concept_params"],
    )

    ordered_base_prompts = [
        gpt_helper.get_teacher_base_prompt(
            config["prog_concept_params"],
            params,
            assume_known_prior=True,
        )
        for params in ordered_student_params
    ]

    descriptions_for_populations = [
        gpt_helper.get_true_student_description(
            config["prog_concept_params"], params, tense="past"
        )
        for params in ordered_student_params
    ]

    gpt_args = {
        "base_prompt": base_prompt,
        "end_prompt": end_prompt,
        "prompts_for_populations": ordered_base_prompts,
        "descriptions_for_populations": descriptions_for_populations,
    }

    return gpt_args


def get_num_possible_inputs(hyp_params):
    """Helper function to get the number of possible inputs.
    Works when input is dict with min_input_val and max_input_val.
    """
    num_possible_inputs = len(
        list(range(hyp_params["min_input_val"], hyp_params["max_input_val"]))
    )
    assert num_possible_inputs != 0, f"{num_possible_inputs} should not be 0"
    return num_possible_inputs


def run_exp(config, wandb_project="pedagogy_lists", exp_notes=None, tag=None):
    # TODO: currently hacky, setting it manually here and in bps.py
    # Better to give as an argument and use that to set both
    torch.set_default_dtype(torch.float64)

    hyp_params = get_concept_config(config["prog_concept_params"])

    num_possible_inputs = get_num_possible_inputs(hyp_params)
    config["num_possible_inputs"] = num_possible_inputs

    teaching_params_lst = config["strategies"]

    # writes the programs to a file
    str_progs, _, _ = get_concept_data(
        config["prog_concept_params"],
        write_data=True,
        out_dir="results/functions",
    )

    outputs_by_inp, tuple_progs_to_outputs_by_inp = get_outputs_by_inp(str_progs, args)
    tuple_progs = tuple(str_progs)

    set_random_seeds(config["seed"])

    print("CONFIG")
    print_dict(config)

    target_prog = hyp_params["target_prog"]
    print(f"Gold program:\t{target_prog}")

    assert (
        len([p for p in str_progs if p == target_prog]) == 1
    ), f"There should be exactly one gold program but there are {len(gold_df)}"

    # used for determining if an example falls into the "fx half"
    fx_half_prog = config["prog_concept_params"]["fx_half_prog"]

    # TODO: pass functions
    interp = Interpreter(env_name=config["env_name"])

    dataset = enumerate_dataset(
        interp,
        target_prog,
        min_val=hyp_params["min_input_val"],
        max_val=hyp_params["max_input_val"],
    )

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

    # If inputs are ints, enumerate all ints from min_input_val to max_input_val
    inps = list(range(hyp_params["min_input_val"], hyp_params["max_input_val"]))
    canonicalized_to_prog, prog_to_canonicalized = canonicalize_progs(
        inps, interp, str_progs
    )

    # Make sure that when "canonicalizing" hypotheses (i.e. treating all programs that give the same outputs as equal), there is only one gold program
    canonicalized_target_progs = get_all_canonicalized_progs(
        target_prog, canonicalized_to_prog, prog_to_canonicalized
    )

    # Make sure there is only one gold program
    assert (
        len(canonicalized_target_progs) == 1
    ), f"len(gold_pcanonicalized_target_progsrogs): {len(canonicalized_target_progs)}"

    # Load list from file (if not already loaded)
    concepts = ConceptLibrary(interp)

    # Get representations of programs (used for getting prior over programs)
    progs_reps = get_prog_reps(interp, str_progs, concepts)

    for t_idx, teaching_params in enumerate(teaching_params_lst):
        run_config = config.copy()
        run_config.update({"teaching_params": teaching_params})

        strategy = teaching_params["strategy"]

        sleep(1)
        print("======================================================")
        print(f"STRATEGY: {strategy} ({t_idx+1}/{len(teaching_params_lst)})")

        set_random_seeds(config["seed"])

        student_concepts = get_concept_library_from_params(
            interp, config["student_concept_params"]
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
        wandb.config.update({"strategy": strategy})

        save_initial_files(
            run_config, student, extra_files_to_save=[hyp_params["prog_file"]]
        )

        progs_to_eval_on_inputs = [
            (fx_half_prog, "x=fx?"),
        ]
        gpt_helper = FunctionGPTHelper(dataset)

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
        )

        run.finish()


def parse_gpt_student_type(teacher):
    """Parse the student type from the GPT teacher."""
    guess_is_fx_knower = int(teacher.student_type) == 2
    guess_is_correct = int(
        guess_is_fx_knower == config["student_concept_params"]["fx_knower"]
    )
    results = {
        "guess_is_fx-knower": int(guess_is_fx_knower),
        "guess_is_correct": guess_is_correct,
        "student_guess": teacher.student_type,
    }
    return results


def enumerate_configs_from_exp(
    args,
    exp,
    keys_to_ignore=[
        "strategies",
        "student_concept_params",
    ],
):
    """
    Helper function to enumerate all possible configurations from an experiment.

    For each experiment:
    - Iterate through all possible combinations of hyperparameters *within* each experiment, excluding the keys_to_ignore.
    - Create student_concept_params and student_populations
    """
    configs = []

    print(f"Reading prog concept params from file: ", args.prog_concept_params_file)
    prog_concept_params = read_json(args.prog_concept_params_file)

    temp_config = {k: v for k, v in exp.items() if k not in keys_to_ignore}
    # Iterate through all possible combinations of hyperparameters (except keys_to_ignore, which are kept the same)
    for combo in itertools.product(*temp_config.values()):
        config = dict(zip(temp_config.keys(), combo))
        config.update({k: exp[k] for k in keys_to_ignore})

        # Add prog_concept_params and prog_concept_params_file
        config.update({"prog_concept_params": prog_concept_params})
        config.update({"prog_concept_params_file": args.prog_concept_params_file})

        # if student_concept_params is None, get all student concepts for this automatically (by calling get_student_concept_params) and run all
        if config["student_concept_params"] is None:
            student_concept_params_lst = get_student_concept_params(
                config["prog_concept_params"],
            )
            for student_concept_params in student_concept_params_lst:
                temp_temp_config = config.copy()
                temp_temp_config["student_concept_params"] = student_concept_params

                # also store all possible student concept configurations, used for initializing teacher
                student_populations = student_concept_params_lst.copy()
                temp_temp_config["student_populations"] = student_populations
                configs.append(temp_temp_config)

    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, default="pedagogy_lists")
    parser.add_argument("--exp_id", type=str, default="all")
    parser.add_argument("--exp_notes", type=str, default=None)
    parser.add_argument("--exp_tag", type=str, default=None)

    parser.add_argument(
        "--prog_concept_params_file", default="src/bps/prog_concepts/0.json"
    )

    parser.add_argument(
        "--outputs_by_inp_file",
        default="tuple_progs_to_outputs_by_inp.pkl",
        type=str,
        help="File to read/write outputs_by_inp to/from",
    )

    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        help="File to read config from. If not None, will override all other args. If None, will run all experiments given by exp_id.",
    )

    args = parser.parse_args()

    experiment = EXPERIMENTS[args.exp_id]

    if args.config_file is not None:
        print(f"Reading config from file: {args.config_file}")
        config = read_json(args.config_file)
        # print_dict(config)

        run_exp(
            config,
            wandb_project=args.wandb_project,
            exp_notes=args.exp_notes,
            tag=args.exp_tag,
        )

    else:
        configs = enumerate_configs_from_exp(args, experiment)

        for exp_idx, config in enumerate(configs):
            print(f"Starting exp {exp_idx+1}/{len(configs)}")
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
