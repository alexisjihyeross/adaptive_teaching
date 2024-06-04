from src.programs.prog_configs import *
from src.programs.functions.teacher import *
from src.programs.functions.dataset import *
from src.programs.interpreter import *
from src.programs.bps import *
from src.programs.concepts import *
from src.programs.functions.exp_configs import EXPERIMENTS
from src.programs.functions.gpt_utils import *
from src.programs.utils import (
    write_outputs_by_inp,
    freeze,
    get_populations,
    initialize_teacher_from_params,
)
from run_functions import get_num_possible_inputs, initialize_gpt_args

import glob
from flask import Flask, render_template, request, redirect, url_for, jsonify
import uuid
import time
from random import Random
import pickle
import os
import argparse
import numpy as np
import json
import dill

from flask import Flask, session
from flask_session import Session

app = Flask(__name__, static_url_path="/static")
app.config["SESSION_TYPE"] = "filesystem"
# app.config["SECRET_KEY"] = "\xf2\xf2\x10j\x01\xb9\xce\xca\xd4T\x04\x88"

Session(app)


# TODO: can't use parse_args with gnunicorn?
def parse_args(return_defaults=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prog_concept_params_dir",
        default="src/programs/functions/target_concepts",
    )
    parser.add_argument(
        "--concepts_to_include",
        # ones in pilot: TODO: confirm
        default=[6, 5, 22, 1, 7, 13, 2, 14, 19, 18, 20],
        nargs="+",
        type=int,
        help="Concept ids to include (corresponding to the i.json files in prog_concept_params_dir).",  # TODO: confirm
    )

    parser.add_argument(
        "--seeds",
        default=[0, 1, 2, 3, 4],
        type=int,
        nargs="+",
        help="Seeds to use for random state. Assumes enumerating all possible exps + seeds (ie not sampling)",
    )

    parser.add_argument(
        "--num_concepts_to_sample",
        default=None,
        type=int,
        help="Number of prog_concepts to sample from the prog_concept_params_dir. If None, then enumerate all possible prog_concepts from the filtered concepts (concepts_to_include).",
    )

    # whether to use existing concepts and not re-sample prog_concept_params and get new experiments_and_seeds
    # TODO: check logic
    parser.set_defaults(use_existing_concepts=False)

    parser.add_argument(
        "--debug", dest="debug", action="store_true", help="Debug mode."
    )
    parser.set_defaults(debug=False)

    parser.add_argument(
        "--exp_dir",
        default="results/human/experiments",
        type=str,
        help="Where to store results.",
    )

    # TODO: Different file for reading vs writing?
    parser.add_argument(
        "--outputs_by_inp_file",
        default="tuple_progs_to_outputs_by_inp.json",
        type=str,
        help="File to read/write outputs_by_inp to/from",
    )
    parser.add_argument("--parsed_progs_file", default=None, type=str)

    if return_defaults:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    return args


def initialize_session(args):
    """Initialize meta info for experiment, such as the user id and other session variables that are used to track each individual human's experiment data.

    Tracks:
    - user_id: unique id for each user (uses global user_idx)
    - user_idx: global variable to create unique user ids (increments by 1 each time a new user is created)

    Using user_id, initializes the following variables:
    - user_timers_started: dict of user_id to whether the timer (associated with chat time) has started
    - user_random_states: dict of user_id to random state
    - user_teachers: dict of user_id to teacher
    - user_session_data: dict of user_id to user session data (e.g. chat messages, guesses, etc.)
        - Includes exp_id, which is the id of the experiment (index into all_experiments_with_seeds)

    Also, add exp_id to:
    - experiments_and_start_times: dict of exp_id to list of start times for each experiment
    - incomplete_experiments: set of exp_ids for incomplete experiments
    """

    global user_idx

    user_id = user_idx

    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    if "user_id" in session:
        # Write to global exp_dir in ALL_index/user_id to keep track of all experiments that aren't cleared at index (bc then cleared) -- used to know if something went wrong (should be empty)
        sub_dir = os.path.join(args.exp_dir, "ALL_index", str(user_id))
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        # Write experiment data to exp_dir/start subdirectory
        write_exp(sub_dir)
    clear_session()

    session["user_id"] = user_id

    # set the user timer started to be False
    user_timers_started[user_id] = False

    user_random_states[user_id] = Random()
    user_session_data[user_id] = {}

    user_idx += 1

    user_session_data[user_id]["visited_chat"] = False
    user_session_data[user_id]["started_learning"] = False
    user_session_data[user_id]["reached_max_possible_inputs"] = False

    user_session_data[user_id]["finish_button_shown"] = False
    user_session_data[user_id]["finish_button_shown_info"] = []

    # Store Prolific IDs (there should only be more than 1 if user reloads the first page)
    user_session_data[user_id]["prolific_ids"] = []

    user_session_data[user_id]["chat_messages"] = []
    user_session_data[user_id]["guesses"] = []
    user_session_data[user_id]["raw_guesses"] = []
    user_session_data[user_id]["task_questions"] = []
    user_session_data[user_id]["end_questions"] = []

    user_session_data[user_id]["exp_start_time"] = time.time()

    set_exp_id()
    exp_id = get_exp_id()
    # Set start time
    experiments_and_start_times[exp_id].append(time.time())
    # Add experiment to incomplete_experiments
    incomplete_experiments.add(exp_id)

    session.modified = True


def enumerate_exps(args):
    """
    Enumerate all possible experiments based on the config files in the prog_concept_param_files in prog_concept_params_dir.

    Returns all_experiments_seeds, a list of tuples (config, seed), for all combinations of:
    - prog_concept_params
    - student parameters
    - teaching strategies in the config
    - seeds
    """

    print("Enumerating experiments...")

    # List of configs
    all_experiments = []

    seeds = args.seeds

    print("Reading prog concept param files from:", args.prog_concept_params_dir)
    prog_concept_param_files = glob.glob(f"{args.prog_concept_params_dir}/*.json")
    print(len(prog_concept_param_files), prog_concept_param_files)

    # Filter out by if in concepts_to_include (should filter if ends with {idx}.json where idx is the concept id)
    print(f"Filtering out concepts to filter: {args.concepts_to_include}")
    # Only include the ones that are in concepts_to_include
    prog_concept_param_files = [
        f
        for f in prog_concept_param_files
        if any([f.endswith(f"/{idx}.json") for idx in args.concepts_to_include])
    ]

    # Filter out the ones that have 'meta_info' in them
    prog_concept_param_files = [
        f for f in prog_concept_param_files if "meta_info" not in f
    ]
    print(
        "Num prog concept param files:",
        len(prog_concept_param_files),
        prog_concept_param_files,
    )

    # This seed used for sampling the prog_concept_param_files
    random.seed(0)
    # If args.num_concepts_to_sample is None, then enumerate all possible prog_concepts in prog_concept_param_files
    # Else, sample args.num_concepts_to_sample prog_concepts from prog_concept_param_files

    if args.num_concepts_to_sample is not None:
        # Sample without replacement
        prog_concept_param_files = random.sample(
            prog_concept_param_files, args.num_concepts_to_sample
        )

    for prog_concept_params_file in prog_concept_param_files:
        config = cleaned_config.copy()

        # TODO: hacky, but add prog_concept_params_file to config
        config["prog_concept_params_file"] = prog_concept_params_file

        prog_concept_params = read_json(prog_concept_params_file)

        # print target prog
        print(prog_concept_params["id"])

        config["prog_concept_params"] = prog_concept_params

        student_concept_params_lst = get_student_concept_params(
            config["prog_concept_params"],
        )
        student_populations = student_concept_params_lst.copy()

        for student_concept_params in student_populations:
            temp_config = config.copy()

            temp_config["student_concept_params"] = student_concept_params
            temp_config["student_populations"] = student_populations

            for teaching_params in temp_config["strategies"]:
                temp_temp_config = temp_config.copy()
                temp_temp_config["teaching_params"] = teaching_params.copy()

                all_experiments.append(temp_temp_config)

    # all_experiments is a list of tuples (config, seed). Enumerate product of all_experiments and seeds. Ordered by seeds, i.e. all seed 0 first, then seed 1, etc.
    all_experiments_seeds = []
    for seed in seeds:
        for config in all_experiments:
            all_experiments_seeds.append((config, seed))
    assert len(all_experiments_seeds) == len(all_experiments) * len(seeds)

    print("Num total experiments:", len(all_experiments_seeds))
    return all_experiments_seeds


def initialize_interpreter(config, hyp_params):
    """Helper function to initialize the interpreter based on the config and hyp_params."""
    assert (
        config["env_name"] == "function"
    ), f'env_name: {config["env_name"]} not supported'

    functions = ["even", "odd", "positive", "prime", "exp", "multiply"]
    functions.extend([f"divisible_{n}" for n in hyp_params["divisible_options"]])
    functions.extend([f"greater_{n}" for n in hyp_params["greater_options"]])
    functions.extend([f"add_{n}" for n in hyp_params["poly_kwargs"]["constants"]])

    interp = Interpreter(config["env_name"], functions=functions)
    return interp


def initialize_experiment(all_experiments_with_seeds, do_print=False):
    """
    - Initialize the config for the experiment.
    - Set the random state for user based on the seed for this experiment.
    - Update other session variables like hints, function options, etc.
    """

    user_id = session["user_id"]

    exp_id = get_exp_id()

    config, seed = all_experiments_with_seeds[exp_id]

    if do_print:
        print(f"user_id: {user_id}")
        print("gold prog id:", config["prog_concept_params"]["id"])
        print("teaching params:", config["teaching_params"])
        print("fx_knower=", config["student_concept_params"]["fx_knower"])
        print("seed:", seed)
        print_dict(config["prog_concept_params"])

    # Set random seed with seed for this experiment
    user_random_states[user_id].seed(seed)

    hyp_params = get_concept_config(
        config["prog_concept_params"],
    )

    num_possible_vals = get_num_possible_inputs(hyp_params)
    config["num_possible_input_vals"] = num_possible_vals
    config["seed"] = seed

    interp = initialize_interpreter(config, hyp_params)
    target_prog = hyp_params["target_prog"]

    dataset = enumerate_dataset(
        interp,
        target_prog,
        input_type=get_input_type(config["env_name"]),
        max_val=config["prog_concept_params"]["max_input_val"],
        min_val=config["prog_concept_params"]["min_input_val"],
    )

    assert (
        dataset.get_len() == num_possible_vals
    ), f"should be equal, but len(dataset): {len(dataset)}; num_possible_vals: {num_possible_vals}"

    str_progs, fs, _ = get_concept_data(
        config["prog_concept_params"],
        write_data=True,
        out_dir="results/human/functions",
    )

    student_populations = config["student_populations"]
    real_student_params = config["student_concept_params"]
    if do_print:
        print(f"Num student populations: {len(student_populations)}")
        for s in student_populations:
            print(s)

        print("Real student:")
        print(real_student_params)

    config["target_prog"] = target_prog

    student_concepts = get_concept_library_from_params(interp, real_student_params)
    tuple_progs = tuple(str_progs)

    concepts = ConceptLibrary(interp)

    parsed_progs = [interp.parse(p) for p in tqdm(str_progs)]
    progs_reps = torch.stack(
        [
            get_prog_vector(parsed, concepts.get_concept_dict())
            for parsed in tqdm(parsed_progs)
        ]
    )

    noise = config["student_noise"]
    outputs_by_inp = (
        tuple_progs_to_outputs_by_inp[tuple_progs]
        if tuple_progs in tuple_progs_to_outputs_by_inp
        else None
    )
    if outputs_by_inp is not None:
        assert dataset.get_len() == len(
            outputs_by_inp.keys()
        ), f"should be equal, but len(dataset): {len(dataset)}; len(outputs_by_inp.keys()): {len(outputs_by_inp.keys())}"

    student = BayesianProgramSynthesizer(
        str_progs,
        interp,
        student_concepts,
        dataset,
        progs_reps=progs_reps,
        noise=noise,
        outputs_by_inp=outputs_by_inp,
        do_print=False,
    )
    if tuple_progs not in tuple_progs_to_outputs_by_inp:
        tuple_progs_to_outputs_by_inp[tuple_progs] = student.outputs_by_inp.copy()
        # Write to json file "tuple_progs_to_outputs_by_inp.json"

    write_outputs_by_inp(
        tuple_progs_to_outputs_by_inp,
        file_name=os.path.join(args.exp_dir, args.outputs_by_inp_file),
    )

    teaching_params = config["teaching_params"]

    user_session_data[user_id]["config"] = config

    gpt_helper = FunctionGPTHelper(dataset)
    teacher, _ = initialize_teacher_from_params(
        student,
        teaching_params,
        dataset,
        target_prog,
        config,
        gpt_helper,
        initialize_teacher,
        initialize_gpt_args,
        random_state=user_random_states[user_id],
    )

    user_teachers[user_id] = teacher

    user_session_data[user_id]["current_output_streak"] = 0
    user_session_data[user_id]["num_correct_output_preds"] = 0
    user_session_data[user_id]["num_total_output_preds"] = 0

    strategy = teaching_params["strategy"]

    if strategy in PROBABILISTIC_STRATEGIES:
        real_pop_idx = get_pop_idx(teacher.populations, student)
        user_session_data[user_id]["real_pop_idx"] = real_pop_idx

        print("real_pop_idx:", real_pop_idx)

        # TODO: the code assumes the real student is the first population?
        assert real_pop_idx == 0, f"real_pop_idx: {real_pop_idx}"

        # initialize with the guess before any data

        curr_guess_idx = teacher.student_pop_idx
        real_student_params = user_session_data[user_id]["config"][
            "student_concept_params"
        ]
        curr_guess_name = get_student_name(curr_guess_idx, real_student_params)
        user_session_data[user_id]["student_pop_guesses"] = [
            (curr_guess_idx, curr_guess_name)
        ]
        print("starting guess of student_pop_idx:", curr_guess_idx, curr_guess_name)

    user_session_data[user_id]["calculator_usage"] = []

    initialize_options_and_hints(config, hyp_params, fs)

    # get the experiment sub_dir
    prolific_ids = user_session_data[user_id]["prolific_ids"]
    exp_dir = get_exp_dir(args.exp_dir, prolific_ids[0])
    assert os.path.exists(exp_dir)
    user_session_data[user_id]["exp_dir"] = exp_dir

    session.modified = True


def initialize_options_and_hints(config, hyp_params, fs):
    """Initialize the function options and hints for the user based on the config and hyp_params."""
    # TODO: double check against initialize_interpreter
    user_id = session["user_id"]

    function1_divisible_options = hyp_params["divisible_options"]
    function1_greater_options = hyp_params["greater_options"]

    fs_from_options = [
        f
        for f in fs
        if not any(
            [
                f.startswith(d)
                for d in [
                    "divisible",
                    "greater",
                ]
            ]
        )
    ]
    fs_from_options.extend([f"(divisible_{n} x_)" for n in function1_divisible_options])
    fs_from_options.extend([f"(greater_{n} x_)" for n in function1_greater_options])

    assert set(fs_from_options) == set(
        fs
    ), f"fs_from_options: {set(fs_from_options)}\nfs: {set(fs)}"
    assert len(set(fs_from_options)) == len(
        set(fs)
    ), f"len fs_from_options: {len(set(fs_from_options))}, len fs: {len(set(fs))}"

    function1_options = get_function1_options()
    function2a_options = get_function2a_options(config)
    function2b_options = get_function2b_options(config)

    user_session_data[user_id]["function1_options"] = function1_options
    user_session_data[user_id]["function2a_options"] = function2a_options
    user_session_data[user_id]["function2b_options"] = function2b_options
    user_session_data[user_id][
        "function1_divisible_options"
    ] = function1_divisible_options
    user_session_data[user_id]["function1_greater_options"] = function1_greater_options

    hint1, hint2 = get_hints()
    user_session_data[user_id]["hint1"] = hint1
    user_session_data[user_id]["hint2"] = hint2

    # TODO: add session.modified = True?


def get_function1_options():
    # TODO: connect to the prog_concept_params; for now, just hardcode
    options = [
        "prime",
        "positive",
        "even",
        "odd",
        "divisible by __",
        "greater than __",
    ]
    return options


def get_function2b_options(config):
    """Helper function to get the options for guesses for b in ax+b based on the config."""
    constant_kwargs = config["prog_concept_params"]["poly_kwargs"]["constants"]
    # Have to return both since the hyp space has both add/subtract constants
    options = [-1 * c for c in constant_kwargs]
    options.extend(constant_kwargs)
    return options


def get_function2a_options(config):
    """Helper function to get the options for guesses for a in ax+b based on the config."""
    coef_kwargs = config["prog_concept_params"]["poly_kwargs"]["coefficients"]
    return coef_kwargs


@app.route("/")
def index():
    """
    Loads the index page.
    - Check if user has submitted prolific id; if so, redirect to task1.
    - If not, initialize the session: calls initialize_session().
    """
    # Check if session exists and user session data has been created for user AND prolific ids have been submitted
    if (
        "user_id" in session
        and session["user_id"] in user_session_data
        and "prolific_ids" in user_session_data[session["user_id"]]
        and len(user_session_data[session["user_id"]]["prolific_ids"]) > 0
    ):
        print("Redirecting to task1...")
        return redirect(url_for("task1"))

    print("Initializing session...")
    initialize_session(args)
    print("Done initializing session.")
    return render_template("index.html")


@app.route("/collect_prolific_id", methods=["POST"])
def collect_prolific_id():
    """
    Collects the prolific id from the user and redirects to task1.
    - Adds the prolific id to the user session data.
    - Initializes the experiment: calls initialize_experiment().
    """
    id = request.form["prolific_id"]
    prolific_ids = user_session_data[session["user_id"]]["prolific_ids"]
    prolific_ids.append(id)
    user_session_data[session["user_id"]]["prolific_ids"] = prolific_ids

    session.modified = True
    initialize_experiment(all_experiments_with_seeds)
    return redirect(url_for("task1"))


@app.route("/task1")
def task1():
    """Loads the task1 page."""
    user_id = session["user_id"]
    remaining_time = get_remaining_time(user_id)
    remaining_time = get_formatted_time(remaining_time)

    return render_template(
        "task1.html",
        prolific_ids=user_session_data[user_id]["prolific_ids"],
        chat_messages=user_session_data[user_id]["chat_messages"],
        time=remaining_time,
    )


@app.route("/task2")
def task2():
    """Loads the task2 page."""
    user_id = session["user_id"]
    remaining_time = get_remaining_time(user_id)
    remaining_time = get_formatted_time(remaining_time)

    return render_template(
        "task2.html",
        prolific_ids=user_session_data[user_id]["prolific_ids"],
        chat_messages=user_session_data[user_id]["chat_messages"],
        time=remaining_time,
    )


@app.route("/task3")
def task3():
    """Loads the task3 page."""
    button_text = (
        "Next"
        if not user_session_data[session["user_id"]]["started_learning"]
        else "Next"
    )
    user_id = session["user_id"]
    remaining_time = get_remaining_time(user_id)
    remaining_time = get_formatted_time(remaining_time)
    hint1, hint2 = get_hints()

    return render_template(
        "task3.html",
        prolific_ids=user_session_data[user_id]["prolific_ids"],
        chat_messages=user_session_data[user_id]["chat_messages"],
        button_text=button_text,
        time=remaining_time,
        hint1=hint1,
        hint2=hint2,
    )


@app.route("/task4")
def task4():
    """Loads the task4 page."""
    button_text = (
        "Next"
        if not user_session_data[session["user_id"]]["started_learning"]
        else "Next"
    )
    user_id = session["user_id"]
    remaining_time = get_remaining_time(user_id)
    remaining_time = get_formatted_time(remaining_time)
    hint1, hint2 = get_hints()

    return render_template(
        "task4.html",
        prolific_ids=user_session_data[user_id]["prolific_ids"],
        chat_messages=user_session_data[user_id]["chat_messages"],
        button_text=button_text,
        time=remaining_time,
        hint1=hint1,
        hint2=hint2,
    )


@app.route("/task5")
def task5():
    """Loads the task5 page."""
    button_text = (
        "Next"
        if not user_session_data[session["user_id"]]["started_learning"]
        else "Next"
    )
    user_id = session["user_id"]
    remaining_time = get_remaining_time(user_id)
    remaining_time = get_formatted_time(remaining_time)
    hint1, hint2 = get_hints()

    return render_template(
        "task5.html",
        prolific_ids=user_session_data[user_id]["prolific_ids"],
        chat_messages=user_session_data[user_id]["chat_messages"],
        button_text=button_text,
        time=remaining_time,
        hint1=hint1,
        hint2=hint2,
    )


def get_exp_dir(exp_dir, prolific_id):
    """Helper function that generate sub directory based on prolific id and date/time. Date time should have month, date, year, hour, minute, second. Used to store the experiment data for each user."""

    # Get current date and time
    current_date_time = time.strftime("%m-%d-%y-%H.%M.%S")

    # Combine the current date and time with the prolific id and file extension to create the filename
    stub = prolific_id + "_" + current_date_time
    sub_dir = os.path.join(exp_dir, stub)

    # Check if the sub directory already exists in the directory
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    else:
        raise ValueError(f"Sub directory already exists: {sub_dir}")
    return sub_dir


# Write current session
def write_exp(exp_dir):
    """
    exp_dir: the directory to write the experiment data to (ie for specific participant)

    - Writes exp_data to exp_data.pkl file in exp_dir
    - Writes outputs_by_inp file in exp_dir
    """

    # Save the keys and values of the session dictionary
    # If the important keys are not in the session dictionary, then give a warning
    important_keys = [
        "user_id",
        "prolific_ids",
        "chat_messages",
        "guesses",
        "raw_guesses",
        "end_questions",
        "task_questions",
        "config",
        "exp_id",
    ]

    exp_data = {}

    # If user_session_data hasn't been created for this user id, then nothing to write
    if session["user_id"] in user_session_data:
        print(f"User id: {session['user_id']}")
        for k, v in user_session_data[session["user_id"]].items():
            if k in ["end_questions", "task_questions"]:
                k += "_answers"
            exp_data.update({k: v})

        for k in important_keys:
            if k not in exp_data:
                print(f"WARNING: {k} not in session dictionary")

        # if strategy is gpt, then also save the base/end prompts and messages/parsed_messages
        # Need to check if config is in session bc might not be if, say, writing to exp_data at the start of the experiment
        if "config" in user_session_data[session["user_id"]] and user_session_data[
            session["user_id"]
        ]["config"]["teaching_params"]["strategy"] in ["gpt", "gpt+probabilistic"]:
            if session["user_id"] in user_teachers:
                teacher = user_teachers[session["user_id"]]
                exp_data.update(
                    {
                        "base_prompt": teacher.base_prompt,
                        "end_prompt": teacher.end_prompt,
                        "gpt_messages": teacher.messages,
                        "parsed_gpt_messages": teacher.parsed_messages,
                        "cost": teacher.get_cost(),
                    }
                )
            else:
                # Not sure why this would happen
                print("user id not in user_teachers")
                print("user teachers:", user_teachers)
                print("user session data:", user_session_data)
                print(
                    "user session user id:",
                    user_session_data[session["user_id"]]["user_id"],
                )
        else:
            print("config not in session or strategy not gpt")

    assert os.path.exists(exp_dir)
    file_name = os.path.join(exp_dir, "exp_data.pkl")

    with open(file_name, "wb") as file:
        print(f"Writing exp data to file: {file_name}")
        dill.dump(exp_data, file)

    exp_data_read = read_exp_data(file_name)
    # can only load some of the values with json
    keys_to_read = [
        "user_id",
        "prolific_ids",
        "chat_messages",
        "guesses",
        "raw_guesses",
        "end_question_answers",
        "config",
        "exp_end_time",
        "exp_start_time",
    ]
    sub_dict = {k: exp_data_read[k] for k in keys_to_read if k in exp_data_read}
    # if some k not in exp_data_read, give warning
    for k in keys_to_read:
        if k not in sub_dict:
            print(f"WARNING: {k} not in exp_data_read")

    # Write outputs_by_inp to json file
    write_outputs_by_inp(
        tuple_progs_to_outputs_by_inp,
        file_name=os.path.join(exp_dir, "tuple_progs_to_outputs_by_inp.json"),
    )


@app.route("/clear_session")
def clear_session():
    """Clear the session."""
    session.clear()
    session.modified = True

    # TODO: also clear user_session_data? for now, not because maybe just want to reset the user_id
    return "Session cleared"


def get_formatted_fx(fx):
    """Helper function to format the f(x) options for display."""
    if fx.startswith("divisible"):
        func, num = parse_function_with_value(fx)
        assert func == "divisible"
        fx = f"<b>divisible by <span class=''>{num}</span></b>"
    elif fx.startswith("greater"):
        func, num = parse_function_with_value(fx)
        assert func == "greater"
        fx = f"<b>greater than <span class=''>{num}</span></b>"
    else:
        fx = f"<b>{fx}</b>"
    return fx


def get_hints():
    """Helper function to get the hints for the user based on the config."""
    config = user_session_data[session["user_id"]]["config"]
    is_fx_knower = bool(config["student_concept_params"]["fx_knower"])

    fx_special_concept = config["prog_concept_params"]["fx_special_concept"]
    gx_special_concept = config["prog_concept_params"]["gx_special_concept"]

    gx_incorrect_concept = config["prog_concept_params"]["gx_spurious"]

    if is_fx_knower:
        hint1 = f"<span class='wug'>wug</span> is undefined when inputs are <b>{get_formatted_fx(fx_special_concept)}</b>"

        if gx_incorrect_concept.startswith("add"):
            # Parse to get val
            _, num = parse_function_with_value(gx_incorrect_concept)
            hint2 = f"When <span class='wug'>wug</span> is defined, <b><span class='constant'>b</span> = {num}</b>"
        else:
            raise ValueError(gx_special_concept)
    else:
        fx_incorrect_concept = config["prog_concept_params"]["fx_spurious"]

        hint1 = f"<span class='wug'>wug</span> is undefined when inputs are {get_formatted_fx(fx_incorrect_concept)}"

        if gx_special_concept.startswith("add"):
            # Parse to get val
            _, num = parse_function_with_value(gx_special_concept)
            hint2 = f"When <span class='wug'>wug</span> is defined, <b><span class='constant'>b</span> = {num}</b>"
        else:
            raise ValueError(gx_special_concept)
    return hint1, hint2


def get_answer():
    """Helper function to get the answer to display at the end page."""
    config = user_session_data[session["user_id"]]["config"]
    fx_special_concept = config["prog_concept_params"]["fx_special_concept"]
    a = config["prog_concept_params"]["gx_a"]
    b = config["prog_concept_params"]["gx_b"]
    answer1 = f"<span class='wug'>wug</span> is undefined when inputs are {get_formatted_fx(fx_special_concept)}"
    answer2 = f"When <span class='wug'>wug</span> is defined, it computes <span class='math'><span class='constant'>a</span>&lowast;x+<span class='constant'>b</span></span> where <b><span class=''><span class='constant'>a</span> = {a}</span></b> and <b><span class=''><span class='constant'>b</span> = {b}</span></b>"
    return answer1, answer2


@app.route("/end")
def end():
    answer1, answer2 = get_answer()
    user_id = session["user_id"]
    user_session_data[user_id]["exp_end_time"] = time.time()
    session.modified = True

    sub_dir = os.path.join(
        user_session_data[user_id]["exp_dir"], "end", time.strftime("%H.%M.%S")
    )
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    # Write experiment data to exp_dir/end subdirectory
    write_exp(sub_dir)

    # Remove experiment from incomplete_experiments
    exp_id = get_exp_id()
    try:
        incomplete_experiments.remove(exp_id)
    except:
        print("Error removing exp id from incomplete_experiments")
        print("exp_id", exp_id)
        print("incomplete_experiments", incomplete_experiments)
    complete_experiments.add(exp_id)

    print(
        f"# complete experiments: {len(complete_experiments)}/{len(all_experiments_with_seeds)}"
    )

    with open(complete_exps_file, "wb") as file:
        print(f"writing all_experiments_with_seeds to file: {complete_exps_file}")
        dill.dump(complete_experiments, file)

    # TODO: have incomplete experiments combined with remaining experiments somewhere?
    print(f"# incomplete experiments: {len(incomplete_experiments)}")
    print("incomplete experiments:", incomplete_experiments)

    clear_session()

    return render_template(
        "end.html", completion_code="C166E0FJ", answer1=answer1, answer2=answer2
    )


def get_exp_id():
    user_id = session["user_id"]
    return user_session_data[user_id]["exp_id"]


# TODO: set this to be based on a combination of remaining experiments AND incomplete experiments?
def set_exp_id():
    """
    If user_id < len(remaining_experiments), then use user_id
    Else, sample from incomplete_experiments (set) based on the start times in experiments_and_start_times (dict) (get the earliest *last* start time)

    TODO: what happens if there is the same experiment with different start times?
    """
    user_id = session["user_id"]
    if user_id < len(remaining_experiments):
        print(
            f"Setting exp id to user id because user_id ({user_id}) < len(remaining_experiments) ({len(remaining_experiments)})"
        )
        exp_id = remaining_experiments[user_id]
    else:
        print("Sampling from incomplete experiments...")
        # Get the earliest experiment
        incomplete_experiments_and_start_times = {
            exp_id: experiments_and_start_times[exp_id]
            for exp_id in incomplete_experiments
        }
        # The start times are a list of all the start times, take the min of the *LAST* start time for each
        exp_id = min(
            incomplete_experiments_and_start_times,
            key=lambda x: incomplete_experiments_and_start_times[x][-1],
        )
        print("incomplete experiments:", incomplete_experiments)
        print(
            "incomplete_experiments_and_start_times:",
            incomplete_experiments_and_start_times,
        )
        print("exp_id:", exp_id)

    # session["exp_id"] = exp_id
    user_session_data[user_id]["exp_id"] = exp_id
    session.modified = True


def read_exp_data(file_name):
    print(f"Reading exp data from {file_name}")
    return dill.load(open(file_name, "rb"))


@app.route("/check")
def check():
    hint1, hint2 = get_hints()
    user_id = session["user_id"]
    remaining_time = get_remaining_time(user_id)
    # session["time_enter_check"] = elapsed_time
    user_session_data[user_id]["time_enter_check"] = remaining_time
    session.modified = True
    print("minutes remaining when entering check: ", remaining_time / 60)
    stop_timer()

    # Write exp
    sub_dir = os.path.join(
        user_session_data[user_id]["exp_dir"], "check", time.strftime("%H.%M.%S")
    )
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    # Write experiment data to exp_dir/chat/{time} subdirectory
    write_exp(sub_dir)

    return render_template("check.html", hint1=hint1, hint2=hint2)


def get_guess_values(last_guess):
    """Helper function to set the default values for the guess select bars"""
    if last_guess is not None:
        function1Selected = last_guess["function1"]

        null_values = ["null", "no_guess", "--", "None", ""]

        function1DivisibleSelected = (
            int(last_guess["function1Divisible"])
            if str(last_guess["function1Divisible"]) not in null_values
            else "--"
        )

        function1GreaterSelected = (
            int(last_guess["function1Greater"])
            if str(last_guess["function1Greater"]) not in null_values
            else "--"
        )

        function2ASelected = (
            int(last_guess["function2A"])
            if str(last_guess["function2A"]) not in null_values
            else "--"
        )

        function2BSelected = (
            int(last_guess["function2B"])
            if str(last_guess["function2B"]) not in null_values
            else "--"
        )
    # if no guesses yet, set to default
    else:
        function1Selected = "--"
        function1DivisibleSelected = "--"
        function1GreaterSelected = "--"
        function2ASelected = "--"
        function2BSelected = "--"

    vals = {
        "function1Selected": function1Selected,
        "function1DivisibleSelected": function1DivisibleSelected,
        "function1GreaterSelected": function1GreaterSelected,
        "function2ASelected": function2ASelected,
        "function2BSelected": function2BSelected,
    }
    for k, v in vals.items():
        print(f"{k}: {v}")

    return vals


def get_last_calc_vals(last_calc):
    """Helper function to get the last calculator values to display in the sidebar."""
    if last_calc is not None:
        last_calc_vals = {
            "a": last_calc["a"],
            "b": last_calc["b"],
            "x": last_calc["x"],
        }
    else:
        last_calc_vals = {
            "a": "--",
            "b": "--",
            "x": 0,
        }
    return last_calc_vals


def print_message(m):
    print_dict(m)


@app.route("/chat")
def chat():
    """Loads the chat page.
    - Get the last guess and last calculator values to display in the sidebar.
    - Save the current experiment data to the exp_dir/chat/{time} subdirectory.
    """
    user_id = session["user_id"]
    remaining_time = get_remaining_time(user_id)
    curr_minutes, curr_seconds = get_minutes_seconds(remaining_time)
    curr_time = get_formatted_time(remaining_time)

    hint1, hint2 = get_hints()
    last_guess = (
        user_session_data[user_id]["guesses"][-1]
        if len(user_session_data[user_id]["guesses"]) > 0
        else None
    )
    last_guess_vals = get_guess_values(last_guess)

    last_calc = (
        user_session_data[user_id]["calculator_usage"][-1]
        if len(user_session_data[user_id]["calculator_usage"]) > 0
        else None
    )
    last_calc_vals = get_last_calc_vals(last_calc)

    # Save current exp data each time the chat page is loaded
    sub_dir = os.path.join(
        user_session_data[user_id]["exp_dir"], "chat", time.strftime("%H.%M.%S")
    )
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    # Write experiment data to exp_dir/chat/{time} subdirectory
    write_exp(sub_dir)

    return render_template(
        "chat.html",
        prolific_ids=user_session_data[user_id]["prolific_ids"],
        chat_messages=user_session_data[user_id]["chat_messages"],
        time=curr_time,
        curr_minutes=curr_minutes,
        curr_seconds=curr_seconds,
        function1_divisible_options=user_session_data[user_id][
            "function1_divisible_options"
        ],
        function1_greater_options=user_session_data[user_id][
            "function1_greater_options"
        ],
        function2a_options=user_session_data[user_id]["function2a_options"],
        function2b_options=user_session_data[user_id]["function2b_options"],
        function1_options=user_session_data[user_id]["function1_options"],
        hint1=hint1,
        hint2=hint2,
        function1Selected=last_guess_vals["function1Selected"],
        function1DivisibleSelected=last_guess_vals["function1DivisibleSelected"],
        function1GreaterSelected=last_guess_vals["function1GreaterSelected"],
        function2ASelected=last_guess_vals["function2ASelected"],
        function2BSelected=last_guess_vals["function2BSelected"],
        last_calc_a=last_calc_vals["a"],
        last_calc_b=last_calc_vals["b"],
        last_calc_x=last_calc_vals["x"],
    )


def get_student_name(pop_idx, real_student_params):
    """Helper function for getting string name of a population guess, used for mapping adaptive teacher's guesses to string. Assumes real pop idx is 0"""

    user_id = session["user_id"]
    real_pop_idx = user_session_data[user_id]["real_pop_idx"]
    assert (
        real_pop_idx == 0
    ), f"This function requires real_pop_idx to be 0 but got real_pop_idx={real_pop_idx}"

    real_student_name = (
        "f-knower" if bool(real_student_params["fx_knower"]) else "g-knower"
    )

    if pop_idx == 0:
        return real_student_name
    else:
        if real_student_name == "f-knower":
            return "g-knower"
        else:
            return "f-knower"


@app.route("/get_response", methods=["GET", "POST"])
def get_response():
    """Helper function to get the response from the teacher and update the student models."""
    teacher_start_time = time.time()
    user_input = request.args.get("input", default="", type=str)
    if user_input == "undefined":
        user_input = None
    else:
        # TODO: what if can't map to int?
        user_input = int(user_input)

    user_id = session["user_id"]
    teacher = user_teachers[user_id]

    try:
        teacher.update_predictions(teacher.last_x, user_input)

        hit_parsing_error = False
    # TODO: ValueError raised when there is an issue parsing gpt input (i.e. when it says have run out of examples) -- use more specific error?
    except ValueError as e:
        print("ValueError:", e)
        print("Treating error as reaching the total number of possible examples")

        # unclear if teacher.observations was updated, so get gold output by calling teacher.get_gold_output

        last_pred = user_input
        last_inp = teacher.last_x
        last_out = teacher.get_gold_output(last_inp)
        hit_parsing_error = True

    # TODO: if hit parsing error, then don't need to store student response
    if not hit_parsing_error:
        # TODO: what is this used for: Only GPT teacher when student ignores prediction??
        teacher.store_student_response(True)

        # Sanity checks
        assert (
            user_input == teacher.observations[-1]["prediction"]
        ), f"user_input: {user_input}, teacher.observations[-1]['prediction']: {teacher.observations[-1]['prediction']}"
        assert (
            teacher.last_x == teacher.observations[-1]["input"]
        ), f"teacher.last_x: {teacher.last_x}, teacher.observations[-1]['input']: {teacher.observations[-1]['input']}"

        num_examples_so_far = len(teacher.observations)

        # TODO: these should be updated *after* select right?
        last_inp = teacher.observations[-1]["input"]
        last_out = teacher.observations[-1]["output"]
        last_pred = teacher.observations[-1]["prediction"]

    teacher.update_student_models(last_inp, last_pred, last_out)

    strategy = user_session_data[user_id]["config"]["teaching_params"]["strategy"]
    if strategy in PROBABILISTIC_STRATEGIES and teacher.adapt_prior_online:

        # log the student_pop_idx in session
        curr_guess_idx = teacher.student_pop_idx
        real_student_params = user_session_data[user_id]["config"][
            "student_concept_params"
        ]
        curr_guess_name = get_student_name(curr_guess_idx, real_student_params)
        user_session_data[user_id]["student_pop_guesses"].append(
            (curr_guess_idx, curr_guess_name)
        )
        print("best guess of student_pop_idx after:", curr_guess_idx, curr_guess_name)

    if last_out == None:
        last_out = "undefined"
    if last_pred == None:
        last_pred = "undefined"

    if last_pred == last_out:
        # deleted to reduce tokens
        teacher_msg = f"That's <span class='correct'>correct</span>. "
        pred_is_correct = True
    else:
        teacher_msg = f"That's <span class='incorrect'>incorrect</span>. <span class='math'>wug({last_inp})={last_out}</span>. "
        pred_is_correct = False

    # if there are still examples to show *and* no parsing error (if hit parsing error, then don't show more examples; treat as if reached max possible examples)
    if (
        not hit_parsing_error
        and num_examples_so_far
        < user_session_data[user_id]["config"]["num_possible_input_vals"]
    ):
        if isinstance(teacher, GPTProgramTeacher):
            x = teacher.select()
        else:
            x, _ = teacher.select()
        teacher_msg += f"What is <span class='math'>wug({x})</span>?"
        teacher.last_x = x
        new_inp = x
    else:
        print("Reached the total number of possible examples")
        teacher_msg += 'You made it through all my examples! You can spend the rest of the time studying the examples I have already given or press "Finish" below to move on.'
        new_inp = None
        # TODO: prevent user from submitting another guess
        # session["reached_max_possible_inputs"] = True
        user_session_data[user_id]["reached_max_possible_inputs"] = True
    model_response = teacher_msg
    # TODO: hacky, get last_x by passing to function instead
    if model_response is None:
        model_response = "undefined"

    if pred_is_correct:
        user_session_data[user_id]["current_output_streak"] += 1
        user_session_data[user_id]["num_correct_output_preds"] += 1
    else:
        user_session_data[user_id]["current_output_streak"] = 0
    user_session_data[user_id]["num_total_output_preds"] += 1

    teacher_end_time = time.time()
    teacher_time = teacher_end_time - teacher_start_time

    session.modified = True

    accuracy = (
        user_session_data[user_id]["num_correct_output_preds"]
        / user_session_data[user_id]["num_total_output_preds"]
    )

    m = {
        "sub_html": model_response,
        "time": teacher_time,
        "last_input": last_inp,
        "last_output": last_out,
        "new_inp": new_inp,
        "streak": user_session_data[user_id]["current_output_streak"],
        "accuracy": accuracy,
        "num_correct": user_session_data[user_id]["num_correct_output_preds"],
        "num_total": user_session_data[user_id]["num_total_output_preds"],
    }
    update_chat_teacher(m)

    j = jsonify(
        {
            "sub_html": model_response,
            "time": teacher_time,
            "accuracy": m["accuracy"],
            "streak": m["streak"],
            "num_correct": m["num_correct"],
            "num_total": m["num_total"],
        }
    )
    return j


@app.route("/get_current_guess")
def get_current_guess():
    if len(user_session_data[session["user_id"]]["guesses"]) == 0:
        return "no_guess"
    else:
        last_guess = user_session_data[session["user_id"]]["guesses"][-1]
        print("last guess:", last_guess)
        return last_guess


# TODO: this returns a message of a different format than the other messages? check in javascript
@app.route("/get_start_message", methods=["GET"])
def get_start_message():
    teacher_start_time = time.time()

    # TODO: generate response
    user_id = session["user_id"]
    teacher = user_teachers[user_id]
    if isinstance(teacher, GPTProgramTeacher):
        x = teacher.select()
    else:
        x, y = teacher.select()

    # TODO: hacky, get last_x by passing to function instead
    teacher.last_x = x
    model_response = f"What is <span class='math'>wug({x})</span>?"

    teacher_end_time = time.time()
    teacher_time = teacher_end_time - teacher_start_time

    m = {
        "sub_html": model_response,
        "new_inp": x,
        "time": teacher_time,
    }
    update_chat_teacher(m)
    return jsonify({"sub_html": model_response})


@app.route("/get_streak", methods=["GET"])
def get_streak():
    return jsonify(
        {"streak": user_session_data[session["user_id"]]["current_output_streak"]}
    )


@app.route("/update_chat_user", methods=["POST"])
def update_chat_user():
    """Helper function to update the chat with the user's message."""
    data = request.get_json()  # Retrieve the data sent by JavaScript
    data["sender"] = "user"

    user_id = session.get("user_id")
    remaining_time = get_remaining_time(session["user_id"])
    data["timestamp"] = remaining_time

    chat_messages = user_session_data[user_id]["chat_messages"]
    chat_messages.append(data)
    user_session_data[user_id]["chat_messages"] = chat_messages
    session.modified = True

    chat_messages = user_session_data[user_id]["chat_messages"]
    if len(chat_messages) >= 3:
        last_msg = chat_messages[-2]
        curr_msg = chat_messages[-1]
        if last_msg["sender"] == "user" and curr_msg["sender"] == "user":
            print("Problem: Last two messages are from user")

            for m_idx, m in enumerate(chat_messages):
                print(m_idx)
                print_message(m)
                print()

    return jsonify({"status": "Success"})  # Respond to the client (JavaScript)


@app.route("/get_last_message", methods=["GET"])
def get_last_message():
    """Helper function to get the last message in the chat."""
    user_id = session.get("user_id")
    chat_messages = user_session_data[user_id]["chat_messages"]

    if len(chat_messages) == 0:
        return jsonify({"last_message_html": None, "last_message_sender": None})

    last_message = chat_messages[-1]
    return jsonify(
        {
            "last_message_html": last_message["html"],
            "last_message_sender": last_message["sender"],
        }
    )


def update_chat_teacher(message):
    user_id = session.get("user_id")
    remaining_time = get_remaining_time(session["user_id"])
    message["sender"] = "teacher"
    message["html"] = (
        '<li class="bot-li"><small class="bot-name">Teacher</small><div class="bot-msg">'
        + message["sub_html"]
        + "</div></li>"
    )

    message["timestamp"] = remaining_time
    chat_messages = user_session_data[user_id]["chat_messages"]
    chat_messages.append(message)
    user_session_data[user_id]["chat_messages"] = chat_messages


@app.route("/submit_questions", methods=["POST"])
def submit_questions():
    """Save the end questions submitted by the user."""
    data = request.get_json()
    print("submitted questions:", data)
    end_questions = user_session_data[session["user_id"]]["end_questions"]
    end_questions.append(data)
    user_session_data[session["user_id"]]["end_questions"] = end_questions
    session.modified = True
    return jsonify({"status": "Success"})  # Respond to the client (JavaScript)


@app.route("/get_submitted_task_questions")
def get_submitted_task_questions():
    """Helper function to check if the user has submitted the task questions."""
    is_submitted = len(user_session_data[session["user_id"]]["task_questions"]) > 0
    return jsonify({"is_submitted": is_submitted})


@app.route("/submit_task_questions", methods=["POST"])
def submit_task_questions():
    """Save the task questions submitted by the user."""
    data = request.get_json()
    task_questions = user_session_data[session["user_id"]]["task_questions"]
    task_questions.append(data)
    user_session_data[session["user_id"]]["task_questions"] = task_questions
    session.modified = True
    return jsonify({"status": "Success"})  # Respond to the client (JavaScript)


@app.route("/update_guess", methods=["POST"])
def update_guess():
    data = request.get_json()  # Retrieve the data sent by JavaScript
    remaining_time = get_remaining_time(session["user_id"])
    data["timestamp"] = remaining_time
    parsed_data = data.copy()

    # Check if any of the functions are "unchanged", if so, replace with last guess
    # Need to replace because "updateCalculatorConstraints()" requires the actual guesses
    for k, v in parsed_data.items():
        if v == "unchanged" and k in [
            "function1",
            "function1Divisible",
            "function1Greater",
            "function2A",
            "function2B",
        ]:
            guesses = user_session_data[session["user_id"]]["guesses"]
            if len(guesses) > 0:
                parsed_data[k] = guesses[-1][k]
            else:
                parsed_data[k] = "no_guess"

    guesses = user_session_data[session["user_id"]]["guesses"]
    guesses.append(parsed_data)
    user_session_data[session["user_id"]]["guesses"] = guesses

    raw_guesses = user_session_data[session["user_id"]]["raw_guesses"]
    raw_guesses.append(data)
    user_session_data[session["user_id"]]["raw_guesses"] = raw_guesses
    session.modified = True
    return jsonify({"status": "Success"})  # Respond to the client (JavaScript)


@app.route("/update_calculator_usage", methods=["POST"])
def update_calculator_usage():
    """Helper function to update the calculator usage."""
    data = request.get_json()
    remaining_time = get_remaining_time(session["user_id"])
    data["timestamp"] = remaining_time
    calc_usage = user_session_data[session["user_id"]]["calculator_usage"]
    calc_usage.append(data)
    user_session_data[session["user_id"]]["calculator_usage"] = calc_usage
    session.modified = True
    return jsonify({"status": "Success"})  # Respond to the client (JavaScript)


@app.route("/update_visited_chat", methods=["POST"])
def update_visited_chat():
    """Helper function to update the visited chat variable."""
    user_session_data[session["user_id"]]["visited_chat"] = True
    session.modified = True
    return jsonify({"status": "Success"})  # Respond to the client (JavaScript)


# Function to update when the finish button is shown
@app.route("/update_early_finish", methods=["POST"])
def update_early_finish():
    data = request.get_json()  # Retrieve the data sent by JavaScript
    reason = data["reason"]  # Extract the message from the data
    update_early_finish_local(reason)

    return jsonify({"status": "Success"})  # Respond to the client (JavaScript)


def update_early_finish_local(reason):
    """Helper function to update the finish button shown info."""
    user_id = session["user_id"]
    user_session_data[user_id]["finish_button_shown"] = True
    remaining_time = get_remaining_time(user_id)
    user_session_data[user_id]["finish_button_shown_info"].append(
        {"time": remaining_time, "reason": reason}
    )


@app.route("/get_visited_chat")
def get_visited_chat():
    """Helper function to check if the user has visited the chat."""
    return jsonify(data=user_session_data[session["user_id"]]["visited_chat"])


@app.route("/update_started_learning", methods=["POST"])
def update_started_learning():
    print("starting timer...")
    start_timer()
    # session["started_learning"] = True
    user_session_data[session["user_id"]]["started_learning"] = True
    session.modified = True
    # print(f"Started learning: {session['started_learning']}")
    return jsonify({"status": "Success"})


@app.route("/get_started_learning")
def get_started_learning():
    """Helper function to check if the user has started learning."""
    if "user_id" not in session or session["user_id"] not in user_session_data:
        started_learning = False
    else:
        started_learning = user_session_data[session["user_id"]]["started_learning"]
    return jsonify(data=started_learning)


# Return metadata helpful for validating whether the user can send a message
@app.route("/get_user_message_data", methods=["GET"])
def get_user_message_data():
    reached_max_possible_inputs = user_session_data[session["user_id"]][
        "reached_max_possible_inputs"
    ]
    if reached_max_possible_inputs:
        update_early_finish_local("reached_max_possible_inputs")

    return jsonify(
        data={
            "startedLearning": user_session_data[session["user_id"]][
                "started_learning"
            ],
            "reachedMaxPossibleInputs": reached_max_possible_inputs,
        }
    )


@app.route("/get_last_sender")
def get_last_sender():
    if len(user_session_data[session["user_id"]]["chat_messages"]) == 0:
        last_sender = None
        num_messages = 0
        last_user_input = None
    else:
        last_sender = user_session_data[session["user_id"]]["chat_messages"][-1][
            "sender"
        ]
        num_messages = len(user_session_data[session["user_id"]]["chat_messages"])

        if last_sender == "user":
            last_user_input = user_session_data[session["user_id"]]["chat_messages"][
                -1
            ]["message"]
        else:
            last_user_input = None
    return jsonify(
        data={
            "last_sender": last_sender,
            "num_messages": num_messages,
            "last_user_input": last_user_input,
        }
    )


@app.route("/")
def home():
    return render_template("index.html", timer=timer)


def get_remaining_time(user_id):
    if user_id in user_timers:
        start_time = user_timers[user_id]
        elapsed = time.time() - start_time
    else:
        # if user_id not in user_timers but user's timer started, then the timer must have run out
        if user_id in user_timers_started and user_timers_started[user_id]:
            elapsed = MAX_TIME
        else:
            elapsed = 0
    return MAX_TIME - elapsed


@app.route("/get_timer")
def get_timer():
    user_id = session["user_id"]
    elapsed = get_remaining_time(user_id)
    formatted_time = get_formatted_time(elapsed)
    return formatted_time


# TODO: join this with get_timer; this returns the integer time in seconds (above string)
@app.route("/get_raw_timer")
def get_raw_timer():
    user_id = session["user_id"]
    return str(get_remaining_time(user_id))


@app.route("/get_all_time")
def get_all_time():
    user_id = session["user_id"]
    remaining_time = get_remaining_time(user_id)
    minutes, seconds = get_minutes_seconds(remaining_time)
    formatted = get_formatted_time(remaining_time)
    return jsonify(
        {
            "minutes": minutes,
            "max_minutes": MAX_TIME // 60,
            "seconds": seconds,
            "raw": remaining_time,
            "formatted": formatted,
        }
    )


def get_formatted_time(time):
    minutes, seconds = get_minutes_seconds(time)
    # Format the result
    formatted_time = f"{minutes:02}:{seconds:02}"
    return formatted_time


def get_minutes_seconds(time):
    # Calculate minutes and seconds
    minutes = int(time) // 60
    seconds = int(time) % 60
    return minutes, seconds


def start_timer():
    user_id = session.get("user_id")
    start_time = time.time()
    user_timers[user_id] = start_time
    user_session_data[user_id]["learning_start_time"] = start_time
    user_timers_started[user_id] = True


def stop_timer():
    user_id = session.get("user_id")

    if user_id in user_timers:
        del user_timers[user_id]
        print("Stopped timer for user: ", user_id)


####### GLOBAL VARIABLES #######
user_idx = 0
user_timers = {}
user_timers_started = {}
user_teachers = {}
user_times = {}  # amount of time spent total for each user
user_exps = {}  # mapping from user_id to idx of exp in all_experiments_with_seeds
user_random_states = {}
# This is a mapping from user_id to a dict including data that used to be in session
user_session_data = {}

global experiments_and_start_times

torch.set_default_dtype(torch.float64)

# Base config
config = EXPERIMENTS["human"]

"""
Get cleaned config, ie for all keys, take first element of list (since main script uses itertools.product to run many experiments); here, just take first element of list. For strategies, don't take first element of list because want to enumerate over these
"""
cleaned_config = {}
# TODO: hacky, but assumes config's values are lists for each key (because main script uses itertools.product to run many experiments); here, just take first element of list, but check that each list is only 1 element long
for k, v in config.items():
    # Ignore taking first element of strategies because want to enumerate over these
    if k == "strategies":
        cleaned_config[k] = v
        continue

    if v is not None:
        assert len(v) == 1, f"Expected list of length 1, but got {v}"
        cleaned_config[k] = v[0]

print("CLEANED CONFIG")
print_dict(cleaned_config)

# for storing tuple of all progs to outputs_by_inp
tuple_progs_to_outputs_by_inp = {}

args = parse_args()

# Track which experiments have been completed
complete_experiments = set()

complete_exps_file = os.path.join(args.exp_dir, "complete_experiments.pkl")

if os.path.exists(complete_exps_file):
    with open(complete_exps_file, "rb") as f:
        complete_experiments = dill.load(f)
    print("Loaded complete exps from: ", complete_exps_file)

incomplete_experiments = set()  # list of indices to all_experiments_with_seeds


MAX_TIME = 600

print("debug mode:", args.debug)

# Write all_experiments_with_seeds to file
all_experiments_with_seeds_file = os.path.join(
    args.exp_dir, "all_experiments_with_seeds.pkl"
)

if args.use_existing_concepts:
    assert os.path.exists(
        args.exp_dir
    ), f"using existing concepts, so exp_dir must exist: {args.exp_dir}"

    print("Loading all experiments with seeds file:")
    all_experiments_with_seeds = pickle.load(
        open(all_experiments_with_seeds_file, "rb")
    )
    print("all experiments with seeds:", len(all_experiments_with_seeds))

else:

    # prog_concept_param_files = glob.glob(f"{args.prog_concept_params_dir}/*.json")
    # print("prog concept param files:", prog_concept_param_files)

    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    # Write args to file
    args_file = os.path.join(args.exp_dir, "args.pkl")
    with open(args_file, "wb") as file:
        print(f"writing args to file: {args_file}")
        dill.dump(args, file)

    # print("prog concept param files:", prog_concept_param_files)

    all_experiments_with_seeds = enumerate_exps(args)

    print("complete experiments:", len(complete_experiments), complete_experiments)

    with open(all_experiments_with_seeds_file, "wb") as file:
        print(
            f"writing all_experiments_with_seeds to file: {all_experiments_with_seeds_file}"
        )
        dill.dump(all_experiments_with_seeds, file)

# if args.outputs_by_inp_file exists, initialize tuple_progs_to_outputs_by_inp by loading the file
outputs_by_inp_file = os.path.join(args.exp_dir, args.outputs_by_inp_file)
if os.path.exists(outputs_by_inp_file):
    print("Reading from outputs_by_inp_file...", outputs_by_inp_file)
    with open(outputs_by_inp_file, "rb") as f:
        # load with pickle
        tuple_progs_to_outputs_by_inp = dill.load(f)

remaining_experiments = [
    idx
    for idx, x in enumerate(all_experiments_with_seeds)
    if idx not in complete_experiments
]

print(
    f"Remaining experiments after removing complete experiments {len(remaining_experiments)}:"
)
print(remaining_experiments)

experiments_and_start_times = {
    exp_id: [] for exp_id in range(len(all_experiments_with_seeds))
}  # mapping from exp_id to start times (in order of when experiment was run)

for exp, times in experiments_and_start_times.items():
    if times != []:
        print(f"Start times for exp {exp}: {times}")

if __name__ == "__main__":
    with app.app_context():
        app.run(
            debug=args.debug,
            host="0.0.0.0",
            port=8081,
            threaded=True,
        )
