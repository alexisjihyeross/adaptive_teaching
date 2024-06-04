import json
import torch
from tqdm import tqdm
import random
import dill
import os
import wandb

from src.programs.interpreter import *

from src.exp_configs import *
from src.programs.bps import BayesianProgramSynthesizer, students_are_equal
from src.programs.concepts import get_concept_library_from_params
from src.utils import write_file
from src.programs.eval import log_results

# from src.eval import get_auc
from src.utils import end_teacher_log


def get_populations(config, student, do_print=False, return_population_params=False):
    """
    Helper function to get populations of students. Returns a list of BayesianProgramSynthesizers.
    Always has the student as the first population.
    """
    student_params = config["student_concept_params"]
    # TODO: remove student_concept_params from student_populations?
    population_params = config["student_populations"]
    # TODO: confirm hypotheses/progs_reps (instead of all_hypotheses/all_progs_reps) is okay -- should be since using student when just initialized
    interp = student.interpreter
    populations = [
        BayesianProgramSynthesizer(
            student.hypotheses.copy(),
            interp,
            get_concept_library_from_params(interp, p),
            student.dataset,
            progs_reps=student.progs_reps.clone(),
            noise=student.noise,
            outputs_by_inp=student.outputs_by_inp.copy(),
            do_print=do_print,
        )
        for p in population_params
    ]

    # Get different populations
    diff_populations = [
        s for s in populations if any(s.concepts.probs != student.concepts.probs)
    ]
    assert (
        len(diff_populations) == len(populations) - 1
    ), f"{len(diff_populations)} != {len(populations) - 1}"

    student_copy = BayesianProgramSynthesizer(
        student.hypotheses.copy(),
        interp,
        get_concept_library_from_params(interp, student_params),
        student.dataset,
        progs_reps=student.progs_reps.clone(),
        outputs_by_inp=student.outputs_by_inp.copy(),
        noise=student.noise,
    )
    populations = [student_copy] + diff_populations

    if return_population_params:
        # get ordered list of population params

        # diff populations should be the same as population_params, but with student_params removed
        diff_population_params = [p for p in population_params if p != student_params]
        population_params = [student_params] + diff_population_params
        assert len(population_params) == len(
            populations
        ), f"{len(population_params)} != {len(populations)}"
        return populations, population_params

    return populations


def get_initial_student_guess(populations, student, teaching_params, random_state=None):
    """Get the initial student guess for the teacher.
    If assume_known_prior, then the initial student guess is the first population (the true student).
    Otherwise, the initial student guess is a random student from the populations.
    """
    if (
        "assume_known_prior" not in teaching_params
        or not teaching_params["assume_known_prior"]
    ):
        # If "assume_known_prior" is not in teaching_params, make sure it's random strategy (which doesn't use a student guess)

        if "assume_known_prior" not in teaching_params:
            assert (
                teaching_params["strategy"] == "random"
            ), f"'assume_known_prior' not in teaching_params but got nonrandom strategy = {teaching_params['strategy']}"

        # Sample a student to be the teacher's belief about the student
        # If random_state is not None, then use that to sample
        if random_state is not None:
            initial_student_guess = random_state.sample(populations, 1)[0]
        else:
            initial_student_guess = random.sample(populations, 1)[0]
    elif teaching_params["assume_known_prior"]:
        initial_student_guess = populations[0]
        assert students_are_equal(initial_student_guess, student)
    else:
        raise NotImplementedError(
            f"teaching_params['assume_known_prior'] = {teaching_params['assume_known_prior']}"
        )

    return initial_student_guess


def save_initial_files(run_config, student, extra_files_to_save=[]):
    """Helper function to save initial files for the run."""
    with open("config.json", "w") as fout:
        json.dump(run_config, fout)
    wandb.save("config.json")

    ######### WRITE AND SAVE FILES ##########
    for f in extra_files_to_save:
        wandb.save(f)
    # wandb.save(hyp_params["prog_file"])

    # Sample prior, then write to temp files and wandb log
    write_lines([p for p in student.concepts.probs], "prior_over_concepts.txt")
    wandb.save("prior_over_concepts.txt")

    write_lines([p for p in student.concepts], "concepts.txt")
    wandb.save("concepts.txt")


def initialize_teaching_args(
    teaching_params,
    config,
    ordered_student_params,
    gpt_helper,
    initialize_gpt_args_func,
):
    """Helper function to initialize teaching args for teacher initialization."""

    strategy = teaching_params["strategy"]
    seed = config["seed"]

    teaching_args = teaching_params.copy()

    # TODO: also do this for adapt_prior_online and update_student_beliefs?
    # TODO: implement a more general way to do this
    # for each dict in ENV_PARAMS, run an experiment for each concept params
    if "assume_known_prior" not in teaching_params:
        assert teaching_params["strategy"] in [
            "random",
        ], f"'assume_known_prior' not in teaching_params but strategy != 'random'"

    gpt_args = None
    if strategy in GPT_STRATEGIES:

        gpt_args = initialize_gpt_args_func(config, gpt_helper, ordered_student_params)

        base_prompt = gpt_args["base_prompt"]
        end_prompt = gpt_args["end_prompt"]

        # TODO: hacky, but these are ignored for gpt+probabilistic
        teaching_args.update({"base_prompt": base_prompt, "end_prompt": end_prompt})
        teaching_args.update({"seed": seed})

        if strategy in GPT_COMBINED_STRATEGIES:
            # get a list of base prompts corresponding to each student population
            prompts_for_populations = gpt_args["prompts_for_populations"]
            descriptions_for_populations = gpt_args["descriptions_for_populations"]

            if teaching_params["assume_known_prior"]:
                raise NotImplementedError(
                    "Assuming known prior not implemented for gpt+probabilistic"
                )

            teaching_args.update(
                {
                    "prompts_for_populations": prompts_for_populations,
                    "descriptions_for_populations": descriptions_for_populations,
                }
            )

    # TODO: use initialize_teacher_from_params in app.py
    teaching_args.pop("strategy")
    teaching_args.pop("id")
    if strategy in ["probabilistic", "gpt+probabilistic"]:
        assert "adapt_prior_online" in teaching_args
        assert "update_student_beliefs" in teaching_args
        loss_type = teaching_params["loss_type"]
        teaching_args["loss_type"] = loss_type

    if strategy in ["probabilistic", "gpt", "ranking", "gpt+probabilistic"]:
        teaching_args.pop("assume_known_prior")

    return teaching_args, gpt_args


def check_teacher_student_hyps(teacher, student, strategy):
    """
    Check that the teacher and student have the same hypotheses.
    """
    if "probabilistic" in strategy:
        assert len(teacher.student_guess.hypotheses) == len(
            student.hypotheses
        ), f"teacher hypotheses: {len(teacher.student_guess.hypotheses)}, student hypotheses: {len(student.hypotheses)}"

        assert all(
            [
                s == t
                for s, t in zip(student.hypotheses, teacher.student_guess.hypotheses)
            ]
        ), f"Student and Teacher hypotheses are unequal:\nStudent: {student.hypotheses}\nTeacher: {teacher.student_guess.hypotheses}"


def initialize_teacher_from_params(
    student,
    teaching_params,
    dataset,
    target_prog,
    config,
    gpt_helper,
    initialize_teacher_func,
    initialize_gpt_args_func,
    random_state=None,
):
    strategy = teaching_params["strategy"]

    # Always has student as first population
    populations, ordered_student_params = get_populations(
        config, student, return_population_params=True
    )

    initial_student_guess = get_initial_student_guess(
        populations, student, teaching_params, random_state=random_state
    )

    teaching_args, gpt_args = initialize_teaching_args(
        teaching_params,
        config,
        ordered_student_params,
        gpt_helper,
        initialize_gpt_args_func,
    )

    teacher = initialize_teacher_func(
        strategy,
        dataset,
        populations,
        initial_student_guess,
        student.interpreter,
        target_prog,
        **teaching_args,
    )

    return teacher, gpt_args


def run_teaching_exp(
    student,
    teaching_params,
    dataset,
    target_prog,
    config,
    gpt_helper,
    initialize_teacher_func,  # function to initialize teacher
    initialize_gpt_args_func,  # function to initialize gpt args
    progs_to_eval_on_inputs=[],
    map_inputs_to_str=False,
    special_datasets={},
    random_state=None,
):
    """Helper function to run experiment with a student and teacher.
    - progs_to_eval_on_inputs: list of tuples of programs to evaluate on inputs (e.g. (fx_half_prog, "x=fx?")) --> evaluates if x satisfies fx_half_prog (i.e. x is a "fx" input)
    """

    strategy = teaching_params["strategy"]

    teacher, gpt_args = initialize_teacher_from_params(
        student,
        teaching_params,
        dataset,
        target_prog,
        config,
        gpt_helper,
        initialize_teacher_func,
        initialize_gpt_args_func,
        random_state=random_state,
    )

    interp = student.interpreter

    # if not isinstance(teacher, GPTProgramTeacher):
    # assert teacher.dataset.get_len() == config["num_possible_inputs"]

    # results to track over iterations
    global_results = {
        "y_correct_lst": [],
        "gold_total_probs": [],
        "dataset_accs_max": [],
        "dataset_accs_sample": [],
        "reason_for_finish": "",
    }

    for i in range(config["num_iterations"]):

        if i >= config["num_possible_inputs"]:
            global_results["reason_for_finish"] = (
                f"num_iterations = total number of possible inputs: {config['num_possible_inputs']}"
            )
            break

        # if isinstance(teacher, GPTProgramTeacher):
        if strategy in GPT_STRATEGIES:
            x = teacher.select()

            if strategy in GPT_COMBINED_STRATEGIES:
                # TODO: automatically move this to select()?
                teacher.clean_messages()

            pred = student.predict(x)
            try:
                y = teacher.update_predictions(x, pred)
            except Exception as e:
                global_results["reason_for_finish"] = (
                    f"Uncaught error in updating predictions, error msg: {e}"
                )
                break
        else:
            x, y = teacher.select()

            pred = student.predict(x)
            teacher.update_predictions(x, pred, out=y)

        gold_y = student.interpreter.run_program(target_prog, x)
        global_results["y_correct_lst"].append(int(y == gold_y))

        # Store that the student learned from the example (TODO: desired behavior always?)
        teacher.store_student_response(True)

        results, global_results = log_results(
            global_results,
            i,
            x,
            y,
            progs_to_eval_on_inputs,
            gold_y,
            pred,
            student,
            teacher,
            dataset,
            target_prog,
            teaching_params,
            map_inputs_to_str=map_inputs_to_str,
            special_datasets=special_datasets,
        )

        try:
            student.update_posterior(x, y)
            teacher.update_student_models(x, pred, y)
        except PosteriorError as e:
            """
            TODO: hacky, but if GPT and y != gold_y,
            there might be an issue in updating posterior
            because of a wrong input (i.e. then there are zero hypotheses left);
            in that case, just terminate early
            """
            # TODO: if do this, what to do with the eval post iterations?
            if strategy in GPT_STRATEGIES:
                global_results["reason_for_finish"] = (
                    "GPT generated an answer that is not the same as "
                    "gold_y, led to no more hypotheses..."
                )
                if sum(global_results["y_correct_lst"]) == len(
                    global_results["y_correct_lst"]
                ):
                    assert False, (
                        "We shouldn't be getting that the student has "
                        "0 hypotheses left if all the generated ys were correct"
                    )
                break
            else:
                print(e)
                assert False

        # More reasons for early finish: no hypotheses left or gold prob is 0.0
        if results["num_hyp_nonzero_nongold"] == 0:
            global_results["reason_for_finish"] = (
                f"num_hyp_nonzero_nongold = 0; "
                f"gold_prob={results['student_learning/gold_total_prob']}; "
                "gold_prob=1.0? {gold_total_prob==1.0}"
            )
            break

        if results["student_learning/gold_total_prob"] == 0.0:
            global_results["reason_for_finish"] = (
                "gold_prob == 0.0; happens when teacher generates incorrect answer"
            )
            break

        check_teacher_student_hyps(teacher, student, strategy)

    end_teacher_log(
        teacher,
        teaching_params,
        global_results,
        config,
        gpt_helper,
        gpt_args,
    )


def get_outputs_by_inp(str_progs, args):
    """
    Helper function to get outputs by input for a set of programs.
    """
    tuple_progs = tuple(str_progs)
    tuple_progs_to_outputs_by_inp = {}

    outputs_by_inp_file = os.path.join(args.outputs_by_inp_file)
    if os.path.exists(outputs_by_inp_file):
        print("Reading from outputs_by_inp_file...", outputs_by_inp_file)
        with open(outputs_by_inp_file, "rb") as f:
            try:
                tuple_progs_to_outputs_by_inp = dill.load(f)
            except:
                try:
                    tuple_progs_to_outputs_by_inp = dill.load(f)
                except:
                    tuple_progs_to_outputs_by_inp = {}

    print(
        "Outputs already computed for these programs? ",
        tuple_progs in tuple_progs_to_outputs_by_inp,
    )

    outputs_by_inp = (
        tuple_progs_to_outputs_by_inp[tuple_progs]
        if tuple_progs in tuple_progs_to_outputs_by_inp
        else None
    )
    return outputs_by_inp, tuple_progs_to_outputs_by_inp


def get_prog_reps(interpreter, str_progs, concepts):
    """Helper function to get program representations."""
    print("Getting prog representations...")
    parsed_progs = [interpreter.parse(p) for p in tqdm(str_progs)]
    progs_reps = torch.stack(
        [
            get_prog_vector(parsed, concepts.get_concept_dict())
            for parsed in tqdm(parsed_progs)
        ]
    )
    return progs_reps


def print_beliefs(student, gold_prog, str_progs):
    print("STUDENT BELIEFS:")
    print(f"Gold prog: {gold_prog} ({student.get_hyp_prob(gold_prog):.2f})")
    for prog in str_progs:
        print(f"({student.get_hyp_prob(prog):.2f}) {prog}")


def parse_function_with_value(func):
    value = int(func.split("_")[-1])
    func_name = func.replace(f"_{value}", "")
    return func_name, value


def freeze(d):
    if isinstance(d, dict):
        return frozenset((key, freeze(value)) for key, value in sorted(d.items()))
    elif isinstance(d, list):
        return tuple(freeze(value) for value in d)
    return d


def write_progs_reps(
    tuple_progs_to_progs_reps, file_name="tuple_progs_to_progs_reps.json"
):
    print("Writing progs reps by inp to file:", file_name)
    with open(file_name, "wb") as f:
        # Write with pickle
        dill.dump(tuple_progs_to_progs_reps, f)


def write_outputs_by_inp(
    tuple_progs_to_outputs_by_inp, file_name="tuple_progs_to_outputs_by_inp.json"
):
    # Helper function to write outputs_by_inp to json file

    with open(file_name, "wb") as f:
        # Write with pickle
        dill.dump(tuple_progs_to_outputs_by_inp, f)


# TODO: I don't think this works for if concept has "ifs"
# Need to add another check for strings
def check_concept_equality(c1, c2):
    # c1 and c2 can be Concepts or Lists (i.e. output of parsed)
    if type(c1) != type(c2):
        return False
    elif isinstance(c1, Primitive):
        return c1.value == c2.value
    elif isinstance(c1, Function):
        return c1.name == c2.name
    elif isinstance(c1, List):
        if len(c1) != len(c2):
            return False
        return all([check_concept_equality(el1, el2) for (el1, el2) in zip(c1, c2)])
    else:
        raise NotImplementedError


# TODO: create a "Parsed" object and make these functions of that
def parsed_to_string_list(parsed):
    str_parsed = []
    for p in parsed:
        if isinstance(p, List):
            str_parsed.append(parsed_to_string_list(p))
        else:
            str_parsed.append(str(p))
    return str_parsed


def parsed_to_string(parsed):
    if isinstance(parsed, Concept):
        return str(parsed)
    s = "("
    for p in parsed:
        if isinstance(p, List):
            s += parsed_to_string(p) + " "
        else:
            s += str(p) + " "
    s = s.strip()
    s += ")"
    return s


def get_prog_vector(parsed_prog, concepts):
    counts_by_concept = torch.zeros(len(concepts))
    if isinstance(parsed_prog, Primitive):
        return counts_by_concept
    counts_by_concept = get_concept_counts(parsed_prog, concepts, counts_by_concept)
    return counts_by_concept


def get_concept_counts(parsed, concepts, current_counts):

    for concept in concepts.values():
        idx = concept["idx"]
        temp_parsed = concept["raw"]
        is_equal = check_concept_equality(parsed, temp_parsed)
        if is_equal:
            current_counts[idx] += 1
            return current_counts

    for tok in parsed:
        found_concept = False
        for concept in concepts.values():
            idx = concept["idx"]
            temp_parsed = concept["raw"]
            is_equal = check_concept_equality(tok, temp_parsed)
            if is_equal:
                current_counts[idx] += 1
                found_concept = True
                break

        # haven't found any compositional concepts, count sub-functions
        if not found_concept and isinstance(tok, List):
            current_counts = get_concept_counts(tok, concepts, current_counts)
    return current_counts


def write_progs(progs, file_name):
    num_total = 0
    num_written = 0
    print(f"Writing programs to file: {file_name}")
    # TODO: This skips programs without x_ in them, set that as an argument?
    with open(file_name, "w") as f:
        for e in tqdm(progs):
            e_string = e.to_string()
            num_total += 1
            if "x_" not in e_string:
                continue
            f.write(f"{e_string}\n")
            num_written += 1
    print("Num total: ", num_total)
    print("Num written (not skipped): ", num_written)


def write_lines(lines, file_name):
    print(f"Writing lines to file: {file_name}")
    with open(file_name, "w") as f:
        for line in lines:
            f.write(f"{line}\n")


def read_lines(file_name):
    print(f"Reading lines from file: {file_name}")
    with open(file_name) as file:
        lines = [line.rstrip() for line in tqdm(file)]
    return lines


# TODO: move to dataset class
def sample_input(input_type="int", min_val=0, max_val=11):
    # treat as int
    if input_type == "int":
        return random.randint(min_val, max_val - 1)
    else:
        raise ValueError


def sample_program(programs):
    return random.sample(programs, 1)[0]


def canonicalize_progs(inps, interpreter, progs):
    print("Canonicalizing programs...")

    canonicalized_to_prog = (
        {}
    )  # Dictionary to store programs with the same outputs (maps from canonicalized indices to progs)
    prog_to_canonicalized = (
        {}
    )  # Dictionary to map each program to idxes of programs with the same outputs

    outputs_to_idx = (
        {}
    )  # Helper dictionary to map a tuple of outputs to a canonicalized index

    canonicalized_counter = 0

    for prog in tqdm(progs):
        outputs = []
        for inp in inps:
            # Compute the output for each input
            output = interpreter.run_program(
                prog, inp, strict=False
            )  # Replace 'compute_output' with your actual computation function
            outputs.append(output)

        # Check if the outputs are already present in the dictionary
        key = tuple(outputs)  # Convert the list of outputs to a tuple for hashing
        if key in outputs_to_idx:
            idx_key = outputs_to_idx[key]
            canonicalized_to_prog[idx_key].append(prog)
        else:
            # Add unique index for given output
            outputs_to_idx[key] = canonicalized_counter
            idx_key = outputs_to_idx[key]
            canonicalized_to_prog[idx_key] = [prog]

            canonicalized_counter += 1
        prog_to_canonicalized[prog] = idx_key

    return canonicalized_to_prog, prog_to_canonicalized


def get_all_canonicalized_progs(prog, canonicalized_to_prog, prog_to_canonicalized):
    """Takes a program prog and returns a list of programs
    that have same canonicalized form"""

    can_idx = prog_to_canonicalized[prog]
    return canonicalized_to_prog[can_idx]
