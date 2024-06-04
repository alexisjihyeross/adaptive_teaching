from src.utils import *
from src.programs.utils import *

import os
import numpy as np


def get_concept_config(concept):
    config = {}
    concept_id = concept["id"]
    config.update(
        {
            "target_prog": concept["gold_prog"],
            "prog_file": f"src/bps/progs/{concept_id}/progs.txt",
            "prog_fx_file": f"src/bps/progs/{concept_id}/fxs.txt",
            "prog_gx_file": f"src/bps/progs/{concept_id}/gxs.txt",
            "min_input_val": concept["min_input_val"],
            "max_input_val": concept["max_input_val"],
            "divisible_options": concept["divisible_options"],
            "greater_options": concept["greater_options"],
            "poly_kwargs": concept["poly_kwargs"],
        }
    )

    return config


def get_incorrect_fx(
    fx,
    divisible_options,
    greater_options,
):

    if fx == "even":
        # Randomly sample a function divisible_by_n where n is not 2 but is even
        # options =  [x for x in divisible_options if x % 2 == 0 and x != 2]
        # spurious = [f'divisible_{x}' for x in options]

        # Set spurious to be divisible by 4 or 6
        options = [4, 6]
        # Make sure divisible by 4 or 6 is in divisible options
        assert all([x in divisible_options for x in options])
        spurious = [f"divisible_{x}" for x in options]

    # For odd, spurious is either prime or divisible by n where n is odd
    elif fx == "odd":
        spurious = []

        # Randomly sample a function divisible_by_n where n is odd
        # options = [x for x in divisible_options if x % 2 != 0]
        # spurious.extend([f'divisible_{x}' for x in options])

        options = [3, 5, 7]
        # Make sure divisible by 3, 5, 7 in divisible options
        assert all([x in divisible_options for x in options])
        spurious.extend([f"divisible_{x}" for x in options])

        # Prime is also spurious
        spurious.append("prime")

    elif fx == "prime":
        spurious = ["odd"]

    # For positive, spurious is greater_n where n is != 0
    elif fx == "positive":
        assert 0 not in greater_options, "Should not be greater_0"
        #        spurious = [f'greater_{x}' for x in greater_options]
        # Randomly sample a function greater_n where n is != 0 and |n| <= 2
        spurious = [f"greater_{x}" for x in greater_options if abs(x) <= 2]

    elif fx.startswith("divisible"):
        # First parse the integer n, and then get all divisible_by options that are not n
        _, n = parse_function_with_value(fx)
        assert n != 2, "Should not be divisible by 2"

        # Temporarily add 2 to divisible options so that we can sample from it, then map to even

        temp_divisible_options = divisible_options.copy()
        temp_divisible_options.append(2)

        # Consider all divisible_x options where x is either a factor of n or a multiple of n
        # If multiple factors, consider the largest; if multiple multiples, consider the smallest
        # Factors
        temp_options = [x for x in temp_divisible_options if x != n and n % x == 0]
        if len(temp_options) > 0:
            temp_options = [max(temp_options)]
        options = temp_options.copy()
        # Multiples
        temp_options = [x for x in temp_divisible_options if x != n and x % n == 0]
        if len(temp_options) > 0:
            temp_options = [min(temp_options)]
        options.extend(temp_options)
        spurious = [f"divisible_{x}" for x in options]

        # # if n is odd, consider other odd numbers
        # if n % 2 != 0:
        #     options = [x for x in divisible_options if x % 2 != 0 and x != n]
        # # if n is even, consider other even numbers
        # else:
        #     options = [x for x in divisible_options if x % 2 == 0 and x != n]
        # spurious = [f'divisible_{x}' for x in options]
        # spurious.extend(['even'])

        # Replace divisible_2 with even if it is in spurious
        if "divisible_2" in spurious:
            spurious.remove("divisible_2")
            spurious.append("even")

    elif fx.startswith("greater"):
        _, n = parse_function_with_value(fx)
        assert n != 0, "Should not be greater_0"

        # Temporarily add 0 to greater options so that we can sample from it, then map to positive
        temp_greater_options = greater_options.copy()
        temp_greater_options.append(0)

        # Consider all greater_x options where abs(x-n) <= 2
        options = [x for x in temp_greater_options if x != n and abs(x - n) <= 2]
        spurious = [f"greater_{x}" for x in options]

        # Replace greater_0 with positive if it is in spurious
        if "greater_0" in spurious:
            spurious.remove("greater_0")
            spurious.append("positive")

        # options = [x for x in greater_options if x != n]
        # spurious = [f'greater_{x}' for x in options]
        # spurious.extend(['positive'])
    else:
        raise ValueError(fx)

    # Used if put_high_prob_on_incorrect to figure out which concept to put high prob on
    # For now, hardcode spuriously correlated concepts
    # TODO: make this more general/set this in concept config?
    # TODO: Figure out how to implement divisible/greater than (when what those are spuriously correlated with depends on the argument)
    # fx_incorrect_concept_map = {
    #     'even': ['divisible'], # TODO: specifically set this to be divisible by an *even* number?
    #     'odd': ['prime', 'divisible'],
    #     'prime': ['odd'],
    #     'positive': ['greater_eq'],
    #     # 'divisible': [],
    #     # 'greater_eq': [],
    #     }

    if len(spurious) == 0:
        raise NotImplementedError(
            f"Should have at least one spurious function for divisible, but got {spurious} for fx={fx}"
        )

    return spurious


def get_incorrect_gx(gold_b, b_options):

    # Sample a function add_n where n != gold_b

    options = [x for x in b_options if x != gold_b]
    spurious = [f"add_{x}" for x in options]
    return spurious


def get_student_concept_params(
    concept,
    use_gx_pref=True,
    put_high_prob_on_incorrect=True,
):
    """
    Takes program concept and returns two kinds of students.
    - use_gx_pref: Whether to also use gx_pref (i.e. decrease preferencein true gx function) to differentiate learners
    - put_high_prob_on_incorrect: For the f-learner, whether to put high prob on a particular incorrect f/g (happens if True), or to put low prob on the correct f/g
    """

    fx_concept = concept["fx_special_concept"]
    gx_concept = concept["gx_special_concept"]

    student_concepts = []

    known_prob = 1e4
    unknown_prob = 1e-04
    default_prob = 1

    # Using fx_concept, get incorrect fx concepts that were already created
    incorrect_fx_list = [concept["fx_spurious"]]
    incorrect_gx_list = [concept["gx_spurious"]]

    # TODO: make sure logic is sound
    for fx_knower in [True, False]:
        # if fx_knower, student knows fx concept, doesn't know gx concept
        if fx_knower:
            concept_probs = {fx_concept: known_prob, "DEFAULT": default_prob}

            # for fx knower, put_high_prob_on_correct controls what to do with gx concepts
            if use_gx_pref:
                # if put_high_prob_on_incorrect, put high prob on incorrect gx concepts
                # else, set low prob on correct gx concept
                if put_high_prob_on_incorrect:
                    # get_incorrect_gx(concept['gx_b'], concept['poly_kwargs']['constants'])
                    concept_probs.update({c: known_prob for c in incorrect_gx_list})
                else:
                    concept_probs.update({gx_concept: unknown_prob})
            # make g default to default prob, but that happens already I believe
            else:
                concept_probs.update({c: default_prob for c in incorrect_fx_list})
                # get_incorrect_gx(concept['gx_b'], concept['poly_kwargs']['constants'])})

        # if not fx_knower, student doesn't know fx concept, knows gx concept (if knows fx concept, doesn't know gx concept)
        else:
            # if use_gx_pref, student knows gx concept, otherwise defauls to default prob
            if use_gx_pref:
                concept_probs = {
                    gx_concept: known_prob,
                    "DEFAULT": default_prob,
                }
            # else, default prob
            else:
                concept_probs = {
                    gx_concept: default_prob,
                    "DEFAULT": default_prob,
                }

            # if not fx_knower, put_high_prob_on_correct controls what to do with fx concepts
            # if put_high_prob_on_incorrect, put high prob on incorrect fx concepts
            # else, set low prob on correct fx concept
            if put_high_prob_on_incorrect:
                # incorrect_fx_list = get_incorrect_fx(fx_concept, concept['divisible_options'], concept['greater_options'])
                concept_probs.update({c: known_prob for c in incorrect_fx_list})
            else:
                concept_probs.update({fx_concept: unknown_prob})

        fx_pref = (
            concept_probs[fx_concept] if fx_concept in concept_probs else default_prob
        )
        concept_id = f"{fx_concept}_{fx_pref}"
        if use_gx_pref:
            # If gx_concept not in concept_probs, defaults to default_prob
            gx_pref = (
                concept_probs[gx_concept]
                if gx_concept in concept_probs
                else default_prob
            )
            concept_id += f"--{gx_concept}_{gx_pref}"
        student_concept = {
            "id": concept_id,
            "fx_knower": fx_knower,
            "concept_probs": concept_probs,
        }
        student_concepts.append(student_concept)

    return student_concepts


def get_concept_data(concept, write_data=True, out_dir=f"results/functions"):
    """
    Creates all possible programs for a given concept configuration.

    :param concept: dict specifying concept configuration
    :param write_data: whether to write data to disk
    :param out_dir: directory to write data to (if write_data is True)
    """
    base_prog = concept["base_prog"]
    if base_prog not in [
        "(if f_x None else g_x)",
    ]:
        raise ValueError(base_prog)

    concept_id = concept["id"]
    min_val, max_val = concept["min_input_val"], concept["max_input_val"]

    fs = get_fs(concept["divisible_options"], concept["greater_options"])
    poly_kwargs = concept["poly_kwargs"]
    gs = get_polynomials(**poly_kwargs)

    progs = get_progs(base_prog, fs, gs)
    print("# progs:", len(progs))

    str_progs = [p["prog"] for p in progs]
    f_xs = [p["f_x"] for p in progs]
    g_xs = [p["g_x"] for p in progs]

    if write_data:
        exp_dir = os.path.join(out_dir, f"progs/{concept_id}")
        print(f"Writing program data to {exp_dir}")
        os.makedirs(exp_dir, exist_ok=True)

        write_lines(str_progs, os.path.join(exp_dir, "progs.txt"))
        write_lines(f_xs, os.path.join(exp_dir, "fxs.txt"))
        write_lines(g_xs, os.path.join(exp_dir, "gxs.txt"))

        prompt_dir = os.path.join(out_dir, f"prompts/{concept_id}")
        print(f"Writing prompts to {prompt_dir}")
        os.makedirs(prompt_dir, exist_ok=True)

    return str_progs, f_xs, g_xs


def get_fs(divisible_options, greater_options):
    fs = ["even", "odd", "prime", "positive"]
    fs.extend([f"divisible_{x}" for x in divisible_options])
    fs.extend([f"greater_{x}" for x in greater_options])

    fs = [f"({f} x_)" for f in fs]

    # Return unique functions in case some of the spurious functions were the same
    fs = list(sorted(set(fs)))
    return fs


def get_polynomials(
    # constant_functions=['add', 'subtract'],
    exponents=[1, 2],
    constants=[1, 2, 3, 4, 5],  # to add/subtract
    coefficients=[1, 2, 3],  # to multiply (not divide to work in int space for now)
    abs_vals=[True, False],
):
    """
    # Creates functions of form: (ABS_OR_NOT (ADD_OR_SUBTRACT (multiply (exp x_ EXP) COEF) CONST))
    # Creates functions of form: (ABS_OR_NOT (add_n (multiply (exp x_ EXP) COEF) CONST))
    """

    base = "(ADD_OR_SUBTRACT (multiply (exp x_ EXP) COEF))"

    fs = []

    for exp in exponents:
        for const in constants:
            for coef in coefficients:
                # for func in constant_functions:
                temp = (
                    base.replace("COEF", str(coef))
                    .replace("EXP", str(exp))
                    .replace("ADD_OR_SUBTRACT", f"add_{const}")
                )
                for do_abs in abs_vals:
                    if do_abs:
                        temp_temp = f"(abs {temp})"
                    else:
                        temp_temp = temp
                    fs.append(temp_temp)

    return list(sorted(fs))


def get_mod_exps(constant_functions=["add", "subtract"]):
    """
    Creates functions of form: (ABS_OR_NOT (ADD_OR_SUBTRACT (multiply (exp x_ EXP) COEF) CONST))
    """

    mod_ns = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # to add/subtract
    constants = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    base = "(ADD_OR_SUBTRACT (mod x_ MOD) CONST)"

    fs = []

    for const in constants:
        for mod_n in mod_ns:
            for func in constant_functions:
                temp = (
                    base.replace("MOD", str(mod_n))
                    .replace("CONST", str(const))
                    .replace("ADD_OR_SUBTRACT", func)
                )
                fs.append(temp)

    return fs


def get_progs(base_prog, fs, gs):

    progs = []
    for f in fs:
        for g in gs:
            p = base_prog.replace("f_x", f).replace("g_x", g)
            prog = {"prog": p, "f_x": f, "g_x": g}
            progs.append(prog)
    #             print(progs[-1])
    return progs


def get_binary_map_gs(min_val, max_val, spurious=["inverse_binary_map"]):
    """Returns max_val maps where the ith integer at the ith map returns 1 and otherwise 0
    (at least that is how it is implemented in the environment -- TODO: double check)

    min_val not used
    """

    gs = []
    for base in ["binary_map"] + spurious:
        for map_idx in range(min_val, max_val):
            gs.append(f"({base} {map_idx} x_)")

    return gs


def get_square_wave_gs(min_val, max_val, spurious=["inverse_square_wave"]):
    """
    Returns switch functions that change every i integers,
    where i is between 0 and 10
    """

    gs = []
    for base in ["square_wave"] + spurious:
        for i in range(1, max_val):
            gs.append(f"({base} {i} x_)")

    return gs


def get_odd_fs(min_val, max_val, spurious=["divisible"]):
    """
    FUNCTIONS NEEDED: even, equals
    """

    fs = [
        "(odd x_)",  # gold
        #         '(equals 0 (mod x_ 2))', # gold
    ]

    for sp in spurious:
        if sp == "divisible":
            temp = "(divisible x_ INT)"
        else:
            raise ValueError(sp)

        # TODO: this range is specifically for visible
        for n in np.arange(3, max_val, 1):
            if n % 2 == 0:
                continue
            fs.append(temp.replace("INT", str(n)))

    return fs


def get_even_fs(min_val, max_val, spurious=["even_and_neq"]):
    """
    FUNCTIONS NEEDED: even, equals
    """

    fs = [
        "(even x_)",  # gold
        #         '(equals 0 (mod x_ 2))', # gold
    ]

    for sp in spurious:
        if sp == "eq":
            temp = "(equals x_ INT)"
        elif sp == "even_and_neq":
            temp = "(even_and_neq x_ INT)"
        elif sp == "mod":
            temp = "(equals (mod x_ INT) 0)"
        elif sp == "divisible":
            temp = "(divisible x_ INT)"
        else:
            raise ValueError(sp)

        # TODO: confirm desired behavior, for divisible, only looking at 2...max_val
        if sp == "divisible":
            # 3 not 2 bc 2 is the same as even
            n_range = np.arange(3, max_val, 1)
        else:
            n_range = np.arange(min_val, max_val, 1)

        for n in n_range:
            if n % 2 != 0:
                continue
            fs.append(temp.replace("INT", str(n)))

    return fs


def get_prime_fs(min_val, max_val, spurious=["prime_and_neq"]):
    """
    FUNCTIONS NEEDED: prime, equals
    (could also do odd/not_even)

    Spurious hypotheses consist of:
    - equals individual primes
    """

    fs = [
        "(prime x_)",
    ]

    assert max_val <= 100

    for sp in spurious:
        if sp == "eq":
            # equals individual primes
            temp = "(equals x_ INT)"
        elif sp == "prime_and_neq":
            temp = "(prime_and_neq x_ INT)"
        # TODO: implement odd as not(even)?
        elif sp == "odd":
            temp = "(odd x_)"
            # No int to replace for odd, so don't iterate through all ptrimes
            # TODO: make fs a set? so that automatically deals with duplicates
            fs.append(temp)
            continue
        else:
            raise ValueError(sp)

        primes = [
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            29,
            31,
            37,
            41,
            43,
            47,
            53,
            59,
            61,
            67,
            71,
            73,
            79,
            83,
            89,
            97,
        ]
        for prime in primes:
            if prime < max_val:
                fs.append(temp.replace("INT", str(prime)))

    """
    - mod [1, 3, 5, 7, 9] == 0: 
        will only have higher prior than prime (gold) if prime has *really* low prior like 0.01 
        (i.e. even with two functions, 0.25^2 > 0.01)
    temp = '(equals (mod x_ INT) 0)'
    for num in [1, 3, 5, 7, 9]:
        fs.append(temp.replace('INT', str(num)))
    """

    return fs


def get_positive_fs(min_val, max_val, spurious=["positive_and_neq"]):
    """
    FUNCTIONS NEEDED: positive, greater_eq (i.e. greater than or equal to)

    spurious:
    - >= ints from min_val to max_val, skipping 1 (bc then it is positive)
    """

    fs = ["(positive x_)"]

    for sp in spurious:
        if sp == "greater_eq":
            temp = "(greater_eq x_ INT)"
        elif sp == "positive_and_neq":
            temp = "(positive_and_neq x_ INT)"

        if sp == "greater_eq":
            n_range = range(1, max_val)
        else:
            n_range = range(min_val, max_val)
        for n in n_range:
            if n == 1:
                continue
            temp_temp = temp.replace("INT", str(n))
            fs.append(temp_temp)

    return fs


def print_prompts(base_prompt, end_prompt):
    print("=====================================\n\nBASE PROMPT:\n\n")
    print(base_prompt)
    print("=====================================\n\nEND PROMPT:\n\n")
    print(end_prompt)
