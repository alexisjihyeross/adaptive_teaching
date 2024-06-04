import shutil
import os
import math
import random
from src.programs.types import *
from src.programs.fractions.lib import *
from src.utils import print_dict, write_dict

int_type = int
list_type = list
bool_type = bool


def random_mapping(seed):
    rng = random.Random(seed)
    mapping = {}
    # TODO: set 1000 to be same as max_val in experiment config
    for i in range(0, 201):
        output = rng.randint(0, 1)
        mapping[i] = output
    return mapping


def int_mapping(special_inp, max_inp=4):
    """
    Creates a mapping where all inputs between 0 and max_inp
    are mapped to 0 *except* special_inp, which is mapped to 1.
    """
    inp_vals = range(0, max_inp)
    mapping = {i: 0 for i in inp_vals}
    mapping[special_inp] = 1
    return mapping


class Env(dict):
    "An environment: a dict of {'var':val} pairs, with an outer Env."

    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer

    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)


def get_env(env_name="standard", **kwargs):
    if env_name == "standard":
        raise NotImplementedError(
            "Need to allow for specifying functions to standard_env too"
        )
    elif env_name == "function":
        return function_env(**kwargs)
    elif env_name == "fraction":
        return fraction_env()
    else:
        raise ValueError(f"No environment found with name {env_name}")


def get_input_type(name):
    """Returns type of inputs to programs (i.e. types of x_ variables)"""
    print(f"Getting input type: {name}")
    if name == "standard":
        return "list"
    elif "function" in name:
        return "int"
    elif "fraction" in name:
        return FractionProblem
    else:
        raise ValueError(f"No input type found with name {name}")


def function_env(num_mappings=100, functions=None):
    """
    An environment with some Scheme standard procedures.
    if functions is not None, then only include those functions in the environment
    """
    env = Env()

    # TODO: make align with "Natural Number" type
    # Moved to global variables
    # int_type = int
    # list_type = list
    # bool_type = bool

    def even(x):
        return x % 2 == 0

    def odd(x):
        return x % 2 != 0

    def is_prime(number):
        if number < 2:
            return False
        for i in range(2, int(number**0.5) + 1):
            if number % i == 0:
                return False
        return True

    def is_positive(n):
        return n > 0

    # TODO: can't have functions return int_types bc what should they return if there is an error parsing program?
    # TODO: Change nth/tail to return int_type?
    # TODO: either filter only one of append/extend/prepend to be able to take the empty list, OR make a separate function that goes from int -> list

    # CURRENTLY: tail, sum, nth, count return ints; these can be made back into lists with listify. Including no [] primitive in enumerating programs
    # If *do* end up including [], then in synthesizer, filter progs where all other non-int-returning functions take an empty list (since should be able to compute the same with listify)
    avail_functions = {
        # Remove append and prepend because extend + listify can compute them
        # TODO: allow for outputting list or int type?
        # is_symmetric defaults to false when takes just one input
        "even": Function(
            "even",
            func=lambda x: even(x),
            input_types=[int_type],
            output_type=bool_type,
            is_symmetric=False,
        ),
        "odd": Function(
            "odd",
            func=lambda x: odd(x),
            input_types=[int_type],
            output_type=bool_type,
            is_symmetric=False,
        ),
        "prime": Function(
            "prime",
            func=lambda x: is_prime(x),
            input_types=[int_type],
            output_type=bool_type,
            is_symmetric=False,
        ),
        "positive": Function(
            "positive",
            func=lambda x: is_positive(x),
            input_types=[int_type],
            output_type=bool_type,
            is_symmetric=False,
        ),
        "multiply": Function(
            "multiply",
            func=lambda x, n: x * n,
            input_types=[int_type, int_type],
            output_type=int_type,
            is_symmetric=True,
        ),
        "exp": Function(
            "exp",
            func=lambda x, n: x**n,
            input_types=[int_type, int_type],
            output_type=int_type,
            is_symmetric=False,
        ),
    }

    env.update(avail_functions)

    # add functions for specific values for these functions, i.e. greater_1, greater_2, ...
    for func in ["divisible", "greater", "add"]:
        env.update(get_functions_with_values(func))

    if functions is not None:
        new_env = Env()
        new_env.update({k: env[k] for k in functions})
        env = new_env

    return env


def get_functions_given_arg(func, n):

    # helper function for divisible that, given an n, returns a local function that checks if x is divisible by n
    # TODO: divisibility: what about when x = 0?
    if func == "divisible":

        def divisible_n(x):
            return x % n == 0

        return divisible_n

    elif func == "greater":

        def greater_n(x):
            return x > n

        return greater_n

    elif func == "add":

        def add_n(x):
            return x + n

        return add_n


def get_functions_with_values(func, min_val=-100, max_val=100):
    """Helper function to return a list of functions that operate on specific values
    For example, if func was 'divisible', then this would return a list of functions
    divisible_n for n from min_val to max_val that check, specifically, for divisibility by n.
    Requires that func is already in env.
    op is the operation that func performs; returns True if op is True, False otherwise.
    """
    functions = {}
    # Do this for values from min_val to max_val for now because doesn't matter if include other functions
    for i in range(min_val, max_val + 1):
        functions[f"{func}_{i}"] = Function(
            f"{func}_{i}",
            func=get_functions_given_arg(func, i),
            input_types=[int_type],
            output_type=bool_type,
            is_symmetric=False,
        )
    return functions


def standard_env():
    "An environment with some Scheme standard procedures."
    env = Env()

    def head(x):
        #        try: return [x[0]]
        #        except: return []
        return x[0]

    def tail(x):
        #        try: return [x[-1]]
        #        except: return []
        return x[-1]

    def nth(x, y):
        #        try:
        #            return [y[x-1]]
        #        except:
        #            return []
        return y[x]

    # TODO: make align with "Natural Number" type
    int_type = int
    list_type = list
    bool_type = bool

    # TODO: can't have functions return int_types bc what should they return if there is an error parsing program?
    # TODO: Change nth/tail to return int_type?
    # TODO: either filter only one of append/extend/prepend to be able to take the empty list, OR make a separate function that goes from int -> list

    # CURRENTLY: tail, sum, nth, count return ints; these can be made back into lists with listify. Including no [] primitive in enumerating programs
    # If *do* end up including [], then in synthesizer, filter progs where all other non-int-returning functions take an empty list (since should be able to compute the same with listify)
    env.update(
        {
            # Remove append and prepend because extend + listify can compute them
            #        'prepend':     Function('prepend', func=lambda x,y: [x]+y, input_types=[int_type, list_type], output_type=list_type),
            #        'append':   Function('append', func=lambda x,y: y+[x], input_types=[int_type, list_type], output_type=list_type),
            "sum": Function(
                "sum",
                func=lambda x: sum(x),
                input_types=[list_type],
                output_type=int_type,
                is_symmetric=True,
            ),
            "add_int": Function(
                "add_int",
                func=lambda x, y: x + y,
                input_types=[int_type, int_type],
                output_type=int_type,
                is_symmetric=True,
            ),
            #        'add':      Function('add', func=lambda x,y: [x+el for el in y], input_types=[int_type, list_type], output_type=list_type, is_symmetric=False),
            # TODO: are you supposed to sort the list, or assume that it's sorted?
            "int_to_list": Function(
                "int_to_list",
                func=lambda x: [x],
                input_types=[int_type],
                output_type=list_type,
                is_symmetric=False,
            ),
            "extend": Function(
                "extend",
                func=lambda x, y: x + y,
                input_types=[list_type, list_type],
                output_type=list_type,
                is_symmetric=False,
            ),
            "reverse": Function(
                "reverse",
                func=lambda x: x[::-1],
                input_types=[list_type],
                output_type=list_type,
                is_symmetric=False,
            ),
            "sort": Function(
                "sort",
                func=lambda x: sorted(x),
                input_types=[list_type],
                output_type=list_type,
                is_symmetric=False,
            ),
            "remove": Function(
                "remove",
                func=lambda x, y: [el for el in y if el != x],
                input_types=[int_type, list_type],
                output_type=list_type,
                is_symmetric=False,
            ),
            "count": Function(
                "count",
                func=lambda x, y: len([el for el in y if el == x]),
                input_types=[int_type, list_type],
                output_type=int_type,
                is_symmetric=False,
            ),
            # TODO: should head return element or [element]?
            #        'head':     Function('head', func=lambda x: head(x), input_types=[list_type], output_type=int_type,
            #                             val_input_args_func=(lambda x: (len(x) > 0))),
            # TODO: should tail return element or [element]?
            "tail": Function(
                "tail",
                func=lambda x: tail(x),
                input_types=[list_type],
                output_type=int_type,
                val_input_args_func=(lambda x: (len(x) > 0)),
                is_symmetric=False,
            ),
            "first_n": Function(
                "first_n",
                func=lambda x, y: [y[i] for i in range(x)],
                input_types=[int_type, list_type],
                output_type=list_type,
                val_input_args_func=(lambda x, y: (x <= len(y) and x > 0)),
                is_symmetric=False,
            ),
            # TODO: change if back to separate evaluation (to not eval both y and z, but only one based on value of x?)
            # TODO: treating like function
            # TODO: setting nth to return [] when index error; is this desired behavior?
            "nth": Function(
                "nth",
                func=lambda x, y: nth(x, y),
                input_types=[int_type, list_type],
                output_type=int_type,
                val_input_args_func=(lambda x, y: (x < len(y) and x >= 0)),
                is_symmetric=False,
            ),
            # TODO: change if back to separate evaluation (to not eval both y and z, but only one based on value of x?)
            # TODO: treating like function
        }
    )

    return env


def fraction_env():
    "An environment for fractions"
    env = Env()

    fraction_type = Fraction
    problem_type = FractionProblem
    bool_type = bool

    def is_common_denoms(problem):
        return problem.frac1.denom == problem.frac2.denom

    # TODO: return type of this func + others?
    def create_common_denoms(problem):
        """Takes in a fraction problem and returns a new one
        where the fractions are mapped to common denominators"""

        common_denom = get_common_denom(problem)
        frac1_multiplier = int(common_denom / problem.frac1.denom)
        frac2_multiplier = int(common_denom / problem.frac2.denom)

        new_frac1_num = problem.frac1.num * frac1_multiplier
        new_frac1_denom = problem.frac1.denom * frac1_multiplier
        assert (
            new_frac1_denom == common_denom
        ), f"New denom should be {common_denom} but is {new_frac1_denom}"

        new_frac2_num = problem.frac2.num * frac2_multiplier
        new_frac2_denom = problem.frac2.denom * frac2_multiplier

        assert (
            new_frac2_denom == common_denom
        ), f"New denom should be {common_denom} but is {new_frac2_denom}"

        new_frac1 = Fraction(new_frac1_num, new_frac1_denom)
        new_frac2 = Fraction(new_frac2_num, new_frac2_denom)

        return FractionProblem(new_frac1, new_frac2, problem.str_op)

    def get_common_denom(problem):
        """Takes in a fraction problem and returns the common denominator"""

        def lcm(a, b):
            return (a * b) // math.gcd(a, b)

        return lcm(problem.frac1.denom, problem.frac2.denom)

    def get_common_denom_no_num(problem):
        """Takes in a fraction problem and maps the denominators to the common denominator, but not the numerators"""

        common_denom = get_common_denom(problem)

        new_frac1 = Fraction(problem.frac1.num, common_denom)
        new_frac2 = Fraction(problem.frac2.num, common_denom)

        return FractionProblem(new_frac1, new_frac2, problem.str_op)

    def apply_op_to_num_and_denom(problem):
        new_num = problem.op(problem.frac1.num, problem.frac2.num)
        new_denom = problem.op(problem.frac1.denom, problem.frac2.denom)
        return Fraction(new_num, new_denom)

    def apply_to_num(problem):
        new_num = problem.op(problem.frac1.num, problem.frac2.num)
        assert is_common_denoms(
            problem
        ), f"Denominators should be common if only applying operation to numerators"
        return Fraction(new_num, problem.frac1.denom)

    # TODO: should this only be for multiplication?
    def cross_apply(problem):
        new_num = problem.op(problem.frac1.num, problem.frac2.denom)
        new_denom = problem.op(problem.frac1.denom, problem.frac2.num)
        return Fraction(new_num, new_denom)

    def cross_multiply(problem):
        assert problem.str_op == "*", f"Operation should be * but is {problem.str_op}"
        new_num = problem.frac1.num * problem.frac2.denom
        new_denom = problem.frac1.denom * problem.frac2.num
        return Fraction(new_num, new_denom)

    def add_nums(problem):
        # return fraction with just the sum of the numerators
        # if denoms are not common, then this is not a valid operation, return None? (TODO: is this robust?)
        if not is_common_denoms(problem):
            return None
        new_num = problem.frac1.num + problem.frac2.num
        return Fraction(new_num, problem.frac1.denom)

    def add_nums_and_denoms(problem):
        assert problem.str_op == "+", f"Operation should be + but is {problem.str_op}"
        new_num = problem.frac1.num + problem.frac2.num
        new_denom = problem.frac1.denom + problem.frac2.denom
        return Fraction(new_num, new_denom)

    def add_both_if_diff_denoms_else_nums(problem):
        # if denoms are different, then add both numerators and denominators
        # if denoms are same, just add numerators
        if not is_common_denoms(problem):
            return add_nums_and_denoms(problem)
        else:
            return add_nums(problem)

    def multiply_nums(problem):
        # return fraction with just the product of the numerators
        # if denoms are not common, then this is not a valid operation, return None? (TODO: is this robust?)
        if not is_common_denoms(problem):
            return None
        new_num = problem.frac1.num * problem.frac2.num
        return Fraction(new_num, problem.frac1.denom)

    def multiply_nums_and_denoms(problem):
        assert problem.str_op == "*", f"Operation should be * but is {problem.str_op}"
        new_num = problem.frac1.num * problem.frac2.num
        new_denom = problem.frac1.denom * problem.frac2.denom
        return Fraction(new_num, new_denom)

    def multiply_nums_if_common_else_both(problem):
        # if denoms are common, then multiply numerators
        # if denoms are different, then multiply both numerators and denominators
        if is_common_denoms(problem):
            return multiply_nums(problem)
        else:
            return multiply_nums_and_denoms(problem)

    def make_common_denoms_and_add_nums(problem):
        # make sure addition problem
        assert problem.str_op == "+", f"Operation should be + but is {problem.str_op}"
        # create common denominators
        problem = create_common_denoms(problem)
        # add numerators
        return add_nums(problem)

    def make_common_denoms_and_multiply_nums(problem):
        # make sure multiplication problem
        assert problem.str_op == "*", f"Operation should be * but is {problem.str_op}"
        # create common denominators
        problem = create_common_denoms(problem)
        # multiply numerators
        return multiply_nums(problem)

    def create_common_denoms_add(problem):
        # make sure addition problem
        assert problem.str_op == "+", f"Operation should be + but is {problem.str_op}"
        # create common denominators
        return create_common_denoms(problem)

    def create_common_denoms_multiply(problem):
        # make sure multiplication problem
        assert problem.str_op == "*", f"Operation should be * but is {problem.str_op}"
        # create common denominators
        return create_common_denoms(problem)

    avail_functions = {
        # TODO: should be considered symmetric?
        "is_common_denoms": Function(
            "is_common_denoms",
            func=lambda x: is_common_denoms(x),
            input_types=[problem_type],
            output_type=bool_type,
            is_symmetric=False,
        ),
        "is_common_denoms_or_add": Function(
            "is_common_denoms_or_add",
            func=lambda x: is_common_denoms(x) or x.str_op == "+",
            input_types=[problem_type],
            output_type=bool_type,
            is_symmetric=False,
        ),
        "create_common_denoms_no_num": Function(
            "create_common_denoms_no_num",
            func=lambda x: get_common_denom_no_num(x),
            input_types=[problem_type],
            output_type=problem_type,
            is_symmetric=False,
        ),
        "apply_op_to_num_and_denom": Function(
            "apply_op_to_num_and_denom",
            func=lambda x: apply_op_to_num_and_denom(x),
            input_types=[problem_type],
            output_type=fraction_type,
            is_symmetric=False,
        ),
        "apply_to_num": Function(
            "apply_to_num",
            func=lambda x: apply_to_num(x),
            input_types=[problem_type],
            output_type=fraction_type,
            is_symmetric=False,
        ),
        "cross_apply": Function(
            "cross_apply",
            func=lambda x: cross_apply(x),
            input_types=[problem_type],
            output_type=fraction_type,
            is_symmetric=False,
        ),
        "is_multiply": Function(
            "is_multiply",
            func=lambda x: x.str_op == "*",
            input_types=[problem_type],
            output_type=bool_type,
            is_symmetric=False,
        ),
        "is_add": Function(
            "is_add",
            func=lambda x: x.str_op == "+",
            input_types=[problem_type],
            output_type=bool_type,
            is_symmetric=False,
        ),
        "always_apply_to_num": Function(
            "always_apply_to_num",
            func=lambda x: True,
            input_types=[problem_type],
            output_type=bool_type,
            is_symmetric=False,
        ),
        "always_apply_to_num_and_denom": Function(
            "always_apply_to_num_and_denom",
            func=lambda x: False,
            input_types=[problem_type],
            output_type=bool_type,
            is_symmetric=False,
        ),
        "create_common_denoms": Function(
            "create_common_denoms",
            func=lambda x: create_common_denoms(x),
            input_types=[problem_type],
            output_type=problem_type,
            is_symmetric=False,
        ),
        "do_nothing_add": Function(
            "do_nothing_add",
            func=lambda x: x,
            input_types=[problem_type],
            output_type=problem_type,
            is_symmetric=False,
        ),
        "do_nothing_multiply": Function(
            "do_nothing_multiply",
            func=lambda x: x,
            input_types=[problem_type],
            output_type=problem_type,
            is_symmetric=False,
        ),
        "add_nums": Function(
            "add_nums",
            func=lambda x: add_nums(x),
            input_types=[problem_type],
            output_type=problem_type,
            is_symmetric=True,
        ),
        "add_nums_and_denoms": Function(
            "add_nums_and_denoms",
            func=lambda x: add_nums_and_denoms(x),
            input_types=[problem_type],
            output_type=problem_type,
            is_symmetric=True,
        ),
        "add_both_if_diff_denoms_else_nums": Function(
            "add_both_if_diff_denoms_else_nums",
            func=lambda x: add_both_if_diff_denoms_else_nums(x),
            input_types=[problem_type],
            output_type=problem_type,
            is_symmetric=True,
        ),
        "multiply_nums": Function(
            "multiply_nums",
            func=lambda x: multiply_nums(x),
            input_types=[problem_type],
            output_type=problem_type,
            is_symmetric=True,
        ),
        "multiply_nums_and_denoms": Function(
            "multiply_nums_and_denoms",
            func=lambda x: multiply_nums_and_denoms(x),
            input_types=[problem_type],
            output_type=problem_type,
            is_symmetric=True,
        ),
        "multiply_nums_if_common_else_both": Function(
            "multiply_nums_if_common_else_both",
            func=lambda x: multiply_nums_if_common_else_both(x),
            input_types=[problem_type],
            output_type=problem_type,
            is_symmetric=True,
        ),
        "cross_multiply": Function(
            "cross_multiply",
            func=lambda x: cross_multiply(x),
            input_types=[problem_type],
            output_type=problem_type,
            is_symmetric=True,
        ),
        "create_common_denoms_add": Function(
            "create_common_denoms_add",
            func=lambda x: create_common_denoms_add(x),
            input_types=[problem_type],
            output_type=problem_type,
            is_symmetric=True,
        ),
        "create_common_denoms_multiply": Function(
            "create_common_denoms_multiply",
            func=lambda x: create_common_denoms_multiply(x),
            input_types=[problem_type],
            output_type=problem_type,
            is_symmetric=True,
        ),
        "make_common_denoms_and_add_nums": Function(
            "make_common_denoms_and_add_nums",
            func=lambda x: make_common_denoms_and_add_nums(x),
            input_types=[problem_type],
            output_type=problem_type,
            is_symmetric=True,
        ),
        "make_common_denoms_and_multiply_nums": Function(
            "make_common_denoms_and_multiply_nums",
            func=lambda x: make_common_denoms_and_multiply_nums(x),
            input_types=[problem_type],
            output_type=problem_type,
            is_symmetric=True,
        ),
    }

    env.update(avail_functions)
    # env.update({k: avail_functions[k] for k in functions})

    return env
