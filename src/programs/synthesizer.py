from tqdm.contrib.itertools import product
from tqdm import tqdm

from src.programs.interpreter import *
from src.programs.utils import *


class Program:
    def __init__(self, str_prog):
        self.prog = prog

    def get_depth():
        pass


# TODO: allow input as argument at any depth level?

# Create function to create a string for a function based on number of args and args given; should be separate from Function class


def filter_expressions_by_return_type(expressions, return_type):
    filtered = [e for e in expressions if e.get_output_type() == return_type]
    return filtered


def validate_return_type(exp, return_type):
    # Returns True if expression has output type matching desired return type
    return exp.get_output_type() == return_type


################ Filter Criteria
# TODO: implement; this is not comprehensive
def filter_prog_simple(func, args):
    if func.name in ["mod"]:
        # filter if all args are non-variable primitives
        if all([(arg.is_primitive() and arg.concept.value != "x_") for arg in args]):
            return True
        # if first two args are both x_, uninteresting
        if all([(arg.is_primitive() and arg.concept.value == "x_") for arg in args]):
            return True
        if args[1].is_primitive() and args[1].concept.value == "0":
            return True

        # TODO: for now, get rid of ones where first number is primitive non-x_ (i.e. 5 mod x)
        # eventually: get rid of?
        if args[0].is_primitive() and args[0].concept.value != "x_":
            return True

        """
        # TODO: COME BACK
        # FOR NOW, filter if not all args are primitives
        if any([(not arg.is_primitive()) for arg in args]): 
            return True
        """
    elif func.name in ["even"]:
        if args[0].is_primitive() and args[0].concept.value != "x_":
            return True
    elif func.name in ["divide"]:
        if args[1].is_primitive() and args[1].concept.value in [0, 1]:
            return True
    elif func.name in ["multiply"]:
        if args[1].is_primitive() and args[1].concept.value in [0, 1]:
            return True
        if args[0].is_primitive() and args[0].concept.value in [0, 1]:
            return True
    elif func.name in ["add_int"]:
        # Filter if one of the ints being added is 0
        if args[1].is_primitive() and args[1].concept.value == 0:
            return True
        if args[0].is_primitive() and args[0].concept.value == 0:
            return True
    elif func.name in ["greater_than"]:
        # filter if both arguments for greater than are equal
        # TODO: hacky, currently just checking the string representations
        # (create func for checking expression equality?)
        if str(args[0]) == str(args[1]):
            return True

    # for all functions: filter if all args are non-variable primitives
    if all([(arg.is_primitive() and arg.concept.value != "x_") for arg in args]):
        return True
    return False


# For standard environments
def filter_prog_standard(func, args):
    # TODO: it would be more robust to somehow use function.val_input_args() by evaluating args that have values

    # For add, filter: adding to empty list (which returns empty list for any number), adding 0
    # For nth, filter: taking eleemnt of empty list (which returns empty list for any number), indexing by 0 since that shouldn't be allowed
    # TODO: add functions first_n, add_int, etc. -- make sure all funcs in interpreter are accounted for
    if func.name in ["int_to_list"]:
        if args[0].is_primitive():
            return True
    elif func.name in ["add", "nth"]:
        if args[1].is_primitive() and args[1].concept.value == []:
            return True
        if args[0].is_primitive() and args[0].concept.value == 0:
            return True
        """
        # Hacky, but apply extra filtering for nth to not allow programs that lead to indexing errors
        # TODO: what about if the list arg is a function that returns a list that is too short?
        if func.name == "nth":
            if args[0] == 0:
                continue
            if args[1].get_output_type() == list and args[0] < len(args[1].value):
                continue
        """
    elif func.name in ["tail", "head", "sort"]:
        if args[0].is_primitive() and args[0].concept.value == []:
            return True
    elif func.name in ["remove"]:
        if args[1].is_primitive() and args[1].concept.value == []:
            return True
    elif func.name in ["add_int"]:
        if args[1].is_primitive() and args[1].is_primitive():
            return True
        if args[0].is_primitive() and args[0].concept.value == 0:
            return True
        if args[1].is_primitive() and args[1].concept.value == 0:
            return True
    elif func.name in ["first_n"]:
        if args[0].is_primitive() and args[0].concept.value == 0:
            return True
    else:
        return False


# class to enumerate all possible programs
class ProgramEnumerator(object):
    def __init__(self, env_name="standard"):
        self.env_name = env_name
        self.env = get_env(env_name)

        # the type of x_ (used in figuring out how to enumerate programs, by get_primitive_expressions()
        # i.e. only give x_ as input to functions w right input types
        # TODO: setting this to be list except when env is simple
        self.input_var_type = int if self.env_name == "simple" else list

    def get_primitive_expressions(self):
        # All the primitives that can be the base arguments (not including list, because assume given)
        # TODO: include lists that aren't given?

        primitives = [Primitive(num) for num in range(0, 10)]
        #    primitives.append(Primitive([]))
        # TODO: have separate "Variable" type?
        primitives.append(Primitive("x_", output_type=self.input_var_type))
        primitives = [Expression(p) for p in primitives]
        return primitives

    def get_functions(self):
        print(self.env)
        functions = list(self.env.values())
        return functions

    def filter_prog(self, func, args):
        if self.env_name == "simple":
            return filter_prog_simple(func, args)
        elif self.env_name == "standard":
            return filter_prog_standard(func, args)
        else:
            raise ValueError()

    def enumerate_expressions(self, max_depth=1, functions=None):
        depth = 0
        primitives = self.get_primitive_expressions()

        if functions is None:
            functions = self.get_functions()

        expressions = primitives
        while depth < max_depth:
            print("DEPTH:", depth)
            print("# expressions:", len(expressions))
            prev_expressions = expressions.copy()
            for func in tqdm(functions):
                # Get the potential arguments for each input argument of the function, based on programs created in prev round
                """
                # Potential args by argument (ordered)
                potential_args = []
                for inp_type in func.input_types:
                    filtered = filter_expressions_by_return_type(prev_expressions, inp_type)
                    potential_args.append(filtered)
                """

                # seen_args stores frozen_sets of arguments that have already been seen
                # this is only used for symmetric functions
                seen_args = set()

                print("\nFunction: ", func.name)
                # Consider all possible arrangements of arguments, continue if not all are correct
                # Need to get rid of duplicate arrangements (same values but different ordering) for functions that are symmetric, like equals/multiply
                args_generator = product(
                    *tuple([prev_expressions] * len(func.input_types))
                )
                num_combs = len(prev_expressions) ** len(func.input_types)
                func_expressions = [None] * (num_combs)
                comb_idx = 0
                comb_filtered_idx = 0
                num_filtered = 0
                for args in args_generator:
                    #                if comb_idx != 0 and comb_idx % 1000000 == 0:
                    #                    print(f"{comb_idx}/{num_combs} ({round(comb_idx/num_combs, 3)})")
                    comb_idx += 1
                    # Check that all the input args return the appropriate type
                    # But can't check that the values are valid with val_input_args()
                    # bc don't know what they evaluate to (for example, when input arg is input variable _x)
                    if not all(
                        [
                            validate_return_type(arg, input_type)
                            for arg, input_type in zip(args, func.input_types)
                        ]
                    ):
                        num_filtered += 1
                        continue
                    # See if we should filter this program according to criterion for each func
                    if self.filter_prog(func, args):
                        num_filtered += 1
                        continue

                    # If the function is symmetric, make sure we don't repeat different orderings of same set of args
                    if func.is_symmetric:
                        arg_set = frozenset(args)
                        if arg_set in seen_args:
                            num_filtered += 1
                            continue

                        seen_args.add(arg_set)

                    temp_expression = FunctionExpression(func, input_args=list(args))
                    func_expressions[comb_filtered_idx] = temp_expression
                    comb_filtered_idx += 1
                # TODO: make sure all other elements in fun_expressions are None
                expressions.extend(func_expressions[:comb_filtered_idx])
                print(f"Filtered {num_filtered}/{num_combs}")

            depth += 1

        return expressions

    def sample_programs(self, max_depth=1):
        if self.env_name == "standard":
            expressions = self.enumerate_expressions(max_depth=max_depth)
        elif self.env_name == "simple":
            """
            gold_prog = '(if (even x_) 2 elif (greater_than x_ 3) 3 else (complex_func x_))'
            # get expressions to replace (even x_)
            functions = [self.env[f] for f in ['equals', 'mod', 'divide', 'multiply']]
            even_expressions = self.enumerate_expressions(max_depth=max_depth, functions=functions)

            for exp in even_expressions:
                if validate_return_type(exp, bool):
                    print(exp)
            return even_expressions
            """
            function_names = ["mod", "multiply", "add_int"]
            functions = [self.env[f] for f in function_names]
            expressions = self.enumerate_expressions(
                max_depth=max_depth, functions=functions
            )

        return expressions
