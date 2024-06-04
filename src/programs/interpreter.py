################ Lispy: Scheme Interpreter in Python

## (c) Peter Norvig, 2010-16; See http://norvig.com/lispy.html

from __future__ import division
import math
import operator as op

import bisect
import re

from src.programs.types import *
from src.programs.environment import *
from src.programs.fractions.lib import *

# TODO: implement True/False?


################ Procedures


class Procedure(object):
    "A user-defined Scheme procedure."

    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env

    def __call__(self, *args):
        return self.eval(self.body, Env(self.parms, args, self.env))


################ Interaction: A REPL


def lispstr(exp):
    "Convert a Python object back into a Lisp-readable string."
    if isinstance(exp, List):
        return "(" + " ".join(map(lispstr, exp)) + ")"
    else:
        return str(exp)


class Interpreter(object):
    def __init__(self, env_name="standard", **kwargs):
        self.env_name = env_name
        self.env = get_env(env_name=env_name, **kwargs)
        self.global_env = self.env
        self.var_name = "x_"

    def parse(self, program):
        "Read a Scheme expression from a string."
        return self.read_from_tokens(self.tokenize(program))

    def tokenize(self, s):
        "Convert a string into a list of tokens."
        return (
            s.replace("(", " ( ")
            .replace(")", " ) ")
            .replace("]", " ] ")
            .replace("[", " [ ")
            .split()
        )

    def read_from_tokens(self, tokens):
        "Read an expression from a sequence of tokens."
        if len(tokens) == 0:
            raise SyntaxError("unexpected EOF while reading")
        token = tokens.pop(0)
        if "(" == token:
            L = []
            while tokens[0] != ")":
                L.append(self.read_from_tokens(tokens))
            tokens.pop(0)  # pop off ')'
            return L
        elif ")" == token:
            raise SyntaxError("unexpected )")
        elif "[" == token:
            L = []
            while tokens[0] != "]":
                L.append(self.read_from_tokens(tokens))
            tokens.pop(0)
            return L
        elif "]" == token:
            raise SyntaxError("unexpected ]")
        else:
            return self.atom(token)

    def atom(self, token):
        "'None' becomes special type; ints become ints; lists become lists; every other token is a symbol."
        # TODO: use Expressions intead of just Symbols (str)/Primitives
        try:
            token = int(token)
            return Primitive(token)
        except ValueError:
            try:
                token = list([int(el) for el in token])
                return Primitive(token)
            except ValueError:
                # try:
                #     tokens = token.split("x")
                try:
                    token = parse_frac(token)
                    # print(f"parsed token: {token}")
                    return token
                except ValueError:

                    if list(token) == []:
                        return Primitive([])
                    elif token == "None":
                        # TODO: what is this output_type used for
                        return Primitive(None, output_type=None)
                    elif token == self.var_name:
                        # TODO: hacky, fit this with rest?
                        return Primitive("x_", output_type=list)
                    # For conditional words like if/elif, return just the token
                    elif token in ["if", "elif", "else"]:
                        return token
                    # Symbol are things in the env, i.e. functions
                    return self.global_env.find(token)[token]

    ################ eval

    def eval_str(self, x):
        return self.eval(self.parse(x))

    def eval_func(self, func_concept, args):
        self.validate_args(func_concept, args)
        return func_concept.func(*args)

    def validate_args(self, func_concept, args):
        # print("validating args")
        # print(func_concept)

        # Validate types
        given_types = [type(arg) for arg in args]
        expected_types = [arg_type for arg_type in func_concept.input_types]
        all_args_correct = all([g == e for g, e in zip(given_types, expected_types)])
        if not all_args_correct:
            error_msg = "Given input args do not match expected types:"
            error_msg = f"\n\nFunction: {func_concept.to_string()}"
            error_msg += "\nGiven:"
            for g in given_types:
                error_msg += f"\n | {g}"
            error_msg += "\n\nExpected:"
            for e in expected_types:
                error_msg += f"\n | {e}"
            raise TypeError(error_msg)

        # Validate values
        validated = func_concept.val_input_args(args)
        if not validated:
            error_msg = "Received incorrect values for function:"
            error_msg += f"\nFunction: {func_concept.to_string()}"
            # Args should be ints/lists at this point, bc already evaluated
            for arg in args:
                error_msg += f"\n | {arg}"
            raise ValueError(error_msg)

    def eval(self, x, env=None):
        # TODO: change arg to some other word, bc use arg in expressions in synthesizer, and those are of different types

        # Leave env is arg in case want to create local environments based on defined variables
        # TODO: can handle (3), [3], but not ([3]) -- What is desired behavior?
        "Evaluate an expression in an environment."
        if env is None:
            env = self.global_env

        # print(x)
        # print(isinstance(x, list))

        # Primitive
        if isinstance(x, Primitive):
            val = x.value

        elif isinstance(x, FractionProblem):
            val = x

        # Function
        elif len(x) > 0 and isinstance(x[0], Function):
            # Eval the first token to get a function
            func_concept = x[0]
            # Eval the args
            args = [self.eval(exp, env) for exp in x[1:]]
            val = self.eval_func(func_concept, args)

        # Conditional: assumes formatted based on as:
        # (if cond0 conseq0 elif cond1 conseq1 else conseq2)
        # TODO: extend to handle arbitrary numbers of conditionals
        # TODO: added x != [] in case empty list is being evaluated, runs into issue... look into this
        elif x != [] and x[0] == "if" and x[3] == "elif":
            (_, cond0, conseq0, _, cond1, conseq1, _, conseq2) = x
            if self.eval(cond0, env):
                val = self.eval(conseq0, env)
            elif self.eval(cond1, env):
                val = self.eval(conseq1, env)
            else:
                val = self.eval(conseq2, env)
        # (if cond0 conseq0 else conseq1)
        # TODO: added x != [] in case empty list is being evaluated, runs into issue... look into this
        elif x != [] and x[0] == "if" and x[3] == "else":
            (_, cond0, conseq0, _, conseq1) = x
            if self.eval(cond0, env):
                val = self.eval(conseq0, env)
            else:
                val = self.eval(conseq1, env)

        # List: evaluate each element separately
        elif isinstance(x, List):
            val = [self.eval(el) for el in x]
        # x should never be a Function because the only non-primitives are functions and those always have arguments
        else:
            raise ValueError
        return val

    def run_program(self, str_prog, inp, strict=False):
        # Return not strict, return None if there is a value error

        # TODO: give x_ as an input? or global var?
        # Assumes programs have x_ for input variable
        pattern = r"(?<=\s|\()x_(?=\s|\))"  # use regex to match 'x_' if it's preceded by whitespace or '(' and followed by whitespace or ')'
        str_prog = re.sub(pattern, str(inp), str_prog).replace(",", "")
        if strict:
            return self.eval_str(str_prog)
        else:
            try:
                return self.eval_str(str_prog)
            except (ValueError, ZeroDivisionError):
                return None
