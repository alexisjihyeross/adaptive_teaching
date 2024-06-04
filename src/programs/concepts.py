import warnings
import torch
from typing import List, Union, Optional

from src.programs.utils import *


def sample_concept_probs(num_concepts):
    concept_probs = torch.rand(num_concepts)
    return concept_probs


def get_concept_library_from_params(interp, params):
    # params are the output of get_student_concept_params()
    concepts = ConceptLibrary(
        interp,
    )
    prior_over_concepts = concepts.probs_dict_to_list(params["concept_probs"])
    concepts.set_concept_probs(probs=prior_over_concepts)
    return concepts


class ConceptLibrary:

    def __init__(
        self,
        interpreter,
    ):

        self.interpreter = interpreter
        self.initialize()
        self.probs = None

    # TODO: let functions be customizable too?
    # TODO: allow partially specified concepts (like '(add (sum x_) ___)' instead of '(add (sum x_) x_)')
    def initialize(
        self,
    ):
        # TODO: will this function always return concept_dict in the same order? (should be?)
        """
        Returns a dictionary of concept_dict from an interpreter (get functions from that)

        Initializes concept library:
        - concept_dict: dict of dicts where outer key is str concept
        - concept_list: list of dicts where idx is idx of concept
        """
        concept_idx = 0

        concept_dict = {}

        concept_list = []
        functions = self.interpreter.global_env

        for func, raw_func in functions.items():
            func_dict = {
                "idx": concept_idx,
                "str": func,
                "type": "function",
                "raw": raw_func,
            }
            concept_dict[func] = func_dict
            concept_list.append(func_dict)
            concept_idx += 1

        self.concept_dict = concept_dict
        self.concept_list = concept_list

    def get_concept_idx(self, concept):
        return self.concept_dict[concept]["idx"]

    def __len__(self):
        return len(self.concept_list)

    # I think this should define what happens when list() is called on a concept library object
    def __iter__(self):
        return iter(self.concept_list)

    def get_concept_dict(self):
        return self.concept_dict

    def probs_dict_to_list(self, probs_dict):
        probs_lst = torch.ones(len(self.concept_list))
        # DEFAULT is special value for setting all concepts not defined
        if "DEFAULT" not in probs_dict:
            assert len(probs_dict) == len(
                self.concept_list
            ), f"len(probs_dict) {len(probs_dict)} != len(self.concept_list) {len(self.concept_list)}\n\nProbs dict: {probs_dict}\nConcept list: {self.concept_list}"
        else:
            probs_lst = probs_lst * probs_dict["DEFAULT"]
        for c, p in probs_dict.items():
            if c == "DEFAULT":
                continue
            c_idx = self.get_concept_idx(c)
            probs_lst[c_idx] = p
        return probs_lst

    # probs can be a list or dict
    def set_concept_probs(self, probs=None, do_print=False):
        if probs is None:
            if do_print:
                print(f"Probs is none, resampling concept probs...")
            probs = sample_concept_probs(len(self.concept_list))
            supplied_probs = False
            self.probs = probs
        elif isinstance(probs, dict):
            probs_list = self.probs_dict_to_list(probs)
            supplied_probs = True
            self.probs = probs_list
        else:
            if do_print:
                print(f"Probs is supplied: {probs}")
            supplied_probs = True
            self.probs = probs

        if do_print:
            self.print()

    def print(self):
        sorted_zip = sorted(
            zip(self.concept_list, self.probs), key=lambda x: x[1], reverse=True
        )
        for concept, prob in sorted_zip:
            print(f"{round(prob.item(), 3)}:\t{concept['str']}")

    def get_concept_probs(self):
        assert (
            self.probs is not None
        ), "Looks like the concept probabilities haven't been set yet"
        return self.probs
