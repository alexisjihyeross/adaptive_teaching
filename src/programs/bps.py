from typing import List, Union, Optional
from tqdm import tqdm
import torch
import warnings
import numpy as np
import collections
import pdb

from src.programs.interpreter import *
from src.programs.synthesizer import *
from src.programs.utils import *
from src.programs.concepts import *


def students_are_equal(s1, s2):
    """Given two BayesianProgramSynthesizer students, check if they are equal."""

    # TODO: also consider other attributes?
    return all(s1.concepts.probs == s2.concepts.probs)


def get_pop_idx(populations, bps):
    """Given a list of populations (ie BayesianProgramSynthesizers), get the index of the population that matches bps"""
    matching_populations = [
        idx for idx, s in enumerate(populations) if students_are_equal(s, bps)
    ]
    assert (
        len(matching_populations) == 1
    ), f"matching populations: {matching_populations}; populations: {populations}"
    return matching_populations[0]


def check_prog_correctness(prog, inputs, labels, interp):
    """Checks program correctness on all inputs/outputs"""
    outputs = [interp.run_program(prog, inp, strict=False) for inp in inputs]
    correct = [o == lab for o, lab in zip(outputs, labels)]
    return all(correct)


# used when there is an error in computing the posterior
class PosteriorError(Exception):
    pass


class BayesianProgramSynthesizer:
    def __init__(
        self,
        hypotheses: list,
        interpreter: Interpreter,
        concepts: ConceptLibrary,
        dataset,
        # used for pre-computing possible outputs
        progs_reps=None,
        canonicalized_to_prog=None,
        prog_to_canonicalized=None,
        outputs_by_inp=None,
        noise=0,  # prob with which student beliefs incorrect labels are generated
        do_print=False,
    ):
        self.dataset = dataset
        self.do_print = do_print

        self.torch_dtype = torch.float64
        torch.set_default_dtype(self.torch_dtype)

        self.noise = noise

        self.interpreter = interpreter
        self.prog_to_canonicalized = prog_to_canonicalized
        self.canonicalized_to_prog = canonicalized_to_prog

        # Hypotheses are programs
        self.all_hypotheses = hypotheses
        self.all_parsed = [self.interpreter.parse(prog) for prog in self.all_hypotheses]
        # Remaining hypotheses

        self.hypotheses = hypotheses
        self.progs_reps = progs_reps
        self.all_progs_reps = progs_reps

        self.concepts = concepts
        self.initialize_prior()
        self.posterior = self.prior.clone()
        self.all_posterior = self.prior.clone()

        self.predict_cache = {}

        if not torch.isclose(
            self.posterior.sum(), torch.tensor([1.0], dtype=self.torch_dtype)
        ):
            raise PosteriorError(
                f"self.posterior.sum() should be 1 but is {self.posterior.sum().item()}"
            )
        if not torch.isclose(
            self.all_posterior.sum(), torch.tensor([1.0], dtype=self.torch_dtype)
        ):
            raise PosteriorError(
                f"self.all_posterior.sum() should be 1 but is {self.all_posterior.sum().item()}"
            )

        # has len(posterior) and has, for each remaining hyp/posterior prob, the orig index in all_hypothess
        self.hyp_indices = torch.arange(len(self.all_hypotheses))

        self.hyp_to_idx = {
            hyp: idx for idx, hyp in zip(self.hyp_indices, self.all_hypotheses)
        }

        if outputs_by_inp is None:
            self.precompute_possible_outputs()
        else:
            self.outputs_by_inp = outputs_by_inp
            self.possible_outputs = set(
                {i for sub_list in self.outputs_by_inp.values() for i in sub_list}
            )

        #        print("possible outputs: ", self.possible_outputs)

        if self.do_print:
            print("Done initializing Bayesian Program Synthesizer.")

    def precompute_possible_outputs(self):
        # precompute all possible outputs; these will be sampled from when a prog is noisy
        if self.do_print:
            print("Precomputing all possible outputs...")
        possible_outputs = set()

        self.outputs_by_inp = {}
        outputs_by_inp = {}
        for x in self.dataset.inputs:
            outputs = self.compute_outputs(x)
            possible_outputs.update(outputs)
            outputs_by_inp[x] = outputs

        self.possible_outputs = possible_outputs
        self.outputs_by_inp = outputs_by_inp
        if self.do_print:
            print("Done.")

    # TODO: change based on prior_over_concepts
    # TODO: move to log space
    def get_prior_over_progs(self, prior_over_concepts):
        torch.set_default_dtype(self.torch_dtype)

        # Raise each func value in prior_over_functions by number of times it appears in program
        # Then multiply all func values with each other
        exponentiated_values = torch.pow(prior_over_concepts, self.all_progs_reps)
        prior_over_progs = torch.prod(exponentiated_values, dim=1)

        prior_over_progs /= prior_over_progs.sum()

        return prior_over_progs

    def update_concepts(self, prior_over_concepts=None):
        torch.set_default_dtype(self.torch_dtype)
        """ Updates underlying concepts (ConceptLibrary object) 
        and resets prior over progs.
        If prior_over_concepts is None, resamples probs over concepts """

        self.concepts.set_concept_probs(probs=prior_over_concepts, do_print=False)
        self.set_prior_over_progs(prior_over_concepts)

    # TODO: call get_prior_over_progs() to reduce redundancy?
    def set_prior_over_progs(self, prior_over_concepts):
        torch.set_default_dtype(self.torch_dtype)
        prior_over_progs = self.get_prior_over_progs(prior_over_concepts)
        self.prior = prior_over_progs

        # also reset posterior
        # TODO: instead re-compute posterior, instead of setting posteriors to priors?
        # (I think this is only called in updating teacher, and there the posteriors are all re-computed?)
        self.posterior = self.prior.clone()
        self.all_posterior = self.prior.clone()

    def get_prog_vector(self, prog):
        counts_by_concept = torch.zeros(len(self.concepts))
        parsed = self.interpreter.parse(prog)
        if isinstance(parsed, Primitive):
            return counts_by_concept
        counts_by_concept = get_concept_counts(
            parsed, self.concepts.get_concept_dict(), counts_by_concept
        )
        return counts_by_concept

    # prior_over_concepts should be indexed based on the concept_indices
    def initialize_prior(self):
        torch.set_default_dtype(self.torch_dtype)
        """ Sets prior over programs and creates program representations. 
        If prior_over_concepts is not supplied, sample and update the concepts and prior_over_concepts (by calling update_concepts()) accordingly.
        """
        if self.do_print:
            print("Initializing prior...")
        if self.progs_reps is None:
            if self.do_print:
                print("Getting program representations...")
            # TODO: all_hypotheses?
            self.progs_reps = torch.stack(
                [
                    get_prog_vector(parsed, self.concepts.get_concept_dict())
                    for parsed in tqdm(self.all_parsed)
                ]
            )
            self.all_progs_reps = self.progs_reps.clone()

        # this will resample probs if self.concepts.probs is None
        self.update_concepts(prior_over_concepts=self.concepts.probs)
        if self.do_print:
            print("Done initializing prior.")

    def get_num_nonzero_hyps(self):
        return len(self.hypotheses)

    def get_nonzero_hyps(self):
        return self.hypotheses

    def get_hyp_idx(self, hyp):
        return self.hyp_to_idx[hyp]

    def get_hyp_prob(self, prog):
        return self.posterior[self.get_hyp_idx(prog)].item()

    def get_total_prob(self, progs):
        prob = 0
        for prog in progs:
            prob += self.posterior[self.get_hyp_idx(prog)].item()
        return prob

    def get_canonical_hyp_prob(self, prog):
        assert self.canonicalized_to_prog is not None
        assert self.prog_to_canonicalized is not None
        can_idx = self.prog_to_canonicalized[prog]
        return self.get_total_prob(self.canonicalized_to_prog[can_idx])

    def compute_outputs(self, x):
        if x in self.outputs_by_inp:
            return self.outputs_by_inp[x]
        if len(self.hypotheses) > 100000:
            outputs = [
                self.interpreter.run_program(prog, x, strict=False)
                for prog in tqdm(self.hypotheses)
            ]
        else:
            #            outputs = [self.interpreter.run_program(prog, x, strict=False) for prog in self.hypotheses]
            outputs = []
            for prog in self.hypotheses:
                output = self.interpreter.run_program(prog, x, strict=False)
                outputs.append(output)
        return outputs

    def update_posterior(self, x: list, y: Union[list, int]):
        raise NotImplementedError()

    def compute_posterior(self, x: Union[list, int], y: Union[list, int]):
        # Compute outputs and check which match target

        if x in self.outputs_by_inp:
            outputs = self.outputs_by_inp[x]
        else:
            # This shouldn't happen for functions where inps have to be in a particular domain
            # print(f"Warning: x not in outputs_by_inp: {x}")
            outputs = self.compute_outputs(x)

        correct = torch.Tensor([o == y for o in outputs])
        prob_noise = self.noise / len(self.possible_outputs)
        prob_y = torch.where(
            correct.bool(),
            torch.ones_like(self.posterior) * ((1 - self.noise) + prob_noise),
            torch.ones_like(self.posterior) * prob_noise,
        )
        new_posterior = prob_y * self.posterior

        # Renormalize
        posterior_sum = new_posterior.sum()

        new_posterior /= posterior_sum

        return (
            self.hypotheses,
            self.hyp_indices,
            self.progs_reps,
            new_posterior,
            new_posterior,
        )

    def predict(self, x, method="sample_pred"):
        if method not in ["sample_prog", "sample_pred", "max_pred"]:
            raise ValueError(method)
        """ Samples a prediction """
        # Sample a program, compute output
        if method == "sample_prog":
            sampled_prog = random.choices(self.all_hypotheses, self.all_posterior)[0]
            # TODO: need to sample incorrect output if noise is not 0
            if self.noise != 0:
                raise NotImplementedError()
            pred = self.interpreter.run_program(sampled_prog, x, strict=False)
        else:
            # TODO: make sure robust
            predict_dist, predict_labels = self.predict_proba(x)
            if method == "sample_pred":
                pred_idx = predict_dist.sample().item()
                pred = predict_labels[pred_idx]
            elif method == "max_pred":
                probs = predict_dist.probs
                pred_idx = np.argmax(probs)
                pred = predict_labels[pred_idx]
        return pred

    def predict_list(self, inputs, **kwargs):
        # TODO: allow predict() to handle lists too? (what to do if a single input is a list?)
        preds = []
        for x in inputs:
            preds.append(self.predict(x, **kwargs))
        return preds

    def predict_proba(self, x: list):
        if x in self.predict_cache:
            # used for multiple calls to get probability distribution
            return self.predict_cache[x]

        preds = self.compute_outputs(x)

        weighted_counts = {}

        for true_pred, prob_hyp in zip(preds, self.posterior.clone()):
            prob_pred = prob_hyp
            if true_pred in weighted_counts:
                weighted_counts[true_pred] += prob_pred
            else:
                weighted_counts[true_pred] = prob_pred

        # not adding noise; TODO: add prediction noise to actual student?
        ordered_preds = sorted(weighted_counts.keys(), key=lambda x: str(x) or "")
        ordered_counts = [weighted_counts[x] for x in ordered_preds]
        pred_distrib = torch.distributions.Categorical(torch.Tensor(ordered_counts))

        self.predict_cache[x] = (pred_distrib, ordered_preds)

        return pred_distrib, ordered_preds

    # TODO: if only one hyp, make sure prob is 1.0
    def update_posterior(self, x: int, y: Union[int, None]):
        torch.set_default_dtype(self.torch_dtype)

        num_hyps_before = len(self.hypotheses)
        (
            new_hypotheses,
            new_hyp_indices,
            new_progs_reps,
            new_posterior,
            new_all_posterior,
        ) = self.compute_posterior(x, y)
        assert (num_hyps_before) == len(self.hypotheses)
        self.hypotheses = new_hypotheses
        self.hyp_indices = new_hyp_indices
        self.progs_reps = new_progs_reps
        self.posterior = new_posterior
        self.all_posterior = new_all_posterior

        if not torch.isclose(
            self.posterior.sum(), torch.tensor([1.0], dtype=self.torch_dtype)
        ):
            raise PosteriorError(
                f"self.posterior.sum() should be 1 but is {self.posterior.sum().item()}"
            )
        if not torch.isclose(
            self.all_posterior.sum(), torch.tensor([1.0], dtype=self.torch_dtype)
        ):
            raise PosteriorError(
                f"self.all_posterior.sum() should be 1 but is {self.all_posterior.sum().item()}"
            )

        # Reset predict cache bc probabilities for hypotheses are changed
        self.predict_cache = {}

    def print_top_hyps(self, num_hyps=10):
        print("Top hypotheses:")
        for hyp, prob in self.get_top_hyps(num_hyps):
            print(f"{hyp}: {prob}")

    def get_top_hyps(self, num_hyps=10):
        # sort self.posterior and self.hypotheses by posterior
        sorted_posterior, sorted_indices = torch.sort(self.posterior, descending=True)
        sorted_hypotheses = [self.hypotheses[i] for i in sorted_indices]
        return list(zip(sorted_hypotheses, sorted_posterior))[:num_hyps]
