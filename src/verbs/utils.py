import pdb
import numpy as np
from tqdm import tqdm
import torch.distributions as dist
import torch
import itertools
from scipy.special import gammaln, psi, beta, betaln

from src.verbs.bnb import BayesianNaiveBayes
from src.verbs.dataset import categorize_verb


def sample_bnb(dataset, seed, alphas=None, betas=None):
    """Returns a bnb with alphas/betas.
    If set to None, BayesianNaiveBayes will initialize to torch.ones (TODO: confirm)"""
    lemmas = dataset.inputs
    categories = list(set(dataset.outputs))
    num_categories = len(categories)
    # TODO: do we ever want random sample?
    #    if alphas is None:
    #        alphas = torch.rand(num_categories)
    bnb = BayesianNaiveBayes(alphas, betas, lemmas, categories, seed)
    return bnb


def get_initial_betas(
    teacher, unknown_label_idx, unknown_concept_beta_val, known_concept_beta_val
):
    """Helper function to get the Beta params for student.
    If known_concept_beta_val = 'MAP', use teacher's actual MAP vals for the class. Same for unknown_concept_beta_val.
    If known_concept_beta_val = 'MAP_inverse', swap first/second parameters of teacher's actual MAP vals for the class. Same for unknown_concept_beta_val.
    Otherwise, use specific values.

    Should not have that known_concept_beta_val is a float but unknown_concept_beta_val = MAP.
    """
    n_classes, n_features, _ = teacher.beta_post_parameters.shape

    # first initialize betas with known parameters
    if known_concept_beta_val == "MAP":
        betas = teacher.beta_post_parameters.clone()
        assert torch.equal(teacher.beta_post_parameters, betas)
    elif known_concept_beta_val == "MAP_inverse":
        betas = teacher.beta_post_parameters.clone()
        betas_1 = betas[:, :, 0].clone()
        betas_2 = betas[:, :, 1].clone()
        # swap first and second parameters of all betas (I think this makes the distribution very different?)
        betas[:, :, 0] = betas_2
        betas[:, :, 1] = betas_1
    elif isinstance(known_concept_beta_val, float) or isinstance(
        known_concept_beta_val, int
    ):
        betas = torch.ones(n_classes, n_features, 2) * known_concept_beta_val
    else:
        raise NotImplementedError(known_concept_beta_val)

    # if they are the same, can leave untouched
    if unknown_concept_beta_val == known_concept_beta_val:
        return betas
    elif unknown_concept_beta_val == "MAP":
        if known_concept_beta_val != "MAP":
            raise ValueError(
                "known_concept_beta_val is not MAP but the unknown_concept_beta_val IS; should not be happening"
            )
    elif unknown_concept_beta_val == "MAP_inverse":
        beta_1 = betas[unknown_label_idx, :, 0].clone()
        beta_2 = betas[unknown_label_idx, :, 1].clone()
        betas[unknown_label_idx, :, 0] = beta_2
        betas[unknown_label_idx, :, 1] = beta_1
    elif isinstance(unknown_concept_beta_val, float) or isinstance(
        unknown_concept_beta_val, int
    ):
        betas[unknown_label_idx, :, :] = (
            torch.ones(1, n_features, 2) * unknown_concept_beta_val
        )
    else:
        raise NotImplementedError(unknown_concept_beta_val)
    return betas


def get_initial_alphas(
    teacher, unknown_label_idx, unknown_concept_alpha_val, known_concept_alpha_val
):
    """Helper function to get the alpha params for student. Use alpha_unknown_concept_alpha_val/known_concept_alpha_val.
    If known_concept_alpha_val = 'MAP', use teacher's actual MAP vals for the class. Same for unknown_concept_alpha_val.
    Otherwise, use specific values.

    Should not have that known_concept_alpha_val is a float but unknown_concept_alpha_val = MAP.
    """
    # Get alphas
    if known_concept_alpha_val == "MAP":
        alphas = teacher.alpha_post_parameters.clone()
        if unknown_concept_alpha_val == "MAP":
            return alphas
        else:
            alphas[unknown_label_idx] = unknown_concept_alpha_val
    else:
        alphas = (
            torch.ones(teacher.alpha_post_parameters.shape) * known_concept_alpha_val
        )
        if unknown_concept_alpha_val == "MAP":
            raise ValueError(
                "known_concept_alpha_val is not MAP but the unknown_concept_alpha_val IS; should not be happening"
            )
        else:
            alphas[unknown_label_idx] = unknown_concept_alpha_val
    return alphas


def sample_bnb_with_unknown(
    dataset,
    seed,
    teacher,
    unknown_label_idx,
    prior_initialize_params,
):
    """Helper function to sample a bnb object with a given unknown concept"""

    alphas = get_initial_alphas(
        teacher,
        unknown_label_idx,
        prior_initialize_params["unknown_concept_alpha_val"],
        prior_initialize_params["known_concept_alpha_val"],
    )
    betas = get_initial_betas(
        teacher,
        unknown_label_idx,
        prior_initialize_params["unknown_concept_beta_val"],
        prior_initialize_params["known_concept_beta_val"],
    )
    bnb = sample_bnb(dataset, seed, alphas=alphas, betas=betas)
    return bnb


def kl_divergence(student, teacher):
    """Computes kl divergence between two Naive Bayes learners who have Dirichlet and Beta priors.
    Because of conjugacy, posterior is product of Dirichlet and Beta distributions.
    KL divergence is the sum of KL divs of each component of posterior.
    """

    # KL divergence between Dirichlet distributions
    #    dirichlet1 = dist.Dirichlet(student.alpha_post_parameters)
    #    dirichlet2 = dist.Dirichlet(teacher.alpha_post_parameters)

    #    kl_dirichlet = dist.kl_divergence(dirichlet1, dirichlet2).sum()

    # equivalent to above and about the same efficiency
    kl_dirichlet = kl_div_dirichlet(
        student.alpha_post_parameters, teacher.alpha_post_parameters
    )

    # KL divergence between Beta distributions
    beta1_params = student.beta_post_parameters
    beta2_params = teacher.beta_post_parameters
    #    kl_beta_sum = 0.0
    kl_beta_sum_2 = 0.0
    #    kl_beta_sum_3 = 0.0
    for i, j in itertools.product(
        range(beta1_params.shape[0]), range(beta1_params.shape[1])
    ):
        #        beta1 = dist.Beta(beta1_params[i, j, 0], beta1_params[i, j, 1])
        #        beta2 = dist.Beta(beta2_params[i, j, 0], beta2_params[i, j, 1])
        #        kl_beta_i = dist.kl_divergence(beta1, beta2)
        #        kl_beta_sum_3 += kl_beta_i

        # this is more efficient than the above
        #        kl_beta_sum += kl_div_betas(beta1_params[i, j, 0], beta2_params[i, j, 0], beta1_params[i, j, 1], beta2_params[i, j, 1])
        kl_beta_sum_2 += kl_div_dirichlet(beta1_params[i, j, :], beta2_params[i, j, :])

    # Total KL divergence
    kl_total = kl_dirichlet + kl_beta_sum_2

    if kl_dirichlet < 0:
        print(f"KL div between Dirichlets < 0: {kl_dirichlet}")
        print(
            f"Dirichlet parameters are equal? {torch.equal(student.alpha_post_parameters, teacher.alpha_post_parameters)}"
        )
        print(
            f"Dirichlet parameters are close? {torch.allclose(student.alpha_post_parameters, teacher.alpha_post_parameters)}"
        )
    else:
        print("KL div between Dirichlets is 0")
    if kl_beta_sum_2 < 0:
        print(f"KL div between Betas < 0: {kl_beta_sum_2}")
        print(
            f"Beta parameters are equal? {torch.equal(student.beta_post_parameters, teacher.beta_post_parameters)}"
        )
        print(
            f"Beta parameters are close? {torch.allclose(student.beta_post_parameters, teacher.beta_post_parameters)}"
        )
    else:
        print("KL div between Betas is 0")

    return kl_total.item()


def kl_div_betas(a1, a2, b1, b2):
    """
    Helper function to calculate KL div between two Beta distributions

    a1, b1: parameters of first Beta (F)
    a2, b2: parameters of first Beta (G)

    Based on: https://math.stackexchange.com/questions/257821/kullback-liebler-divergence#comment564291_257821
    """

    # Calculate KL divergence between two Beta distributions
    KL = (
        gammaln(a2)
        + gammaln(b2)
        + gammaln(a1 + b1)
        - gammaln(a2 + b2)
        - gammaln(a1)
        - gammaln(b1)
        + (a1 - a2) * (psi(a1) - psi(a1 + b1))
        + (b1 - b2) * (psi(b1) - psi(a1 + b1))
    )
    return KL


def kl_div_dirichlet(alpha1, alpha2):
    """
    Helper function to calculate KL div between two Dirichlet distributions

    alpha1: parameters of first Dirichlet (P)
    alpha2: parameters of second Dirichlet (Q)

    Based on: https://statproofbook.github.io/P/dir-kl.html
    """

    KL = (
        gammaln(alpha1.sum())
        - gammaln(alpha2.sum())
        + (gammaln(alpha2) - gammaln(alpha1)).sum()
        + ((alpha1 - alpha2) * (psi(alpha1) - psi(alpha1.sum()))).sum()
    )
    return KL
