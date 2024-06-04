from scipy.stats import entropy
import pdb
import torch

from src.programs.interpreter import *
from src.programs.utils import *
from src.programs.bps import *


def compute_best_population(
    populations,
    student,
    loss_type,
    interp,
    progs_correct_on_warmup_preds=None,
    pred=None,
    progs_correct_on_pred=None,
    progs_correct_on_seen=None,
):
    progs = student.all_hypotheses
    losses_by_population = [None] * len(populations)

    print("computing best population...")

    for idx, pop in enumerate(populations):
        pop_estimate = pop.probs
        prior_over_progs = student.get_prior_over_progs(pop_estimate)
        (
            loss,
            _,
        ) = get_loss(
            progs_correct_on_seen,
            progs_correct_on_pred,
            pred,
            loss_type,
            prior_over_progs,
            interp,
            progs,
            student,
            progs_correct_on_warmup_preds=progs_correct_on_warmup_preds,
        )
        losses_by_population[idx] = loss
        print("pop idx: ", idx, "loss: ", loss)
    return np.argmin(losses_by_population)


def eval_prior(prior_estimate, student, compute_progs_prior=False):
    true_prior = student.concepts.probs
    # kl_div doesn't make sense because this isn't a normalized distribution
    #    kl_div = entropy(prior_estimate, qk=true_prior)
    #    normalized_estimate = torch.nn.functional.normalize(prior_estimate, dim=0)
    #    normalized_true_prior = torch.nn.functional.normalize(true_prior, dim=0)
    #    dist_of_normalized = torch.linalg.vector_norm(normalized_estimate - normalized_true_prior)
    dist = torch.linalg.vector_norm(prior_estimate - true_prior)
    result = {
        "prior_estimates": prior_estimate,
        "estimate_dists": dist,
        #            "normalized_estimate_dists": dist_of_normalized,
    }
    if compute_progs_prior:
        prior_progs_estimate = student.get_prior_over_progs(prior_estimate.detach())
        assert torch.isclose(
            prior_progs_estimate.sum(), torch.tensor([1.0])
        ), f"prior_progs_estimate should sum to 1, but sums to {prior_progs_estimate.sum().item()}"
        assert torch.isclose(
            student.prior.sum(), torch.tensor([1.0])
        ), f"student prior should sum to 1, but sums to {student.prior.sum()}"
        kl_div_progs = entropy(prior_progs_estimate, student.prior)
        kl_div_concepts = entropy(prior_estimate, student.concepts.probs)
        result.update(
            {
                "prior_progs_estimates": prior_progs_estimate,
                "kl_divs_progs_prior": kl_div_progs,
                "kl_div_concepts": kl_div_concepts,
            }
        )
    return result


def derive_prior(
    student,
    prior_var,
    optimizer=None,
    lr=1e-3,
    patience=5,
    max_iter=50,
    progs_correct_on_seen=None,
    progs_correct_on_pred=None,
    pred=None,
    loss_type="diff",
    progs_correct_on_warmup_preds=None,
):
    best_loss = torch.inf
    streak = 0
    best_idx = None
    best_estimate = prior_var.clone()

    variables = [prior_var]

    if optimizer is None:
        if lr is None:
            raise ValueError(
                "If optimizer is not supplied, \
                    need lr for initializing optimizer"
            )
        else:
            optimizer = torch.optim.Adam(variables, lr=lr)

    progs = student.all_hypotheses
    interp = student.interpreter

    # TODO: not sure if this is right
    teacher_concepts = ConceptLibrary(
        interp, compositions, unknown_concepts=[], known_concepts=[]
    )
    teacher_concepts.set_concept_probs(probs=teacher_prior_over_concepts_init)
    teacher = BayesianProgramSynthesizer(
        progs,
        interp,
        teacher_concepts,
        progs_reps=student.all_progs_reps,
    )
    teacher.progs_reps = student.progs_reps

    if progs_correct_on_seen is None:
        progs_correct_on_seen = get_progs_correct_on_seen(
            student.seen_data, student.seen_labels, interp, progs
        )

    # TODO: this returns a list of results, whereas for lr it returns a dict of lists; standardize
    results = []
    for idx in tqdm(range(max_iter)):
        prior_var, loss, optimizer, grad, misc_results = prior_step(
            optimizer,
            student,
            teacher,
            prior_var,
            progs_correct_on_seen,
            progs_correct_on_pred=progs_correct_on_pred,
            pred=pred,
            loss_type=loss_type,
            progs_correct_on_warmup_preds=progs_correct_on_warmup_preds,
        )
        #        print(f"\n{idx}: loss={loss}; grad={grad}")
        prior_estimate = prior_var.detach().clone()
        result = eval_prior(prior_estimate, student)
        result.update(
            {"loss": loss, "prior_iter": idx, "grad_norm": torch.linalg.norm(grad)}
        )
        result.update(misc_results)
        results.append(result)
        if loss <= best_loss:
            best_loss = loss
            best_idx = idx
            best_estimate = prior_var
            streak = 0
        else:
            streak += 1

        if streak >= patience:
            print(f"breaking at {idx} (patience={patience})")
            print("best loss:", best_loss)
            print("best idx:", best_idx)
            break

    return results, best_estimate, optimizer


def get_progs_correct_on_seen(seen_data, seen_labels, interp, progs):
    # Helper function to get a (num_seen x num_progs) tensor of whether the progs return the correct outputs on the seen data
    # Store the outputs of all programs on the seen data
    # Gets a list of lists where each sub_list contains outputs for each program
    progs_correct_on_seen = []
    for seen_inp, seen_label in zip(seen_data, seen_labels):
        correct_by_prog = [
            run_program(interp, prog, seen_inp) == seen_label for prog in progs
        ]
        progs_correct_on_seen.append(correct_by_prog)

    progs_correct_on_seen = torch.Tensor(progs_correct_on_seen)
    return progs_correct_on_seen


def get_loss(
    progs_correct_on_seen,
    progs_correct_on_pred,
    pred,
    loss_type,
    prior_over_progs,
    interp,
    progs,
    student,
    progs_correct_on_warmup_preds=None,
):
    # TODO: hacky, but currently getting prior_over_priogs by calling bps function; make separate

    loss = torch.Tensor([0])

    # TODO: why is warmup data being treated as data without preds?
    if progs_correct_on_warmup_preds is not None:
        for pred_correct_vector in progs_correct_on_warmup_preds:
            prob_of_pred = pred_correct_vector.dot(prior_over_progs)
            log_prob_of_pred = prob_of_pred.log()
            if loss_type in ["mle"]:
                loss -= log_prob_of_pred
            else:
                raise NotImplementedError(loss_type)

    # TODO: enumerate through teacher seen inputs/predictions instead of student.seen_data
    for idx, x in enumerate(student.seen_data):

        # do we care about minimizing the prob of programs that were learned to be wrong?
        if progs_correct_on_pred is None:
            assert (
                pred is not None
            ), "Need to supply pred (before updating posterior for this point) if progs_correct_on_pred is None"
            progs_correct_on_curr_pred = torch.Tensor(
                [run_program(interp, prog, x) == pred for prog in progs]
            ).long()
        else:
            progs_correct_on_curr_pred = progs_correct_on_pred[idx, :].long()

        # Get all progs correct on inputs before the current input
        correct_on_seen = progs_correct_on_seen[:idx, :]
        # Each "correct" is a sub list with outputs for all programs
        correct_on_all_seen = correct_on_seen.all(dim=0).long()

        #            int(check_prog_correctness(prog, student.seen_data, student.seen_labels, interp)) for prog in progs])

        #        print("correct on all seen: ", correct_on_all_seen.shape)
        #        print("progs correct on curr pred: ", progs_correct_on_curr_pred.shape)
        #        print("prior over progs: ", prior_over_progs.shape)
        prob_of_pred = (
            correct_on_all_seen * progs_correct_on_curr_pred * prior_over_progs
        )
        sum_prob = prob_of_pred.sum()

        true_prob_of_pred = (
            correct_on_all_seen * progs_correct_on_curr_pred * student.prior
        )

        log_prob_of_pred = sum_prob.log()

        test_loss_term = (prob_of_pred.sum() - true_prob_of_pred.sum()).abs().log()
        #        print(f"{idx}:")
        #        print("correct on all seen:", correct_on_all_seen)
        #        print("progs correct on curr pred:", progs_correct_on_curr_pred)
        #        print("prob_of_pred:", prob_of_pred)
        #        print("normalized prob_of_pred:", prob_of_pred / prob_of_pred.sum())
        #        print("true prob_of_pred:", true_prob_of_pred)
        #        print("normalized true prob_of_pred:", true_prob_of_pred / true_prob_of_pred.sum())
        #        print("posterior:", student.all_posterior)
        #        print("prior_over_progs:", prior_over_progs.round(decimals=3))
        #        print("student prior:", student.prior.round(decimals=3))

        # Negative because want to minimize negative logprob
        if loss_type in ["mle", "map"]:
            loss -= log_prob_of_pred
        elif loss_type == "diff":
            loss += test_loss_term
        #        elif loss_type == "map":
        #            prior_prior = torch.distributions.uniform.Uniform(torch.zeros(prior_var.shape[0]), torch.ones(prior_var.shape[0]))
        #            prior_prob = prior_prior.log_prob(prior_var)
        #            loss -= prior_prob + log_prob_of_pred
        else:
            raise ValueError

    if loss_type == "map":
        prior_prior = torch.distributions.uniform.Uniform(
            torch.zeros(prior_var.shape[0]), torch.ones(prior_var.shape[0])
        )
        prior_prob = prior_prior.log_prob(
            prior_var
        ).sum()  # TODO: sum these? this is saying the probability of this prior is multiplied product of each
        #        print("loss", loss)
        #        print("prior prob: ", prior_prob)
        #        print("MAP")
        loss -= prior_prob
    else:
        prior_prob = None
    return loss, prior_prob


def prior_step(
    optimizer,
    student,
    teacher,
    prior_var,
    progs_correct_on_seen,
    progs_correct_on_pred=None,
    pred=None,
    loss_type="mle",
    progs_correct_on_warmup_preds=None,
):

    teacher.set_prior_over_progs(prior_var)

    prior_over_progs = teacher.prior
    progs = teacher.all_hypotheses
    interp = teacher.interpreter

    # Derive prior on all the seen data
    loss, prior_prob = get_loss(
        progs_correct_on_seen,
        progs_correct_on_pred,
        pred,
        loss_type,
        prior_over_progs,
        interp,
        progs,
        student,
        progs_correct_on_warmup_preds=progs_correct_on_warmup_preds,
    )

    loss.backward()

    grad = prior_var.grad.detach().clone()

    optimizer.step()

    optimizer.zero_grad()
    prior_var.data = torch.clamp(prior_var, min=1e-6)

    # TODO: clamp?

    #    print("grad:", grad)
    #    misc_logging = {"diff_in_preds_norm": diff_in_preds_norm, "p_theta": prior_prob}
    misc_logging = {}
    return prior_var, loss.item(), optimizer, grad, misc_logging
