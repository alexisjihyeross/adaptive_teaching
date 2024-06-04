from src.programs.bps import get_pop_idx
from src.eval import compute_accuracy
import wandb


def get_prob_results(
    bps,
    target_prog,
):
    """Helper function to get the probability of the gold program
    (uses probability of all canonicalized hypotheses)."""

    # TODO: Canonicalizing shouldn't do anything because code assumes that there is only one gold program
    hyp_prob = bps.get_canonical_hyp_prob(target_prog)
    results = {"student_learning/gold_total_prob": hyp_prob}

    return results


def log_results(
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
    map_inputs_to_str=False,
    special_datasets={},
):
    results = {
        "iter": i,
        "data/iter": i,
        "efficiency/iter": i,
        "data/x": x if not map_inputs_to_str else str(x),
        # evaluate if x satisfied fx
        "data/y": y if not map_inputs_to_str else str(y),
        "data/gold_y": gold_y if not map_inputs_to_str else str(gold_y),
        "data/y==gold?": int(y == gold_y),
        "preds/pred": pred if not map_inputs_to_str else str(pred),
        "preds/iter": i,
        "preds/pred_is_correct": int(pred == y),
    }

    for prog, name in progs_to_eval_on_inputs:
        results.update(
            {
                f"data/{name}": int(student.interpreter.run_program(prog, x)),
            }
        )

    results.update(
        {
            "num_hyp_nonzero": student.get_num_nonzero_hyps(),  # before seeing (x, y)
            "num_hyp_nonzero_nongold": len(
                [hyp for hyp in student.hypotheses if hyp != target_prog]
            ),  # before seeing (x, y)
        }
    )

    if teaching_params["strategy"] in ["probabilistic", "gpt+probabilistic"]:
        results.update(eval_probabilistic_student_model(student, teacher, i))

    prob_results = get_prob_results(
        student,
        target_prog,
    )

    results.update(prob_results)

    gold_total_prob = prob_results["student_learning/gold_total_prob"]

    preds_max = student.predict_list(dataset.inputs, method="max_pred")
    preds_sample = student.predict_list(dataset.inputs, method="sample_pred")
    acc_max = compute_accuracy(dataset.outputs, preds_max)
    acc_sample = compute_accuracy(dataset.outputs, preds_sample)

    results.update(
        {
            "student_learning/iter": i,
            "student_learning/acc_on_dataset_MAX": acc_max,
            "student_learning/acc_on_dataset_SAMPLE": acc_sample,
        }
    )

    for dataset_name, special_dataset in special_datasets.items():
        temp_preds_max = student.predict_list(special_dataset.inputs, method="max_pred")
        temp_preds_sample = student.predict_list(
            special_dataset.inputs, method="sample_pred"
        )
        temp_acc_max = compute_accuracy(special_dataset.outputs, temp_preds_max)
        temp_acc_sample = compute_accuracy(special_dataset.outputs, temp_preds_sample)
        results.update(
            {
                f"subset_accuracies_SAMPLE/acc_on_{dataset_name}": temp_acc_sample,
                f"subset_accuracies_MAX/acc_on_{dataset_name}": temp_acc_max,
            }
        )
    results.update(
        {"subset_accuracies_SAMPLE/iter": i, "subset_accuracies_MAX/iter": i}
    )

    print(f"iter: {i}")
    print(f" | {x} -> {y}")
    print(f" | gold prob total: {gold_total_prob}")

    global_results["gold_total_probs"].append(gold_total_prob)
    global_results["dataset_accs_max"].append(acc_max)
    global_results["dataset_accs_sample"].append(acc_sample)

    # Log
    wandb.log(results)

    return results, global_results


def eval_probabilistic_student_model(student, teacher, i):
    """
    Helper function to evaluate a probabilistic teacher's model of the student.
    """

    # Used to sanity check
    chosen_pop_idx = get_pop_idx(teacher.populations, teacher.student_guess)

    # student_pop_idx only is an attribute for probabilistic teacher
    assert (
        chosen_pop_idx == teacher.student_pop_idx
    ), f"these two pop indices should be the same, but {chosen_pop_idx} != {teacher.student_pop_idx}"

    true_student_idx = get_pop_idx(teacher.populations, student)
    assert (
        true_student_idx == 0
    ), f"Assuming student is always first population, but {true_student_idx} != 0"
    results = {
        "student_model/iter": i,
        "student_model/pop_idx_correct": int(chosen_pop_idx == true_student_idx),
    }
    return results
