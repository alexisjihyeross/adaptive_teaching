TEACHING_PARAMS = {
    "random": {
        "id": "random",  # used as name
        "strategy": "random",
    },
    "ranking_known": {
        "id": "ranking_known",  # used as name
        "strategy": "ranking",
        "assume_known_prior": True,
    },
    "ranking_unknown": {
        "id": "ranking",  # used as name
        "strategy": "ranking",
        "assume_known_prior": False,
    },
    "non-adaptive_known": {
        "id": "non-adaptive_known",  # used as name
        "strategy": "probabilistic",
        "adapt_prior_online": False,
        "assume_known_prior": True,
        "update_student_beliefs": True,
        "loss_type": "mle",
    },
    "non-adaptive": {
        "id": "non-adaptive",  # used as name
        "strategy": "probabilistic",
        "adapt_prior_online": False,
        "assume_known_prior": False,
        "update_student_beliefs": True,
        "loss_type": "mle",
    },
    "atom": {
        "id": "atom",  # used as name
        "strategy": "probabilistic",
        "adapt_prior_online": True,
        "assume_known_prior": False,
        "update_student_beliefs": True,
        "loss_type": "mle",
    },
    "gpt4": {
        "id": "gpt4",  # used as name
        "model_name": "gpt-4-0314",
        "strategy": "gpt",
        "use_gold_output": True,
        "filter_duplicates": False,
        "assume_known_prior": False,
    },
    "gpt4_known": {
        "id": "gpt4_known",  # used as name
        "model_name": "gpt-4-0314",
        "strategy": "gpt",
        "use_gold_output": True,
        "filter_duplicates": False,
        "assume_known_prior": True,
    },
    "gpt4_atom_combo_overwrite": {
        "id": "gpt4_atom_combo_overwrite",  # used as name
        "model_name": "gpt-4-0314",
        "strategy": "gpt+probabilistic",
        "loss_type": "mle",
        "use_gold_output": True,
        "filter_duplicates": False,
        "assume_known_prior": False,
        "adapt_prior_online": True,
        "combine_strategy": "overwrite_prompt",
        "update_student_beliefs": True,
    },
}
