###### STUDENT PARAMS ######

betas_initialize_params_1 = {
    "unknown_concept_alpha_val": "MAP",
    "known_concept_alpha_val": "MAP",
    "unknown_concept_beta_val": 1,
    "known_concept_beta_val": "MAP",
}

betas_initialize_params_2 = {
    "unknown_concept_alpha_val": 1,
    "known_concept_alpha_val": 1,
    "unknown_concept_beta_val": 1,
    "known_concept_beta_val": "MAP",
}

betas_initialize_params_3 = {
    "unknown_concept_alpha_val": "MAP",
    "known_concept_alpha_val": "MAP",
    "unknown_concept_beta_val": "MAP_inverse",
    "known_concept_beta_val": "MAP",
}

correct_initialize_params = {
    "unknown_concept_alpha_val": "MAP",
    "known_concept_alpha_val": "MAP",
    "unknown_concept_beta_val": "MAP",
    "known_concept_beta_val": "MAP",
}

alphas_initialize_params_1 = {
    "unknown_concept_alpha_val": 1,
    "known_concept_alpha_val": "MAP",  # if None, use params of actual posterior
    "unknown_concept_beta_val": 1,
    "known_concept_beta_val": 1,
}

alphas_initialize_params_2 = {
    "unknown_concept_alpha_val": 1,
    "known_concept_alpha_val": 1000,  # if None, use params of actual posterior
    "unknown_concept_beta_val": 1,
    "known_concept_beta_val": 1,
}

both_initialize_params = {
    "unknown_concept_alpha_val": 1,
    "known_concept_alpha_val": "MAP",  # if None, use params of actual posterior
    "unknown_concept_beta_val": 1,
    "known_concept_beta_val": "MAP",
}

student_params = {
    "student_type": "bnb",
    "unknown_concept": "y_to_ied",
    # 'prior_initialize_params': betas_initialize_params_1,
    "prior_initialize_params": both_initialize_params,
}

STRATEGIES = [
    {
        "strategy": "probabilistic",
        "id": "non-adaptive_known",
        "num_samples": 500,
        "adapt_prior_online": False,
        "assume_known_prior": True,
        "update_student_beliefs": True,
    },
    {
        "strategy": "probabilistic",
        "id": "non-adaptive",
        "num_samples": 500,
        "adapt_prior_online": False,
        "assume_known_prior": False,
        "update_student_beliefs": True,
    },
    {
        "strategy": "probabilistic",
        "id": "atom",
        "num_samples": 500,
        "adapt_prior_online": True,
        "assume_known_prior": False,
        "update_student_beliefs": True,
    },
    {
        "strategy": "ranking",
        "id": "ranking",
        "num_samples": 500,
        "assume_known_prior": False,
    },
    {
        "strategy": "ranking",
        "id": "ranking_known",
        "num_samples": 500,
        "assume_known_prior": True,
    },
    {
        "strategy": "gpt",
        "id": "gpt4",
        "model_name": "gpt-4-0314",
        "use_gold_output": True,
        "assume_known_prior": False,
    },
    {
        "strategy": "gpt",
        "id": "gpt4_known",
        "model_name": "gpt-4-0314",
        "use_gold_output": True,
        "assume_known_prior": True,
    },
]
