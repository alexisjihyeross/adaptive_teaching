from src.programs.prog_configs import *
from src.programs.exp_configs import TEACHING_PARAMS

EXPERIMENTS = {
    "synthetic": {
        # student_concept_params and strategies are the same for all configs created from experiment
        "student_concept_params": None,  # when student_concept_params is None, created automatically in run_functions.py
        "strategies": [
            TEACHING_PARAMS["random"],
            TEACHING_PARAMS["ranking_known"],
            TEACHING_PARAMS["ranking_unknown"],
            TEACHING_PARAMS["non-adaptive_known"],
            TEACHING_PARAMS["non-adaptive"],
            TEACHING_PARAMS["atom"],
            TEACHING_PARAMS["gpt4"],
            TEACHING_PARAMS["gpt4_known"],
            TEACHING_PARAMS["gpt4_atom_combo_overwrite"],
        ],
        # for all other hyperparameters, configs are created by getting all possible combinations of values
        "seed": [0, 1, 2],
        "num_iterations": [40],
        "env_name": ["function"],
        "student_noise": [0.05],
    },
    "human": {
        "student_concept_params": None,
        "strategies": [
            TEACHING_PARAMS["random"],
            TEACHING_PARAMS["atom"],
            TEACHING_PARAMS["gpt4"],
        ],
        "num_iterations": [40],
        "env_name": ["function"],
        "student_noise": [0.02],
    },
}
