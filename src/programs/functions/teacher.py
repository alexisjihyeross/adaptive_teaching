from src.programs.bps import *
from src.programs.utils import *
from src.programs.prior import *
from src.programs.functions.gpt_utils import *
from src.programs.teacher import (
    GPTProgramTeacher,
    GPTProbabilisticProgramTeacher,
    RandomProgramTeacher,
    ProbabilisticProgramTeacher,
    RankingProgramTeacher,
)


def initialize_teacher(strategy, dataset, populations, *args, **kwargs):
    if strategy == "random":
        teacher = RandomProgramTeacher(dataset, *args)
    elif strategy == "probabilistic":
        teacher = ProbabilisticProgramTeacher(dataset, populations, *args, **kwargs)
    elif strategy == "ranking":
        teacher = RankingProgramTeacher(dataset, populations, *args, **kwargs)
    elif strategy == "gpt":
        gpt_helper = FunctionGPTHelper(dataset)
        teacher = GPTProgramTeacher(gpt_helper, dataset, *args, **kwargs)
    elif strategy == "gpt+probabilistic":
        gpt_helper = FunctionGPTHelper(dataset)
        teacher = GPTProbabilisticProgramTeacher(
            populations, gpt_helper, dataset, *args, **kwargs
        )
    else:
        raise ValueError(strategy)

    return teacher
