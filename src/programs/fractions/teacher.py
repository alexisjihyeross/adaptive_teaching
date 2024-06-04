from src.programs.bps import *
from src.programs.utils import *
from src.programs.prior import *
from src.programs.fractions.gpt_utils import *
from src.programs.teacher import (
    GPTProgramTeacher,
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
        gpt_helper = FractionGPTHelper()
        teacher = GPTProgramTeacher(gpt_helper, dataset, *args, **kwargs)
    return teacher
