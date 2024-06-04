from src.programs.utils import *
from src.dataset import Dataset
from src.programs.fractions.lib import *


# assumes max_denom is the actual last value we want (ie if we want 1/10, max_denom is 10)
def enumerate_fraction_dataset(
    interpreter,
    gold_prog,
    input_type=FractionProblem,
    min_denom=2,
    max_denom=10,
    min_num=1,
    max_num=5,
    ops=["+", "*"],
    filter_criterion=None,
):
    if input_type != FractionProblem:
        raise NotImplementedError(
            "Enumerating dataset only supported for input_type='int' because can't sample lengths"
        )

    # Enumerate all fractions with denominator up to max_denom, numerator up to max_denom
    # Then enumerate all fraction problems that can be made from these fractions with both possible operations ('+' or '*')
    fracs = []
    for denom in range(min_denom, max_denom + 1):
        for numer in range(min_num, max_num + 1):
            fracs.append(Fraction(numer, denom))

    frac_problems = []
    for frac1 in fracs:
        for frac2 in fracs:
            for op in ops:
                if filter_criterion is None:
                    frac_problems.append(FractionProblem(frac1, frac2, op))
                else:
                    if filter_criterion == "common_denoms":
                        if frac1.denom == frac2.denom:
                            frac_problems.append(FractionProblem(frac1, frac2, op))
                    else:
                        raise NotImplementedError(
                            f"filter_criterion: {filter_criterion}"
                        )

    inps, outs = [], []
    for frac_prob in frac_problems:
        y = interpreter.run_program(gold_prog, frac_prob, strict=False)
        inps.append(frac_prob)
        outs.append(y)

    return FractionDataset(min_num, min_denom, max_num, max_denom, inps, outs)


class FractionDataset(Dataset):
    def __init__(self, min_num, min_denom, max_num, max_denom, *args):
        super().__init__(*args)
        self.min_num = min_num
        self.min_denom = min_denom
        self.max_num = max_num
        self.max_denom = max_denom

    def check_input_validity(self, inp):
        assert isinstance(inp, FractionProblem)

        # TODO: for now, not checking whether fractions are in range, since this is used in gpt interactions
        # assert inp.frac1.denom <= self.max_denom
        # assert inp.frac2.denom <= self.max_denom
        return True

    def check_output_validity(self, out):
        return isinstance(out, Fraction)

    def get_unique_outputs(self):
        return list(set(self.outputs))
