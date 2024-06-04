from src.programs.utils import *
from src.dataset import Dataset


def enumerate_dataset(interpreter, gold_prog, input_type="int", max_val=10, min_val=0):
    if input_type != "int":
        raise NotImplementedError(
            "Enumerating dataset only supported for input_type='int' because can't sample lengths"
        )
    inps, outs = [], []

    for x in range(min_val, max_val):
        y = interpreter.run_program(gold_prog, x, strict=False)
        inps.append(x)
        outs.append(y)

    assert min_val in inps, f"{min_val} not in {inps}"
    assert max_val - 1 in inps, f"{max_val - 1} not in {inps}"

    return FunctionDataset(min_val, max_val, inps, outs)


class FunctionDataset(Dataset):
    def __init__(self, min_input_val, max_input_val, *args):
        super().__init__(*args)
        self.min_input_val = min_input_val
        self.max_input_val = max_input_val

    def check_input_validity(self, inp):
        return inp in range(self.min_input_val, self.max_input_val)

    def check_output_validity(self, out):
        # can be int or None
        return isinstance(out, int) or out is None
