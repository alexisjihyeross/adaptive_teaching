import pandas as pd


class Dataset:
    def __init__(
        self,
        inputs,
        outputs,
    ):
        self.inputs = inputs
        self.outputs = outputs

        num_unique_inputs = len(set(self.inputs))
        # get a dict version for fast retrieval of labels (i.e. for verbs)
        if num_unique_inputs != len(self.inputs):
            self.dict = None
        else:
            self.dict = {i: o for i, o in zip(self.inputs, self.outputs)}

        self.unique_outputs = self.get_unique_outputs()

    def get_len(self):
        return len(self.inputs)

    # TODO: without sorted(list()) returns in different orders?
    def get_unique_outputs(self):
        return sorted(list(set(self.outputs)), key=lambda x: x or -1)

    def select_indices(self, indices):
        selected_inputs = [inp for idx, inp in enumerate(self.inputs) if idx in indices]
        selected_outputs = [
            out for idx, out in enumerate(self.outputs) if idx in indices
        ]
        self.inputs = selected_inputs
        self.outputs = selected_outputs

    def delete_index(self, index):
        selected_inputs = [inp for idx, inp in enumerate(self.inputs) if idx != index]
        selected_outputs = [out for idx, out in enumerate(self.outputs) if idx != index]
        self.inputs = selected_inputs
        self.outputs = selected_outputs

    def to_dataframe(self):
        return pd.DataFrame({"input": self.inputs, "output": self.outputs})

    def get_label(self, inp):
        assert (
            self.dict is not None
        ), f"Can't get label for input because this dictionary does not have a one-to-X mapping (has repeats of inputs in the dataset)"
        return self.dict[inp]

    def check_input_validity(self, inp):
        raise NotImplementedError

    def check_output_validity(self, out):
        raise NotImplementedError

    def sample(self, random_state, num_samples):
        """Sample num_samples number of examples with random_state"""
        sampled_indices = random_state.sample(range(self.get_len()), num_samples)
        sampled_inputs = [self.inputs[idx] for idx in sampled_indices]
        sampled_outputs = [self.outputs[idx] for idx in sampled_indices]
        return sampled_inputs, sampled_outputs
