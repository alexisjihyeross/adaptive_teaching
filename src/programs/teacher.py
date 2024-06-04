from src.utils import read_file, print_dict
from src.programs.bps import *
from src.programs.prior import *
from src.teacher import Teacher, GPTTeacher
from src.programs.fractions.lib import FractionProblem


# TODO: set local random seed (like in verbs)
# TODO: change to ProgramTeacher
class ProgramTeacher(Teacher):
    """Base Class"""

    def __init__(
        self,
        dataset,
        student_guess,
        interpreter,
        gold_prog,
        **kwargs,  # TODO hacky: add in case of multiple inheritance
    ):
        self.interpreter = interpreter
        self.gold_prog = gold_prog
        self.seen_inps = set()
        self.observations = []
        self.student_guess = student_guess
        self.progs = self.student_guess.all_hypotheses.copy()
        self.progs_reps = self.student_guess.all_progs_reps.clone()

        # Used to figure out which inputs have been used for update_student_models; used in pilot in case of refresh
        # self.inputs_seen_for_student_models = set()

        super().__init__(dataset, **kwargs)

    def update_student_models(self, inp, pred, out):
        # self.inputs_seen_for_student_models.add(inp)
        pass

    # Scores all examples in the dataset
    def score_all_examples(self):
        scores = {}
        for idx in range(self.dataset.get_len()):
            x = self.dataset.inputs[idx]
            y = self.dataset.outputs[idx]
            score = self.score(x, y)
            assert (x, y) not in scores, f"datapoint ({x}, {y}) already in scores"
            scores[(x, y)] = score
        return scores

    # TODO: redundant with save_interaction in base teacher? --> I think this exists because GPT needs a function for parsing the label?
    def update_predictions(self, inp, pred, out=None):
        if out is None:
            out = self.interpreter.run_program(self.gold_prog, inp)
        self.observations.append({"input": inp, "output": out, "prediction": pred})

    def update_student_guess(self):
        pass

    def get_unique_input(self, x):
        if isinstance(x, list):
            return tuple(x)
        elif isinstance(x, int):
            return x
        elif isinstance(x, FractionProblem):
            return str(x)
        else:
            raise ValueError()

    # TODO: are seen_inps used for anything? standardize to observations?
    def select(self, add_seen_input=True):
        x, y = self.sample()
        if add_seen_input:
            self.add_inp(x)
        return x, y

    def add_inp(self, x):
        unique_x = self.get_unique_input(x)
        self.seen_inps.add(unique_x)

    def sample(self):
        raise NotImplementedError

    def get_num_seen(self):
        return len(self.seen_inps)


class RandomProgramTeacher(ProgramTeacher):
    """Randomly select examples"""

    def __init__(self, *args):
        super().__init__(*args)
        self.remaining_indices = random.sample(
            range(self.dataset.get_len()), self.dataset.get_len()
        )

    def score(self, inp, out):
        if self.get_unique_input(inp) in self.seen_inps:
            return 0
        else:
            # TODO: hacky
            # can't use remaining_indices bc already popped from in selecting; want to use seen_inps (in case seen_inps not added to yet)
            # num_remaining = self.dataset.get_len() - len(self.seen_inps)
            num_remaining = len(self.remaining_indices)
            return 1 / num_remaining

    def sample(self):
        idx = self.remaining_indices.pop()
        x = self.dataset.inputs[idx]
        y = self.dataset.outputs[idx]
        return x, y


# TODO: only allow GPT to generate examples that are up to max_input_length (param in run_bps.py)
class GPTProgramTeacher(GPTTeacher, ProgramTeacher):
    def __init__(
        self,
        gpt_helper,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # TODO: hacky, but EXAMPLE_TO_REPLACE is replaced in get_output function

        self.gpt_helper = gpt_helper

    def parse_input(self, text):
        return self.gpt_helper.parse_input(text)

    def parse_output(self, text):
        return self.gpt_helper.parse_output(text)

    def get_input(self, response, parsed_message):
        """Tries parsing output, and if fails, responds to GPT with canned response.
        Does so in following steps:
        1. Try parsing input; if succeeds, update parsed_message and return
        2. If fails, assume that GPT tried to finish teaching, in which
            case the student can respond *once*: "I would like to keep learning. Can I have another..."
            - Add student's response to self.messages (along with empty? parsed messages)
            - Re-call GPT with student response and add GPT response to self.messages
            - Repeat step 1 (if hits 2 again, leads to error)
        """

        num_parsing_errors = 0
        # If output was parsed but no input, assume that GPT tries to finish teaching
        while True:
            try:
                new_inp, input_pattern = self.parse_input(response)
                parsed_message.update(
                    {"next_inp": (new_inp), "input_pattern": str(input_pattern)}
                )

                input_is_valid = self.dataset.check_input_validity(new_inp)
                # invalid if input is not in range
                if not input_is_valid:
                    response = self.get_student_invalid_ex_response(new_inp)
                    self.generated_invalid_input = True
                    self.messages.append({"role": "user", "content": response})
                    # Append two parsed messages, one for current parsed GPT response
                    # and one for student response (latter should be empty)
                    self.parsed_messages.append(parsed_message)
                    self.parsed_messages.append({})
                    response = self.call()
                    num_parsing_errors += 1
                # break if input is valid; otherwise, keep trying
                else:
                    break
            except ValueError as e:
                # only allow for one parsing error
                if num_parsing_errors > 0:
                    raise ValueError(f"Too many parsing errors: {e}")
                student_response = (
                    "I would like to keep learning. Can I have another example?"
                )
                print(
                    (
                        f"Warning: Didn't find an input in the response. "
                        f"Error: '{e}'. Trying again with student response: '{student_response}'"
                    )
                )
                self.messages.append({"role": "user", "content": student_response})
                # Append two empty, one for current parsed GPT response
                # and one for student response (latter should be empty)
                self.parsed_messages.append(parsed_message)
                self.parsed_messages.append({})
                parsed_message = {}
                num_parsing_errors += 1
                response = self.call()
        if num_parsing_errors == 0:
            self.last_input_re_generated = False
        else:
            self.last_input_re_generated = True
        return response, parsed_message

    # TODO:
    def parse_student_type(self, response):
        return self.gpt_helper.parse_student_type(response)

    def get_formatted_inp_out(self, inp, out):
        return self.gpt_helper.get_formatted_inp_out(inp, out)

    def get_formatted_inp_question(self, inp):
        return self.gpt_helper.get_formatted_inp_question(inp)

    def get_student_no_output_response(self, inp):
        return self.gpt_helper.get_student_no_output_response(inp)

    def get_student_default_response(self, inp):
        return self.gpt_helper.get_student_default_response(inp)

    def get_student_no_learning_response(self, inp, out):
        return self.gpt_helper.get_student_no_learning_response(inp, out)

    def get_student_invalid_ex_response(self, inp):
        return self.gpt_helper.get_student_invalid_ex_response(inp)

    def get_student_invalid_output_response(self, inp):
        return self.gpt_helper.get_student_invalid_output_response(inp)

    def get_student_diff_answer_response(self, inp):
        return self.gpt_helper.get_student_diff_answer_response(inp)

    def get_gold_output(self, inp):
        gold_out = self.interpreter.run_program(self.gold_prog, inp)
        return gold_out


class ScoringProgramTeacher(ProgramTeacher):
    """Base class for teachers that score examples"""

    def __init__(self, dataset, populations, *args):
        super().__init__(dataset, *args)
        self.populations = populations
        # Initialize student_pop_idx based on student_guess (hacky bc that is passed to ProgramTeacher in args?)
        self.student_pop_idx = get_pop_idx(populations, self.student_guess)

    def score(self, inp, out):
        """Helper function for sample to score the remaining indices"""
        _, _, _, _, new_posterior = self.student_guess.compute_posterior(inp, out)
        # only works when no hyps zeroed out and new_posterior always has same length as original posterior
        # TODO: need to get total prob (of all equiv hypotheses?) -- I think there are no equiv hyps, but confirm
        new_gold_prob = new_posterior[
            self.student_guess.get_hyp_idx(self.gold_prog)
        ].item()
        return new_gold_prob


class RankingProgramTeacher(ScoringProgramTeacher):
    """Uses gold labels as labels (not predictions)
    Samples num_samples examples and scores them in the beginning based on the student's prior, then goes through them in order
    """

    def __init__(self, dataset, populations, *args):
        super().__init__(dataset, populations, *args)

        self.initialize_ranking()

    def initialize_ranking(self):
        self.indices = list(range(self.dataset.get_len()))
        self.scored_indices = [
            (
                idx,
                self.dataset.inputs[idx],
                self.dataset.outputs[idx],
                self.score(self.dataset.inputs[idx], self.dataset.outputs[idx]),
            )
            for idx in self.indices
        ]

        # Sort by scores
        self.scored_indices = sorted(
            self.scored_indices, key=lambda x: x[3], reverse=True
        )

        print("Initialized ranking")
        print("Top 10 examples:")
        for idx, inp, out, score in self.scored_indices[:10]:
            print(f"{idx}\t{inp} -> {out}\t{score}")

        # index of next example to return
        self.curr_idx = 0

    def sample(self):
        idx, x, y, score = self.scored_indices[self.curr_idx]
        print(f"Sampling data idx {idx}, ranking idx {self.curr_idx}")
        self.curr_idx += 1
        return x, y


# Only implemented for populations
class ProbabilisticProgramTeacher(ScoringProgramTeacher):
    def __init__(
        self,
        dataset,
        populations,
        *args,
        loss_type="mle",
        adapt_prior_online=True,
        update_student_beliefs=True,  # whether to update student beliefs (ie posterior) based on observed datapoints
        pred_noise=None,
        num_unique_preds=215,  # used when pred_noise is not None
    ):
        super().__init__(dataset, populations, *args)

        self.loss_type = loss_type
        self.losses_by_population = {}
        self.adapt_prior_online = adapt_prior_online
        self.update_student_beliefs = update_student_beliefs

        self.pred_noise = pred_noise
        self.num_unique_preds = num_unique_preds

        # keep track of which programs are correct on the observed inputs/labels
        self.progs_correct_on_seen = None
        # keep track of which programs are correct on the observed predictions from students
        self.progs_correct_on_pred = None

        self.dataset = dataset
        self.remaining_indices = list(range(self.dataset.get_len()))

    def sample(self):
        # TODO: maybe this can be made more efficient

        best_idx = None
        best_prob = -np.inf

        probs = []
        for idx in self.remaining_indices:
            inp = self.dataset.inputs[idx]
            out = self.dataset.outputs[idx]
            new_gold_prob = self.score(inp, out)
            probs.append(new_gold_prob)
            if new_gold_prob > best_prob:
                best_idx = idx
                best_prob = new_gold_prob

        best_x = self.dataset.inputs[best_idx]
        best_y = self.dataset.outputs[best_idx]

        # print("best idx: ", best_idx)
        # for idx, prob in sorted(
        #     zip(self.remaining_indices, probs), key=lambda x: x[1], reverse=True
        # )[:5]:
        #     print(f"{idx}: {prob} ({self.dataset.inputs[idx]})")

        # TODO: make more efficient (with set operation)
        self.remaining_indices = [i for i in self.remaining_indices if i != best_idx]

        return best_x, best_y

    def update_predictions(
        self,
        inp,
        pred,
        out=None,
    ):
        if out is None:
            out = self.interpreter.run_program(self.gold_prog, inp)

        self.observations.append({"input": inp, "output": out, "prediction": pred})

    def update_student_models(self, inp, pred, out):
        if self.adapt_prior_online:
            """Update representations of each population's probability(pred)
            and update their posteriors"""

            for pop_idx, pop in enumerate(self.populations):

                def get_pop_log_prob():
                    # gets log prob of pred under the population; if pred not in population's prediction space, returns 0
                    pred_distrib, ordered_preds = pop.predict_proba(inp)

                    log_probs = torch.log(pred_distrib.probs)

                    try:
                        pred_idx = ordered_preds.index(pred)
                        if self.loss_type == "mle":
                            loss = log_probs[pred_idx].item()

                            # pop.print_top_hyps(5)

                        else:
                            raise NotImplementedError(
                                f"Only implemented for MLE (not {self.loss_type})"
                            )
                    except ValueError as e:
                        print(f"Got exception: {e}")
                        print(f"Got impossible prediction: {pred}")
                        print("Setting the loss to 0...")
                        # TODO: not sure if robust
                        loss = 0

                    return loss

                loss = get_pop_log_prob()

                # TODO: have default pred_noise set to 0 bc equivalent?
                # if self.pred_noise is not None, add probability of a random prediction
                # TODO: hacky, but re-exponentiate and then take log of full thing
                if self.pred_noise is not None:
                    loss = (1 - self.pred_noise) * np.exp(loss) + self.pred_noise * (
                        1 / self.num_unique_preds
                    )
                    loss = np.log(loss)

                if pop_idx in self.losses_by_population:
                    self.losses_by_population[pop_idx] += loss
                else:
                    self.losses_by_population[pop_idx] = loss

                if self.update_student_beliefs:
                    pop.update_posterior(inp, out)
                self.populations[pop_idx] = pop

            # Update student guess (i.e. best pop)
            self.update_student_guess()

        # If not adapting the prior, still need to update the posterior for the student
        # TODO: or do we want to just simulate what the posterior would be??
        else:
            if self.update_student_beliefs:
                self.student_guess.update_posterior(inp, out)

        # self.inputs_seen_for_student_models.add(inp)

    def update_student_guess(self):
        best_pop_idx = self.compute_best_population()
        best_population = self.populations[best_pop_idx]

        # print("New best population:", best_pop_idx)

        self.student_guess = best_population
        self.student_pop_idx = best_pop_idx

    # TODO: this is redundant with src/nb/teacher: create Probabilistic parent class?
    def compute_best_population(self):
        """Get the idx of the population that maximizes log probability  of observed predictions"""

        sorted_populations = [
            (pop_idx, loss)
            for pop_idx, loss in sorted(
                self.losses_by_population.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        # Randomly sample if there are multiple populations tied for best
        # TODO: Make sure robust
        best_populations = [
            pop_idx
            for pop_idx, loss in sorted_populations
            if loss == sorted_populations[0][1]
        ]
        if len(best_populations) > 1:
            print(f"Warning: multiple populations tied for best: {best_populations}")
            selected = random.choice(best_populations)
            print("selected:", selected)
            return selected

        return sorted_populations[0][0]


# class combining GPT and ProbabilisticProgramTeacher
# TODO: hacky, but copying a lot of ProbabilisticProgramTeacher manually
class GPTProbabilisticProgramTeacher(GPTProgramTeacher):
    def __init__(
        self,
        populations,
        *args,
        combine_strategy="overwrite_prompt",
        loss_type="mle",
        adapt_prior_online=True,
        update_student_beliefs=True,  # whether to update student beliefs (ie posterior) based on observed datapoints
        pred_noise=None,  # probability with which (the teacher believes) the student gives a random prediction
        prompts_for_populations=None,
        descriptions_for_populations=None,
        num_unique_preds=215,  # used when pred_noise is not None
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # overwrite_prompt: overwrite the beginning prompt with a new prompt containing the student type inferred by ATOM
        # rejection_sample: sample from GPT, then reject if it doesn't match the type of example inferred by ATOM
        # append the student guess to the context
        if combine_strategy not in [
            "overwrite_prompt",
            "rejection_sample",
            "append_to_context",
        ]:
            raise ValueError(f"Invalid combine_strategy: {combine_strategy}")

        if combine_strategy not in ["overwrite_prompt", "append_to_context"]:
            raise NotImplementedError(f"Only implemented for overwrite_prompt")

        if combine_strategy == "overwrite_prompt":
            assert (
                prompts_for_populations is not None
            ), f"prompts_for_populations cannot be None if combine_strategy=overwrite_prompt"

        if combine_strategy == "append_to_context":
            assert (
                descriptions_for_populations is not None
            ), f"descriptions_for_populations cannot be None if combine_strategy=append_to_context"

        # make sure adapt_prior_online is True
        if not adapt_prior_online or not update_student_beliefs:
            raise ValueError(
                "both adapt_prior_online and update_student_beliefs must be True for GPTProbabilisticProgramTeacher"
            )

        self.combine_strategy = combine_strategy

        self.student_pop_idx = get_pop_idx(populations, self.student_guess)

        self.prompts_for_populations = prompts_for_populations
        self.descriptions_for_populations = descriptions_for_populations
        self.populations = populations

        self.loss_type = loss_type
        self.losses_by_population = {}
        self.adapt_prior_online = adapt_prior_online
        self.update_student_beliefs = update_student_beliefs

        self.pred_noise = pred_noise
        print("pred noise:", pred_noise)
        self.num_unique_preds = num_unique_preds

        if self.combine_strategy == "overwrite_prompt":
            # override the initial base prompt based on randomly sampled student guess
            self.set_base_prompt()

    def set_base_prompt(self):
        """Update the base prompt with the student guess"""
        print("Updating base prompt with guess...")
        # Get new base prompt
        new_prompt = self.prompts_for_populations[self.student_pop_idx]

        # Overwrite prompt with new prompt
        self.messages[0] = {"role": "user", "content": new_prompt}

    def append_to_context(self):
        """Appends to context message that describes the current population guess"""
        student_description = self.descriptions_for_populations[self.student_pop_idx]
        message = f"Note: Based on the student's predictions so far, the student is likely a {student_description}"
        # TODO: should this be a system message? or a user message?
        # TODO: should we change the base prompt to include examples of these messages?
        self.messages.append({"role": "system", "content": message})
        # add a dummy parsed message
        self.parsed_messages.append({})

    # for update_student_models, use ProbabilisticProgramTeacher's method
    def update_student_models(self, inp, pred, out):
        old_student_idx = self.student_pop_idx
        ProbabilisticProgramTeacher.update_student_models(self, inp, pred, out)
        new_student_idx = self.student_pop_idx

        old_prompt = self.messages[0]["content"]

        # if combine_strategy is overwrite_prompt, then set the base prompt
        if self.combine_strategy == "overwrite_prompt":
            self.set_base_prompt()

            if old_student_idx != new_student_idx:
                assert old_prompt != self.messages[0]["content"]

        elif self.combine_strategy == "append_to_context":
            # append the student guess to the context
            self.append_to_context()

    def compute_best_population(self):
        return ProbabilisticProgramTeacher.compute_best_population(self)

    def update_student_guess(self):
        ProbabilisticProgramTeacher.update_student_guess(self)

    def clean_messages(self):
        """Cleans up the messages by removing any messages that are not relevant to the current conversation
        e.g. for append_to_context, remove all messages that start with "Note: Based on the student's predictions so far" -- will be called *after* select() is called (so remove anything not needed for future examples)
        TODO: move to select()?
        """
        if self.combine_strategy == "append_to_context":

            # remove all messages that start with "Note: Based on the student's predictions so far"; remove corresponding parsed messages too
            new_messages, new_parsed_messages = [], []

            # TODO: this is a hacky way of cleaning up the messages
            for message, parsed_message in zip(self.messages, self.parsed_messages):
                if not message["content"].startswith(
                    "Note: Based on the student's predictions so far"
                ):
                    new_messages.append(message)
                    new_parsed_messages.append(parsed_message)

            self.messages = new_messages
            self.parsed_messages = new_parsed_messages
