import re
import random
from tqdm import tqdm

from src.teacher import *
from src.verbs.dataset import get_verb_category, VerbCategoryError
from src.verbs.gpt_utils import *


def initialize_teacher(strategy, dataset, *args, **kwargs):
    if strategy == "random":
        ds = RandomVerbsTeacher(dataset, *args)
    elif strategy == "ranking":
        ds = ApproximateRankingVerbsTeacher(dataset, *args, **kwargs)
    elif strategy == "probabilistic":
        ds = ApproximateProbabilisticVerbsTeacher(dataset, *args, **kwargs)
    elif strategy == "gpt":
        gpt_helper = VerbsGPTHelper(dataset)
        ds = GPTVerbsTeacher(gpt_helper, dataset, *args, **kwargs)
    else:
        raise ValueError(f"Unrecognized strategy: {strategy}")
    return ds


class VerbsTeacher(Teacher):
    def __init__(self, dataset, seed=42):
        self.seen_inps = set()
        self.observations = []
        self.random = random.Random(seed)

        super().__init__(
            dataset,
        )

    def select(self):
        x, y = self.sample()
        self.seen_inps.add(x)
        return x, y

    def update_predictions(self, inp, pred, out):
        self.observations.append({"input": inp, "output": out, "prediction": pred})


class RandomVerbsTeacher(VerbsTeacher):
    """Randomly select examples"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset_len = self.dataset.get_len()
        self.shuffled_indices = self.random.sample(range(dataset_len), dataset_len)

    def sample(self):
        idx = self.shuffled_indices.pop()
        x = self.dataset.inputs[idx]
        y = self.dataset.outputs[idx]
        return x, y


class ScoringVerbsTeacher(VerbsTeacher):
    """Parent class for teachers that need to score examples based on the student's current belief distribution"""

    def __init__(
        self,
        dataset,
        teacher_nb,
        student_pop_idx,
        populations,
        num_samples=100,
        **kwargs,
    ):
        super().__init__(dataset, **kwargs)
        self.num_samples = num_samples

        self.teacher_nb = teacher_nb
        self.populations = populations

        self.student_pop_idx = student_pop_idx
        self.student_guess = populations[student_pop_idx]

    def score(self, inp, out):
        alpha_post_parameters, beta_post_parameters = (
            self.student_guess.transform_and_simulate_fit([inp], [out])
        )
        teacher_pi, teacher_phi = self.teacher_nb.get_posterior_mean()
        log_prob = self.student_guess.get_posterior_log_prob(
            teacher_pi,
            teacher_phi,
            alpha_post_parameters=alpha_post_parameters,
            beta_post_parameters=beta_post_parameters,
        )

        return log_prob.item()


class ApproximateRankingVerbsTeacher(ScoringVerbsTeacher):
    """
    Uses gold labels as labels (not predictions)
    Samples num_samples examples and scores them in the beginning based on the student's prior, then goes through them in order
    """

    def __init__(
        self,
        dataset,
        teacher_nb,
        student_pop_idx,
        populations,
        **kwargs,
    ):
        super().__init__(dataset, teacher_nb, student_pop_idx, populations, **kwargs)

        self.initialize_ranking()

    def initialize_ranking(self):
        """Sample indices to use for the entire run"""
        self.sampled_remaining_indices = self.random.sample(
            range(self.dataset.get_len()), self.num_samples
        )
        # Scoring remaining indices that were sampled upfront, instead of resampling
        self.scored_indices = [
            (
                idx,
                self.dataset.inputs[idx],
                self.dataset.outputs[idx],
                self.score(self.dataset.inputs[idx], self.dataset.outputs[idx]),
            )
            for idx in tqdm(
                self.sampled_remaining_indices,
                total=len(self.sampled_remaining_indices),
            )
        ]
        # Sort by scores
        self.scored_indices = sorted(
            self.scored_indices, key=lambda tup: tup[-1], reverse=True
        )

        # index of next example to return
        self.curr_idx = 0

    def sample(self):
        idx, x, y, score = self.scored_indices[self.curr_idx]
        print(f"Sampling data idx {idx}, ranking idx {self.curr_idx}")
        self.curr_idx += 1
        return x, y


class ApproximateProbabilisticVerbsTeacher(ScoringVerbsTeacher):
    """Uses gold labels as labels (not predictions)"""

    def __init__(
        self,
        dataset,
        teacher_nb,
        student_pop_idx,
        populations,
        adapt_prior_online=False,
        update_student_beliefs=True,  # whether to update student beliefs (ie posterior) based on observed datapoints
        **kwargs,
    ):
        super().__init__(dataset, teacher_nb, student_pop_idx, populations, **kwargs)
        self.losses_by_population = {}

        self.adapt_prior_online = adapt_prior_online
        self.update_student_beliefs = update_student_beliefs

        self.remaining_indices = set(range(self.dataset.get_len()))

        # Sample indices to use for the entire run
        self.sampled_remaining_indices = self.random.sample(
            self.remaining_indices, self.num_samples
        )

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
        return sorted_populations[0][0]

    def update_student_guess(self):
        """Set student_guess to be best guess"""
        best_pop_idx = self.compute_best_population()
        best_population = self.populations[best_pop_idx]

        # Set student_nb to be the best guess
        self.student_guess = best_population
        self.student_pop_idx = best_pop_idx

    def update_predictions(self, inp, pred, out):
        """Update representations of each population's probability(pred)
        and update their posteriors"""

        if self.adapt_prior_online:

            pred_is_valid = self.dataset.check_output_validity(pred)

            if pred_is_valid:
                # Iterate through populations and compute probability of inp, then update posterior
                for idx, pop in enumerate(self.populations):
                    # compute prob before updating on this inp
                    probs = pop.predict_proba([inp], return_log=True)[0]

                    # Get predicted probability of prediction
                    label_idx = pop.get_label_idx(pred)
                    assert label_idx == self.teacher_nb.get_label_idx(pred)
                    loss = probs[label_idx]

                    if idx in self.losses_by_population:
                        self.losses_by_population[idx] += loss
                    else:
                        self.losses_by_population[idx] = loss

                    if self.update_student_beliefs:
                        # update posterior on this input/output
                        pop.transform_and_fit([inp], [out])

                self.update_student_guess()

            else:

                print(
                    f"Not updating representations of student populations because the actual prediction is invalid: {pred}"
                )

        # If not adapting the prior, still need to update the posterior for the student
        # TODO: or do we want to just simulate what the posterior would be??
        else:
            if self.update_student_beliefs:
                self.student_guess.transform_and_fit([inp], [out])

        self.observations.append({"input": inp, "output": out, "prediction": pred})

    def sample(self, verbose=False):
        # Scoring remaining indices that were sampled upfront, instead of resampling

        sampled_indices = self.sampled_remaining_indices
        scored_indices = [
            (
                idx,
                self.dataset.inputs[idx],
                self.dataset.outputs[idx],
                self.score(self.dataset.inputs[idx], self.dataset.outputs[idx]),
            )
            for idx in tqdm(sampled_indices, total=len(sampled_indices))
        ]
        # TODO: sort zipped by scores
        sorted_scored_indices = sorted(
            scored_indices, key=lambda tup: tup[-1], reverse=True
        )
        if verbose:
            print(sorted_scored_indices)
            print("top 5:")
            for idx, inp, out, score in sorted_scored_indices[:5]:
                print(f"{idx}\t{inp} -> {out}\t{score}")
            print("bottom 5:")
            for idx, inp, out, score in sorted_scored_indices[-5:]:
                print(f"{idx}\t{inp} -> {out}\t{score}")

        best_idx = sorted_scored_indices[0][0]
        inp = self.dataset.inputs[best_idx]
        out = self.dataset.outputs[best_idx]

        # self.remaining_indices now not used? TODO: confirm
        self.remaining_indices.remove(best_idx)
        self.sampled_remaining_indices.remove(best_idx)
        return inp, out


class GPTVerbsTeacher(GPTTeacher, VerbsTeacher):
    def __init__(
        self,
        gpt_helper,
        *args,
        **kwargs,
    ):
        self.gpt_helper = gpt_helper
        super().__init__(*args, **kwargs)
        # TODO: pass as argument?
        self.categories = ["+ed", "+d", "y_to_ied", "+consonant+ed"]

    def parse_input(self, text):
        """
        Parses inputs. Matches patterns:
        - What type of verb is 'LEMMA'?
        """

        for pattern in [
            r"What type of verb is '([A-Za-z]+)'?",
        ]:
            match = re.search(pattern, text)
            if match:
                lemma = match.group(1)

                return lemma, pattern

        raise ValueError(f"Found unmatched input pattern: '{text}'")

    def parse_output(self, text):
        """
        Parses outputs. Matches patterns:
        - 'LEMMA' is [a/an] 'CATEGORY' verb
        """

        # call function in utils (bc used by other things)
        lemma, category, pattern = parse_output(text)
        if category not in self.categories:
            print(f"Warning: unrecognized category ({category})")
        return lemma, category, pattern

    def parse_student_type(self, response):
        found_cats = []
        for cat in self.categories:
            if f"'{cat}'" in response:
                found_cats.append(cat)

        if len(found_cats) == 0:
            print(f"Warning: No categories found in response: {response}")
            assert False
        elif len(found_cats) > 1:
            print(f"Warning: Multiple categories found in response: {response}")
            assert False

        return found_cats[0]

    def get_student_no_output_response(self, inp):
        # The response that the student should give when no output is found in GPT's generation
        response = f"Sorry, I didn't understand. I can only learn from examples, and they need to be formatted as: 'LEMMA' is a 'CATEGORY' verb. Can you please tell me the category for the previous lemma, '{inp}', and give me a new lemma to give my guess for?"
        return response

    def get_formatted_inp_out(self, inp, out):
        # calls GPT util
        return self.gpt_helper.get_formatted_inp_out(inp, out)

    def get_formatted_inp_question(self, inp):
        # calls GPT util
        return self.gpt_helper.get_formatted_inp_question(inp)

    def get_student_default_response(self, inp):
        response = (
            f"Sorry, I still didn't understand. "
            "Can you give me the category for '{inp}'"
            " and give me a new lemma?"
        )
        return response

    def get_student_invalid_output_response(self, inp):
        raise NotImplementedError

    def get_student_no_learning_response(self, inp, out):
        """Get response for when teacher gives an invalid example"""

        response = (
            "I'm sorry, I couldn't learn from the "
            "last example that you gave, "
            f"'{self.get_formatted_inp_out(inp, out)}'. "
            "I can only learn from verbs that are in the four categories "
            "('+ed', '+d', 'y_to_ied', and '+consonant+ed'). "
            "I will ignore that example and give my prediction for the new example. "
        )
        return response

    def get_student_invalid_ex_response(self, inp):

        # This assumes can "peek ahead" and get labels for inputs, which we can only do if using gold outputs (otherwise would need to get the label from GPT's next message)
        if not self.use_gold_output:
            raise NotImplementedError()

        formatted_inp = self.gpt_helper.get_formatted_inp(inp)

        response = (
            f"Sorry, I can't learn from that example, {formatted_inp}, "
            f"because it is not one of the valid categories. "
            "I can only learn from verbs that are in the four categories "
            "('+ed', '+d', 'y_to_ied', and '+consonant+ed'). "
            f"Can you give me a new example?"
        )
        return response

    def get_student_diff_answer_response(self, inp):
        # The response that the student should give when GPT gives an answer for the wrong input
        student_response = (
            f"You gave me the category for '{parsed_inp}', "
            f"but I expected the category for '{inp}'. "
            f"Can you give me the category for '{inp}' "
            f"and then give me a new lemma?"
        )
        return student_response

    def get_gold_output(self, inp):
        cat = get_verb_category(inp, self.dataset)
        if cat not in self.categories:
            print(f"Warning: unrecognized category ({cat}) for verb {inp}")
        return cat

    def get_input(self, response, parsed_message):
        """Tries parsing input, and if fails, responds to GPT with canned response.
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
                    {"next_inp": new_inp, "input_pattern": input_pattern}
                )

                # if using gold output, can "peek ahead" and get labels for inputs
                if self.use_gold_output:
                    try:
                        gold_out = self.get_gold_output(new_inp)
                        input_is_valid = self.dataset.check_output_validity(gold_out)
                    # if error getting label, treating as an input with an invalid output too
                    except VerbCategoryError:
                        input_is_valid = False
                    # invalid if input is not in range
                    if not input_is_valid:
                        self.generated_invalid_input = True
                        response = self.get_student_invalid_ex_response(new_inp)
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
                # if not using gold output, always break
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
