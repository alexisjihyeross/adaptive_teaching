import re

from src.gpt import GPT


class Teacher:
    """Base Class"""

    def __init__(
        self,
        dataset,
    ):
        self.inputs = []
        self.outputs = []
        # TODO: do we need inputs and outputs if have dataset?
        self.dataset = dataset
        self.student_predictions = []

        self.observations = []

        super().__init__()

    def select(self):
        raise NotImplementedError

    def update_prior(self):
        raise NotImplementedError

    def store_student_response(self, update):
        """
        Keep track of whether the example was
        ignored or updated on by the student.
        Modifies last observation in self.observations.

        Used by GPT update_predictions() to figure out what student should say to the teacher.
        """

        if not update:
            print("Student didn't learn from last example...")
        assert (
            "student_updated_bool" not in self.observations[-1]
        ), "Shouldn't update student response twice, last observation already has student_updated_bool: {}".format(
            self.observations[-1]
        )
        self.observations[-1].update({"student_updated_bool": update})
        assert "student_updated_bool" in self.observations[-1]

    def get_last_student_response(self):
        """Get the last student response (bool of whether the student
        updated on the example or not).
        If no observations yet (i.e. first interaction), default to True
        TODO: not sure if we want this default"""
        if len(self.observations) == 0:
            return True

        assert "student_updated_bool" in self.observations[-1]
        return self.observations[-1]["student_updated_bool"]


class GPTTeacher(GPT, Teacher):
    def __init__(
        self,
        *args,
        use_gold_output=False,
        filter_duplicates=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_gold_output = use_gold_output

        # set to be True if input is found to be invalid in get_input(); used for logging purposes
        self.generated_invalid_input = False

        # whether the generation for the last input was generated twice; if so, don't add "That's correct/incorrect"
        self.last_input_re_generated = False

        # set to be true if student_updated_on_last is False in update_predictions(); used for logging purposes
        self.generated_student_no_learning_response = False

        # whether to filter duplicates from GPT-4 generated datapoints
        self.filter_duplicates = filter_duplicates

        # TODO: not robust if inputs are supposed to be None
        # here assumine normal inputs are not None
        self.next_inp = None

        self.prompt = self.base_prompt

        self.last_output = None

        assert self.end_prompt is not None

    def parse_input(self, text):
        raise NotImplementedError()

    def parse_output(self, text):
        raise NotImplementedError()

    def get_input(self, response, parsed_message):
        raise NotImplementedError()

    def parse_student_type(self, response):
        raise NotImplementedError()

    def parse_output_correctness(self, text):
        # for checking whether 'correct/Correct' but NOT 'incorrect/Incorrect/wrong' are in the text
        for pattern in [
            r"^(?=.*?[cC]orrect)(?!.*?(?:\b(?:incorrect|Incorrect|wrong|Wrong|not)\b))",
            r"^(?=.*?[gG]reat)(?!.*?(?:\b(?:incorrect|Incorrect|wrong|Wrong|not)\b))",
            r"^(?=.*?[eE]xcellent)(?!.*?(?:\b(?:incorrect|Incorrect|wrong|Wrong|not)\b))",
            r"^(?=.*?[aA]mazing)(?!.*?(?:\b(?:incorrect|Incorrect|wrong|Wrong|not)\b))",
            r"^(?=.*?[aA]wesome)(?!.*?(?:\b(?:incorrect|Incorrect|wrong|Wrong|not)\b))",
            r"^(?=.*?[rR]ight)(?!.*?(?:\b(?:incorrect|Incorrect|wrong|Wrong|not)\b))",
        ]:
            if re.match(pattern, text):
                return True, pattern

        for pattern in ["(?i)(wrong|incorrect)"]:
            if re.match(pattern, text):
                return False, pattern

        # TODO: hacky, but return None, None if can't parse
        return None, None

    def sample_input(self):
        # i.e. have seen inputs before (in which case should have saved the input)
        if self.next_inp is not None:
            assert len(self.observations) > 0
            x = self.next_inp
            self.next_inp = None
            return x

        # i.e. this is the first input
        message = self.call()
        try:
            x, input_pattern = self.parse_input(message)
            # breakpoint()
            self.parsed_messages.append(
                {"next_inp": str(x), "input_pattern": str(input_pattern)}
            )
        #            assert isinstance(x, int)
        except ValueError as e:
            print(e)
            print("GPT generated output that can't be parsed. updating prompt")
            print("tried to parse: ", message)
            #                self.messages.append({"role": "user", "content": "Please give a properly formatted input and output."})
            #                continue
            assert False
        return x

    def select(self):
        if not self.filter_duplicates:
            x = self.sample_input()
            return x

        # filter duplicates
        while True:
            x = self.sample_input()
            unique_x = self.get_unique_input(x)
            if unique_x not in self.seen_inps:
                self.seen_inps.add(x)
                break
            else:
                print("resampling because duplicate...")
        self.last_output = f"Input: {x}; Output: {y}"
        return x, y

    def get_student_type(self):
        self.messages.append(
            {
                "role": "user",
                "content": "I would like to stop learning now. " + self.end_prompt,
            }
        )
        response = self.call()
        self.parsed_messages.append(
            {}
        )  # TODO: hacky, append for the user question + system response
        self.parsed_messages.append(
            {}
        )  # TODO: hacky, append for the user question + system response
        # self.print_history()
        parsed_response = self.parse_student_type(response)
        self.parsed_messages.append({"student_type": parsed_response})
        self.print_history()

        self.student_type = parsed_response

    def get_output(self, inp, response, pred):
        """Tries parsing output, and if fails, responds to GPT with canned response.
        Does so in following steps:
        1. Try parsing output; if succeeds, update parsed_message and return
        2. If fails, try parsing to see if message says the prediction is correct, in which
            case the output can be inferred. If so, update parsed_message and return
        3. Otherwise (couldn't parse output number AND message didn't say pred was correct),
            allow student to respond *once* with "Sorry I didn't understand. I can only learn from examples..."
            - Add student's response to self.messages (along with empty? parsed messages)
            - Re-call GPT with student response and add GPT response to self.messages
            - Repeat steps 1-2 (if hits 3 again, leads to error)
        """

        num_parsing_errors = 0

        # TODO: Create function for parsing

        parsed_message = {}
        while True:
            try:
                parsed_inp, out, output_pattern = self.parse_output(response)
                parsed_message.update(
                    {
                        "inp": str(parsed_inp),
                        "label": str(out),
                        "output_pattern": str(output_pattern),
                    }
                )
                break
            except ValueError as e:
                is_correct, correct_pattern = self.parse_output_correctness(response)
                # TODO: is this robust?
                # If response just says prediction was correct, infer the parsed input/out
                # If not correct, need to have been able to parse output to get the answer
                if is_correct:
                    parsed_inp, out = inp, pred
                    parsed_message.update(
                        {
                            "inp": str(parsed_inp),
                            "label": str(out),
                            "correct_pattern": str(correct_pattern),
                        }
                    )
                    break

                # else, allow for one parsing error with canned student response
                if num_parsing_errors > 0:
                    raise ValueError(e)
                print(e)
                student_response = self.get_student_no_output_response(inp)
                print(
                    f"Warning: Didn't find an output in the response. Error: '{e}'. Trying again with student response: '{student_response}'"
                )
                self.messages.append({"role": "user", "content": student_response})

                # if got here, nothing to parse, so append empty for student response AND current parsed_message
                # and reset parsed_message
                # TODO: isn't parsed_message here always empty?
                self.parsed_messages.append(parsed_message)
                self.parsed_messages.append({})  # TODO: hacky
                parsed_message = {}
                num_parsing_errors += 1
                response = self.call()
        return response, parsed_message

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
                    {"next_inp": str(new_inp), "input_pattern": str(input_pattern)}
                )
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
        return response, parsed_message

    def update_predictions(self, inp, pred):
        """Gives prediction to GPT Model and parses response to get label for that input + next input"""

        pred_msg = ""

        # If student didn't learn from previous example, tell GPT
        student_updated_on_last = self.get_last_student_response()
        # TODO: if student doesn't update on last, treated as "example" in main script evals, which maybe we don't want
        if not student_updated_on_last:
            last_inp = self.observations[-1]["input"]
            last_out = self.observations[-1]["output"]
            pred_msg += self.get_student_no_learning_response(last_inp, last_out)
            self.generated_student_no_learning_response = True
        pred_msg += self.get_formatted_inp_out(inp, pred)
        self.messages.append({"role": "user", "content": pred_msg})
        self.parsed_messages.append({})  # TODO: hacky

        # Get response to pred. Should include gold answer *and* next input
        response = self.call()

        gold_out = self.get_gold_output(inp)
        if self.use_gold_output:
            """Only parse the response for the input.
            Then construct GPT response to be 'That's (in)correct. wug()...'
            Overwrite last message (GPT-generated) in self.messages
            Append parsed message
            """
            out = gold_out
            parsed_message = {}
            response, parsed_message = self.get_input(response, parsed_message)
            new_inp = parsed_message["next_inp"]
            if self.last_input_re_generated:
                # if the input was re-generated, then assumes previous message already gave label for previous input
                gpt_message = f"{self.get_formatted_inp_question(new_inp)}"
            elif out == pred:
                gpt_message = (
                    f"That's correct. {self.get_formatted_inp_out(inp, pred)}. "
                    f"{self.get_formatted_inp_question(new_inp)}"
                )
            else:
                gpt_message = (
                    f"That's incorrect. {self.get_formatted_inp_out(inp, out)}. "
                    f"{self.get_formatted_inp_question(new_inp)}"
                )
            # TODO: hacky, but currently overwriting actual gpt generated message
            orig_gpt_message = self.messages[-1]["content"]
            self.messages[-1] = {"role": "assistant", "content": gpt_message}
            self.parsed_messages.append(
                {
                    "next_inp": str(new_inp),
                    "label": str(out),
                    "original_gpt_message": orig_gpt_message,
                }
            )
            self.next_inp = new_inp
        else:
            """
            First try parsing the output; if doesn't work, say couldn't understand ask for answer.
            Then, try parsing input; if doesn't work, say would like to keep learning.
            """
            response, parsed_message = self.get_output(inp, response, pred)
            out = parsed_message["label"]
            parsed_inp = parsed_message["inp"]
            response, parsed_message = self.get_input(response, parsed_message)
            new_inp = parsed_message["next_inp"]

            # this parsed_message should have the output and the next inp
            self.parsed_messages.append(parsed_message)

            # If these are equal, it's probably a parsing issue bc new input should not be the same as the old one
            # TODO: For now, just give canned response and regenerate
            # TODO: why doesn't this exist for gold outputs?
            if new_inp == parsed_inp:
                print(
                    (
                        f"Warning: The new input given by ChatGPT {new_inp} is "
                        "the same as what was parsed for the "
                        "old one {parsed_inp}; response: {response}"
                    )
                )
                student_response = self.get_student_default_response(inp)
                self.messages.append({"role": "user", "content": student_response})
                self.parsed_messages.append({})
                response = self.call()

                parsed_message = {}
                parsed_inp, out, output_pattern = self.parse_output(response)
                parsed_message.update(
                    {"inp": parsed_inp, "label": out, "output_pattern": output_pattern}
                )
                new_inp, input_pattern = self.parse_input(response)
                parsed_message.update(
                    {"next_inp": new_inp, "input_pattern": input_pattern}
                )
                assert (
                    new_inp != parsed_inp
                ), f"new_inp: {new_inp}; parsed_inp: {parsed_inp}; response: '{response}'"

                self.parsed_messages.append(parsed_message)

            if out != gold_out:
                print(
                    f"Warning: The answer given by ChatGPT {out} != gold answer {gold_out}"
                )

            # TODO: is it ok to give this after the previous?
            """
            If the parsed input != inp, re-ask with canned response, 
            then try reparsing for input and output (simply)
            """
            # TODO: the message with the incorrect input is still parsed
            # and added to parsed_messages (above), but not given to student right?
            if str(parsed_inp) != str(inp):
                print(
                    (
                        f"Warning: The parsed input {parsed_inp} is not the same "
                        "as what was expected {inp}; response: '{response}'"
                    )
                )
                student_response = self.get_student_diff_answer_response(inp)
                self.messages.append({"role": "user", "content": student_response})
                self.parsed_messages.append({})
                response = self.call()

                parsed_message = {}
                parsed_inp, out, output_pattern = self.parse_output(response)
                parsed_message.update(
                    {"inp": parsed_inp, "label": out, "output_pattern": output_pattern}
                )
                new_inp, input_pattern = self.parse_input(response)
                parsed_message.update(
                    {"next_inp": new_inp, "input_pattern": input_pattern}
                )
                assert str(parsed_inp) == str(
                    inp
                ), f"parsed_inp: {parsed_inp}; inp: {inp}; response: '{response}'"
                self.parsed_messages.append(parsed_message)

            print("Setting next inp...", new_inp)
            self.next_inp = new_inp

        # self.print_history()

        # TODO: add something for when outputs are ignored?
        self.observations.append({"input": inp, "output": out, "prediction": pred})

        return out

    # TODO:

    def get_formatted_inp_out(self, inp, out):
        raise NotImplementedError

    def get_formatted_inp_question(self, inp):
        raise NotImplementedError

    def get_student_no_output_response(self, inp):
        raise NotImplementedError

    def get_student_default_response(self, inp):
        raise NotImplementedError

    def get_student_no_learning_response(self, inp, out):
        raise NotImplementedError

    def get_student_diff_answer_response(self, inp):
        raise NotImplementedError

    def get_gold_output(self, inp):
        raise NotImplementedError
