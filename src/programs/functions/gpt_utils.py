import re

# from pilot.app import get_formatted_fx
from src.programs.utils import parse_function_with_value


class FunctionGPTHelper:
    def __init__(self, dataset):
        self.dataset = dataset

    def parse_input(self, text):
        """
        Parses inputs. Matches patterns:
        - What is wug(inp)?
        """

        for pattern in [
            r"What\s+is\s+wug\((-?\d+)\)\?",
            r"what\s+is\s+wug\((-?\d+)\)\?",
            r"What\s+is\s+your\s+guess\s+for\s+wug\((-?\d+)\)\?",
            r"What[^0-9]+wug\((-?\d+)\)\?",
            r"[^0-9]+try\s+wug\((-?\d+)\)",
            r"What.*wug\((-?\d+)\)\s+is?",
            r"what.*wug\((-?\d+)\)\s+is?",
            r"guess\s+what\s+.*wug\((-?\d+)\)\s+is?",
            r".*wug\((-?\d+)\)\?",  # matches ...wug(INT)?
        ]:
            match = re.search(pattern, text)
            if match:
                n = int(match.group(1))

                return n, pattern

        raise ValueError(f"Found unmatched input pattern: '{text}'")

    def parse_output(self, text):
        """
        Parses outputs. Matches patterns:
        - wug(inp)=out?
        """

        # also match None
        # TODO: currently only matching 'None' in first 2
        for pattern in [
            r"wug\((-?\d+)\)=(-?\d+|None)",
            r"wug\((-?\d+)\)\s+=\s+(-?\d+|None)",
            r"wug\((-?\d+)\)\s+is[^0-9]*(-?\d+)[\s.!;,]",  # matches wug(INT) is.[no other numbers]..(OUT)..[punctuation]
            r"wug\((-?\d+)\)\s+should\s+be\s+(-?\d+)",
            r"wug\((-?\d+)\)\s+is\s+actually\s+(-?\d+)",
        ]:
            match = re.search(pattern, text)
            if match:
                inp = match.group(1)
                out = match.group(2)
                inp = int(inp)
                if out == "None":
                    out = None
                else:
                    try:
                        out = int(out)
                    except:
                        out = list(out)

                assert not isinstance(inp, str)
                assert not isinstance(out, str)
                return inp, out, pattern

        raise ValueError(f"Found unmatched output pattern: '{text}'")

    def get_formatted_inp_out(self, inp, out):
        """
        Return string that gives formatted input and label.
        Maps None to 'undefined'.
        """
        if out is None:
            out = "undefined"
        return f"wug({inp})={out}"

    def get_formatted_inp_question(self, inp):
        """
        Return string that gives formatted input question
        """
        return f"What is wug({inp})?"

    def get_formatted_inp(self, inp):
        """
        Return string that gives formatted input
        """
        return f"wug({inp})"

    def get_student_no_output_response(self, inp):
        # The response that the student should give when no output is found in GPT's generation
        response = f"Sorry, I didn't understand. I can only learn from examples, and they need to be formatted as: wug(INPUT)=ANSWER. Can you please tell me the answer for the previous example, wug({inp}), and give me a new example to give my guess for?"
        return response

    def get_student_default_response(self, inp):
        response = (
            f"Sorry, I still didn't understand. "
            "Can you give me the answer for wug({inp})"
            " and give me a new example?"
        )
        return response

    def get_student_invalid_ex_response(self, inp):
        formatted_inp = self.get_formatted_inp(inp)
        response = (
            f"Sorry, I can't learn from that last example, {formatted_inp}, "
            f"because wug is only defined for inputs between "
            f"{self.dataset.min_input_val} and "
            f"{self.dataset.max_input_val - 1} (inclusive). "
            f"Can you give me a new example?"
        )
        return response

    def get_student_diff_answer_response(self, inp):
        # The response that the student should give when GPT gives an answer for the wrong input
        student_response = (
            f"You gave me the answer for {parsed_inp}, "
            f"but I expected the answer for {inp}. "
            f"Can you give me the answer for wug({inp}) "
            f"and then give me a new example?"
        )
        return student_response

    def parse_student_type(self, response):
        if "1" in response:
            if "2" in response:
                print("Warning: Both 1/2 found in response")
            return 1
        elif "2" in response:
            return 2
        else:
            print(
                "Warning: Hackily selecting that the answer here is (2) based on no other matches"
            )
            return 2

    def get_student_no_learning_response(self, inp, out):
        # The response that the student should give when GPT gives an invalid example
        raise NotImplementedError()

    def get_student_invalid_output_response(self, inp):
        raise NotImplementedError()

    #########################################
    ################ PROMPTS ################
    #########################################

    def get_student_descriptions(self, prog_concept):
        # Always lists g-knower first
        gold_b = prog_concept["gx_b"]
        gx_incorrect_concept = prog_concept["gx_spurious"]
        _, incorrect_b = parse_function_with_value(gx_incorrect_concept)
        fx_incorrect_description = prog_concept["fx_spurious_description"]
        fx_correct_description = prog_concept["fx_concept_description"]
        descriptions = f"""1) Students who correctly think that b={gold_b} but incorrectly think wug is undefined when inputs are {fx_incorrect_description}
    2) Students who correctly think that wug is undefined when inputs are {fx_correct_description} but incorrectly think that b={incorrect_b}"""
        return descriptions

    def get_true_student_description(
        self, prog_concept, student_concept_params, tense="present"
    ):
        is_fx_knower = student_concept_params["fx_knower"]
        gx_incorrect_concept = prog_concept["gx_spurious"]
        _, incorrect_b = parse_function_with_value(gx_incorrect_concept)

        if tense not in ["present", "past"]:
            raise ValueError(f"Invalid tense: {tense}")

        if is_fx_knower:
            if tense == "present":
                return f"""student who correctly thinks that wug is undefined when inputs are {prog_concept['fx_concept_description']} but incorrectly thinks that b={incorrect_b}"""
            elif tense == "past":
                return f"""student who, at the start of this teaching interaction, correctly thought that wug was undefined when inputs were {prog_concept['fx_concept_description']} but incorrectly thought that b={incorrect_b}"""
        else:
            if tense == "present":
                return f"""student who correctly thinks that b={prog_concept['gx_b']} but incorrectly thinks that wug is undefined when inputs are {prog_concept['fx_spurious_description']}"""
            elif tense == "past":
                return f"""student who, at the start of this teaching interaction, correctly thought that b={prog_concept['gx_b']} but incorrectly thought that wug was undefined when inputs were {prog_concept['fx_spurious_description']}"""

    def get_teacher_base_prompt(
        self,
        prog_concept,
        student_concept_params,
        assume_known_prior=False,
    ):
        if assume_known_prior:
            base = self.get_teacher_known_prompt(prog_concept, student_concept_params)
        else:
            base = self.get_teacher_unknown_prompt(prog_concept)

        return base

    def get_teacher_known_prompt(self, prog_concept, student_concept_params):
        # max_val should be max_val *not* inclusive

        gold_description = prog_concept["gold_description"]
        min_val = prog_concept["min_input_val"]
        max_val = prog_concept["max_input_val"]
        gold_b = prog_concept["gx_b"]

        a_min_val = min(prog_concept["poly_kwargs"]["coefficients"])
        a_max_val = max(prog_concept["poly_kwargs"]["coefficients"])

        b_min_val = min(prog_concept["poly_kwargs"]["constants"])
        b_max_val = max(prog_concept["poly_kwargs"]["constants"])

        assert set(prog_concept["poly_kwargs"]["coefficients"]) == set(
            range(a_min_val, a_max_val + 1)
        ), f"coefficients: {prog_concept['poly_kwargs']['coefficients']}"

        assert set(prog_concept["poly_kwargs"]["constants"]) == set(
            range(b_min_val, b_max_val + 1)
        ), f"constants: {prog_concept['poly_kwargs']['constants']}"

        divisible_n_min_val = min(prog_concept["divisible_options"])
        divisible_n_max_val = max(prog_concept["divisible_options"])

        greater_n_min_val = min(prog_concept["greater_options"])
        greater_n_max_val = max(prog_concept["greater_options"])

        assert set(prog_concept["divisible_options"]) == set(
            range(divisible_n_min_val, divisible_n_max_val + 1)
        ), f"divisible_options: {prog_concept['divisible_options']}"

        assert set(prog_concept["greater_options"]) == set(
            range(greater_n_min_val, greater_n_max_val + 1)
        ), f"greater_options: {prog_concept['greater_options']}"

        gold_a = prog_concept["gx_a"]
        gold_b = prog_concept["gx_b"]

        student_description = self.get_true_student_description(
            prog_concept, student_concept_params
        )

        base = f"""You are GPT-teacher, an expert teacher. Your goal is to teach a student what a mystery machine called wug does. This machine takes in numbers and outputs numbers. However, it only works for some numbers and is undefined for others. Your goal is to teach the student on what inputs wug is undefined, and when it is defined, what it does. You should do so as efficiently as possible with helpful input/output examples, such as edge cases. 

    The wug machine works as follows: {gold_description}

    You're going to be interacting with a student who is learning how wug works. The student knows that wug is sometimes undefined. The student also knows that when wug is defined, it computes something of the form a*x+b. In the real wug machine, a={gold_a} and b={gold_b}. However, the student does not know this. The student only knows that a is a constant number between {a_min_val} and {a_max_val} (inclusive) and that b is a constant number between {b_min_val} and {b_max_val} (inclusive).

    The student knows that wug is undefined when the input is one of the following:
    - prime
    - positive
    - even
    - odd
    - divisible by n for n between {divisible_n_min_val} and {divisible_n_max_val} (inclusive)
    - greater than n for n between {greater_n_min_val} and {greater_n_max_val} (inclusive)

    Students have varying previous exposure to wug, and so they understand different parts of how wug works. The student you will be interacting with is a {student_description}. 

    Please make sure to follow these instructions:
    - You are only allowed to give students example inputs, and ask them to guess outputs. You may not explain aspects of the concept to them directly, or ask any other questions. Anything other than inputs and outputs will be ignored by the student.
    - Please format input/output examples as: wug(INPUT)=ANSWER
    - wug only works for numbers between {min_val} to {max_val-1} (inclusive), so restrict the inputs you choose to that range. Any inputs outside of that range will be ignored by the student.
    - Keep teaching with inputs and outputs until the student says they would like to stop, even if you think you have covered the full input range.

    For example, your interactions will look like the following, where capital words indicate placeholders for actual numbers:

    Your interactions will look like the following: 
    System: What is wug(INPUT)?
    User: wug(INPUT)=GUESS
    System: That's [correct/incorrect]. wug(INPUT)=ANSWER. What is wug(NEW INPUT)?

    Please start by asking the student for their guess on an input."""

        return base

    def get_teacher_unknown_prompt(self, prog_concept):
        # max_val should be max_val *not* inclusive
        gold_description = prog_concept["gold_description"]
        min_val = prog_concept["min_input_val"]
        max_val = prog_concept["max_input_val"]
        gold_b = prog_concept["gx_b"]

        a_min_val = min(prog_concept["poly_kwargs"]["coefficients"])
        a_max_val = max(prog_concept["poly_kwargs"]["coefficients"])

        b_min_val = min(prog_concept["poly_kwargs"]["constants"])
        b_max_val = max(prog_concept["poly_kwargs"]["constants"])

        assert set(prog_concept["poly_kwargs"]["coefficients"]) == set(
            range(a_min_val, a_max_val + 1)
        ), f"coefficients: {prog_concept['poly_kwargs']['coefficients']}"

        assert set(prog_concept["poly_kwargs"]["constants"]) == set(
            range(b_min_val, b_max_val + 1)
        ), f"constants: {prog_concept['poly_kwargs']['constants']}"

        divisible_n_min_val = min(prog_concept["divisible_options"])
        divisible_n_max_val = max(prog_concept["divisible_options"])

        greater_n_min_val = min(prog_concept["greater_options"])
        greater_n_max_val = max(prog_concept["greater_options"])

        assert set(prog_concept["divisible_options"]) == set(
            range(divisible_n_min_val, divisible_n_max_val + 1)
        ), f"divisible_options: {prog_concept['divisible_options']}"

        assert set(prog_concept["greater_options"]) == set(
            range(greater_n_min_val, greater_n_max_val + 1)
        ), f"greater_options: {prog_concept['greater_options']}"

        gold_a = prog_concept["gx_a"]
        gold_b = prog_concept["gx_b"]

        student_descriptions = self.get_student_descriptions(prog_concept)

        base = f"""You are GPT-teacher, an expert teacher. Your goal is to teach a student what a mystery machine called wug does. This machine takes in numbers and outputs numbers. However, it only works for some numbers and is undefined for others. Your goal is to teach the student on what inputs wug is undefined, and when it is defined, what it does. You should do so as efficiently as possible with helpful input/output examples, such as edge cases. 

    The wug machine works as follows: {gold_description}

    You're going to be interacting with a student who is learning how wug works. The student knows that wug is sometimes undefined. The student also knows that when wug is defined, it computes something of the form a*x+b. In the real wug machine, a={gold_a} and b={gold_b}. However, the student does not know this. The student only knows that a is a constant number between {a_min_val} and {a_max_val} (inclusive) and that b is a constant number between {b_min_val} and {b_max_val} (inclusive).

    The student knows that wug is undefined when the input is one of the following:
    - prime
    - positive
    - even
    - odd
    - divisible by n for n between {divisible_n_min_val} and {divisible_n_max_val} (inclusive)
    - greater than n for n between {greater_n_min_val} and {greater_n_max_val} (inclusive)

    Students have varying previous exposure to wug, and so they understand different parts of how wug works. There are two kinds of students:
    {student_descriptions}

    Please make sure to follow these instructions:
    - You are only allowed to give students example inputs, and ask them to guess outputs. You may not explain aspects of the concept to them directly, or ask any other questions. Anything other than inputs and outputs will be ignored by the student.
    - Please format input/output examples as: wug(INPUT)=ANSWER
    - wug is only defined for numbers between {min_val} to {max_val-1} (inclusive), so restrict the inputs you choose to that range.
    - Keep teaching with inputs and outputs until the student says they would like to stop, even if you think you have covered the full input range.

    For example, your interactions will look like the following, where capital words indicate placeholders for actual numbers:

    Your interactions will look like the following: 
    System: What is wug(INPUT)?
    User: wug(INPUT)=GUESS
    System: That's [correct/incorrect]. wug(INPUT)=ANSWER. What is wug(NEW INPUT)?

    Please start by asking the student for their guess on an input."""

        return base

    def get_teacher_end_prompt(self, prog_concept):
        student_descriptions = self.get_student_descriptions(prog_concept)
        end = f"""Based on this interaction, which kind of student do you think I was at the start of this teaching session:
    {student_descriptions}

    Please select (1) or (2)."""

        return end

    def eval_student_type(self, teacher, config, gpt_args):
        """Parse the student type from the GPT teacher.
        Determines if the guess is an fx-knower or not based on the integer. (Assumes fx-knower is 2, gx-knower is 1).
        TODO: Hacky: Not all args are used, but keeping for consistency with other parse functions.
        """
        guess_is_fx_knower = int(teacher.student_type) == 2
        guess_is_correct = int(
            guess_is_fx_knower == config["student_concept_params"]["fx_knower"]
        )
        results = {
            "guess_is_fx-knower": int(guess_is_fx_knower),
            "guess_is_correct": guess_is_correct,
            "student_guess": teacher.student_type,
        }
        return results
