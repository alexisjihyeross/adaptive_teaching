import re

from src.programs.utils import parse_function_with_value
from src.programs.fractions.lib import FractionProblem, Fraction


class FractionGPTHelper:

    def __init__(self):
        pass

    def parse_input(self, text):
        """
        Parses inputs. Matches patterns like:
        - What is a/b+c/d?
        """

        # match What is a/b+c/d? or What is a/b*c/d? where a, b, c, d are integers
        for pattern in [
            r"What\s+is\s+(\d+)/(\d+)(\+|\*)(\d+)/(\d+)\?",
            r"what\s+is\s+(\d+)/(\d+)(\+|\*)(\d+)/(\d+)\?",
            # Leave room for space between fraction/operator
            r"What\s+is\s+(\d+)/(\d+)\s*(\+|\*)\s*(\d+)/(\d+)\?",
        ]:
            match = re.search(pattern, text)
            if match:
                num1 = int(match.group(1))
                denom1 = int(match.group(2))
                num2 = int(match.group(4))
                denom2 = int(match.group(5))
                operation = match.group(3)

                frac1 = Fraction(num1, denom1)
                frac2 = Fraction(num2, denom2)
                inp = FractionProblem(frac1, frac2, operation)

                return inp, pattern

        raise ValueError(f"Found unmatched input pattern: '{text}'")

    def parse_output(self, text):
        """
        Parses outputs. Matches patterns like:
        - 1/2+3/4=5/6 or 1/2*3/4=5/6
        """

        # match num1/denom1+num2/denom2=num3/denom3 or num1/denom1*num2/denom2=num3/denom3 where num1, denom1, num2, denom2, num3, denom3 are integers, also get operation
        for pattern in [
            r"(\d+)/(\d+)(\+|\*)(\d+)/(\d+)=(\d+)/(\d+)",
        ]:
            match = re.search(pattern, text)
            if match:
                num1 = int(match.group(1))
                denom1 = int(match.group(2))
                num2 = int(match.group(4))
                denom2 = int(match.group(5))
                num3 = int(match.group(6))
                denom3 = int(match.group(7))
                operation = match.group(3)

                frac1 = Fraction(num1, denom1)
                frac2 = Fraction(num2, denom2)
                inp = FractionProblem(frac1, frac2, operation)
                out = Fraction(num3, denom3)

                return inp, out, pattern

        raise ValueError(f"Found unmatched output pattern: '{text}'")

    def get_formatted_inp_out(self, inp, out):
        """
        Return string that gives formatted input and label.
        Maps None to 'undefined'.
        """
        return f"{inp}={out}"

    def get_formatted_inp_question(self, inp):
        """
        Return string that gives formatted input question
        """
        return f"What is {inp}?"

    def get_formatted_inp(self, inp):
        """
        Return string that gives formatted input
        """
        return f"{inp}"

    def get_student_no_output_response(self, inp):
        # The response that the student should give when no output is found in GPT's generation
        response = f"Sorry, I didn't understand. I can only learn from examples, and they need to be formatted as: a/b+c/d=e/f or a/b*c/d=e/f. Can you please tell me the answer for the previous example, {self.get_formatted_inp(inp)}, and give me a new example to give my guess for?"
        return response

    def get_student_default_response(self, inp):
        response = (
            f"Sorry, I still didn't understand. "
            "Can you give me the answer for {self.get_formatted_inp(inp)}"
            " and give me a new example?"
        )
        return response

    def get_student_no_learning_response(self, inp, out):
        # The response that the student should give when GPT gives an invalid example
        raise NotImplementedError

    def get_student_invalid_ex_response(self, inp):
        # The response that the student should give when GPT gives an invalid example
        formatted_inp = self.get_formatted_inp(inp)
        response = (
            f"Sorry, I can't learn from that last example, {formatted_inp}. "
            f"Can you give me a new example?"
        )
        print(
            "Warning: invalid example given to GPT, but student giving non-specific response"
        )
        print("response:", response)
        return response

    def get_student_invalid_ex_response(self, inp):
        formatted_inp = self.get_formatted_inp(inp)
        response = (
            f"Sorry, I can't learn from that last example, {formatted_inp}. "
            f"Can you give me a new example?"
        )
        print(
            "Warning: invalid example given to GPT, but student giving non-specific response"
        )
        print("response:", response)
        return response

    def get_student_invalid_output_response(self, inp):
        raise NotImplementedError

    def get_student_diff_answer_response(self, inp):
        # The response that the student should give when GPT gives an answer for the wrong input
        raise NotImplementedError("Not implemented yet")

    def parse_student_type(self, response):
        if "1" in response:
            if "2" in response:
                print("Warning: Both 1/2 found in response")
            if "3" in response:
                print("Warning: Both 1/3 found in response")
            return 1
        elif "2" in response:
            if "3" in response:
                print("Warning: Both 2/3 found in response")
            if "1" in response:
                print("Warning: Both 2/1 found in response")
            return 2
        elif "3" in response:
            if "1" in response:
                print("Warning: Both 3/1 found in response")
            if "2" in response:
                print("Warning: Both 3/2 found in response")
            return 3
        else:
            print(
                "Warning: Hackily selecting that the answer here is (2) based on no other matches"
            )
            return 2

    #########################################
    ################ PROMPTS ################
    #########################################

    def get_student_descriptions(self, population_concept_params):
        descriptions = ""
        for idx, _ in enumerate(population_concept_params):
            desc = f"{idx+1}) "
            desc += population_concept_params[idx]["plural_description"]
            descriptions += desc

            if idx != len(population_concept_params) - 1:
                descriptions += "\n"
        return descriptions

    def get_true_student_description(self, student_concept_params):
        return student_concept_params["single_description"]

    def get_teacher_base_prompt(
        self,
        true_student_concept_params,
        population_concept_params,
        assume_known_prior=False,
    ):
        print(
            "Getting teacher base prompt for assume_known_prior={}".format(
                assume_known_prior
            )
        )
        if assume_known_prior:
            base = self.get_teacher_known_prompt(true_student_concept_params)
        else:
            base = self.get_teacher_unknown_prompt(population_concept_params)

        return base

    def get_teacher_unknown_prompt(self, student_concept_params):
        student_descriptions = self.get_student_descriptions(student_concept_params)

        base = f"""You are GPT-teacher, an expert teacher. Your goal is to teach a student how to multiply and add fractions as efficiently as possible with helpful examples.

    You will be interacting with a student who has spent some time with fraction arithmetic but still has some misconceptions about how it works. There are {len(student_concept_params)} kinds of students: 
    {student_descriptions}
    You should try to figure out which kind of student you are interacting with and then teach them accordingly.

    Please make sure to follow these instructions:
    - You are only allowed to give students example fraction problems, and ask them to guess the outputs. You may not explain any concepts to them directly, or ask any other questions. Anything other than example fraction problems and answers will be ignored by the student.
    - The student has not learned how to simplify fractions yet, so please do not simplify the fractions in your examples. Leave the answers in their unsimplified form. The student will also not simplify their answer.
    - Please only use fractions with positive numerators and denominators.
    - Do not teach arithmetic with mixed numbers or whole numbers.
    - Only teach fraction addition and multiplication. Please format input/output examples as: a/b+c/d=e/f for addition or a/b*c/d=e/f for multiplication.
    - Keep teaching with fraction problems and outputs until the student says they would like to stop, even if you think you have covered the full input range.

    For example, your interactions will look like the following, where capital words indicate placeholders for actual verb lemmas and categories:

    Your interactions will look like the following (where letters are placeholders for actual numbers): 
    System: What is a/b+c/d?
    User: a/b+c/d=e/f 
    System: That's [correct/incorrect]. a/b+c/d=x/y. What is g/h+i/j? 

    Please start by asking the student for their guess on a fraction example."""

        return base

    def get_teacher_known_prompt(self, true_concept_params):
        true_student_description = self.get_true_student_description(
            true_concept_params
        )

        base = f"""You are GPT-teacher, an expert teacher. Your goal is to teach a student how to multiply and add fractions as efficiently as possible with helpful examples.

    You will be interacting with a student who has spent some time with fraction arithmetic but still has some misconceptions about how it works. The student you will be interacting with is a {true_student_description}. 

    Please make sure to follow these instructions:
    - You are only allowed to give students example fraction problems, and ask them to guess the outputs. You may not explain any concepts to them directly, or ask any other questions. Anything other than example fraction problems and answers will be ignored by the student.
    - The student has not learned how to simplify fractions yet, so please do not simplify the fractions in your examples. Leave the answers in their unsimplified form. The student will also not simplify their answer.
    - Please only use fractions with positive numerators and denominators.
    - Do not teach arithmetic with mixed numbers or whole numbers.
    - Only teach fraction addition and multiplication. Please format input/output examples as: a/b+c/d=e/f for addition or a/b*c/d=e/f for multiplication.
    - Keep teaching with fraction problems and outputs until the student says they would like to stop, even if you think you have covered the full input range.

    For example, your interactions will look like the following, where capital words indicate placeholders for actual verb lemmas and categories:

    Your interactions will look like the following (where letters are placeholders for actual numbers): 
    System: What is a/b+c/d?
    User: a/b+c/d=e/f 
    System: That's [correct/incorrect]. a/b+c/d=x/y. What is g/h+i/j? 

    Please start by asking the student for their guess on a fraction example."""

        return base

    def get_teacher_end_prompt(self, student_concept_params):
        student_descriptions = self.get_student_descriptions(student_concept_params)
        assert len(student_concept_params) == 2
        end = f"""Based on this interaction, which kind of student do you think I was at the start of this teaching session:
    {student_descriptions}

    Please select (1) or (2)."""

        return end

    def eval_student_type(self, teacher, config, gpt_args):
        """Parse the student type from the GPT response.
        TODO: kind of hacky, but when gpt prompts are created, the population_params are stored in gpt_args so that they can be used by this function.
        """

        # teacher.student_type is not 0-indexed but true_student_idx is
        guess_student_idx = teacher.student_type - 1
        guess_id = [pop["id"] for pop in gpt_args["population_params"]][
            guess_student_idx
        ]
        guess_is_add_generalizer = int(guess_id == "add_generalizer")
        guess_is_correct = int(gpt_args["true_student_idx"] == guess_student_idx)

        results = {
            "guess_id": guess_id,
            "guess_is_add_generalizer": guess_is_add_generalizer,
            "guess_is_correct": guess_is_correct,
        }
        return results
