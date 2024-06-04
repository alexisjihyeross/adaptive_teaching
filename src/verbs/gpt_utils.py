import re

########################################
############### GPT UTILS ##############
########################################


class VerbsGPTHelper:
    def __init__(self, dataset):
        self.dataset = dataset

    def parse_output(self, text):
        """
        Parses outputs. Matches patterns:
        - 'LEMMA' is [a/an] 'CATEGORY' verb
        """

        for pattern in [
            r"'([A-Za-z]+)' is (?:a|an) '(\+?[A-Za-z_+]+)' verb",
            r"'([A-Za-z]+)' is (?:a|an) '(\+?[A-Za-z_-]+)' verb",  # allowing '-' bc GPT might accidentally output it
        ]:
            match = re.search(pattern, text)
            if match:
                lemma = match.group(1)
                category = match.group(2)

                return lemma, category, pattern

        raise ValueError(f"Found unmatched output pattern: '{text}'")

    def get_formatted_inp_out(self, inp, out):
        """
        Return string that gives formatted input and label
        """
        return f"'{inp}' is a '{out}' verb"

    def get_formatted_inp_question(self, inp):
        """
        Return string that gives formatted input question
        """
        return f"What type of verb is '{inp}'?"

    def get_formatted_inp(self, inp):
        return f"'{inp}'"

    def get_teacher_unknown_prompt(self):
        base = f"""You are GPT-teacher, an expert teacher. Your goal is to teach a student how to conjugate English past tense verbs as efficiently as possible with helpful examples.

    Specifically, your goal is to teach students about four categories of past tense verbs:
    - '+ed': add 'ed' to the verb lemma
    - '+d': add 'd' to the verb lemma
    - 'y_to_ied': if the verb lemma ends in a 'y', replace the 'y' with 'ied'
    - '+consonant+ed': if the verb lemma ends in a consonant, double the last consonant and add 'ed'

    Different students have different confusion points, but each student has one verb category that they are the least familiar with. While teaching the student, you should aim to infer what verb category they are the least familiar with in order to teach and correct their misconceptions most efficiently.

    Please make sure to follow these instructions:
    - You are only allowed to give students example verb lemmas, and ask them to guess verb categories. You may not explain any concepts to them directly, or ask any other questions. Anything other than example verb lemmas and categories will be ignored by the student.
    - Please format input/output examples as: 'LEMMA' is a 'CATEGORY' verb
    - Keep teaching until the student says they would like to stop, even if you think they understand the verb categories.
    - You are only allowed to teach students about verbs in the four categories ('+ed', '+d', 'y_to_ied', and '+consonant+ed'). Please do not give examples from other categories, like irregular verbs.

    For example, your interactions will look like the following, where capital words indicate placeholders for actual verb lemmas and categories:

    Your interactions will look like the following: 
    System: What type of verb is 'LEMMA'?
    User: 'LEMMA' is a 'CATEGORY' verb
    System: That's [correct/incorrect]. 'LEMMA' is a 'CATEGORY' verb. What type of verb is 'LEMMA'?

    Please start by asking the student for their guess on a lemma."""

        return base

    def get_teacher_base_prompt(
        self,
        student_concept_params,
        assume_known_prior=False,
    ):

        unknown_concept = student_concept_params["unknown_concept"]

        if assume_known_prior:
            base = self.get_teacher_known_prompt(unknown_concept)
        else:
            base = self.get_teacher_unknown_prompt()

        return base

    def get_teacher_known_prompt(self, unknown_concept):
        base = f"""You are GPT-teacher, an expert teacher. Your goal is to teach a student how to conjugate English past tense verbs as efficiently as possible with helpful examples.

    Specifically, your goal is to teach students about four categories of past tense verbs:
    - '+ed': add 'ed' to the verb lemma
    - '+d': add 'd' to the verb lemma
    - 'y_to_ied': if the verb lemma ends in a 'y', replace the 'y' with 'ied'
    - '+consonant+ed': if the verb lemma ends in a consonant, double the last consonant and add 'ed'

    Different students have different confusion points, but each student has one verb category that they are the least familiar with. The student you will be interacting with is the least familiar with the '{unknown_concept}' category.

    Please make sure to follow these instructions:
    - You are only allowed to give students example verb lemmas, and ask them to guess verb categories. You may not explain any concepts to them directly, or ask any other questions. Anything other than example verb lemmas and categories will be ignored by the student.
    - Please format input/output examples as: 'LEMMA' is a 'CATEGORY' verb
    - Keep teaching until the student says they would like to stop, even if you think they understand the verb categories.
    - You are only allowed to teach students about verbs in the four categories ('+ed', '+d', 'y_to_ied', and '+consonant+ed'). Please do not give examples from other categories, like irregular verbs.

    For example, your interactions will look like the following, where capital words indicate placeholders for actual verb lemmas and categories:

    Your interactions will look like the following: 
    System: What type of verb is 'LEMMA'?
    User: 'LEMMA' is a 'CATEGORY' verb
    System: That's [correct/incorrect]. 'LEMMA' is a 'CATEGORY' verb. What type of verb is 'LEMMA'?

    Please start by asking the student for their guess on a lemma."""

        return base

    def get_teacher_end_prompt(self):
        end = f"""Based on this interaction, which verb category do you think I was the least familiar with at the start of this teaching session?"""

        return end

    def get_student_base_prompt(self, unknown_category):
        base = f"""You are GPT-student, a student who is learning how to conjugate English past tense verbs. You are specifically learning about four categories of past tense verbs:
    - '+ed': add 'ed' to the verb lemma
    - '+d': add 'd' to the verb lemma
    - 'y_to_ied': if the verb lemma ends in a 'y', replace the 'y' with 'ied'
    - '+consonant+ed': if the verb lemma ends in a consonant, double the last consonant and add 'ed'

    At the start of the interaction, you have incomplete understanding about these categories.  Your goal is to simulate how well a real student would learn in response to examples they've observed.

    You will be interacting with a teacher who gives you examples of verbs and asks you for guesses on the examples before telling you the correct verb categories. Over the course of teaching, you will learn how to correctly categorize verbs and get more examples correct based on the examples you're shown. At several points, the teacher will quiz you to assess your current understanding of verb categories. 

    You are the least familiar with the '{unknown_category}' verb category, meaning that you start out less likely to predict '{unknown_category}' for examples than the other categories.

    Please make sure to follow these instructions:
    - You can only learn from example verb lemmas and categories. Please ignore anything else that the teacher says, like explanations of concepts. 
    - You may not ask the teacher questions or give any information other than your guesses on verb categories given to you by the teacher.
    - Please format input/output examples as: 'LEMMA' is a 'CATEGORY' verb.
    - Please assume that the teacher will only give examples of real verb lemmas, so always provide your best guess on verb category (even if you think the given example is not a real verb).
    - Please limit your guesses to the four categories of verbs: '+ed', '+d', 'y_to_ied', '+consonant+ed'. 

    Your interactions will look like the following, where capital words indicate placeholders for actual lemmas and categories:

    User: What type of verb is 'LEMMA'?
    System: 'LEMMA' is a 'CATEGORY' verb
    User: That's [correct/incorrect]. 'LEMMA' is a 'CATEGORY' verb. What type of verb is 'LEMMA'?

    During quizzes, you will be given multiple verb lemmas at a time, i.e.:

    User: QUIZ: What types of verbs are 1) 'LEMMA' 2) 'LEMMA'... 
    System: 1) 'LEMMA' is a 'CATEGORY' verb 2) 'LEMMA is a 'CATEGORY' verb..."""

        return base

    def eval_student_type(self, teacher, config, gpt_args):
        student_guess = teacher.student_type
        student_guess_is_correct = student_guess == config["unknown_concept"]

        results = {
            "student_guess": student_guess,
            "student_guess_is_correct": int(student_guess_is_correct),
        }

        assert student_guess in self.dataset.unique_outputs
        return results
