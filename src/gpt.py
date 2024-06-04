import openai
import os
import pickle
import time


class GPT:
    def __init__(
        self,
        *args,
        model_name="gpt-3.5-turbo-0301",
        base_prompt=None,
        end_prompt=None,
        skip_cache=False,
        seed=0,
        cache_file="gpt_cache.pkl",
        **kwargs,
    ):
        print("Initializing GPT object...")
        super().__init__(*args, **kwargs)

        # used for openai API
        self.seed = seed

        # base_prompt should never be None, but end_prompt can be
        assert base_prompt is not None
        self.base_prompt = base_prompt
        self.end_prompt = end_prompt

        self.skip_cache = skip_cache
        self.cache_file = cache_file

        self.cache = {}
        self.load_cache()

        self.model_name = model_name

        self.messages = [
            {"role": "user", "content": self.base_prompt},
        ]
        self.parsed_messages = [{} for _ in range(len(self.messages))]

        self.usages = []

    def load_cache(self):
        cache = {}
        if not self.skip_cache and os.path.exists(self.cache_file):
            try:
                cache = pickle.load(open(self.cache_file, "rb"))
            except Exception as e:
                print("Error loading cache:", e)
                try:
                    cache = pickle.load(open(self.cache_file, "rb"))
                except Exception as e:
                    print("Error again...", e)
                    print("Not re-loading cache")
        self.cache.update(cache)

    def get_cache_key(self, max_tokens, temperature, messages, seed):
        # messages is a list of dicts, so we need to convert to something hashable
        tuple_messages = tuple([tuple(m.items()) for m in messages])
        return (self.model_name, tuple_messages, max_tokens, temperature, seed)

    def initialize(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.organization = os.getenv("OPENAI_ORGANIZATION")

    def get_cost(self):
        cost = 0
        if self.model_name == "gpt-4-0314":
            prompt_tok_cost = 0.03 / 1000
            completion_tok_cost = 0.06 / 1000
        else:
            raise NotImplementedError()

        for usage in self.usages:
            cost += prompt_tok_cost * usage["prompt_tokens"]
            cost += completion_tok_cost * usage["completion_tokens"]
        return cost

    def call(self, max_tokens=100, temperature=0.5):

        key = self.get_cache_key(max_tokens, temperature, self.messages, self.seed)

        if key in self.cache:
            output = self.cache[key]
            self.messages.append({"role": "assistant", "content": output})
            return output

        else:
            """Calls openai.ChatCompletion with self.messages
            and appends output as new message"""
            while True:
                try:
                    completion = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=self.messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        seed=self.seed,
                    )
                    break
                except openai.error.RateLimitError:
                    # TODO: hacky
                    print("rate limit error; waiting 10 secs...")
                    time.sleep(10)
                    continue
            output = completion.choices[0].message["content"]
            self.messages.append({"role": "assistant", "content": output})
            self.usages.append(completion.usage)

            # Update cache
            if not self.skip_cache:
                self.cache[key] = output

        return output

    def write_cache(self):
        pickle.dump(self.cache, open(self.cache_file, "wb"))

    def print_history(self):
        if len(self.messages) != len(self.parsed_messages):
            print(
                f"Warning: len(self.messages) ({len(self.messages)}) != len(self.parsed_messages) ({len(self.parsed_messages)}). There is probably something wrong with parsing messages."
            )
        for m, parsed in zip(self.messages, self.parsed_messages):
            print(m)
            print(parsed)
            print("")
