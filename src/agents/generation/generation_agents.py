from openai import OpenAI
from src.agent_constructor.agent import Agent
from src.agent_constructor.core import Text, Document


class QueryGenerator(Agent):
    def __init__(self, url: str, model_name: str, options: int = 3):
        super().__init__("query_generator")
        self.dummy_mode = not (url and model_name)
        self.options = options

        if not self.dummy_mode:
            self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, task: Text) -> Text:
        if self.dummy_mode:
            return task

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are search query generator"},
                {"role": "user", "content": f"""Given original query: `{task}`, generate {self.options} variations of rewriting original query
                
                Yours output template (only options, nothing else!!!):
                - "option 1"
                - "option 2"
                .
                .
                .
                - "option {self.options}"
                """}
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.split('\n')


@ComponentRegistry.register_component(ComponentNames.RANDOM_WORD_GENERATOR)
class RandomWordGenerator(Agent):
    def __init__(self, url: str, model_name: str):
        super().__init__("random_word_generator")
        self.dummy_mode = not (url and model_name)
        self.words = words
        if not self.dummy_mode:
            self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def get_random_items(self,
                         items: List,
                         min_count: int = 3,
                         max_count: int = 5,
                         ) -> List[str]:

        max_possible = min(max_count, len(items))
        min_possible = min(min_count, len(items))

        count = random.randint(min_possible, max_possible)
        return random.sample(items, count)

    def run(self, query: Text, example: Text) -> Text:
        if self.dummy_mode:
            return query  # , context

        random_words = self.get_random_items(self.words)
        # print(random_words)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a professional task rephraser"},
                {"role": "user", "content": f"""
                You are given a task: {query}.
                Rephrase it using the given set of words: {random_words}.
                Your paraphrasing should result in a new problem text. 
                It's crucial that the meaning of the problem remains the same. 
                In your answer, include only the paraphrased text, nothing else.
                """},
            ],
            temperature=0.6,
            max_tokens=500
        )

        rephrased_promt: Text = response.choices[0].message.content

        # print(rephrased_promt)

        new_response = self.client.chat.completions.create(
            model=self.model_name,
           messages=[
                {"role": "system", "content": "You are professional python-coder"},
                {"role": "user", "content": rephrased_promt +
                    f"""
                        To solve the problem pay attention to the examples: {example}.
                        In your response, include only the Python code and nothing else.
                        Remember to keep the code as simple as possible and avoid overcomplicating it where possible.
                        The simpler the code, the better. Your response should contain nothing but code.
                        VERY IMPORTANT: Do not write anything in the answer except the code, it should not contain any text, only code.
                    """}
            ],
            temperature=0.1,
            max_tokens=500
        )
        return new_response.choices[0].message.content


@ComponentRegistry.register_component(ComponentNames.SHOTS_GENERATOR)
class ZeroFewShotGenerator(Agent):
    def __init__(self, url: str, model_name: str):
        super().__init__("zero_few_shot_generator")
        self.dummy_mode = not (url and model_name)
        self.topics = topics
        if not self.dummy_mode:
            self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, query: Text, example: Text) -> tuple[Text, Text, Text]:
        if self.dummy_mode:
            return query

        zero_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a professional Python developer"},
                {"role": "user", "content": f"""
                    You have a task: {query}.
                    Solve it like an expert with 10 years of experience.
                    In your response, include only the Python code and nothing else.
                    Remember to keep the code as simple as possible and avoid overcomplicating it where possible.
                    The simpler the code, the better. Your response should contain nothing but code.
                    VERY IMPORTANT: Do not write anything in the answer except the code, it should not contain any text, only code.
                """}
                ],
            temperature=0.1,
        )
        zero_example: Text = zero_response.choices[0].message.content

        selected_topic = random.choice(self.topics)
        # print(f"selected_topic - {selected_topic}")
        topic_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a professional Python developer"},
                {"role": "user", "content": f"""
                    Write a code example on this topic: {selected_topic}
                    In your response, include only the Python code and nothing else.
                    Remember to keep the code as simple as possible and avoid overcomplicating it where possible.
                    The simpler the code, the better. Your response should contain nothing but code.
                    VERY IMPORTANT: Do not write anything in the answer except the code, it should not contain any text, only code.
                """}
            ],
            temperature=0.1,
        )
        topic_example: Text = topic_response.choices[0].message.content

        few_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a professional Python developer"},
                {"role": "user", "content": f"""
                    You have a task: {query}.
                    Take this as an example: {example}
                    Write another code example that is similar in structure or content to the ones provided to you.
                """}
            ],
            temperature=0.1,
        )
        few_example: Text = few_response.choices[0].message.content

        return zero_example, topic_example, few_example


@ComponentRegistry.register_component(ComponentNames.INSTRUCT_GENERATOR)
class InstuctGenerator(Agent):
    def __init__(self, url: str, model_name: str):
        super().__init__("instuct_generator")
        self.dummy_mode = not (url and model_name)
        if not self.dummy_mode:
            self.client = OpenAI(base_url=url, api_key="vllm")

        self.model_name = model_name

    def run(self, example: Text) -> Text:
        if self.dummy_mode:
            return example

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a professional Python developer and Prompt-engineer"},
                {"role": "user", "content": f"""
                Create a prompt for a large language model that will generate a solution like this: {example}
                The response should contain only the prompt you created and nothing else.
                For example:
                Your input:
                ```python
                    Z = np.random.random((3,3,3))
                    print(Z)
                ```
                Your output:
                Generate Python code that creates a 3D NumPy array with shape (3,3,3) filled with random floats between 0 and 1. 
                Then print the array. Output only the code, no explanations.
                """}
            ],
            temperature=0.1,
        )
        instruction: str = response.choices[0].message.content

        return instruction
