from abc import ABC, abstractmethod
from pydantic import BaseModel


class Agent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"


class Example(BaseModel):
    question: str
    answer: str
    feedback: str
    refined: str


def get_examples():
        return[
            Example(
                question = """You are given a DataFrame containing sales information for various products across different categories. 
                Your task is to add a new column, Percentage_of_Category_Sales, which for each product will show what percentage of its category's total sales that specific product represents.""",
                answer = """
                        import pandas as pd

                        data = {
                            'Category': ['Electronics', 'Electronics', 'Books', 'Books', 'Books', 'Clothing', 'Clothing'],
                            'Product': ['Laptop', 'Mouse', 'Novel', 'Textbook', 'Magazine', 'Shirt', 'Jeans'],
                            'Sales': [1200, 50, 30, 80, 10, 40, 60]
                        }
                        df = pd.DataFrame(data)

                        print(df)

                        category_total_sales = df.groupby('Category')['Sales'].sum()

                        print(category_total_sales)

                        df['Percentage_of_Category_Sales'] = (df['Sales'] / category_total_sales) * 100

                        print(df)
                    """,
                feedback =   """The line category_total_sales = df.groupby('Category')['Sales'].sum() correctly calculates the sum of sales for each category.
                            The result is a Series where the index consists of category names ('Electronics', 'Books', 'Clothing') and the values are the corresponding sales sums.
                            When you perform arithmetic operations between two Series (or a Series and a DataFrame), Pandas, by default, attempts to align the operands by their indexes.
                            The index of df['Sales'] is a numerical range (0, 1, 2, 3, 4, 5, 6), which is the standard DataFrame index.
                            The index of category_total_sales consists of string category names ('Books', 'Clothing', 'Electronics').
                            Since the indexes of df['Sales'] (numbers) and category_total_sales (strings) are completely different and have no common values, Pandas cannot find matches for most elements:
                            For df['Sales'].iloc[0] (value 1200, index 0), Pandas looks for index 0 in category_total_sales. No such index exists there.
                            For category_total_sales.loc['Electronics'] (value 1250), Pandas looks for the index 'Electronics' in df['Sales']. No such index exists there.
                            As a result of this alignment operation, Pandas fills values for which no match is found with NaN (Not a Number). In our case, since the indexes do not match at all in type and value, almost all (or all) elements in the new Percentage_of_Category_Sales column will become NaN.
                            """,
                refined = """import pandas as pd

                            data = {
                                'Category': ['Electronics', 'Electronics', 'Books', 'Books', 'Books', 'Clothing', 'Clothing'],
                                'Product': ['Laptop', 'Mouse', 'Novel', 'Textbook', 'Magazine', 'Shirt', 'Jeans'],
                                'Sales': [1200, 50, 30, 80, 10, 40, 60]
                            }
                            df = pd.DataFrame(data)
                            df['Category_Total_Sales_Correct'] = df.groupby('Category')['Sales'].transform('sum')
                            df['Percentage_of_Category_Sales_Correct'] = (df['Sales'] / df['Category_Total_Sales_Correct']) * 100

                            print("\nDataFrame after CORRECT implementation:")
                            print(df)"""
            ),
            Example(
                question = """Given a sales table:

                            Product,Sales_qty,Price_per_unit
                            A,10,100
                            B,5,200
                            C,20,50

                            You need to add a column "Revenue" equal to the product of "Sales_qty" and "Price_per_unit" using Pandas.
                        """,
                answer = """
                        import pandas as pd

                        data = {
                            'Product': ['A', 'B', 'C'],
                            'Sales_qty': [10, 5, 20],
                            'Price_per_unit': [100, 200, 50]
                        }

                        df = pd.DataFrame(data)

                        # Mistake: addition used instead of multiplication
                        df['Revenue'] = df['Sales_qty'] + df['Price_per_unit']

                        print(df)
                    """,
                feedback =   """Instead of multiplication ( * ), addition ( + ) is used, so "Revenue" contains the sum of sales quantity and price, not their product.
                                This fundamentally changes the meaning:

                                For product A, instead of 10 * 100 = 1000, it shows 10 + 100 = 110.

                                This leads to incorrect data analysis because revenue is wrongly calculated""",
                refined = """
                                import pandas as pd

                                data = {
                                    'Product': ['A', 'B', 'C'],
                                    'Sales_qty': [10, 5, 20],
                                    'Price_per_unit': [100, 200, 50]
                                }

                                df = pd.DataFrame(data)

                                df['Revenue'] = df['Revenue'] = df['Sales_qty'] * df['Price_per_unit']


                                print(df)
                            """
            ),                     
        ]

class SelfRefine(Agent):
    """
    Agent that implements Self-Refine: Iterative Refinement with Self-Feedback.

    Prompts in original paper are task-specific.
    """

    def __init__(self, name: str = "Self-Refine", critique_only: bool = True):
        super().__init__(name)
        self.critique_only = critique_only
        self.refrain_feedback: str =  """You are a Python programming assistant.
        You will be given a problem and implementation in Python. 
        Your goal is to write feedback for the implementation step for:
        1. Logical errors in reasoning 
        2. Syntax and semantic correctness 
        3. Conceptual misunderstandings 
        4. Potential bugs or edge cases 
        5. Alignment with problem requirements You will need this as a hint when you 
        try again later. Only provide the few sentence description in your answer, 
        not the implementation."""

        self.refrain_refine: str = """Now fix the implementation of problem using feedback. 
                                    Return refined implementation"""

    def llm(self, prompt: str) -> str:
        # TODO: implement actual LLM call
        return ""

    def feedback_prompt(
        self, question: str, answer: str, examples: list[Example] = []
    ) -> str:
        few_shot: str = """Examples:
        """
        for example in examples:
            few_shot += f"""
            Problem: {example.question}
            Implementation: {example.answer}
            {self.refrain_feedback}
            Feedback: {example.feedback}
            """

        prompt = f""" END OF EXAMPLES
        Problem: {question}
        Implementation: {answer}
        {self.refrain_feedback}
        Feedback:
        """

        final_prompt = few_shot + prompt
        return final_prompt

    def feedback(self, question: str, answer: str, examples: list[Example] = []) -> str:
        feedback_prompt = self.feedback_prompt(question, answer, examples)
        result = self.llm(feedback_prompt)
        return result

    def refine_prompt(
        self, question: str, answer: str, feedback: str, examples: list[Example] = []
    ) -> str:
        few_shot: str = """Examples:
         """
        for example in examples:
            few_shot += f"""
            Problem: {example.question}
            Implementation: {example.answer}
            {self.refrain_feedback}
            Feedback: {example.feedback}
            {self.refrain_refine}
            Refined Implementation: {example.refined}
            """

        prompt = f""" END OF EXAMPLES
        Problem: {question}
        Implementation: {answer}
        {self.refrain_feedback}
        Feedback: {feedback}
        {self.refrain_refine}
        Refined Implementation:
        """

        final_prompt = few_shot + prompt
        return final_prompt

    def refine(
        self, question: str, answer: str, feedback: str, examples: list[Example] = []
    ) -> str:
        refine_prompt = self.refine_prompt(question, answer, feedback, examples)
        result = self.llm(refine_prompt)
        return result

    def run(self, question: str, answer: str):
        feedback = self.feedback(question, answer)
        if self.critique_only:
            return feedback

        improved_answer = self.refine(question, answer, feedback)
        return improved_answer
