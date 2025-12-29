from abc import ABC, abstractmethod
from src.agent_constructor.agent import Agent
from langchain_tavily import TavilySearch
from langchain_core.tools import BaseTool
from typing import List
from langchain_experimental.utilities import PythonREPL
from typing import Annotated
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_experimental.utilities import PythonREPL
from typing import Annotated
from langchain_core.tools import tool
from pydantic import BaseModel


class Example(BaseModel):
    question: str
    answer: str
    critique: str
    correct_answer: str

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
                critique =   """The line category_total_sales = df.groupby('Category')['Sales'].sum() correctly calculates the sum of sales for each category.
                            The result is a Series where the index consists of category names ('Electronics', 'Books', 'Clothing') and the values are the corresponding sales sums.
                            When you perform arithmetic operations between two Series (or a Series and a DataFrame), Pandas, by default, attempts to align the operands by their indexes.
                            The index of df['Sales'] is a numerical range (0, 1, 2, 3, 4, 5, 6), which is the standard DataFrame index.
                            The index of category_total_sales consists of string category names ('Books', 'Clothing', 'Electronics').
                            Since the indexes of df['Sales'] (numbers) and category_total_sales (strings) are completely different and have no common values, Pandas cannot find matches for most elements:
                            For df['Sales'].iloc[0] (value 1200, index 0), Pandas looks for index 0 in category_total_sales. No such index exists there.
                            For category_total_sales.loc['Electronics'] (value 1250), Pandas looks for the index 'Electronics' in df['Sales']. No such index exists there.
                            As a result of this alignment operation, Pandas fills values for which no match is found with NaN (Not a Number). In our case, since the indexes do not match at all in type and value, almost all (or all) elements in the new Percentage_of_Category_Sales column will become NaN.
                            """,
                refine_answer = """import pandas as pd

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
                critique =   """Instead of multiplication ( * ), addition ( + ) is used, so "Revenue" contains the sum of sales quantity and price, not their product.
                                This fundamentally changes the meaning:

                                For product A, instead of 10 * 100 = 1000, it shows 10 + 100 = 110.

                                This leads to incorrect data analysis because revenue is wrongly calculated""",
                refine_answer = """
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

class Critic(Agent):
    """
    Agent that implements CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing.
    Finds faults in model response via external tools or few-shot prompting.

    Few-shot examples in original paper are task-specific.
    """

    def __init__(
        self,
        name: str = "CRITIC",
        use_tools: bool = False,
        tools: dict = {},
        examples: list[Example] = [],
    ):
        super().__init__(name)
        self.refrain =     """You are a Python programming assistant.
        You will be given a problem and implementation in Python and critique of implementation. 
        Your goal is to make the correct implementation based on the critique
        Return only correct implementation"""
        self.use_tools = use_tools
        self.tools = self.get_tools()
        self.examples = examples

    def llm(self, prompt: str) -> str:
        # TODO: Replace with actual LLM call
        return ""

    def get_python_repl_tool(self):
        repl = PythonREPL()

        def python_repl_tool(
            code: Annotated[str, "The python code to execute."],
        ) -> str:
            """Use this to execute python code. If you want to see the output of a value,
            you should print it out with print(...). This is visible to the user."""
            try:
                result = repl.run(code)
            except BaseException as e:
                return f"Failed to execute. Error: {repr(e)}"
            return f"Successfully executed:\n : {result}"

    return python_repl_tool

    def get_web_search_tool(self, max_results: int = 3) -> BaseTool:
        wrapper = DuckDuckGoSearchAPIWrapper(
            max_results=max_results,
        )
        tool = DuckDuckGoSearchResults(
            api_wrapper=wrapper
        )
        tool.description = (
            """Search information using Internet. Use English language for searching"""
        )
        return tool   

    def get_code_tool(self):

        def code_tool(answer: str) -> str:
            promt = f"You only need to extract the Python code from the following text. I want to compile this code. And You can't fix it. Text: {answer} "
            return self.llm(promt)

        return code_tool

    def get_problem_tool(self):

        def problem_tool(problem: str) -> str:
            promt = f"You should create a query to search the internet for implementations to problem. Problem: {problem} "
            return self.llm(promt)

        return  problem_tool
        
    def get_tools(self) -> Dict[str, Any]:
        return {
            "problem" : self.get_problem_tool(),
            "websearch" : self.get_web_search_tool(),
            "getcode" : self.get_code_tool(),
            "replcode": self.get_repl_code_tool(),
        }

    def make_correct_prompt(self, question: str, answer: str, critique: str) -> str:
        few_shot: str = """EXAMPLES:
        """
        for example in self.examples:
            few_shot += f"""
                Problem: {example.question}

                Proposed implementation: {example.answer}

                Critique: {example.critique}

                Correct answer: {example.refine_answer}
            """

        prompt = f"""END OF EXAMPLES
            {self.refrain}
            Problem: {question}
            Proposed implementation: {answer}

            

            Critique: {critique}

            Correct answer:
        """

        final_prompt = few_shot + prompt
        return final_prompt

    def get_web_implementations(self, question: str) -> List[str]:
        problem = self.tools["problem"](question)
        search_result = self.tools["websearch"].invoke(f"I have got a problem in Python: {problem}.")
        implementations_list = self.llm(f"""Get a list of links from text. Return only this list, separated by commas. 
        Examples: 'https://stackoverflow.com/questions/5453026/string-to-list-in-python, https://stackoverflow.com/questions/5453027/string-to-list-in-python'. Text: {search_result} """)
        return implementations_list.split(', ')

    def tool_critique(self, question: str, response: str) -> str:
        # TODO: implement tool calls
        tool_criticisms = []
        web_implementations = self.get_web_implementations(question)
        for implementation in web_implementations:    
            code = self.tools["getcode"](implementation)
            tool_criticisms.append(f"Possible implementation: {code}")
        code = self.tools["getcode"](response)
        replcode_result = self.tools["replcode"](code)
        tool_criticisms.append(f"Code's compilation results: {replcode_result}")
        return "\n".join(tool_criticisms)

    def llm_critique(self, question: str, response: str) -> str:
        critique = self.tool_critique(question, response)
        prompt = self.make_correct_prompt(question, response, critique)
        res = self.llm(prompt)
        return res

    def run(self, question: str, response: str) -> str:
        # TODO: verify then correct

        if self.use_tools:
            return self.llm_correct(question, response, self.tool_critique(question, response))
            
        return response