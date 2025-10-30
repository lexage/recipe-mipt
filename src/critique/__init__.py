from pydantic import BaseModel


class Critic:
    def __init__(self, evaluate_flag: bool = False):
        self.evaluate_flag = evaluate_flag

    def llm(self, prompt: str | list) -> str:
        return ""

    def evaluate(self, question: str, answer: str) -> str | None:
        if self.evaluate_flag:
            evaluation = ""
        else:
            evaluation = None
        return evaluation

    def criticise(self, question: str, answer: str, evaluation: str | None) -> str:
        critique = ""
        return critique

    def run_one(self, question: str, answer: str) -> str:
        evaluation = self.evaluate(question, answer)
        critique = self.criticise(question, answer, evaluation)
        return critique

    def run(self, question: str, answer: str) -> str:
        return self.run_one(question, answer)


class Problem(BaseModel):
    question: str
    answer: str
    evaluation: str | None = None
    critique: str | None = None


class ComplexCritic(Critic):
    def __init__(self, decompose_flag: bool = False, evaluate_flag: bool = False):
        self.decompose_flag = decompose_flag
        self.evaluate_flag = evaluate_flag

    def decompose(self, problem: Problem) -> list[Problem]:
        return []

    def synthesise(self, subproblems: list[Problem]) -> Problem:
        return Problem(question="", answer="")

    def run_many(self, problem: Problem) -> Problem:
        problems = self.decompose(problem)
        solutions = []
        for problem in problems:
            problem.critique = self.run_one(problem.question, problem.answer)
            solutions.append(problem)
        solved_problem = self.synthesise(solutions)
        return solved_problem

    def run(self, question: str, answer: str) -> str:
        if self.decompose_flag:
            problem = Problem(question=question, answer=answer)
            solved_problem = self.run_many(problem)
            return solved_problem.critique
        else:
            return self.run_one(question, answer)
