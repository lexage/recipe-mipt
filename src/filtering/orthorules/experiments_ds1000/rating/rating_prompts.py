# TODO: rewrite prompts for DS-1000

rule_generation_prompt = """
Generate 50 specific rules for rating data from the training dataset (SlimPajama), in order to select a high-quality subset to train large language models (LLMs) that will improve their performance on Coding benchmark evaluation. The descriptions of the training data and downstream task are provided below. The rules should focus on various aspects such as data quality, relevance, diversity, and other characteristics that would be beneficial for Coding benchmark.

Description of training data:
The SlimPajama dataset is a large-scale dataset. It is designed to be a compact, high-quality dataset curated for pre-training large language models. The dataset includes a diverse range of texts, sourced from various domains such as web pages, books, and academic articles, providing a rich and varied training corpus for developing robust and versatile language models.


Description of downstream task:
The Code Generation LM Evaluation Harness, part of the BigCode project, is a framework designed to evaluate large language models (LLMs) on their ability to generate code. It provides a structured environment to assess the performance of these models across various programming tasks and languages. The harness supports automated evaluation metrics and facilitates benchmark comparisons, making it a valuable tool for researchers and developers aiming to enhance the code generation capabilities of LLMs.

Requirements for the Rules:
Each rule should be concise and specific.
The rules could be basic text quality rules or task-relaated quality rules.
The rules should be written in clear, natural language and be easy to understand.
Now, please generate the rules for me in natural language.
"""


rules = [
    "Syntax Highlighting: Include texts that contain syntax highlighting or structured code comments.",
    "Grammar Quality: Exclude texts with excessive spelling and grammatical errors.",
    "Programming Keywords: Prioritize samples containing programming language keywords and constructs.",
    "Language Focus: Exclude texts that are predominantly non-English unless they are code snippets.",
    "Concept Explanation: Select texts with clear, concise explanations of programming concepts.",
    "Reputable Sources: Prioritize texts from reputable sources like well-known programming blogs and documentation sites.",
    "Minimum Length: Exclude texts that contain less than 50 words as they may not provide sufficient context.",
    "Best Practices: Include examples that demonstrate best coding practices.",
    "Language Diversity: Prioritize texts that include diverse programming languages covered in the BigCode project.",
    "Error Solutions: Select texts that provide examples of common programming errors and their solutions.",
    "Technique Comparison: Include texts with comparative discussions of different coding techniques or tools.",
    "Current Practices: Exclude texts with outdated or deprecated coding practices.",
    "Algorithm Explanation: Prioritize texts that include algorithm explanations with code snippets.",
    "Duplication Check: Exclude samples that are heavily duplicated within the dataset.",
    "API Usage: Select samples that demonstrate use of APIs from well-known software libraries.",
    "Multi-Language Code: Include texts with embedded code in multiple programming languages.",
    "Development Paradigms: Prioritize texts that discuss software development paradigms (e.g., object-oriented programming).",
    "Relevance Check: Exclude non-relevant texts like purely historical accounts of programming without technical details.",
    "Decision Context: Include texts that provide context on why certain coding decisions are made.",
    "Annotated Code: Prioritize texts that contain code with annotations explaining each part of the code.",
    "Step-by-Step Code: Include samples where code is broken down into step-by-step explanations.",
    "Non-Promotional: Exclude texts that are purely promotional or sales-focused.",
    "Debugging Techniques: Select texts that discuss debugging techniques with code examples.",
    "Tool Comparison: Include texts that compare different programming tools or environments.",
    "Technical Focus: Prioritize articles or excerpts from technical books that focus on programming.",
    "Academic Pseudo-Code: Include texts from academic papers that contain pseudo-code or algorithms.",
    "Content Density: Exclude texts that are excessively verbose without substantive content.",
    "Optimization Tips: Prioritize texts that provide insights into code optimization.",
    "Advanced Topics: Include texts that cover advanced programming topics like concurrency or security.",
    "Architecture Patterns: Select texts that discuss architectural patterns with code examples.",
    "Programming Paradigms: Prioritize examples that demonstrate functional or logic-based programming.",
    "Technical Emphasis: Exclude samples that focus solely on non-technical aspects of IT projects.",
    "Quality Solutions: Include forum and Q&A entries with high-quality code solutions.",
    "Documentation Inclusion: Select project documentation and readme files that include example usage of code.",
    "Commented Code: Prioritize texts with code that includes comprehensive inline comments.",
    "Jargon Balance: Exclude texts with a high density of technical jargon unless accompanied by clear explanations or code.",
    "Executable Snippets: Include code snippets that are functional and can be executed without modifications.",
    "Complexity Discussion: Prioritize texts that explain the computational complexity of algorithms with examples.",
    "Integration Showcase: Include texts that showcase the integration of different technologies or languages.",
    "Version Control: Select samples that explain version control practices with code snippets.",
    "Cross-Platform Coding: Prioritize texts that discuss cross-platform coding challenges and solutions.",
    "Interactive Tutorials: Include interactive coding tutorials or walkthroughs.",
    "Proprietary Code: Exclude any samples containing proprietary code without proper authorization.",
    "Scalability Focus: Select examples that discuss the scalability of code or systems.",
    "Accessibility Coding: Prioritize samples that address coding for accessibility or internationalization.",
    "Performance Analysis: Include texts that analyze the performance of different coding approaches.",
    "Content Relevance: Exclude texts that mix code with irrelevant images or multimedia that don’t add educational value.",
    "Ethical Coding: Prioritize texts that discuss ethical considerations in programming.",
    "Tool Usage: Include examples of how to use popular development tools and environments through coding tutorials.",
    "Project Scope: Select texts that clearly define the scope and objectives of programming projects."
]


def rating_prompt(text_A):
    prompt = f'''
You are a helpful assistant. We are training a language model using the SlimPajama dataset to enhance our model's performance on the coding benchmark evaluation. Evaluate the following example from SlimPajama dataset and assign a quality score between 0 and 1 (0 indicates the worst quality, and 1 indicates perfect quality).

Respond only with a single float number.

Example:
{text_A}
'''
    return prompt


def rule_rating_prompt(rule, text_A):
    prompt = f'''
You are a helpful assistant. We are training a language model using the SlimPajama dataset to enhance our model's performance on the coding benchmark evaluation. Evaluate the following example from SlimPajama dataset and assign a quality score between 0 and 1 (0 indicates the worst quality, and 1 indicates perfect quality) according to the provided rule:
{rule}

Respond only with a single float number.

Example:
{text_A}
'''
    return prompt