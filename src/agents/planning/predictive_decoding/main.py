import os
import sys
sys.path.insert(0, os.getcwd())

from openai import OpenAI

import json
import torch
import io
import argparse
import numpy as np
from src.agent_constructor.agent import Agent
from typing import Any, Optional, Union, List, Dict


class MPCSample(Agent):  # the algorithm should be stateless, and generates a whole plan / code / chain of actions at once.
    def __init__(
        self,
        model_url: str = "http://localhost:7215/v1",
        model_name: str = "Qwen/Qwen1.5-32B-Chat-AWQ",
        name: str = "MPCSample",
        prompt_path: Optional[str] = None,
        lookahead_thought_length: int = 3,
        lookahead_token_length: Optional[int] = None,    # the length of the lookahead token sequence, default use thought length as evaluation chunk
        reward_threshold: float = 1.0,
        beam_size: int = 8,
        beam_temperature: float = 0.7,
        select_temperature: float = 0.1,
        n_generate_sample: int = 8,
        value_type: str = "logp",
        do_sample: bool = True,
        use_memory: bool = True,
        max_problem_size: int = 50,
        examples: Optional[Union[str, List[Dict[str, str]]]] = None
    ):
        super().__init__(name)
        
        self.model_name = model_name
        
        self.client = OpenAI(
            base_url=model_url,
            api_key="vllm"
        )
        
        if prompt_path is not None:
            self.prompts = json.load(open(prompt_path, 'r'))
        else:
            self.prompts = {
                "prompt": "default_prompt_template",
                "system_msg": "default_system_message" 
            }
            
        self.examples = self.load_examples(examples)
        
        self.problem_size = max_problem_size
        self.n_gram = self.problem_size
        
        self.reward_threshold = reward_threshold 
        self.lookahead_decision_length = lookahead_thought_length
        self.lookahead_token_length = lookahead_token_length
        
        self.do_sample = do_sample
        self.beam_temperature = beam_temperature
        self.select_temperature = select_temperature
        self.beam_size = beam_size
        self.n_generate_sample = n_generate_sample
        self.value_type = value_type
        self.use_memory = use_memory
        
    
    def load_examples(
        self, 
        examples: Optional[Union[str, List[Dict[str, str]]]]
    ) -> List[str]:
        """Load few-shot examples from file or use provided examples."""
        if isinstance(examples, str) and examples.endswith('.jsonl'):
            examples_list = []
            with open(examples, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        examples_list.append(json.loads(line))
            return [example["example"] for example in examples_list]
        elif isinstance(examples, list):
            return [example["example"] for example in examples]
        else:
            print(f"Warning: Invalid examples type.")
            return []
        
        
    def make_prompt(self, prompt, question, memory=None):
        # may need to reimplement
        if memory is None:
            memory = self.memory
            
        user_content = ""
        if self.examples:
            user_content += "Here are some examples of problems and their solution plans in the desired format:\n\n"
            for i, example in enumerate(self.examples, 1):
                user_content += f"Example {i}:\n{example}\n\n"
            user_content += "Now make a plan to solve the following problem. Provide only the plan in the same format, without code.\n\n"
            
        user_content += prompt
            
        # with io.StringIO() as f:
        #     f.write(question)
        #     model_input = f.getvalue()

        with io.StringIO() as f:    
            # f.write(prompt)
            for a in memory:
                if a is not None:
                    f.write(f"{a}")
            answer_prefix = f.getvalue()
                    
        return user_content, answer_prefix
        
        
    def update_trajectory_pool(self, outputs, reward=None, id=None, memory=None):
        
        # update the trajectory pool with the generated action rollouts by llm
        
        # if we have the id, means this is for parallel generation, therefore we need to update only the corresponding state for that question
        
        action_rollouts = outputs["action_chain"]
        
        if memory is None:
            memory = self.memory[id] if id is not None else self.memory
        
        history_rollouts = [a for a in memory if a is not None]
        
        item = []
        
        item.append({"Action": None, "Verified": False, "Reward": reward})
        
        for action in history_rollouts:
            item.append({"Action": action, "Verified": True, "Reward": reward})
            
        for action in action_rollouts:
            item.append({"Action": action, "Verified": None, "Reward": reward})
        
        if id is not None:
            self.trajectory_pool[id].append(item)
        else:
            self.trajectory_pool.append(item)
    
    
    def parse_action_sequence(self, action_output, parse_prefix="", id=None, memory=None):  
        def _get_start_end_token_id(original_text, text, tokens):
            if text not in original_text:
                text = text.strip()
                if text not in original_text:
                    return 0, -1
                
            cnt_length = [len(token) for token in tokens]
            cumulated_cnt_length = np.cumsum(cnt_length)
            index_start =  original_text.index(text) # processed action index in action
            index_end = index_start + len(text)
                
            token_start = np.argmax(cumulated_cnt_length >= index_start)
            token_end = np.argmax(cumulated_cnt_length >= index_end)
            
            if token_end < token_start:
                token_end = -1
                
            return token_start + 1, token_end + 1 # +1 to account for the >= sign, instead of > sign
        
        def _clean_action(action):
            action = action.replace("`\n", "")
            return action + "\n"
        
        prefix = parse_prefix
        
        if isinstance(action_output, str): # no logprob information
            
            if memory is None:
                memory = self.memory[id] if id is not None else self.memory
            
            all_prefix = [prefix] + [a for a in memory if a is not None]
            
            action = action_output
            
            for prefix in all_prefix:
                if prefix in action: # added, in case there is repeat of prompt inside the generation
                    action = action.split(prefix)[1]
                        
            action = action.lstrip('\n')
            
            if action == "":
                return None, None
            
            # Here is the start of the action chain:
            if '\n' in action:
                all_actions = action.split('\n')
            else:
                all_actions = [action]
            
            first_action = all_actions[0] + '\n'
            action_chain = [a + '\n' for a in all_actions][: self.lookahead_decision_length] # only keep the first n actions
            print(action_chain)
            print()
            
            return {"action": first_action, "action_chain": action_chain}, first_action
        
        elif isinstance(action_output, dict): # need logprob information
            
            action_text_output = action_output["text"]
            action_logprobs = action_output["logprobs"]
            action_tokens = action_output["tokens"]
            
            if memory is None:
                memory = self.memory[id] if id is not None else self.memory
            
            all_prefix = [prefix] + [a for a in memory if a is not None]
            
            token_start, token_end = 0, -1
            action = action_text_output
            
            for prefix in all_prefix:
                if prefix in action: # added, in case there is repeat of prompt inside the generation
                    action = action.split(prefix)[1]
                    
             # remove all '\n' in the beginning
            action = action.lstrip('\n')
            
            if action == "":
                return None, None
            
            if self.lookahead_token_length is not None: # limit the length of the lookahead token sequence as a chunk for lookahead
                token_start, token_end = _get_start_end_token_id(action_text_output, action, action_tokens)
                token_end = min(token_end, token_start + self.lookahead_token_length)
                action = "".join(action_tokens[token_start:token_end])
                
                # Here is the start of the action chain:
                if '\n' in action:
                    all_actions = action.split('\n')
                else:
                    all_actions = [action]
                
                first_action = all_actions[0] + '\n'
                action_chain = [a + '\n' for a in all_actions]
                
                action_logprobs = action_logprobs[token_start:token_end]
                
                action_prob = np.exp(sum(action_logprobs)) 
                if token_end - token_start > 0:
                    action_length = token_end - token_start
                else:
                    action_length = len(action_tokens)
                    
                action_prob = action_prob ** (1 / action_length) # normalize by the length of the action
                
                action_chain = [a for a in action_chain if a.strip()!=""]
            
                return {"action": first_action, "action_chain": action_chain, "action_prob": action_prob}, first_action
            
            else: # limit the number of thoughts in the lookahead
                
                # Here is the start of the action chain:
                if '\n' in action:
                    all_actions = action.split('\n')
                else:
                    all_actions = [action]
                
                first_action = all_actions[0] + '\n'
                action_chain = [_clean_action(a) for a in all_actions][: self.lookahead_decision_length] # only keep the first n actions
                
                token_start, token_end = _get_start_end_token_id(action_text_output, "".join(action_chain), action_tokens)
                
                action_logprobs = action_logprobs[token_start:token_end]
                
                action_prob = np.exp(sum(action_logprobs))
                if token_end - token_start > 0:
                    action_length = token_end - token_start
                else:
                    action_length = len(action_tokens)
                    
                action_prob = action_prob ** (1 / action_length)  # normalize by the length of the action
                action_chain = [a for a in action_chain if a.strip()!=""]
                
                return {"action": first_action, "action_chain": action_chain, "action_prob": action_prob}, first_action
        
        else:
            raise NotImplementedError
    
    
    def get_valid_actions(self, action_history, id=None):       
        def is_valid_action(action):
            return action is not None and action.strip() != ""
        
        all_results = []
        
        if id is not None:
            trajectory_pool = self.trajectory_pool[id]
        else:
            trajectory_pool = self.trajectory_pool
        
        for trajectory in trajectory_pool:
            start = max(1, len(trajectory) - self.n_gram + 1)
            
            for i in range(start):
                n = min([len(trajectory) - i, self.n_gram, len(action_history) + 1])
                
                n_gram_list = [trajectory[i + s]["Action"] for s in range(n)]
                n_gram_reward = [trajectory[i + s]["Reward"] for s in range(n)][-1]
                
                match = (action_history[-n + 1:] == n_gram_list[:-1])
                
                if match:
                    all_results.append((n_gram_list[-1], n_gram_reward))
                    
        all_results = [item for item in all_results if is_valid_action(item[0])]
        return all_results
    
      
    def lookahead_decision_model(self, reward_threshold=1.0):
        # given the look ahead predictions, make the next action
        
        # ! todo: choose the best action when there are multiple options
        
        action_history = [None] + [action for action in self.memory if action is not None]
        
        all_valid_action_values = self.get_valid_actions(action_history)
        
        if len(all_valid_action_values) < 1:
            return None
        
        all_valid_values = np.array([item[1] for item in all_valid_action_values])
        all_valid_actions = [item[0] for item in all_valid_action_values]
        
        if all_valid_values.max() < reward_threshold:
            return None
        
        if self.do_sample: 
            probs = np.exp(all_valid_values/self.select_temperature)
            probs = probs / probs.sum()
            
            all_action_prob_pairs = dict()
            
            for (action, prob) in zip(all_valid_actions, probs):
                if action not in all_action_prob_pairs:
                    all_action_prob_pairs[action] = prob
                else:
                    all_action_prob_pairs[action] += prob
            
            all_valid_actions = list(all_action_prob_pairs.keys())
            probs = list(all_action_prob_pairs.values())
            
            sample = torch.multinomial(torch.tensor(probs), 1).item()
            action = all_valid_actions[sample]
        else:
            action = all_valid_actions[np.argmax(all_valid_values)]
            
        return action
    
    
    def reflection_tips(self): 
        
        # determine if the model is stuck and requires self-reflection, used sparingly
                
        reflection = ""
        
        if len(self.trajectory_pool) > 0 :
            all_actions = ",".join(list(set([trajectory[-1]["Action"] for trajectory in self.trajectory_pool if trajectory[-1]["Action"] is not None])))
            
            question = self.prompts["question"]
            reflection += f"I have generated {all_actions}, but none of them are correct. I need to revise them to solve the problem {question}."
            
            # if reflection is multiline, follow the format of python comment
            indent = "    "
            reflection = indent + "# " + reflection.replace("\n", "\n# ")

        if reflection != "":
            return True, reflection
        else:
            return False, None


    def run(self, question, prompts=None, **kwargs):
        if "end_suffix" in kwargs:
            end_suffix = kwargs["end_suffix"]
        else:
            end_suffix = None
            
        if prompts is not None:
            self.prompts = prompts
        self.prompts["question"] = question
        
        self.trajectory_pool = []
        
        args = {
            "n_generate_sample": self.beam_size,
            "max_iters": self.problem_size,
            "max_tokens": 500,# if self.lookahead_token_length is None else self.lookahead_token_length,
            "temperature": self.beam_temperature,
            "top_p": 1.0,
            "stop": ["\n\n"],           
            "logprobs": (self.value_type == "logp"),
            "value_type": self.value_type
        }
        
        args = argparse.Namespace(**args)
        
        iter = 0
        self.memory = [None] * self.problem_size
        reflection_tips = ""
        
        failed_attempts = 0
        max_failed_attempts = 3 

        while iter < args.max_iters:
            if not self.use_memory:
                self.trajectory_pool = [] # don't keep memory, but re-init the trajectory pool every time
            
            input_prompt, answer_prefix = self.make_prompt(self.prompts["prompt"], question)
            system_message = self.prompts["system_msg"]
            
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": input_prompt})
            if answer_prefix:
                messages.append({"role": "assistant", "content": answer_prefix})
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                    n=args.n_generate_sample,
                    stop=args.stop if args.stop else None,
                    logprobs=args.logprobs,
                )
                
                action_sequence_samples = []
                for choice in response.choices:
                    if args.logprobs:
                        logprobs_content = choice.logprobs.content if choice.logprobs else []
                        logprobs_list = [token.logprob for token in logprobs_content]
                        tokens_list = [token.token for token in logprobs_content]
                        action_sequence_samples.append({
                            "text": choice.message.content,
                            "logprobs": logprobs_list,
                            "tokens": tokens_list
                        })
                    else:
                        action_sequence_samples.append(choice.message.content)
                
                success = True
            except Exception as e:
                print(f"OpenAI API error: {e}")
                success = False
                action_sequence_samples = []
            
            # success, action_sequence_samples = self.llm_model.generate_with_config(system_message, input_prompt, generation_config, answer_prefix=answer_prefix)
            
            if success:
                for action_sequence in action_sequence_samples:
                    parse_prefix = self.prompts["prompt"]
                    processed_output, action = self.parse_action_sequence(action_sequence, parse_prefix=parse_prefix)

                    if action is None: 
                        continue
                    reward = 0
                    if args.value_type == "logp":
                        reward = processed_output["action_prob"]
                    else:
                        raise NotImplementedError
                    
                    self.update_trajectory_pool(processed_output, reward=reward)
            else:
                print("Failed to generate action sequence.")
                return False, None
                
            reward_threshold = self.reward_threshold if reflection_tips == "" else 0  # if reflection is needed, lower the threshold so that the model won't get stuck
            action = self.lookahead_decision_model(reward_threshold=reward_threshold)
            
            if action is not None:
                self.memory[iter] = action
                iter += 1
                failed_attempts = 0
                reflection_tips = self.reflection_tips()
                
                if iter > args.max_iters:
                    break
                
                if end_suffix is not None and end_suffix in action.strip():
                    break
            else:
                failed_attempts += 1
                reflection_tips = self.reflection_tips()
                if reflection_tips[0]:
                    self.memory[iter] = reflection_tips[1]
                    break
                elif failed_attempts >= max_failed_attempts:
                    break
                      
        if success:
            with io.StringIO() as f:
                # iterate through the state
                for a  in self.memory:
                    if a is not None:
                        f.write(f"{a}\n")

                full_output = f.getvalue()

            return True, full_output
            
        return False, None