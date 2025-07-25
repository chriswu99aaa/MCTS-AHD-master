
import numpy as np
# from copy import deepcopy

from source.interface_LLM import InterfaceAPI as InterfaceLLM
# from .prompts import Prompts

# from pathlib import Path
import re
from typing import List, Tuple
from source.evolution import Evolution
import random

class BOInterface():
    def __init__(self,api_endpoint, api_key, llm_model, debug_mode, interface_prob,timeout,population,**kwargs):

        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs
    

        # Initialize the LLM interface
        self.llm_interface = InterfaceLLM(
            api_endpoint=api_endpoint,
            api_key=api_key,
            model_LLM=llm_model,
            debug_mode=debug_mode
        )
        self.timeout = timeout

        self.interface_eval = interface_prob
        self.evolution = Evolution(
            api_endpoint=api_endpoint,
            api_key=api_key,
            model_LLM=llm_model,
            debug_mode=debug_mode,
            prompts=interface_prob.prompts,
            **kwargs
        )
        # Initialize the prompts
        self.prompts = interface_prob.prompts
    
        self.prompt_task = self.prompts.get_task()
        self.prompt_func_name = self.prompts.get_func_name()
        self.prompt_func_inputs = self.prompts.get_func_inputs()
        self.prompt_func_outputs = self.prompts.get_func_outputs()
        self.prompt_inout_inf = self.prompts.get_inout_inf()
        self.prompt_other_inf = self.prompts.get_other_inf()
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"
        
        self.population = population

    def get_heuristics(self, prompt):
        """
        Initialize the Bayesian optimization process.
        """
        response = self.llm_interface.get_response(
            prompt_content=prompt,
            temp=0.8
        )
        # print("LLM Response:", response)
        return response
    
    
    def get_initial_prompt(self):
        """
        Get initial heuristics for Bayesian optimization.
        """ 
        prompt  =  self.prompt_task  + "\n"
        prompt += "First, describe the design idea and main steps of your algorithm in one sentence. "
        prompt += f"The description must be inside a brace outside the code implementation.\n"
        prompt += f"Next, implement it in Python as a function named '{self.prompt_func_name}'.\n"
        prompt += f"This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}."
        prompt += f" The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}.\n"
        prompt += self.prompt_inout_inf + " " + self.prompt_other_inf + "\n"
        prompt += "Do not give additional explanations." + "\n"
        return prompt
    

    def extract_heuristics(self, response) -> List[Tuple[str, str]]:
        """
        Extract code and algorithm from the LLM response.
        Args:
            response (str): The response from the LLM containing the code and description.
        Returns:
            List[Tuple[str, str]]: A list of tuples where each tuple contains the description and the corresponding code.
        """
        # Extract the description from the response
        description = re.findall(r'\{(.*?)\}', response, re.DOTALL)

        code = re.findall(r"```python(.*?)```", response, re.DOTALL)

        print("Extracted Descriptions:", description)
        print("Extracted Code Snippets:", code)
        heuristics = []
        # if len(description) != len(code):
        #     raise ValueError("The number of descriptions and code snippets do not match.")
        
        for d, c in zip(description, code):
            desc = d.strip().replace('\n', ' ')
            code_string = c.strip()
            heuristic = {
                'description': desc,
                'code': code_string}
            heuristics.append(heuristic)
        
        return heuristics
    

    def generate_heuristic_by_action(self, action: str):
        """
        Use specified action to generate a new heuristic function using the Evolution class.

        Parameters:
            action (str): One of ['i1', 'e1', 'e2', 'm1', 'm2', 's1']

        Returns:
            code (str): The generated code for the heuristic function.
            algorithm (str): The description of the heuristic algorithm.
        """
        if action == "i1":
            code, algorithm = self.evolution.i1()

        elif action in ("e1", "e2", "s1"):
            # 这些方法都期望一个 dict 列表
            if action == "s1":
                parents = self.select_best_parents(5)
                indivs = [self.node_to_dict(p) for p in parents]
                code, algorithm = self.evolution.s1(indivs)
            elif action == "e1":
                parents = self.select_best_parents(3)
                indivs = [self.node_to_dict(p) for p in parents]
                code, algorithm = self.evolution.e1(indivs)
            else:
                parents = self.select_best_parents(3)
                indivs = [self.node_to_dict(p) for p in parents]
                code, algorithm = self.evolution.e2(indivs)

        elif action in ("m1", "m2"):
            # 这些方法都期望单个 dict
            parent = self.select_best_parents()[0]
            print(parent)
            indiv = self.node_to_dict(parent)
            if action == "m1":
                code, algorithm = self.evolution.m1(indiv)
            else:
                code, algorithm = self.evolution.m2(indiv)
        else:
            raise ValueError(f"Unknown action: {action}")

        return code, algorithm
    

    def select_best_parents(self, num=1):
        """
        Select the best parent from the population based on the path length.
        """
        if not self.population:
            raise ValueError("Population is empty.")
        
        if len(self.population) < num:
            num = len(self.population)

        # Sort the population by path length and return the best one
        sorted_nodes = sorted(self.population, key=lambda node: node.tour_length)
        return sorted_nodes[:min(num, len(sorted_nodes))]
    

    def select_random_parents(self, num=5):
        """
        Select the best parents from the population based on the path length.
        """
        if not self.population:
            raise ValueError("Population is empty.")
        
        if len(self.population) < num:
            num = len(self.population)
        k = min(num, len(self.population))
        return random.sample(self.population, k)

    def node_to_dict(self, node):
        return {
            "algorithm": node.algorithm,
            "code":      node.code,
            "objective": node.tour_length
        }
    
    def get_embedding(self, prompt_content):
        """
        Get the embedding of the prompt content using the LLM interface.
        """
        return self.llm_interface.get_embedding(prompt_content)
    



        
        



    
