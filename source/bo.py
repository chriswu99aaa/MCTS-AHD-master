
import importlib
import logging
import numpy as np
from pathlib import Path

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C

from source.bo_interface import BOInterface

from os import path
import sys

from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
import traceback
import math

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.objective import IdentityMCObjective
from botorch.optim import optimize_acqf,optimize_acqf_discrete
from botorch.acquisition.logei import qLogExpectedImprovement
from gpytorch.kernels import ScaleKernel, RBFKernel
import torch

class HeuristicNode:
    # def __init__(self, code, algorithm, parent=None, action=None, tour_length=None, expanded_nodes=None):
    def __init__(self, code, algorithm, parent=None, action=None, tour_length=None):
        self.code = code
        self.algorithm = algorithm
        self.parent = parent
        self.action = action  # action taken to generate this node
        self.children = []
        self.tour_length = tour_length
        # self.expanded_nodes = expanded_nodes
        self.feature_vector = None  # embedding vector for the code
    
    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        return f"HeuristicNode(description={self.algorithm}, path_length={self.tour_length})"

    def set_tour_length(self, tour_length):
        self.tour_length = tour_length

    def set_expanded_nodes(self, expanded_nodes):
        self.expanded_nodes = expanded_nodes


class BayesianOptimizer:
    def __init__(self,  paras, problem, **kwargs):
        # self.root = root_dir
        # self.ws = workspace_dir
        self.problem = problem
        self.paras = paras
        # LLM settings
        self.use_local_llm = paras.llm_use_local
        self.url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint  # currently only API2D + GPT
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        # BO settings
        self.n_init = paras.n_init               # e.g. 10
        self.n_iters = paras.n_iters             # e.g. 20
        self.bounds = paras.bounds                # e.g. [[0,1],[0,1],...]
        self.dim = len(self.bounds)

        # LLM settings
        self.use_local_llm = kwargs.get('use_local_llm', False)
        assert isinstance(self.use_local_llm, bool)
        if self.use_local_llm:
            assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
            assert isinstance(kwargs.get('url'), str)
            self.url = kwargs.get('url')

        # containers
        self.X = []  # shape (n_obs, d)
        self.Y = []  # scalar cost

        self.count = 0
        # surrogate
        kernel = C(1.0) * (RBF(length_scale=1.0) + Matern(length_scale=1.0, nu=2.5))
        self.gp = GaussianProcessRegressor(kernel=kernel,
                                           alpha=paras.noise, #later set to automatically learned noise level by white kernel
                                           normalize_y=True,
                                           n_restarts_optimizer=5)

        # Experimental settings
        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id
        self.output_path = paras.exp_output_path
        self.debug_mode = paras.exp_debug_mode  # if debug

        self.timeout = self.paras.eva_timeout

        self.heuristics = []

        self.population = []

        self.action_types = ['i1', 'e1', 'e2', 'm1', 'm2', 's1']  # action types for evolution

        self.tour_lengths = []  # to store tour lengths
        self.expanded_nodes = []  # to store expanded nodes


    def embed_code(self, code):
        """
        Use OpenAI API to embed the code snippet.
        Returns a numpy array of the embedding.
        """
        try:
            embedding = self.bo_interface.get_embedding(code)
            return np.array(embedding)
        except Exception as e:
            print(f"OpenAI embedding Fail: {e}")
            traceback.print_exc()
 
            return np.zeros(1536)  # text-embedding-ada-002的维度

    def encode_action(self, action):
        """
        Encode the action using one-hot encoding.
        """
        if not hasattr(self, 'action_encoder'):
            self.action_encoder = OneHotEncoder(sparse_output=False)
            # 使用所有可能的action类型进行fit
            actions_array = np.array(self.action_types).reshape(-1, 1)
            self.action_encoder.fit(actions_array)
        

        if action is None:
            action = 'i1'  # 默认action
        action_vector = self.action_encoder.transform([[action]])[0]
        return action_vector


    def create_feature_vector(self, code, action):
        """
        Create a feature vector by combining the code embedding and action one-hot encoding.
        The feature vector will be of shape (1536 + 6,).
        """
        code_embedding = self.embed_code(code)
        action_onehot = self.encode_action(action)
        
        # 拼接特征：[code_embedding(1536维) + action_onehot(6维)]
        feature_vector = np.concatenate([code_embedding, action_onehot])
        return feature_vector


    # def evaluate_heuristics(self, code):
    #     """
    #     TODO
    #     """
    #     """
    #     1. Write generated heuristic code via LLM (using your existing client)
    #     2. Call MCTS-AHD eval script to get cost
    #     """

    #     # 1) 将 LLM 生成的代码写入 gpt.py
    #     prob_dir = self.problem.problems_path
    #     if not path.exists(prob_dir):
    #         raise FileNotFoundError(f"Problem directory {prob_dir} does not exist.")
    #     gpt_file = path.join(prob_dir, "gpt_bo.py")
    #     with open(gpt_file, "w", encoding="utf-8") as f:
    #         f.write(code)


    #     for m in ("gpt_bo", "eval_bo"):
    #         if m in sys.modules:
    #             del sys.modules[m]
    #     # 2) load eval.py
    #     eval_path = path.join(prob_dir, "eval_bo.py")
    #     spec = importlib.util.spec_from_file_location("eval_mod", eval_path)
    #     eval_mod = importlib.util.module_from_spec(spec)
    #     spec.loader.exec_module(eval_mod)
    #     eval_fn = eval_mod.eval_heuristic
    
    #     # 3) run few instances of the problem
    #     dataset_dir = path.join(prob_dir, "dataset")
    #     ds_file_path = path.join(dataset_dir, f"train{self.paras.n_instances}_dataset.npy")
            
    #     if not path.isfile(ds_file_path):
    #         # dynamically load problems/<problem>/gen_inst.py
    #         gen_inst_path = path.join(prob_dir, "gen_inst.py")
    #         spec = importlib.util.spec_from_file_location("gen_inst", gen_inst_path)
    #         gen_inst = importlib.util.module_from_spec(spec)
    #         spec.loader.exec_module(gen_inst)
    #         # call the dataset generation function
    #         gen_inst.generate_datasets()
    #         # 确保文件已经生成
    #         if not path.isfile(ds_file_path):
    #             raise RuntimeError(f"Failed to generate {ds_file_path}")
           
    #     else:
    #         # file exists
    #         data = np.load(ds_file_path)

    #     data = np.load(ds_file_path)
    #     few = min(len(data), 5)   # 只试前 5 条，加速实验

    #     # 4) 循环调用 eval_heuristic，收集 tour_length 和 expanded_nodes
    #     lengths, expands = [], []
    #     for i in range(few):
    #         tour_len, exp_nodes = eval_fn(data[i])
    #         lengths.append(tour_len)
    #         expands.append(exp_nodes)

    #     # 5) 返回平均指标
    #     return float(np.mean(lengths)), int(np.mean(expands))
    


    def initialize(self):
        print("- Initialization Start -")
        interface_problem = self.problem

        self.bo_interface = BOInterface(self.api_endpoint, self.api_key, self.llm_model,
                                        self.use_local_llm, interface_problem, use_local_llm=self.use_local_llm, url=self.url,
                                        timeout=self.timeout, population=self.population)
        
        # Generate initial heuristic without retry logic since there's only one node
        code, description = self.bo_interface.generate_heuristic_by_action(action="i1")
        print(f"Initial response: {code}, {description}")

        tour_length = self.problem.batch_evaluate([code], 0)[0]  # Evaluate the initial heuristic
        print(f"Initial tour length: {tour_length}")
        
        # Create root node regardless of tour_length value to ensure initialization succeeds
        root = HeuristicNode(algorithm=description, code=code, parent=None, action="i1", tour_length=tour_length)
        
        # Calculate and cache feature vector
        feature_vector = self.create_feature_vector(code, "i1")
        root.feature_vector = feature_vector  # Cache the feature vector
        
        print(f"Root node created: {root}")
        self.population.append(root)
        self.tour_lengths.append(tour_length)
        
        # Add to training data
        self.X.append(feature_vector)
        self.Y.append(tour_length)
        
        # Expand the root node to generate initial population
        self.expand(root)
        
        print(f"- Initialization Completed - Population size: {len(self.population)}")


    def fit(self):
        """
        Fit the Gaussian Process model to the collected data.
        """
        
        train_x = torch.tensor(np.vstack(self.X), dtype=torch.float32)
        train_y = torch.tensor(np.array(self.Y), dtype=torch.float32).unsqueeze(-1)
        model = SingleTaskGP(train_x, train_y, covar_module=ScaleKernel(RBFKernel()))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def acquisition_function(self, model, candidate_features):
        """
        TODO the input should be the five action generated heuristics
        Define the acquisition function for Bayesian optimization.
        """

        num_samples = 30 
        input_dim = candidate_features.shape[1]

        num_candidates = candidate_features.shape[0]
        if num_candidates > 0:
            indices = torch.randperm(num_candidates)[:min(num_samples, num_candidates)]
            optimal_inputs = candidate_features[indices]

            if len(optimal_inputs) < num_samples:
                repeat_times = (num_samples + len(optimal_inputs) - 1) // len(optimal_inputs)
                optimal_inputs = optimal_inputs.repeat(repeat_times, 1)[:num_samples]
        else:
            optimal_inputs = torch.randn((num_samples, input_dim), dtype=torch.float32)
        
        # Retry logic for qPredictiveEntropySearch with increasing jitter/threshold
        search_success = False
        jitter_vals = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        threshold_vals = [1e-2, 1e-1]
        for jitter in jitter_vals:
            for threshold in threshold_vals:
                try:
                    acq_func = qPredictiveEntropySearch(
                        model=model,
                        optimal_inputs=optimal_inputs,
                        maximize=False,
                        X_pending=None,
                        max_ep_iterations=50,
                        ep_jitter=jitter,
                        test_jitter=jitter,
                        threshold=threshold
                    )
                    best_candidate, _ = optimize_acqf_discrete(
                        acq_function=acq_func,
                        q=1,
                        choices=candidate_features,
                    )
                    for i, candidate in enumerate(candidate_features):
                        if torch.allclose(candidate, best_candidate.squeeze(), atol=1e-6):
                            search_success = True
                            return i
                except Exception as e:
                    print(f"qPredictiveEntropySearch failed with jitter={jitter}, threshold={threshold}: {e}")
                    continue
        if not search_success:
            print("All qPredictiveEntropySearch attempts failed. Falling back to basic acquisition function.")
            return self._fallback_acquisition(model, candidate_features)

        print("No matching candidate found.")
        logging.error("No matching candidate found.")
        return 0

    def _fallback_acquisition(self, model, candidate_features):
        """
        Fallback acquisition function that selects the best candidate based on the model's predictions.
        """
        from botorch.acquisition.monte_carlo import qExpectedImprovement
    
        # 计算当前最佳性能值
        if len(self.Y) > 0:
            best_f = torch.tensor(min(self.Y), dtype=torch.float32)
        else:
            best_f = torch.tensor(float('inf'), dtype=torch.float32)
        
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        
        acq_func = qLogExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=IdentityMCObjective(),
        )
        try:
            # 为每个候选点添加batch维度并评估
            candidates_with_batch = candidate_features.unsqueeze(1)
            acq_values = acq_func(candidates_with_batch)
            best_idx = torch.argmax(acq_values).item()
            return best_idx
        except Exception as e:
            print(f"Fallback acquisition also failed: {e}")
            return 0


    def expand(self, node: HeuristicNode):
        """
        Expand the node by generating new heuristics for each action, then batch evaluate for efficiency.
        """
        max_retries = 5
        ABS_MAX_ALLOW = 1e12

        for action in self.action_types:
            if action == 'i1':
                continue
            print(f"Generating heuristic for action: {action}")

            candidates = []
            # 1. 批量生成候选 code/algorithm
            for retry in range(max_retries):
                try:
                    code, algorithm = self.bo_interface.generate_heuristic_by_action(action)
                    if self.check_duplicate_code(self.population, code):
                        print(f"Duplicate code detected for action {action}, retry {retry + 1}")
                        continue
                    candidates.append({'action': action, 'code': code, 'algorithm': algorithm})
                    break  # 只要生成一个有效 code 就跳出 retry
                except Exception as e:
                    print(f"Error generating heuristic for action {action}, retry {retry + 1}: {e}")
                    continue
            if not candidates:
                print(f"Failed to generate candidate for action {action} after {max_retries} retries")
                continue

            # 2. 批量评估所有 code
            codes = [c['code'] for c in candidates]
            try:
                tour_lengths = self.problem.batch_evaluate(codes, 0)
            except Exception as e:
                print(f"Batch evaluate failed for action {action}: {e}")
                continue

            # 3. 检查每个候选的有效性
            for idx, candidate in enumerate(candidates):
                code = candidate['code']
                algorithm = candidate['algorithm']
                act = candidate['action']
                tour_length = tour_lengths[idx]

                # validate if the tour_length is valid
                try:
                    val = float(tour_length)
                except (TypeError, ValueError):
                    print(f"Invalid tour_length type for action {act}")
                    continue
                if (not math.isfinite(val)) or val <= 0 or val > ABS_MAX_ALLOW:
                    print(f"Invalid tour_length value {val} for action {act}")
                    continue
                if self.check_duplicate_obj(self.population, tour_length):
                    print(f"Duplicate objective {tour_length} detected for action {act}")
                    continue

                # 成功生成有效且不重复的启发函数
                child = HeuristicNode(algorithm=algorithm, code=code, parent=node, action=act, tour_length=tour_length)
                feature_vector = self.create_feature_vector(code, act)
                child.feature_vector = feature_vector
                node.add_child(child)
                self.population.append(child)
                self.tour_lengths.append(tour_length)
                self.X.append(feature_vector)
                self.Y.append(tour_length)
                print(f"Successfully generated heuristic for action {act}, tour_length: {tour_length}")


    def check_duplicate_obj(self, population, obj):
        for ind in population:
            if obj == ind.tour_length:
                return True
        return False

    def check_duplicate_code(self, population, code):
        for ind in population:
            if code == ind.code:
                return True
        return False
    
    def _get_node_feature_vector(self, node: HeuristicNode):
        if node.feature_vector is None:
            node.feature_vector = self.create_feature_vector(node.code, node.action)
        return node.feature_vector

    def run(self):
        import json
            
        self.initialize()  # Initialize the optimizer with the first heuristic

        # === Bayesian Optimization Main Loop ===
        for iteration in range(self.n_iters):
            print(f"\n=== Baeysian Optimization Iteration {iteration + 1}/{self.n_iters} ===")
            
            # 1. Fit GP model with current population
            if len(self.X) > 0:
                model = self.fit()
                print(f"GP Model Fitted with Data: {len(self.X)}")
            else:
                print("Inadquate Data")
                continue
            
            # 2. Prepare candidate set
            candidate_features = []
            valid_nodes = []
            
            for node in self.population:
                if node.tour_length is not None and node.tour_length != float('inf'):
                    feature_vector = self._get_node_feature_vector(node)
                    candidate_features.append(torch.tensor(feature_vector, dtype=torch.float32))
                    valid_nodes.append(node)
            
            if len(candidate_features) == 0:
                print("No Candidate Set Found, Break")
                break
                
            candidate_features = torch.stack(candidate_features)
            print(f"Candidate Set Size: {candidate_features.shape}")
            
            # 3. Use acquisition function to select the next node
            try:
                best_idx = self.acquisition_function(model, candidate_features)
                selected_node = valid_nodes[best_idx]
                print(f"Selected Node: {selected_node.algorithm[:50]}..., Performance: {selected_node.tour_length}")
            except Exception as e:
                print(f"Acquisition Function Excution Failed: {e}")
                # Fallback to basic strategy
                best_node = min(valid_nodes, key=lambda x: x.tour_length)
                selected_node = best_node
                print(f"Fallback To Basic Strategy: {selected_node.tour_length}")
            
            # 4. Expand the selected node
            old_population_size = len(self.population)
            self.expand(selected_node)
            new_nodes = len(self.population) - old_population_size
            print(f"New Heuristic Node: {new_nodes}")
            
            # 5. Output current best performance
            if self.Y:
                current_best = min(self.Y)
                print(f"Current Best Performance: {current_best}")
            
            # 6. Save population and best individual (following MCTS logic)
            self.count = iteration + 1
            
            # Convert population to serializable format
            population_data = []
            for node in self.population:
                node_data = {
                    "algorithm": node.algorithm,
                    "code": node.code,
                    "objective": node.tour_length,
                    "action": node.action
                }
                population_data.append(node_data)
            
            # Save population to a file
            filename = self.output_path + "population_generation_" + str(self.count) + ".json"
            with open(filename, 'w') as f:
                json.dump(population_data, f, indent=5)

            # Find and save the best individual
            best_node = min(self.population, key=lambda x: x.tour_length if x.tour_length is not None else float('inf'))
            best_data = {
                "algorithm": best_node.algorithm,
                "code": best_node.code,
                "objective": best_node.tour_length,
                "action": best_node.action
            }
            
            # Save the best one to a file
            best_filename = self.output_path + "best_population_generation_" + str(self.count) + ".json"
            with open(best_filename, 'w') as f:
                json.dump(best_data, f, indent=5)
        
        # === Return The Final Result (following MCTS format) ===
        if not self.population:
            return "", ""
        
        best_node = min(self.population, key=lambda x: x.tour_length if x.tour_length is not None else float('inf'))
        
        print(f"\n=== Optimization Complete ===")
        print(f"Best Performance: {best_node.tour_length}")
        print(f"Total Evaluations: {len(self.Y)}")
        print(f"Population Size: {len(self.population)}")

        # Return best code and the filename of the last saved best individual (like MCTS)
        final_filename = self.output_path + "best_population_generation_" + str(self.count) + ".json"
        return best_node.code, final_filename


    def save_results(self, results):
        """
        Save the results of the Bayesian optimization.
        """
        output_path = Path(self.problem.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save the best code and its path
        self.count += 1
        best_code_path = output_path / f"best_code{self.count}.py"
        with open(best_code_path, 'w') as f:
            f.write(results['best_code'])
        
        logging.info(f"Best code saved to {best_code_path}")
        return results['best_code'], str(best_code_path)
