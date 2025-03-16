import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import LlamaForCausalLM, LlamaTokenizer
import umap
import networkx as nx
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import copy
import os
import json
import re
import warnings
import time
import multiprocessing
from joblib import Parallel, delayed
from umap import UMAP
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import pearsonr, spearmanr, ttest_ind
import heapq
from functools import lru_cache
from collections import defaultdict, Counter


def convert_numpy_types(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                         np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
        return complex(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.void):
        return None
    elif isinstance(obj, dict):
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(x) for x in obj]
    return obj


class ActivationCache:
    def __init__(self, max_cache_size=100):
        self.activation_data = {}
        self.max_cache_size = max_cache_size
        self.access_count = {}
        
    def store(self, key, tensor, detach=True, cpu=True, to_numpy=False):
        if len(self.activation_data) >= self.max_cache_size:
            lru_key = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.activation_data[lru_key]
            del self.access_count[lru_key]
            
        if detach:
            tensor = tensor.detach()
        if cpu:
            tensor = tensor.cpu()
        if to_numpy:
            tensor = tensor.numpy()
            
        self.activation_data[key] = tensor
        self.access_count[key] = 0
        
    def get(self, key, default=None):
        if key in self.activation_data:
            self.access_count[key] += 1
            return self.activation_data[key]
        return default
    
    def clear(self):
        self.activation_data.clear()
        self.access_count.clear()
        
    def items(self):
        for key, value in self.activation_data.items():
            self.access_count[key] += 1
            yield key, value
            
    def keys(self):
        return self.activation_data.keys()
    
    def __len__(self):
        return len(self.activation_data)
    
    def __contains__(self, key):
        return key in self.activation_data


class LlamaActivationHook:
    def __init__(self, model, selective_capture=True):
        self.model = model
        self.activation_cache = ActivationCache()
        self.hooks = []
        self.selective_capture = selective_capture
        self.capture_layers = set()
        self._setup_hooks()
        
    def set_capture_layers(self, layer_indices):
        self.capture_layers = set(layer_indices)
        
    def _setup_hooks(self):
        self.remove_hooks()
        
        for layer_idx, layer in enumerate(self.model.model.layers):
            self.hooks.append(layer.register_forward_hook(
                lambda mod, inp, out, layer_idx=layer_idx: self._maybe_cache_activation(
                    f"layer_input_{layer_idx}", inp[0], layer_idx
                )
            ))
            
            self.hooks.append(layer.self_attn.register_forward_hook(
                lambda mod, inp, out, layer_idx=layer_idx: self._maybe_cache_activation(
                    f"attn_output_{layer_idx}", out[0], layer_idx
                )
            ))
            
            self.hooks.append(layer.register_forward_hook(
                lambda mod, inp, out, layer_idx=layer_idx: self._maybe_cache_activation(
                    f"residual_output_{layer_idx}", inp[0] + out[0], layer_idx
                )
            ))
            
            self.hooks.append(layer.mlp.register_forward_hook(
                lambda mod, inp, out, layer_idx=layer_idx: self._maybe_cache_activation(
                    f"ffn_output_{layer_idx}", out[0], layer_idx
                )
            ))
            
            self.hooks.append(layer.mlp.register_forward_hook(
                lambda mod, inp, out, layer_idx=layer_idx: self._cache_coefficient_scores(
                    layer_idx, mod, inp[0]
                )
            ))
    
    def _maybe_cache_activation(self, key, tensor, layer_idx):
        if not self.selective_capture or layer_idx in self.capture_layers:
            self.activation_cache.store(key, tensor)
    
    def _cache_coefficient_scores(self, layer_idx, module, tensor):
        if not self.selective_capture or layer_idx in self.capture_layers:
            with torch.no_grad():
                gate_out = module.act_fn(module.gate_proj(tensor))
                up_out = module.up_proj(tensor)
                coeffs = gate_out * up_out
                self.activation_cache.store(f"coefficient_scores_{layer_idx}", coeffs)
    
    def clear_cache(self):
        self.activation_cache.clear()
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def sanitize_filename(name):
    sanitized = re.sub(r'[\\/*?:"<>|=]', '_', name)
    sanitized = re.sub(r'[^\w\-_.]', '_', sanitized)
    return sanitized


class NeuronRanker:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.neuron_scores = defaultdict(float)
        self.neuron_metadata = {}
        
    def score_neurons_by_activation_diff(self, original_acts, modified_acts, position=-1, top_k=100):
        important_neurons = []
        
        for key in original_acts.keys():
            if not key.startswith("coefficient_scores_"):
                continue
                
            layer_idx = int(key.split("_")[-1])
            
            orig_tensor = original_acts.get(key)
            mod_tensor = modified_acts.get(key)
            
            if orig_tensor is None or mod_tensor is None:
                continue
                
            orig_activations = orig_tensor[0, position].cpu().numpy()
            mod_activations = mod_tensor[0, position].cpu().numpy()
            
            activation_diffs = np.abs(orig_activations - mod_activations)
            
            if len(activation_diffs) > 0:
                top_indices = np.argsort(activation_diffs)[::-1][:top_k]
                
                for idx in top_indices:
                    neuron_id = (layer_idx, int(idx))
                    score = float(activation_diffs[idx])
                    
                    if score > 1e-4:
                        important_neurons.append({
                            "layer": layer_idx,
                            "neuron": int(idx),
                            "score": score,
                            "orig_activation": float(orig_activations[idx]),
                            "mod_activation": float(mod_activations[idx])
                        })
                        
                        self.neuron_scores[(layer_idx, int(idx))] += score
        
        return sorted(important_neurons, key=lambda x: x["score"], reverse=True)
    
    def score_neurons_by_attribution(self, attribution_graphs, threshold=0.5, weight_by_layer=True):
        important_neurons = []
        
        # Fixed: Create layer_weights for all possible layers (0-31)
        # This ensures we have weights for any layer that might appear in the attribution graphs
        max_layer = 31  # For Llama models with 32 layers (0-31)
        layer_weights = {i: 0.5 + 0.5 * (i / max_layer) for i in range(max_layer + 1)} if weight_by_layer else {i: 1.0 for i in range(max_layer + 1)}
        
        for graph_name, graph in attribution_graphs.items():
            for node, data in graph.nodes(data=True):
                if "coefficient_scores" in node and data.get("attribution_score", 0) > threshold:
                    layer = int(node.split("_")[-1])
                    
                    incoming_edges = graph.in_edges(node, data=True)
                    edge_weight_sum = sum(edge_data.get("weight", 0) for _, _, edge_data in incoming_edges)
                    
                    weighted_score = data.get("attribution_score", 0) * layer_weights[layer]
                    
                    for neuron_idx in range(min(100, self.model.config.intermediate_size)):
                        neuron_id = (layer, neuron_idx)
                        
                        score = weighted_score * (1.0 + 0.5 * edge_weight_sum)
                        self.neuron_scores[neuron_id] += score
                        
                        if score > threshold:
                            important_neurons.append({
                                "layer": layer,
                                "neuron": neuron_idx,
                                "score": score,
                                "attribution": data.get("attribution_score", 0),
                                "edge_weight_sum": edge_weight_sum
                            })
        
        return sorted(important_neurons, key=lambda x: x["score"], reverse=True)
    
    def get_top_neurons(self, k=50, min_score=0.0):
        scored_neurons = [(layer, neuron, score) for (layer, neuron), score in self.neuron_scores.items() if score > min_score]
        return sorted(scored_neurons, key=lambda x: x[2], reverse=True)[:k]
    
    def add_neuron_interpretations(self, top_neurons, model, tokenizer, top_k_tokens=5):
        for layer, neuron, _ in top_neurons:
            if (layer, neuron) not in self.neuron_metadata:
                tokens = self._project_neuron_to_vocab(model, layer, neuron, tokenizer, top_k=top_k_tokens)
                self.neuron_metadata[(layer, neuron)] = {
                    "top_tokens": tokens
                }
    
    def _project_neuron_to_vocab(self, model, layer_idx, neuron_idx, tokenizer, top_k=5):
        with torch.no_grad():
            fc2_weights = model.model.layers[layer_idx].mlp.down_proj.weight.data
            fc2_vector = fc2_weights[:, neuron_idx]
            
            final_layernorm = model.model.norm.weight.data
            lm_head = model.lm_head.weight.data
            
            fc2_vector = fc2_vector * torch.rsqrt(torch.tensor(1e-6)) * final_layernorm
            logits = torch.matmul(fc2_vector, lm_head.t())
            
            top_token_indices = torch.argsort(logits, descending=True)[:top_k].cpu().numpy()
            top_tokens = [tokenizer.decode([idx]) for idx in top_token_indices]
            
            return top_tokens
    
    def cluster_neurons_by_similarity(self, model, n_clusters=5):
        top_neurons = self.get_top_neurons(k=min(100, len(self.neuron_scores)))
        
        if len(top_neurons) < n_clusters:
            return {}
            
        neuron_vectors = []
        neuron_ids = []
        
        with torch.no_grad():
            for layer, neuron, _ in top_neurons:
                fc2_weights = model.model.layers[layer].mlp.down_proj.weight.data
                fc2_vector = fc2_weights[:, neuron].cpu().numpy()
                neuron_vectors.append(fc2_vector)
                neuron_ids.append((layer, neuron))
        
        neuron_array = np.vstack(neuron_vectors)
        similarities = cosine_similarity(neuron_array)
        
        clustering = AgglomerativeClustering(
            n_clusters=min(n_clusters, len(neuron_ids)),
            affinity='precomputed',
            linkage='average'
        ).fit(1 - similarities)
        
        clusters = defaultdict(list)
        for i, (layer, neuron) in enumerate(neuron_ids):
            cluster_id = int(clustering.labels_[i])
            neuron_info = {
                "layer": layer,
                "neuron": neuron,
                "score": self.neuron_scores[(layer, neuron)],
                "top_tokens": self.neuron_metadata.get((layer, neuron), {}).get("top_tokens", [])
            }
            clusters[cluster_id].append(neuron_info)
        
        for cluster_id in clusters:
            clusters[cluster_id] = sorted(clusters[cluster_id], key=lambda x: x["score"], reverse=True)
        
        return dict(clusters)


class ComparativeNeuronAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.activation_hook = LlamaActivationHook(model)
        self.neuron_ranker = NeuronRanker(model, tokenizer)
        self.important_neurons = {}
        self.neuron_interpretations = {}
        self.timing_info = {}
        self.num_layers = len(self.model.model.layers)
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.num_heads
    
    def analyze_arithmetic_head(self, input_text, target_head=(17, 22), top_k_neurons=30, only_last_position=True):
        start_time = time.time()
        
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.model.device)
        
        if only_last_position:
            deep_layers = list(range(max(0, target_head[0]-2), min(self.num_layers, target_head[0]+3)))
            self.activation_hook.set_capture_layers(deep_layers)
        
        self.activation_hook.clear_cache()
        with torch.no_grad():
            original_outputs = self.model(input_ids)
        original_logits = original_outputs.logits[0, -1, :].cpu()
        original_probs = torch.softmax(original_logits, dim=0)
        original_activations = copy.deepcopy(self.activation_hook.activation_cache.activation_data)
        
        layer_idx, head_idx = target_head
        head_size = self.model.config.hidden_size // self.model.config.num_attention_heads
        head_start = head_idx * head_size
        head_end = head_start + head_size
        
        with torch.no_grad():
            original_weights = self.model.model.layers[layer_idx].self_attn.o_proj.weight[:, head_start:head_end].clone()
            
            self.model.model.layers[layer_idx].self_attn.o_proj.weight[:, head_start:head_end] = 0
        
            self.activation_hook.clear_cache()
            modified_outputs = self.model(input_ids)
            
            self.model.model.layers[layer_idx].self_attn.o_proj.weight[:, head_start:head_end] = original_weights
        
        modified_logits = modified_outputs.logits[0, -1, :].cpu()
        modified_probs = torch.softmax(modified_logits, dim=0)
        modified_activations = copy.deepcopy(self.activation_hook.activation_cache.activation_data)
        
        important_neurons = self.neuron_ranker.score_neurons_by_activation_diff(
            original_activations, 
            modified_activations,
            position=-1,
            top_k=top_k_neurons
        )
        
        interpretations = {}
        for neuron_info in important_neurons[:top_k_neurons]:
            layer = neuron_info["layer"]
            neuron = neuron_info["neuron"]
            
            top_tokens = self._project_neuron_to_vocab(layer, neuron, top_k=5)
            interpretations[f"layer_{layer}_neuron_{neuron}"] = {
                "tokens": top_tokens,
                "score": neuron_info["score"],
                "activation_diff": neuron_info["orig_activation"] - neuron_info["mod_activation"]
            }
        
        top_k_outputs = 5
        original_top_indices = torch.argsort(original_probs, descending=True)[:top_k_outputs].cpu().numpy()
        modified_top_indices = torch.argsort(modified_probs, descending=True)[:top_k_outputs].cpu().numpy()
        
        original_top_tokens = [(self.tokenizer.decode([idx]), original_probs[idx].item()) 
                              for idx in original_top_indices]
        modified_top_tokens = [(self.tokenizer.decode([idx]), modified_probs[idx].item()) 
                              for idx in modified_top_indices]
        
        self.timing_info["cna_analysis_time"] = time.time() - start_time
        
        self.important_neurons[input_text] = important_neurons
        self.neuron_interpretations[input_text] = interpretations
        
        cna_result = {
            "input_text": input_text,
            "target_head": target_head,
            "original_predictions": original_top_tokens,
            "modified_predictions": modified_top_tokens,
            "probability_change": {
                self.tokenizer.decode([idx]): (modified_probs[idx].item() - original_probs[idx].item()) 
                for idx in original_top_indices
            },
            "important_neurons": important_neurons[:top_k_neurons],
            "neuron_interpretations": interpretations,
            "analysis_time": self.timing_info["cna_analysis_time"]
        }
        
        return cna_result
    
    def _project_neuron_to_vocab(self, layer_idx, neuron_idx, top_k=5):
        with torch.no_grad():
            fc2_weights = self.model.model.layers[layer_idx].mlp.down_proj.weight.data
            fc2_vector = fc2_weights[:, neuron_idx]
            
            final_layernorm = self.model.model.norm.weight.data
            lm_head = self.model.lm_head.weight.data
            
            fc2_vector = fc2_vector * torch.rsqrt(torch.tensor(1e-6)) * final_layernorm
            logits = torch.matmul(fc2_vector, lm_head.t())
            
            top_token_indices = torch.argsort(logits, descending=True)[:top_k].cpu().numpy()
            top_tokens = [self.tokenizer.decode([idx]) for idx in top_token_indices]
            
            return top_tokens


class EnhancedContrastiveAttributionMapper:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", cache_dir=None, dtype=torch.float16):
        print(f"Loading model: {model_name}")
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = LlamaForCausalLM.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            torch_dtype=dtype,
            device_map="auto"
        )
        self.model.eval()
        
        self.activation_hook = LlamaActivationHook(self.model, selective_capture=True)
        
        self.neuron_ranker = NeuronRanker(self.model, self.tokenizer)
        
        self.manifolds = {}
        self.attribution_graphs = {}
        self.concept_clusters = {}
        self.timing_info = {}
        self.example_cache = {}
        
        self.num_layers = len(self.model.model.layers)
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.num_heads
        print(f"Model loaded: {self.num_layers} layers, {self.num_heads} heads")
        
        self.cna_analyzer = ComparativeNeuronAnalyzer(self.model, self.tokenizer)
    
    def generate_contrastive_set(self, base_inputs, dimensions, max_variations=5):
        start_time = time.time()
        contrastive_sets = {"base": base_inputs}
        
        limited_dimensions = {}
        for dim_name, variations in dimensions.items():
            if len(variations) > max_variations:
                if isinstance(variations[0], int) and all(isinstance(x, int) for x in variations):
                    samples = [min(variations), max(variations)]
                    mid_samples = np.linspace(min(variations), max(variations), max_variations-2)
                    samples.extend([int(x) for x in mid_samples if int(x) not in samples])
                    limited_dimensions[dim_name] = samples[:max_variations]
                else:
                    limited_dimensions[dim_name] = variations[:max_variations]
            else:
                limited_dimensions[dim_name] = variations
        
        for dim_name, variations in limited_dimensions.items():
            for variation in variations:
                variation_name = f"{dim_name}_{variation}"
                contrastive_sets[variation_name] = []
                
                for base_input in base_inputs:
                    cache_key = (base_input, dim_name, str(variation))
                    if cache_key in self.example_cache:
                        modified_input = self.example_cache[cache_key]
                    else:
                        modified_input = self._apply_variation(base_input, dim_name, variation)
                        self.example_cache[cache_key] = modified_input
                    
                    contrastive_sets[variation_name].append(modified_input)
        
        self.timing_info["contrastive_set_generation_time"] = time.time() - start_time
        
        print(f"Generated {len(contrastive_sets)} contrastive sets:")
        for name, examples in contrastive_sets.items():
            print(f"  {name}: {examples[:2]}{'...' if len(examples) > 2 else ''}")
            
        return contrastive_sets
    
    def _apply_variation(self, input_text, dimension, variation):
        if dimension == "operation":
            for op in ["+", "-", "*", "/"]:
                if op in input_text:
                    return input_text.replace(op, variation)
            
            op_mapping = {
                "plus": "plus", "minus": "minus", 
                "times": "times", "divided by": "divided by"
            }
            for text_op, replacement in zip(op_mapping.keys(), op_mapping.values()):
                if text_op in input_text:
                    return input_text.replace(text_op, variation)     
                    
        elif dimension == "first_digit":
            if "+" in input_text or "-" in input_text or "*" in input_text or "/" in input_text:
                parts = input_text.replace("=", "").split("+")
                if len(parts) != 2:
                    parts = input_text.replace("=", "").split("-")
                if len(parts) != 2:
                    parts = input_text.replace("=", "").split("*")
                if len(parts) != 2:
                    parts = input_text.replace("=", "").split("/")              
                if len(parts) == 2:
                    return input_text.replace(parts[0], str(variation))
            
            num_words = ["zero", "one", "two", "three", "four", "five", 
                          "six", "seven", "eight", "nine"]
            for i, word in enumerate(num_words):
                if input_text.startswith(word):
                    replacement = str(variation) if isinstance(variation, int) else variation
                    if isinstance(variation, int) and variation < 10:
                        replacement = num_words[variation]
                    return input_text.replace(word, replacement, 1)
        
        elif dimension == "second_digit":
            if "+" in input_text or "-" in input_text or "*" in input_text or "/" in input_text:
                parts = input_text.replace("=", "").split("+")
                if len(parts) != 2:
                    parts = input_text.replace("=", "").split("-")
                if len(parts) != 2:
                    parts = input_text.replace("=", "").split("*")
                if len(parts) != 2:
                    parts = input_text.replace("=", "").split("/")
                
                if len(parts) == 2:
                    return input_text.replace(parts[1], str(variation))
            
            num_words = ["zero", "one", "two", "three", "four", "five", 
                          "six", "seven", "eight", "nine"]
            
            for op in ["plus", "minus", "times", "divided by"]:
                if op in input_text:
                    parts = input_text.split(op)
                    if len(parts) == 2:
                        for i, word in enumerate(num_words):
                            if word in parts[1]:
                                replacement = str(variation) if isinstance(variation, int) else variation
                                if isinstance(variation, int) and variation < 10:
                                    replacement = num_words[variation]
                                return input_text.replace(word, replacement, 1)
                    
        elif dimension == "format":
            if variation == "word":
                mapping = {
                    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
                    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
                    "+": " plus ", "-": " minus ", "*": " times ", "/": " divided by ",
                    "=": " equals "
                }
                result = input_text
                for digit, word in mapping.items():
                    result = result.replace(digit, word)
                return result
            
            elif variation == "digit":
                mapping = {
                    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
                    "plus": "+", "minus": "-", "times": "*", "divided by": "/",
                    "equals": "="
                }
                result = input_text
                for word, digit in mapping.items():
                    result = result.replace(word, digit)
                return result
        
        print(f"Warning: Could not apply variation '{variation}' to dimension '{dimension}' for input '{input_text}'")
        return input_text
    
    def collect_activations(self, inputs, focus_layers=None):
        activations = {}
        
        if focus_layers is not None:
            self.activation_hook.set_capture_layers(focus_layers)
        
        for input_text in tqdm(inputs, desc="Collecting activations"):
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.model.device)
            
            self.activation_hook.clear_cache()
            
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            activations[input_text] = copy.deepcopy(self.activation_hook.activation_cache.activation_data)
        
        return activations
    
    def compute_activation_manifolds(self, contrastive_sets, layer_indices=None):
        start_time = time.time()
        
        if layer_indices is None:
            layer_indices = np.linspace(0, self.num_layers-1, min(8, self.num_layers)).astype(int).tolist()
        
        print(f"Computing activation manifolds for {len(contrastive_sets)} contrastive sets")
        print(f"Using layers: {layer_indices}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        
            all_manifolds = {}
            
            self.activation_hook.set_capture_layers(layer_indices)
            
            for set_name, inputs in contrastive_sets.items():
                print(f"Processing set: {set_name}")
                
                activations = self.collect_activations(inputs, focus_layers=layer_indices)
                
                layer_activations = {}
                for layer_idx in layer_indices:
                    layer_activations[f"layer_{layer_idx}"] = []
                    
                    for input_text, input_activations in activations.items():
                        layer_output = input_activations.get(f"layer_input_{layer_idx}", None)
                        if layer_output is not None:
                            last_pos_activation = layer_output[0, -1, :].cpu().numpy().astype(np.float64)
                            
                            last_pos_activation = np.nan_to_num(last_pos_activation, nan=0.0, posinf=1e5, neginf=-1e5)
                            clip_threshold = 1e6
                            last_pos_activation = np.clip(last_pos_activation, -clip_threshold, clip_threshold)
                            last_pos_activation = np.sign(last_pos_activation) * np.log1p(np.abs(last_pos_activation))
                            
                            layer_activations[f"layer_{layer_idx}"].append(last_pos_activation)
                
                manifolds = {}
                for layer_name, layer_acts in layer_activations.items():
                    if len(layer_acts) > 2:
                        try:
                            stacked_acts = np.stack(layer_acts)
                            
                            epsilon = 1e-4
                            stacked_acts += np.random.normal(0, epsilon, stacked_acts.shape)
                            
                            stacked_acts = stacked_acts - np.mean(stacked_acts, axis=0, keepdims=True)
                            
                            min_samples = min(len(stacked_acts), len(stacked_acts[0]))
                            
                            if min_samples > 5:
                                try:
                                    n_components = min(min(50, stacked_acts.shape[0] - 1), stacked_acts.shape[1])
                                    pca = PCA(n_components=n_components, svd_solver='full', whiten=True)
                                    reduced_acts = pca.fit_transform(stacked_acts)
                                except Exception as e:
                                    print(f"Full PCA failed for {layer_name}, trying with randomized PCA: {e}")
                                    pca = PCA(n_components=min(min(10, stacked_acts.shape[0] - 1), stacked_acts.shape[1]), 
                                            svd_solver='randomized', 
                                            whiten=True,
                                            random_state=42)
                                    reduced_acts = pca.fit_transform(stacked_acts)
                                
                                try:
                                    n_neighbors = min(8, stacked_acts.shape[0] - 1)
                                    
                                    with warnings.catch_warnings():
                                        warnings.filterwarnings("ignore")
                                        umap_reducer = UMAP(
                                            n_components=2,
                                            n_neighbors=n_neighbors,
                                            min_dist=0.3,
                                            metric='euclidean',
                                            random_state=42
                                        )
                                        manifold = umap_reducer.fit_transform(reduced_acts)
                                    manifolds[layer_name] = manifold
                                except Exception as e:
                                    print(f"UMAP failed for {layer_name}, falling back to t-SNE: {e}")
                                    from sklearn.manifold import TSNE
                                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, stacked_acts.shape[0]//2))
                                    manifolds[layer_name] = tsne.fit_transform(reduced_acts)
                            else:
                                pca_2d = PCA(n_components=min(2, stacked_acts.shape[0] - 1))
                                manifolds[layer_name] = pca_2d.fit_transform(stacked_acts)
                        except Exception as e:
                            print(f"ERROR: Failed to compute manifold for {layer_name} in set {set_name}: {e}")
                            manifolds[layer_name] = np.zeros((len(layer_acts), 2))
                    elif len(layer_acts) == 2:
                        manifolds[layer_name] = np.array([[0, 0], [1, 0]])
                    elif len(layer_acts) == 1:
                        manifolds[layer_name] = np.array([[0, 0]])
                    else:
                        manifolds[layer_name] = np.zeros((0, 2))
                    
                all_manifolds[set_name] = manifolds
        
        self.manifolds = all_manifolds
        self.timing_info["manifold_computation_time"] = time.time() - start_time
        
        print("Activation manifolds computed successfully")
        return all_manifolds
    
    def construct_attribution_graphs(self, contrastive_sets, reference_set="base", focus_layers=None):
        start_time = time.time()
        
        if reference_set not in contrastive_sets:
            raise ValueError(f"Reference set '{reference_set}' not found in contrastive sets")
        
        if focus_layers is None:
            focus_layers = np.linspace(0, self.num_layers-1, min(8, self.num_layers)).astype(int).tolist()
        
        self.activation_hook.set_capture_layers(focus_layers)
        
        attribution_graphs = {}
        
        for set_name, inputs in contrastive_sets.items():
            if set_name == reference_set:
                continue
            
            print(f"Constructing attribution graph for: {set_name} vs {reference_set}")
            
            G = nx.DiGraph()
            
            test_input = inputs[0]
            reference_input = contrastive_sets[reference_set][0]
            
            test_acts = self.collect_activations([test_input], focus_layers=focus_layers)[test_input]
            ref_acts = self.collect_activations([reference_input], focus_layers=focus_layers)[reference_input]
            
            for layer_idx in focus_layers:
                G.add_node(f"layer_input_{layer_idx}", type="layer_input", layer=layer_idx)
                G.add_node(f"attn_output_{layer_idx}", type="attn_output", layer=layer_idx)
                G.add_node(f"residual_output_{layer_idx}", type="residual_output", layer=layer_idx)
                G.add_node(f"ffn_output_{layer_idx}", type="ffn_output", layer=layer_idx)
                G.add_node(f"coefficient_scores_{layer_idx}", type="coefficient_scores", layer=layer_idx)
                
                for act_type in ["layer_input", "attn_output", "residual_output", "ffn_output", "coefficient_scores"]:
                    node_name = f"{act_type}_{layer_idx}"
                    
                    test_tensor = test_acts.get(node_name, None)
                    ref_tensor = ref_acts.get(node_name, None)
                    
                    if test_tensor is not None and ref_tensor is not None:
                        diff = (test_tensor[0, -1] - ref_tensor[0, -1]).pow(2).sum().sqrt().item()
                        G.nodes[node_name]["attribution_score"] = diff
                    else:
                        G.nodes[node_name]["attribution_score"] = 0.0
            
            for layer_idx in focus_layers:
                G.add_edge(f"layer_input_{layer_idx}", f"attn_output_{layer_idx}")
                G.add_edge(f"layer_input_{layer_idx}", f"residual_output_{layer_idx}")
                G.add_edge(f"attn_output_{layer_idx}", f"residual_output_{layer_idx}")
                G.add_edge(f"residual_output_{layer_idx}", f"coefficient_scores_{layer_idx}")
                G.add_edge(f"coefficient_scores_{layer_idx}", f"ffn_output_{layer_idx}")
                G.add_edge(f"residual_output_{layer_idx}", f"ffn_output_{layer_idx}")
                
                if layer_idx < max(focus_layers) and layer_idx + 1 in focus_layers:
                    G.add_edge(f"ffn_output_{layer_idx}", f"layer_input_{layer_idx + 1}")
            
            for u, v in G.edges():
                source_score = G.nodes[u].get("attribution_score", 0.0)
                target_score = G.nodes[v].get("attribution_score", 0.0)
                edge_weight = abs(target_score - source_score)
                G[u][v]["weight"] = edge_weight
            
            attribution_graphs[set_name] = G
        
        self.attribution_graphs = attribution_graphs
        self.timing_info["attribution_graph_time"] = time.time() - start_time
        
        neuron_importance = self.neuron_ranker.score_neurons_by_attribution(attribution_graphs)
        
        print("Attribution graphs constructed successfully")
        return attribution_graphs
    
    def identify_concept_clusters(self, contrastive_sets, n_clusters=5, focus_layers=None):
        start_time = time.time()
        
        print("Identifying concept clusters in activation space")
        
        if focus_layers is None:
            focus_layers = list(range(max(0, self.num_layers-8), self.num_layers))
        
        self.activation_hook.set_capture_layers(focus_layers)
        
        concept_clusters = {}
        
        for set_name, inputs in contrastive_sets.items():
            print(f"Finding clusters for set: {set_name}")
            
            activations = self.collect_activations(inputs, focus_layers=focus_layers)
            
            layer_clusters = {}
            
            for layer_idx in focus_layers:
                ffn_acts = []
                input_mapping = []
                
                for i, (input_text, input_activations) in enumerate(activations.items()):
                    coef_scores = input_activations.get(f"coefficient_scores_{layer_idx}", None)
                    if coef_scores is not None:
                        last_pos_activation = coef_scores[0, -1, :].cpu().numpy()
                        ffn_acts.append(last_pos_activation)
                        input_mapping.append(i)
                
                if len(ffn_acts) > 0:
                    stacked_acts = np.stack(ffn_acts)
                    
                    try:
                        if stacked_acts.shape[0] > 10 and stacked_acts.shape[1] > 100:
                            n_components = min(stacked_acts.shape[0] - 1, 50)
                            pca = PCA(n_components=n_components)
                            reduced_acts = pca.fit_transform(stacked_acts)
                        else:
                            reduced_acts = stacked_acts
                            
                        clustering = DBSCAN(
                            eps=0.5, 
                            min_samples=max(2, len(reduced_acts) // 10)
                        ).fit(reduced_acts)
                        
                        layer_clusters[f"layer_{layer_idx}"] = {
                            "labels": clustering.labels_.tolist(),
                            "input_mapping": input_mapping,
                            "n_clusters": len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                        }
                        
                        neuron_importance_by_cluster = defaultdict(list)
                        for i, label in enumerate(clustering.labels_):
                            if label != -1:
                                input_idx = input_mapping[i]
                                input_text = inputs[input_idx]
                                
                                if f"coefficient_scores_{layer_idx}" in activations[input_text]:
                                    coeffs = activations[input_text][f"coefficient_scores_{layer_idx}"][0, -1]
                                    top_neuron_idxs = torch.topk(coeffs, k=5)[1].cpu().numpy()
                                    for neuron_idx in top_neuron_idxs:
                                        neuron_importance_by_cluster[label].append(int(neuron_idx))
                        
                        cluster_neurons = {}
                        for label, neurons in neuron_importance_by_cluster.items():
                            neuron_counts = Counter(neurons)
                            top_neurons = [n for n, _ in neuron_counts.most_common(5)]
                            cluster_neurons[label] = top_neurons
                            
                        layer_clusters[f"layer_{layer_idx}"]["cluster_neurons"] = cluster_neurons
                        
                    except Exception as e:
                        print(f"Error clustering layer {layer_idx}: {e}")
                        layer_clusters[f"layer_{layer_idx}"] = {
                            "labels": [-1] * len(ffn_acts),
                            "input_mapping": input_mapping,
                            "n_clusters": 0
                        }
            
            concept_clusters[set_name] = layer_clusters
        
        for set_name, layer_data in concept_clusters.items():
            for layer_name, cluster_data in layer_data.items():
                if "cluster_neurons" in cluster_data:
                    neuron_interpretations = {}
                    layer_idx = int(layer_name.split("_")[1])
                    
                    for cluster_id, neurons in cluster_data["cluster_neurons"].items():
                        neuron_interpretations[cluster_id] = {}
                        for neuron in neurons:
                            tokens = self._project_neuron_to_vocab(layer_idx, neuron, top_k=5)
                            neuron_interpretations[cluster_id][neuron] = tokens
                    
                    concept_clusters[set_name][layer_name]["neuron_interpretations"] = neuron_interpretations
        
        self.concept_clusters = concept_clusters
        self.timing_info["concept_clustering_time"] = time.time() - start_time
        
        print("Concept clusters identified successfully")
        return concept_clusters
    
    def _project_neuron_to_vocab(self, layer_idx, neuron_idx, top_k=5):
        with torch.no_grad():
            fc2_weights = self.model.model.layers[layer_idx].mlp.down_proj.weight.data
            fc2_vector = fc2_weights[:, neuron_idx]
            
            final_layernorm = self.model.model.norm.weight.data
            lm_head = self.model.lm_head.weight.data
            
            fc2_vector = fc2_vector * torch.rsqrt(torch.tensor(1e-6)) * final_layernorm
            logits = torch.matmul(fc2_vector, lm_head.t())
            
            top_token_indices = torch.argsort(logits, descending=True)[:top_k].cpu().numpy()
            top_tokens = [self.tokenizer.decode([idx]) for idx in top_token_indices]
            
            return top_tokens
    
    def analyze_with_hybrid_approach(self, input_text, target_head=(17, 22), 
                                     contrastive_dimensions=None, n_important_neurons=50):
        start_time = time.time()
        print(f"Starting hybrid analysis for input: {input_text}")
        
        results = {
            "input": input_text,
            "target_head": target_head,
            "timing": {},
            "important_neurons": [],
            "prediction_path": {},
            "manifolds": {},
            "cluster_analysis": {}
        }
        
        cna_start = time.time()
        cna_result = self.cna_analyzer.analyze_arithmetic_head(
            input_text, 
            target_head=target_head,
            top_k_neurons=n_important_neurons
        )
        cna_time = time.time() - cna_start
        results["timing"]["cna_analysis"] = cna_time
        
        cna_neurons = [(n["layer"], n["neuron"]) for n in cna_result["important_neurons"]]
        print(f"CNA identified {len(cna_neurons)} important neurons in {cna_time:.2f}s")
        
        cam_start = time.time()
        if contrastive_dimensions is None:
            contrastive_dimensions = {}
            
            for op, variations in [
                ("+", ["-", "*"]),
                ("-", ["+", "/"]),
                ("*", ["+", "/"]),
                ("/", ["-", "*"])
            ]:
                if op in input_text:
                    contrastive_dimensions["operation"] = variations
                    break
            
            if any(d in input_text for d in "0123456789"):
                digits = [c for c in input_text if c.isdigit()]
                if len(digits) >= 1:
                    first_digit = int(digits[0])
                    contrastive_dimensions["first_digit"] = [
                        (first_digit + 1) % 10,
                        (first_digit + 2) % 10
                    ]
                
                if len(digits) >= 2:
                    second_digit = int(digits[1])
                    contrastive_dimensions["second_digit"] = [
                        (second_digit + 1) % 10,
                        (second_digit + 2) % 10
                    ]
        
        base_input = [input_text]
        contrastive_sets = self.generate_contrastive_set(base_input, contrastive_dimensions)
        
        focus_layers = sorted(set(layer for layer, _ in cna_neurons))
        
        extended_focus_layers = set(focus_layers)
        for layer in focus_layers:
            if layer > 0:
                extended_focus_layers.add(layer - 1)
            if layer < self.num_layers - 1:
                extended_focus_layers.add(layer + 1)
        focus_layers = sorted(extended_focus_layers)
        
        manifolds = self.compute_activation_manifolds(contrastive_sets, layer_indices=focus_layers)
        attribution_graphs = self.construct_attribution_graphs(
            contrastive_sets, 
            focus_layers=focus_layers
        )
        
        cam_time = time.time() - cam_start
        results["timing"]["cam_analysis"] = cam_time
        
        merge_start = time.time()
        
        cam_neurons = []
        attribution_threshold = 0.5
        for graph_name, graph in attribution_graphs.items():
            for node, data in graph.nodes(data=True):
                if "coefficient_scores" in node and data.get("attribution_score", 0) > attribution_threshold:
                    layer = int(node.split("_")[-1])
                    for neuron_idx in range(min(100, self.model.config.intermediate_size)):
                        cam_neurons.append((layer, neuron_idx))
        
        cam_neurons = list(set(cam_neurons))
        
        overlap_neurons = set(cna_neurons).intersection(set(cam_neurons))
        all_neurons = list(set(cna_neurons).union(set(cam_neurons)))
        
        neuron_interpretations = {}
        for layer, neuron in all_neurons:
            tokens = self._project_neuron_to_vocab(layer, neuron, top_k=5)
            neuron_interpretations[(layer, neuron)] = tokens
        
        neuron_scores = {}
        for layer, neuron in all_neurons:
            score = 0
            
            if (layer, neuron) in cna_neurons:
                for i, n in enumerate(cna_result["important_neurons"]):
                    if n["layer"] == layer and n["neuron"] == neuron:
                        score += 5.0 * (1.0 - i / len(cna_result["important_neurons"]))
                        break
            
            for graph_name, graph in attribution_graphs.items():
                node_name = f"coefficient_scores_{layer}"
                if node_name in graph.nodes:
                    attr_score = graph.nodes[node_name].get("attribution_score", 0)
                    score += 2.0 * attr_score
            
            neuron_scores[(layer, neuron)] = score
        
        ranked_neurons = sorted(
            [(layer, neuron, neuron_scores.get((layer, neuron), 0)) 
             for layer, neuron in all_neurons],
            key=lambda x: x[2], 
            reverse=True
        )
        
        layer_neuron_clusters = {}
        for layer in set(layer for layer, _, _ in ranked_neurons):
            layer_neurons = [(neuron, score) for l, neuron, score in ranked_neurons if l == layer]
            layer_neurons.sort(key=lambda x: x[1], reverse=True)
            
            interpretations = {}
            for neuron, _ in layer_neurons[:10]:
                interpretations[neuron] = neuron_interpretations.get((layer, neuron), [])
            
            layer_neuron_clusters[layer] = {
                "neurons": [n for n, _ in layer_neurons[:10]],
                "scores": [s for _, s in layer_neurons[:10]],
                "interpretations": interpretations
            }
        
        merge_time = time.time() - merge_start
        results["timing"]["merge_analysis"] = merge_time
        
        results["important_neurons"] = [
            {"layer": l, "neuron": n, "score": s, "tokens": neuron_interpretations.get((l, n), [])}
            for l, n, s in ranked_neurons[:n_important_neurons]
        ]
        
        results["overlap_analysis"] = {
            "cna_count": len(cna_neurons),
            "cam_count": len(cam_neurons),
            "overlap_count": len(overlap_neurons),
            "jaccard_similarity": len(overlap_neurons) / len(set(cna_neurons).union(set(cam_neurons))) if all_neurons else 0,
            "in_both": [{"layer": l, "neuron": n} for l, n in overlap_neurons]
        }
        
        results["prediction_path"] = {
            "original_predictions": cna_result["original_predictions"],
            "modified_predictions": cna_result["modified_predictions"],
            "probability_changes": cna_result["probability_change"],
            "layer_clusters": layer_neuron_clusters
        }
        
        results["manifolds"] = {
            set_name: {layer: manifold.tolist() for layer, manifold in set_manifolds.items()}
            for set_name, set_manifolds in manifolds.items()
        }
        
        results["timing"]["total_time"] = time.time() - start_time
        
        print(f"Hybrid analysis completed in {results['timing']['total_time']:.2f}s")
        return results
    
    def visualize_results(self, hybrid_results, save_dir="hybrid_results"):
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        layers = []
        neurons = []
        scores = []
        colors = []
        
        for neuron_info in hybrid_results["important_neurons"][:30]:
            layers.append(neuron_info["layer"])
            neurons.append(neuron_info["neuron"])
            scores.append(neuron_info["score"] * 100)
            
            is_in_overlap = any(n["layer"] == neuron_info["layer"] and n["neuron"] == neuron_info["neuron"] 
                               for n in hybrid_results["overlap_analysis"]["in_both"])
            
            if is_in_overlap:
                colors.append("purple")
            elif neuron_info["score"] > 2.0:
                colors.append("blue")
            else:
                colors.append("red")
        
        plt.scatter(layers, neurons, s=scores, c=colors, alpha=0.7)
        
        plt.xlabel("Layer")
        plt.ylabel("Neuron Index")
        plt.title(f"Important Neurons for '{hybrid_results['input']}'")
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Both Methods'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='CNA'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='CAM')
        ]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "important_neurons.png"))
        plt.close()
        
        important_layers = sorted(set(n["layer"] for n in hybrid_results["important_neurons"][:10]))
        
        for layer in important_layers:
            plt.figure(figsize=(12, 8))
            
            for i, (set_name, manifolds) in enumerate(hybrid_results["manifolds"].items()):
                if f"layer_{layer}" in manifolds:
                    manifold = np.array(manifolds[f"layer_{layer}"])
                    if len(manifold) > 0:
                        plt.scatter(
                            manifold[:, 0], 
                            manifold[:, 1], 
                            label=set_name,
                            alpha=0.8
                        )
            
            plt.title(f"Activation Manifold for Layer {layer}")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"manifold_layer_{layer}.png"))
            plt.close()
        
        try:
            from wordcloud import WordCloud
            
            all_tokens = []
            for neuron in hybrid_results["important_neurons"][:20]:
                all_tokens.extend(neuron["tokens"])
            
            if all_tokens:
                text = " ".join(all_tokens)
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.title(f"Token Concepts for '{hybrid_results['input']}'")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "neuron_concepts.png"))
                plt.close()
        except ImportError:
            print("WordCloud package not available, skipping word cloud visualization")
        
        plt.figure(figsize=(10, 6))
        
        orig_tokens = [t for t, _ in hybrid_results["prediction_path"]["original_predictions"]]
        orig_probs = [p for _, p in hybrid_results["prediction_path"]["original_predictions"]]
        
        mod_tokens = [t for t, _ in hybrid_results["prediction_path"]["modified_predictions"]]
        mod_probs = [p for _, p in hybrid_results["prediction_path"]["modified_predictions"]]
        
        x = np.arange(len(orig_tokens))
        width = 0.35
        
        plt.bar(x - width/2, orig_probs, width, label='Original', color='green')
        plt.bar(x + width/2, mod_probs, width, label='After Head Ablation', color='red')
        
        plt.xlabel('Token')
        plt.ylabel('Probability')
        plt.title(f"Prediction Changes for '{hybrid_results['input']}'")
        plt.xticks(x, orig_tokens)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "prediction_changes.png"))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        
        methods = ["CNA", "CAM", "Hybrid"]
        times = [
            hybrid_results["timing"]["cna_analysis"],
            hybrid_results["timing"]["cam_analysis"],
            hybrid_results["timing"]["total_time"]
        ]
        
        plt.bar(methods, times, color=['blue', 'red', 'purple'])
        plt.ylabel('Time (seconds)')
        plt.title('Analysis Performance Comparison')
        
        for i, v in enumerate(times):
            plt.text(i, v + 0.05, f"{v:.2f}s", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "performance_comparison.png"))
        plt.close()
        
        with open(os.path.join(save_dir, "hybrid_results.json"), "w") as f:
            json.dump(convert_numpy_types(hybrid_results), f, indent=2)
        
        print(f"Visualizations saved to {save_dir}")


def run_comparative_analysis(input_examples=None):
    print("Starting comprehensive analysis of arithmetic mechanisms...")
    
    if input_examples is None:
        input_examples = [
            "3+5=",
            "7-2=",
            "4*3=",
            "8/2=",
            "15+27=",
        ]
    
    mapper = EnhancedContrastiveAttributionMapper()
    
    all_results = {}
    
    for input_text in input_examples:
        print(f"\n{'='*50}\nAnalyzing: {input_text}\n{'='*50}")
        
        if "+" in input_text:
            target_head = (17, 22)
        elif "-" in input_text:
            target_head = (17, 22)
        elif "*" in input_text:
            target_head = (20, 18)
        elif "/" in input_text:
            target_head = (14, 19)
        else:
            target_head = (17, 22)
        
        results = mapper.analyze_with_hybrid_approach(input_text, target_head=target_head)
        
        output_dir = sanitize_filename(f"results_{input_text}")
        mapper.visualize_results(results, save_dir=output_dir)
        
        all_results[input_text] = results
        
        print(f"\nAnalysis summary for '{input_text}':")
        print(f"  CNA identified {results['overlap_analysis']['cna_count']} neurons in {results['timing']['cna_analysis']:.2f}s")
        print(f"  CAM identified {results['overlap_analysis']['cam_count']} neurons in {results['timing']['cam_analysis']:.2f}s")
        print(f"  Overlap: {results['overlap_analysis']['overlap_count']} neurons (Jaccard: {results['overlap_analysis']['jaccard_similarity']:.2f})")
        print(f"  Top prediction: {results['prediction_path']['original_predictions'][0][0]} ({results['prediction_path']['original_predictions'][0][1]:.4f})")
        
        if results['prediction_path']['modified_predictions'][0][0] != results['prediction_path']['original_predictions'][0][0]:
            print(f"  WARNING: Ablating head {target_head} changes prediction to {results['prediction_path']['modified_predictions'][0][0]}")
        
        top_neuron = results['important_neurons'][0]
        print(f"  Most important neuron: Layer {top_neuron['layer']}, Neuron {top_neuron['neuron']}")
        print(f"  Associated tokens: {', '.join(top_neuron['tokens'])}")
    
    create_comparative_summary(all_results, "comparison_summary")
    
    print("\nComparative analysis complete!")
    return all_results


def create_comparative_summary(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    examples = list(all_results.keys())
    cna_times = [r["timing"]["cna_analysis"] for r in all_results.values()]
    cam_times = [r["timing"]["cam_analysis"] for r in all_results.values()]
    hybrid_times = [r["timing"]["total_time"] for r in all_results.values()]
    
    x = np.arange(len(examples))
    width = 0.25
    
    plt.bar(x - width, cna_times, width, label='CNA', color='blue')
    plt.bar(x, cam_times, width, label='CAM', color='red')
    plt.bar(x + width, hybrid_times, width, label='Hybrid', color='purple')
    
    plt.xlabel('Input Examples')
    plt.ylabel('Time (seconds)')
    plt.title('Performance Comparison')
    plt.xticks(x, examples, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_comparison.png"))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    
    cna_counts = [r["overlap_analysis"]["cna_count"] for r in all_results.values()]
    cam_counts = [r["overlap_analysis"]["cam_count"] for r in all_results.values()]
    overlap_counts = [r["overlap_analysis"]["overlap_count"] for r in all_results.values()]
    
    plt.bar(x - width, cna_counts, width, label='CNA only', color='blue')
    plt.bar(x, cam_counts, width, label='CAM only', color='red')
    plt.bar(x + width, overlap_counts, width, label='Overlap', color='purple')
    
    plt.xlabel('Input Examples')
    plt.ylabel('Neuron Count')
    plt.title('Neuron Identification Comparison')
    plt.xticks(x, examples, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "neuron_count_comparison.png"))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    
    jaccard_values = [r["overlap_analysis"]["jaccard_similarity"] for r in all_results.values()]
    
    plt.bar(examples, jaccard_values, color='purple')
    plt.xlabel('Input Examples')
    plt.ylabel('Jaccard Similarity')
    plt.title('Neuron Identification Agreement (Jaccard Similarity)')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "jaccard_similarity.png"))
    plt.close()
    
    plt.figure(figsize=(14, 8))
    
    for i, (example, results) in enumerate(all_results.items()):
        layers = [n["layer"] for n in results["important_neurons"][:20]]
        
        plt.subplot(len(all_results), 1, i+1)
        plt.hist(layers, bins=range(0, 33), alpha=0.7)
        plt.title(f"Layer Distribution for '{example}'")
        plt.xlabel("Layer")
        plt.ylabel("Neuron Count")
        plt.xlim(0, 32)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layer_distribution.png"))
    plt.close()
    
    summary_data = []
    for example, results in all_results.items():
        summary_data.append({
            "Input": example,
            "CNA Time (s)": f"{results['timing']['cna_analysis']:.2f}",
            "CAM Time (s)": f"{results['timing']['cam_analysis']:.2f}",
            "Hybrid Time (s)": f"{results['timing']['total_time']:.2f}",
            "CNA Neurons": results['overlap_analysis']['cna_count'],
            "CAM Neurons": results['overlap_analysis']['cam_count'],
            "Overlap Neurons": results['overlap_analysis']['overlap_count'],
            "Jaccard Similarity": f"{results['overlap_analysis']['jaccard_similarity']:.2f}",
            "Original Prediction": f"{results['prediction_path']['original_predictions'][0][0]} ({results['prediction_path']['original_predictions'][0][1]:.2f})",
            "Top Neuron": f"L{results['important_neurons'][0]['layer']}_N{results['important_neurons'][0]['neuron']}"
        })
    
    import pandas as pd
    pd.DataFrame(summary_data).to_csv(os.path.join(output_dir, "summary_table.csv"), index=False)
    
    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        json.dump(convert_numpy_types(all_results), f, indent=2)
    
    print(f"Comparative summary saved to {output_dir}")


if __name__ == "__main__":
    print("Starting Enhanced Contrastive Attribution Mapping analysis...")
    all_results = run_comparative_analysis()
    print("Analysis complete!")
