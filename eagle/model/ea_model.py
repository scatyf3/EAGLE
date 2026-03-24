import copy
import json
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
#from .modeling_qwen2_kv import LlamaForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .cnets1 import Model as Model1
from .configs import EConfig


class EaModel(nn.Module):

    def __init__(
            self,
            use_eagle3,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        self.use_eagle3 = use_eagle3
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        if use_eagle3:
            self.ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
        else:
            self.ea_layer = Model1(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        if self.use_eagle3 and config.vocab_size==config.draft_vocab_size:
            del self.ea_layer.d2t,self.ea_layer.t2d
        load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            use_eagle3=True,
            base_model_path=None,
            ea_model_path=None,
            total_token=60,
            depth=7,
            top_k=10,
            threshold=1.0,
            **kwargs,
    ):
        # assert Type=="LLaMA" or "Mixtral"
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen2ForCausalLM':
            base_model = KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen3ForCausalLM':
            base_model = KVQwen3ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")

        try:
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            use_eagle3,
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )

        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.ea_layer.total_tokens = total_token - 1

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @staticmethod
    def _build_streaming_keep_indices(seq_len, sink_size, window_size, device):
        """Keep a small sink prefix plus a recent tail window, in original order."""
        sink_size = max(0, int(sink_size))
        window_size = max(0, int(window_size))
        if seq_len <= sink_size + window_size:
            return None

        prefix_len = min(sink_size, seq_len)
        tail_start = max(prefix_len, seq_len - window_size)
        prefix = torch.arange(0, prefix_len, device=device, dtype=torch.long)
        tail = torch.arange(tail_start, seq_len, device=device, dtype=torch.long)
        if prefix.numel() == 0:
            return tail
        if tail.numel() == 0:
            return prefix
        return torch.cat([prefix, tail], dim=0)

    @staticmethod
    def _parse_head_set(heads):
        if heads is None:
            return None
        out = set()
        for h in heads:
            try:
                out.add(int(h))
            except Exception:
                continue
        return out

    def _map_attn_heads_to_kv_heads(self, full_attn_heads, num_kv_heads):
        """Map attention-head ids to KV-head ids for grouped-query attention.

        Draft attention traces are reported on attention heads, while KV cache is
        stored on KV heads. For Qwen-style GQA, each KV head is shared by
        `num_attention_heads / num_kv_heads` attention heads.
        """
        if not full_attn_heads:
            return set()

        try:
            num_attn_heads = int(self.ea_layer.layers[0].self_attn.num_heads)
        except Exception:
            num_attn_heads = int(num_kv_heads)

        group_size = max(1, num_attn_heads // max(1, int(num_kv_heads)))
        full_kv = set()
        for h in full_attn_heads:
            if h < 0:
                continue
            kv_h = int(h) // group_size
            if 0 <= kv_h < int(num_kv_heads):
                full_kv.add(kv_h)
        return full_kv

    def _apply_streaming_kv_prune(self, sink_size, window_size, full_draft_heads=None):
        """Prune draft-head KV cache only (do not modify target KV or input_ids).

        Two modes:
        1) full_draft_heads is None: physical KV truncate for all draft heads
           (saves memory and compute).
          2) full_draft_heads is provided: treat those as protected/global-like heads
              and keep their KV full; apply sink+window masking to other heads
              (logical head-wise prune).
           This preserves tensor shape for compatibility.
        """
        if not (hasattr(self.ea_layer, "stable_kv") and self.ea_layer.stable_kv is not None):
            return False

        first_layer = self.ea_layer.stable_kv[0]
        if not isinstance(first_layer, (list, tuple)) or len(first_layer) < 2:
            return False

        seq_len = int(first_layer[0].shape[2])
        keep_indices = EaModel._build_streaming_keep_indices(
            seq_len,
            sink_size=sink_size,
            window_size=window_size,
            device=first_layer[0].device,
        )
        if keep_indices is None:
            return False

        full_draft_heads = self._parse_head_set(full_draft_heads)

        # Mode A: keep original behavior (all draft heads share one pruned KV tensor).
        if not full_draft_heads:
            pruned_stable_kv = []
            for layer_kv in self.ea_layer.stable_kv:
                if isinstance(layer_kv, (list, tuple)) and len(layer_kv) >= 2:
                    k_cache = layer_kv[0]
                    v_cache = layer_kv[1]
                    k_idx = keep_indices.to(k_cache.device)
                    v_idx = keep_indices.to(v_cache.device)
                    k_new = k_cache.index_select(2, k_idx)
                    v_new = v_cache.index_select(2, v_idx)
                    if isinstance(layer_kv, tuple):
                        pruned_stable_kv.append((k_new, v_new))
                    else:
                        pruned_stable_kv.append([k_new, v_new])
                else:
                    pruned_stable_kv.append(layer_kv)

            if isinstance(self.ea_layer.stable_kv, tuple):
                self.ea_layer.stable_kv = tuple(pruned_stable_kv)
            else:
                self.ea_layer.stable_kv = pruned_stable_kv

            # Keep draft-side token history aligned with draft KV length.
            if hasattr(self, "draft_input_ids") and self.draft_input_ids is not None:
                max_idx = int(keep_indices[-1].item()) if keep_indices.numel() > 0 else -1
                if self.draft_input_ids.shape[1] > max_idx:
                    self.draft_input_ids = self.draft_input_ids.index_select(
                        1, keep_indices.to(self.draft_input_ids.device)
                    )
            return True

        # Mode B: head-wise logical prune (protected/global-like heads full, others masked).
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=first_layer[0].device)
        keep_mask[keep_indices] = True
        drop_mask = ~keep_mask

        pruned_stable_kv = []
        for layer_kv in self.ea_layer.stable_kv:
            if isinstance(layer_kv, (list, tuple)) and len(layer_kv) >= 2:
                k_cache = layer_kv[0]
                v_cache = layer_kv[1]
                num_kv_heads = int(k_cache.shape[1])
                full_kv_heads = self._map_attn_heads_to_kv_heads(full_draft_heads, num_kv_heads)

                k_new = k_cache.clone()
                v_new = v_cache.clone()
                local_drop_mask = drop_mask.to(k_new.device)

                # Zero out dropped-token KV for non-important heads only.
                for kv_h in range(num_kv_heads):
                    if kv_h in full_kv_heads:
                        continue
                    k_new[:, kv_h, local_drop_mask, :] = 0
                    v_new[:, kv_h, local_drop_mask, :] = 0

                if isinstance(layer_kv, tuple):
                    pruned_stable_kv.append((k_new, v_new))
                else:
                    pruned_stable_kv.append([k_new, v_new])
            else:
                pruned_stable_kv.append(layer_kv)

        if isinstance(self.ea_layer.stable_kv, tuple):
            self.ea_layer.stable_kv = tuple(pruned_stable_kv)
        else:
            self.ea_layer.stable_kv = pruned_stable_kv

        return True

    def _parse_head_set(self, head_input):
        """Parse induction heads from various input formats.
        
        Args:
            head_input: Can be:
                - set of ints (head IDs)
                - list of ints (head IDs)
                - dict with 'protected_global_heads' or similar key
                - None (returns empty set)
        
        Returns:
            set of head IDs
        """
        if head_input is None:
            return set()
        if isinstance(head_input, (set, list)):
            return set(head_input)
        if isinstance(head_input, dict):
            # Try common keys for protected/global heads
            for key in ['protected_global_heads', 'draft_induction_heads', 'induction_heads', 'heads']:
                if key in head_input:
                    heads = head_input[key]
                    if isinstance(heads, list):
                        return set(heads)
                    if isinstance(heads, dict):
                        # If structured as {layer: set_of_heads}, extract layer 0 heads
                        if 0 in heads:
                            return set(heads[0])
                    return set(heads)
        return set()

    def _apply_h2o_kv_prune(self, heavy_budget, recent_budget, full_draft_heads=None):
        """Apply H2O KV pruning to the draft model's stable_kv, with optional head-wise preservation.

        Keeps the `heavy_budget` tokens with highest accumulated attention scores
        (heavy hitters) plus the `recent_budget` most recent tokens, discarding
        the rest. Scores are accumulated from each draft-model forward pass via
        `ea_layer.last_h2o_attn_weights`.

        Args:
            heavy_budget: Number of heavy-hitter tokens to keep
            recent_budget: Number of most-recent tokens to keep
            full_draft_heads: Set or list of attention head ids to keep full KV cache for
                             (e.g., induction heads). Other heads will be pruned.
        """
        if not (hasattr(self.ea_layer, "stable_kv") and self.ea_layer.stable_kv is not None):
            return False
        if not (hasattr(self.ea_layer, "last_h2o_attn_weights") and
                self.ea_layer.last_h2o_attn_weights is not None):
            return False

        attn = self.ea_layer.last_h2o_attn_weights  # [num_heads, q_len, kv_len]
        first_layer_kv = self.ea_layer.stable_kv[0]
        seq_len = int(first_layer_kv[0].shape[2])
        num_heads = int(attn.shape[0])

        # Accumulate attention scores: sum over query positions → [num_heads, kv_len]
        new_scores = attn.float().sum(dim=1).cpu()
        if self.h2o_scores is None:
            self.h2o_scores = new_scores
        else:
            old_n = self.h2o_scores.shape[1]
            if seq_len > old_n:
                ext = torch.zeros(
                    self.h2o_scores.shape[0], seq_len - old_n,
                    dtype=self.h2o_scores.dtype,
                )
                self.h2o_scores = torch.cat([self.h2o_scores, ext], dim=1)
            self.h2o_scores[:, :seq_len] += new_scores[:, :seq_len]

        total_budget = heavy_budget + recent_budget
        if seq_len <= total_budget:
            return False

        # If full_draft_heads provided, keep head-wise (logical mask); else physical prune
        full_draft_heads = self._parse_head_set(full_draft_heads) if full_draft_heads else set()

        if not full_draft_heads:
            # Physical pruning: truncate KV cache
            avg_scores = self.h2o_scores[:, :seq_len].mean(dim=0)
            recent_start = seq_len - recent_budget
            actual_heavy = min(heavy_budget, recent_start)
            if actual_heavy > 0:
                _, heavy_idx = avg_scores[:recent_start].topk(actual_heavy, largest=True, sorted=False)
            else:
                heavy_idx = torch.tensor([], dtype=torch.long)
            recent_idx = torch.arange(recent_start, seq_len, dtype=torch.long)
            keep_indices, _ = torch.cat([heavy_idx, recent_idx]).sort()

            pruned_stable_kv = []
            for layer_kv in self.ea_layer.stable_kv:
                if isinstance(layer_kv, (list, tuple)) and len(layer_kv) >= 2:
                    k, v = layer_kv[0], layer_kv[1]
                    k_idx = keep_indices.to(k.device)
                    k_new = k.index_select(2, k_idx)
                    v_new = v.index_select(2, keep_indices.to(v.device))
                    if isinstance(layer_kv, tuple):
                        pruned_stable_kv.append((k_new, v_new))
                    else:
                        pruned_stable_kv.append([k_new, v_new])
                else:
                    pruned_stable_kv.append(layer_kv)

            if isinstance(self.ea_layer.stable_kv, tuple):
                self.ea_layer.stable_kv = tuple(pruned_stable_kv)
            else:
                self.ea_layer.stable_kv = pruned_stable_kv

            # Keep accumulated scores aligned with pruned KV.
            self.h2o_scores = self.h2o_scores[:, keep_indices]

            # Keep draft_input_ids aligned if present.
            if hasattr(self, "draft_input_ids") and self.draft_input_ids is not None:
                if self.draft_input_ids.shape[1] == seq_len:
                    self.draft_input_ids = self.draft_input_ids.index_select(
                        1, keep_indices.to(self.draft_input_ids.device)
                    )
        else:
            # Head-wise logical pruning: keep shape; mask dropped tokens to 0 for non-important heads
            keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=attn.device)
            avg_scores = self.h2o_scores[:, :seq_len].mean(dim=0)
            recent_start = seq_len - recent_budget

            # Mark recent tokens
            keep_mask[recent_start:] = True

            # Mark heavy hitters
            actual_heavy = min(heavy_budget, recent_start)
            if actual_heavy > 0:
                _, heavy_idx = avg_scores[:recent_start].topk(actual_heavy, largest=True, sorted=False)
                keep_mask[heavy_idx] = True

            drop_mask = ~keep_mask

            # Prune: zero out dropped-token KV for non-important heads only
            pruned_stable_kv = []
            for layer_kv in self.ea_layer.stable_kv:
                if isinstance(layer_kv, (list, tuple)) and len(layer_kv) >= 2:
                    k, v = layer_kv[0], layer_kv[1]
                    k_new = k.clone()
                    v_new = v.clone()
                    local_drop_mask = drop_mask.to(k_new.device)

                    # Zero out dropped-token KV for non-important heads only
                    for head_idx in range(num_heads):
                        if head_idx not in full_draft_heads:
                            k_new[:, head_idx, local_drop_mask, :] = 0
                            v_new[:, head_idx, local_drop_mask, :] = 0

                    if isinstance(layer_kv, tuple):
                        pruned_stable_kv.append((k_new, v_new))
                    else:
                        pruned_stable_kv.append([k_new, v_new])
                else:
                    pruned_stable_kv.append(layer_kv)

            if isinstance(self.ea_layer.stable_kv, tuple):
                self.ea_layer.stable_kv = tuple(pruned_stable_kv)
            else:
                self.ea_layer.stable_kv = pruned_stable_kv

        return True

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
            streaming_kv_prune=False,
            streaming_sink_size=4,
            streaming_window_size=2044,
            streaming_full_draft_heads=None,
            draft_attn_debug=False,
            h2o_kv_prune=False,
            h2o_heavy_budget=200,
            h2o_recent_budget=2000,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()
        self.draft_input_ids = input_ids.clone()
        self.last_draft_attn_trace = None
        self.h2o_scores = None
        
        # Load induction heads mapping if H2O pruning is enabled
        h2o_full_draft_heads = None
        if h2o_kv_prune:
            try:
                induction_heads_path = "outputs/induction_heads_mapping.json"
                if os.path.exists(induction_heads_path):
                    with open(induction_heads_path, "r") as f:
                        induction_heads_data = json.load(f)
                    # Extract protected/global head indices from draft model layer 0
                    if isinstance(induction_heads_data, dict):
                        draft_heads = induction_heads_data.get("protected_global_heads")
                        if draft_heads is None:
                            draft_heads = induction_heads_data.get("draft_induction_heads")
                    else:
                        draft_heads = None
                    if draft_heads is not None:
                        # Extract head indices (assuming format: [{"layer": 0, "head": X}, ...])
                        h2o_full_draft_heads = set()
                        for head_info in draft_heads:
                            if isinstance(head_info, dict) and head_info.get("layer") == 0:
                                h2o_full_draft_heads.add(head_info["head"])
            except Exception as e:
                print(f"Warning: Failed to load induction heads mapping: {e}")
                h2o_full_draft_heads = None

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        # prefill
        torch.cuda.synchronize()
        _prefill_t0 = time.time()
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor,
            h2o_collect_attn=h2o_kv_prune,
        )
        if streaming_kv_prune:
            _ = self._apply_streaming_kv_prune(
                sink_size=streaming_sink_size,
                window_size=streaming_window_size,
                full_draft_heads=streaming_full_draft_heads,
            )
        if h2o_kv_prune:
            _ = self._apply_h2o_kv_prune(
                heavy_budget=h2o_heavy_budget,
                recent_budget=h2o_recent_budget,
                full_draft_heads=h2o_full_draft_heads,
            )
        torch.cuda.synchronize()
        prefill_time = time.time() - _prefill_t0
        new_token = 0
        draft_attn_trace = [] if draft_attn_debug else None
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # Target model forward, get logits
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            # verification
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            # Adjusting the input sequence, draft model forward
            prev_len = input_ids.shape[1]
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p,
                draft_attn_debug=draft_attn_debug,
                h2o_collect_attn=h2o_kv_prune,
            )

            # Check stop tokens only on newly accepted tokens from this decode step.
            accepted_ids = input_ids[0, prev_len:]
            accepted_list = accepted_ids.tolist()

            if draft_attn_debug:
                draft_attn_trace.append(
                    {
                        "decode_step_idx": int(idx),
                        "accepted_token_ids": [int(x) for x in accepted_list],
                        "draft_topk_trace": getattr(self.ea_layer, "last_topk_draft_attn_trace", None),
                    }
                )

            if is_llama3 and stop_token_id in accepted_list:
                break

            if self.tokenizer.eos_token_id in accepted_list:
                break

            if streaming_kv_prune:
                _ = self._apply_streaming_kv_prune(
                    sink_size=streaming_sink_size,
                    window_size=streaming_window_size,
                    full_draft_heads=streaming_full_draft_heads,
                )
            if h2o_kv_prune:
                _ = self._apply_h2o_kv_prune(
                    heavy_budget=h2o_heavy_budget,
                    recent_budget=h2o_recent_budget,
                    full_draft_heads=h2o_full_draft_heads,
                )
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if draft_attn_debug:
            self.last_draft_attn_trace = draft_attn_trace
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx, prefill_time

    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
            attn_debug=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()
        self.last_attn_trace = None

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        attn_trace = [] if attn_debug else None
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(
                input_id,
                use_cache=True,
                past_key_values=past_key_values,
                output_attentions=attn_debug,
            )
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            if attn_debug:
                # attentions: tuple(num_layers), each [bsz, num_heads, q_len, kv_len]
                # q_len is usually 1 in incremental decoding.
                per_layer = {}
                total_len = input_ids.shape[1]
                for layer_idx, layer_attn in enumerate(outputs.attentions):
                    if layer_attn is None:
                        continue
                    # Keep only scores to previous tokens (exclude self at the last key position).
                    # Result shape: [num_heads, total_len - 1]
                    per_layer[f"layer_{layer_idx:03d}"] = layer_attn[0, :, -1, : total_len - 1].detach().cpu()

                attn_trace.append(
                    {
                        "step_idx": int(new_token - 1),
                        "accepted_token_id": int(input_id[0, 0].item()),
                        "seq_len_after_accept": int(total_len),
                        "attn_to_previous_tokens": per_layer,
                    }
                )

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            if attn_debug:
                self.last_attn_trace = attn_trace
            return input_ids
        else:
            if attn_debug:
                self.last_attn_trace = attn_trace
            return input_ids, new_token, idx

    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            # with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break