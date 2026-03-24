"""Head-level KV cache pruning for efficient long-context inference.

Implements selective KV cache retention:
- Induction heads: retain full KV cache (important for next-token prediction)
- Other heads: apply streaming-llm start_recent strategy (keep first start_size + last recent_size tokens)
"""

import torch
import torch.nn as nn


class HeadPruningConfig:
    """Configuration for head-level pruning."""
    
    def __init__(self, full_heads=None, start_size=1024, recent_size=1024):
        """
        Args:
            full_heads: dict mapping layer_name -> set of head indices to keep full KV
                       e.g. {'layer_016': {14, 15}, 'layer_021': {8, 9}}
            start_size: number of initial tokens to retain in start_recent strategy
            recent_size: number of recent tokens to retain in start_recent strategy
        """
        self.full_heads = full_heads or {}
        self.start_size = start_size
        self.recent_size = recent_size
    
    @staticmethod
    def from_induction_heads(induction_heads_list):
        """Create config from induction heads list.
        
        Args:
            induction_heads_list: list of {"layer": "layer_XXX", "head": H} dicts
        """
        full_heads = {}
        for item in induction_heads_list:
            layer = item['layer']
            head = item['head']
            if layer not in full_heads:
                full_heads[layer] = set()
            full_heads[layer].add(head)
        return HeadPruningConfig(full_heads)


def apply_head_pruning_to_model(model, pruning_config):
    """Register head pruning config on model.
    
    This stores the config as an attribute; the actual KV cache inference
    logic must be implemented in the attention forward or generate loop.
    """
    model.head_pruning_config = pruning_config
    return model


def apply_start_recent_kv_to_head(
    key_cache, value_cache,
    start_size, recent_size,
    seq_dim=2
):
    """Apply start_recent KV cache strategy to a single head's cache.
    
    Args:
        key_cache, value_cache: KV tensors with current_length attribute
        start_size: keep first N tokens
        recent_size: keep last N tokens
        seq_dim: dimension along which sequence is
    
    Returns:
        pruned key_cache, value_cache (same format)
    """
    current_len = int(key_cache.current_length.item())
    
    if current_len <= start_size + recent_size:
        # Cache is small enough, keep all
        return key_cache, value_cache
    
    # Prune: keep [0:start_size] and [-(recent_size):]
    indices_to_keep = (
        list(range(start_size)) +
        list(range(current_len - recent_size, current_len))
    )
    indices_to_keep = torch.tensor(indices_to_keep, device=key_cache.device, dtype=torch.long)
    
    # Index along seq_dim
    pruned_key = torch.index_select(key_cache, seq_dim, indices_to_keep)
    pruned_value = torch.index_select(value_cache, seq_dim, indices_to_keep)
    
    # Update current_length
    pruned_key.current_length = torch.tensor(
        min(current_len, start_size + recent_size),
        device=key_cache.device,
        dtype=key_cache.current_length.dtype
    )
    pruned_value.current_length = torch.tensor(
        min(current_len, start_size + recent_size),
        device=value_cache.device,
        dtype=value_cache.current_length.dtype
    )
    
    return pruned_key, pruned_value


def prune_kv_cache_by_config(model, pruning_config, layer_name, seq_dim=2):
    """Apply head-level pruning to a model layer's KV cache.
    
    Args:
        model: transformers model with past_key_values
        pruning_config: HeadPruningConfig instance
        layer_name: str like "layer_016"
        seq_dim: sequence dimension in KV tensors
    """
    if not hasattr(model, 'past_key_values') or model.past_key_values is None:
        return
    
    # Map layer name to index
    # "layer_016" -> 16
    try:
        layer_idx = int(layer_name.split('_')[1])
    except (ValueError, IndexError):
        return
    
    if layer_idx >= len(model.past_key_values):
        return
    
    key_cache, value_cache = model.past_key_values[layer_idx]
    
    # Check which heads should be pruned
    full_head_indices = pruning_config.full_heads.get(layer_name, set())
    num_heads = key_cache.shape[1] if len(key_cache.shape) > 2 else key_cache.shape[0]
    
    # For simplicity, apply same strategy to all heads in this layer
    # (More sophisticated: per-head pruning would index along head dim)
    # Here we apply to per-head KV if they exist
    
    if len(key_cache.shape) == 4:  # [batch, num_heads, seq, head_dim]
        for h in range(num_heads):
            if h not in full_head_indices:
                # Prune this head via start_recent
                k_h = key_cache[:, h:h+1, :, :]
                v_h = value_cache[:, h:h+1, :, :]
                k_p, v_p = apply_start_recent_kv_to_head(
                    k_h, v_h,
                    pruning_config.start_size,
                    pruning_config.recent_size,
                    seq_dim=2
                )
                key_cache[:, h:h+1, :, :] = k_p
                value_cache[:, h:h+1, :, :] = v_p
    else:
        # Fallback: apply to entire layer
        k_p, v_p = apply_start_recent_kv_to_head(
            key_cache, value_cache,
            pruning_config.start_size,
            pruning_config.recent_size,
            seq_dim=seq_dim
        )
        model.past_key_values[layer_idx] = (k_p, v_p)


# ===== Integration with fastchat / generate =====

def should_prune_kv_at_step(step_count, prune_interval=10):
    """Decide whether to apply pruning at current generation step.
    
    Pruning every N steps is more efficient than every step.
    """
    return step_count % prune_interval == 0
