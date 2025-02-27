# Copyright 2023 Cheng Li
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import fire

from llm_analysis.constant import (DTYPE_CONFIG_DIR_NAME, GPU_CONFIG_DIR_NAME,
                                   MODEL_CONFIG_DIR_NAME)
from llm_analysis.logger import logger

try:
    from transformers import AutoConfig
except ImportError:
    logger.warning(
        f"cannot import AutoConfig from transformers, `transformers` is not installed, HuggingFace will not be available to use for model config retrieval"
    )
    AutoConfig = None


class EnhancedJSONEncoder(json.JSONEncoder):

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


@dataclass
class ModelConfig:
    """Configuration class for transformer-based models with support for various attention mechanisms."""

    # Required fields
    name: str  # Model configuration name
    num_layers: int  # Number of transformer layers (blocks)
    n_head: int  # Number of attention heads
    hidden_dim: int  # Hidden dimension size
    vocab_size: int  # Vocabulary size

    # Optional fields with defaults
    max_seq_len: int | None = None  # Maximum sequence length

    # Attention mechanism configuration
    num_key_value_heads: int | None = None  # Number of key-value heads for attention
    num_key_value_groups: int | None = field(
        default=None,
        repr=True)  # Number of key-value groups (calculated in post_init)

    # Feed-forward network configuration
    ffn_embed_dim: int | None = None  # Hidden dimension of feed-forward network
    expansion_ratio: float | None = (
        None  # Expansion ratio for hidden_dim to ffn_embed_dim
    )

    # Model type and architecture configuration
    model_type: str = "unknown"  # Model architecture type
    mlp_gated_linear_units: bool = False  # Whether to use gated linear units for MLP

    # Mixture of Experts (MoE) configuration
    moe_num_experts: int | None = None  # Number of experts for MoE
    moe_num_shared_experts: int | None = None  # Number of shared experts for MoE
    moe_top_k: int | None = None  # Top-k experts to use per token
    moe_intermediate_size: int | None = None  # Intermediate size of MoE layer
    first_k_dense_replace: int | None = (
        None  # Number of dense layers to replace with MoE
    )
    q_lora_rank: int | None = None  # Rank for Q LoRA (Low-Rank Adaptation)
    kv_lora_rank: int | None = None  # Rank for KV LoRA (Low-Rank Adaptation)
    qk_nope_head_dim: int | None = (
        None  # Head dimension for QK NoPE (No Position Embedding)
    )
    qk_rope_head_dim: int | None = (
        None  # Head dimension for QK RoPE (Rotary Position Embedding)
    )

    def __post_init__(self) -> None:
        """
        Initialize derived properties and perform validation on the configuration.

        This method:
        1. Sets ffn_embed_dim and expansion_ratio if not provided
        2. Sets num_key_value_heads to n_head if not provided
        3. Validates attention head configuration
        4. Calculates num_key_value_groups
        """
        # Handle feed-forward network dimension settings
        if self.ffn_embed_dim is None and self.expansion_ratio is None:
            self.ffn_embed_dim = self.hidden_dim * 4
            self.expansion_ratio = 4.0
        elif self.ffn_embed_dim is None and self.expansion_ratio is not None:
            self.ffn_embed_dim = int(self.hidden_dim * self.expansion_ratio)
        elif self.expansion_ratio is None and self.ffn_embed_dim is not None:
            self.expansion_ratio = self.ffn_embed_dim / self.hidden_dim

        # Handle attention mechanism settings
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.n_head

        # Validate attention head configuration
        if self.n_head % self.num_key_value_heads != 0:
            raise ValueError(
                f"n_head ({self.n_head}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})")

        # Calculate number of key-value groups
        self.num_key_value_groups = self.n_head / self.num_key_value_heads

        # Check if this is a Mixture of Experts model
        is_moe_model = False
        if (self.moe_num_experts or self.moe_num_shared_experts
                or self.moe_top_k or self.moe_intermediate_size):
            is_moe_model = True

        if is_moe_model and (self.moe_intermediate_size is None):
            self.moe_intermediate_size = self.ffn_embed_dim

    def __str__(self) -> str:
        """Return a formatted string representation of the configuration."""
        config_dict = asdict(self)
        formatted_items = [f"{k}={v}" for k, v in config_dict.items()]
        return f"ModelConfig({', '.join(formatted_items)})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return asdict(self)

    @property
    def attention_type(self) -> str:
        """
        Returns the attention mechanism type used in the model.

        Returns:
            str: "MHA" for Multi-Head Attention
                 "MQA" for Multi-Query Attention
                 "GQA" for Grouped-Query Attention
        """
        if self.num_key_value_heads == self.n_head:
            if self.q_lora_rank is not None and self.kv_lora_rank is not None:
                return "MLA"  # Multi-head Latent Attention
            return "MHA"  # Multi-Head Attention
        elif self.num_key_value_heads == 1:
            return "MQA"  # Multi-Query Attention
        else:
            return "GQA"  # Grouped-Query Attention


@dataclass
class GPUConfig:
    name: str  # GPU config name
    mem_per_GPU_in_GB: float  # memory per GPU in GB
    hbm_bandwidth_in_GB_per_sec: float  # GPU HBM bandwidth in GB/s
    intra_node_bandwidth_in_GB_per_sec: float  # intra node GPU bandwidth in GB/s
    intra_node_min_message_latency: (
        float  # minimum intra node message latency in seconds
    )
    peak_fp16_TFLOPS: float  # peak Tensor TFLOPS for FP16
    peak_i8_TFLOPS: float = None  # peak Tensor TFLOPS for INT8
    peak_i4_TFLOPS: float = None  # peak Tensor TFLOPS for INT4
    inter_node_bandwidth_in_GB_per_sec: float = (
        200  # inter node bandwidth in GB/s, assuming Mellanox 200Gbps HDR Infiniband
    )

    def __post_init__(self):
        if self.peak_i8_TFLOPS is None:
            self.peak_i8_TFLOPS = 2 * self.peak_fp16_TFLOPS
        if self.peak_i4_TFLOPS is None:
            self.peak_i4_TFLOPS = 4 * self.peak_fp16_TFLOPS


@dataclass
class DtypeConfig:
    name: str = "w16a16e16"  # dtype config name
    weight_bits: int = 16  # number of bits for weight
    activation_bits: int = 16  # number of bits for activation
    embedding_bits: int = 16  # number of bits for the embedding
    linear_weight_bits: int = 16  # number of bits for weight in linear layer
    linear_activation_bits: int = 16  # number of bits for activation in linear layer


@dataclass
class ParallelismConfig:
    tp_size: int = (
        1  # tensor parallelism size, Megatron-LM tensor parallelism implementation
    )
    pp_size: int = (
        1  # pipeline parallelism size, Megatron-LM pipeline parallelism implementation
    )
    dp_size: int = (
        1  # sharded data parallelism size, PyTorch FSDP or DeepSpeed Zero parallelism implementation
    )
    rdp_size: int = 1  # replicated data parallelism size, PyTorch HSDP implementation
    ep_size: int = 1  # expert parallelism size
    sp_size: int = (
        None  # sequence parallelism size, Megatron-LM sequence parallelism implementation
    )

    def __post_init__(self):
        if self.sp_size is None:
            self.sp_size = self.tp_size


# model name and configurations mapping populated from MODEL_CONFIG_DIR_NAME
model_configs = {}

# gpu name and configurations mapping populated from MODEL_CONFIG_DIR_NAME
# https://gist.github.com/joshlk/bbb1aca6e70b11d251886baee6423dcb
gpu_configs = {}

# dtype name and configurations mapping populated from MODEL_CONFIG_DIR_NAME
dtype_configs = {}


def canonical_model_name(name: str) -> str:
    return name.replace("/", "_")


def dump_configs(configs: dict, config_dir_name: str) -> None:
    """Dump configs to json files under config_dir_name.

    Args:
        configs (dict): a dict of configs
        config_dir_name (str): the name of the output directory
    """
    for k, v in configs.items():
        with open(
                Path(__file__).parent / Path(config_dir_name, f"{k}.json"),
                "w") as f:
            json.dump(v, f, cls=EnhancedJSONEncoder, indent=4)
    logger.info(f"dumped {len(configs)} configs to {config_dir_name}")


def get_model_config_from_hf(name: str, ) -> ModelConfig:
    """Get model config from HuggingFace transformers library `AutoConfig`; if the model
    does not exist, try updating the transformers library.

    Args:
        name (str): the model id of a pretrained model configuration hosted inside a model repo on huggingface.co

    Returns:
        ModelConfig: a dataclass for llm-analysis model config

    Raises:
        Exception: When configuration values cannot be determined
    """
    if AutoConfig is None:
        logger.warning(
            f"Cannot import AutoConfig from transformers, `transformers` is not installed, HuggingFace will not be available to use for model config retrieval"
        )
        return None

    # Add error handling for failed loading
    try:
        hf_config = AutoConfig.from_pretrained(name, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load config from HF for {name}: {str(e)}")
        raise Exception(
            f"Could not load model configuration from HF: {str(e)}")

    # Model type extraction with proper default
    model_type = hf_config.model_type if hasattr(hf_config,
                                                 "model_type") else "unknown"

    # Extract number of layers
    if hasattr(hf_config, "num_hidden_layers"):
        num_layers = hf_config.num_hidden_layers
    elif hasattr(hf_config, "n_layers"):
        num_layers = hf_config.n_layers
    else:
        raise Exception(
            "Could not determine number of layers: missing 'num_hidden_layers' and 'n_layers' attributes"
        )

    # Extract attention heads
    if hasattr(hf_config, "num_attention_heads"):
        n_head = hf_config.num_attention_heads
    elif hasattr(hf_config, "n_heads"):
        n_head = hf_config.n_heads
    else:
        raise Exception(
            "Could not determine number of attention heads: missing 'num_attention_heads' and 'n_heads' attributes"
        )

    # Extract hidden dimension
    if hasattr(hf_config, "hidden_size"):
        hidden_dim = hf_config.hidden_size
    elif hasattr(hf_config, "d_model"):
        hidden_dim = hf_config.d_model
    else:
        raise Exception(
            "Could not determine hidden dimension: missing 'hidden_size' and 'd_model' attributes"
        )

    # Extract feed-forward network dimension
    if hasattr(hf_config, "ffn_embed_dim"):
        ffn_embed_dim = hf_config.ffn_embed_dim
    elif hasattr(hf_config, "intermediate_size"):
        ffn_embed_dim = hf_config.intermediate_size
    elif hasattr(hf_config, "expansion_ratio"):
        ffn_embed_dim = int(hidden_dim * hf_config.expansion_ratio)
    else:
        raise Exception(
            "Could not determine ffn dimension: missing 'ffn_embed_dim' or 'intermediate_size' or '        expansion_ratio' attributes"
        )

    # Check for gated linear units
    mlp_gated_linear_units = False
    expansion_ratio = ffn_embed_dim / hidden_dim
    if (expansion_ratio == 3.5
            and model_type == "llama") or model_type == "deepseek_v3":
        mlp_gated_linear_units = True

    # Extract MoE experts count, default to None if not found
    if hasattr(hf_config, "moe_num_experts"):
        moe_num_experts = hf_config.moe_num_experts
    elif hasattr(hf_config, "num_local_experts"):
        moe_num_experts = hf_config.num_local_experts
    elif hasattr(hf_config, "n_routed_experts"):
        moe_num_experts = hf_config.n_routed_experts
    else:
        moe_num_experts = None

    # Extract MoE top-k, default to None if not found
    if hasattr(hf_config, "num_experts_per_tok"):
        moe_top_k = hf_config.num_experts_per_tok
    elif hasattr(hf_config, "moe_top_k"):
        moe_top_k = hf_config.moe_top_k
    else:
        moe_top_k = None

    # Extract MoE shared experts, default to None if not found
    if hasattr(hf_config, "moe_num_shared_experts"):
        moe_num_shared_experts = hf_config.moe_num_shared_experts
    elif hasattr(hf_config, "n_shared_experts"):
        moe_num_shared_experts = hf_config.n_shared_experts
    else:
        moe_num_shared_experts = None

    # Extract optional attributes
    first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", None)
    moe_intermediate_size = getattr(hf_config, "moe_intermediate_size", None)
    q_lora_rank = getattr(hf_config, "q_lora_rank", None)
    kv_lora_rank = getattr(hf_config, "kv_lora_rank", None)
    qk_nope_head_dim = getattr(hf_config, "qk_nope_head_dim", None)
    qk_rope_head_dim = getattr(hf_config, "qk_rope_head_dim", None)

    # Check for vocab_size with validation
    if not hasattr(hf_config, "vocab_size"):
        raise Exception(
            "Could not determine vocabulary size: missing 'vocab_size' attribute"
        )

    vocab_size = hf_config.vocab_size
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError(f"Invalid vocabulary size: {vocab_size}")

    # Safe model name canonicalization
    try:
        canonical_name = canonical_model_name(name)
    except Exception as e:
        logger.warning(f"Failed to get canonical model name: {str(e)}")
        canonical_name = name

    config = ModelConfig(
        name=canonical_name,
        max_seq_len=(hf_config.max_position_embeddings if hasattr(
            hf_config, "max_position_embeddings") else None),
        num_layers=num_layers,
        n_head=n_head,
        hidden_dim=hidden_dim,
        ffn_embed_dim=ffn_embed_dim,
        vocab_size=vocab_size,
        model_type=model_type,
        num_key_value_heads=(hf_config.num_key_value_heads if hasattr(
            hf_config, "num_key_value_heads") else None),
        moe_num_experts=moe_num_experts,
        moe_top_k=moe_top_k,
        moe_intermediate_size=moe_intermediate_size,
        mlp_gated_linear_units=mlp_gated_linear_units,
        first_k_dense_replace=first_k_dense_replace,
        moe_num_shared_experts=moe_num_shared_experts,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
    )

    return config


def read_configs(config_dir_name: str, type="model") -> dict:
    """Read configs from a directory."""
    configs = {}
    for filename in os.listdir(config_dir_name):
        filepath = os.path.join(config_dir_name, filename)
        with open(filepath, "r") as f:
            config_json = json.load(f)
            if type == "model":
                config = ModelConfig(**config_json)
            elif type == "gpu":
                config = GPUConfig(**config_json)
            elif type == "dtype":
                config = DtypeConfig(**config_json)
            else:
                assert False, f"unknown config type when reading: {type}"
            if config.name not in configs:
                configs[config.name] = config
    logger.info(f"Loaded {len(configs)} configs from {config_dir_name}")
    return configs


def get_hf_models_by_type_and_task(
    model_type: str = "opt",
    task: str = None,
    min_downloads: int = 10000,
    top_k: int = 6,
    full_info: bool = False,
) -> list:
    """Get a HuggingFace model name list by model type and task, filtered by popularity
    (minimal number of downloads)

    Args:
        model_type (str, optional): model type, e.g., gpt, llama, opt, bloom. Defaults to "opt".
        task (str, optional): model task, e.g., text-generation, fill-mask. Defaults to "text-generation".
        min_downloads (int, optional): minimal number of downloads to filter the models. Defaults to 10000.
        top_k (int, optional): _description_. Defaults to 6.
        full_info (bool, optional): whether to return full model information, if False, just return the list of model names. Defaults to False.

    Returns:
        list: a list of HuggingFace model information
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error(
            f"cannot import HfApi from huggingface_hub, lease install huggingface_hub first"
        )
    api = HfApi()
    models = api.list_models(filter=model_type)
    logger.info(f"found {len(models)} models of type {model_type}")
    # sort by number of downloads
    ordered = sorted(
        models,
        reverse=True,
        key=lambda t: t.downloads if hasattr(t, "downloads") else 0,
    )
    ret = []
    for m in ordered:
        if hasattr(m, "downloads") and m.downloads > min_downloads:
            if task:
                if hasattr(m, "pipeline_tag") and m.pipeline_tag == task:
                    ret.append(m)
            else:
                ret.append(m)
    top_k = max(1, min(top_k, len(ret)))
    logger.info(f"take top {top_k} of the list of found models")
    if full_info:
        return ret[:top_k]
    return [r.modelId for r in ret][:top_k]


def populate_model_and_gpu_configs() -> None:
    """Populate model, gpu, and data type configs from the pre-defined json files."""
    global model_configs, gpu_configs, dtype_configs
    model_configs = read_configs(Path(__file__).parent /
                                 Path(MODEL_CONFIG_DIR_NAME),
                                 type="model")
    gpu_configs = read_configs(Path(__file__).parent /
                               Path(GPU_CONFIG_DIR_NAME),
                               type="gpu")

    dtype_configs = read_configs(Path(__file__).parent /
                                 Path(DTYPE_CONFIG_DIR_NAME),
                                 type="dtype")
    logger.info(
        f"Populated {len(model_configs)} model configs, {len(gpu_configs)} gpu configs, {len(dtype_configs)} dtype configs"
    )


def list_model_configs() -> None:
    """List all predefined model configs."""
    logger.info(model_configs.keys())


def list_gpu_configs() -> None:
    """List all predefined gpu configs."""
    logger.info(gpu_configs.keys())


def list_dtype_configs() -> None:
    """List all predefined data type configs."""
    logger.info(dtype_configs.keys())


def get_model_config_by_name(name_or_path: str) -> ModelConfig:
    """Get model config from the populated mapping by name, or from model config json file path, if not found from the previous methods, try to get it from HuggingFace."""
    if name_or_path in model_configs:
        return model_configs[name_or_path]
    if os.path.isfile(name_or_path) and ".json" in name_or_path:
        try:
            with open(name_or_path, "r") as f:
                config_json = json.load(f)
                config = ModelConfig(**config_json)
                if config.name not in model_configs:
                    model_configs[config.name] = config
            return config
        except Exception as e:
            raise ValueError(f"unknown model config name: {e}")
    model_config = get_model_config_from_hf(name_or_path)
    if model_config is None:
        raise ValueError(
            f"unknown model config name: {name_or_path}, and none is found on HuggingFace Hub"
        )
    return model_config


def get_gpu_config_by_name(name: str) -> GPUConfig:
    """Get gpu config from the populated mapping by name."""
    if name not in gpu_configs:
        raise ValueError(f"unknown gpu config name: {name}")
    return gpu_configs[name]


def get_dtype_config_by_name(name: str) -> DtypeConfig:
    """Get data type config from the populated mapping by name."""
    if name not in dtype_configs:
        raise ValueError(f"unknown quant config name: {name}")
    return dtype_configs[name]


def dump_model_config_by_name(name: str,
                              config_dir_name: str = MODEL_CONFIG_DIR_NAME
                              ) -> None:
    """Dump a model config from either the populated `model_configs` or Hugging Face by
    name to `config_dir_name`

    Args:
        name (str): model name, e,g., gpt2, facebook/opt-1.3b, decapoda-research/llama-7b-hf, etc.
        config_dir_name (str, optional): _description_. Defaults to MODEL_CONFIG_DIR_NAME.
    """
    model_config = get_model_config_by_name(name)
    dump_configs({model_config.name: model_config}, config_dir_name)
    logger.info(f"dumped model config {model_config} to {config_dir_name}")


def dump_hf_model_configs_by_type_and_task(
    model_type: str = "opt",
    task: str = None,
    min_downloads: int = 10000,
    top_k: int = 6,
    config_dir_name: str = MODEL_CONFIG_DIR_NAME,
) -> None:
    """Dump model configs from HuggingFace by type and task to `config_dir_name`

    Args:
        model_type (str, optional): model type, e.g., gpt, llama, opt, bloom. Defaults to "opt".
        task (str, optional): model task, e.g., text-generation, fill-mask. Defaults to "text-generation".
        min_downloads (int, optional): minimal number of downloads to filter the models. Defaults to 10000.
        top_k (int, optional): _description_. Defaults to 6.
        config_dir_name (str, optional): _description_. Defaults to MODEL_CONFIG_DIR_NAME.
    """
    model_list = get_hf_models_by_type_and_task(
        model_type=model_type,
        task=task,
        min_downloads=min_downloads,
        top_k=top_k,
        full_info=False,
    )
    for m in model_list:
        dump_model_config_by_name(m, config_dir_name)
    logger.info(
        f"In total, dumped {len(model_list)} model configs of model_type={model_type}, task={task}, to {config_dir_name}"
    )


populate_model_and_gpu_configs()

if __name__ == "__main__":
    logger.setLevel(logging.getLevelName("INFO"))
    fire.Fire(
        {
            "list_model_configs":
            list_model_configs,
            "list_gpu_configs":
            list_gpu_configs,
            "list_dtype_configs":
            list_dtype_configs,
            "get_model_config_by_name":
            get_model_config_by_name,
            "get_gpu_config_by_name":
            get_gpu_config_by_name,
            "get_dtype_config_by_name":
            get_dtype_config_by_name,
            "get_hf_models_by_type_and_task":
            get_hf_models_by_type_and_task,
            "dump_model_config_by_name":
            dump_model_config_by_name,
            "dump_hf_model_configs_by_type_and_task":
            dump_hf_model_configs_by_type_and_task,
        },
        serialize=lambda x: (json.dumps(x, cls=EnhancedJSONEncoder, indent=4)
                             if dataclasses.is_dataclass(x) else x),
    )
