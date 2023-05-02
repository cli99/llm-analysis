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
from dataclasses import dataclass
from pathlib import Path

import fire

from llm_analysis.constant import (
    DTYPE_CONFIG_DIR_NAME,
    GPU_CONFIG_DIR_NAME,
    MODEL_CONFIG_DIR_NAME,
)
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
    name: str  # model config name
    num_layers: int  # number of transformer layers (blocks)
    n_head: int  # number of attention heads
    hidden_dim: int  # hidden dimension
    vocab_size: int  # vocabulary size
    max_seq_len: int = None  # max sequence length
    ffn_embed_dim: int = (
        None  # hidden dimension of FFN, default to 4 * hidden_dim
    )
    model_type: str = (
        None  # model type as tagged on Hugging Face (e.g., gpt2, opt, llama.)
    )

    def __post_init__(self):
        if self.ffn_embed_dim is None:
            self.ffn_embed_dim = self.hidden_dim * 4

    def __str__(self):
        return dataclasses.asdict(self).__str__()


@dataclass
class GPUConfig:
    name: str  # GPU config name
    mem_per_GPU_in_GB: float  # memory per GPU in GB
    hbm_bandwidth_in_GB_per_sec: float  # GPU HBM bandwidth in GB/s
    intra_node_bandwidth_in_GB_per_sec: float  # intra node GPU bandwidth in GB/s
    intra_node_min_message_latency: float  # minimum intra node message latency in seconds
    peak_fp16_TFLOPS: float  # peak Tensor TFLOPS for FP16
    peak_i8_TFLOPS: float = None  # peak Tensor TFLOPS for INT8
    peak_i4_TFLOPS: float = None  # peak Tensor TFLOPS for INT4
    inter_node_bandwidth_in_GB_per_sec: float = 200  # inter node bandwidth in GB/s, assuming Mellanox 200Gbps HDR Infiniband

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


@dataclass
class ParallelismConfig:
    tp_size: int = 1  # tensor parallelism size, Megatron-LM tensor parallelism implementation
    pp_size: int = 1  # pipeline parallelism size, Megatron-LM pipeline parallelism implementation
    dp_size: int = (
        1  # data parallelism size, DeepSpeed Zero parallelism implementation
    )
    sp_size: int = 1  # sequence parallelism size, Megatron-LM sequence parallelism implementation


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
            Path(__file__).parent / Path(config_dir_name, f"{k}.json"), "w"
        ) as f:
            json.dump(v, f, cls=EnhancedJSONEncoder, indent=4)
    logger.info(f"dumped {len(configs)} configs to {config_dir_name}")


def get_model_config_from_hf(
    name: str,
) -> ModelConfig:
    """Get model config from HuggingFace transformers library `AutoConfig`; if
    the model does not exist, try updating the transformers library.

    Args:
        name (str): the model id of a pretrained model configuration hosted inside a model repo on huggingface.co

    Returns:
        ModelConfig: a dataclass for llm-analysis model config
    """
    if AutoConfig is None:
        logger.warning(
            f"cannot import AutoConfig from transformers, `transformers` is not installed, HuggingFace will not be available to use for model config retrieval"
        )
        return None
    hf_config = AutoConfig.from_pretrained(name, trust_remote_code=True)
    config = ModelConfig(
        name=canonical_model_name(name),
        max_seq_len=hf_config.max_position_embeddings
        if hasattr(hf_config, "max_position_embeddings")
        else None,
        num_layers=hf_config.num_hidden_layers,
        n_head=hf_config.num_attention_heads,
        hidden_dim=hf_config.hidden_size,
        vocab_size=hf_config.vocab_size,
        model_type=hf_config.model_type
        if hasattr(hf_config, "model_type")
        else None,
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
    """Get a HuggingFace model name list by model type and task, filtered by
    popularity (minimal number of downloads)

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
    """Populate model, gpu, and data type configs from the pre-defined json
    files."""
    global model_configs, gpu_configs, dtype_configs
    model_configs = read_configs(
        Path(__file__).parent / Path(MODEL_CONFIG_DIR_NAME), type="model"
    )
    gpu_configs = read_configs(
        Path(__file__).parent / Path(GPU_CONFIG_DIR_NAME), type="gpu"
    )

    dtype_configs = read_configs(
        Path(__file__).parent / Path(DTYPE_CONFIG_DIR_NAME), type="dtype"
    )
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


def get_model_config_by_name(name: str) -> ModelConfig:
    """Get model config from the populated mapping by name, if not found, try
    to get it from HuggingFace."""
    if name in model_configs:
        return model_configs[name]
    model_config = get_model_config_from_hf(name)
    if model_config is None:
        raise (
            f"unknown model config name: {name}, and none found on HuggingFace Hub"
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


def dump_model_config_by_name(
    name: str, config_dir_name: str = MODEL_CONFIG_DIR_NAME
) -> None:
    """Dump a model config from either the populated `model_configs` or Hugging
    Face by name to `config_dir_name`

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
    """Dump model configs from HuggingFace by type and task to
    `config_dir_name`

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
            "list_model_configs": list_model_configs,
            "list_gpu_configs": list_gpu_configs,
            "list_dtype_configs": list_dtype_configs,
            "get_model_config_by_name": get_model_config_by_name,
            "get_gpu_config_by_name": get_gpu_config_by_name,
            "get_dtype_config_by_name": get_dtype_config_by_name,
            "get_hf_models_by_type_and_task": get_hf_models_by_type_and_task,
            "dump_model_config_by_name": dump_model_config_by_name,
            "dump_hf_model_configs_by_type_and_task": dump_hf_model_configs_by_type_and_task,
        },
        serialize=lambda x: json.dumps(x, cls=EnhancedJSONEncoder, indent=4)
        if dataclasses.is_dataclass(x)
        else x,
    )
