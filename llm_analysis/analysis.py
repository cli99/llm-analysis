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

import json
import logging
import os
from enum import Enum
from functools import total_ordering
from pprint import pformat
from typing import Union

import fire

from llm_analysis.config import (
    DtypeConfig,
    GPUConfig,
    ModelConfig,
    ParallelismConfig,
    get_dtype_config_by_name,
    get_gpu_config_by_name,
    get_model_config_by_name,
)
from llm_analysis.constant import *
from llm_analysis.logger import logger
from llm_analysis.utils import _latency_to_string, _num_to_string, within_range


class ActivationRecomputation(Enum):
    NONE = 0
    """No activation recomputation; requires the most amount of memory."""

    SELECTIVE = 1
    """Selectively checkpoints and recomputes only parts of each transformer layer that
    take up a considerable amount of memory but are not computationally expensive to
    recompute, i.e. QK^T matrix multiply, softmax, softmax dropout, and attention over
    V."""

    FULL = 2
    """Full activation recomputation stores the input to EVERY transformer layer, which
    is sharded across the tensor parallel group, thus requiring an extra all-gather
    (ignored for now) per layer and add communication overhead; requires the lease
    amount of memory; requires an extra forward pass."""


@total_ordering
class DSZeRO(Enum):
    NONE = 0
    """No DeepSPeed ZeRO; requires the most amount of memory."""

    STAGE_1 = 1
    """ZeRO stage 1 shards the optimizer states across the data parallel group."""

    STAGE_2 = 2
    """ZeRO stage 2 shards the optimizer states and gradients across the data parallel
    group."""

    STAGE_3 = 3
    """ZeRO stage 3 shards the optimizer states, gradients, and model weights across the
    data parallel group."""

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class LLMAnalysis:
    """Given the specified model, GPU, data type, parallelism
    configuration/implementation, LLMAnalysis estimates the latency and memory usage of
    LLMs for training or inference.

    Refer to the `train` and `infer` entry functions for usage details.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        gpu_config: GPUConfig,
        dtype_config: DtypeConfig = DtypeConfig(),
        parallelism_config: ParallelismConfig = ParallelismConfig(),
        achieved_tflops: float = None,
        achieved_memory_bandwidth_GBs: float = None,
        flops_efficiency: float = None,
        hbm_memory_efficiency: float = None,
        intra_node_memory_efficiency: float = INTRA_NODE_MEMORY_EFFICIENCY,
        inter_node_memory_efficiency: float = INTER_NODE_MEMORY_EFFICIENCY,
    ) -> None:
        """LLMAnalysis constructor.

        Args:
            model_config (ModelConfig): model configuration
            gpu_config (GPUConfig): GPU configuration
            dtype_config (DtypeConfig, optional): data type configuration. Defaults to DtypeConfig().
            parallelism_config (ParallelismConfig, optional): parallelism configuration. Defaults to ParallelismConfig().
            achieved_tflops (float, optional): achieved TFLOPS per GPU. If specified, will override the flops_efficiency passed in. Defaults to None.
            achieved_memory_bandwidth_GBs (float, optional): achieved GPU memory bandwidth in GB/s. If specified, will override the hbm_memory_efficiency passed in. Defaults to None.
            flops_efficiency (float, optional): flops efficiency, ranging from 0 to 1. Defaults to None.
            hbm_memory_efficiency (float, optional): GPU HBM memory efficiency, ranging from 0 to 1. Defaults to None.
            intra_node_memory_efficiency (float, optional): intra-node memory efficiency, ranging from 0 to 1. Defaults to INTRA_NODE_MEMORY_EFFICIENCY.
            inter_node_memory_efficiency (float, optional): inter-node memory efficiency, ranging from 0 to 1. Defaults to INTER_NODE_MEMORY_EFFICIENCY.
        """
        self.model_config = model_config
        self.gpu_config = gpu_config
        self.parallelism_config = parallelism_config
        self.dtype_config = dtype_config
        self.intra_node_memory_efficiency = intra_node_memory_efficiency
        self.inter_node_memory_efficiency = inter_node_memory_efficiency

        if achieved_memory_bandwidth_GBs and hbm_memory_efficiency:
            logger.info(
                "both achieved_memory_bandwidth_GBs and hbm_memory_efficiency are set, using achieved_memory_bandwidth_GBs({achieved_memory_bandwidth_GBs} GB/s) to calculate hbm_memory_efficiency"
            )
            self.hbm_memory_efficiency = (
                achieved_memory_bandwidth_GBs /
                gpu_config.hbm_bandwidth_in_GB_per_sec)
        elif hbm_memory_efficiency:
            self.hbm_memory_efficiency = hbm_memory_efficiency
        elif achieved_memory_bandwidth_GBs:
            self.hbm_memory_efficiency = (
                achieved_memory_bandwidth_GBs /
                gpu_config.hbm_bandwidth_in_GB_per_sec)
        else:
            self.hbm_memory_efficiency = HBM_MEMORY_EFFICIENCY

        assert self.hbm_memory_efficiency > 0 and self.hbm_memory_efficiency <= 1, (
            "hbm_memory_efficiency must be in (0, 1], check the achieved_memory_bandwidth_GBs and hbm_memory_efficiency passed in"
        )
        logger.info(f"hbm_memory_efficiency: {self.hbm_memory_efficiency}")

        if achieved_tflops and flops_efficiency:
            logger.info(
                "both achieved_tflops and flops_efficiency are set, using achieved_tflops({achieved_tflops} TFLOPS) to calculate flops_efficiency"
            )
            self.flops_efficiency = (achieved_tflops /
                                     gpu_config.peak_fp16_TFLOPS)
        elif flops_efficiency:
            self.flops_efficiency = flops_efficiency
        elif achieved_tflops:
            self.flops_efficiency = (achieved_tflops /
                                     gpu_config.peak_fp16_TFLOPS)
        else:
            self.flops_efficiency = FLOPS_EFFICIENCY

        assert self.flops_efficiency > 0 and self.flops_efficiency <= 1, (
            "flops_efficiency must be in (0, 1], check the achieved_tflops and"
            " flops_efficiency passed in")
        logger.info(f"flops_efficiency: {self.flops_efficiency}")
        if self.flops_efficiency > 0.55:
            logger.warning(
                "Note that Megatron-LM reported up to 0.6 flops efficiency for large scale model training"
            )

        if self.parallelism_config.sp_size > 1:
            assert (
                self.parallelism_config.sp_size ==
                self.parallelism_config.tp_size
            ), ("if sequence parallelism size > 1, it must be equal to tensor"
                " parallelism size using Megatron-LM sequence"
                " parallelism")

        self.total_num_params = self.get_num_params_total()
        self.total_num_active_params = self.get_num_active_params_total()

    def update_model_config(self, model_config: ModelConfig) -> None:
        self.model_config = model_config

    def update_gpu_config(self, gpu_config: GPUConfig) -> None:
        self.gpu_config = gpu_config

    def update_dtype_config(self, dtype_config: DtypeConfig) -> None:
        self.dtype_config = dtype_config

    def update_parallelism_config(
            self, parallelism_config: ParallelismConfig) -> None:
        self.parallelism_config = parallelism_config

    def update_intra_node_memory_efficiency(
            self, intra_node_memory_efficiency: float) -> None:
        self.intra_node_memory_efficiency = intra_node_memory_efficiency

    def update_inter_node_memory_efficiency(
            self, inter_node_memory_efficiency: float) -> None:
        self.inter_node_memory_efficiency = inter_node_memory_efficiency

    def update_float_efficiency(self, flops_efficiency: float) -> None:
        self.flops_efficiency = flops_efficiency

    def get_gpu_hbm_bandwidth(self) -> float:
        return (self.gpu_config.hbm_bandwidth_in_GB_per_sec *
                self.hbm_memory_efficiency)

    def get_intra_node_bandwidth(self) -> float:
        return (self.gpu_config.intra_node_bandwidth_in_GB_per_sec *
                self.intra_node_memory_efficiency)

    def get_inter_node_bandwidth(self) -> float:
        return (self.gpu_config.inter_node_bandwidth_in_GB_per_sec *
                self.inter_node_memory_efficiency)

    def get_TFLOPS_per_gpu(self) -> float:
        """Get the expected TFLOPS per GPU for the specified data type configuration/GPU
        (adjusted by flops_efficiency)

        Returns:
            float: TFLOPS per GPU
        """
        wbits = self.dtype_config.weight_bits
        abits = self.dtype_config.activation_bits
        higher_bits = max(
            wbits, abits)  # gemm dtype/TFLOPS is determined by the higher bits
        if higher_bits == 4:
            gemm_TFOPS = self.gpu_config.peak_i4_TFLOPS
        elif higher_bits == 8:
            gemm_TFOPS = self.gpu_config.peak_i8_TFLOPS
        else:
            assert (higher_bits == 16
                    ), "weight_bits and activation_bits must be 4, 8, or 16"
            gemm_TFOPS = self.gpu_config.peak_fp16_TFLOPS
        return gemm_TFOPS * self.flops_efficiency

    def get_pivot(self) -> float:
        """Return the pivot point, defined as (model_weights / hbm_bandwidth) /
        (model_flops / TFLOPS_per_gpu)

        Returns:
            float: pivot point
        """
        pivot = (self.get_TFLOPS_per_gpu() * 10**3 *
                 self.dtype_config.activation_bits / BITS_PER_BYTE /
                 self.get_gpu_hbm_bandwidth() / 2)
        return pivot

    def get_num_params_embedding(self, shared_embedding: bool = True) -> int:
        """Get the number of parameters in the embedding layer.

        Args:
            shared_embedding (bool, optional):  whether the output embedding \
                shares weights with the input embedding. Defaults to True.

        Returns:
            int: the number of parameters in the embedding layer
        """
        num_params_input_embedding = (self.model_config.hidden_dim *
                                      self.model_config.vocab_size)
        num_params_output_embedding = (self.model_config.hidden_dim *
                                       self.model_config.vocab_size
                                       if not shared_embedding else 0)
        return num_params_input_embedding + num_params_output_embedding

    def get_num_params_per_layer_attn(self) -> int:
        """Get the number of parameters in the attention linear layers, including the
        query/key/value projection and output matrices.

        Returns:
            int: the number of parameters in the attention linear layers
        """
        num_heads_per_gpu = max(
            self.model_config.num_key_value_heads /
            self.parallelism_config.tp_size,
            1)  # At least on attention head on each tensor-parallel GPU
        num_key_value_heads = num_heads_per_gpu * self.parallelism_config.tp_size

        return 2 * self.model_config.hidden_dim**2 + 2 * self.model_config.hidden_dim * (
            self.model_config.hidden_dim * num_key_value_heads /
            self.model_config.n_head)

    def get_num_params_per_layer_mlp(self) -> int:
        """Get the number of parameters in the MLP linear layers, including the
        intermediate and output matrices.

        Returns:
            int: the number of parameters in the two MLP linear layers
        """
        return 2 * self.model_config.hidden_dim * self.model_config.ffn_embed_dim * self.model_config.moe_num_experts

    def get_num_params_per_layer_router(self) -> int:
        if self.model_config.moe_num_experts > 1:
            return self.model_config.hidden_dim * self.model_config.moe_num_experts
        else:
            return 0

    def get_num_params_per_layer_layernorm(self) -> int:
        return 2 * self.model_config.hidden_dim

    def get_num_params_last_layernorm(self) -> int:
        return self.model_config.hidden_dim

    def get_num_params_per_layer(self) -> int:
        """Get the number of parameters in a transformer layer, including the attention
        and MLP linear layers.

        Returns:
            int: the number of parameters in a transformer layer
        """

        return (self.get_num_params_per_layer_attn() +
                self.get_num_params_per_layer_mlp() +
                self.get_num_params_per_layer_router() +
                self.get_num_params_per_layer_layernorm())

    def get_num_active_params_per_layer(self) -> int:
        """Get the number of active parameters in a transformer layer, including the
        attention and MoE MLP linear layers.

        Returns:
            int: the number of parameters in a transformer layer
        """

        return (
            self.get_num_params_per_layer_attn() +
            self.get_num_params_per_layer_mlp() * self.model_config.moe_top_k /
            self.model_config.moe_num_experts +
            self.get_num_params_per_layer_router() +
            self.get_num_params_per_layer_layernorm())

    def get_num_params_total(self) -> int:
        """Get the total number of parameters in the model, including all the
        transformer layers and the embedding layer.

        Returns:
            int: the total number of parameters in the model
        """
        return (
            self.model_config.num_layers * self.get_num_params_per_layer() +
            self.get_num_params_embedding() +
            self.get_num_params_last_layernorm())

    def get_num_active_params_total(self) -> int:
        """Get the total number of parameters in the model, including all the
        transformer layers and the embedding layer.

        Returns:
            int: the total number of parameters in the model
        """
        return (self.model_config.num_layers *
                self.get_num_active_params_per_layer() +
                self.get_num_params_embedding() +
                self.get_num_params_last_layernorm())

    def get_weight_memory_per_layer(
            self,
            ds_zero: DSZeRO = DSZeRO.NONE,
            return_breakdown: bool = False) -> Union[float, tuple]:
        """Get the memory (in bytes) required to store the weights of a transformer
        layer, given the number of parameters in a transformer layer, the data type used
        for the weights, the tensor parallelism size, and the DeepSpeed ZeRO stage. WIth
        ZeRO Stage 3, the weights are sharded across data parallel groups.

        Args:
            ds_zero (DSZeRO, optional): which DeepSpeed ZeRO stage to use. Defaults to DSZeRO.NONE (disabled).

        Returns:
            Union[float, tuple]: the memory (in bytes) required to store the weights of a transformer layer, or a tuple of its breakdown
        """
        if ds_zero == DSZeRO.STAGE_3:
            sharded_dp_size = self.parallelism_config.dp_size
            mlp_sharded_dp_size = self.parallelism_config.dp_size / self.parallelism_config.ep_size
        else:
            sharded_dp_size = 1
            mlp_sharded_dp_size = 1

        weight_memory_attn_per_layer = self.get_num_params_per_layer_attn(
        ) * self.dtype_config.weight_bits / BITS_PER_BYTE / self.parallelism_config.tp_size / sharded_dp_size

        weight_memory_mlp_per_layer = (
            self.get_num_params_per_layer_mlp() /
            self.parallelism_config.ep_size +
            self.get_num_params_per_layer_router()
        ) * self.dtype_config.weight_bits / BITS_PER_BYTE / self.parallelism_config.tp_size / mlp_sharded_dp_size

        weight_memory_layernorm_per_layer = self.get_num_params_per_layer_layernorm(
        ) * self.dtype_config.weight_bits / BITS_PER_BYTE / self.parallelism_config.tp_size / sharded_dp_size

        weight_memory_per_layer = weight_memory_attn_per_layer + weight_memory_mlp_per_layer + weight_memory_layernorm_per_layer

        logger.info(
            f'weight_memory_attn_per_layer: {_num_to_string(weight_memory_attn_per_layer)}B, weight_memory_mlp_per_layer: {_num_to_string(weight_memory_mlp_per_layer)}B, weight_memory_layernorm_per_layer: {_num_to_string(weight_memory_layernorm_per_layer)}B'
        )

        if return_breakdown:
            return weight_memory_per_layer, weight_memory_attn_per_layer, weight_memory_mlp_per_layer, weight_memory_layernorm_per_layer
        return weight_memory_per_layer

    def get_weight_memory_last_layernorm(self, ds_zero: DSZeRO = DSZeRO.NONE):
        weight_memory_last_layernorm = self.get_num_params_last_layernorm(
        ) * self.dtype_config.weight_bits / BITS_PER_BYTE / self.parallelism_config.tp_size
        if ds_zero == DSZeRO.STAGE_3:
            weight_memory_last_layernorm /= self.parallelism_config.dp_size
        return weight_memory_last_layernorm

    def get_memory_optimizer_state_and_gradient_embedding(
        self,
        master_weights_dtype_bytes: int = BYTES_FP32,
        other_op_bytes: int = None,
        ds_zero: DSZeRO = DSZeRO.NONE,
    ) -> tuple:
        if other_op_bytes is None:
            op_bytes_per_params = BYTES_FP32 + 2 * BYTES_FP32  # adam optimizer
        else:
            op_bytes_per_params = (other_op_bytes + master_weights_dtype_bytes)

        memory_optimizer_state_embedding = op_bytes_per_params * self.get_num_params_embedding(
        ) / self.parallelism_config.tp_size
        if ds_zero >= DSZeRO.STAGE_1:
            memory_optimizer_state_embedding /= self.parallelism_config.dp_size

        memory_gradient_embedding = master_weights_dtype_bytes * self.get_num_params_embedding(
        ) / self.parallelism_config.tp_size
        if ds_zero >= DSZeRO.STAGE_2:
            memory_gradient_embedding /= self.parallelism_config.dp_size

        return memory_optimizer_state_embedding, memory_gradient_embedding

    def get_memory_optimizer_state_and_gradient_per_layer(
        self,
        master_weights_dtype_bytes: int = BYTES_FP32,
        other_op_bytes: int = None,
        ds_zero: DSZeRO = DSZeRO.NONE,
    ) -> tuple:
        """Get the memory (in bytes) required to store the gradients and optimizer
        states of a transformer layer. The optimizer states include the master weights
        and other states such as momentum. The gradients need to be upcasted to the same
        data type as the optimizer master weights before being applied.

        The default assumes using Adam optimizer (https://arxiv.org/abs/1412.6980), which requires the full-precision master weights (`master_weights_dtype_bytes=4`), momentum and variance (`other_op_bytes=8`).
        For other optimizers, use `master_weights_dtype_bytes` and `other_op_bytes` to express the bytes needed. For example, with lion optimizer (https://arxiv.org/abs/2302.06675), `other_op_bytes = 4` as it only requires FP32 momentum.

        With DeepSpeed ZeRO stage 1 and above, the optimizer states are sharded across data parallel groups.
        With ZeRO stage 2 and above, the gradients are sharded across the data parallel group.
        With FSDP SHARD_GRAD_OP or FULL_SHARD, the gradients and optimizer states are sharded across data parallel groups.

        Args:
            master_weights_dtype_bytes (int): the number of bytes in the data type for the optimizer master weights. Defaults to BYTES_FP32.
            other_op_bytes (int): the number of bytes in the optimizer state. Defaults to None, which assumes using Adam optimizer.
            ds_zero (DSZeRO, optional): which DeepSpeed ZeRO stage to use. Defaults to DSZeRO.NONE (disabled, no sharding).

        Returns:
            tuple: a tuple of the memory (in bytes) required to store the optimizer states and gradients of a transformer layer
        """

        if ds_zero >= DSZeRO.STAGE_1:
            sharded_dp_size = self.parallelism_config.dp_size
            mlp_sharded_dp_size = self.parallelism_config.dp_size / self.parallelism_config.ep_size
        else:
            sharded_dp_size = 1
            mlp_sharded_dp_size = 1

        if other_op_bytes is None:
            op_bytes_per_params = BYTES_FP32 + 2 * BYTES_FP32  # adam optimizer
        else:
            op_bytes_per_params = (other_op_bytes + master_weights_dtype_bytes)

        memory_optimizer_state_mlp_per_layer = op_bytes_per_params * self.get_num_params_per_layer_mlp(
        ) / self.parallelism_config.ep_size / self.parallelism_config.tp_size / mlp_sharded_dp_size

        memory_optimizer_state_others_per_layer = op_bytes_per_params * (
            (self.get_num_params_per_layer_attn() +
             +self.get_num_params_per_layer_router() +
             self.get_num_params_per_layer_layernorm())
        ) / self.parallelism_config.tp_size / sharded_dp_size

        memory_optimizer_state_per_layer = memory_optimizer_state_mlp_per_layer + memory_optimizer_state_others_per_layer

        if ds_zero >= DSZeRO.STAGE_2:
            sharded_dp_size = self.parallelism_config.dp_size
            mlp_sharded_dp_size = self.parallelism_config.dp_size / self.parallelism_config.ep_size
        else:
            sharded_dp_size = 1
            mlp_sharded_dp_size = 1

        memory_gradient_mlp_per_layer = 1 * self.get_num_params_per_layer_mlp(
        ) * master_weights_dtype_bytes / self.parallelism_config.ep_size / self.parallelism_config.tp_size / mlp_sharded_dp_size

        memory_gradient_others_per_layer = 1 * (
            self.get_num_params_per_layer_attn() +
            +self.get_num_params_per_layer_router() +
            self.get_num_params_per_layer_layernorm()
        ) * master_weights_dtype_bytes / self.parallelism_config.tp_size / sharded_dp_size

        memory_gradient_per_layer = memory_gradient_mlp_per_layer + memory_gradient_others_per_layer

        return memory_optimizer_state_per_layer, memory_gradient_per_layer

    def get_memory_optimizer_state_and_gradient_last_layernorm(
        self,
        master_weights_dtype_bytes: int = BYTES_FP32,
        other_op_bytes: int = None,
        ds_zero: DSZeRO = DSZeRO.NONE,
    ) -> tuple:
        if other_op_bytes is None:
            op_bytes_per_params = BYTES_FP32 + 2 * BYTES_FP32  # adam optimizer
        else:
            op_bytes_per_params = (other_op_bytes + master_weights_dtype_bytes)

        memory_optimizer_state_last_layernorm = op_bytes_per_params * self.get_num_params_last_layernorm(
        ) / self.parallelism_config.tp_size
        if ds_zero >= DSZeRO.STAGE_1:
            memory_optimizer_state_last_layernorm /= self.parallelism_config.dp_size

        memory_gradient_last_layernorm = master_weights_dtype_bytes * self.get_num_params_last_layernorm(
        ) / self.parallelism_config.tp_size
        if ds_zero >= DSZeRO.STAGE_2:
            memory_gradient_last_layernorm /= self.parallelism_config.dp_size

        return memory_optimizer_state_last_layernorm, memory_gradient_last_layernorm

    def get_memory_embedding(
        self,
        ds_zero: DSZeRO = DSZeRO.NONE,
    ) -> float:
        """Get the memory (in bytes) required to store the embedding layer, given the
        number of parameters in the embedding layer, the data type (defaults to FP32)
        used for the weights, and the tensor parallelism size (Megatron-LM partitions
        the embedding layer across the tensor parallel groups).

        Args:
            ds_zero (DSZeRO, optional): which DeepSpeed ZeRO stage to use. Defaults to DSZeRO.NONE (disabled, no sharding).

        Returns:
            float: the memory (in bytes) required to store the embedding layer
        """
        dtype_bytes = self.dtype_config.embedding_bits / BITS_PER_BYTE
        memory_embedding = (self.get_num_params_embedding() /
                            self.parallelism_config.tp_size) * dtype_bytes
        if ds_zero == DSZeRO.STAGE_3:
            memory_embedding /= self.parallelism_config.dp_size

        return memory_embedding

    def get_activation_memory_per_layer_attn(
        self,
        batch_size: int,
        seq_len: int,
        is_inference: bool = True,
        flash_attn: bool = True,
        softmax_dropout: bool = False,
        attn_dropout: bool = True,
        activation_recomputation:
        ActivationRecomputation = ActivationRecomputation.NONE,
    ) -> float:
        """Get the memory (in bytes) required to store the activations of the
        attention in a transformer layer, given the batch size, sequence
        length, whether it is inference or training, the activation
        recomputation strategy, and the activation data type. The `attn`
        activations include the input to Q/K/V gemm, QK^T matrix multiply,
        softmax, softmax dropout attention over V, the input to the attention
        output Gemm; if training, also include the softmax dropout mask and
        attention dropout mask; Refer to https://arxiv.org/abs/2205.05198 for
        details.

        Args:
            batch_size (int): micro batch size
            seq_len (int): sequence length
            is_inference (bool, optional): whether it is inference or not. Defaults to True.
            flash_attn (bool, optional): whether to use Flash Attention. Defaults to True.
            softmax_dropout (bool, optional): whether to apply dropout after softmax. Defaults to False.
            activation_recomputation (ActivationRecomputation, optional): activation recomputation strategy. Defaults to ActivationRecomputation.NONE.

        Returns:
            float: the memory (in bytes) required to store the activations of the attention in a transformer layer
        """
        tp_size = self.parallelism_config.tp_size
        sp_size = self.parallelism_config.sp_size
        hidden_dim = self.model_config.hidden_dim
        n_head = self.model_config.n_head
        bytes_per_activation = (self.dtype_config.activation_bits /
                                BITS_PER_BYTE)

        if (not is_inference
            ) and activation_recomputation == ActivationRecomputation.FULL:
            return (seq_len * batch_size * hidden_dim * bytes_per_activation /
                    sp_size)

        attn_compute = 0
        if activation_recomputation != activation_recomputation.SELECTIVE:
            if flash_attn:
                memory_attn_compute = (2 * seq_len * batch_size * hidden_dim +
                                       4 * n_head * seq_len * batch_size
                                       ) * bytes_per_activation / tp_size
            else:
                memory_attn_compute = 2 * n_head * seq_len**2 * batch_size * bytes_per_activation / tp_size
            if softmax_dropout:
                # dropout mask only requires a single byte per element
                memory_attn_compute += n_head * seq_len**2 * batch_size / tp_size
        else:
            memory_attn_compute = 0

        if is_inference:
            return max(
                3 * bytes_per_activation * seq_len * batch_size * hidden_dim /
                sp_size, memory_attn_compute)

        activation_memory_per_layer_attn = (
            seq_len * batch_size * hidden_dim / sp_size +
            4 * seq_len * batch_size * hidden_dim /
            tp_size) * bytes_per_activation + memory_attn_compute

        if attn_dropout:
            activation_memory_per_layer_attn += seq_len * batch_size * hidden_dim / sp_size

        return activation_memory_per_layer_attn

    def get_activation_memory_per_layer_mlp(
        self,
        batch_size: int,
        seq_len: int,
        is_inference: bool = True,
        activation_recomputation:
        ActivationRecomputation = ActivationRecomputation.NONE,
        mlp_activation_quant_bits: int = None,
        mlp_1linear_quant_bits: int = None,
        mlp_gelu_input_quant_bits: int = None,
        mlp_2linear_quant_bits: int = None,
        recompute_gelu: bool = False,
        gated_linear_units: bool = False,
        with_dropout: bool = False,
    ) -> float:
        """Get the memory (in bytes) required to store the activations of the
        MLP in a transformer layer, given the batch size, sequence length, and
        whether it is inference or training, the activation recomputation
        strategy, and the activation data type. The `mlp` activations include
        the input to the two linear layers. Refer to
        https://arxiv.org/abs/2205.05198 for details.

        Args:
            batch_size (int): micro batch size
            seq_len (int): sequence length
            is_inference (bool, optional): whether it is inference or not. Defaults to True.
            activation_recomputation (ActivationRecomputation, optional): activation recomputation strategy. Defaults to ActivationRecomputation.NONE.
            mlp_activation_quant_bits (int, optional): number of bits to quantize MLP activations; if set, override the values for mlp_1linear_quant_bits, mlp_gelu_input_quant_bits and mlp_2linear_quant_bits. Defaults to None.
            mlp_1linear_quant_bits (int, optional): number of bits to quantize the input activations of the first linear layer. Defaults to None.
            mlp_gelu_input_quant_bits (int, optional): number of bits to quantize the GELU input activations. Defaults to None.
            mlp_2linear_quant_bits (int, optional): number of bits to quantize the input activations of the second linear layer. Defaults to None.
            recompute_gelu (bool, optional): whether to recompute gelu in backward pass.
            gated_linear_units (bool, optional): whether to use gated linear units.

        Returns:
            float: the memory (in bytes) required to store the activations of the MLP in a transformer layer
        """
        if (not is_inference
            ) and activation_recomputation == ActivationRecomputation.FULL:
            return 0

        tp_size = self.parallelism_config.tp_size
        sp_size = self.parallelism_config.sp_size
        ep_size = self.parallelism_config.ep_size
        hidden_dim = self.model_config.hidden_dim

        bytes_per_activation = (self.dtype_config.activation_bits /
                                BITS_PER_BYTE)

        bytes_per_1linear_input = bytes_per_gelu_input = bytes_per_2linear_input = bytes_per_activation
        if mlp_1linear_quant_bits:
            bytes_per_1linear_input = mlp_1linear_quant_bits / BITS_PER_BYTE
        if mlp_gelu_input_quant_bits:
            bytes_per_gelu_input = mlp_gelu_input_quant_bits / BITS_PER_BYTE
        if mlp_2linear_quant_bits:
            bytes_per_2linear_input = mlp_2linear_quant_bits / BITS_PER_BYTE
        if mlp_activation_quant_bits:
            bytes_per_1linear_input = mlp_activation_quant_bits / BITS_PER_BYTE
            bytes_per_gelu_input = mlp_activation_quant_bits / BITS_PER_BYTE
            bytes_per_2linear_input = mlp_activation_quant_bits / BITS_PER_BYTE

        num_experts_per_gpu = self.model_config.moe_num_experts / ep_size

        if is_inference:
            return max(
                bytes_per_1linear_input,
                bytes_per_gelu_input * self.model_config.expansion_ratio
            ) * seq_len * batch_size * hidden_dim * self.model_config.moe_top_k / tp_size

        activation_memory_per_layer_mlp = bytes_per_1linear_input * seq_len * batch_size * hidden_dim * num_experts_per_gpu * self.model_config.moe_top_k / sp_size

        if recompute_gelu and gated_linear_units:
            # swiglu decreases the expansion ratio by 2/3 to get isoparam
            activation_memory_per_layer_mlp += (
                1 / 3 * bytes_per_gelu_input
            ) * self.model_config.expansion_ratio * seq_len * batch_size * hidden_dim * num_experts_per_gpu * self.model_config.moe_top_k / tp_size
        elif recompute_gelu:
            activation_memory_per_layer_mlp += bytes_per_gelu_input * self.model_config.expansion_ratio * seq_len * batch_size * hidden_dim * num_experts_per_gpu * self.model_config.moe_top_k / tp_size
        else:
            activation_memory_per_layer_mlp += (
                bytes_per_gelu_input + bytes_per_2linear_input
            ) * self.model_config.expansion_ratio * seq_len * batch_size * hidden_dim * num_experts_per_gpu * self.model_config.moe_top_k / tp_size

        # dropout mask only requires a single byte per element
        if with_dropout:
            drop_out_mask = seq_len * batch_size * hidden_dim / sp_size
            activation_memory_per_layer_mlp += drop_out_mask

        return activation_memory_per_layer_mlp

    def get_activation_memory_per_layernorm(
        self,
        batch_size: int,
        seq_len: int,
        activation_recomputation:
        ActivationRecomputation = ActivationRecomputation.NONE,
        dtype_bytes: int = BYTES_FP32,
    ) -> float:
        """Get the memory (in bytes) required to store the activations of a
        single layernorm in a transformer layer, given the batch size, sequence
        length. Refer to https://arxiv.org/abs/2205.05198 for details.

        Args:
            batch_size (int): micro batch size
            seq_len (int): sequence length
            activation_recomputation (ActivationRecomputation, optional): \
                activation recomputation strategy. Defaults to ActivationRecomputation.NONE.
            dtype_bytes (int, optional): number of bytes in the data type for the \
                layernorm activation. Defaults to BYTES_FP32. Need to be at least FP16 to maintain accuracy.

        Returns:
            float: the memory (in bytes) required to store the activations of a single layernorm in a transformer layer
        """
        if activation_recomputation == ActivationRecomputation.FULL:
            return 0
        return (seq_len * batch_size * self.model_config.hidden_dim /
                self.parallelism_config.sp_size) * dtype_bytes

    def get_activation_memory_output_embedding(self, batch_size: int,
                                               seq_len: int) -> float:
        """Get the memory (in bytes) required to store the activations of output embedding (logits)"""
        return self.model_config.vocab_size * batch_size * seq_len * self.dtype_config.activation_bits / BITS_PER_BYTE / self.parallelism_config.tp_size

    def get_activation_memory_per_layer(
        self,
        batch_size: int,
        seq_len: int,
        is_inference: bool = True,
        activation_recomputation:
        ActivationRecomputation = ActivationRecomputation.NONE,
        layernorm_dtype_bytes: int = BYTES_FP32,
        flash_attn: bool = True,
        softmax_dropout: bool = False,
        mlp_activation_quant_bits: int = None,
        mlp_1linear_quant_bits: int = None,
        mlp_gelu_input_quant_bits: int = None,
        mlp_2linear_quant_bits: int = None,
        mlp_recompute_gelu: bool = False,
        mlp_gated_linear_units: bool = False,
        return_breakdown: bool = False,
    ) -> Union[float, tuple]:
        """Get the memory (in bytes) required to store the activations of a
        transformer layer, given the batch size, sequence length, and whether
        it is inference or training, the activation recomputation strategy, and
        the activation data type. Refer to https://arxiv.org/abs/2205.05198 for
        details. For inference, this assumes the maximum tensor buffer reuse.

        Args:
            batch_size (int):
            seq_len (int): sequence length
            is_inference (bool, optional): whether it is inference or not. Return the max memory activation tensor size between layernorm/attn/mlp. Defaults to True.
            activation_recomputation (ActivationRecomputation, optional): \
                activation recomputation strategy. Defaults to ActivationRecomputation.NONE.
            layernorm_dtype_bytes (int, optional): number of bytes in the data type for \
                the layernorm activations. Defaults to BYTES_FP32. Often has to be FP32 in training to maintain model accuracy.
            flash_attn (bool, optional): whether to use Flash Attention. Defaults to True.
            softmax_dropout (bool, optional): whether to apply dropout after softmax. Defaults to False.
            mlp_activation_quant_bits (int, optional): number of bits to quantize MLP activations; if set, override the values for mlp_1linear_quant_bits, mlp_gelu_input_quant_bits and mlp_2linear_quant_bits. Defaults to None.
            mlp_1linear_quant_bits (int, optional): number of bits to quantize the input activations of the first linear layer. Defaults to None.
            mlp_gelu_input_quant_bits (int, optional): number of bits to quantize the GELU input activations. Defaults to None.
            mlp_2linear_quant_bits (int, optional): number of bits to quantize the input activations of the second linear layer. Defaults to None.            mlp_recompute_gelu (bool, optional): whether to recompute the gelu activation in the MLP backward pass. Defaults to False.
            mlp_gated_linear_units (bool, optional): whether to use gated linear units in the MLP. Defaults to False.
        Returns:
            Union[float, tuple]: the memory (in bytes) required to store the activations of a transformer layer or a tuple of its breakdown
        """
        if (not is_inference
            ) and activation_recomputation == ActivationRecomputation.FULL:
            activation_memory_per_layer = (seq_len * batch_size *
                                           self.model_config.hidden_dim *
                                           self.dtype_config.activation_bits /
                                           BITS_PER_BYTE /
                                           self.parallelism_config.tp_size)
            if return_breakdown:
                return activation_memory_per_layer, 0, 0, 0
            else:
                return activation_memory_per_layer

        activation_memory_per_layer_attn = (
            self.get_activation_memory_per_layer_attn(
                batch_size,
                seq_len,
                is_inference,
                flash_attn=flash_attn,
                softmax_dropout=softmax_dropout,
                activation_recomputation=activation_recomputation))

        activation_memory_per_layer_mlp = (
            self.get_activation_memory_per_layer_mlp(
                batch_size,
                seq_len,
                is_inference,
                activation_recomputation,
                mlp_activation_quant_bits=mlp_activation_quant_bits,
                mlp_1linear_quant_bits=mlp_1linear_quant_bits,
                mlp_gelu_input_quant_bits=mlp_gelu_input_quant_bits,
                mlp_2linear_quant_bits=mlp_2linear_quant_bits,
                recompute_gelu=mlp_recompute_gelu,
                gated_linear_units=mlp_gated_linear_units,
            ))

        activation_memory_per_layernorm = self.get_activation_memory_per_layernorm(
            batch_size,
            seq_len,
            activation_recomputation,
            layernorm_dtype_bytes,
        )

        if is_inference:
            activation_memory_per_layer = max(activation_memory_per_layer_attn,
                                              activation_memory_per_layer_mlp,
                                              activation_memory_per_layernorm)
            logger.info(
                f"activation_memory_per_layer for batch_size {batch_size}:"
                f" {_num_to_string(activation_memory_per_layer)}B"
                f" (max(attn, mlp, layernorm): max({_num_to_string(activation_memory_per_layer_attn)}B ,"
                f" {_num_to_string(activation_memory_per_layer_mlp)}B , 2 *"
                f" {_num_to_string(2*activation_memory_per_layernorm)}B))")
        else:
            activation_memory_per_layer = (activation_memory_per_layer_attn +
                                           activation_memory_per_layer_mlp +
                                           2 * activation_memory_per_layernorm)
            logger.info(
                f"activation_memory_per_layer for micro batch size {batch_size}:"
                f" {_num_to_string(activation_memory_per_layer)}B"
                f" (attn + mlp + layernorm: {_num_to_string(activation_memory_per_layer_attn)}B +"
                f" {_num_to_string(activation_memory_per_layer_mlp)}B + 2 *"
                f" {_num_to_string(activation_memory_per_layernorm)}B)")
        if return_breakdown:
            return activation_memory_per_layer, activation_memory_per_layer_attn, activation_memory_per_layer_mlp, 2 * activation_memory_per_layernorm
        return activation_memory_per_layer

    def get_memory_kv_cache_per_layer(
        self,
        batch_size: int,
        seq_len: int,
        kv_cache_dtype_bytes: int = None,
    ) -> float:
        """Get the memory (in bytes) required to store the key and value cache for a
        transformer layer in inference, given the batch size, sequence length,
        activation data type, and tensor parallelism size.

        Args:
            batch_size (int): batch size
            seq_len (int): sequence length
            kv_cache_dtype_bytes (int, optional): number of bytes in the data type for the kv_cache. Defaults to None. Often has to be at least FP16 in inference to maintain model accuracy.

        Returns:
            float: the memory (in bytes) required to store the key and value cache for a transformer layer in inference
        """
        if kv_cache_dtype_bytes is None:
            kv_cache_dtype_bytes = (self.dtype_config.activation_bits /
                                    BITS_PER_BYTE)
        head_dim = self.model_config.hidden_dim / self.model_config.n_head
        num_heads_per_gpu = max(
            self.model_config.num_key_value_heads /
            self.parallelism_config.tp_size,
            1)  # At least on attention head on each tensor-parallel GPU

        memory_kv_cache_per_layer = (2 * batch_size * seq_len * head_dim *
                                     num_heads_per_gpu) * kv_cache_dtype_bytes
        logger.debug(
            f"memory_kv_cache_per_layer = {_num_to_string(memory_kv_cache_per_layer)} B"
        )

        return memory_kv_cache_per_layer

    def get_num_flops_fwd_per_layer_attn(self, batch_size: int,
                                         seq_len: int) -> int:
        """Get the number of floating point operations (flops) for the forward pass of
        the attention module in a transformer layer, given the batch size and sequence
        length. The count is model-specific and does not depend on the parallelism
        strategy.

        Args:
            batch_size (int): batch size
            seq_len (int): sequence length

        Returns:
            int: the number of floating point operations for the forward pass of the attention module in a transformer layer
        """
        return (4 * batch_size * seq_len * self.model_config.hidden_dim**2 +
                4 * batch_size * seq_len * self.model_config.hidden_dim**2 /
                self.model_config.num_key_value_groups +
                4 * batch_size * seq_len**2 * self.model_config.hidden_dim
                )  # q proj + attn_out + k proj + v proj + compute_attn

    def get_num_flops_fwd_per_layer_mlp(self, batch_size: int,
                                        seq_len: int) -> int:
        """Get the number of floating point operations (flops) for the forward pass of
        the MLP module in a transformer layer, given the batch size and sequence length.
        The count is model-specific and does not depend on the parallelism strategy.s.

        Args:
            batch_size (int): batch size
            seq_len (int): sequence length

        Returns:
            int: the number of floating point operations for the forward pass of the MLP module in a transformer layer
        """
        return 16 * batch_size * seq_len * self.model_config.hidden_dim**2

    def get_num_flops_fwd_per_layer(
        self,
        batch_size: int,
        seq_len: int,
    ) -> int:
        """Get the number of floating point operations (flops) for the forward pass of a
        transformer layer, given the batch size and sequence length. The count is model-
        specific and does not depend on the parallelism strategy.

        Args:
            batch_size (int): batch size
            seq_len (int): sequence length

        Returns:
            int: the number of floating point operations for the forward pass of a transformer layer
        """
        return self.get_num_flops_fwd_per_layer_attn(
            batch_size, seq_len) + self.get_num_flops_fwd_per_layer_mlp(
                batch_size, seq_len)

    def get_num_flops_fwd_total(self, batch_size: int, seq_len: int) -> int:
        """Get the number of floating point operations (flops) for the forward pass of
        the entire transformer, given the batch size and sequence length. The count is
        model-specific and does not depend on the parallelism strategy.

        Args:
            batch_size (int): batch size
            seq_len (int): sequence length

        Returns:
            int: the number of floating point operations for the forward pass of the entire transformer
        """
        hidden_dim = self.model_config.hidden_dim
        vocab_size = self.model_config.vocab_size
        num_layers = self.model_config.num_layers

        num_flops_logit_layer = (2 * batch_size * seq_len * hidden_dim *
                                 vocab_size)  # logit compute

        num_flops_fwd_total = (
            self.get_num_flops_fwd_per_layer(batch_size, seq_len) * num_layers
            + num_flops_logit_layer)

        # validate only when using Multi Head Attention (MHA)
        if self.model_config.num_key_value_groups == 1:
            assert within_range(
                num_flops_fwd_total,
                (24 * batch_size * num_layers * seq_len * hidden_dim**2 *
                 (1 + seq_len / (6 * hidden_dim) + vocab_size /
                  (12 * num_layers * hidden_dim))),
                TOLERANCE,
            )

        return num_flops_fwd_total

    def get_num_flops_bwd_total(self, batch_size: int, seq_len: int) -> int:
        """Get the number of floating point operations (flops) for the backward pass of
        the entire transformer, estimated as the twice the number of flops for the
        forward pass. The count is model-specific and does not depend on the parallelism
        strategy.

        Args:
            batch_size (int): batch size
            seq_len (int): sequence length

        Returns:
            int: the number of floating point operations for the backward pass of the entire transformer
        """
        return 2 * self.get_num_flops_fwd_total(batch_size, seq_len)

    def get_num_flops_total_selective_recompute_attn(self, batch_size: int,
                                                     seq_len: int) -> int:
        """Get the number of floating point operations (flops) for recomputation when
        using selective activation recomputation. The count is model-specific and does
        not depend on the parallelism strategy.

        Args:
            batch_size (int): batch size
            seq_len (int): sequence length

        Returns:
            int: the number of floating point operations for recomputation when using selective activation recomputation
        """
        return (4 * batch_size * seq_len**2 *
                self.model_config.hidden_dim) * self.model_config.num_layers

    def get_latency_fwd_per_layer_attn(
        self,
        batch_size: int,
        seq_len: int,
        is_inference: bool = True,
        activation_recomputation:
        ActivationRecomputation = ActivationRecomputation.NONE,
    ) -> float:
        """Get the latency for the forward pass of the attention module in a transformer
        layer, given the batch size and sequence length. The latency is the max of the
        compute latency and the memory latency, assuming the compute and memory
        operations are perfectly overlapped.

        Args:
            batch_size (int): batch size
            seq_len (int): sequence length
            is_inference (bool, optional): whether it is inference or not. Defaults to True.
            activation_recomputation (ActivationRecomputation, optional): activation recomputation strategy. Defaults to ActivationRecomputation.NONE.

        Returns:
            float: the latency in seconds for the forward pass of the attention module in a transformer layer
        """
        tp_size = self.parallelism_config.tp_size

        compute_latency = (
            self.get_num_flops_fwd_per_layer_attn(batch_size, seq_len) /
            tp_size / (self.get_TFLOPS_per_gpu() * 10**12))

        weight_memory = (self.get_num_params_per_layer_attn() *
                         self.dtype_config.weight_bits / BITS_PER_BYTE)
        weight_memory_latency = (weight_memory / tp_size /
                                 (self.get_gpu_hbm_bandwidth() * 10**9))

        activation_memory = self.get_activation_memory_per_layer_attn(
            batch_size, seq_len, is_inference, activation_recomputation)
        activation_memory_latency = activation_memory / (
            self.get_gpu_hbm_bandwidth() * 10**9)

        memory_latency = weight_memory_latency + activation_memory_latency

        logger.debug(
            "latency_fwd_per_layer_attn:"
            f" {round(max(compute_latency, memory_latency)*1000, 3)} ms"
            " (max(compute_latency, weight_memory_latency+"
            " activation_memory_latency) ="
            f" max({round(compute_latency*1000, 3)},"
            f" ({round(weight_memory_latency*1000, 3)} +"
            f" {round(activation_memory_latency*1000, 3)})))")

        return max(compute_latency, memory_latency)

    def get_latency_fwd_per_layer_mlp(
        self,
        batch_size: int,
        seq_len: int,
        is_inference: bool = True,
        activation_recomputation:
        ActivationRecomputation = ActivationRecomputation.NONE,
    ) -> float:
        """Get the latency for the forward pass of the MLP module in a transformer
        layer, given the batch size and sequence length. The latency is the max of the
        compute latency and the memory latency, assuming the compute and memory
        operations are perfectly overlapped.

        Args:
            batch_size (int): batch size
            seq_len (int): sequence length
            is_inference (bool, optional): whether it is inference or not. Defaults to True.
            activation_recomputation (ActivationRecomputation, optional): activation recomputation strategy. Defaults to ActivationRecomputation.NONE.

        Returns:
            float: the latency in seconds for the forward pass of the MLP module in a transformer layer
        """
        tp_size = self.parallelism_config.tp_size

        compute_latency = (
            self.get_num_flops_fwd_per_layer_mlp(batch_size, seq_len) /
            tp_size / (self.get_TFLOPS_per_gpu() * 10**12))

        weight_memory = (self.get_num_params_per_layer_mlp() *
                         self.dtype_config.weight_bits / BITS_PER_BYTE)
        weight_memory_latency = (weight_memory / tp_size /
                                 (self.get_gpu_hbm_bandwidth() * 10**9))

        activation_memory = self.get_activation_memory_per_layer_mlp(
            batch_size, seq_len, is_inference, activation_recomputation)
        activation_memory_latency = activation_memory / (
            self.get_gpu_hbm_bandwidth() * 10**9)

        memory_latency = weight_memory_latency + activation_memory_latency

        logger.debug(
            "latency_fwd_per_layer_mlp:"
            f" {round(max(compute_latency, memory_latency)*1000, 3)} ms"
            " (max(compute_latency, weight_memory_latency+"
            " activation_memory_latency) ="
            f" max({round(compute_latency*1000, 3)},"
            f" ({round(weight_memory_latency*1000, 3)} +"
            f" {round(activation_memory_latency*1000, 3)})))")

        return max(compute_latency, memory_latency)

    def get_latency_fwd_per_layer_layernorm(
        self,
        batch_size: int,
        seq_len: int,
        activation_recomputation:
        ActivationRecomputation = ActivationRecomputation.NONE,
        dtype_bytes: int = BYTES_FP32,
    ) -> float:
        """Get the latency for the forward pass of a single layernorm in a transformer
        layer, given the batch size, sequence length, activation recomputation strategy,
        and data type. The latency is the memory latency as layernorm is a memory-bound
        operation.

        Args:
            batch_size (int): batch size
            seq_len (int): sequence length
            activation_recomputation (ActivationRecomputation, optional): activation recomputation strategy. Defaults to ActivationRecomputation.NONE.
            dtype_bytes (int, optional): number of bytes in the data type for the layernorm activation. Defaults to BYTES_FP32. Need to be at least FP16 to maintain accuracy.

        Returns:
            float: the latency in seconds for the forward pass of a single layernorm in a transformer layer
        """
        activation_memory = self.get_activation_memory_per_layernorm(
            batch_size,
            seq_len,
        )
        activation_memory_latency = activation_memory / (
            self.get_gpu_hbm_bandwidth() * 10**9)
        return activation_memory_latency

    def get_latency_fwd_per_layer_tp_comm(self, batch_size: int, seq_len: int,
                                          dtype_bytes: int) -> float:
        """Get the latency of a single allreduce communication across the tensor
        parallel group in the forward pass of a transformer layer, given the batch size,
        sequence length, and data type, and assuming  a ring allreduce implementation.
        The latency is the max of the latency for the allreduce and the minimum message
        latency through intra-node connect (Note that tensor parallelism size <= number
        of GPUs per node).

        Args:
            batch_size (int): batch size
            seq_len (int): sequence length
            dtype_bytes (int): number of bytes in the data type

        Returns:
            float: the latency in seconds for a single allreduce communication across the tensor parallel group in the forward pass of a transformer layer
        """
        tp_size = self.parallelism_config.tp_size
        if tp_size == 1:
            return 0

        elems_per_all_reduce = (2 * batch_size * seq_len *
                                self.model_config.hidden_dim * (tp_size - 1) /
                                tp_size)
        latency_per_all_reduce = (
            elems_per_all_reduce * dtype_bytes /
            (self.gpu_config.intra_node_bandwidth_in_GB_per_sec * 10**9))

        return max(
            latency_per_all_reduce,
            self.gpu_config.intra_node_min_message_latency,
        )

    def get_latency_fwd_per_layer(
        self,
        batch_size: int,
        seq_len: int,
        is_inference: bool = True,
        activation_recomputation:
        ActivationRecomputation = ActivationRecomputation.NONE,
        layernorm_dtype_bytes: int = BYTES_FP32,
    ) -> tuple:
        """Get the latency for the forward pass of a transformer layer, given the batch
        size, sequence length, training or inference, activation recomputation strategy,
        and layernorm data type. The latency is the sum of the latency for the attention
        module, MLP module, two layernorms, and two (Megatron-LM tp implementation)
        allreduce communications across the tensor parallel group.

        Args:
            batch_size (int): batch size
            seq_len (int): sequence length
            is_inference (bool, optional): whether it is inference or not. Defaults to True.
            activation_recomputation (ActivationRecomputation, optional): activation recomputation strategy. Defaults to ActivationRecomputation.NONE.
            layernorm_dtype_bytes (int, optional): number of bytes in the data type for the layernorm activations. Defaults to BYTES_FP32. Often has to be FP32 in training to maintain model accuracy.

        Returns:
            tuple: a tuple of the latency in seconds for the forward pass of a transformer layer and its breakdown dict
        """
        latency_fwd_per_layer_attn = self.get_latency_fwd_per_layer_attn(
            batch_size, seq_len, is_inference, activation_recomputation)

        latency_fwd_per_layer_mlp = self.get_latency_fwd_per_layer_mlp(
            batch_size, seq_len, is_inference, activation_recomputation)

        latency_fwd_per_layer_layernorm = (
            self.get_latency_fwd_per_layer_layernorm(
                batch_size,
                seq_len,
                activation_recomputation,
                layernorm_dtype_bytes,
            ))

        latency_fwd_per_layer_tp_comm = self.get_latency_fwd_per_layer_tp_comm(
            batch_size,
            seq_len,
            self.dtype_config.activation_bits / BITS_PER_BYTE,
        )

        latency_per_layer = (latency_fwd_per_layer_attn +
                             latency_fwd_per_layer_mlp +
                             2 * latency_fwd_per_layer_layernorm +
                             2 * latency_fwd_per_layer_tp_comm)

        logger.debug("latency_fwd_per_layer_layernorm:"
                     f" {round(latency_fwd_per_layer_layernorm*1000, 3)} ms,"
                     " latency_fwd_per_layer_tp_comm:"
                     f" {round(latency_fwd_per_layer_tp_comm*1000, 3)} ms")

        logger.debug(
            f"latency_per_layer: {round(latency_per_layer*1000, 3)} ms"
            f" ({round(latency_fwd_per_layer_attn*1000, 3)} +"
            f" {round(latency_fwd_per_layer_mlp*1000, 3)} +"
            f" {round(2*latency_fwd_per_layer_layernorm*1000, 3)} +"
            f" {round(2*latency_fwd_per_layer_tp_comm*1000, 3)})")

        breakdown_per_layer = {
            "attn": latency_fwd_per_layer_attn,
            "mlp": latency_fwd_per_layer_mlp,
            "layernorm": 2 * latency_fwd_per_layer_layernorm,
            "tp_comm": 2 * latency_fwd_per_layer_tp_comm,
        }

        return latency_per_layer, breakdown_per_layer

    def get_latency_fwd_input_embedding(
            self,
            batch_size: int,
            seq_len: int,
            dtype_bytes: int = BYTES_FP32) -> float:
        """Get the latency for the forward pass of the input embedding layer, given the
        batch size, sequence length, and data type of the embedding weight.

        Args:
            batch_size (int): batch size
            seq_len (int): sequence length
            dtype_bytes (int, optional): number of bytes in the data type for the embedding weight. Defaults to BYTES_FP32.

        Returns:
            float: the latency in seconds for the forward pass of the input embedding layer
        """
        memory_latency = (self.model_config.vocab_size *
                          self.model_config.hidden_dim * dtype_bytes /
                          (self.get_gpu_hbm_bandwidth() * 10**9))
        comm_latency = self.get_latency_fwd_per_layer_tp_comm(
            batch_size, seq_len, dtype_bytes)
        return memory_latency + comm_latency

    def get_latency_fwd_output_embedding_loss(self, batch_size: int,
                                              seq_len: int) -> float:
        """Get the latency for the forward pass of the output embedding layer (computing the logits). The operation is compute bound. With tensor parallelism size > 1, an allgather communicates `batch_size * seq_len` elements, which is ignored here. Refer to https://arxiv.org/abs/1909.08053 for more details.

        Args:
            batch_size (int): batch size
            seq_len (int): sequence length

        Returns:
            float: the latency in seconds for the forward pass of the output embedding layer
        """
        compute_latency = (2 * seq_len * batch_size *
                           self.model_config.vocab_size *
                           self.model_config.hidden_dim /
                           self.parallelism_config.tp_size /
                           (self.get_TFLOPS_per_gpu() * 10**12))
        return compute_latency

    def get_latency_fwd(
        self,
        batch_size: int,
        seq_len: int,
        is_inference: bool = True,
        activation_recomputation:
        ActivationRecomputation = ActivationRecomputation.NONE,
        layernorm_dtype_bytes: int = BYTES_FP32,
        breakdown_prefix: str = "",
    ) -> tuple:
        """Get the latency for the forward pass of the transformer, given the batch
        size, sequence length, and whether it is inference or not, the activation
        recomputation strategy, and the number of bytes in the data type for the
        layernorm activations.

        Args:
            batch_size (int): batch size
            seq_len (int): sequence length
            is_inference (bool, optional): whether it is inference or not. Defaults to True.
            activation_recomputation (ActivationRecomputation, optional): activation recomputation strategy. Defaults to ActivationRecomputation.NONE.
            layernorm_dtype_bytes (int, optional): number of bytes in the data type for the layernorm activations. Defaults to BYTES_FP32. Often has to be FP32 in training to maintain model accuracy.
            breakdown_prefix (str, optional): prefix for the breakdown dict keys. Defaults to "".
        Returns:
            tuple: a tuple of the latency in seconds for the forward pass of the transformer and its breakdown dict
        """
        num_layers_per_gpu = int(self.model_config.num_layers /
                                 self.parallelism_config.pp_size)

        (
            latency_fwd_per_layer,
            breakdown_per_layer,
        ) = self.get_latency_fwd_per_layer(
            batch_size,
            seq_len,
            is_inference,
            activation_recomputation,
            layernorm_dtype_bytes,
        )

        latency_fwd_all_layers = latency_fwd_per_layer * num_layers_per_gpu

        latency_fwd_input_embedding = self.get_latency_fwd_input_embedding(
            batch_size,
            seq_len,
            dtype_bytes=self.dtype_config.embedding_bits / BITS_PER_BYTE,
        )

        latency_fwd_output_embedding_loss = (
            self.get_latency_fwd_output_embedding_loss(batch_size, seq_len))

        total_latency = (latency_fwd_all_layers + latency_fwd_input_embedding +
                         latency_fwd_output_embedding_loss)

        logger.debug("latency_fwd_all_layers:"
                     f" {round(latency_fwd_all_layers*1000, 3)} ms"
                     f" ({round(latency_fwd_per_layer*1000, 3)} ms x"
                     f" {num_layers_per_gpu}), latency_fwd_input_embedding:"
                     f" {round(latency_fwd_input_embedding*1000, 3)} ms,"
                     " latency_fwd_output_embedding_loss:"
                     f" {round(latency_fwd_output_embedding_loss*1000, 3)} ms")

        logger.debug(f"latency_fwd_total: {round(total_latency*1000, 3)} ms"
                     f" ({round(latency_fwd_all_layers*1000, 3)} +"
                     f" {round(latency_fwd_input_embedding*1000, 3)} +"
                     f" {round(latency_fwd_output_embedding_loss*1000, 3)})")

        total_breakdown = {
            breakdown_prefix + "latency_fwd_attn":
            breakdown_per_layer["attn"] * num_layers_per_gpu,
            breakdown_prefix + "latency_fwd_mlp":
            breakdown_per_layer["mlp"] * num_layers_per_gpu,
            breakdown_prefix + "latency_fwd_layernorm":
            breakdown_per_layer["layernorm"] * num_layers_per_gpu,
            breakdown_prefix + "latency_fwd_tp_comm":
            breakdown_per_layer["tp_comm"] * num_layers_per_gpu,
            breakdown_prefix + "latency_fwd_input_embedding":
            latency_fwd_input_embedding,
            breakdown_prefix + "latency_fwd_output_embedding_loss":
            latency_fwd_output_embedding_loss,
        }
        return total_latency, total_breakdown

    def print_config(self, name="Training Configs") -> None:
        config_str = f"\n{name.center(PRINT_LINE_WIDTH, '-')}\n"
        config_str += f"{pformat(self.model_config)}\n"
        config_str += f"{pformat(self.gpu_config)}\n"
        config_str += f"{pformat(self.dtype_config)}\n"
        config_str += f"{pformat(self.parallelism_config)}\n"
        logger.info(config_str)

    def get_configs_desc(self) -> str:
        return f"{self.model_config.name}-{self.gpu_config.name}-{self.dtype_config.name}-tp{self.parallelism_config.tp_size}-pp{self.parallelism_config.pp_size}-dp{self.parallelism_config.dp_size}-sp{self.parallelism_config.sp_size}-fe{round(self.flops_efficiency, 2)}-ep{self.parallelism_config.ep_size}-hbme{round(self.hbm_memory_efficiency, 2)}"

    def get_readable_summary_dict(self,
                                  summary_dict: dict,
                                  title="Summary") -> str:
        log_str = f"\n{title.center(PRINT_LINE_WIDTH, '-')}\n"
        for key, value in summary_dict.items():
            if "num_tokens" in key or "num_params" in key or "num_active_params" in key or "flops" in key:
                log_str += f"{key}: {_num_to_string(value, divisor=1000)}\n"
            elif "gpu_hours" == key:
                log_str += f"{key}: {int(value)}\n" if value else ""
            elif "memory" in key and "efficiency" not in key:
                log_str += f"{key}: {_num_to_string(value)}B\n"
            elif "latency" in key:
                log_str += f"{key}: {_latency_to_string(value)}\n"
            else:
                log_str += f"{key}: {value}\n"
        log_str += f"{'-' * PRINT_LINE_WIDTH}\n"
        return log_str

    def output_summary_dict(
        self,
        summary_dict: dict,
        output_dir: str,
        print_human_readable: bool = True,
        output_file_suffix: str = "",
    ):
        file_name = self.get_configs_desc(
        ) + output_file_suffix + "-summary.json"

        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created directory: {output_dir}")
            except:
                logger.error(f"Failed to create output_dir {output_dir}")
                exit()
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory"
        with open(os.path.join(output_dir, file_name), "w") as f:
            json.dump(summary_dict, f, indent=4)
        logger.info(
            f"Summary written to {os.path.join(output_dir, file_name)}")
        if print_human_readable:
            log_str = self.get_readable_summary_dict(summary_dict)
            file_name = self.get_configs_desc(
            ) + output_file_suffix + "-summary-readable.txt"
            with open(os.path.join(output_dir, file_name), "w") as f:
                f.write(log_str)
            logger.info(
                f"Readable summary written to {os.path.join(output_dir, file_name)}"
            )

    def inference(
        self,
        batch_size_per_gpu: int = 1,
        seq_len: int = 512,
        num_tokens_to_generate: int = 32,
        use_kv_cache: bool = True,
        ds_zero: DSZeRO = DSZeRO.NONE,
        layernorm_dtype_bytes: int = BYTES_FP16,
        kv_cache_dtype_bytes: int = None,
        cost_per_gpu_hour: float = None,
        output_dir: str = None,
        output_file_suffix: str = "",
    ) -> dict:
        """Inference analysis given the configs and inputs.

        Args:
            batch_size_per_gpu (int, optional): batch size per gpu. Defaults to 1.
            seq_len (int, optional): number of input tokens. Defaults to 512.
            num_tokens_to_generate (int, optional): number of tokens to generate for generative models. Defaults to 32.
            use_kv_cache (bool, optional): whether to use kv_cache. Defaults to True.
            ds_zero (DSZeRO, optional): which DeepSpeed ZeRO stage to use. Defaults to DSZeRO.NONE (disabled).
            layernorm_dtype_bytes (int, optional): number of bytes in the data type for the layernorm activations. Defaults to BYTES_FP32. Often has to be at least FP16 in inference to maintain model accuracy.
            kv_cache_dtype_bytes (int, optional): number of bytes in the data type for the kv_cache. Defaults to None. Often has to be at least FP16 in inference to maintain model accuracy.
            cost_per_gpu_hour (float, optional): dollar cost per GPU hour. Defaults to None.
            output_dir (str, optional): if set to a directory path, write the return summary dict out to the directory with the setup. Defaults to None.
            output_dir (str, optional): if set to a directory path, write the return summary dict out to the directory with the setup. Defaults to None.

        Returns:
            dict: a summary dict of the training analysis
        """
        if self.model_config.max_seq_len is not None:
            assert (
                seq_len <= self.model_config.max_seq_len
            ), f"seq_len must be less than model max_seq_len ({self.model_config.max_seq_len})"

        self.print_config("Inference Configs")

        logger.info(f"\n{'Analysis'.center(PRINT_LINE_WIDTH, '-')}")

        if kv_cache_dtype_bytes is None:
            kv_cache_dtype_bytes = (self.dtype_config.activation_bits /
                                    BITS_PER_BYTE)
            logger.info(
                "kv_cache_dtype_bytes not specified, setting to the same as"
                f" the activation data type : {kv_cache_dtype_bytes}")

        num_layers_per_gpu = int(self.model_config.num_layers /
                                 self.parallelism_config.pp_size)
        if self.model_config.num_layers % self.parallelism_config.pp_size:
            logger.info(
                "num_layers not be divisible by pp_size, taking the floor")

        weight_memory_embedding_per_gpu = self.get_memory_embedding(ds_zero)
        weight_memory_layers_per_gpu, weight_memory_attn_per_gpu, weight_memory_mlp_per_gpu, weight_memory_layernorm_per_gpu = [
            x * self.model_config.num_layers
            for x in self.get_weight_memory_per_layer(ds_zero,
                                                      return_breakdown=True)
        ]
        weight_memory_last_layernorm = self.get_weight_memory_last_layernorm(
            ds_zero)
        weight_memory_per_gpu = (weight_memory_layers_per_gpu +
                                 weight_memory_embedding_per_gpu +
                                 weight_memory_last_layernorm)

        memory_left = (self.gpu_config.mem_per_GPU_in_GB * 1024**3 -
                       weight_memory_per_gpu)
        assert memory_left > 0, (
            "model is too large (requiring"
            f" {_num_to_string(weight_memory_per_gpu)}B) to fit in total GPU"
            " memory")

        logger.info(
            f"weight_memory_per_gpu: {_num_to_string(weight_memory_per_gpu)}B"
            " (embedding + attn + mlp + layernorm:"
            f" {_num_to_string(weight_memory_embedding_per_gpu)}B + {_num_to_string(weight_memory_attn_per_gpu)}B + {_num_to_string(weight_memory_mlp_per_gpu)}B + {_num_to_string(weight_memory_layernorm_per_gpu)}B), memory_left:"
            f" {_num_to_string(memory_left)}B")

        # With pipeline parallelism, each stage contains L/p layers so the first stage must store p ×L/p = L layers worth of activations regardless of the pipeline parallel size p; activation memory required for the input embeddings, the last layer-norm, and the output layer are ignored here. Refer to https://arxiv.org/abs/2205.05198 for more details.
        prefill_activation_memory_per_layer_batch_size_1 = self.get_activation_memory_per_layer(
            1,
            seq_len,
            is_inference=True,
            layernorm_dtype_bytes=layernorm_dtype_bytes,
        )
        prefill_activation_memory_embedding_output_batch_size_1 = self.get_activation_memory_output_embedding(
            1, seq_len)

        prefill_activation_memory_batch_size_1 = max(
            prefill_activation_memory_per_layer_batch_size_1,
            prefill_activation_memory_embedding_output_batch_size_1)

        prefill_max_batch_size_per_gpu = int(
            memory_left / prefill_activation_memory_batch_size_1)
        logger.info(
            f"prefill_activation_memory_batch_size_1: {_num_to_string(prefill_activation_memory_batch_size_1)}B,"
            " prefill_max_batch_size_per_gpu:"
            f" {prefill_max_batch_size_per_gpu}")

        prefill_activation_memory_per_layer = self.get_activation_memory_per_layer(
            batch_size_per_gpu,
            seq_len,
            is_inference=True,
            layernorm_dtype_bytes=layernorm_dtype_bytes,
        )
        prefill_activation_memory_embedding_output = self.get_activation_memory_output_embedding(
            batch_size_per_gpu, seq_len)
        prefill_activation_memory_per_gpu = max(
            prefill_activation_memory_per_layer,
            prefill_activation_memory_embedding_output)

        logger.info("prefill_activation_memory_per_gpu with batch_size_per_gpu"
                    f" {batch_size_per_gpu}:"
                    f" {_num_to_string(prefill_activation_memory_per_gpu)}B")
        assert memory_left > prefill_activation_memory_per_gpu, (
            "prefill activation memory is too large with batch_size_per_gpu ="
            f" {batch_size_per_gpu} to fit in GPU memory(requiring"
            f" {_num_to_string(prefill_activation_memory_per_gpu)}B),"
            " memory_left after fitting in model weights:"
            f" {_num_to_string(memory_left)}B, prefill_max_batch_size_per_gpu:"
            f" {prefill_max_batch_size_per_gpu}")

        prefill_num_flops_fwd_total = self.get_num_flops_fwd_total(
            batch_size_per_gpu, seq_len)
        logger.info(
            "prefill_num_flops_fwd_total:"
            f" {_num_to_string(prefill_num_flops_fwd_total, divisor=1000)}")

        prefill_latency, prefill_latency_breakdown = self.get_latency_fwd(
            batch_size_per_gpu,
            seq_len,
            is_inference=True,
            layernorm_dtype_bytes=layernorm_dtype_bytes,
            breakdown_prefix="prefill_",
        )

        if use_kv_cache:
            if (batch_size_per_gpu *
                (seq_len + num_tokens_to_generate) < self.get_pivot()):
                logger.warning(
                    "kv_cache is only useful when batch_size *"
                    " (seq+num_tokens_to_generate)"
                    f" ({batch_size_per_gpu * (seq_len+num_tokens_to_generate)}) is larger"
                    f" than ({round(self.get_pivot(), 3)}), which is the pivot"
                    " point")
            kv_cache_memory_per_gpu = self.get_memory_kv_cache_per_layer(
                batch_size_per_gpu,
                seq_len + num_tokens_to_generate,
                kv_cache_dtype_bytes=kv_cache_dtype_bytes,
            ) * num_layers_per_gpu

            # load and store kv cache
            kv_cache_latency = (2 * kv_cache_memory_per_gpu /
                                (self.get_gpu_hbm_bandwidth() * 10**9))

            decode_activation_memory_per_layer = self.get_activation_memory_per_layer(
                batch_size_per_gpu,
                1,
                is_inference=True,
                layernorm_dtype_bytes=layernorm_dtype_bytes,
            )
            decode_activation_memory_embedding_output = self.get_activation_memory_output_embedding(
                batch_size_per_gpu, 1)
            decode_activation_memory_per_gpu = max(
                decode_activation_memory_per_layer,
                decode_activation_memory_embedding_output)

            logger.info(
                "kv_cache_memory_per_gpu:"
                f" {_num_to_string(kv_cache_memory_per_gpu)}B,"
                " decode_activation_memory_per_gpu:"
                f" {_num_to_string(decode_activation_memory_per_gpu)}B")

            decode_max_batch_size_per_gpu = int(
                memory_left /
                ((decode_activation_memory_per_gpu + kv_cache_memory_per_gpu) /
                 batch_size_per_gpu))
            assert memory_left > (
                kv_cache_memory_per_gpu + decode_activation_memory_per_gpu
            ), ("kv_cache and activation memory with batch_size_per_gpu ="
                f" {batch_size_per_gpu} is too large to fit in GPU memory"
                " (requiring"
                f" {_num_to_string((kv_cache_memory_per_gpu + decode_activation_memory_per_gpu))}B),"
                " memory_left after fitting in model weights:"
                f" {_num_to_string(memory_left)}B,"
                " decode_max_batch_size_per_gpu:"
                f" {decode_max_batch_size_per_gpu}")
        else:
            decode_max_batch_size_per_gpu = int(
                memory_left / prefill_activation_memory_batch_size_1)
            logger.info("decode_activation_memory_batch_size_1:"
                        f" {prefill_activation_memory_batch_size_1},"
                        " decode_max_batch_size_per_gpu:"
                        f" {decode_max_batch_size_per_gpu}")

            assert batch_size_per_gpu <= decode_max_batch_size_per_gpu, (
                f"batch_size_per_gpu {batch_size_per_gpu} is too large to fit"
                " in GPU memory, decode_max_batch_size_per_gpu:"
                f" {decode_max_batch_size_per_gpu}")

            decode_activation_memory_per_layer = (
                self.get_activation_memory_per_layer(
                    batch_size_per_gpu,
                    seq_len + num_tokens_to_generate,
                    is_inference=True,
                    layernorm_dtype_bytes=layernorm_dtype_bytes,
                ))
            decode_activation_memory_per_gpu = (
                decode_activation_memory_per_layer)
            kv_cache_memory_per_gpu = 0
            kv_cache_latency = 0

        decode_num_flops_fwd_total = self.get_num_flops_fwd_total(
            batch_size_per_gpu,
            1 if use_kv_cache else (seq_len + num_tokens_to_generate) // 2,
        )
        logger.info(
            "decode_num_flops_fwd_total:"
            f" {_num_to_string(decode_num_flops_fwd_total, divisor=1000)}")

        decode_latency, decode_latency_breakdown = self.get_latency_fwd(
            batch_size_per_gpu,
            1 if use_kv_cache else (seq_len + num_tokens_to_generate) // 2,
            is_inference=True,
            layernorm_dtype_bytes=layernorm_dtype_bytes,
            breakdown_prefix="decode_",
        )

        if use_kv_cache:
            decode_latency += kv_cache_latency

        total_decode_latency = decode_latency * num_tokens_to_generate
        total_per_token_latency = (
            prefill_latency + total_decode_latency) / num_tokens_to_generate

        decode_tokens_per_sec = batch_size_per_gpu * 1 / decode_latency
        prefill_tokens_per_sec = batch_size_per_gpu * seq_len / prefill_latency
        total_tokens_per_sec = batch_size_per_gpu / total_per_token_latency

        summary_dict = {
            "batch_size_per_gpu": batch_size_per_gpu,
            "seq_len": seq_len,
            "tp_size": self.parallelism_config.tp_size,
            "ep_size": self.parallelism_config.ep_size,
            "pp_size": self.parallelism_config.pp_size,
            "num_tokens_to_generate": num_tokens_to_generate,
            "flops_efficiency": self.flops_efficiency,
            "hbm_memory_efficiency": self.hbm_memory_efficiency,
            "layernorm_dtype_bytes": layernorm_dtype_bytes,
            "use_kv_cache": use_kv_cache,
            "kv_cache_latency": kv_cache_latency,
            "kv_cache_memory_per_gpu": kv_cache_memory_per_gpu,
            "weight_memory_per_gpu": weight_memory_per_gpu,
            "weight_memory_embedding_per_gpu": weight_memory_embedding_per_gpu,
            "prefill_activation_memory_per_gpu":
            prefill_activation_memory_per_gpu,
            "prefill_max_batch_size_per_gpu": prefill_max_batch_size_per_gpu,
            "prefill_num_flops_fwd_total": prefill_num_flops_fwd_total,
            "decode_activation_memory_per_gpu":
            decode_activation_memory_per_gpu,
            "decode_max_batch_size_per_gpu": decode_max_batch_size_per_gpu,
            "decode_activation_memory_per_gpu":
            decode_activation_memory_per_gpu,
            "decode_num_flops_fwd_total": decode_num_flops_fwd_total,
            "prefill_latency": prefill_latency,
        }

        summary_dict.update(prefill_latency_breakdown)
        summary_dict.update({
            "decode_latency": decode_latency,
        })
        summary_dict.update(decode_latency_breakdown)
        summary_dict.update({
            "total_decode_latency":
            total_decode_latency,
            "total_latency":
            prefill_latency + decode_latency * num_tokens_to_generate,
            "total_per_token_latency":
            total_per_token_latency,
        })

        summary_dict.update({
            "prefill_tokens_per_sec": prefill_tokens_per_sec,
            "decode_tokens_per_sec": decode_tokens_per_sec,
            "total_tokens_per_sec": total_tokens_per_sec,
        })

        if cost_per_gpu_hour:
            num_gpus = self.parallelism_config.pp_size * self.parallelism_config.tp_size

            def compute_cost_per_1k_tokens(tokens_per_sec):
                return 1000 * cost_per_gpu_hour * num_gpus / 3600 / tokens_per_sec

            prefill_cost_per_1k_tokens = compute_cost_per_1k_tokens(
                prefill_tokens_per_sec)
            decode_cost_per_1k_tokens = compute_cost_per_1k_tokens(
                decode_tokens_per_sec)
            total_cost_per_1k_tokens = compute_cost_per_1k_tokens(
                total_tokens_per_sec)
            summary_dict.update({
                "prefill_cost_per_1k_tokens":
                prefill_cost_per_1k_tokens,
                "decode_cost_per_1k_tokens":
                decode_cost_per_1k_tokens,
                "total_cost_per_1k_tokens":
                total_cost_per_1k_tokens
            })

        logger.info(self.get_readable_summary_dict(summary_dict))

        if output_dir is not None:
            self.output_summary_dict(summary_dict,
                                     output_dir,
                                     print_human_readable=True,
                                     output_file_suffix=output_file_suffix)

        return summary_dict

    def config_batch_size_and_gradient_accumulation_steps(
        self,
        max_batch_size_per_gpu: int,
        batch_size_per_gpu: int = None,
        gradient_accumulation_steps: int = None,
        global_batch_size: int = None,
    ) -> tuple:
        """Configure batch_size_per_gpu, gradient_accumulation_steps and
        global_batch_size (effective batch size). If none is given, find a
        maximum batch_size_per_gpu while satisfying the constraint `global_batch_size ==
        batch_size_per_gpu * gradient_accumulation_steps * dp_size`.

        Args:
            max_batch_size_per_gpu (int): the max batch size per gpu before OOM
            batch_size_per_gpu (int, optional): batch size per GPU. Defaults to None.
            gradient_accumulation_steps (int, optional): gradient accumulation steps. Defaults to None.
            global_batch_size (int, optional): global batch size (effective batch size). Defaults to None.

        Returns:
            tuple: (batch_size_per_gpu, gradient_accumulation_steps, global_batch_size)
        """
        assert_msg = (f"note that global_batch_size == batch_size_per_gpu *"
                      f" gradient_accumulation_steps * dp_size")
        dp_size = self.parallelism_config.dp_size
        if (global_batch_size and batch_size_per_gpu
                and gradient_accumulation_steps):
            assert (global_batch_size == batch_size_per_gpu *
                    gradient_accumulation_steps * dp_size), assert_msg
        elif global_batch_size and batch_size_per_gpu:
            # gradient_accumulation_steps is None, the other two are not None
            gradient_accumulation_steps = global_batch_size // (
                batch_size_per_gpu * dp_size)
            assert (global_batch_size % (batch_size_per_gpu * dp_size) == 0
                    and gradient_accumulation_steps > 0
                    ), "no valid gradient_accumulation_steps, {assert_msg}"
        elif global_batch_size and gradient_accumulation_steps:
            # batch_size_per_gpu is None, the other two are not None
            batch_size_per_gpu = global_batch_size // (
                gradient_accumulation_steps * dp_size)
            assert (global_batch_size %
                    (gradient_accumulation_steps * dp_size) == 0
                    and batch_size_per_gpu > 0
                    ), "no valid batch_size_per_gpu, {assert_msg}"
        elif batch_size_per_gpu and gradient_accumulation_steps or batch_size_per_gpu:
            # batch_size_per_gpu is not None
            assert (
                batch_size_per_gpu <= max_batch_size_per_gpu
            ), f"batch_size_per_gpu {batch_size_per_gpu} must be <= max_batch_size_per_gpu {max_batch_size_per_gpu}, {assert_msg}"
            if gradient_accumulation_steps is None:
                gradient_accumulation_steps = 1
            global_batch_size = (batch_size_per_gpu *
                                 gradient_accumulation_steps * dp_size)
        elif global_batch_size:
            # batch_size_per_gpu and gradient_accumulation_steps are None
            assert (
                global_batch_size % dp_size == 0
            ), f"global_batch_size must be divisible by dp_size, {assert_msg}"

            if max_batch_size_per_gpu >= global_batch_size // dp_size:
                batch_size_per_gpu = global_batch_size // dp_size
                gradient_accumulation_steps = 1
            else:
                prod = global_batch_size // dp_size
                batch_size_per_gpu = next(d for d in range(
                    prod,
                    0,
                    -1,
                ) if prod % d == 0 and d <= max_batch_size_per_gpu)
                gradient_accumulation_steps = global_batch_size // (
                    batch_size_per_gpu * dp_size)
            logger.info("batch_size_per_gpu not set, using batch_size_per_gpu"
                        f" {batch_size_per_gpu} (max_batch_size_per_gpu ="
                        f" {max_batch_size_per_gpu})")
        else:
            # (global_batch_size and batch_size_per_gpu are None) or (all are None)
            batch_size_per_gpu = max_batch_size_per_gpu
            gradient_accumulation_steps = (1 if
                                           gradient_accumulation_steps is None
                                           else gradient_accumulation_steps)
            global_batch_size = (batch_size_per_gpu *
                                 gradient_accumulation_steps *
                                 self.parallelism_config.dp_size)
            logger.info("batch_size_per_gpu not set, using batch_size_per_gpu"
                        f" {batch_size_per_gpu} (max_batch_size_per_gpu ="
                        f" {max_batch_size_per_gpu})")

        return (
            batch_size_per_gpu,
            gradient_accumulation_steps,
            global_batch_size,
        )

    def training(
        self,
        batch_size_per_gpu: int = None,
        gradient_accumulation_steps: int = None,
        global_batch_size: int = None,
        seq_len: int = None,
        total_num_tokens: int = None,
        activation_recomputation:
        ActivationRecomputation = ActivationRecomputation.NONE,
        ds_zero: DSZeRO = DSZeRO.NONE,
        layernorm_dtype_bytes: int = BYTES_FP32,
        master_weights_dtype_bytes: int = BYTES_FP32,
        other_op_bytes: int = None,
        flash_attn: bool = True,
        softmax_dropout: bool = False,
        mlp_activation_quant_bits: int = None,
        mlp_1linear_quant_bits: int = None,
        mlp_gelu_input_quant_bits: int = None,
        mlp_2linear_quant_bits: int = None,
        mlp_recompute_gelu: bool = False,
        mlp_gated_linear_units: bool = False,
        output_dir: str = None,
        output_file_suffix: str = "",
    ) -> dict:
        """Training analysis given the configs and inputs.

        Args:
            batch_size_per_gpu (int, optional): batch size per gpu (micro batch size). Defaults to None.
            gradient_accumulation_steps (int, optional): gradient accumulation steps. Defaults to None.
            global_batch_size (int, optional): global batch size. Defaults to None.
            seq_len (int, optional): sequence length. Defaults to None.
            total_num_tokens (int, optional): total number of tokens used for training. Defaults to None.
            activation_recomputation (ActivationRecomputation, optional): activation recomputation strategy. Defaults to ActivationRecomputation.NONE.
            ds_zero (DSZeRO, optional): which DeepSpeed ZeRO stage to use. Defaults to DSZeRO.NONE (disabled).
            layernorm_dtype_bytes (int, optional): number of bytes in the data type for the layernorm activations. Defaults to BYTES_FP32. Often has to be FP32 in training to maintain model accuracy.
            master_weights_dtype_bytes (int): the number of bytes in the data type for the optimizer master weights. Defaults to BYTES_FP32.
            other_op_bytes (int, optional): the number of bytes in the optimizer state. Defaults to None, which assumes using Adam optimizer.
            flash_attn (bool, optional): whether to use Flash Attention. Defaults to True.
            softmax_dropout (bool, optional): whether to apply dropout after softmax. Defaults to False.
            mlp_activation_quant_bits (int, optional): number of bits to quantize MLP activations; if set, override the values for mlp_1linear_quant_bits, mlp_gelu_input_quant_bits and mlp_2linear_quant_bits. Defaults to None.
            mlp_1linear_quant_bits (int, optional): number of bits to quantize the input activations of the first linear layer. Defaults to None.
            mlp_gelu_input_quant_bits (int, optional): number of bits to quantize the GELU input activations. Defaults to None.
            mlp_2linear_quant_bits (int, optional): number of bits to quantize the input activations of the second linear layer. Defaults to None.
            mlp_recompute_gelu (bool, optional): whether to recompute the gelu activation in the MLP backward pass. Defaults to False.
            mlp_gated_linear_units (bool, optional): whether to use gated linear units in the MLP. Defaults to False.
            output_dir (str, optional): if set to a directory path, write the return summary dict out to the directory with the setup. Defaults to None.

        Returns:
            dict: a summary dict of the training analysis
        )
        """
        if seq_len is None:
            assert (
                self.model_config.max_seq_len
                is not None), "seq_len must be set if max_seq_len is not set"
            seq_len = self.model_config.max_seq_len
            logger.info(f"seq_len not set, using max_seq_len {seq_len}")
        else:
            assert (seq_len <= self.model_config.max_seq_len
                    ), "seq_len must be less than model max_seq_len"

        self.print_config("Training Configs")

        if ds_zero == DSZeRO.NONE:
            logger.warning(
                f"DeepSpeed ZeRO is disabled, consider using ZeRO to reduce memory usage"
            )

        logger.info(f"\n{'Analysis'.center(PRINT_LINE_WIDTH, '-')}")

        num_layers_per_gpu = int(self.model_config.num_layers /
                                 self.parallelism_config.pp_size)
        if self.model_config.num_layers % self.parallelism_config.pp_size:
            logger.info(
                "num_layers not be divisible by pp_size, taking the floor")

        weight_memory_embedding_per_gpu = self.get_memory_embedding(ds_zero)

        weight_memory_layers_per_gpu, weight_memory_attn_per_gpu, weight_memory_mlp_per_gpu, weight_memory_layernorm_per_gpu = [
            x * num_layers_per_gpu
            for x in self.get_weight_memory_per_layer(ds_zero,
                                                      return_breakdown=True)
        ]
        weight_memory_last_layernorm = self.get_weight_memory_last_layernorm(
            ds_zero)
        weight_memory_per_gpu = (weight_memory_embedding_per_gpu +
                                 weight_memory_layers_per_gpu +
                                 weight_memory_last_layernorm)

        optimizer_state_memory_per_layer, gradient_memory_per_layer = self.get_memory_optimizer_state_and_gradient_per_layer(
            master_weights_dtype_bytes, other_op_bytes, ds_zero)

        optimizer_state_memory_embedding, gradient_memory_embedding = self.get_memory_optimizer_state_and_gradient_embedding(
            master_weights_dtype_bytes, other_op_bytes, ds_zero)

        optimizer_state_memory_last_layernorm, gradient_memory_last_layernorm = self.get_memory_optimizer_state_and_gradient_last_layernorm(
            master_weights_dtype_bytes, other_op_bytes, ds_zero)

        optimizer_state_memory_per_gpu = optimizer_state_memory_per_layer * num_layers_per_gpu + optimizer_state_memory_embedding + optimizer_state_memory_last_layernorm
        gradient_memory_per_gpu = gradient_memory_per_layer * num_layers_per_gpu + gradient_memory_embedding + gradient_memory_last_layernorm

        memory_left = (self.gpu_config.mem_per_GPU_in_GB * 1024**3 -
                       weight_memory_per_gpu - optimizer_state_memory_per_gpu -
                       gradient_memory_per_gpu)

        logger.info(
            f"weight_memory_per_gpu: {_num_to_string(weight_memory_per_gpu)}B"
            " (embedding_memory:"
            f" {_num_to_string(weight_memory_embedding_per_gpu)}B),"
            " optimizer_state_memory_per_gpu:"
            f" {_num_to_string(optimizer_state_memory_per_gpu)}B,"
            " gradient_memory_per_gpu:"
            f" {_num_to_string(gradient_memory_per_gpu)}B, memory_left:"
            f" {_num_to_string(memory_left)}B")

        if memory_left < 0:
            logger.warning(
                "model weight/optimizer stage/gradient is too large (requiring"
                f" {_num_to_string(weight_memory_per_gpu)}B /"
                f" {_num_to_string(optimizer_state_memory_per_gpu)}B /"
                f" {_num_to_string(gradient_memory_per_gpu)}B) to fit in total GPU"
                " memory")

        # With pipeline parallelism, each stage contains L/p layers so the first stage must store p ×L/p = L layers worth of activations regardless of the pipeline parallel size p; activation memory required for the input embeddings, the last layer-norm, and the output layer are ignored here. Refer to https://arxiv.org/abs/2205.05198 for more details.

        activation_memory_batch_size_1, activation_memory_attn_batch_size_1, mlp_activation_memory_batch_size_1, layernorm_activation_memory_batch_size_1 = [
            x * self.model_config.num_layers
            for x in self.get_activation_memory_per_layer(
                1,
                seq_len,
                is_inference=False,
                activation_recomputation=activation_recomputation,
                layernorm_dtype_bytes=layernorm_dtype_bytes,
                flash_attn=flash_attn,
                softmax_dropout=softmax_dropout,
                mlp_activation_quant_bits=mlp_activation_quant_bits,
                mlp_1linear_quant_bits=mlp_1linear_quant_bits,
                mlp_gelu_input_quant_bits=mlp_gelu_input_quant_bits,
                mlp_2linear_quant_bits=mlp_2linear_quant_bits,
                mlp_recompute_gelu=mlp_recompute_gelu,
                mlp_gated_linear_units=mlp_gated_linear_units,
                return_breakdown=True,
            )
        ]
        activation_memory_embedding_output_batch_size_1 = self.get_activation_memory_output_embedding(
            1, seq_len)
        logger.info(
            f"activation_memory_embedding_output for micro batch size 1: {_num_to_string(activation_memory_embedding_output_batch_size_1)}B"
        )
        activation_memory_batch_size_1 += activation_memory_embedding_output_batch_size_1
        activation_memory_batch_size_1 += self.get_activation_memory_per_layernorm(
            1,
            seq_len,
            activation_recomputation,
            layernorm_dtype_bytes,
        )

        max_batch_size_per_gpu = int(memory_left //
                                     activation_memory_batch_size_1)

        if memory_left < activation_memory_batch_size_1:
            logger.warning(
                f"memory_left {_num_to_string(memory_left)} < activation_memory_batch_size_1 {_num_to_string(activation_memory_batch_size_1)}"
            )

        logger.info(
            f"activation_memory for micro batch size 1: {_num_to_string(activation_memory_batch_size_1)}B, max_batch_size_per_gpu: {max_batch_size_per_gpu}"
        )

        (
            batch_size_per_gpu,
            gradient_accumulation_steps,
            global_batch_size,
        ) = self.config_batch_size_and_gradient_accumulation_steps(
            max_batch_size_per_gpu,
            batch_size_per_gpu,
            gradient_accumulation_steps,
            global_batch_size,
        )

        if batch_size_per_gpu == 1:
            activation_memory_per_gpu, activation_memory_attn_per_gpu, activation_memory_mlp_per_gpu, activation_memory_layernorm_per_gpu = activation_memory_batch_size_1, activation_memory_attn_batch_size_1, mlp_activation_memory_batch_size_1, layernorm_activation_memory_batch_size_1
        else:
            activation_memory_per_gpu, activation_memory_attn_per_gpu, activation_memory_mlp_per_gpu, activation_memory_layernorm_per_gpu = [
                x * self.model_config.num_layers
                for x in self.get_activation_memory_per_layer(
                    batch_size_per_gpu,
                    seq_len,
                    is_inference=False,
                    activation_recomputation=activation_recomputation,
                    layernorm_dtype_bytes=layernorm_dtype_bytes,
                    flash_attn=flash_attn,
                    softmax_dropout=softmax_dropout,
                    mlp_activation_quant_bits=mlp_activation_quant_bits,
                    mlp_1linear_quant_bits=mlp_1linear_quant_bits,
                    mlp_gelu_input_quant_bits=mlp_gelu_input_quant_bits,
                    mlp_2linear_quant_bits=mlp_2linear_quant_bits,
                    mlp_recompute_gelu=mlp_recompute_gelu,
                    return_breakdown=True,
                )
            ]
        activation_memory_embedding_output_per_gpu = self.get_activation_memory_output_embedding(
            batch_size_per_gpu, seq_len)
        activation_memory_per_gpu += activation_memory_embedding_output_per_gpu
        activation_memory_per_gpu += self.get_activation_memory_per_layernorm(
            batch_size_per_gpu,
            seq_len,
            activation_recomputation,
            layernorm_dtype_bytes,
        )

        logger.info("activation_memory_per_gpu with micro batch size"
                    f" {batch_size_per_gpu}:"
                    f" {_num_to_string(activation_memory_per_gpu)}B")
        if memory_left < activation_memory_per_gpu:
            logger.warning(
                "activation memory is too large with batch_size_per_gpu ="
                f" {batch_size_per_gpu} to fit in GPU memory (requiring"
                f" {_num_to_string(activation_memory_per_gpu)}B, memory_left after"
                " fitting in model weights, gradients, and optimizer states ="
                f" {_num_to_string(memory_left)}B, max_batch_size_per_gpu ="
                f" {max_batch_size_per_gpu})")
        memory_left -= activation_memory_per_gpu

        num_flops_fwd_total = self.get_num_flops_fwd_total(
            batch_size_per_gpu, seq_len)
        num_flops_bwd_total = self.get_num_flops_bwd_total(
            batch_size_per_gpu, seq_len)

        if activation_recomputation == ActivationRecomputation.FULL:
            num_flops_recompute = num_flops_fwd_total
        elif activation_recomputation == ActivationRecomputation.SELECTIVE:
            num_flops_recompute = (
                self.get_num_flops_total_selective_recompute_attn(
                    batch_size_per_gpu, seq_len))
            if num_flops_recompute < 0.05 * num_flops_fwd_total:
                logger.warning(
                    f"num_flops_recompute ({num_flops_recompute}) is too large to"
                    " ignore")
        elif activation_recomputation == ActivationRecomputation.NONE:
            num_flops_recompute = 0

        num_flops_total_per_micro_batch = (num_flops_fwd_total +
                                           num_flops_bwd_total +
                                           num_flops_recompute)

        logger.info(
            "num_flops_total_per_micro_batch:"
            f" {_num_to_string(num_flops_total_per_micro_batch, divisor=1000)} ({_num_to_string(num_flops_fwd_total, divisor=1000)} fwd"
            f" + {_num_to_string(num_flops_bwd_total, divisor=1000)} bwd +"
            f" {_num_to_string(num_flops_recompute, divisor=1000)} recompute)")

        latency_fwd, latency_fwd_breakdown = self.get_latency_fwd(
            batch_size_per_gpu,
            seq_len,
            is_inference=False,
            activation_recomputation=activation_recomputation,
            layernorm_dtype_bytes=layernorm_dtype_bytes,
        )

        mp_size = (self.parallelism_config.tp_size *
                   self.parallelism_config.pp_size)

        latency_per_micro_batch = num_flops_total_per_micro_batch / (
            mp_size * self.get_TFLOPS_per_gpu() * 1e12)

        latency_per_iter = (latency_per_micro_batch *
                            gradient_accumulation_steps)

        logger.info(
            "latency_per_micro_batch:"
            f" {round(latency_per_micro_batch * 1000, 3)} ms, latency_fwd:"
            f" {round(latency_fwd * 1000, 3)} ms, \nlatency_per_iter:"
            f" {round(latency_per_iter * 1000, 3)} ms"
            f" ({round(latency_per_micro_batch * 1000, 3)} ms *"
            f" {gradient_accumulation_steps} gradient_accumulation_steps)")

        total_num_gpus = (self.parallelism_config.tp_size *
                          self.parallelism_config.pp_size *
                          self.parallelism_config.dp_size)

        if total_num_tokens is not None:
            if total_num_tokens < 20 * self.total_num_params:
                logger.warning(
                    "according to the Chinchilla paper"
                    " (https://arxiv.org/abs/2203.15556), to train a"
                    " compute-optimal LLM, \nwe need around 20 text tokens"
                    " per parameter, the given total_num_tokens /"
                    " total_num_tokens ="
                    f" {round(total_num_tokens/self.total_num_params, 3)} ")
            num_iters = int(total_num_tokens / (global_batch_size * seq_len))
            total_training_latency = latency_per_iter * num_iters
            logger.info(
                f"total_training_latency: {round(total_training_latency, 3)} s"
                f" = {round(total_training_latency/3600/24, 3)} days"
                f" ({round(latency_per_iter * 1000, 3)} ms x"
                f" {num_iters} iters)")
            if self.model_config.moe_num_experts == 1:
                # dense models
                estimated_total_training_latency = (
                    (8 if activation_recomputation
                     == ActivationRecomputation.FULL else 6) *
                    self.total_num_params * total_num_tokens /
                    (total_num_gpus * self.get_TFLOPS_per_gpu() * 1e12))
                if not within_range(total_training_latency,
                                    estimated_total_training_latency, 0.05):
                    logger.warning(
                        f"total_training_latency ({total_training_latency}) is too"
                        " different from estimated_total_training_latency"
                        f" ({estimated_total_training_latency})")

        else:
            total_training_latency = None

        gpu_hours = (total_training_latency * total_num_gpus /
                     3600 if total_training_latency is not None else None)

        summary_dict = {
            "batch_size_per_gpu":
            batch_size_per_gpu,
            "max_batch_size_per_gpu":
            max_batch_size_per_gpu,
            "gradient_accumulation_steps":
            gradient_accumulation_steps,
            "global_batch_size":
            global_batch_size,
            "dp_size":
            self.parallelism_config.dp_size,
            "tp_size":
            self.parallelism_config.tp_size,
            "pp_size":
            self.parallelism_config.pp_size,
            "sp_size":
            self.parallelism_config.sp_size,
            "ep_size":
            self.parallelism_config.ep_size,
            "ds_zero":
            DSZeRO(ds_zero).name,
            "total_num_gpus":
            total_num_gpus,
            "seq_len":
            seq_len,
            "total_num_tokens":
            total_num_tokens,
            "num_params_total":
            self.total_num_params,
            "num_active_params_total":
            self.total_num_active_params,
            "activation_recomputation":
            ActivationRecomputation(activation_recomputation).name,
            "layernorm_dtype_bytes":
            layernorm_dtype_bytes,
            "mlp_activation_quant_bits":
            mlp_activation_quant_bits,
            "mlp_recompute_gelu":
            mlp_recompute_gelu,
            "achieved_flops":
            self.get_TFLOPS_per_gpu(),
            "flops_efficiency":
            self.flops_efficiency,
            "hbm_memory_efficiency":
            self.hbm_memory_efficiency,
            "num_flops_total_per_micro_batch":
            num_flops_total_per_micro_batch,
            "weight_memory_per_gpu":
            weight_memory_per_gpu,
            "weight_memory_embedding_per_gpu":
            weight_memory_embedding_per_gpu,
            "weight_memory_attn_per_gpu":
            weight_memory_attn_per_gpu,
            "weight_memory_mlp_per_gpu":
            weight_memory_mlp_per_gpu,
            "weight_memory_layernorm_per_gpu":
            weight_memory_layernorm_per_gpu,
            "gradient_memory_per_gpu":
            gradient_memory_per_gpu,
            "optimizer_state_memory_per_gpu":
            optimizer_state_memory_per_gpu,
            "(weight+op_state+grad)_memory_per_gpu":
            weight_memory_per_gpu + gradient_memory_per_gpu +
            optimizer_state_memory_per_gpu,
            "activation_memory_batch_size_1":
            activation_memory_batch_size_1,
            "activation_memory_per_gpu":
            activation_memory_per_gpu,
            "activation_memory_attn_per_gpu":
            activation_memory_attn_per_gpu,
            "activation_memory_mlp_per_gpu":
            activation_memory_mlp_per_gpu,
            "activation_memory_layernorm_per_gpu":
            activation_memory_layernorm_per_gpu,
            "activation_memory_embedding_output_per_gpu":
            activation_memory_embedding_output_per_gpu,
            "(weight+op_state+grad+act)_memory_per_gpu":
            weight_memory_per_gpu + gradient_memory_per_gpu +
            optimizer_state_memory_per_gpu + activation_memory_per_gpu,
            "memory_left_per_gpu":
            memory_left,
            "latency_per_micro_batch":
            latency_per_micro_batch,
            "latency_fwd":
            latency_fwd,
        }
        summary_dict.update(latency_fwd_breakdown)
        summary_dict.update({
            "latency_per_iter": latency_per_iter,
            "total_training_latency": total_training_latency,
            "gpu_hours": gpu_hours,
        })

        logger.info(self.get_readable_summary_dict(summary_dict))

        if output_dir is not None:
            self.output_summary_dict(summary_dict,
                                     output_dir,
                                     print_human_readable=True,
                                     output_file_suffix=output_file_suffix)

        return summary_dict


def infer(
    model_name="facebook_opt-1.3b",
    gpu_name="a100-sxm-40gb",
    dtype_name="w16a16e16",
    log_level="INFO",
    batch_size_per_gpu=1,
    ds_zero: int = 0,
    dp_size: int = 1,
    tp_size: int = 1,
    pp_size: int = 1,
    sp_size: int = None,
    seq_len=512,
    num_tokens_to_generate=32,
    use_kv_cache: bool = True,
    layernorm_dtype_bytes: int = BYTES_FP16,
    kv_cache_dtype_bytes: int = None,
    achieved_tflops: float = None,
    achieved_memory_bandwidth_GBs: float = None,
    flops_efficiency: float = None,
    hbm_memory_efficiency: float = None,
    intra_node_memory_efficiency=INTRA_NODE_MEMORY_EFFICIENCY,
    inter_node_memory_efficiency=INTER_NODE_MEMORY_EFFICIENCY,
    cost_per_gpu_hour: float = None,
    output_dir: str = None,
    output_file_suffix: str = "",
) -> dict:
    """_summary_

    Args:
        model_name (str, optional): model name to query the pre-defined `model_configs` dict, or model config json file path, if not found, query Hugging Face to construct ModelConfig. Defaults to "facebook_opt-1.3b".
        gpu_name (str, optional): gpu name to query the pre-defined `gpu_configs` dict. Defaults to "a100-sxm-40gb".
        dtype_name (str, optional): data type name to pre-defined `dtype_configs` dict. Defaults to "w16a16e16".
        log_level (str, optional): logging level. Defaults to "INFO".
        batch_size_per_gpu (int, optional): batch size per GPU. Defaults to 1.
        ds_zero (int, optional): which DeepSpeed ZeRO stage to use. See `DSZeRO`. Defaults to 0.
        dp_size (int, optional): data parallelism size. Defaults to None.
        tp_size (int, optional): tensor parallelism size. Defaults to 1.
        pp_size (int, optional): pipeline parallelism size. Defaults to 1.
        sp_size (int, optional): sequence parallelism size. Defaults to tp_size.
        seq_len (int, optional): input sequence length. Defaults to 512.
        num_tokens_to_generate (int, optional): number of tokens to generate for generative models. Defaults to 32.
        use_kv_cache (bool, optional): whether to use kv cache. Defaults to True.
        layernorm_dtype_bytes (int, optional): number of bytes in the data type for the layernorm activations. Defaults to BYTES_FP32. Often has to be at least FP16 in inference to maintain model accuracy.
        kv_cache_dtype_bytes (int, optional): number of bytes in the data type for the kv_cache. Defaults to None. Often has to be at least FP16 in inference to maintain model accuracy.
        achieved_tflops (float, optional): achieved TFLOPS per GPU. If specified, will override the flops_efficiency passed in. Defaults to None.
        achieved_memory_bandwidth_GBs (float, optional): achieved GPU memory bandwidth in GB/s. If specified, will override the hbm_memory_efficiency passed in. Defaults to None.
        flops_efficiency (float, optional): flops efficiency, ranging from 0 to 1. Defaults to None.
        hbm_memory_efficiency (float, optional): GPU HBM memory efficiency, ranging from 0 to 1. Defaults to HBM_MEMORY_EFFICIENCY.
        intra_node_memory_efficiency (float, optional):  intra-node memory efficiency, ranging from 0 to 1. Defaults to INTRA_NODE_MEMORY_EFFICIENCY.
        inter_node_memory_efficiency (float, optional):  inter-node memory efficiency, ranging from 0 to 1. Defaults to INTER_NODE_MEMORY_EFFICIENCY.
        cost_per_gpu_hour (float, optional): dollar cost per GPU hour. Defaults to None.
        output_dir (str, optional): if set to a directory path, write the return summary dict out to the directory with the setup. Defaults to None.. Defaults to None.
        output_file_suffix (str, optional): suffix of the output file. Defaults to "".

    Returns:
        dict: a summary dictionary of the inference analysis
    """

    model_config = get_model_config_by_name(model_name)
    gpu_config = get_gpu_config_by_name(gpu_name)
    dtype_config = get_dtype_config_by_name(dtype_name)
    parallel_config = ParallelismConfig(
        tp_size=tp_size,
        pp_size=pp_size,
        sp_size=sp_size if sp_size else tp_size,
        dp_size=dp_size)

    logger.setLevel(logging.getLevelName(log_level))

    analysis = LLMAnalysis(
        model_config,
        gpu_config,
        dtype_config,
        parallel_config,
        achieved_tflops=achieved_tflops,
        achieved_memory_bandwidth_GBs=achieved_memory_bandwidth_GBs,
        flops_efficiency=flops_efficiency,
        hbm_memory_efficiency=hbm_memory_efficiency,
        intra_node_memory_efficiency=intra_node_memory_efficiency,
        inter_node_memory_efficiency=inter_node_memory_efficiency,
    )

    if analysis.hbm_memory_efficiency > 0.8:
        logger.warning(
            "Note that benchmarks show closer to 0.6-0.7 hbm_memory_efficiency in inference workloads"
        )

    summary_dict = analysis.inference(
        batch_size_per_gpu=batch_size_per_gpu,
        seq_len=seq_len,
        num_tokens_to_generate=num_tokens_to_generate,
        use_kv_cache=use_kv_cache,
        ds_zero=DSZeRO(ds_zero),
        layernorm_dtype_bytes=layernorm_dtype_bytes,
        kv_cache_dtype_bytes=kv_cache_dtype_bytes,
        cost_per_gpu_hour=cost_per_gpu_hour,
        output_dir=output_dir,
        output_file_suffix=output_file_suffix,
    )

    return summary_dict


def train(
    model_name="facebook_opt-1.3b",
    gpu_name="a100-sxm-40gb",
    dtype_name="w16a16e16",
    log_level="INFO",
    batch_size_per_gpu: int = None,
    gradient_accumulation_steps: int = None,
    global_batch_size: int = None,
    seq_len: int = None,
    total_num_tokens: int = None,
    activation_recomputation: int = 0,
    ds_zero: int = 0,
    dp_size: int = None,
    tp_size: int = 1,
    pp_size: int = 1,
    sp_size: int = None,
    ep_size: int = 1,
    total_num_gpus: int = None,
    layernorm_dtype_bytes: int = BYTES_FP32,
    master_weights_dtype_bytes: int = BYTES_FP32,
    other_op_bytes: int = None,
    flash_attn: bool = True,
    softmax_dropout: bool = False,
    mlp_activation_quant_bits: int = None,
    mlp_1linear_quant_bits: int = None,
    mlp_gelu_input_quant_bits: int = None,
    mlp_2linear_quant_bits: int = None,
    mlp_recompute_gelu: bool = False,
    mlp_gated_linear_units: bool = False,
    achieved_tflops: float = None,
    flops_efficiency: float = None,
    hbm_memory_efficiency: float = HBM_MEMORY_EFFICIENCY,
    intra_node_memory_efficiency=INTRA_NODE_MEMORY_EFFICIENCY,
    inter_node_memory_efficiency=INTER_NODE_MEMORY_EFFICIENCY,
    num_gpus_per_node: int = NUM_GPUS_PER_NODE,
    output_dir: str = None,
    output_file_suffix: str = "",
) -> dict:
    """Entry point function of training analysis for the command line interface. This
    uses pre-defined name-to-configuration mapping and common arguments to construct
    LLMAnalysis.

    Args:
        model_name (str, optional): model name to query the pre-defined `model_configs` dict, or model config json file path, if not found, query Hugging Face to construct ModelConfig. Defaults to "facebook_opt-1.3b".
        gpu_name (str, optional): gpu name to query the pre-defined `gpu_configs` dict. Defaults to "a100-sxm-40gb".
        dtype_name (str, optional): data type name to pre-defined `dtype_configs` dict. Defaults to "w16a16e16".
        log_level (str, optional): logging level. Defaults to "INFO".
        batch_size_per_gpu (int, optional): batch size per GPU (micro batch size). Defaults to None.
        gradient_accumulation_steps (int, optional): gradient accumulation steps. Defaults to None.
        global_batch_size (int, optional): global batch size. Defaults to None.
        seq_len (int, optional): sequence length. Defaults to None.
        total_num_tokens (int, optional): total number of tokens used for training. Defaults to None.
        activation_recomputation (int, optional): activation recomputation strategy. See `ActivationRecomputation`. Defaults to 0.
        ds_zero (int, optional): which DeepSpeed ZeRO stage to use. See `DSZeRO`. Defaults to 0.
        dp_size (int, optional): data parallelism size. Defaults to None.
        tp_size (int, optional): tensor parallelism size. Defaults to 1.
        pp_size (int, optional): pipeline parallelism size. Defaults to 1.
        sp_size (int, optional): sequence parallelism size. Defaults to tp_size.
        ep_size (int, optional): expert parallelism size. Defaults to 1.
        total_num_gpus (int, optional): total number of GPUs used for training. Defaults to None.
        layernorm_dtype_bytes (int, optional): number of bytes in the data type for the layernorm activations. Often has to be FP32 in training to maintain model accuracy. Defaults to BYTES_FP32.
        master_weights_dtype_bytes (int): the number of bytes in the data type for the optimizer master weights. Defaults to BYTES_FP32.
        other_op_bytes (int, optional): the number of bytes in the optimizer state. Defaults to None, which assumes using Adam optimizer.
        flash_attn (bool, optional): whether to use Flash Attention. Defaults to True.
        softmax_dropout (bool, optional): whether to apply dropout after softmax. Defaults to False.
        mlp_activation_quant_bits (int, optional): number of bits to quantize MLP activations; if set, override the values for mlp_1linear_quant_bits, mlp_gelu_input_quant_bits and mlp_2linear_quant_bits. Defaults to None.
        mlp_1linear_quant_bits (int, optional): number of bits to quantize the input activations of the first linear layer. Defaults to None.
        mlp_gelu_input_quant_bits (int, optional): number of bits to quantize the GELU input activations. Defaults to None.
        mlp_2linear_quant_bits (int, optional): number of bits to quantize the input activations of the second linear layer. Defaults to None.
        mlp_activation_quant_bits (int, optional): number of bits for the quantized MLP activation. Defaults to None.
        mlp_recompute_gelu (bool, optional): whether to recompute the GELU activation in the MLP backward pass. Defaults to False.
        mlp_gated_linear_units (bool, optional): whether to use gated linear units in the MLP. Defaults to False.
        achieved_tflops (float, optional): achieved TFLOPS per GPU. Defaults to None.
        flops_efficiency (float, optional): flops efficiency, ranging from 0 to 1. Defaults to None.
        hbm_memory_efficiency (float, optional): GPU HBM memory efficiency, ranging from 0 to 1. Defaults to HBM_MEMORY_EFFICIENCY.
        intra_node_memory_efficiency (float, optional):  intra-node memory efficiency, ranging from 0 to 1. Defaults to INTRA_NODE_MEMORY_EFFICIENCY.
        inter_node_memory_efficiency (float, optional):  inter-node memory efficiency, ranging from 0 to 1. Defaults to INTER_NODE_MEMORY_EFFICIENCY.
        num_gpus_per_node (int, optional): number of GPUs per node. Defaults to NUM_GPUS_PER_NODE (8).
        output_dir (str, optional): if set to a directory path, write the return summary dict out to the directory with the setup. Defaults to None.
    Returns:
        dict: a summary dictionary of the training analysis
    """
    logger.setLevel(logging.getLevelName(log_level))

    assert tp_size <= num_gpus_per_node, (
        f"tp_size must be <= {num_gpus_per_node}(num_gpus_per_node), tensor"
        " parallelism requires high communication bandwidth to be efficient"
        " and is best kept within a single node where high bandwidth NVLink"
        " is available.")

    if total_num_gpus and dp_size:
        assert (
            total_num_gpus == dp_size * tp_size * pp_size
        ), "total_num_gpus must be equal to dp_size * tp_size * pp_size"
    elif total_num_gpus:
        assert (total_num_gpus % (tp_size * pp_size) == 0
                ), f"total_num_gpus must be a multiple of tp_size * pp_size"
        dp_size = total_num_gpus // (tp_size * pp_size)
    elif dp_size:
        total_num_gpus = dp_size * tp_size * pp_size
    else:
        dp_size = 1

    model_config = get_model_config_by_name(model_name)
    gpu_config = get_gpu_config_by_name(gpu_name)
    dtype_config = get_dtype_config_by_name(dtype_name)
    parallel_config = ParallelismConfig(
        tp_size=tp_size,
        pp_size=pp_size,
        dp_size=dp_size,
        sp_size=sp_size if sp_size else tp_size,
        ep_size=ep_size)

    analysis = LLMAnalysis(
        model_config,
        gpu_config,
        dtype_config,
        parallel_config,
        hbm_memory_efficiency=hbm_memory_efficiency,
        intra_node_memory_efficiency=intra_node_memory_efficiency,
        inter_node_memory_efficiency=inter_node_memory_efficiency,
        achieved_tflops=achieved_tflops,
        flops_efficiency=flops_efficiency,
    )

    summary_dict = analysis.training(
        batch_size_per_gpu=batch_size_per_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        total_num_tokens=total_num_tokens,
        activation_recomputation=ActivationRecomputation(
            activation_recomputation),
        ds_zero=DSZeRO(ds_zero),
        layernorm_dtype_bytes=layernorm_dtype_bytes,
        master_weights_dtype_bytes=master_weights_dtype_bytes,
        other_op_bytes=other_op_bytes,
        flash_attn=flash_attn,
        softmax_dropout=softmax_dropout,
        mlp_activation_quant_bits=mlp_activation_quant_bits,
        mlp_1linear_quant_bits=mlp_1linear_quant_bits,
        mlp_gelu_input_quant_bits=mlp_gelu_input_quant_bits,
        mlp_2linear_quant_bits=mlp_2linear_quant_bits,
        mlp_recompute_gelu=mlp_recompute_gelu,
        mlp_gated_linear_units=mlp_gated_linear_units,
        output_dir=output_dir,
        output_file_suffix=output_file_suffix,
    )

    return summary_dict


if __name__ == "__main__":
    fire.Fire(serialize=lambda x: json.dumps(x, indent=4))
