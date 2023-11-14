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

from llm_analysis.analysis import ActivationRecomputation, DSZeRO, LLMAnalysis
from llm_analysis.config import (
    ParallelismConfig,
    get_dtype_config_by_name,
    get_gpu_config_by_name,
    get_model_config_by_name,
)
from llm_analysis.utils import _latency_to_string, _num_to_string, within_range

TOLERANCE = 0.05


# megatron-lm paper https://arxiv.org/abs/2104.04473 Table 2
def test_training_megatron_lm_1():
    model_name = "facebook_opt-175b"
    dtype_name = "w16a16e16"
    gpu_name = "a100-sxm-40gb"
    total_num_tokens = 300e9

    activation_recomputation = ActivationRecomputation.FULL
    tp_size = 8
    pp_size = 12
    total_num_gpus = 384
    dp_size = total_num_gpus // (tp_size * pp_size)
    batch_size_per_gpu = 8

    achieved_tflops = 153  # reported in the paper

    model_config = get_model_config_by_name(model_name)
    gpu_config = get_gpu_config_by_name(gpu_name)
    dtype_config = get_dtype_config_by_name(dtype_name)
    parallel_config = ParallelismConfig(tp_size=tp_size,
                                        pp_size=pp_size,
                                        dp_size=dp_size)

    analysis = LLMAnalysis(
        model_config,
        gpu_config,
        dtype_config,
        parallel_config,
        achieved_tflops=achieved_tflops,
    )

    summary_dict = analysis.training(
        batch_size_per_gpu=batch_size_per_gpu,
        total_num_tokens=total_num_tokens,
        activation_recomputation=activation_recomputation,
    )

    assert within_range(
        summary_dict["total_training_latency_using_flops"] / 3600 / 24, 84,
        TOLERANCE)

    assert (_latency_to_string(
        summary_dict["total_training_latency_using_flops"]) == "84.82 days")

    assert _num_to_string(summary_dict["num_params_total"]) == "162.58 G"


# megatron-lm paper https://arxiv.org/abs/2104.04473 Table 2
def test_training_megatron_lm_2():
    model_name = "facebook_opt-175b"
    dtype_name = "w16a16e16"
    gpu_name = "a100-sxm-40gb"
    total_num_tokens = 300e9

    activation_recomputation = ActivationRecomputation.FULL
    tp_size = 8
    pp_size = 12
    total_num_gpus = 1536
    dp_size = total_num_gpus // (tp_size * pp_size)
    batch_size_per_gpu = 8

    achieved_tflops = 141  # reported in the paper

    model_config = get_model_config_by_name(model_name)
    gpu_config = get_gpu_config_by_name(gpu_name)
    dtype_config = get_dtype_config_by_name(dtype_name)
    parallel_config = ParallelismConfig(tp_size=tp_size,
                                        pp_size=pp_size,
                                        dp_size=dp_size)

    analysis = LLMAnalysis(
        model_config,
        gpu_config,
        dtype_config,
        parallel_config,
        achieved_tflops=achieved_tflops,
    )

    summary_dict = analysis.training(
        batch_size_per_gpu=batch_size_per_gpu,
        total_num_tokens=total_num_tokens,
        activation_recomputation=activation_recomputation,
    )

    assert within_range(
        summary_dict["total_training_latency_using_flops"] / 3600 / 24, 23,
        TOLERANCE)


# megatron-lm paper https://arxiv.org/abs/2104.04473 Table 2
def test_training_megatron_lm_3():
    model_name = "mt-nlg-530b"
    dtype_name = "w16a16e16"
    gpu_name = "a100-sxm-40gb"
    total_num_tokens = 300e9

    activation_recomputation = ActivationRecomputation.FULL
    tp_size = 8
    pp_size = 35
    total_num_gpus = 560
    dp_size = total_num_gpus // (tp_size * pp_size)
    batch_size_per_gpu = 1

    achieved_tflops = 171  # reported in the paper

    model_config = get_model_config_by_name(model_name)
    gpu_config = get_gpu_config_by_name(gpu_name)
    dtype_config = get_dtype_config_by_name(dtype_name)
    parallel_config = ParallelismConfig(tp_size=tp_size,
                                        pp_size=pp_size,
                                        dp_size=dp_size)

    analysis = LLMAnalysis(
        model_config,
        gpu_config,
        dtype_config,
        parallel_config,
        achieved_tflops=achieved_tflops,
    )

    summary_dict = analysis.training(
        batch_size_per_gpu=batch_size_per_gpu,
        total_num_tokens=total_num_tokens,
        activation_recomputation=ActivationRecomputation(
            activation_recomputation),
    )

    assert within_range(
        summary_dict["total_training_latency_using_flops"] / 3600 / 24, 156,
        TOLERANCE)


# megatron-lm paper https://arxiv.org/abs/2104.04473 Table 2
def test_training_zero3_1():
    model_name = "facebook_opt-175b"
    dtype_name = "w16a16e16"
    gpu_name = "a100-sxm-40gb"
    total_num_tokens = 300e9

    batch_size_per_gpu = 1
    activation_recomputation = ActivationRecomputation.FULL
    tp_size = 1
    pp_size = 1
    total_num_gpus = 384
    dp_size = total_num_gpus // (tp_size * pp_size)
    batch_size_per_gpu = 4

    achieved_tflops = 144  # reported in the paper

    model_config = get_model_config_by_name(model_name)
    gpu_config = get_gpu_config_by_name(gpu_name)
    dtype_config = get_dtype_config_by_name(dtype_name)
    parallel_config = ParallelismConfig(tp_size=tp_size,
                                        pp_size=pp_size,
                                        dp_size=dp_size)

    analysis = LLMAnalysis(
        model_config,
        gpu_config,
        dtype_config,
        parallel_config,
        achieved_tflops=achieved_tflops,
    )

    summary_dict = analysis.training(
        batch_size_per_gpu=batch_size_per_gpu,
        total_num_tokens=total_num_tokens,
        activation_recomputation=activation_recomputation,
        ds_zero=DSZeRO.STAGE_3,
    )

    assert within_range(
        summary_dict["total_training_latency_using_flops"] / 3600 / 24, 90,
        TOLERANCE)


# megatron-lm paper https://arxiv.org/abs/2104.04473 Table 2
def test_training_zero3_2():
    model_name = "mt-nlg-530b"
    dtype_name = "w16a16e16"
    gpu_name = "a100-sxm-40gb"
    total_num_tokens = 300e9

    activation_recomputation = ActivationRecomputation.FULL
    tp_size = 1
    pp_size = 1
    total_num_gpus = 1120
    dp_size = total_num_gpus // (tp_size * pp_size)
    batch_size_per_gpu = 2

    achieved_tflops = 98  # reported in the paper

    model_config = get_model_config_by_name(model_name)
    gpu_config = get_gpu_config_by_name(gpu_name)
    dtype_config = get_dtype_config_by_name(dtype_name)
    parallel_config = ParallelismConfig(tp_size=tp_size,
                                        pp_size=pp_size,
                                        dp_size=dp_size)

    analysis = LLMAnalysis(
        model_config,
        gpu_config,
        dtype_config,
        parallel_config,
        achieved_tflops=achieved_tflops,
    )

    summary_dict = analysis.training(
        batch_size_per_gpu=batch_size_per_gpu,
        total_num_tokens=total_num_tokens,
        activation_recomputation=activation_recomputation,
        ds_zero=DSZeRO.STAGE_3,
    )

    assert within_range(
        summary_dict["total_training_latency_using_flops"] / 3600 / 24, 136,
        TOLERANCE)


# deepspeed megatron mt-nlg-530b paper https://arxiv.org/abs/2201.11990
def test_training_mt_nlg_1():
    model_name = "mt-nlg-530b"
    dtype_name = "w16a16e16"
    gpu_name = "a100-sxm-40gb"
    total_num_tokens = 270e9

    activation_recomputation = ActivationRecomputation.FULL
    tp_size = 8
    pp_size = 35
    total_num_gpus = 2240
    dp_size = total_num_gpus // (tp_size * pp_size)
    global_batch_size = 1920

    achieved_tflops = 126  # reported in the paper

    model_config = get_model_config_by_name(model_name)
    gpu_config = get_gpu_config_by_name(gpu_name)
    dtype_config = get_dtype_config_by_name(dtype_name)
    parallel_config = ParallelismConfig(tp_size=tp_size,
                                        pp_size=pp_size,
                                        dp_size=dp_size)

    analysis = LLMAnalysis(
        model_config,
        gpu_config,
        dtype_config,
        parallel_config,
        achieved_tflops=achieved_tflops,
    )

    summary_dict = analysis.training(
        global_batch_size=global_batch_size,
        total_num_tokens=total_num_tokens,
        activation_recomputation=activation_recomputation,
        ds_zero=DSZeRO.STAGE_3,
    )

    assert within_range(summary_dict["latency_per_iter"], 60.0, TOLERANCE)


# deepspeed megatron mt-nlg-530b paper https://arxiv.org/abs/2201.11990
def test_training_mt_nlg_2():
    model_name = "mt-nlg-530b"
    dtype_name = "w16a16e16"
    gpu_name = "a100-sxm-40gb"
    total_num_tokens = 270e9

    activation_recomputation = ActivationRecomputation.FULL
    tp_size = 8
    pp_size = 35
    total_num_gpus = 2800
    dp_size = total_num_gpus // (tp_size * pp_size)
    global_batch_size = 1920

    achieved_tflops = 121  # reported in the paper

    model_config = get_model_config_by_name(model_name)
    gpu_config = get_gpu_config_by_name(gpu_name)
    dtype_config = get_dtype_config_by_name(dtype_name)
    parallel_config = ParallelismConfig(tp_size=tp_size,
                                        pp_size=pp_size,
                                        dp_size=dp_size)

    analysis = LLMAnalysis(
        model_config,
        gpu_config,
        dtype_config,
        parallel_config,
        achieved_tflops=achieved_tflops,
    )

    summary_dict = analysis.training(
        global_batch_size=global_batch_size,
        total_num_tokens=total_num_tokens,
        activation_recomputation=activation_recomputation,
        ds_zero=DSZeRO.STAGE_3,
        flash_attn=False)

    assert within_range(summary_dict["latency_per_iter_using_flops"], 50.2,
                        TOLERANCE)

    assert _latency_to_string(
        summary_dict["latency_per_iter_using_flops"]) == "49.98 s"
