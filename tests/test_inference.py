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

from llm_analysis.utils import within_range
from llm_analysis.analysis import LLMAnalysis
from llm_analysis.config import (
    ParallelismConfig,
    get_dtype_config_by_name,
    get_gpu_config_by_name,
    get_model_config_by_name,
)

TOLERANCE = 0.1


# megatron-lm paper https://arxiv.org/abs/2104.04473 Table 2
def test_fastertransformer_13b_tp1():
    model_name = "test-13b"
    dtype_name = "w16a16e16"
    gpu_name = "a100-sxm-40gb"

    tp_size = 1
    batch_size_per_gpu = 1
    model_config = get_model_config_by_name(model_name)
    gpu_config = get_gpu_config_by_name(gpu_name)
    dtype_config = get_dtype_config_by_name(dtype_name)
    parallel_config = ParallelismConfig(tp_size=tp_size)

    analysis = LLMAnalysis(
        model_config,
        gpu_config,
        dtype_config,
        parallel_config,
    )

    summary_dict = analysis.inference(
        batch_size_per_gpu=batch_size_per_gpu,
        seq_len=512,
        num_tokens_to_generate=10,
    )

    assert within_range(summary_dict["decode_latency"] * 1000, 17.07,
                        TOLERANCE)


def test_llama2_70b():
    model_name = "upstage/Llama-2-70b-instruct-v2"
    dtype_name = "w16a16e16"
    gpu_name = "a100-sxm-80gb"

    tp_size = 2
    batch_size_per_gpu = 1
    model_config = get_model_config_by_name(model_name)
    gpu_config = get_gpu_config_by_name(gpu_name)
    dtype_config = get_dtype_config_by_name(dtype_name)
    parallel_config = ParallelismConfig(tp_size=tp_size)

    analysis = LLMAnalysis(model_config, gpu_config, dtype_config,
                           parallel_config)

    summary_dict = analysis.inference(
        batch_size_per_gpu=batch_size_per_gpu,
        seq_len=512,
        num_tokens_to_generate=512,
    )

    assert within_range(summary_dict["total_decode_latency"], 180.06,
                        TOLERANCE)
