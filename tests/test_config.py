# Copyright 2023 chengli
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

from llm_analysis.config import (DtypeConfig, GPUConfig, ModelConfig,
                                 get_dtype_config_by_name,
                                 get_gpu_config_by_name,
                                 get_model_config_by_name)


def test_get_model_config_by_name():
    model_name = "facebook/opt-125m"
    model_config = get_model_config_by_name(model_name)
    assert isinstance(model_config, ModelConfig)
    assert model_config.num_layers == 12
    assert model_config.hidden_dim == 768


def test_get_gpu_config_by_name():
    gpu_name = "a100-sxm-40gb"
    gpu_config = get_gpu_config_by_name(gpu_name)
    assert isinstance(gpu_config, GPUConfig)
    assert gpu_config.mem_per_GPU_in_GB == 40
    assert gpu_config.hbm_bandwidth_in_GB_per_sec == 1555


def test_get_dtype_config_by_name():
    dtype_name = "w4a4e16"
    dtype_config = get_dtype_config_by_name(dtype_name)
    assert isinstance(dtype_config, DtypeConfig)
    assert dtype_config.weight_bits == 4
    assert dtype_config.activation_bits == 4
    assert dtype_config.embedding_bits == 16
