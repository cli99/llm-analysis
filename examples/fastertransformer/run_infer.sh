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

output_dir='outputs_infer'

if [[ ! -e $output_dir ]]; then
    mkdir $output_dir
elif [[ ! -d $output_dir ]]; then
    echo "$output_dir already exists but is not a directory" 1>&2
fi

python -m llm_analysis.analysis infer --model_name test-13b --gpu_name a100-sxm-40gb --tp_size 1 --seq_len 512 --num_tokens_to_generate 10 --flops_efficiency 1 --hbm_memory_efficiency 1 --output_dir ${output_dir}

python -m llm_analysis.analysis infer --model_name test-13b --gpu_name a100-sxm-40gb --tp_size 1 --seq_len 512 --num_tokens_to_generate 10 --flops_efficiency 0.7 --hbm_memory_efficiency 0.9 --output_dir ${output_dir}

python -m llm_analysis.analysis infer --model_name test-13b --gpu_name a100-sxm-40gb --tp_size 2 --seq_len 512 --num_tokens_to_generate 10 --flops_efficiency 1 --hbm_memory_efficiency 1 --output_dir ${output_dir}

python -m llm_analysis.analysis infer --model_name test-13b --gpu_name a100-sxm-40gb --tp_size 2 --seq_len 512 --num_tokens_to_generate 10 --flops_efficiency 0.7 --hbm_memory_efficiency 0.87 --output_dir ${output_dir}
