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

output_dir="outputs_train"
global_batch_size=4000000
gpu_name="a100-sxm-80gb"
flops_efficiency=0.5
hbm_memory_efficiency=0.9

if [[ ! -e $output_dir ]]; then
    mkdir $output_dir
elif [[ ! -d $output_dir ]]; then
    echo "$output_dir already exists but is not a directory" 1>&2
fi

python -m llm_analysis.analysis train --model_name decapoda-research_llama-7b-hf --gpu_name ${gpu_name} --total_num_tokens 1e12 --activation_recomputation 1 --tp_size 8 --pp_size 8 --sp_size 8 --global_batch_size ${global_batch_size} --total_num_gpus 2048 --flops_efficiency ${flops_efficiency} --hbm_memory_efficiency ${hbm_memory_efficiency} --output_dir ${output_dir}

python -m llm_analysis.analysis train --model_name decapoda-research_llama-13b-hf --gpu_name ${gpu_name} --total_num_tokens 1e12 --activation_recomputation 1 --tp_size 8 --pp_size 8 --sp_size 8 --global_batch_size ${global_batch_size} --total_num_gpus 2048 --flops_efficiency ${flops_efficiency} --hbm_memory_efficiency ${hbm_memory_efficiency} --output_dir ${output_dir}

python -m llm_analysis.analysis train --model_name decapoda-research_llama-30b-hf --gpu_name ${gpu_name} --total_num_tokens 1.4e12 --activation_recomputation 1 --tp_size 8 --pp_size 8 --sp_size 8 --global_batch_size ${global_batch_size} --total_num_gpus 2048 --flops_efficiency ${flops_efficiency} --hbm_memory_efficiency ${hbm_memory_efficiency} --output_dir ${output_dir}

python -m llm_analysis.analysis train --model_name decapoda-research_llama-65b-hf --gpu_name ${gpu_name} --total_num_tokens 1.4e12 --activation_recomputation 1 --tp_size 8 --pp_size 8 --sp_size 8 --global_batch_size ${global_batch_size} --total_num_gpus 2048 --flops_efficiency ${flops_efficiency} --hbm_memory_efficiency ${hbm_memory_efficiency} --output_dir ${output_dir}
