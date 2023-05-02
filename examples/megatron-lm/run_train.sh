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

if [[ ! -e $output_dir ]]; then
    mkdir $output_dir
elif [[ ! -d $output_dir ]]; then
    echo "$output_dir already exists but is not a directory" 1>&2
fi

# 84 days
python -m llm_analysis.analysis train --model_name megatron-lm-175b --gpu_name a100-sxm-40gb --total_num_tokens 300e9  --activation_recomputation 2 --tp_size 8 --pp_size 12  --global_batch_size 1536 --total_num_gpus 384 --achieved_tflops 153 --output_dir ${output_dir}

# 43 days
python -m llm_analysis.analysis train --model_name megatron-lm-175b --gpu_name a100-sxm-40gb --total_num_tokens 300e9  --activation_recomputation 2 --tp_size 8 --pp_size 12  --global_batch_size 1536 --total_num_gpus 768 --achieved_tflops 149 --output_dir ${output_dir}

# 23 days
python -m llm_analysis.analysis train --model_name megatron-lm-175b --gpu_name a100-sxm-40gb --total_num_tokens 300e9  --activation_recomputation 2 --tp_size 8 --pp_size 12  --global_batch_size 1536 --total_num_gpus 1536 --achieved_tflops 141 --output_dir ${output_dir}

# 156 days
python -m llm_analysis.analysis train --model_name megatron-lm-530b --gpu_name a100-sxm-40gb --total_num_tokens 300e9  --activation_recomputation 2 --tp_size 8 --pp_size 35  --global_batch_size 2240 --total_num_gpus 560 --achieved_tflops 171 --output_dir ${output_dir}

# 80 days
python -m llm_analysis.analysis train --model_name megatron-lm-530b --gpu_name a100-sxm-40gb --total_num_tokens 300e9  --activation_recomputation 2 --tp_size 8 --pp_size 35  --global_batch_size 2240 --total_num_gpus 1120 --achieved_tflops 167 --output_dir ${output_dir}

# 42 days
python -m llm_analysis.analysis train --model_name megatron-lm-530b --gpu_name a100-sxm-40gb --total_num_tokens 300e9  --activation_recomputation 2 --tp_size 8 --pp_size 35  --global_batch_size 2240 --total_num_gpus 2240 --achieved_tflops 159 --output_dir ${output_dir}
