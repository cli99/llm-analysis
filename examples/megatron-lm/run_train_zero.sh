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

output_dir="outputs_train_zero"

if [[ ! -e $output_dir ]]; then
    mkdir $output_dir
elif [[ ! -d $output_dir ]]; then
    echo "$output_dir already exists but is not a directory" 1>&2
fi

# 90 days
python -m llm_analysis.analysis train --model_name megatron-lm-175b --gpu_name a100-sxm-40gb --total_num_tokens 300e9  --activation_recomputation 2 --tp_size 1 --pp_size 1 --ds_zero 3 --global_batch_size 1536 --total_num_gpus 384 --achieved_tflops 144 --output_dir ${output_dir}

# 74
python -m llm_analysis.analysis train --model_name megatron-lm-175b --gpu_name a100-sxm-40gb --total_num_tokens 300e9  --activation_recomputation 2 --tp_size 1 --pp_size 1 --ds_zero 3 --global_batch_size 1536 --total_num_gpus 768 --achieved_tflops 88 --output_dir ${output_dir}

# 74
python -m llm_analysis.analysis train --model_name megatron-lm-175b --gpu_name a100-sxm-40gb --total_num_tokens 300e9  --activation_recomputation 2 --tp_size 1 --pp_size 1 --ds_zero 3 --global_batch_size 1536 --total_num_gpus 1536 --achieved_tflops 44 --output_dir ${output_dir}

# 169 days
python -m llm_analysis.analysis train --model_name megatron-lm-530b --gpu_name a100-sxm-40gb --total_num_tokens 300e9  --activation_recomputation 2 --tp_size 1 --pp_size 1 --ds_zero 3 --global_batch_size 2560 --total_num_gpus 640 --achieved_tflops 138 --output_dir ${output_dir}

# 137 days
python -m llm_analysis.analysis train --model_name megatron-lm-530b --gpu_name a100-sxm-40gb --total_num_tokens 300e9  --activation_recomputation 2 --tp_size 1 --pp_size 1 --ds_zero 3 --global_batch_size 2240 --total_num_gpus 1120 --achieved_tflops 98 --output_dir ${output_dir}

# 140 days
python -m llm_analysis.analysis train --model_name megatron-lm-530b --gpu_name a100-sxm-40gb --total_num_tokens 300e9  --activation_recomputation 2 --tp_size 1 --pp_size 1 --ds_zero 3 --global_batch_size 2240 --total_num_gpus 2240 --achieved_tflops 48 --output_dir ${output_dir}
