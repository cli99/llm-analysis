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

gpu_name='a100-sxm-80gb'
dtype_name="w16a16e16"
output_dir='outputs_infer'
model_name=upstage/Llama-2-70b-instruct-v2
batch_size_per_gpu=1
tp_size=2
output_file_suffix="-bs${batch_size_per_gpu}"
cost_per_gpu_hour=2.21
seq_len=256
num_tokens_to_generate=512
flops_efficiency=0.7
hbm_memory_efficiency=0.9
achieved_tflops=200                # will overwrite the flops_efficiency above
achieved_memory_bandwidth_GBs=1200 # will overwrite the hbm_memory_efficiency above

if [[ ! -e $output_dir ]]; then
    mkdir $output_dir
elif [[ ! -d $output_dir ]]; then
    echo "$output_dir already exists but is not a directory" 1>&2
fi

python -m llm_analysis.analysis infer --model_name=${model_name} --gpu_name=${gpu_name} --dtype_name=${dtype_name} -output_dir=${output_dir} --output-file-suffix=${output_file_suffix} \
    --seq_len=${seq_len} --num_tokens_to_generate=${num_tokens_to_generate} --batch_size_per_gpu=${batch_size_per_gpu} \
    --tp_size=${tp_size} \
    --cost_per_gpu_hour=${cost_per_gpu_hour} \
    --flops_efficiency=${flops_efficiency} --hbm_memory_efficiency=${hbm_memory_efficiency} --log_level DEBUG
# --achieved_tflops=${achieved_tflops} --achieved_memory_bandwidth_GBs=${achieved_memory_bandwidth_GBs}
