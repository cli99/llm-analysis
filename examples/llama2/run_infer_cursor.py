from llm_analysis.config import (
    DtypeConfig,
    GPUConfig,
    ModelConfig,
    ParallelismConfig,
    get_dtype_config_by_name,
    get_gpu_config_by_name,
    get_model_config_by_name,
)
from llm_analysis.analysis import LLMAnalysis
import csv

gpu_name="a100-sxm-80gb"
dtype_name="w16a16e16"
model_name="upstage/Llama-2-70b-instruct-v2"
tp_size=2
cost_per_gpu_hour=2.21
flops_efficiency=0.6
hbm_memory_efficiency=0.6
output_file="infer_cursor_compare.csv"

analysis = LLMAnalysis(
    get_model_config_by_name(model_name),
    get_gpu_config_by_name(gpu_name),
    get_dtype_config_by_name(dtype_name),
    ParallelismConfig(tp_size=tp_size),
    flops_efficiency=flops_efficiency,
    hbm_memory_efficiency=hbm_memory_efficiency,
)

experiments = [(1, 128, 242), (2, 128, 512), (4, 128, 512), (1, 512, 512), (2, 512, 512), (4, 512, 512), (1, 512, 304), (8, 512, 512), (16, 512, 512), (32, 512, 512), (1, 1024, 512), (2, 1024, 512), (4, 1024, 512), (8, 1024, 512), (16, 1024, 512), (1, 3595, 512), (2, 3595, 512), (4, 3595, 512)]

with open(output_file, mode='w') as csv_file:
    fieldnames = ["Batch Size", "Prompt Tokens", "Completion Tokens", "Time to first token (s)", "Time for completion (s)", "Tokens/second", "Price/1k prompt tokens", "Price /1k Completion tokens"]
    writer = csv.writer(csv_file)
    writer.writerow(fieldnames)
    selected_keys = ["batch_size_per_gpu", "seq_len", "num_tokens_to_generate", "prefill_latency", "total_decode_latency", "total_tokens_per_sec", "prefill_cost_per_1k_tokens", "decode_cost_per_1k_tokens"]

    for idx, (batch_size_per_gpu, seq_len, num_tokens_to_generate) in enumerate(experiments):
        summary_dict = analysis.inference(
                batch_size_per_gpu=batch_size_per_gpu,
                seq_len=seq_len,
                num_tokens_to_generate=num_tokens_to_generate,
                cost_per_gpu_hour=cost_per_gpu_hour,
            )
        writer.writerow([summary_dict[key] for key in selected_keys])
