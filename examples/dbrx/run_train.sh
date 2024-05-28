model_name=dbrx # model name (hf model name) or model config json file path
seq_len=4096
gpu_name='h100-sxm-80gb'    # python -m llm_analysis.config list_gpu_configs
dtype_name="w16a16e16"      # 16-bit weights, activations, and embedding
batch_size_per_gpu=2        # device_train_microbatch_size
num_gpus=3072               # num_gpus
activation_recomputation=0  # 0: no activation recomputation; 1: checkpoint attention compute; 2: checkpoint attention ; 3: checkpoint layernorm-attention-layernorm; 4: checkpoint attention the entire transformer layer
dp_size=128                 # data parallelization size for sharding
ep_size=8                   # expert parallelization size, moe_dp_sharding_size = dp_size / ep_size
tp_size=1                   # tensor parallelization size
ds_zero=3                   # dp sharding strategy, https://github.com/cli99/llm-analysis#parallelism-scheme
mlp_activation_quant_bits=8 # number of bits used for mlp activation
mlp_recompute_gelu=True     # whether to recompute the gelu in mlp backward
flops_efficiency=0.35       # mfu
hbm_memory_efficiency=1     # gpu memory efficiency
intra_node_memory_efficiency=0.8
inter_node_memory_efficiency=0.8
total_num_tokens=12e12       # number of tokens to train on
master_weights_dtype_bytes=4 # FP32 master weights
other_op_bytes=4             # lion optimizer
output_dir=output_dbrx
output_file_prefix="bs${batch_size_per_gpu}-ar${activation_recomputation}-zero${ds_zero}-"
layernorm_dtype_bytes=2

python -m llm_analysis.analysis train --model_name=${model_name} --seq_len=${seq_len} --gpu_name=${gpu_name} --dtype_name=${dtype_name} --output_dir=${output_dir} --output-file-prefix=${output_file_prefix} --activation_recomputation ${activation_recomputation} --ds_zero ${ds_zero} --batch_size_per_gpu=${batch_size_per_gpu} --total_num_gpus=${num_gpus} --dp_size=${dp_size} --tp_size=${tp_size} --ep_size=${ep_size} --flops_efficiency=${flops_efficiency} --hbm_memory_efficiency=${hbm_memory_efficiency} --total_num_tokens ${total_num_tokens} --mlp_activation_quant_bits ${mlp_activation_quant_bits} --layernorm_dtype_bytes=${layernorm_dtype_bytes} --mlp_recompute_gelu ${mlp_recompute_gelu} --master_weights_dtype_bytes ${master_weights_dtype_bytes} --other_op_bytes ${other_op_bytes} --intra_node_memory_efficiency ${intra_node_memory_efficiency} --inter_node_memory_efficiency ${inter_node_memory_efficiency}
