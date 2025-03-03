import streamlit as st

from llm_analysis.analysis import (BYTES_FP16, BYTES_FP32,
                                   ActivationRecomputation, DSZeRO,
                                   LLMAnalysis)
from llm_analysis.config import (DtypeConfig, GPUConfig, ModelConfig,
                                 ParallelismConfig, get_model_config_by_name)
from llm_analysis.constant import (HBM_MEMORY_EFFICIENCY,
                                   INTER_NODE_MEMORY_EFFICIENCY,
                                   INTRA_NODE_MEMORY_EFFICIENCY,
                                   NUM_GPUS_PER_NODE)


def main():
    st.set_page_config(page_title="LLM Analysis", layout="wide")

    st.title("LLM Analysis")

    st.header("Training")

    # Initialize session state with default values if not already set
    if "initialized" not in st.session_state:
        try:
            # Load DeepSeek-V3 config as default
            default_model = get_model_config_by_name("deepseek-ai/DeepSeek-V3")
            st.session_state.initialized = True
            st.session_state.model_name = default_model.name
            st.session_state.hidden_dim = default_model.hidden_dim
            st.session_state.n_head = default_model.n_head
            st.session_state.num_layers = default_model.num_layers
            st.session_state.vocab_size = default_model.vocab_size
            st.session_state.expansion_ratio = default_model.expansion_ratio
            st.session_state.mlp_gated_linear_units = default_model.mlp_gated_linear_units
            st.session_state.moe_num_experts = default_model.moe_num_experts or 1
            st.session_state.moe_top_k = default_model.moe_top_k or 1
            st.session_state.moe_intermediate_size = default_model.moe_intermediate_size or 0
            st.session_state.moe_num_shared_experts = default_model.moe_num_shared_experts or 0
            st.session_state.num_key_value_heads = default_model.num_key_value_heads or default_model.n_head
            st.session_state.num_key_value_groups = default_model.num_key_value_groups or 1
            st.session_state.max_seq_len = default_model.max_seq_len or 4096
        except Exception as e:
            # Fallback to basic defaults if loading fails
            st.error(
                f"Error loading default model config: {str(e)}. Using basic defaults."
            )
            st.session_state.initialized = True
            st.session_state.model_name = "deepseek-ai/DeepSeek-V3"
            st.session_state.hidden_dim = 4096
            st.session_state.n_head = 32
            st.session_state.num_layers = 32
            st.session_state.vocab_size = 32000
            st.session_state.expansion_ratio = 4.0
            st.session_state.mlp_gated_linear_units = False
            st.session_state.moe_num_experts = 1
            st.session_state.moe_top_k = 1
            st.session_state.moe_intermediate_size = 0
            st.session_state.moe_num_shared_experts = 0
            st.session_state.num_key_value_heads = 32
            st.session_state.num_key_value_groups = 1
            st.session_state.max_seq_len = 4096

    # Create columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Configuration")

        # Model configuration mode selection
        config_mode = st.radio(
            "Configuration Mode",
            ["Load from HuggingFace", "Custom Configuration"],
            help=
            "Choose whether to load a model from HuggingFace or create a custom configuration"
        )

        if config_mode == "Load from HuggingFace":
            st.markdown(
                "Enter a model name on HuggingFace (e.g., deepseek-ai/DeepSeek-V3, meta-llama/Llama-3.3-70B-Instruct)"
            )
            selected_model = st.text_input("HuggingFace Model Name",
                                           "deepseek-ai/DeepSeek-V3")
            load_button = st.button("Load Model Configuration")

            if load_button:
                try:
                    model_config = get_model_config_by_name(selected_model)
                    st.success(
                        f"Successfully loaded configuration for {selected_model}"
                    )
                    # Update session state with loaded config
                    st.session_state.model_name = model_config.name
                    st.session_state.hidden_dim = model_config.hidden_dim
                    st.session_state.n_head = model_config.n_head
                    st.session_state.num_layers = model_config.num_layers
                    st.session_state.vocab_size = model_config.vocab_size
                    st.session_state.expansion_ratio = model_config.expansion_ratio
                    st.session_state.mlp_gated_linear_units = model_config.mlp_gated_linear_units
                    st.session_state.moe_num_experts = model_config.moe_num_experts or 1
                    st.session_state.moe_top_k = model_config.moe_top_k or 1
                    st.session_state.moe_intermediate_size = model_config.moe_intermediate_size or 0
                    st.session_state.moe_num_shared_experts = model_config.moe_num_shared_experts or 0
                    st.session_state.num_key_value_heads = model_config.num_key_value_heads or model_config.n_head
                    st.session_state.num_key_value_groups = model_config.num_key_value_groups or 1
                    st.session_state.max_seq_len = model_config.max_seq_len or 4096
                except Exception as e:
                    st.error(f"Error loading model config: {str(e)}")
                    # Keep existing session state values if loading fails

        # Model configuration fields that can be adjusted
        st.markdown("##### Model Configuration")
        st.session_state.model_name = st.text_input(
            "Model Name", value=st.session_state.model_name)
        st.session_state.hidden_dim = st.number_input(
            "Hidden Dimension",
            value=st.session_state.hidden_dim,
            step=128,
            help="Hidden dimension size of the model")
        ffn_embed_dim = st.number_input(
            "FFN Dimension",
            value=int(st.session_state.hidden_dim *
                      st.session_state.expansion_ratio),
            step=128,
            help="Hidden dimension of feed-forward network")
        # Update expansion ratio based on FFN dimension
        st.session_state.expansion_ratio = ffn_embed_dim / st.session_state.hidden_dim

        st.session_state.n_head = st.number_input(
            "Number of Attention Heads",
            value=st.session_state.n_head,
            step=1,
            help="Number of attention heads")
        st.session_state.num_key_value_heads = st.number_input(
            "Number of KV Heads",
            min_value=1,
            max_value=st.session_state.n_head,
            value=st.session_state.num_key_value_heads,
            step=1,
            help=
            "Number of key-value heads (must divide number of attention heads)"
        )
        if st.session_state.n_head % st.session_state.num_key_value_heads != 0:
            st.error(
                f"Number of attention heads ({st.session_state.n_head}) must be divisible by number of KV heads ({st.session_state.num_key_value_heads})"
            )
        st.session_state.num_key_value_groups = st.session_state.n_head // st.session_state.num_key_value_heads

        st.session_state.num_layers = st.number_input(
            "Number of Layers",
            value=st.session_state.num_layers,
            step=1,
            help="Number of transformer layers")
        st.session_state.vocab_size = st.number_input(
            "Vocabulary Size",
            value=st.session_state.vocab_size,
            step=1000,
            help="Size of the vocabulary")
        st.session_state.max_seq_len = st.number_input(
            "Maximum Sequence Length",
            value=st.session_state.max_seq_len,
            step=128,
            help="Maximum sequence length the model can handle")
        st.session_state.mlp_gated_linear_units = st.checkbox(
            "MLP Gated Linear Units",
            value=st.session_state.mlp_gated_linear_units,
            help="Use gated linear units in MLP layers")

        # MoE Configuration
        # Store MoE state in session state if not already present
        if "use_moe" not in st.session_state:
            st.session_state.use_moe = st.session_state.moe_num_experts > 1

        # Update use_moe checkbox based on session state
        st.session_state.use_moe = st.checkbox(
            "Use Mixture of Experts",
            value=st.session_state.use_moe,
            help="Enable Mixture of Experts architecture")

        if st.session_state.use_moe:
            st.markdown("##### Mixture of Experts (MoE) Configuration")
            moe_col1, moe_col2 = st.columns(2)
            with moe_col1:
                st.session_state.moe_num_experts = st.number_input(
                    "Number of Experts",
                    min_value=1,
                    value=st.session_state.moe_num_experts,
                    step=1,
                    help="Total number of experts")
                st.session_state.moe_top_k = st.number_input(
                    "Top-K Experts",
                    min_value=1,
                    max_value=st.session_state.moe_num_experts,
                    value=min(st.session_state.moe_top_k,
                              st.session_state.moe_num_experts),
                    step=1,
                    help="Number of experts to route each token to")
            with moe_col2:
                st.session_state.moe_num_shared_experts = st.number_input(
                    "Number of Shared Experts",
                    min_value=0,
                    value=st.session_state.moe_num_shared_experts,
                    step=1,
                    help="Number of experts shared across all groups")
                st.session_state.moe_intermediate_size = st.number_input(
                    "MoE Intermediate Size",
                    min_value=0,
                    value=st.session_state.moe_intermediate_size
                    if st.session_state.moe_intermediate_size > 0 else
                    ffn_embed_dim,
                    step=128,
                    help="Intermediate size for MoE layers")
        else:
            # Reset MoE parameters when disabled
            st.session_state.moe_num_experts = 1
            st.session_state.moe_top_k = 1
            st.session_state.moe_num_shared_experts = 0
            st.session_state.moe_intermediate_size = 0

        # Update the MoE state when loading a new model configuration
        if load_button and 'model_config' in locals():
            st.session_state.use_moe = (model_config.moe_num_experts or 1) > 1
            if st.session_state.use_moe:
                st.session_state.moe_num_experts = model_config.moe_num_experts
                st.session_state.moe_top_k = model_config.moe_top_k
                st.session_state.moe_intermediate_size = model_config.moe_intermediate_size or ffn_embed_dim
                st.session_state.moe_num_shared_experts = model_config.moe_num_shared_experts or 0

    with col2:
        st.subheader("Hardware Configuration")
        gpu_name = st.selectbox(
            "GPU Type", ["A100-80GB", "A100-40GB", "H100-80GB", "Custom"])

        if gpu_name == "Custom":
            mem_per_GPU_in_GB = st.number_input("Memory per GPU (GB)",
                                                value=80,
                                                step=1)
            gpu_flops_16bit = st.number_input("GPU FLOPs 16-bit (TFLOPS)",
                                              value=312,
                                              step=1)
            gpu_flops_8bit = st.number_input("GPU FLOPs 8-bit (TFLOPS)",
                                             value=624,
                                             step=1)
            gpu_flops_4bit = st.number_input("GPU FLOPs 4-bit (TFLOPS)",
                                             value=1248,
                                             step=1)
            gpu_hbm_bandwidth = st.number_input("GPU HBM Bandwidth (GB/s)",
                                                value=2039,
                                                step=1)
        else:
            # Pre-defined GPU configs
            gpu_configs = {
                "A100-80GB": {
                    "mem": 80,
                    "flops_16bit": 312,
                    "flops_8bit": 624,
                    "flops_4bit": 1248,
                    "bandwidth": 2039
                },
                "A100-40GB": {
                    "mem": 40,
                    "flops_16bit": 312,
                    "flops_8bit": 624,
                    "flops_4bit": 1248,
                    "bandwidth": 1555
                },
                "H100-80GB": {
                    "mem": 80,
                    "flops_16bit": 989,
                    "flops_8bit": 1979,
                    "flops_4bit": 3958,
                    "bandwidth": 3350
                },
            }
            selected_gpu = gpu_configs[gpu_name]
            mem_per_GPU_in_GB = selected_gpu["mem"]
            gpu_flops_16bit = selected_gpu["flops_16bit"]
            gpu_flops_8bit = selected_gpu["flops_8bit"]
            gpu_flops_4bit = selected_gpu["flops_4bit"]
            gpu_hbm_bandwidth = selected_gpu["bandwidth"]

        st.subheader("Parallelism Configuration")
        total_num_gpus = st.number_input("Total Number of GPUs",
                                         value=8,
                                         min_value=1,
                                         step=1)
        tp_size = st.number_input("Tensor Parallelism Size",
                                  value=1,
                                  min_value=1,
                                  step=1)
        pp_size = st.number_input("Pipeline Parallelism Size",
                                  value=1,
                                  min_value=1,
                                  step=1)
        dp_size = st.number_input("Data Parallelism Size",
                                  value=8,
                                  min_value=1,
                                  step=1)
        ep_size = st.number_input("Expert Parallelism Size",
                                  value=1,
                                  min_value=1,
                                  step=1)
        sp_size = st.number_input("Sequence Parallelism Size",
                                  value=1,
                                  min_value=1,
                                  step=1)

        st.subheader("Data Type Configuration")
        dtype_name = st.selectbox("Data Type",
                                  ["FP16", "BF16", "FP8", "INT8", "INT4"])

    # Training parameters
    st.subheader("Training Setup")
    col3, col4 = st.columns(2)

    with col3:
        batch_size_per_gpu = st.number_input("Batch Size per GPU",
                                             value=1,
                                             min_value=1,
                                             step=1)
        gradient_accumulation_steps = st.number_input(
            "Gradient Accumulation Steps", value=1, min_value=1, step=1)
        global_batch_size = st.number_input("Global Batch Size (optional)",
                                            value=0,
                                            min_value=0,
                                            step=1)
        seq_len = st.number_input("Sequence Length", value=2048, step=128)
        total_num_tokens = st.number_input("Total Number of Tokens (billions)",
                                           value=300.0,
                                           min_value=0.0,
                                           max_value=10000.0,
                                           step=10.0)
        total_num_tokens *= 1_000_000_000  # Convert to actual token count

    with col4:
        activation_recomputation = st.selectbox("Activation Recomputation", [
            "None", "Attention Compute", "Attention", "Norm-Attention-Norm",
            "Full"
        ])
        activation_recomputation_value = {
            "None": 0,
            "Attention Compute": 1,
            "Attention": 2,
            "Norm-Attention-Norm": 3,
            "Full": 4
        }[activation_recomputation]

        ds_zero = st.selectbox("DeepSpeed ZeRO Stage",
                               ["None", "Stage 1", "Stage 2", "Stage 3"])
        ds_zero_value = {
            "None": 0,
            "Stage 1": 1,
            "Stage 2": 2,
            "Stage 3": 3
        }[ds_zero]

        layernorm_dtype = st.selectbox("LayerNorm Data Type", ["FP16", "FP32"])
        layernorm_dtype_bytes = BYTES_FP16 if layernorm_dtype == "FP16" else BYTES_FP32

        flash_attn = st.checkbox("Use Flash Attention", value=True)
        softmax_dropout = st.checkbox("Softmax Dropout", value=False)

    # Advanced training settings
    with st.expander("Advanced Training Settings"):
        col5, col6 = st.columns(2)

        with col5:
            fwd_prefetch = st.checkbox("Forward Prefetch", value=True)
            bwd_prefetch = st.checkbox("Backward Prefetch", value=True)
            mlp_recompute_act = st.checkbox("MLP Recompute Activation",
                                            value=False)

            master_weights_dtype = st.selectbox("Master Weights Data Type",
                                                ["FP16", "FP32"])
            master_weights_dtype_bytes = BYTES_FP16 if master_weights_dtype == "FP16" else BYTES_FP32

        with col6:
            flops_efficiency = st.slider("FLOPS Efficiency",
                                         min_value=0.0,
                                         max_value=1.0,
                                         value=0.5,
                                         step=0.05)
            hbm_memory_efficiency_value = 0.8  # Default fallback value
            if isinstance(HBM_MEMORY_EFFICIENCY,
                          (int, float)) and 0 <= HBM_MEMORY_EFFICIENCY <= 1:
                hbm_memory_efficiency_value = float(HBM_MEMORY_EFFICIENCY)
            hbm_memory_efficiency = st.slider(
                "HBM Memory Efficiency",
                min_value=0.0,
                max_value=1.0,
                value=hbm_memory_efficiency_value,
                step=0.05)
            intra_node_efficiency_value = 0.7  # Default fallback value
            if isinstance(
                    INTRA_NODE_MEMORY_EFFICIENCY,
                (int, float)) and 0 <= INTRA_NODE_MEMORY_EFFICIENCY <= 1:
                intra_node_efficiency_value = float(
                    INTRA_NODE_MEMORY_EFFICIENCY)

            intra_node_memory_efficiency = st.slider(
                "Intra-Node Memory Efficiency",
                min_value=0.0,
                max_value=1.0,
                value=intra_node_efficiency_value,
                step=0.05)

            inter_node_efficiency_value = 0.6  # Default fallback value
            if isinstance(
                    INTER_NODE_MEMORY_EFFICIENCY,
                (int, float)) and 0 <= INTER_NODE_MEMORY_EFFICIENCY <= 1:
                inter_node_efficiency_value = float(
                    INTER_NODE_MEMORY_EFFICIENCY)

            inter_node_memory_efficiency = st.slider(
                "Inter-Node Memory Efficiency",
                min_value=0.0,
                max_value=1.0,
                value=inter_node_efficiency_value,
                step=0.05)
            num_gpus_per_node = st.number_input("Number of GPUs per Node",
                                                value=NUM_GPUS_PER_NODE,
                                                min_value=1,
                                                step=1)

    # Output directory
    output_dir = st.text_input("Output Directory (optional)", "")
    output_file_prefix = st.text_input("Output File Prefix (optional)", "")
    output_file_suffix = st.text_input("Output File Suffix (optional)", "")

    # Run analysis button
    if st.button("Run Training Analysis"):
        # Create configurations
        model_config = ModelConfig(
            name=st.session_state.model_name,
            hidden_dim=st.session_state.hidden_dim,
            n_head=st.session_state.n_head,
            num_layers=st.session_state.num_layers,
            vocab_size=st.session_state.vocab_size,
            expansion_ratio=st.session_state.expansion_ratio,
            mlp_gated_linear_units=st.session_state.mlp_gated_linear_units,
            moe_num_experts=st.session_state.moe_num_experts,
            moe_top_k=st.session_state.moe_top_k,
            moe_intermediate_size=st.session_state.moe_intermediate_size,
            moe_num_shared_experts=st.session_state.moe_num_shared_experts,
            num_key_value_heads=st.session_state.num_key_value_heads,
            num_key_value_groups=st.session_state.num_key_value_groups,
            max_seq_len=st.session_state.max_seq_len)

        gpu_config = GPUConfig(
            name=gpu_name,
            mem_per_GPU_in_GB=mem_per_GPU_in_GB,
            peak_fp16_TFLOPS=gpu_flops_16bit,
            peak_i8_TFLOPS=gpu_flops_8bit,
            peak_i4_TFLOPS=gpu_flops_4bit,
            hbm_bandwidth_in_GB_per_sec=gpu_hbm_bandwidth,
            intra_node_bandwidth_in_GB_per_sec=300,  # Default value
            intra_node_min_message_latency=0.00001  # Default value
        )

        # Map dtype selection to config
        dtype_mapping = {
            "FP16": {
                "name": "FP16",
                "weight_bits": 16,
                "activation_bits": 16,
                "embedding_bits": 16,
                "linear_weight_bits": 16,
                "linear_activation_bits": 16
            },
            "BF16": {
                "name": "BF16",
                "weight_bits": 16,
                "activation_bits": 16,
                "embedding_bits": 16,
                "linear_weight_bits": 16,
                "linear_activation_bits": 16
            },
            "FP8": {
                "name": "FP8",
                "weight_bits": 8,
                "activation_bits": 8,
                "embedding_bits": 16,
                "linear_weight_bits": 8,
                "linear_activation_bits": 8
            },
            "INT8": {
                "name": "INT8",
                "weight_bits": 8,
                "activation_bits": 8,
                "embedding_bits": 16,
                "linear_weight_bits": 8,
                "linear_activation_bits": 8
            },
            "INT4": {
                "name": "INT4",
                "weight_bits": 4,
                "activation_bits": 8,
                "embedding_bits": 16,
                "linear_weight_bits": 4,
                "linear_activation_bits": 8
            },
        }

        dtype_config = DtypeConfig(**dtype_mapping[dtype_name])

        parallelism_config = ParallelismConfig(tp_size=tp_size,
                                               pp_size=pp_size,
                                               dp_size=dp_size,
                                               ep_size=ep_size,
                                               sp_size=sp_size)

        # Create analyzer
        analyzer = LLMAnalysis(
            model_config=model_config,
            gpu_config=gpu_config,
            dtype_config=dtype_config,
            parallelism_config=parallelism_config,
            flops_efficiency=flops_efficiency,
            hbm_memory_efficiency=hbm_memory_efficiency,
            intra_node_memory_efficiency=intra_node_memory_efficiency,
            inter_node_memory_efficiency=inter_node_memory_efficiency)

        # Run analysis
        with st.spinner("Running training analysis..."):
            results = analyzer.training(
                batch_size_per_gpu=batch_size_per_gpu,
                gradient_accumulation_steps=gradient_accumulation_steps,
                global_batch_size=global_batch_size
                if global_batch_size > 0 else None,
                seq_len=seq_len,
                total_num_tokens=total_num_tokens,
                activation_recomputation=ActivationRecomputation(
                    activation_recomputation_value),
                ds_zero=DSZeRO(ds_zero_value),
                fwd_prefetch=fwd_prefetch,
                bwd_prefetch=bwd_prefetch,
                layernorm_dtype_bytes=layernorm_dtype_bytes,
                master_weights_dtype_bytes=master_weights_dtype_bytes,
                flash_attn=flash_attn,
                softmax_dropout=softmax_dropout,
                mlp_recompute_act=mlp_recompute_act,
                output_dir=output_dir if output_dir else None,
                output_file_prefix=output_file_prefix
                if output_file_prefix else None,
                output_file_suffix=output_file_suffix
                if output_file_suffix else None)

        # Display results
        st.subheader("Training Analysis Results")

        # Create tabs for different result categories
        res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs(
            ["Performance", "Memory", "Latency", "Scaling"])

        with res_tab1:
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Total Parameters",
                          f"{results.get('num_params_total', 0):,}")
                st.metric(
                    "Training Throughput",
                    f"{results.get('training_throughput', 0):,.0f} tokens/sec")
                st.metric(
                    "Training Time",
                    f"{results.get('training_time_in_days', 0):.2f} days")

            with col_p2:
                st.metric("FLOPS Efficiency",
                          f"{results.get('flops_efficiency', 0):.2f}")
                st.metric("HBM Memory Efficiency",
                          f"{results.get('hbm_memory_efficiency', 0):.2f}")
                st.metric("Global Batch Size",
                          f"{results.get('global_batch_size', 0):,}")

        with res_tab2:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric(
                    "Weight Memory per GPU",
                    f"{results.get('weight_memory_per_gpu', 0) / (1024**3):.2f} GB"
                )
                st.metric(
                    "Optimizer Memory per GPU",
                    f"{results.get('optimizer_memory_per_gpu', 0) / (1024**3):.2f} GB"
                )
                st.metric(
                    "Gradient Memory per GPU",
                    f"{results.get('gradient_memory_per_gpu', 0) / (1024**3):.2f} GB"
                )

            with col_m2:
                st.metric(
                    "Activation Memory per GPU",
                    f"{results.get('activation_memory_per_gpu', 0) / (1024**3):.2f} GB"
                )
                st.metric(
                    "Total Memory per GPU",
                    f"{results.get('total_memory_per_gpu', 0) / (1024**3):.2f} GB"
                )
                st.metric("Memory Utilization",
                          f"{results.get('memory_utilization', 0):.2f}")

        with res_tab3:
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                st.metric("Forward Pass Time",
                          f"{results.get('fwd_time', 0) * 1000:.2f} ms")
                st.metric("Backward Pass Time",
                          f"{results.get('bwd_time', 0) * 1000:.2f} ms")
                st.metric(
                    "Optimizer Step Time",
                    f"{results.get('optimizer_step_time', 0) * 1000:.2f} ms")

            with col_l2:
                st.metric("Communication Time",
                          f"{results.get('comm_time', 0) * 1000:.2f} ms")
                st.metric("Step Time",
                          f"{results.get('step_time', 0) * 1000:.2f} ms")
                st.metric("Samples per Second",
                          f"{results.get('samples_per_sec', 0):.2f}")

        with res_tab4:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.metric(
                    "Strong Scaling Efficiency",
                    f"{results.get('strong_scaling_efficiency', 0):.2f}")
                st.metric("Weak Scaling Efficiency",
                          f"{results.get('weak_scaling_efficiency', 0):.2f}")

            with col_s2:
                st.metric(
                    "Model Parallel Efficiency",
                    f"{results.get('model_parallel_efficiency', 0):.2f}")
                st.metric("Data Parallel Efficiency",
                          f"{results.get('data_parallel_efficiency', 0):.2f}")


if __name__ == "__main__":
    main()
