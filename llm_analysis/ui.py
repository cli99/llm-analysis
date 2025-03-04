import logging
from io import StringIO

import streamlit as st

from llm_analysis.analysis import (BYTES_FP16, BYTES_FP32,
                                   ActivationRecomputation, DSZeRO,
                                   LLMAnalysis)
from llm_analysis.config import (DtypeConfig, GPUConfig, ModelConfig,
                                 ParallelismConfig, get_dtype_config_by_name,
                                 get_gpu_config_by_name,
                                 get_model_config_by_name, list_dtype_configs,
                                 list_gpu_configs)
from llm_analysis.constant import (HBM_MEMORY_EFFICIENCY,
                                   INTER_NODE_MEMORY_EFFICIENCY,
                                   INTRA_NODE_MEMORY_EFFICIENCY,
                                   NUM_GPUS_PER_NODE)


def main():

    st.set_page_config(page_title="LLM Analysis", layout="wide")

    # Create a StringIO object to capture log output
    log_output = StringIO()
    # Create a StreamHandler that writes to StringIO
    stream_handler = logging.StreamHandler(log_output)
    # Set the format for the handler
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    # Add the handler to the root logger
    logging.getLogger().addHandler(stream_handler)
    # Set the logging level
    logging.getLogger().setLevel(getattr(logging, "INFO"))

    st.title("LLM Analysis")

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
            # Initialize MoE parameters with -1 if not set
            st.session_state.moe_num_experts = default_model.moe_num_experts if default_model.moe_num_experts is not None else -1
            st.session_state.moe_top_k = default_model.moe_top_k if default_model.moe_top_k is not None else -1
            st.session_state.moe_intermediate_size = default_model.moe_intermediate_size if default_model.moe_intermediate_size is not None else -1
            st.session_state.moe_num_shared_experts = default_model.moe_num_shared_experts if default_model.moe_num_shared_experts is not None else -1
            st.session_state.num_key_value_heads = default_model.num_key_value_heads or default_model.n_head
            st.session_state.num_key_value_groups = default_model.num_key_value_groups or 1
            st.session_state.max_seq_len = default_model.max_seq_len or 4096
            # Add new parameters with None as default
            st.session_state.first_k_dense_replace = default_model.first_k_dense_replace if hasattr(
                default_model, 'first_k_dense_replace') else None
            st.session_state.q_lora_rank = default_model.q_lora_rank if hasattr(
                default_model, 'q_lora_rank') else None
            st.session_state.kv_lora_rank = default_model.kv_lora_rank if hasattr(
                default_model, 'kv_lora_rank') else None
            st.session_state.qk_nope_head_dim = default_model.qk_nope_head_dim if hasattr(
                default_model, 'qk_nope_head_dim') else None
            st.session_state.qk_rope_head_dim = default_model.qk_rope_head_dim if hasattr(
                default_model, 'qk_rope_head_dim') else None
            # Initialize MoE state
            st.session_state.use_moe = default_model.moe_num_experts is not None and default_model.moe_num_experts > 0
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
            st.session_state.moe_num_experts = -1
            st.session_state.moe_top_k = -1
            st.session_state.moe_intermediate_size = -1
            st.session_state.moe_num_shared_experts = -1
            st.session_state.num_key_value_heads = 32
            st.session_state.num_key_value_groups = 1
            st.session_state.max_seq_len = 4096
            # Add new parameters with default values
            st.session_state.first_k_dense_replace = 3
            st.session_state.q_lora_rank = 1536
            st.session_state.kv_lora_rank = 512
            st.session_state.qk_nope_head_dim = 128
            st.session_state.qk_rope_head_dim = 64
            # Initialize MoE state
            st.session_state.use_moe = False

    # Also initialize use_moe if it's not in session state (for cases where initialized is already True)
    if "use_moe" not in st.session_state:
        st.session_state.use_moe = st.session_state.moe_num_experts > 0

    # Initialize GPU configuration session state if not already set
    if "gpu_initialized" not in st.session_state:
        st.session_state.gpu_initialized = True
        st.session_state.current_gpu_name = "h800-sxm-80gb"
        st.session_state.mem_per_GPU_in_GB = 80.0
        st.session_state.gpu_flops_16bit = 989.0
        st.session_state.gpu_flops_8bit = 1979.0
        st.session_state.gpu_flops_4bit = 3958.0
        st.session_state.gpu_hbm_bandwidth = 3200.0
        st.session_state.intra_node_bandwidth = 200.0
        st.session_state.inter_node_bandwidth = 50.0
        st.session_state.intra_node_latency = 0.000008

    # Initialize data type configuration session state if not already set
    if "dtype_initialized" not in st.session_state:
        st.session_state.dtype_initialized = True
        st.session_state.current_dtype_name = "w16a16e16"
        st.session_state.weight_bits = 16
        st.session_state.activation_bits = 16
        st.session_state.embedding_bits = 16
        st.session_state.linear_weight_bits = 16
        st.session_state.linear_activation_bits = 16

    def update_gpu_config():
        """Callback to update GPU configuration when selection changes"""
        selected_name = st.session_state.gpu_type_select_main
        st.session_state.current_gpu_name = selected_name

        if selected_name != "Custom":
            try:
                selected_gpu = get_gpu_config_by_name(selected_name)
                # Update all GPU-related session state values
                st.session_state.mem_per_GPU_in_GB = float(
                    selected_gpu.mem_per_GPU_in_GB)
                st.session_state.gpu_flops_16bit = float(
                    selected_gpu.peak_fp16_TFLOPS)
                st.session_state.gpu_flops_8bit = float(
                    selected_gpu.peak_i8_TFLOPS)
                st.session_state.gpu_flops_4bit = float(
                    selected_gpu.peak_i4_TFLOPS)
                st.session_state.gpu_hbm_bandwidth = float(
                    selected_gpu.hbm_bandwidth_in_GB_per_sec)
                st.session_state.intra_node_bandwidth = float(
                    selected_gpu.intra_node_bandwidth_in_GB_per_sec)
                st.session_state.inter_node_bandwidth = float(
                    selected_gpu.inter_node_bandwidth_in_GB_per_sec)
                st.session_state.intra_node_latency = float(
                    selected_gpu.intra_node_min_message_latency)

                # Update the input widget values by setting their keys in session state
                st.session_state["mem_per_GPU_in_GB_main"] = float(
                    selected_gpu.mem_per_GPU_in_GB)
                st.session_state["gpu_flops_16bit_main"] = float(
                    selected_gpu.peak_fp16_TFLOPS)
                st.session_state["gpu_flops_8bit_main"] = float(
                    selected_gpu.peak_i8_TFLOPS)
                st.session_state["gpu_flops_4bit_main"] = float(
                    selected_gpu.peak_i4_TFLOPS)
                st.session_state["gpu_hbm_bandwidth_main"] = float(
                    selected_gpu.hbm_bandwidth_in_GB_per_sec)
                st.session_state["intra_node_bandwidth_main"] = float(
                    selected_gpu.intra_node_bandwidth_in_GB_per_sec)
                st.session_state["inter_node_bandwidth_main"] = float(
                    selected_gpu.inter_node_bandwidth_in_GB_per_sec)
                st.session_state["intra_node_latency_main"] = float(
                    selected_gpu.intra_node_min_message_latency)
            except Exception as e:
                st.error(f"Error loading GPU config: {str(e)}")
        else:
            # Reset to default Custom values
            default_values = {
                "mem_per_GPU_in_GB": 80.0,
                "gpu_flops_16bit": 989.0,
                "gpu_flops_8bit": 1979.0,
                "gpu_flops_4bit": 3958.0,
                "gpu_hbm_bandwidth": 3200.0,
                "intra_node_bandwidth": 200.0,
                "inter_node_bandwidth": 50.0,
                "intra_node_latency": 0.000008
            }

            # Update both session state and input widget values
            for key, value in default_values.items():
                setattr(st.session_state, key, value)
                st.session_state[f"{key}_main"] = value

    def update_dtype_config():
        """Callback to update data type configuration when selection changes"""
        selected_name = st.session_state.dtype_type_select
        if selected_name != "Custom":
            try:
                selected_dtype = get_dtype_config_by_name(selected_name)
                st.session_state.weight_bits = selected_dtype.weight_bits
                st.session_state.activation_bits = selected_dtype.activation_bits
                st.session_state.embedding_bits = selected_dtype.embedding_bits
                st.session_state.linear_weight_bits = selected_dtype.linear_weight_bits
                st.session_state.linear_activation_bits = selected_dtype.linear_activation_bits
            except Exception as e:
                st.error(f"Error loading data type config: {str(e)}")
        else:
            # Reset to default Custom values (FP16)
            st.session_state.weight_bits = 16
            st.session_state.activation_bits = 16
            st.session_state.embedding_bits = 16
            st.session_state.linear_weight_bits = 16
            st.session_state.linear_activation_bits = 16
        st.session_state.current_dtype_name = selected_name

    st.header("Training")

    # Model Configuration section
    st.subheader("Model Configuration")
    # Model configuration mode selection
    config_mode = st.radio(
        "Configuration Mode",
        ["Load from HuggingFace", "Custom Configuration"],
        help=
        "Choose whether to load a model from HuggingFace or create a custom configuration",
        horizontal=True)

    if config_mode == "Load from HuggingFace":
        col_name, col_button = st.columns([0.7, 0.3])
        with col_name:
            selected_model = st.text_input(
                "HuggingFace Model Name",
                "deepseek-ai/DeepSeek-V3",
                help=
                "e.g., deepseek-ai/DeepSeek-V3, meta-llama/Llama-3.3-70B-Instruct"
            )
        with col_button:
            load_button = st.button("Load Model Configuration")

        if load_button:
            try:
                model_config = get_model_config_by_name(selected_model)
                st.success(
                    f"Successfully loaded configuration for {selected_model}")
                # Update session state with loaded config
                st.session_state.model_name = model_config.name
                st.session_state.hidden_dim = model_config.hidden_dim
                st.session_state.n_head = model_config.n_head
                st.session_state.num_layers = model_config.num_layers
                st.session_state.vocab_size = model_config.vocab_size
                st.session_state.expansion_ratio = model_config.expansion_ratio
                st.session_state.mlp_gated_linear_units = model_config.mlp_gated_linear_units
                st.session_state.moe_num_experts = model_config.moe_num_experts if model_config.moe_num_experts is not None else -1
                st.session_state.moe_top_k = model_config.moe_top_k if model_config.moe_top_k is not None else -1
                st.session_state.moe_intermediate_size = model_config.moe_intermediate_size if model_config.moe_intermediate_size is not None else -1
                st.session_state.moe_num_shared_experts = model_config.moe_num_shared_experts if model_config.moe_num_shared_experts is not None else -1
                st.session_state.num_key_value_heads = model_config.num_key_value_heads or model_config.n_head
                st.session_state.num_key_value_groups = model_config.num_key_value_groups or 1
                st.session_state.max_seq_len = model_config.max_seq_len or 4096
                # Update MoE state after loading configuration
                st.session_state.use_moe = st.session_state.moe_num_experts > 0
                # Add new parameters
                st.session_state.first_k_dense_replace = model_config.first_k_dense_replace if hasattr(
                    model_config, 'first_k_dense_replace') else 3
                st.session_state.q_lora_rank = model_config.q_lora_rank if hasattr(
                    model_config, 'q_lora_rank') else 1536
                st.session_state.kv_lora_rank = model_config.kv_lora_rank if hasattr(
                    model_config, 'kv_lora_rank') else 512
                st.session_state.qk_nope_head_dim = model_config.qk_nope_head_dim if hasattr(
                    model_config, 'qk_nope_head_dim') else 128
                st.session_state.qk_rope_head_dim = model_config.qk_rope_head_dim if hasattr(
                    model_config, 'qk_rope_head_dim') else 64
            except Exception as e:
                st.error(f"Error loading model config: {str(e)}")

    # Model configuration fields in two columns
    config_col1, config_col2 = st.columns(2)
    with config_col1:
        st.session_state.model_name = st.text_input(
            "Model Name", value=st.session_state.model_name)
        st.session_state.hidden_dim = st.number_input(
            "Hidden Dimension", value=st.session_state.hidden_dim, step=128)
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

    with config_col2:
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
        st.session_state.use_moe = st.checkbox("Is Mixture of Experts (MoE)",
                                               value=st.session_state.use_moe,
                                               help="Is Mixture of Experts")

    # MoE Configuration (if enabled)
    if st.session_state.use_moe:
        st.markdown("##### Mixture of Experts (MoE) Configuration")
        # MoE Configuration
        moe_col1, moe_col2 = st.columns(2)
        with moe_col1:
            st.session_state.moe_num_experts = st.number_input(
                "Number of Experts",
                min_value=1,
                value=max(1, st.session_state.moe_num_experts),
                step=1,
                help="Total number of experts")
            st.session_state.moe_top_k = st.number_input(
                "Top-K Experts",
                min_value=1,
                max_value=st.session_state.moe_num_experts,
                value=max(
                    1,
                    min(st.session_state.moe_top_k,
                        st.session_state.moe_num_experts)),
                step=1,
                help="Number of experts to route each token to")
            st.session_state.first_k_dense_replace = st.number_input(
                "First K Dense Replace",
                value=st.session_state.first_k_dense_replace
                if st.session_state.first_k_dense_replace is not None else 3,
                min_value=0,
                step=1,
                help="Number of initial layers to replace with dense layers")
            st.session_state.q_lora_rank = st.number_input(
                "Q LoRA Rank",
                value=st.session_state.q_lora_rank
                if st.session_state.q_lora_rank is not None else 1536,
                min_value=1,
                step=128,
                help="Rank for Q LoRA adaptation")
            st.session_state.kv_lora_rank = st.number_input(
                "KV LoRA Rank",
                value=st.session_state.kv_lora_rank
                if st.session_state.kv_lora_rank is not None else 512,
                min_value=1,
                step=128,
                help="Rank for KV LoRA adaptation")
        with moe_col2:
            st.session_state.moe_num_shared_experts = st.number_input(
                "Number of Shared Experts",
                min_value=0,
                value=max(0, st.session_state.moe_num_shared_experts),
                step=1,
                help="Number of experts shared across all groups")
            st.session_state.moe_intermediate_size = st.number_input(
                "MoE Intermediate Size",
                min_value=0,
                value=max(0, st.session_state.moe_intermediate_size) if
                st.session_state.moe_intermediate_size > 0 else ffn_embed_dim,
                step=128,
                help="Intermediate size for MoE layers")
            st.session_state.qk_nope_head_dim = st.number_input(
                "QK NoPE Head Dimension",
                value=st.session_state.qk_nope_head_dim
                if st.session_state.qk_nope_head_dim is not None else 128,
                min_value=1,
                step=32,
                help="Head dimension for QK without positional encoding")
            st.session_state.qk_rope_head_dim = st.number_input(
                "QK RoPE Head Dimension",
                value=st.session_state.qk_rope_head_dim
                if st.session_state.qk_rope_head_dim is not None else 64,
                min_value=1,
                step=32,
                help="Head dimension for QK with rotary positional encoding")
    else:
        # Reset MoE parameters when disabled
        st.session_state.moe_num_experts = -1
        st.session_state.moe_top_k = -1
        st.session_state.moe_num_shared_experts = -1
        st.session_state.moe_intermediate_size = -1
        # Reset advanced MoE parameters to None
        st.session_state.first_k_dense_replace = None
        st.session_state.q_lora_rank = None
        st.session_state.kv_lora_rank = None
        st.session_state.qk_nope_head_dim = None
        st.session_state.qk_rope_head_dim = None

    # Create a row for Data Type, Hardware, and Parallelism configurations
    config_col1, config_col2, config_col3 = st.columns(3)

    with config_col1:
        st.subheader("Data Type Configuration")
        dtype_configs = list_dtype_configs()

        if not dtype_configs:
            st.error(
                "No data type configurations were loaded. Using Custom configuration only."
            )
            dtype_configs = []

        # Data Type Selection
        dtype_name = st.selectbox(
            "Data Type Name",
            dtype_configs + ["Custom"],
            index=dtype_configs.index("w16a16e16")
            if "w16a16e16" in dtype_configs else len(dtype_configs),
            help=
            "Select from predefined data type configurations or create a custom one",
            key="dtype_type_select",
            on_change=update_dtype_config)

        if st.button(
                "Reset Data Type",
                help=
                "Reset data type configuration to the default values for the selected configuration"
        ):
            update_dtype_config()
            st.success(
                f"Data type configuration reset to default values for {dtype_name}"
            )

        # Bit Width Parameters
        weight_bits = st.number_input("Weight Bits",
                                      min_value=1,
                                      max_value=32,
                                      step=1,
                                      key="weight_bits")
        activation_bits = st.number_input("Activation Bits",
                                          min_value=1,
                                          max_value=32,
                                          step=1,
                                          key="activation_bits")
        embedding_bits = st.number_input("Embedding Bits",
                                         min_value=1,
                                         max_value=32,
                                         step=1,
                                         key="embedding_bits")
        linear_weight_bits = st.number_input("Linear Weight Bits",
                                             min_value=1,
                                             max_value=32,
                                             step=1,
                                             key="linear_weight_bits")
        linear_activation_bits = st.number_input("Linear Activation Bits",
                                                 min_value=1,
                                                 max_value=32,
                                                 step=1,
                                                 key="linear_activation_bits")

    with config_col2:
        st.subheader("Hardware Configuration")
        gpu_configs = list_gpu_configs()

        if not gpu_configs:
            st.error(
                "No GPU configurations were loaded. Using Custom configuration only."
            )
            gpu_configs = []

        # GPU Type Selection
        gpu_name = st.selectbox(
            "GPU Type Name",
            gpu_configs + ["Custom"],
            index=gpu_configs.index("h800-sxm-80gb")
            if "h800-sxm-80gb" in gpu_configs else len(gpu_configs),
            help=
            "Select from predefined GPU configurations or create a custom one",
            key="gpu_type_select_main",
            on_change=update_gpu_config)

        if st.button(
                "Reset GPU Type",
                help=
                "Reset GPU configuration to the default values for the selected GPU type",
                key="reset_gpu_main"):
            update_gpu_config()
            st.success(
                f"GPU configuration reset to default values for {gpu_name}")

        # GPU Parameters
        st.number_input("Memory per GPU (GB)",
                        key="mem_per_GPU_in_GB_main",
                        min_value=0.0,
                        step=1.0,
                        format="%.1f")
        st.number_input("GPU FLOPs 16-bit (TFLOPS)",
                        key="gpu_flops_16bit_main",
                        min_value=0.0,
                        step=1.0,
                        format="%.1f")
        st.number_input("GPU FLOPs 8-bit (TFLOPS)",
                        key="gpu_flops_8bit_main",
                        min_value=0.0,
                        step=1.0,
                        format="%.1f")
        st.number_input("GPU FLOPs 4-bit (TFLOPS)",
                        key="gpu_flops_4bit_main",
                        min_value=0.0,
                        step=1.0,
                        format="%.1f")
        st.number_input("GPU HBM Bandwidth (GB/s)",
                        key="gpu_hbm_bandwidth_main",
                        min_value=0.0,
                        step=1.0,
                        format="%.1f")
        st.number_input("Intra-node Bandwidth (GB/s)",
                        key="intra_node_bandwidth_main",
                        min_value=0.0,
                        step=1.0,
                        format="%.1f")
        st.number_input("Inter-node Bandwidth (GB/s)",
                        key="inter_node_bandwidth_main",
                        min_value=0.0,
                        step=1.0,
                        format="%.1f")
        st.number_input("Intra-node Message Latency (s)",
                        key="intra_node_latency_main",
                        min_value=0.0,
                        format="%.5f",
                        step=0.00001)

    with config_col3:
        st.subheader("Parallelism Configuration")
        total_num_gpus = st.number_input("Total Number of GPUs",
                                         value=8,
                                         min_value=1,
                                         step=1)
        num_gpus_per_node = st.number_input("Number of GPUs per Node",
                                            value=NUM_GPUS_PER_NODE,
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

    # Training Setup section
    st.subheader("Training Setup")
    train_col1, train_col2, train_col3 = st.columns(3)
    with train_col1:
        batch_size_per_gpu = st.number_input("Batch Size per GPU",
                                             value=1,
                                             min_value=1,
                                             step=1)
        gradient_accumulation_steps = st.number_input(
            "Gradient Accumulation Steps", value=1, min_value=1, step=1)
        global_batch_size = st.number_input("Global Batch Size (optional)",
                                            value=-1,
                                            min_value=-1,
                                            step=1)
        seq_len = st.number_input("Sequence Length", value=2048, step=128)
        total_num_tokens = st.number_input("Total Number of Tokens (billions)",
                                           value=300.0,
                                           min_value=0.0,
                                           max_value=10000.0,
                                           step=10.0)
        total_num_tokens *= 1_000_000_000  # Convert to actual token count

    with train_col2:
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

        master_weights_dtype = st.selectbox("Master Weights Data Type",
                                            ["FP16", "FP32"])
        master_weights_dtype_bytes = BYTES_FP16 if master_weights_dtype == "FP16" else BYTES_FP32

        fwd_prefetch = st.checkbox("Forward Prefetch", value=True)
        bwd_prefetch = st.checkbox("Backward Prefetch", value=True)
        mlp_recompute_act = st.checkbox("MLP Recompute Activation",
                                        value=False)
        flash_attn = st.checkbox("Use Flash Attention", value=True)
        softmax_dropout = st.checkbox("Softmax Dropout", value=False)

    with train_col3:
        flops_efficiency = st.slider("FLOPS Efficiency",
                                     min_value=0.0,
                                     max_value=1.0,
                                     value=0.5,
                                     step=0.05)

        hbm_memory_efficiency_value = 0.8  # Default fallback value
        if isinstance(HBM_MEMORY_EFFICIENCY,
                      (int, float)) and 0 <= HBM_MEMORY_EFFICIENCY <= 1:
            hbm_memory_efficiency_value = float(HBM_MEMORY_EFFICIENCY)
        hbm_memory_efficiency = st.slider("HBM Memory Efficiency",
                                          min_value=0.0,
                                          max_value=1.0,
                                          value=hbm_memory_efficiency_value,
                                          step=0.05)

        intra_node_efficiency_value = 0.7  # Default fallback value
        if isinstance(INTRA_NODE_MEMORY_EFFICIENCY,
                      (int, float)) and 0 <= INTRA_NODE_MEMORY_EFFICIENCY <= 1:
            intra_node_efficiency_value = float(INTRA_NODE_MEMORY_EFFICIENCY)
        intra_node_memory_efficiency = st.slider(
            "Intra-Node Memory Efficiency",
            min_value=0.0,
            max_value=1.0,
            value=intra_node_efficiency_value,
            step=0.05)

        inter_node_efficiency_value = 0.6  # Default fallback value
        if isinstance(INTER_NODE_MEMORY_EFFICIENCY,
                      (int, float)) and 0 <= INTER_NODE_MEMORY_EFFICIENCY <= 1:
            inter_node_efficiency_value = float(INTER_NODE_MEMORY_EFFICIENCY)
        inter_node_memory_efficiency = st.slider(
            "Inter-Node Memory Efficiency",
            min_value=0.0,
            max_value=1.0,
            value=inter_node_efficiency_value,
            step=0.05)

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
            max_seq_len=st.session_state.max_seq_len,
            # Add new parameters
            first_k_dense_replace=st.session_state.first_k_dense_replace,
            q_lora_rank=st.session_state.q_lora_rank,
            kv_lora_rank=st.session_state.kv_lora_rank,
            qk_nope_head_dim=st.session_state.qk_nope_head_dim,
            qk_rope_head_dim=st.session_state.qk_rope_head_dim)

        gpu_config = GPUConfig(
            name=st.session_state.current_gpu_name,
            mem_per_GPU_in_GB=st.session_state.mem_per_GPU_in_GB,
            peak_fp16_TFLOPS=st.session_state.gpu_flops_16bit,
            peak_i8_TFLOPS=st.session_state.gpu_flops_8bit,
            peak_i4_TFLOPS=st.session_state.gpu_flops_4bit,
            hbm_bandwidth_in_GB_per_sec=st.session_state.gpu_hbm_bandwidth,
            intra_node_bandwidth_in_GB_per_sec=st.session_state.
            intra_node_bandwidth,
            inter_node_bandwidth_in_GB_per_sec=st.session_state.
            inter_node_bandwidth,
            intra_node_min_message_latency=st.session_state.intra_node_latency)

        parallelism_config = ParallelismConfig(tp_size=tp_size,
                                               pp_size=pp_size,
                                               dp_size=dp_size,
                                               ep_size=ep_size,
                                               sp_size=sp_size)

        # Create dtype config with the current values from session state
        dtype_config = DtypeConfig(
            name=st.session_state.current_dtype_name,
            weight_bits=st.session_state.weight_bits,
            activation_bits=st.session_state.activation_bits,
            embedding_bits=st.session_state.embedding_bits,
            linear_weight_bits=st.session_state.linear_weight_bits,
            linear_activation_bits=st.session_state.linear_activation_bits)

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
                output_dir=None,
                output_file_prefix=None,
                output_file_suffix=None)

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

    # Add debug window at the bottom
    with st.expander("Debug Logs", expanded=False):
        st.text(log_output.getvalue())


if __name__ == "__main__":
    main()
