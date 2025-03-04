import logging
from io import StringIO

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from llm_analysis.analysis import (BYTES_FP16, BYTES_FP32,
                                   ActivationRecomputation, DSZeRO,
                                   LLMAnalysis)
from llm_analysis.config import (DtypeConfig, GPUConfig, ModelConfig,
                                 ParallelismConfig, get_dtype_config_by_name,
                                 get_gpu_config_by_name,
                                 get_model_config_by_name, list_dtype_configs,
                                 list_gpu_configs)
from llm_analysis.constant import (INTER_NODE_MEMORY_EFFICIENCY,
                                   NUM_GPUS_PER_NODE)


def main():

    st.set_page_config(page_title="LLM Analysis", layout="wide")

    # Add custom CSS for print layout
    st.markdown("""
        <style>
        .stMetric [data-testid="stMetricValue"] {
            font-size: 1rem;
        }

        /* Print-specific styles */
        @media print {
            /* Ensure content scales properly */
            body {
                width: 100%;
                margin: 0;
                padding: 0;
            }

            /* Preserve Streamlit's layout structure */
            .main .block-container {
                max-width: none !important;
                padding: 0 !important;
            }

            /* Preserve multi-column layout */
            div[data-testid="column"] {
                display: table-cell !important;
                width: 33.33% !important;
                padding: 0.5cm !important;
                vertical-align: top !important;
            }

            div[data-testid="stHorizontalBlock"] {
                display: table !important;
                width: 100% !important;
                table-layout: fixed !important;
            }

            /* Ensure metrics are visible and properly sized */
            .stMetric {
                break-inside: avoid;
                margin-bottom: 0.5cm;
            }

            /* Hide UI elements not needed in print */
            .stButton,
            button[kind="secondary"],
            .stSpinner,
            .stSelectbox,
            .stDownloadButton,
            .stToolbar {
                display: none !important;
            }

            /* Ensure charts print properly */
            .js-plotly-plot {
                break-inside: avoid;
                page-break-inside: avoid;
                margin: 0.5cm 0;
                width: 100% !important;
            }

            /* Keep chart legends visible */
            .legend {
                display: block !important;
                visibility: visible !important;
                opacity: 1 !important;
            }

            /* Ensure SVG elements maintain proper dimensions */
            svg.main-svg {
                max-height: 400px !important;
                width: 100% !important;
            }

            /* Ensure proper page breaks */
            .element-container {
                break-inside: avoid;
            }
        }
        </style>
    """,
                unsafe_allow_html=True)

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
    st.divider()

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
        try:
            # Load H800 SXM config as default
            default_gpu = get_gpu_config_by_name("h800-sxm-80gb")
            st.session_state.gpu_initialized = True
            st.session_state.current_gpu_name = default_gpu.name
            st.session_state.mem_per_GPU_in_GB = float(
                default_gpu.mem_per_GPU_in_GB)
            st.session_state.gpu_flops_16bit = float(
                default_gpu.peak_fp16_TFLOPS)
            st.session_state.gpu_flops_8bit = float(default_gpu.peak_i8_TFLOPS)
            st.session_state.gpu_flops_4bit = float(default_gpu.peak_i4_TFLOPS)
            st.session_state.gpu_hbm_bandwidth = float(
                default_gpu.hbm_bandwidth_in_GB_per_sec)
            st.session_state.intra_node_bandwidth = float(
                default_gpu.intra_node_bandwidth_in_GB_per_sec)
            st.session_state.inter_node_bandwidth = float(
                default_gpu.inter_node_bandwidth_in_GB_per_sec)
            st.session_state.intra_node_latency = float(
                default_gpu.intra_node_min_message_latency)
        except Exception as e:
            # Fallback to hardcoded H800 values if loading fails
            st.error(
                f"Error loading GPU config: {str(e)}. Using hardcoded H800 values."
            )
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
        try:
            # Load w16a16e16l8 config as default
            default_dtype = get_dtype_config_by_name("w16a16e16l8")
            st.session_state.dtype_initialized = True
            st.session_state.current_dtype_name = default_dtype.name
            st.session_state.weight_bits = default_dtype.weight_bits
            st.session_state.activation_bits = default_dtype.activation_bits
            st.session_state.embedding_bits = default_dtype.embedding_bits
            st.session_state.linear_weight_bits = default_dtype.linear_weight_bits
            st.session_state.linear_activation_bits = default_dtype.linear_activation_bits
            st.session_state.master_weights_dtype = "FP32"
            st.session_state.mlp_activation_quant_bits = 8
        except Exception as e:
            # Fallback to hardcoded w16a16e16l8 values if loading fails
            st.error(
                f"Error loading data type config: {str(e)}. Using hardcoded w16a16e16l8 values."
            )
            st.session_state.dtype_initialized = True
            st.session_state.current_dtype_name = "w16a16e16l8"
            st.session_state.weight_bits = 16
            st.session_state.activation_bits = 16
            st.session_state.embedding_bits = 16
            st.session_state.linear_weight_bits = 16
            st.session_state.linear_activation_bits = 16
            st.session_state.master_weights_dtype = "FP32"
            st.session_state.mlp_activation_quant_bits = 8

    def update_gpu_config():
        """Callback to update GPU configuration when selection changes"""
        selected_name = st.session_state.gpu_type_select
        st.session_state.current_gpu_name = selected_name

        if selected_name != "Custom":
            try:
                selected_gpu = get_gpu_config_by_name(selected_name)
                # Update GPU-related session state values
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
            except Exception as e:
                st.error(f"Error loading GPU config: {str(e)}")
        else:
            # Reset to default Custom values
            st.session_state.mem_per_GPU_in_GB = 80.0
            st.session_state.gpu_flops_16bit = 989.0
            st.session_state.gpu_flops_8bit = 1979.0
            st.session_state.gpu_flops_4bit = 3958.0
            st.session_state.gpu_hbm_bandwidth = 3200.0
            st.session_state.intra_node_bandwidth = 200.0
            st.session_state.inter_node_bandwidth = 50.0
            st.session_state.intra_node_latency = 0.000008

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
    st.divider()

    # Model Configuration section
    st.subheader("Model Architecture")
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
    st.divider()
    config_col1, config_col2, config_col3 = st.columns(3)

    with config_col1:
        st.subheader("Precision Settings")
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
            index=dtype_configs.index("w16a16e16l8")
            if "w16a16e16l8" in dtype_configs else len(dtype_configs),
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
        st.subheader("GPU Specifications")
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
            key="gpu_type_select",
            on_change=update_gpu_config)

        if st.button(
                "Reset GPU Type",
                help=
                "Reset GPU configuration to the default values for the selected GPU type"
        ):
            update_gpu_config()
            st.success(
                f"GPU configuration reset to default values for {gpu_name}")

        # GPU Parameters
        st.number_input("Memory per GPU (GB)",
                        key="mem_per_GPU_in_GB",
                        min_value=0.0,
                        step=1.0,
                        format="%.1f")
        st.number_input("GPU FLOPs 16-bit (TFLOPS)",
                        key="gpu_flops_16bit",
                        min_value=0.0,
                        step=1.0,
                        format="%.1f")
        st.number_input("GPU FLOPs 8-bit (TFLOPS)",
                        key="gpu_flops_8bit",
                        min_value=0.0,
                        step=1.0,
                        format="%.1f")
        st.number_input("GPU FLOPs 4-bit (TFLOPS)",
                        key="gpu_flops_4bit",
                        min_value=0.0,
                        step=1.0,
                        format="%.1f")
        st.number_input("GPU HBM Bandwidth (GB/s)",
                        key="gpu_hbm_bandwidth",
                        min_value=0.0,
                        step=1.0,
                        format="%.1f")
        st.number_input("Intra-node Bandwidth (GB/s)",
                        key="intra_node_bandwidth",
                        min_value=0.0,
                        step=1.0,
                        format="%.1f")
        st.number_input("Inter-node Bandwidth (GB/s)",
                        key="inter_node_bandwidth",
                        min_value=0.0,
                        step=1.0,
                        format="%.1f")
        st.number_input("Intra-node Message Latency (s)",
                        key="intra_node_latency",
                        min_value=0.0,
                        format="%.5f",
                        step=0.00001)

    with config_col3:
        st.subheader("Parallelism Configuration")
        total_num_gpus = st.number_input("Total Number of GPUs",
                                         value=2048,
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
                                  value=16,
                                  min_value=1,
                                  step=1)
        dp_size = st.number_input("Data Parallelism Size",
                                  value=128,
                                  min_value=1,
                                  step=1)
        ep_size = st.number_input("Expert Parallelism Size",
                                  value=64,
                                  min_value=1,
                                  step=1)
        sp_size = st.number_input("Sequence Parallelism Size",
                                  value=1,
                                  min_value=1,
                                  step=1)

    # Training Parameters section
    st.divider()
    st.subheader("Training Setup & Optimization")
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
        seq_len = st.number_input("Sequence Length", value=4096, step=128)
        total_num_tokens = st.number_input("Total Number of Tokens (billions)",
                                           value=14.8,
                                           min_value=0.0,
                                           max_value=10000.0,
                                           step=0.1)
        total_num_tokens *= 1_000_000_000_000  # Convert to actual token count
        mlp_activation_quant_bits = st.number_input(
            "MLP Activation Quantization Bits",
            min_value=1,
            max_value=32,
            value=st.session_state.mlp_activation_quant_bits,
            step=1,
            help="Number of bits for MLP activation quantization")
        other_op_bytes = st.number_input(
            "Optimizer State Bytes",
            min_value=1,
            value=4,
            step=1,
            help=
            "Number of bytes in the optimizer state per parameter. Defaults to 4 (assumes using Adam optimizer)"
        )

    with train_col2:
        activation_recomputation = st.selectbox("Activation Recomputation", [
            "None", "Attention Compute", "Attention", "Norm-Attention-Norm",
            "Full"
        ],
                                                index=2)
        activation_recomputation_value = {
            "None": 0,
            "Attention Compute": 1,
            "Attention": 2,
            "Norm-Attention-Norm": 3,
            "Full": 4
        }[activation_recomputation]

        ds_zero = st.selectbox("DeepSpeed ZeRO Stage",
                               ["None", "Stage 1", "Stage 2", "Stage 3"],
                               index=1)
        ds_zero_value = {
            "None": 0,
            "Stage 1": 1,
            "Stage 2": 2,
            "Stage 3": 3
        }[ds_zero]

        layernorm_dtype = st.selectbox("LayerNorm Data Type", ["FP16", "FP32"])
        layernorm_dtype_bytes = BYTES_FP16 if layernorm_dtype == "FP16" else BYTES_FP32

        master_weights_dtype = st.selectbox("Master Weights Data Type",
                                            ["FP16", "FP32"],
                                            index=1,
                                            key="master_weights_dtype")
        master_weights_dtype_bytes = BYTES_FP16 if master_weights_dtype == "FP16" else BYTES_FP32

        fwd_prefetch = st.checkbox("Forward Prefetch", value=True)
        bwd_prefetch = st.checkbox("Backward Prefetch", value=True)
        mlp_recompute_act = st.checkbox("MLP Recompute Activation", value=True)
        flash_attn = st.checkbox("Use Flash Attention", value=True)
        ignore_comm_latency = st.checkbox(
            "Ignore Communication Latency",
            value=True,
            help="Ignore communication cost in analysis")
        softmax_dropout = st.checkbox("Softmax Dropout", value=False)

    with train_col3:
        flops_efficiency = st.slider("FLOPS Efficiency",
                                     min_value=0.0,
                                     max_value=1.0,
                                     value=0.2,
                                     step=0.05)

        hbm_memory_efficiency = st.slider("HBM Memory Efficiency",
                                          min_value=0.0,
                                          max_value=1.0,
                                          value=0.8,
                                          step=0.05)

        intra_node_memory_efficiency = st.slider(
            "Intra-Node Memory Efficiency",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
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
            inter_node_memory_efficiency=inter_node_memory_efficiency,
            ignore_comm_latency=ignore_comm_latency)

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
                mlp_activation_quant_bits=mlp_activation_quant_bits,
                other_op_bytes=other_op_bytes,
                output_dir=None,
                output_file_prefix=None,
                output_file_suffix=None)

        # Display results
        st.divider()
        st.subheader("Analysis Results")
        st.divider()

        # Configuration Summary
        st.markdown("#### Configuration")
        conf_col1, conf_col2, conf_col3, conf_col4 = st.columns(4)
        with conf_col1:
            st.metric("Batch Size per GPU",
                      f"{results.get('batch_size_per_gpu', 0)}")
            st.metric("Max Batch Size per GPU",
                      f"{results.get('max_batch_size_per_gpu', 0)}")
            st.metric("Gradient Accumulation Steps",
                      f"{results.get('gradient_accumulation_steps', 0)}")
            st.metric("Global Batch Size",
                      f"{results.get('global_batch_size', 0)}")
            st.metric("Sequence Length", f"{results.get('seq_len', 0)}")
            st.metric("Total Tokens",
                      f"{results.get('total_num_tokens', 0)/1e12:.1f} T")

        with conf_col2:
            st.metric("Data Parallel Size", f"{results.get('dp_size', 0)}")
            st.metric("Tensor Parallel Size", f"{results.get('tp_size', 0)}")
            st.metric("Pipeline Parallel Size", f"{results.get('pp_size', 0)}")
            st.metric("Sequence Parallel Size", f"{results.get('sp_size', 0)}")
            st.metric("Expert Parallel Size", f"{results.get('ep_size', 0)}")
            st.metric("Total GPUs", f"{results.get('total_num_gpus', 0)}")

        with conf_col3:
            st.metric("Total Parameters",
                      f"{results.get('num_params_total', 0)/1e9:.2f} B")
            st.metric(
                "Active Parameters",
                f"{results.get('num_active_params_total', 0)/1e9:.2f} B")
            st.metric("ZeRO Stage", f"{ds_zero}")
            st.metric("Activation Recomputation",
                      f"{activation_recomputation}")
            st.metric("MLP Activation Bits", f"{mlp_activation_quant_bits}")

        with conf_col4:
            st.metric(
                "FLOPS Efficiency",
                f"{results.get('flops_efficiency', 0):.2f}",
                help="Achieved FLOPS as a fraction of peak theoretical FLOPS")
            st.metric(
                "HBM Memory Efficiency",
                f"{results.get('hbm_memory_efficiency', 0):.2f}",
                help=
                "Achieved memory bandwidth as a fraction of peak theoretical bandwidth"
            )

        # Performance Metrics
        st.divider()
        st.markdown("#### Performance Metrics")
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        with perf_col1:
            st.markdown("**Throughput**")
            st.metric(
                "FLOPS per Micro-batch",
                f"{results.get('num_flops_total_per_micro_batch', 0)/1e12:.2f} T",
                help="Total floating point operations performed per micro-batch"
            )
            st.metric("Tokens per Second",
                      f"{results.get('device_tokens_per_sec', 0):,.0f}",
                      help="Number of tokens processed per second per GPU")

        with perf_col2:
            st.markdown("**Training Cost**")
            st.metric(
                "Training Time",
                f"{int(results.get('total_training_latency', 0)/3600/24)} days",
                help="Total time required to complete the training")
            st.metric("GPU Hours",
                      f"{results.get('gpu_hours', 0):,.0f}",
                      help="Total GPU compute hours needed for training")

        # Memory Summary
        st.divider()
        st.markdown("#### Memory Usage (per GPU)")
        mem_col1, mem_col2, mem_col3 = st.columns(3)
        with mem_col1:
            st.markdown("**Breakdown**")
            st.metric(
                "Weight Memory",
                f"{results.get('weight_memory_per_gpu', 0) / (1024**3):.2f} GB",
                help=
                "Memory required to store model weights. Depends on model size and data type configuration."
            )
            st.metric(
                "Gradient Memory",
                f"{results.get('gradient_memory_per_gpu', 0) / (1024**3):.2f} GB",
                help=
                "Memory needed for storing gradients during backpropagation.")
            st.metric(
                "Optimizer State Memory",
                f"{results.get('optimizer_state_memory_per_gpu', 0) / (1024**3):.2f} GB",
                help=
                "Memory used by optimizer states, including master weights and other optimizer states (e.g., momentum and variance in Adam)."
            )
            st.metric(
                "Activation Memory",
                f"{results.get('activation_memory_per_gpu', 0) / (1024**3):.2f} GB",
                help=
                "Memory required for storing activations during forward pass. Can be reduced using activation recomputation."
            )

        with mem_col2:
            st.markdown("**Totals**")
            st.metric(
                "Weight + Optimizer State",
                f"{results.get('(weight+op_state)_memory_per_gpu', 0) / (1024**3):.2f} GB",
                help=
                "Total memory needed for weights and optimizer states. This memory is always occupied during training."
            )
            st.metric(
                "Weight + Optimizer + Gradient",
                f"{results.get('(weight+op_state+grad)_memory_per_gpu', 0) / (1024**3):.2f} GB",
                help=
                "Weight + Optimizer + Gradient. This represents memory usage during backpropagation phase."
            )
            st.metric(
                "Weight + Optimizer + Activations",
                f"{results.get('(weight+op_state+act)_memory_per_gpu', 0) / (1024**3):.2f} GB",
                help=
                "Weight + Optimizer + Activation . This represents the typical memory usage during forward pass."
            )
            st.metric(
                "Estimated Peak Memory",
                f"{results.get('estimated_peak_memory_per_gpu', 0) / (1024**3):.2f} GB",
                help=
                "Peak memory is calculated as: Optimizer State + Weights + max(Activations, Gradients) + max(Backward Prefetch Memory, Loss Backward Memory). This represents the maximum memory usage during training, accounting for memory that can be reused between different phases."
            )

        with mem_col3:
            st.markdown("**Forward Memory Distribution**")
            # Create pie chart for memory usage breakdown
            weight_mem = results.get('weight_memory_per_gpu', 0) / (1024**3)
            optimizer_mem = results.get('optimizer_state_memory_per_gpu',
                                        0) / (1024**3)
            activation_mem = results.get('activation_memory_per_gpu',
                                         0) / (1024**3)

            # Memory Distribution pie chart
            fig = go.Figure(data=[
                go.Pie(
                    labels=['Weights', 'Optimizer State', 'Activations'],
                    values=[weight_mem, optimizer_mem, activation_mem],
                    textinfo='percent',
                    textposition='inside',
                    insidetextorientation='horizontal',
                    hovertemplate=
                    "<b>%{label}</b><br>%{value:.2f} GB<br>%{percent}",
                    hole=0.4,
                    marker=dict(colors=['#2ecc71', '#3498db', '#e74c3c']),
                    name="",  # This removes the trace_0 prefix
                    textfont=dict(
                        color="#ffffff"
                    )  # Use white text for better contrast with pie colors
                )
            ])
            fig.update_layout(**get_pie_chart_layout())
            st.plotly_chart(fig, use_container_width=True)

        # Latency Summary
        st.divider()
        st.markdown("#### Latency Breakdown")
        lat_col1, lat_col2, lat_col3 = st.columns(3)
        with lat_col1:
            st.markdown("**Forward Pass Timing**")
            fwd_total = results.get('latency_fwd', 0) * 1000
            attn_time = results.get('latency_fwd_attn', 0) * 1000
            mlp_time = results.get('latency_fwd_mlp', 0) * 1000
            ln_time = results.get('latency_fwd_layernorm', 0) * 1000
            comm_time = (results.get('latency_fwd_tp_comm', 0) +
                         results.get('latency_fwd_sharded_dp_comm', 0)) * 1000
            input_embed_time = results.get('latency_fwd_input_embedding',
                                           0) * 1000
            output_embed_time = results.get(
                'latency_fwd_output_embedding_loss', 0) * 1000

            st.metric("Total Forward", f"{fwd_total:.2f} ms")
            st.metric("Attention", f"{attn_time:.2f} ms")
            st.metric("MLP", f"{mlp_time:.2f} ms")
            st.metric("LayerNorm", f"{ln_time:.2f} ms")
            st.metric("Communication", f"{comm_time:.2f} ms")
            st.metric("Input Embedding", f"{input_embed_time:.2f} ms")
            st.metric("Output Embedding/Loss", f"{output_embed_time:.2f} ms")

        with lat_col2:
            st.markdown("**Overall Timing**")
            st.metric(
                "Per Micro-batch",
                f"{results.get('latency_per_micro_batch', 0) * 1000:.2f} ms")
            st.metric("Per Iteration",
                      f"{results.get('latency_per_iter', 0) * 1000:.2f} ms")

        with lat_col3:
            st.markdown("**Forward Time Distribution**")
            # Create pie chart for latency breakdown
            comm_time = max(
                0, fwd_total - (attn_time + mlp_time + ln_time +
                                input_embed_time + output_embed_time))

            fig = go.Figure(data=[
                go.Pie(
                    labels=[
                        'Attention', 'MLP', 'LayerNorm', 'Input Embed',
                        'Output Embed', 'Communication'
                    ],
                    values=[
                        attn_time, mlp_time, ln_time, input_embed_time,
                        output_embed_time, comm_time
                    ],
                    textinfo='percent',
                    textposition='inside',
                    insidetextorientation='horizontal',
                    hovertemplate=
                    "<b>%{label}</b><br>%{value:.2f} ms<br>%{percent}",
                    hole=0.4,
                    marker=dict(colors=[
                        '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#e67e22',
                        '#e74c3c'
                    ]),
                    name="",  # This removes the trace_0 prefix
                    textfont=dict(
                        color="#ffffff"
                    )  # Use white text for better contrast with pie colors
                )
            ])
            fig.update_layout(**get_pie_chart_layout())
            st.plotly_chart(fig, use_container_width=True)

    # Add debug window at the bottom
    st.divider()
    with st.expander("Debug Logs", expanded=False):
        st.text(log_output.getvalue())

    show_print_button = """
        <script>
            function print_page(obj) {
                // Hide button during print
                obj.style.display = "none";

                // Set optimal print settings
                const style = document.createElement('style');
                style.textContent = `
                    @page {
                        size: landscape;
                        margin: 1cm;
                    }
                    @media print {
                        html, body {
                            height: auto !important;
                            overflow: visible !important;
                            -webkit-print-color-adjust: exact !important;
                            print-color-adjust: exact !important;
                        }
                    }
                `;
                document.head.appendChild(style);

                // Wait for charts to finish rendering and handle print
                setTimeout(() => {
                    // Force all charts to proper dimensions before printing
                    const charts = document.querySelectorAll('.js-plotly-plot');
                    charts.forEach(chart => {
                        if (chart && chart.layout) {
                            Plotly.relayout(chart, {
                                'autosize': true,
                                'width': null,
                                'height': 400
                            });
                        }
                    });

                    // Print after a short delay to ensure charts are resized
                    setTimeout(() => {
                        parent.window.print();
                        document.head.removeChild(style);
                        obj.style.display = "block";
                    }, 200);
                }, 500);
            }
        </script>
        <button style="
            padding: 0.5rem 1rem;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin: 1rem 0;
            font-size: 1rem;
            display: block;
        " onclick="print_page(this)">
            Export to PDF (Landscape)
        </button>
        """
    components.html(show_print_button)


def get_pie_chart_layout():
    return dict(showlegend=True,
                legend=dict(orientation="h",
                            yanchor="bottom",
                            y=-0.2,
                            xanchor="center",
                            x=0.5,
                            bgcolor='rgba(0,0,0,0)',
                            borderwidth=0,
                            font=dict(size=14)),
                height=400,
                margin=dict(t=30, l=0, r=0, b=80),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)')


if __name__ == "__main__":
    main()
