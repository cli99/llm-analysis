# MODEL_NAME=NousResearch/Llama-2-7b-hf
MODEL_NAME=upstage/Llama-2-70b-instruct-v2

python -m llm_analysis.config dump_model_config_by_name --name $MODEL_NAME
