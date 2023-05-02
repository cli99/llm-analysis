# llm-analysis

[![PyPI](https://img.shields.io/pypi/v/llm-analysis.svg)](https://pypi.org/project/llm-analysis/)
[![Read the Docs](https://readthedocs.org/projects/llm-analysis/badge/)](https://llm-analysis.readthedocs.io/)
[![Tests](https://github.com/cli99/llm-analysis/workflows/tests/badge.svg)](https://github.com/cli99/llm-analysis/actions?workflow=tests)
[![Codecov](https://codecov.io/gh/cli99/llm-analysis/branch/main/graph/badge.svg)](https://codecov.io/gh/cli99/llm-analysis)
[![GitHub license](https://img.shields.io/github/license/cli99/llm-analysis)](https://github.com/cli99/llm-analysis/blob/main/LICENSE)

> Latency and Memory Analysis of Transformer Models for Training and Inference

- [llm-analysis](#llm-analysis)
  - [Overview](#overview)
    - [Examples (updating)](#examples-updating)
  - [Quick Start](#quick-start)
    - [Using the `LLMAnalysis` class](#using-the-llmanalysis-class)
    - [Using the Entry Point Functions for Command Line](#using-the-entry-point-functions-for-command-line)
    - [How to Set FLOPS and Memory Efficiency](#how-to-set-flops-and-memory-efficiency)
  - [Current Scope (expanding) and Limitations](#current-scope-expanding-and-limitations)
  - [TODOs (stay tuned :radio:)](#todos-stay-tuned-radio)
  - [Citation](#citation)
  - [Contributing](#contributing)
  - [Useful Links](#useful-links)

## Overview

Many formulas or equations are floating around in papers, blogs, etc., about how to calculate training or inference latency and memory for Large Language Models (LLMs) or Transformers. Rather than doing math on papers or typing in Excel sheets, `let's automate the boring stuff with llm-analysis` :gear:!

Given the specified model, GPU, data type, and parallelism configurations, llm-analysis estimates the latency and memory usage of LLMs for training or inference. With llm-analysis, one can easily try out different training/inference setups theoretically, and better understand the system performance for different scenarios.

llm-analysis helps answer questions such as:
- what batch size, data type, parallelism scheme to use to get a `feasible` (not getting OOM) and `optimal` (maximizing throughput with a latency constraint) setup for training or inference
- `time` it takes with the given setup to do training or inference and the `cost` (GPU-hours)
- how the latency/memory changes if using a different model, GPU type, number of GPU, data type for weights and activations, parallelism configuration (suggesting the performance benefit of `modeling change`, `hardware improvement`, `quantization`, `parallelism`, etc.)

### Examples (updating)

Check the example use cases. With llm-analysis, you can do such analysis in minutes :rocket:!
- [LLaMA Models](examples/llama)
- [Megatron-LM Models](examples/megatron-lm/)
- [Megatron Turing NLG 530B](examples/megatron-turing-nlg)
- [FasterTransformer](examples/fastertransformer/)
- OPT Models (WIP)
- BLOOM Models (WIP)
## Quick Start

- To install llm-analysis from pypi:
  ```sh
  pip install llm-analysis
  ```

- To install the latest development build:

  ```sh
  pip install --upgrade git+https://github.com/cli99/llm-analysis.git@main
  ```

- To install from source, clone the repo and run `pip install .` or `poetry install` (install [poetry](https://python-poetry.org/) by `pip install poetry`).

### Using the `LLMAnalysis` class

To integrate llm-analysis in your code, use the `LLMAnalysis` class. Refer to doc [LLMAnalysis](https://llm-analysis.readthedocs.io/) for details.

`LLMAnalysis` is constructed with flops and memory efficiency numbers and the following [configuration classes](https://llm-analysis.readthedocs.io/en/latest/config.html):

- `ModelConfig` covers model information, i.e. max sequence length, number of transformer layers, number of attention heads, hidden dimension, vocabulary size
- `GPUConfig` covers GPU compute and memory specifications
- `DtypeConfig` covers the number of bits used for the model weight, activation, and embedding
- `ParallelismConfig` covers Tensor Parallelism (`tp`), Pipeline Parallelism (`pp`), Sequence Parallelism (`sp`), and Data Parallelism (`dp`).

Then `LLMAnalysis` can be queried with different arguments through the [training](https://llm-analysis.readthedocs.io/en/latest/analysis.html#llm_analysis.analysis.LLMAnalysis.training) and [inference](https://llm-analysis.readthedocs.io/en/latest/analysis.html#llm_analysis.analysis.LLMAnalysis.inference) methods.
### Using the Entry Point Functions for Command Line

llm-analysis provides two entry functions, [train](https://llm-analysis.readthedocs.io/en/latest/analysis.html#llm_analysis.analysis.train) and [infer](https://llm-analysis.readthedocs.io/en/latest/analysis.html#llm_analysis.analysis.infer), for ease of use through the command line interface. Run

```sh
python -m llm_analysis.analysis train --help
```
 or
```sh
python -m llm_analysis.analysis infer --help
```
to check the options.

`train` and `infer` use the pre-defined name-to-configuration mappings (`model_configs`, `gpu_configs`, `dtype_configs`) and other user-input arguments to construct the `LLMAnalysis` and do the query.
The pre-defined mappings are populated at the runtime from the model, GPU, and data type configuration `json` files under [model_configs](llm_analysis/model_configs), [gpu_configs](llm_analysis/gpu_configs), and [dtype_configs](llm_analysis/dtype_configs). To add a new model, GPU or data type to the mapping for query, just add a `json` description file to the corresponding folder.

llm-analysis also supports retrieving `ModelConfig` from Hugging Face with the model name (thus no need to add the model configuration to the `model_configs` folder). E.g. use [`EleutherAI/gpt-neox-20b`](https://huggingface.co/EleutherAI/gpt-neox-20b) as `model_name` when calling the `train` or `infer` entry functions.

A list of handy commands is provided to query against the pre-defined mappings as well as Hugging Face, or to dump configurations, for example,

```sh
python -m llm_analysis.config get_model_config_by_name decapoda-research/llama-7b-hf

```
gets the `ModelConfig` from the populated mapping by name, if not found, tries to get it from HuggingFace.

Run ```python -m llm_analysis.config --help``` for details.

### How to Set FLOPS and Memory Efficiency

Setting flops and memory efficiency to `1` (default) gives the lower bound of training or inference latency, as it assumes the peak hardware perforamnce (which is never the case).
A close-to-reality flops or memory efficiency can be found by benchmarking and profiling using the input dimensions in the model (providing such scripts is on the [TODOs](#todos-stay-tuned-radio) list).

If one has to make assumptions, for flops efficiency, literature reports up to `0.5` for large scale model training, and up to `0.7` for inference; `0.9` can be an aggressive target for memory efficiencies.

## Current Scope (expanding) and Limitations

- tp, pp, and sp assume using [`Megatron-LM`](https://github.com/NVIDIA/Megatron-LM) for training and [`FasterTransformer`](https://github.com/NVIDIA/FasterTransformer) for inference, and the dp assumes using [`DeepSpeed ZeRO`](https://github.com/microsoft/DeepSpeed)
- the parallelism strategy used in llm-analysis follows [Megatron-Turing NLG 530B](https://arxiv.org/abs/2201.11990) where tp is used within a node; then pp is used if the model is still too large to fit in GPU memory; then extra GPUs are used for dp
- supporting both full and selective activation recomputation, as described in [Reducing Activation Recomputation
in Large Transformer Models](https://arxiv.org/abs/2205.05198)
- tp communication is calculated as using `ring allreduce`; pp and dp communications across nodes are ignored for now
- data types are expressed with the number of bits, only `16`, `8`, and `4` bits data types are modeled for now.
- pre-training and fine-tuning are modeled the same (controlled by `total_num_tokens` passed to the `train` entry function), thus only full (all model parameters) fine-tuning is supported for now
- training assumes using the [Adam optimizer](https://arxiv.org/abs/1412.6980)
- training time only counts forward and backward for now
- inference assumes perfect overlapping of compute and memory operations

Check the TODOs below for what's next and stay tuned :radio:!

## [TODOs](#todos) (stay tuned :radio:)

The following features/improvements are on the roadmap. Stay tuned, and any contributions are welcome! :smile:

- [ ] Add more model case studies
- [ ] Add scripts to benchmark and profile the latency, FLOPS and memory efficiency in real workloads
- [ ] Improve the analytical model for inference
- [ ] Support efficient fine-tuning methods such as [LoRA](https://github.com/microsoft/LoRA) or [Adapters](https://arxiv.org/abs/2303.16199)
- [ ] Support other optimizers in training analysis
- [ ] Add pp (across nodes) and dp (across and within a node) communications analysis
- [ ] Add configuration/optimization advising
- [ ] An interactive web UI with data visualization
- [ ] Support other model architectures beyond encoders and decoders
- [ ] Support CPU offloading (weight, kv cache, etc.) analysis in training and inference
- [ ] Support sparse model inference
- [ ] Support CPU for inference analysis
- [ ] ...

## Citation

If you use llm-analysis in your work, please cite:

```
Cheng Li. (2023). LLM-Analysis: Latency and Memory Analysis of Transformer Models for Training and Inference. GitHub repository, https://github.com/cli99/llm-analysis.
```
or
```
@misc{llm-analysis-chengli,
  author = {Cheng Li},
  title = {LLM-Analysis: Latency and Memory Analysis of Transformer Models for Training and Inference},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cli99/llm-analysis}},
}
```
## Contributing
Contributions and suggestions are welcome.

llm-analysis uses [pre-commit](https://pre-commit.com/) to ensure code formatting is consistent. For pull requests with code contribution, please install the pre-commit (`pip install pre-commit`) as well as the used hooks (`pip install` in the repo), and format the code (runs automatically before each git commit) before submitting the PR.

## Useful Links

1. [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
2. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054v3)
3. [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
4. [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](https://arxiv.org/abs/2201.11990)
5. [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
6. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
7. [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)
8. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
9. [Understanding INT4 Quantization for Transformer Models: Latency Speedup, Composability, and Failure Cases](https://arxiv.org/abs/2301.12017)
10. [A Comprehensive Study on Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2303.08302)
