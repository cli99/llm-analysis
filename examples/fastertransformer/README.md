# Example Analyses with FasterTransformer

[kipply's blog](https://kipp.ly/blog/transformer-inference-arithmetic/#comparing-against-real-benchmarks) posts some real benchmarks for a [13B parameter model](../../llm_analysis/model_configs/test-13b.json) with [FasterTransformer](https://github.com/NVIDIA/FasterTransformer).

For inference with `512` context length, batch size `1` and `10` generated tokens, the blog reports the empirical results below on `1 ` and `2` A100 40GB GPU:

| number of GPUs (tp_size) | prefill latency (ms) | decode latency (ms) | total latency (ms) |
|------------------------|----------------------|---------------------|--------------------|
| 1                      | 63.17                | 22.0                | 283.6              |
| 2                      | 45.45                | 13.5                | 180.86             |

## Inference Analysis

[run_infer.sh](run_infer.sh) reproduces the setup used in the blog post and the latency analyses are presented below.


Note that as the blog described, the highest efficiency numbers observed in profiling are taken for `flops_efficiency` and `hbm_memory_efficiency`; different parts of the model achieve different compuate or memory efficiency, e.g., the one of the MLP gets `0.72` for `flops_efficency` while the attention projections only achieves 0.54. Taking the lowest observed in profiling gives a `safer lower-bound` of model latency.

- 1 GPU (no tensor parallelism):

Latencies with ideal compute and memory efficiency (`--flops_efficiency 1 --hbm_memory_efficiency 1`), and with reported `flops_efficiency = 0.7` and `hbm_memory_efficiency = 0.9`(observed by profiling in the blog) are compared to the empirical results:

| latency (ms) | empirical | llm-analysis with ideal efficiency | llm-analysis with profiled efficiency | diff. between with profiled efficiency and empirical (%) |
|--------------|-----------|------------------------------------|---------------------------------------|----------------------------------------------------------|
| prefill      | 63.17     | 43.7                               | 62.15                                 | -1.61                                                    |
| decode       | 22.0      | 17.07                              | 18.97                                 | -13.8                                                    |
| total        | 283.6     | 214.41                             | 251.84                                | -11.2                                                    |

- 2 GPUs (`tp_size = 2`):

Latencies with ideal compute and memory efficiency (`--flops_efficiency 1 --hbm_memory_efficiency 1`), and with reported `flops_efficiency = 0.7` and `hbm_memory_efficiency = 0.87`(observed by profiling in the blog) are compared to the empirical results:

| latency (ms) | empirical | llm-analysis with ideal efficiency | llm-analysis with profiled efficiency | diff. between with profiled efficiency and empirical (%) |
|--------------|-----------|------------------------------------|---------------------------------------|----------------------------------------------------------|
| prefill      | 45.45     | 23.7                               | 32.98                                 | -27.4                                                    |
| decode       | 13.5      | 9.35                               | 10.32                                 | -23.5                                                    |
| total        | 180.86    | 117.2                              | 136.15                                | -24.7                                                    |

## References
- [Transformer Inference Arithmetic](https://kipp.ly/blog/transformer-inference-arithmetic/)
