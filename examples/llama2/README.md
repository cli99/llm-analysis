# Example Analyses with Llama 2 Models

[Llama 2](https://ai.meta.com/llama/) is a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from `7` billion to `70` billion parameters. The analyses below are performed with llm-analysis as a showcase.

- [Example Analyses with Llama 2 Models](#example-analyses-with-llama-2-models)
  - [Inference with Llama 2 70B](#inference-with-llama-2-70b)
  - [References](#references)

## Inference with Llama 2 70B

The Appendix of [Why GPT-3.5 is (mostly) cheaper than Llama 2](https://www.cursor.so/blog/llama-inference) reports measurements with  `Llama-2-70B` on 2 A100-80GB GPUs. The tables below show llm-analysis results using the experiment setups with different efficiency numbers.
The cost per GPU hour uses `$2.21`. For other details, check `run_infer_cursor.py`.

- `flops_efficiency=0.7 and hbm_memory_efficiency=0.9`

| Batch Size | Prompt Tokens | Completion Tokens | Time to first token (s) | Time for completion (s) | Tokens/second | Price/1k prompt tokens | Price /1k Completion tokens |
| ---------- | ------------- | ----------------- | ----------------------- | ----------------------- | ------------- | ---------------------- | --------------------------- |
| 1          | 128           | 242               | 0.040                   | 9.425                   | 25.568        | 0.000383               | 0.0478                      |
| 2          | 128           | 512               | 0.062                   | 20.026                  | 50.977        | 0.000296               | 0.0240                      |
| 4          | 128           | 512               | 0.123                   | 20.148                  | 101.031       | 0.000295               | 0.0121                      |
| 1          | 512           | 512               | 0.124                   | 20.000                  | 25.442        | 0.000298               | 0.0480                      |
| 2          | 512           | 512               | 0.248                   | 20.096                  | 50.334        | 0.000297               | 0.0241                      |
| 4          | 512           | 512               | 0.496                   | 20.288                  | 98.537        | 0.000297               | 0.0122                      |
| 1          | 512           | 304               | 0.124                   | 11.864                  | 25.359        | 0.000298               | 0.0479                      |
| 8          | 512           | 512               | 0.991                   | 20.673                  | 189.069       | 0.000297               | 0.0062                      |
| 16         | 512           | 512               | 1.982                   | 21.442                  | 349.726       | 0.000297               | 0.0032                      |
| 32         | 512           | 512               | 3.963                   | 22.981                  | 608.077       | 0.000297               | 0.0017                      |
| 1          | 1024          | 512               | 0.251                   | 20.047                  | 25.224        | 0.000301               | 0.0481                      |
| 2          | 1024          | 512               | 0.502                   | 20.190                  | 49.488        | 0.000301               | 0.0242                      |
| 4          | 1024          | 512               | 1.004                   | 20.476                  | 95.348        | 0.000301               | 0.0123                      |
| 8          | 1024          | 512               | 2.007                   | 21.048                  | 177.666       | 0.000301               | 0.0063                      |
| 16         | 1024          | 512               | 4.014                   | 22.191                  | 312.615       | 0.000301               | 0.0033                      |
| 1          | 3595          | 512               | 0.936                   | 20.282                  | 24.130        | 0.000320               | 0.0486                      |
| 2          | 3595          | 512               | 1.872                   | 20.660                  | 45.446        | 0.000320               | 0.0248                      |
| 4          | 3595          | 512               | 3.745                   | 21.416                  | 81.398        | 0.000320               | 0.0128                      |

## References

- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
