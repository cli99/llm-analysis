# Example Analyses with Llama 2 Models

[Llama 2](https://ai.meta.com/llama/) is a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from `7` billion to `70` billion parameters. The analyses below are performed with llm-analysis as a showcase.

- [Example Analyses with Llama 2 Models](#example-analyses-with-llama-2-models)
  - [Inference with Llama 2 70B](#inference-with-llama-2-70b)
  - [References](#references)

## Inference with Llama 2 70B

The Appendix of [Why GPT-3.5 is (mostly) cheaper than Llama 2](https://www.cursor.so/blog/llama-inference) reports measurements with  `Llama-2-70B` on 2 A100-80GB GPUs. The tables below show llm-anlaysis results using the experiment setups with different efficiency numbers.
The cost per GPU hour uses `$2.21`. For other details, check `run_infer_cursor.py`.

- `flops_efficiency=0.6 and hbm_memory_efficiency=0.6`

| Batch Size | Prompt Tokens | Completion Tokens | Time to first token (s) | Time for completion (s) | Tokens/second | Price/1k prompt tokens | Price /1k Completion tokens |
| ---------- | ------------- | ----------------- | ------------------------ | ----------------------- | ------------- | ---------------------- | --------------------------- |
| 1          | 128           | 242               | 0.055                    | 13.185                  | 18.277        | 0.000532               | 0.06689                     |
| 2          | 128           | 512               | 0.080                    | 28.025                  | 36.434        | 0.000386               | 0.03360                     |
| 4          | 128           | 512               | 0.158                    | 28.209                  | 72.197        | 0.000378               | 0.01691                     |
| 1          | 512           | 512               | 0.159                    | 27.986                  | 18.192        | 0.000381               | 0.06711                     |
| 2          | 512           | 512               | 0.318                    | 28.130                  | 35.995        | 0.000381               | 0.03373                     |
| 4          | 512           | 512               | 0.635                    | 28.420                  | 70.487        | 0.000381               | 0.01704                     |
| 1          | 512           | 304               | 0.159                    | 16.600                  | 18.140        | 0.000381               | 0.06704                     |
| 8          | 512           | 512               | 1.270                    | 28.999                  | 135.323       | 0.000381               | 0.00869                     |
| 16         | 512           | 512               | 2.539                    | 30.156                  | 250.556       | 0.000381               | 0.00452                     |
| 32         | 512           | 512               | 5.078                    | 32.472                  | 436.335       | 0.000380               | 0.00243                     |
| 1          | 1024          | 512               | 0.321                    | 28.056                  | 18.043        | 0.000385               | 0.06728                     |
| 2          | 1024          | 512               | 0.642                    | 28.271                  | 35.416        | 0.000385               | 0.03390                     |
| 4          | 1024          | 512               | 1.284                    | 28.701                  | 68.301        | 0.000385               | 0.01721                     |
| 8          | 1024          | 512               | 2.568                    | 29.560                  | 127.488       | 0.000385               | 0.00886                     |
| 16         | 1024          | 512               | 5.136                    | 31.280                  | 224.957       | 0.000385               | 0.00469                     |
| 1          | 3595          | 512               | 1.192                    | 28.408                  | 17.297        | 0.000407               | 0.06812                     |
| 2          | 3595          | 512               | 2.384                    | 28.976                  | 32.653        | 0.000407               | 0.03474                     |
| 4          | 3595          | 512               | 4.767                    | 30.111                  | 58.719        | 0.000407               | 0.01805                     |

- `flops_efficiency=0.7 and hbm_memory_efficiency=0.9`

| Batch Size | Prompt Tokens | Completion Tokens | Time to first token (s) | Time for completion (s) | Tokens/second | Price/1k prompt tokens | Price /1k Completion tokens |
| ---------- | ------------- | ----------------- | ------------------------ | ----------------------- | ------------- | ---------------------- | --------------------------- |
| 1          | 128           | 242               | 0.039                    | 8.894                   | 27.090        | 0.000374               | 0.04512                     |
| 2          | 128           | 512               | 0.068                    | 18.903                  | 53.976        | 0.000326               | 0.02267                     |
| 4          | 128           | 512               | 0.136                    | 19.026                  | 106.878       | 0.000325               | 0.01141                     |
| 1          | 512           | 512               | 0.137                    | 18.877                  | 26.928        | 0.000328               | 0.04527                     |
| 2          | 512           | 512               | 0.273                    | 18.974                  | 53.204        | 0.000327               | 0.02275                     |
| 4          | 512           | 512               | 0.546                    | 19.167                  | 103.892       | 0.000327               | 0.01149                     |
| 1          | 512           | 304               | 0.137                    | 11.197                  | 26.823        | 0.000328               | 0.04522                     |
| 8          | 512           | 512               | 1.092                    | 19.553                  | 198.402       | 0.000327               | 0.00586                     |
| 16         | 512           | 512               | 2.183                    | 20.326                  | 363.938       | 0.000327               | 0.00305                     |
| 32         | 512           | 512               | 4.366                    | 21.872                  | 624.440       | 0.000327               | 0.00164                     |
| 1          | 1024          | 512               | 0.276                    | 18.924                  | 26.666        | 0.000331               | 0.04538                     |
| 2          | 1024          | 512               | 0.552                    | 19.067                  | 52.193        | 0.000331               | 0.02286                     |
| 4          | 1024          | 512               | 1.104                    | 19.354                  | 100.106       | 0.000331               | 0.01160                     |
| 8          | 1024          | 512               | 2.208                    | 19.928                  | 185.037       | 0.000331               | 0.00597                     |
| 16         | 1024          | 512               | 4.416                    | 21.075                  | 321.363       | 0.000331               | 0.00316                     |
| 1          | 3595          | 512               | 1.025                    | 19.159                  | 25.367        | 0.000350               | 0.04594                     |
| 2          | 3595          | 512               | 2.049                    | 19.537                  | 47.437        | 0.000350               | 0.02343                     |
| 4          | 3595          | 512               | 4.098                    | 20.294                  | 83.961        | 0.000350               | 0.01217                     |


## References

- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
