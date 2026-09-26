| Component                       |                                                        Recommendation |
| ------------------------------- | --------------------------------------------------------------------: |
| **Parameters**                  |                                                    **~15–25 million** |
| Vocabulary                      |                                                      **8,000 tokens** |
| Context window                  |                                                        **256 tokens** |
| Embedding dimension (`d_model`) |                                                               **256** |
| Transformer blocks              |                                                                 **6** |
| Attention heads                 |                                                                 **8** |
| Head dimension                  |                                                                **32** |
| FFN hidden dimension            |                                                             **1,024** |
| Activation                      |                                                              **GELU** |
| Attention                       |                                  **Causal multi-head self-attention** |
| Positional encoding             |                             **RoPE** or learned positional embeddings |
| Normalization                   |                                                           **RMSNorm** |
| Output                          |                                            Linear → vocabulary logits |
| Training precision              |      **FP16 weights/activations + FP32 accumulation where practical** |
| Optimizer                       |                                                             **AdamW** |
| Batch size                      |                                                   **32–64 sequences** |
| Sequence length                 |                                                               **256** |
| Gradient accumulation           |                                                               **4–8** |
| Training examples               |                                              **~100k–500k sequences** |
| Training time                   | **Several hours to ~1 day**, depending heavily on implementation/data |
| Framework                       |                                     **CuPy only for GPU computation** |
| Data processing                 |                                                            **Polars** |
| Configuration/checkpoints       |                                                              **JSON** |
