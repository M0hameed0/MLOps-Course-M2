---
library_name: transformers
license: apache-2.0
base_model: distilbert-base-uncased
tags:
- generated_from_trainer
model-index:
- name: imdb-distilbert-sentiment
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# imdb-distilbert-sentiment

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on an unknown dataset.
It achieves the following results on the evaluation set:
- eval_loss: 0.6947
- eval_model_preparation_time: 0.0
- eval_accuracy: 0.5067
- eval_f1: 0.6621
- eval_runtime: 85.1474
- eval_samples_per_second: 3.523
- eval_steps_per_second: 0.223
- step: 0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 2

### Framework versions

- Transformers 4.56.0
- Pytorch 2.8.0+cpu
- Datasets 4.0.0
- Tokenizers 0.22.0
