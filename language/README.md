## Environment Setup
1. Create and activate conda environment
```bash
conda create -n instant python=3.8
conda activate instant
```
2. Install Pytorch 1.13.1 with CUDA 11.6:
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

3. Install required library
```bash
pip install transformers datasets evaluate scikit-learn accelerate==1.0.1
```

## Train
Run `python train.py <args>` in your terminal.

Some common args:
- `--var`: The energy threshold use
- `--over_sampling`: The number of over sampling rank
- `--model_name`: Model used for fine-tuning

## Supported Configurations

1. Datasets

The project supports various tasks from the **GLUE Benchmark**. Specify your choice using the `--dataset` flag.

| Dataset ID                  | Task Type                              |
|----------------------------|-----------------------------------------|
| `cola`, `sst2`             | Single-Sentence Tasks                   |
| `mrpc`,      | Similarity and Paraphrase Tasks         |
| `mnli`, `qnli`, `rte` | Natural Language Inference Tasks     |

2. Models

You can utilize any sequence classification model from the **Hugging Face Hub** via the `--model_name` argument.

Tested models include:

- `bert-base-uncased` *(Default)*
- `distilbert-base-uncased`

Training example:
```bash
bash train_example.sh
```
