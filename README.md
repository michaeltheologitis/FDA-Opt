# Communication-Efficient Federated Fine-Tuning of Language Models via Dynamic Update Schedules

## Abstract
Federated learning (FL) makes it possible to train models on data that would otherwise remain untapped and inaccessible. Simultaneously, pre-trained language models (LMs) have emerged as indispensable tools in modern workflows. These models exhibit extraordinary capabilities and are easily adapted to downstream tasks. This opens one of the most exciting frontiers in FL: fine-tuning LMs. However, a persistent challenge in FL is the frequent, rigid communication of parameters---a problem magnified by the sheer size of these modern models. Currently, the FedOpt family of algorithms is the prevailing approach in FL, though it relies on fixed, heuristic intervals for model synchronization. Recently, the FDA algorithm introduced a dynamic alternative by monitoring training progress, but it came with its own drawbacks---namely, a hard-to-tune threshold parameter and a rigid synchronization scheme. In this work, we introduce the FDA-Opt family of algorithms---a unified generalization that extends the principles behind both FDA and FedOpt, while resolving their core limitations. We evaluate our approach on fine-tuning LMs across a range of downstream NLP tasks, and demonstrate that it consistently outperforms FedOpt---even when FDA-Opt operates under hyper-parameter settings originally optimized for its competitors. In other words, we show that FDA-Opt is a practical, drop-in replacement for FedOpt in modern FL libraries and systems---it requires no additional configuration and delivers superior performance out of the box.

## Enviroment
Requires python version 3.12.
```bash
pip install torch torchvision torchaudio datasets transformers pandas evaluate scikit-learn scipy matplotlib sentencepiece protobuf
```

## Paper Experiments

### Create the unique experiments
```bash
bash paper-experiments.sh
```
### Run Locally

Here, we utilize two GPUs (cuda:0 and cuda:1), each running up to *three* concurrent experiments. This choice is subject to available GPU RAM.
```bash
python -m simulator --device_limits cuda:0=3 cuda:1=3
```

You can monitor the *stderr* and *stdout* at **results/output/**

### Visualize results

Go to **notebooks** and run **paper-results.ipynb**.

## Example: Run two Experiments

We will run **FedAdam** and **FDA-Adam** on **MRPC**. 

### Create the 2 unique experiments

We first create the experiment for **FedAdam** (Adam at the server and SGD at the client) with some hyper-parameters.

```bash
python -m fdaopt.miscellaneous.create_hyperparameters --total_rounds 100 --ds_name mrpc --num_clients 10 --clients_per_round 10 --server_opt_name Adam --client_opt_name SGD --server_opt_args lr=0.001 --client_opt_args lr=1e-05
```

We then create the experiment for **FDA-Adam** with the simple addition of the ``--fda`` flag.

```bash
python -m fdaopt.miscellaneous.create_hyperparameters --total_rounds 100 --ds_name mrpc --num_clients 10 --clients_per_round 10 --server_opt_name Adam --client_opt_name SGD --server_opt_args lr=0.001 --client_opt_args lr=1e-05 --fda
```

### Run Locally

Here, we utilize one GPU (cuda:0) running the *two* experiments concurrently.
```bash
python -m simulator --device_limits cuda:0=2
```

### Results

You can monitor the *stderr* and *stdout* at **results/output/**

The test information (metrics, etc.) will be saved on **results/round_metrics/**
