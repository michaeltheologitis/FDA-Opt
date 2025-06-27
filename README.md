# FDA-Opt: Communication-Efficient Federated Fine-Tuning of Language Models

## Abstract
Federated Learning (FL) enables the utilization of vast, previously inaccessible data sources. At the same time, pre-trained Language Models (LMs) have taken the world by storm and for good reason. They exhibit remarkable emergent abilities and are readily adapted to downstream tasks. This opens one of the most exciting frontiers in FL: fine-tuning LMs. Yet, a persistent challenge in FL is the frequent, rigid communication of parameters -- a problem magnified by the sheer size of these contemporary models. The FedOpt family of algorithms has become the go-to approach for FL, relying on fixed but arbitrary intervals for model exchanges. Recently, the FDA algorithm prescribed a dynamic approach by monitoring the training progress. However, it introduced a hard-to-calibrate parameter and imposed a rigid synchronization scheme. In this work, we address these limitations by proposing the FDA-Opt family of algorithms -- a unified generalization of both FDA and FedOpt. Our experimental evaluation focuses on fine-tuning LMs on downstream NLP tasks and demonstrates that FDA-Opt outperforms FedOpt even when it is configured with hyper-parameters specifically optimized for the latter. In other words, we show that FDA-Opt is a practical, drop-in replacement for FedOpt in modern FL libraries and systems: it requires no additional configuration and delivers superior performance out of the box.

## Cite

If you find our work useful:
```bibtex
@misc{theologitis2025communication,
    title={FDA-Opt: Communication-Efficient Federated Fine-Tuning of Language Models},
    author={Michail Theologitis and Vasilis Samoladas and Antonios Deligiannakis},
    year={2025},
    eprint={2505.04535},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

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
