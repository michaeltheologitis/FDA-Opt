# Communication-Efficient Federated Learning of Language Models with Generalized Federated Dynamic Averaging

## Abstract
Federated Learning (FL) enables the utilization of vast, previously
inaccessible data sources. At the same time, pre-traned Language Models (LMs) have taken the world by storm and for good reason. They exhibit remarkable emergent abilities and are readily adapted to downstream tasks. This opens one of the most exciting frontiers in FL: fine-tuning LMs. Yet, a persistent challenge in FL is the frequent, rigid communication of parameters---a problem magnified by the sheer size of these modern models. The **FedOpt** family of algorithms has become the go-to approach for FL, relying on fixed but arbitrary intervals for model exchanges. Recently, the **FDA** algorithm prescribed a dynamic approach by monitoring the training progress. However, it introduced a hard-to-calibrate parameter and imposed a rigid synchronization scheme. In this work, we address these limitations by proposing the **FDA-Opt** family of algorithms---a unified generalization of both **FDA** and **FedOpt**. **FDA-Opt** eliminates the need for tuning additional parameters with a novel dynamic scheme, incorporates accelerated optimizers, and alleviates the original synchronization bottleneck. Our experimental evaluation demonstrates that **FDA-Opt** outperforms **FedOpt** even when it is configured with hyper-parameters specifically optimized for the latter. In other words, we show that **FDA-Opt** can seamlessly replace **FedOpt**.

## Enviroment
Requires python version 3.12.
```bash
pip install torch torchvision torchaudio datasets transformers pandas evaluate scikit-learn scipy matplotlib sentencepiece protobuf
```

## Paper Experiments

### Create the 510 unique experiments
```bash
bash paper-experiments.sh
```
### Run Locally

Here, we utilize two GPUs (cuda:0 and cuda:1), each running up to *three* concurrent experiments. This choice is subject to available GPU RAM.
```bash
python -m simulator --device_limits cuda:0=3 cuda:1=3
```

You can monitor the stderr and stdout at **results/output/**.

### Visualize results

Go to **notebooks** and run **paper-results.ipynb**.
