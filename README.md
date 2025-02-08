# FDA-Opt

```bash
conda create -n fda-opt python=3.12
conda activate fda-opt
```

```bash
pip install torch torchvision torchaudio datasets transformers pandas evaluate scikit-learn scipy matplotlib
```

```bash
bash paper-experiments.sh
```

```bash
python -m simulator --device_limits cuda:0=3 cuda:1=3
```

go on the notebooks/paper-results.ipynb