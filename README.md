# FDA-Opt

```bash
python -m fdaopt.miscellaneous.create_hyperparameters --checkpoint roberta-base --ds_path glue --ds_name mrpc --num_labels 2 --num_clients 100 --clients_per_round 10 --alpha 1. --batch_size 8 --local_epochs 1 --total_rounds 1000 --server_opt_name Adam --client_opt_name SGD --server_opt_args lr=0.001 betas='(0.9, 0.99)'
```

```bash
python -m fdaopt.miscellaneous.create_hyperparameters --server_opt_name AdamW --client_opt_name SGD --server_opt_args lr=0.0001 weight_decay=0.001 --client_opt_args lr=0.001
```