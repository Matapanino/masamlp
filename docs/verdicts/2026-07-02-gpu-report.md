# masaMLP GPU verification — torch 2.11.0+cu128, cuda=True

## pytest (device / ensemble / realmlp)

```
.....s............................                                       [100%]

```

## CUDA smoke — torch 2.11.0+cu128, Tesla T4

| model | cuda acc | fit s |
|---|---|---|
| resnet | 0.997 | 2.5 |
| realmlp | 0.975 | 0.3 |
| ft_transformer | 0.980 | 3.0 |
| tab_transformer | 0.831 | 0.3 |
| danet | 0.961 | 185.7 |
| tabr | 0.996 | 4.2 |
| modernnca | 0.983 | 0.6 |
| gandalf | 0.912 | 1.0 |
| grn | 0.995 | 0.4 |
| lnn | 0.995 | 1.4 |

AMP auto (resnet reg): rmse=0.1845
save(cuda) -> load -> predict parity: True

## gpu_speed.py --rows 30000 --skip-cpu

```
torch 2.11.0+cu128  cuda=True (Tesla T4)
resnet                       device=cuda  amp=False fit     6.6s  rmse 0.2107
resnet                       device=cuda  amp=auto  fit     6.2s  rmse 0.2044
realmlp (TD-S recipe)        device=cuda  amp=False fit    12.2s  rmse 0.1854
realmlp (TD-S recipe)        device=cuda  amp=auto  fit    13.8s  rmse 0.1864
ft_transformer               device=cuda  amp=False fit    19.2s  rmse 0.2937
ft_transformer               device=cuda  amp=auto  fit    23.3s  rmse 0.2509
tabr                         device=cuda  amp=False fit    10.1s  rmse 0.1967
tabr                         device=cuda  amp=auto  fit    20.8s  rmse 0.1925

-- n_ens=8 (lnn): loop vs vectorized --
lnn n_ens=8 [loop]           device=cuda  amp=False fit    34.6s  rmse 0.1677
lnn n_ens=8 [vectorized]     device=cuda  amp=False fit     8.2s  rmse 0.1572

```

pytest exit code: 0