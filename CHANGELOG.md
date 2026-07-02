# Changelog

## 0.1.0 (unreleased)

Initial release.

- `MasaRegressor` / `MasaClassifier` sklearn-compatible estimators with
  `fit(X, y, sample_weight=..., eval_set=...)`, early stopping on any metric,
  and directory-format save/load.
- Models: `resnet` and `ft_transformer` (Gorishniy et al. 2021), `realmlp`
  (Holzmüller et al. 2024, TD-S architecture with the full training recipe
  in `masamlp.realmlp_params`), `tab_transformer` (Huang et al. 2020),
  `danet` (Chen et al. AAAI 2022), `tabr` (retrieval-augmented, Gorishniy
  et al. 2023), `modernnca` (Ye et al. 2024, soft-nearest-neighbor),
  `gandalf` (Joseph & Raj 2022, GFLU with t-softmax feature masks),
  `grn` (stacked TFT Gated Residual Networks), and `lnn` (experimental
  CfC-based liquid network for static tabular data), plus a
  `register_model` hook for custom architectures (token-based models via
  `embedding_kind = "tokens"`).
- `n_ens` seed ensembling on both estimators (pytabkit semantics: members
  seeded `random_state + i`, predictions averaged on the transformed scale);
  save/load stores all members.
- RealMLP insights as composable estimator options: `numeric_scaler="rssc"`,
  `cat_encoding="onehot"`, numeric embedding zoo
  (`num_embedding="pbld"/"plr"/"pl"/"periodic"`), learnable input scaling
  (`num_scaling`), `lr_scheduler="coslog4"` with per-group learning-rate
  factors, `optimizer_betas`, and regression `clip_predictions`.
- Objective plugin system: per-sample torch losses with a uniform
  sample-weight contract; built-ins for regression (squared error, MAE,
  Huber, quantile, Poisson) and classification (binary logistic, multiclass
  softmax, both with label smoothing).
- Metric plugin system ported from repleafgbm (`get_metric` / `make_metric`).
- Built-in preprocessing: quantile/standard/robust numeric scaling, median
  imputation, categorical index encoding with embeddings.
- Device support: CPU, CUDA (bf16 AMP, optional `torch.compile`), and MPS,
  behind `device="auto"`.
