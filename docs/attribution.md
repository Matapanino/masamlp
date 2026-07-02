# Attribution

masaMLP implements published architectures; the value it adds is the
extensibility layer (sample weights, custom objectives/metrics, early
stopping on any metric), not the model designs.

## Architectures

- **TabularResNet** — Gorishniy, Rubachev, Khrulkov, Babenko, *Revisiting
  Deep Learning Models for Tabular Data*, NeurIPS 2021
  (arXiv:2106.11959). Reference implementation: `rtdl_revisiting_models`
  (MIT).
- **Numeric embeddings (PLR / periodic)** — Gorishniy, Rubachev, Babenko,
  *On Embeddings for Numerical Features in Tabular Deep Learning*, NeurIPS
  2022 (arXiv:2203.05556). Reference: `rtdl_num_embeddings` (MIT).
- **DANet** — Chen, Liao, Chen, Wang, Wu, *DANets: Deep Abstract Networks
  for Tabular Data Classification and Regression*, AAAI 2022
  (arXiv:2112.02962). Clean-room reimplementation following the MIT-licensed
  official repository (github.com/WhatAShot/DANet).
- **RealMLP** — Holzmüller, Ludwig, Klein, *Better by Default: Strong
  Pre-Tuned MLPs and Boosted Trees on Tabular Data*, NeurIPS 2024
  (arXiv:2407.04491). The `realmlp` model and its preprocessing/training
  recipe follow the author's MIT-licensed standalone reference
  (dholzmueller/realmlp-td-s_standalone); the PBLD embedding variant follows
  the Apache-2.0 pytabkit implementation, rewritten in-house.
- **TabR** — Gorishniy, Rubachev, Kartashev, Shlenskii, Kotelnikov, Babenko,
  *TabR: Unlocking the Power of Retrieval-Augmented Tabular Deep Learning*,
  ICLR 2024 (arXiv:2307.14338). Clean-room reimplementation following the
  MIT-licensed official repository (yandex-research/tabular-dl-tabr), using
  torch-native search instead of faiss.
- **CfC cell (used by TabularLNN)** — Hasani, Lechner, Amini, et al.,
  *Closed-form Continuous-time Neural Networks*, Nature Machine
  Intelligence 2022. Reference implementation: `ncps` (Apache-2.0),
  reimplemented in-house. The static-tabular adaptation is ours and
  experimental — see lnn.md.
- **Ghost Batch Normalization** — Hoffer, Hubara, Soudry, *Train longer,
  generalize better*, NeurIPS 2017 (arXiv:1705.08741).
- **sparsemax** — Martins, Astudillo (ICML 2016); **1.5-entmax** — Peters,
  Niculae, Martins (ACL 2019). Both implemented in-house from the exact
  sorting-based algorithms.

## Libraries that informed the design

- **pytabkit** (Apache-2.0) — RealMLP/TabM and the benchmark-grade training
  recipes; masaMLP intentionally complements it with an extensibility-first
  API rather than competing on model zoo breadth.
- **repleafgbm** (MIT, same author) — the estimator surface, metric registry
  (`get_metric`/`make_metric`), and repository conventions are shared with
  it; masaMLP is its neural sibling.
- **LightGBM / pytorch-tabnet** — the `fit(..., eval_set=...)` +
  `evals_result_` interaction model.
