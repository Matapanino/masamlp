"""RealMLP-TD fidelity knobs (0.9.0) — and the guarantee that turning none of
them on changes nothing.

The three fidelity levers ported from pytabkit's RealMLP-TD in 0.9.0 —
``init_mode``, ``weight_decay_mode``, and the configurable per-group lr
factors — all default to the 0.8.0 behaviour.  Two independent tests hold
that line:

* :func:`test_defaults_match_legacy_arguments` — same process, same seed:
  the plain default path and the path with every new argument spelled out at
  its legacy value must produce **bit-identical** parameters and predictions.
* :func:`test_default_init_matches_080_fixture` — the seeded initial
  parameters of a default build are compared, ``atol=0``, against tensors
  recorded from **0.8.0** (``tests/fixtures/realmlp_init_v080.npz``).  Init is
  pure ``torch.randn``/``torch.rand`` under a manual seed, so this fixture is
  stable across torch versions in a way a trained-weights fixture would not
  be; the end-to-end fit fixture below is therefore version-gated.

Regenerate the fixtures (only ever from a checkout whose behaviour is the
reference) with::

    python tests/test_realmlp_fidelity.py --regen
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from masamlp.classifier import MasaClassifier
from masamlp.models import build_model
from masamlp.presets import realmlp_params, realmlp_td_params
from masamlp.regressor import MasaRegressor

FIXTURES = Path(__file__).resolve().parent / "fixtures"
INIT_FIXTURE = FIXTURES / "realmlp_init_v080.npz"
FIT_FIXTURE = FIXTURES / "realmlp_fit_v080.npz"

# The two architectures the fixture pins: the plain default and the TD preset's.
_INIT_CASES = {
    "default": ({}, 4, [3], 1),
    "td": (
        {
            "num_scaling": True,
            "activation": "selu",
            "use_parametric_act": True,
            "act_lr_factor": 0.1,
            "dropout": 0.15,
            "dropout_schedule": "flat_cos",
            "plr_lr_factor": 0.1,
            "d_num_embedding": 4,
            "n_frequencies": 16,
            "sigma": 0.1,
            "cat_emb_dim": 8,
            "hidden_sizes": [64, 64],
        },
        5,
        [4, 3],
        2,
    ),
}


def _seeded_build(name: str):
    params, n_num, cards, out_dim = _INIT_CASES[name]
    torch.manual_seed(1234)
    return build_model(
        "realmlp", dict(params), n_num, cards, out_dim, "pbld" if name == "td" else None
    )


def _fit_case():
    """A tiny seeded end-to-end fit of both RealMLP presets."""
    rng = np.random.default_rng(0)
    xr = rng.normal(size=(240, 5))
    yr = 2.0 * xr[:, 0] - xr[:, 1] + rng.normal(0, 0.1, 240)
    xc = rng.normal(size=(240, 5))
    yc = (xc[:, 0] + xc[:, 1] > 0).astype(int)
    reg = MasaRegressor(
        **{**realmlp_params("regression"), "n_epochs": 4, "device": "cpu", "random_state": 0}
    ).fit(xr, yr)
    clf = MasaClassifier(
        **{
            **realmlp_td_params("classification"),
            "n_epochs": 4,
            "device": "cpu",
            "random_state": 0,
            "model_params": {
                **realmlp_td_params("classification")["model_params"],
                "hidden_sizes": [32, 32],
            },
        }
    ).fit(xc, yc)
    return {"reg": reg.predict(xr), "clf": clf.predict_proba(xc)[:, 1]}


# ---------------------------------------------------------------------------
# The no-change guarantee
# ---------------------------------------------------------------------------
def test_defaults_match_legacy_arguments():
    """Spelling every 0.9.0 fidelity argument out at its legacy value must be
    bit-for-bit the same model as not passing them at all."""
    legacy = {
        "init_mode": "ntp",
        "scale_lr_factor": 6.0,
        "first_layer_lr_factor": 1.0,
        "bias_lr_factor": 0.1,
    }
    torch.manual_seed(7)
    plain = build_model("realmlp", {"hidden_sizes": [16, 16], "num_scaling": True}, 4, [3], 1, None)
    torch.manual_seed(7)
    spelled = build_model(
        "realmlp",
        {"hidden_sizes": [16, 16], "num_scaling": True, **legacy},
        4,
        [3],
        1,
        None,
    )
    a = dict(plain.named_parameters())
    b = dict(spelled.named_parameters())
    assert a.keys() == b.keys()
    for name in a:
        assert torch.equal(a[name], b[name]), f"{name} differs"

    x_num, x_cat = torch.randn(8, 4), torch.randint(0, 3, (8, 1))
    plain.eval(), spelled.eval()
    with torch.no_grad():
        assert torch.equal(plain(x_num, x_cat), spelled(x_num, x_cat))

    groups_a = [(g["lr_factor"], g.get("wd_factor", 1.0), len(g["params"]))
                for g in plain.param_groups()]
    groups_b = [(g["lr_factor"], g.get("wd_factor", 1.0), len(g["params"]))
                for g in spelled.param_groups()]
    assert groups_a == groups_b


def test_weight_decay_mode_picks_the_optimizer():
    """``"auto"`` (the default) reproduces each optimizer's own convention;
    the explicit modes force one regardless of the optimizer's name."""
    from masamlp.core.trainer import TrainerConfig, _make_optimizer

    assert TrainerConfig().weight_decay_mode == "auto"
    assert MasaClassifier(model="realmlp").weight_decay_mode == "auto"

    groups = [{"params": [torch.nn.Parameter(torch.zeros(2))]}]
    cases = {
        ("adamw", "auto"): torch.optim.AdamW,
        ("adam", "auto"): torch.optim.Adam,
        ("adamw", "coupled"): torch.optim.Adam,
        ("adam", "decoupled"): torch.optim.AdamW,
    }
    for (name, mode), expected in cases.items():
        opt = _make_optimizer(name, [dict(g) for g in groups], 1e-3, 0.1, None, mode)
        assert type(opt) is expected, (name, mode)
    with pytest.raises(ValueError, match="weight_decay_mode"):
        _make_optimizer("adamw", [dict(g) for g in groups], 1e-3, 0.1, None, "nope")


def test_label_smoothing_schedule_scales_and_restores():
    """The schedule multiplies the objective's epsilon per step and hands the
    objective back at its original value."""
    from masamlp.core.objectives import BinaryLogistic
    from masamlp.core.trainer import _coslog4

    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, 4))
    y = (X[:, 0] > 0).astype(int)
    seen: list[float] = []

    class Recording(BinaryLogistic):
        def per_sample_loss(self, y_true, raw_pred):
            seen.append(self.label_smoothing)
            return super().per_sample_loss(y_true, raw_pred)

    obj = Recording(0.2)
    m = MasaClassifier(
        model="realmlp", model_params={"hidden_sizes": [16]}, objective=obj,
        n_epochs=3, batch_size=64, device="cpu", random_state=0,
        label_smoothing_schedule="coslog4",
    ).fit(X, y)
    assert m is not None
    assert seen[0] == pytest.approx(0.2 * _coslog4(0.0))       # first step at t=0
    assert max(seen) <= 0.2 + 1e-12 and max(seen) > 0.0
    assert obj.label_smoothing == 0.2                           # restored

    with pytest.raises(ValueError, match="label_smoothing_schedule"):
        MasaClassifier(
            model="realmlp", model_params={"hidden_sizes": [8]}, n_epochs=1,
            device="cpu", label_smoothing_schedule="nope",
        ).fit(X, y)


def test_drop_last_drops_the_short_batch():
    """With ``drop_last`` every epoch runs ``floor(n/batch_size)`` steps."""
    from masamlp.core.objectives import BinaryLogistic

    rng = np.random.default_rng(0)
    X = rng.normal(size=(250, 4))
    y = (X[:, 0] > 0).astype(int)

    def steps(drop_last: bool) -> int:
        count = [0]

        class Counting(BinaryLogistic):
            def per_sample_loss(self, y_true, raw_pred):
                count[0] += 1
                return super().per_sample_loss(y_true, raw_pred)

        MasaClassifier(
            model="realmlp", model_params={"hidden_sizes": [8]}, objective=Counting(),
            n_epochs=2, batch_size=100, device="cpu", random_state=0, drop_last=drop_last,
        ).fit(X, y)
        return count[0]

    assert steps(False) == 2 * 3      # 100 + 100 + 50
    assert steps(True) == 2 * 2       # the 50-row tail is dropped


def test_data_driven_init_gives_unit_std_preactivations():
    """``init_mode="std+he5"`` rescales every layer so its pre-activations
    have unit sample std on the init batch, and biases are non-zero."""
    torch.manual_seed(0)
    model = build_model(
        "realmlp",
        {"hidden_sizes": [32, 32], "num_scaling": True, "init_mode": "std+he5",
         "zero_init_output": False},
        4, [3], 1, None,
    )
    assert model.needs_data_init is True
    x_num = torch.randn(512, 4) * 7.0 + 3.0        # deliberately off-scale
    x_cat = torch.randint(0, 3, (512, 1))
    model.data_init(x_num, x_cat)
    model.eval()

    h = model.embedding(x_num, x_cat)
    first = model.trunk[0]
    pre = (h @ first.weight) / np.sqrt(first.in_features)
    np.testing.assert_allclose(
        pre.std(dim=0, correction=0).detach().numpy(), np.ones(32), rtol=1e-5
    )
    assert float(first.bias.detach().abs().max()) > 0.0
    # ...and the default mode leaves the model untouched.
    torch.manual_seed(0)
    plain = build_model("realmlp", {"hidden_sizes": [32, 32], "num_scaling": True},
                        4, [3], 1, None)
    assert plain.needs_data_init is False
    before = plain.trunk[0].weight.detach().clone()
    plain.data_init(x_num, x_cat)
    assert torch.equal(plain.trunk[0].weight, before)


def test_scale_position_first_layer_moves_the_gain():
    """``scale_position="first_layer"`` relocates the diagonal gain from the
    raw numerics to the first hidden layer's input (RealMLP-TD)."""
    torch.manual_seed(0)
    model = build_model(
        "realmlp",
        {"hidden_sizes": [16], "num_scaling": True, "scale_position": "first_layer",
         "d_num_embedding": 4, "n_frequencies": 8},
        5, [3], 1, "pbld",
    )
    assert model.embedding.scaling is None
    assert model.front_scale is not None
    assert model.front_scale.scale.numel() == model.embedding.d_out   # 5*4 + cat dim
    scale_group = next(g for g in model.param_groups() if g["lr_factor"] == 6.0)
    assert scale_group["params"][0] is model.front_scale.scale
    # It is a real forward-path parameter (read at the trunk: the output
    # layer is zero-initialized, so the logits are 0 either way).
    x_num, x_cat = torch.randn(8, 5), torch.randint(0, 3, (8, 1))
    model.eval()
    with torch.no_grad():
        def trunk_out():
            return model.trunk(model.front_scale(model.embedding(x_num, x_cat)))

        base = trunk_out().clone()
        model.front_scale.scale.mul_(0.0)
        assert not torch.equal(trunk_out(), base)


def test_onehot_max_categories_moves_the_split():
    """The estimator now exposes the hybrid encoder's cardinality threshold."""
    import pandas as pd

    from masamlp.data.preprocessing import TabularPreprocessor

    df = pd.DataFrame({"n": np.arange(60.0),
                       "c": [f"v{i % 12}" for i in range(60)]})
    low = TabularPreprocessor("rssc", cat_encoding="hybrid", onehot_max_categories=9).fit(df)
    high = TabularPreprocessor("rssc", cat_encoding="hybrid", onehot_max_categories=17).fit(df)
    assert low.onehot_pos_ == [] and low.embed_pos_ == [0]
    assert high.onehot_pos_ == [0] and high.embed_pos_ == []
    assert MasaClassifier(model="realmlp").onehot_max_categories == 9


def test_default_init_matches_080_fixture():
    """Seeded default init is byte-identical to what 0.8.0 produced."""
    ref = np.load(INIT_FIXTURE)
    for name in _INIT_CASES:
        model = _seeded_build(name)
        for pname, tensor in model.named_parameters():
            key = f"{name}.{pname}"
            assert key in ref, f"fixture has no {key} (architecture changed?)"
            np.testing.assert_array_equal(
                tensor.detach().numpy(), ref[key], err_msg=f"init changed for {key}"
            )
        extra = [k for k in ref.files
                 if k.startswith(f"{name}.")
                 and k[len(name) + 1:] not in dict(model.named_parameters())]
        assert not extra, f"fixture parameters vanished: {extra}"


@pytest.mark.skipif(
    not FIT_FIXTURE.exists(), reason="fit fixture not generated"
)
def test_fit_predictions_match_080_fixture():
    """A full tiny fit under both presets reproduces 0.8.0's predictions.

    Gated on the torch version the fixture was recorded with: matmul/reduction
    kernels are free to change between releases, so an inequality here across
    versions would be a torch fact, not a masaMLP regression.
    """
    ref = np.load(FIT_FIXTURE)
    if str(ref["torch_version"]) != torch.__version__:
        pytest.skip(
            f"fixture recorded on torch {ref['torch_version']}, running {torch.__version__}"
        )
    out = _fit_case()
    for key, value in out.items():
        np.testing.assert_array_equal(value, ref[key], err_msg=f"{key} predictions changed")


# ---------------------------------------------------------------------------
# Fixture generation
# ---------------------------------------------------------------------------
def _regen() -> None:
    FIXTURES.mkdir(exist_ok=True)
    init = {}
    for name in _INIT_CASES:
        model = _seeded_build(name)
        for pname, tensor in model.named_parameters():
            init[f"{name}.{pname}"] = tensor.detach().numpy()
    np.savez(INIT_FIXTURE, **init)
    fit = _fit_case()
    np.savez(FIT_FIXTURE, torch_version=np.array(torch.__version__), **fit)
    print(f"wrote {INIT_FIXTURE} ({len(init)} tensors) and {FIT_FIXTURE} "
          f"(torch {torch.__version__})")


if __name__ == "__main__":
    _regen()


def test_pytabkit_fidelity_preset():
    """``pytabkit_fidelity=True`` flips exactly the three structural choices
    (plus the two batching/encoding details) and nothing else."""
    base = realmlp_td_params("classification")
    fid = realmlp_td_params("classification", pytabkit_fidelity=True)
    assert base == realmlp_td_params("classification")          # not mutated
    changed = {k for k in set(base) | set(fid) if base.get(k) != fid.get(k)}
    assert changed == {"model_params", "onehot_max_categories", "drop_last"}
    mp_changed = {
        k for k in set(base["model_params"]) | set(fid["model_params"])
        if base["model_params"].get(k) != fid["model_params"].get(k)
    }
    assert mp_changed == {"init_mode", "zero_init_output", "scale_position"}
    assert fid["onehot_max_categories"] == 8 and fid["drop_last"] is True


def test_pytabkit_fidelity_preset_trains(clf_data):
    """The fidelity preset is a runnable recipe end to end, not just a dict."""
    X, y, Xv, yv = clf_data
    m = MasaClassifier(
        **{
            **realmlp_td_params("classification", pytabkit_fidelity=True),
            "model_params": {
                **realmlp_td_params("classification", pytabkit_fidelity=True)["model_params"],
                "hidden_sizes": [32, 32],
            },
            "n_epochs": 12, "batch_size": 64, "device": "cpu", "random_state": 0,
        }
    ).fit(X, y)
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(yv, m.predict_proba(Xv)[:, 1]) > 0.85
