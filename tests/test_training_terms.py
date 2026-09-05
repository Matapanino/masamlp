"""WP2 regression gates: arithmetic first, then trainer/state integration."""

import pytest
import torch

from masamlp.core.objectives import SquaredError
from masamlp.core.trainer import weighted_loss


@pytest.mark.parametrize("weight, expected", [
    ([[1., 1.], [1., 3.]], 9.0),
    ([[1., 0.], [1., 0.]], 2.5),
    ([[0., 0.], [0., 0.]], 0.),
])
def test_independent_member_normalization(weight, expected):
    raw = torch.tensor([[[1.], [2.]], [[3.], [4.]]], requires_grad=True)
    y = torch.zeros_like(raw)
    got = weighted_loss(SquaredError(), y, raw, torch.tensor(weight), member_batches=True)
    torch.testing.assert_close(got, torch.tensor(expected))
    got.backward()
    assert torch.isfinite(raw.grad).all()
    if expected == 0:
        assert torch.count_nonzero(raw.grad) == 0


def test_auxiliary_and_fixed_regularizer_gradient():
    from masamlp.core.training_terms import ModelRegularizer, RowTerm, TrainingTerms

    p = torch.tensor(2., requires_grad=True)
    raw = p.expand(2, 1)
    terms = TrainingTerms(
        auxiliary=(RowTerm(p * torch.tensor([1., 3.]), coefficient=0.5),),
        regularizers=(ModelRegularizer(p.square(), normalizer=4., coefficient=3.),),
    )
    loss = weighted_loss(SquaredError(), torch.zeros_like(raw), raw,
                         torch.tensor([1., 3.]), terms=terms)
    # p² + .5 * p * (1 + 9)/4 + 3*p²/4 = 9.5; derivative = 8.25.
    torch.testing.assert_close(loss, torch.tensor(9.5))
    loss.backward()
    torch.testing.assert_close(p.grad, torch.tensor(8.25))


@pytest.mark.parametrize("n, scale", [(2, 1.), (7, 19.)])
def test_regularizer_independent_of_batch_size_and_weight_scale(n, scale):
    from masamlp.core.training_terms import ModelRegularizer, TrainingTerms

    p = torch.tensor(2., requires_grad=True)
    raw = p.expand(n, 1)
    loss = weighted_loss(SquaredError(), raw.detach(), raw, torch.ones(n) * scale,
                         terms=TrainingTerms(regularizers=(
                             ModelRegularizer(p.square(), normalizer=4., coefficient=3.),)))
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(3.))
    torch.testing.assert_close(p.grad, torch.tensor(3.))


def test_auxiliary_member_alignment_and_zero_weights():
    from masamlp.core.training_terms import RowTerm, TrainingTerms

    raw = torch.zeros(2, 2, 1, requires_grad=True)
    aux = torch.tensor([[1., 100.], [3., 200.]], requires_grad=True)
    loss = weighted_loss(SquaredError(), raw.detach(), raw,
                         torch.tensor([[1., 0.], [3., 0.]]), member_batches=True,
                         terms=TrainingTerms(auxiliary=(RowTerm(aux),)))
    torch.testing.assert_close(loss, torch.tensor(1.25))
    loss.backward()
    torch.testing.assert_close(aux.grad, torch.tensor([[.125, 0.], [.375, 0.]]))


def test_terms_validate_shapes_and_normalizers():
    from masamlp.core.training_terms import ModelRegularizer, RowTerm, TrainingTerms

    raw = torch.zeros(2, 1)
    with pytest.raises(ValueError, match="shape"):
        weighted_loss(SquaredError(), raw, raw, None,
                      terms=TrainingTerms(auxiliary=(RowTerm(torch.zeros(2, 1)),)))
    with pytest.raises(ValueError, match="scalar"):
        ModelRegularizer(torch.zeros(2), normalizer=1.)
    for bad in [0., -1., float("inf"), float("nan")]:
        with pytest.raises(ValueError, match="normalizer"):
            ModelRegularizer(torch.tensor(1.), normalizer=bad)
    with pytest.raises(ValueError, match="coefficient"):
        RowTerm(torch.ones(2), coefficient=float("nan"))


class TinyModel(torch.nn.Module):
    def __init__(self, embedding=None, out_dim=1):
        super().__init__()
        self.output_layer = torch.nn.Linear(2, out_dim)
        self.aux_head = torch.nn.Linear(2, 1)
        self.register_buffer("penalty_normalizer", torch.tensor(4.))

    def forward(self, x_num, x_cat):
        return self.output_layer(x_num)


class TermsModel(TinyModel):
    coefficient = 0.5

    def training_terms(self, batch, raw):
        from masamlp.core.training_terms import ModelRegularizer, RowTerm, TrainingTerms

        if not self.training:
            raise AssertionError("training_terms must never run at prediction")
        return TrainingTerms(
            auxiliary=(RowTerm((self.aux_head(batch.x_num).squeeze(-1)
                                - batch.x_num[..., 0]).square(), self.coefficient),),
            regularizers=(ModelRegularizer(self.output_layer.weight.square().sum(),
                                           normalizer=4., coefficient=self.coefficient),),
        )


def _fit(model, *, epochs=3, weight=None, eval_sets=(), metrics=(), **kwargs):
    from masamlp.core.trainer import Trainer, TrainerConfig
    from masamlp.data.dataset import TabularData

    x = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 16
    data = TabularData(x, torch.empty(8, 0, dtype=torch.long), x[:, :1], weight)
    result = Trainer().fit(model, SquaredError(), data, list(eval_sets), list(metrics),
                           TrainerConfig(n_epochs=epochs, batch_size=4, device="cpu",
                                         amp=False, random_state=7, **kwargs))
    return result, data


def test_zero_terms_exact_training_identity():
    import copy

    baseline = TinyModel()
    candidate = TermsModel()
    candidate.load_state_dict(copy.deepcopy(baseline.state_dict()))
    candidate.coefficient = 0.
    _fit(baseline)
    _fit(candidate)
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[key], value, rtol=0, atol=0)


def test_all_zero_batch_skips_regularizer_and_optimizer():
    import copy

    model = TermsModel()
    before = copy.deepcopy(model.state_dict())
    _fit(model, weight=torch.zeros(8), weight_decay=0.5)
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


def test_nonzero_auxiliary_updates_head():
    model = TermsModel()
    before = model.aux_head.weight.detach().clone()
    _fit(model)
    assert not torch.equal(before, model.aux_head.weight)


def test_vectorized_terms_rejected():
    from masamlp.core.ensemble import check_vectorizable

    with pytest.raises(ValueError, match="training_terms.*loop"):
        check_vectorizable(TermsModel())


def test_checkpoint_and_save_load_equivalence(tmp_path, monkeypatch):
    import copy

    import numpy as np

    from masamlp import MasaRegressor, make_metric
    from masamlp.models import _MODEL_REGISTRY

    monkeypatch.setitem(_MODEL_REGISTRY, "wp2_test_terms", TermsModel)
    x = np.arange(32, dtype=np.float32).reshape(16, 2) / 32
    kw = dict(model="wp2_test_terms", numeric_scaler="none", target_standardize=False,
              random_state=7, device="cpu", amp=False, batch_size=8)
    # Constant metric makes epoch 0 best; later steps must be rolled back,
    # including the auxiliary head, which is outside forward's prediction path.
    model = MasaRegressor(**kw, n_epochs=5, early_stopping_rounds=1,
                          eval_metric=make_metric(lambda y, p: 1., name="constant", minimize=True))
    model.fit(x, x[:, 0], eval_set=[(x, x[:, 0])])
    one = MasaRegressor(**kw, n_epochs=1).fit(x, x[:, 0])
    assert model.best_iteration_ == 0
    for key, value in one.model_.state_dict().items():
        torch.testing.assert_close(model.model_.state_dict()[key], value, rtol=0, atol=0)
    before = copy.deepcopy(model.model_.state_dict())
    with pytest.warns(UserWarning, match="custom objects"):
        model.save_model(str(tmp_path / "model"))
    loaded = MasaRegressor.load_model(str(tmp_path / "model"))
    assert loaded.objective_ is None
    np.testing.assert_array_equal(model.predict(x), loaded.predict(x))
    for key, value in before.items():
        torch.testing.assert_close(loaded.model_.state_dict()[key], value, rtol=0, atol=0)


def test_documented_example_runs_and_loads(tmp_path):
    from pathlib import Path

    import numpy as np

    from masamlp.models import _MODEL_REGISTRY

    doc = Path(__file__).resolve().parents[1] / "docs" / "training-terms.md"
    source = doc.read_text().split("```python")[2].split("```")[0]
    namespace = {}
    try:
        exec(source, namespace)
        x = np.arange(24, dtype=np.float32).reshape(8, 3) / 24
        cls = namespace["MasaRegressor"]
        est = cls(model="joint_example", device="cpu", n_epochs=2, amp=False)
        est.fit(x, x[:, 1])
        est.save_model(str(tmp_path))
        loaded = cls.load_model(str(tmp_path))
        np.testing.assert_array_equal(est.predict(x), loaded.predict(x))
    finally:
        _MODEL_REGISTRY.pop("joint_example", None)


def test_independent_member_weight_rescaling_preserves_loss_and_gradient():
    raw = torch.tensor([[[1.], [2.]], [[3.], [4.]]], requires_grad=True)
    w = torch.tensor([[1., 1.], [1., 3.]])
    y = torch.zeros_like(raw)
    first = weighted_loss(SquaredError(), y, raw, w, member_batches=True)
    grad = torch.autograd.grad(first, raw)[0]
    second = weighted_loss(SquaredError(), y, raw, w * torch.tensor([1e-9, 17.]),
                           member_batches=True)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(grad, torch.autograd.grad(second, raw)[0])


def test_shared_member_auxiliary_uses_broadcast_sample_weights():
    from masamlp.core.training_terms import RowTerm, TrainingTerms

    raw = torch.zeros(2, 2, 1)
    aux = torch.tensor([[1., 4.], [9., 16.]])
    loss = weighted_loss(SquaredError(), torch.zeros(2, 1), raw, torch.tensor([1., 3.]),
                         terms=TrainingTerms(auxiliary=(RowTerm(aux),)))
    torch.testing.assert_close(loss, torch.tensor(10.))


def test_independent_member_trainer_aligns_hook_batch_indices():
    from masamlp.core.trainer import Trainer, TrainerConfig
    from masamlp.data.dataset import TabularData

    class IndexedTerms(TermsModel):
        wants_batch_indices = True
        supports_member_batches = True
        k = 2

        def training_terms(self, batch, raw):
            idx = self.current_batch_indices
            assert idx.shape == (4, 2)
            torch.testing.assert_close(batch.x_num[..., 0], idx.float() / 8)
            assert raw.shape == (4, 2, 1)
            return super().training_terms(batch, raw)

    x = torch.arange(8, dtype=torch.float32).unsqueeze(-1).expand(-1, 2) / 8
    data = TabularData(x, torch.empty(8, 0, dtype=torch.long), x[:, :1],
                       torch.tensor([0., 0., 1., 2., 3., 1., 1., 1.]))
    model = IndexedTerms()
    Trainer().fit(model, SquaredError(), data, [], [], TrainerConfig(
        n_epochs=2, batch_size=4, device="cpu", amp=False, random_state=7,
        share_training_batches=False,
    ))
    assert model.current_batch_indices is None
