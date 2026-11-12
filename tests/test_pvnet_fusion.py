import torch

from rocast.models.pvnet_romania import CalibratedPinballLoss, PVNetRomaniaFusion

B, T = 2, 192


def _inputs():
    g = torch.Generator().manual_seed(0)
    return dict(
        sat=torch.randn(B, 5, 8, 24, 40, generator=g),
        hist=torch.randn(B, 96, 6, generator=g),
        e2=torch.randn(B, T, 16, 24, 40, generator=g),
        exo=torch.randn(B, T, 12, generator=g),
    )


def _model(cap=3550.0):
    return PVNetRomaniaFusion(f_hist=6, f_exo=12, d_model=16, capacity_ceiling_mw=cap)


def test_shape_and_monotonicity():
    out = _model()(**_inputs())
    assert out.shape == (B, T, 5)
    assert (out[..., 1:] >= out[..., :-1]).all()


def test_capacity_clamp():
    m = _model(cap=1.0)
    torch.nn.init.constant_(m.out.bias, 5.0)
    out = m(**_inputs())
    assert out.max() <= 1.0 and out.min() >= 0.0


def test_night_zero_for_solar():
    elev = torch.ones(B, T)
    elev[:, :50] = -10.0
    out = _model()(**_inputs(), solar_elevation=elev)
    assert (out[:, :50] == 0.0).all()
    assert (out[:, 50:] > 0.0).any()


def test_pinball_loss_and_grad():
    loss_fn = CalibratedPinballLoss()
    y = torch.ones(B, T)
    pred = torch.zeros(B, T, 5, requires_grad=True)
    loss = loss_fn(pred, y)
    # y - yhat = 1 -> loss = mean(q) = 0.5
    assert abs(loss.item() - 0.5) < 1e-6
    loss.backward()
    assert pred.grad is not None
