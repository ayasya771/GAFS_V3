import pytest
import torch

from gafs.models.critic import Critic
from gafs.models.generator_tft import TFTGenerator
from gafs.models.layers import GRN, VariableSelection

B, K, H, F, A, C, Z = 4, 30, 10, 6, 3, 3, 8


@pytest.fixture(scope="module")
def generator():
    torch.manual_seed(0)
    return TFTGenerator(
        n_features=F, n_assets=A, cond_dim=C, lookback=K, horizon=H,
        hidden=32, heads=4, z_dim=Z, dropout=0.0,
    )


def test_grn_shapes_and_context():
    grn = GRN(16, 32, output_size=24, context_size=8, dropout=0.0)
    x = torch.randn(B, 5, 16)
    c = torch.randn(B, 8)
    out = grn(x, c)
    assert out.shape == (B, 5, 24)
    assert torch.isfinite(out).all()


def test_vsn_weights_sum_to_one():
    vsn = VariableSelection(F, 16, dropout=0.0)
    x = torch.randn(B, K, F)
    fused, weights = vsn(x)
    assert fused.shape == (B, K, 16)
    assert weights.shape == (B, K, F)
    assert torch.allclose(weights.sum(-1), torch.ones(B, K), atol=1e-5)


def test_generator_output_shape_and_stochasticity(generator):
    x = torch.randn(B, K, F)
    c = torch.randn(B, C)
    y1 = generator(x, c)
    y2 = generator(x, c)
    assert y1.shape == (B, H, A)
    assert torch.isfinite(y1).all()
    assert not torch.allclose(y1, y2)
    generator.eval()
    z = generator.sample_noise(B)
    assert torch.allclose(generator(x, c, z), generator(x, c, z))
    generator.train()


def test_generator_uses_conditioning(generator):
    generator.eval()
    x = torch.randn(1, K, F)
    z = generator.sample_noise(1)
    c1 = torch.zeros(1, C)
    c2 = torch.full((1, C), 3.0)
    assert not torch.allclose(generator(x, c1, z), generator(x, c2, z))
    generator.train()


def test_generator_causal_attention_mask(generator):
    mask = generator.attn_mask
    assert mask.shape == (H, K + H)
    assert not mask[:, :K].any()
    assert not mask[0, K]
    assert mask[0, K + 1 :].all()


def test_generator_sample_batches(generator):
    x = torch.randn(1, K, F)
    c = torch.randn(1, C)
    paths = generator.sample(x, c, n_samples=10, batch_size=4)
    assert paths.shape == (10, H, A)


@pytest.mark.parametrize("arch", ["resnet", "transformer"])
def test_critic_score_and_features(arch):
    torch.manual_seed(0)
    critic = Critic(
        n_assets=A, n_features=F, cond_dim=C, horizon=H,
        hidden=32, arch=arch, channels=(16, 32), ctx_channels=8, proj_dim=16,
        heads=4, dropout=0.0,
    )
    y = torch.randn(B, H, A)
    x = torch.randn(B, K, F)
    c = torch.randn(B, C)
    score, feat = critic(y, x, c, return_features=True)
    assert score.shape == (B,)
    assert feat.shape == (B, critic.feature_dim)
    z = critic.project(feat)
    assert z.shape == (B, 16)
    assert torch.isfinite(score).all() and torch.isfinite(z).all()


def test_critic_conditioning_matters():
    torch.manual_seed(0)
    critic = Critic(
        n_assets=A, n_features=F, cond_dim=C, horizon=H,
        hidden=32, arch="resnet", channels=(16, 32), ctx_channels=8, proj_dim=16,
        heads=4, dropout=0.0,
    ).eval()
    y = torch.randn(B, H, A)
    x = torch.randn(B, K, F)
    s1 = critic(y, x, torch.zeros(B, C))
    s2 = critic(y, x, torch.full((B, C), 2.0))
    assert not torch.allclose(s1, s2)
