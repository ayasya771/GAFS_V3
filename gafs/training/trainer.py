"""WGAN-GP + SimCLR training loop.

Per generator update the critic trains `n_critic` times on fresh batches with
the gradient penalty (lambda = 10) plus an NT-Xent term on its projection
head, computed from two augmented views of the real windows. The generator
then minimises -E[D(fake)] plus latent coverage terms (uniformity + MMD in
the critic's contrastive space), which forces samples to populate the
organised latent space instead of collapsing to a single safe path.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import TrainConfig
from ..models.critic import Critic, build_critic
from ..models.generator_tft import TFTGenerator
from ..utils import CSVLogger, ensure_dir
from .augmentations import augment
from .losses import (
    critic_wgan_loss,
    generator_wgan_loss,
    gradient_penalty,
    nt_xent,
    rbf_mmd,
    uniformity,
)


def _infinite(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


class GANTrainer:
    def __init__(
        self,
        generator: TFTGenerator,
        critic: Critic,
        cfg: TrainConfig,
        model_meta: dict,
        device: torch.device,
        out_dir: str | Path = "outputs/run",
    ):
        self.g = generator.to(device)
        self.d = critic.to(device)
        self.cfg = cfg
        self.model_meta = dict(model_meta)
        self.device = device
        self.out_dir = ensure_dir(out_dir)

        betas = (cfg.beta1, cfg.beta2)
        self.opt_g = torch.optim.Adam(self.g.parameters(), lr=cfg.lr_g, betas=betas)
        self.opt_d = torch.optim.Adam(self.d.parameters(), lr=cfg.lr_d, betas=betas)

        self.g_ema = copy.deepcopy(self.g).eval()
        for p in self.g_ema.parameters():
            p.requires_grad_(False)

        self.step = 0
        self.history: list[dict[str, float]] = []
        self.logger = CSVLogger(self.out_dir / "history.csv")

    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _critic_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        cfg = self.cfg
        x, cond, y = batch["x_hist"], batch["cond"], batch["y"]
        with torch.no_grad():
            y_fake = self.g(x, cond)

        d_real = self.d(y, x, cond)
        d_fake = self.d(y_fake, x, cond)
        adv = critic_wgan_loss(d_real, d_fake)
        gp = gradient_penalty(self.d, y, y_fake, x, cond)

        v1, v2 = augment(y), augment(y)
        _, f1 = self.d(v1, x, cond, return_features=True)
        _, f2 = self.d(v2, x, cond, return_features=True)
        con = nt_xent(self.d.project(f1), self.d.project(f2), cfg.tau)

        loss = adv + cfg.lambda_gp * gp + cfg.w_contrastive * con
        self.opt_d.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.d.parameters(), cfg.grad_clip)
        self.opt_d.step()

        return {
            "d_adv": float(adv.detach()),
            "d_gp": float(gp.detach()),
            "d_con": float(con.detach()),
            "wasserstein": float((d_real.mean() - d_fake.mean()).detach()),
        }

    def _generator_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        cfg = self.cfg
        x, cond, y = batch["x_hist"], batch["cond"], batch["y"]

        y_fake = self.g(x, cond)
        score, feat_fake = self.d(y_fake, x, cond, return_features=True)
        g_adv = generator_wgan_loss(score)

        z_fake = F.normalize(self.d.project(feat_fake), dim=1)
        with torch.no_grad():
            _, feat_real = self.d(y, x, cond, return_features=True)
            z_real = F.normalize(self.d.project(feat_real), dim=1)

        l_unif = uniformity(z_fake) if cfg.w_uniform > 0 else y_fake.new_zeros(())
        l_mmd = rbf_mmd(z_fake, z_real) if cfg.w_mmd > 0 else y_fake.new_zeros(())

        loss = g_adv + cfg.w_uniform * l_unif + cfg.w_mmd * l_mmd
        self.opt_g.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_d.zero_grad(set_to_none=True)
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.g.parameters(), cfg.grad_clip)
        self.opt_g.step()
        self._update_ema()

        return {
            "g_adv": float(g_adv.detach()),
            "g_unif": float(l_unif.detach()),
            "g_mmd": float(l_mmd.detach()),
        }

    @torch.no_grad()
    def _update_ema(self) -> None:
        decay = self.cfg.ema_decay
        for p_ema, p in zip(self.g_ema.parameters(), self.g.parameters()):
            p_ema.lerp_(p, 1.0 - decay)
        for b_ema, b in zip(self.g_ema.buffers(), self.g.buffers()):
            b_ema.copy_(b)

    def fit(self, dataset, steps: int | None = None, verbose: bool = True) -> list[dict]:
        cfg = self.cfg
        steps = steps or cfg.steps
        if len(dataset) < cfg.batch_size:
            raise ValueError(
                f"Dataset has {len(dataset)} windows, fewer than batch_size={cfg.batch_size}."
            )
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=cfg.num_workers,
        )
        stream = _infinite(loader)
        t_start = time.time()

        for _ in range(steps):
            self.step += 1
            d_logs: dict[str, float] = {}
            for _ in range(cfg.n_critic):
                d_logs = self._critic_step(self._to_device(next(stream)))
            g_logs = self._generator_step(self._to_device(next(stream)))

            row = {"step": self.step, **d_logs, **g_logs,
                   "elapsed_s": round(time.time() - t_start, 1)}
            bad = [k for k, v in row.items() if isinstance(v, float) and not math.isfinite(v)]
            if bad:
                raise RuntimeError(
                    f"Non-finite training values at step {self.step}: {bad}. "
                    "Lower the learning rates or raise n_critic and rerun."
                )
            if self.step % cfg.log_every == 0 or self.step == 1:
                self.history.append(row)
                self.logger.log(row)
                if verbose:
                    print(
                        f"step {self.step:>6d}  W={row['wasserstein']:+.4f}  "
                        f"gp={row['d_gp']:.4f}  con={row['d_con']:.4f}  "
                        f"g_adv={row['g_adv']:+.4f}  unif={row['g_unif']:+.4f}  "
                        f"mmd={row['g_mmd']:.4f}",
                        flush=True,
                    )
            if cfg.ckpt_every > 0 and self.step % cfg.ckpt_every == 0:
                self.save_checkpoint(self.out_dir / f"ckpt_step{self.step}.pt")

        self.save_checkpoint(self.out_dir / "ckpt_final.pt")
        return self.history

    def load_state(self, path: str | Path) -> int:
        """Resume from a checkpoint saved by this trainer; returns the step."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.g.load_state_dict(ckpt["generator"])
        self.g_ema.load_state_dict(ckpt["generator_ema"])
        self.d.load_state_dict(ckpt["critic"])
        self.opt_g.load_state_dict(ckpt["opt_g"])
        self.opt_d.load_state_dict(ckpt["opt_d"])
        self.step = int(ckpt["step"])
        return self.step

    def save_checkpoint(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": self.step,
                "generator": self.g.state_dict(),
                "generator_ema": self.g_ema.state_dict(),
                "critic": self.d.state_dict(),
                "opt_g": self.opt_g.state_dict(),
                "opt_d": self.opt_d.state_dict(),
                "train_config": asdict(self.cfg),
                "model_meta": self.model_meta,
            },
            path,
        )
        return path


def build_models(model_meta: dict) -> tuple[TFTGenerator, Critic]:
    """Construct generator + critic from a serialisable dimension dict."""
    m = model_meta
    generator = TFTGenerator(
        n_features=m["n_features"],
        n_assets=m["n_assets"],
        cond_dim=m["cond_dim"],
        lookback=m["lookback"],
        horizon=m["horizon"],
        hidden=m["hidden"],
        heads=m["heads"],
        z_dim=m["z_dim"],
        dropout=m["dropout"],
        lstm_layers=m["lstm_layers"],
    )
    critic = build_critic(
        n_assets=m["n_assets"],
        n_features=m["n_features"],
        cond_dim=m["cond_dim"],
        horizon=m["horizon"],
        hidden=m["hidden"],
        arch=m["critic_arch"],
        channels=list(m["critic_channels"]),
        ctx_channels=m["critic_ctx_channels"],
        proj_dim=m["proj_dim"],
        heads=m["heads"],
        dropout=m["dropout"],
    )
    return generator, critic


def load_generator(path: str | Path, map_location: str = "cpu", ema: bool = True) -> tuple[TFTGenerator, dict]:
    """Load a trained generator (EMA weights by default) plus its meta dict."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    generator, _ = build_models(ckpt["model_meta"])
    key = "generator_ema" if (ema and "generator_ema" in ckpt) else "generator"
    generator.load_state_dict(ckpt[key])
    generator.eval()
    return generator, ckpt["model_meta"]
