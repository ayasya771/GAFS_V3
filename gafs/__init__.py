"""GAFS: Generative Adversarial Financial Simulation.

Institutional-grade synthetic financial scenario engine combining:

* a Conditional Temporal Fusion Transformer (TFT) generator with per-step
  Gaussian noise injection,
* a Wasserstein GAN critic trained with a gradient penalty (WGAN-GP), and
* SimCLR-style time-series contrastive learning (NT-Xent) that regularises
  the latent space against mode collapse, so rare tail events are not
  averaged away.

Subpackages
-----------
data        acquisition (Yahoo, FRED, Binance, Alpaca, Dukascopy), synthetic
            market generation, preprocessing and windowing
models      TFT generator, WGAN-GP critic (1D ResNet / Transformer)
training    augmentations, losses (WGAN-GP, NT-Xent, uniformity, MMD), trainer
evaluation  stylized-facts battery and real-vs-synthetic reporting
simulation  conditioned scenario generation and macro stress shocks
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
