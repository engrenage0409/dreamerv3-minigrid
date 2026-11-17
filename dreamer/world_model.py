from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .rssm import RSSM


class ImageEncoder(nn.Module):
    def __init__(self, image_shape=(3, 7, 7), embed_dim=256):
        super().__init__()
        c, h, w = image_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(), nn.Linear(64 * ((h + 1) // 2) * ((w + 1) // 2), embed_dim),
            nn.LayerNorm(embed_dim), nn.SiLU(),
        )

    def forward(self, image):
        return self.net(image)


class WorldModel(nn.Module):
    """Image encoder + discrete RSSM + image/reward/continuation decoders."""

    def __init__(self, action_dim: int, image_shape=(3, 7, 7), embed_dim=256,
                 deter_dim=256, stoch_groups=16, classes=16):
        super().__init__()
        self.encoder = ImageEncoder(image_shape, embed_dim)
        self.rssm = RSSM(action_dim, embed_dim, deter_dim, stoch_groups, classes)
        feat_dim = deter_dim + stoch_groups * classes
        self.decoder = nn.Sequential(nn.Linear(feat_dim, 512), nn.SiLU(),
                                      nn.Linear(512, int(torch.tensor(image_shape).prod())))
        self.reward = nn.Sequential(nn.Linear(feat_dim, 256), nn.SiLU(), nn.Linear(256, 1))
        self.continue_head = nn.Sequential(nn.Linear(feat_dim, 256), nn.SiLU(), nn.Linear(256, 1))
        self.image_shape = image_shape

    def get_feat(self, state):
        return torch.cat([state["deter"], state["stoch"]], -1)

    def decode(self, state):
        logits = self.decoder(self.get_feat(state))
        return logits.reshape(*logits.shape[:-1], *self.image_shape)

    def forward(self, images, actions, is_first=None):
        """Sequence input: images [B,T,C,H,W], actions [B,T,A]."""
        b, t = images.shape[:2]
        embeds = self.encoder(images.reshape(b * t, *images.shape[2:])).reshape(b, t, -1)
        state = self.rssm.initial(b, images.device)
        posts, priors = [], []
        zero_action = torch.zeros_like(actions[:, 0])
        for i in range(t):
            action = zero_action if i == 0 else actions[:, i - 1]
            first = is_first[:, i] if is_first is not None else None
            post, prior = self.rssm.obs_step(state, action, embeds[:, i], first)
            posts.append(post)
            priors.append(prior)
            state = post
        post = {k: torch.stack([s[k] for s in posts], 1) for k in posts[0]}
        prior = {k: torch.stack([s[k] for s in priors], 1) for k in priors[0]}
        feat = self.get_feat(post)
        return {"post": post, "prior": prior, "image_logits": self.decode(post),
                "reward": self.reward(feat).squeeze(-1),
                "continue_logits": self.continue_head(feat).squeeze(-1)}


def balanced_kl(post_logits, prior_logits, balance=0.8, free_nats=1.0):
    """Balanced categorical KL, with free-nats floor per batch/time position."""
    post = torch.distributions.OneHotCategorical(logits=post_logits)
    prior = torch.distributions.OneHotCategorical(logits=prior_logits)
    lhs = torch.distributions.kl_divergence(
        torch.distributions.OneHotCategorical(logits=post_logits.detach()), prior)
    rhs = torch.distributions.kl_divergence(
        post, torch.distributions.OneHotCategorical(logits=prior_logits.detach()))
    # Sum independent stochastic groups, then apply free nats.
    kl = balance * lhs.sum(-1) + (1.0 - balance) * rhs.sum(-1)
    return torch.clamp(kl, min=free_nats).mean()


def world_model_loss(model, batch, kl_balance=0.8, free_nats=1.0):
    out = model(batch["images"], batch["actions"], batch.get("is_first"))
    image_loss = F.binary_cross_entropy_with_logits(
        out["image_logits"], batch["images"], reduction="none").mean((2, 3, 4)).mean()
    reward_loss = F.mse_loss(out["reward"], batch["rewards"])
    cont_loss = F.binary_cross_entropy_with_logits(
        out["continue_logits"], batch["continues"])
    kl_loss = balanced_kl(out["post"]["logits"], out["prior"]["logits"],
                          kl_balance, free_nats)
    total = image_loss + reward_loss + cont_loss + kl_loss
    return total, {"image": image_loss.detach(), "reward": reward_loss.detach(),
                   "continue": cont_loss.detach(), "kl": kl_loss.detach()}
