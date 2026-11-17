from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical


class RSSM(nn.Module):
    """Discrete-stochastic RSSM core (Dreamer-style, compact baseline)."""

    def __init__(self, action_dim: int, embed_dim: int, deter_dim: int = 256,
                 stoch_groups: int = 16, classes: int = 16):
        super().__init__()
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_groups = stoch_groups
        self.classes = classes
        self.stoch_dim = stoch_groups * classes
        self.gru = nn.GRUCell(self.stoch_dim + action_dim, deter_dim)
        self.prior = nn.Sequential(nn.Linear(deter_dim, deter_dim), nn.SiLU(),
                                   nn.Linear(deter_dim, stoch_groups * classes))
        self.posterior = nn.Sequential(nn.Linear(deter_dim + embed_dim, deter_dim),
                                       nn.SiLU(),
                                       nn.Linear(deter_dim, stoch_groups * classes))

    def _sample(self, logits: torch.Tensor, sample: bool = True):
        logits = logits.reshape(*logits.shape[:-1], self.stoch_groups, self.classes)
        probs = logits.softmax(-1)
        if sample:
            stoch = OneHotCategorical(probs=probs).sample()
        else:
            stoch = F.one_hot(probs.argmax(-1), self.classes).to(probs.dtype)
        return stoch.flatten(-2), logits

    def initial(self, batch_size: int, device):
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        logits = torch.zeros(batch_size, self.stoch_groups, self.classes, device=device)
        stoch = F.one_hot(torch.zeros(batch_size, self.stoch_groups, dtype=torch.long,
                                      device=device), self.classes).float().flatten(1)
        return {"deter": deter, "stoch": stoch, "logits": logits}

    def img_step(self, prev: dict, action: torch.Tensor, sample: bool = True):
        deter = self.gru(torch.cat([prev["stoch"], action], -1), prev["deter"])
        stoch, logits = self._sample(self.prior(deter), sample)
        return {"deter": deter, "stoch": stoch, "logits": logits}

    def obs_step(self, prev: dict, action: torch.Tensor, embed: torch.Tensor,
                 is_first: torch.Tensor | None = None, sample: bool = True):
        if is_first is not None:
            keep = (1.0 - is_first.float()).unsqueeze(-1)
            prev = {k: v * keep for k, v in prev.items()}
        prior = self.img_step(prev, action, sample)
        logits = self.posterior(torch.cat([prior["deter"], embed], -1))
        stoch, logits = self._sample(logits, sample)
        post = {"deter": prior["deter"], "stoch": stoch, "logits": logits}
        return post, prior
