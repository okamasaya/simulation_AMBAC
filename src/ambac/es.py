"""numpy のみで書いた小さな方策と進化戦略（OpenAI-ES）.

方策: 多層パーセプトロン（tanh）。パラメータは 1 本のベクトルで扱い、
母集団分のパラメータを一括で評価できるようにしている。
"""
from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, sizes):
        self.sizes = list(sizes)
        self.shapes = []
        for a, b in zip(self.sizes[:-1], self.sizes[1:]):
            self.shapes += [(a, b), (b,)]
        self.n_params = int(sum(np.prod(s) for s in self.shapes))

    def init(self, rng, scale=0.5):
        out = []
        for s in self.shapes:
            out.append(rng.normal(0, scale / np.sqrt(s[0]), s).ravel() if len(s) == 2 else np.zeros(s))
        return np.concatenate(out)

    def unpack(self, P):
        """P: (pop, n_params) → 層ごとの重み (pop, a, b), (pop, b)."""
        P = np.atleast_2d(P)
        out, i = [], 0
        for s in self.shapes:
            k = int(np.prod(s))
            out.append(P[:, i:i + k].reshape(P.shape[0], *s))
            i += k
        return out

    def __call__(self, P, x):
        """P: (pop, n_params), x: (pop, batch, in) → (pop, batch, out), 出力は tanh で [-1,1]."""
        W = self.unpack(P)
        h = x
        for j in range(0, len(W), 2):
            h = np.tanh(np.einsum("pbi,pio->pbo", h, W[j]) + W[j + 1][:, None, :])
        return h


class OpenAIES:
    """対称サンプリング・順位変換・Adam による ES."""

    def __init__(self, x0, sigma=0.05, lr=0.03, n_pairs=32, seed=0):
        self.x = np.array(x0, float)
        self.sigma, self.lr, self.n_pairs = sigma, lr, n_pairs
        self.rng = np.random.default_rng(seed)
        self.m = np.zeros_like(self.x); self.v = np.zeros_like(self.x); self.t = 0

    def ask(self):
        self.eps = self.rng.normal(size=(self.n_pairs, self.x.size))
        return np.concatenate([self.x + self.sigma * self.eps, self.x - self.sigma * self.eps])

    def tell(self, fitness):
        """fitness: 大きいほど良い (2*n_pairs,)"""
        r = np.empty(len(fitness)); r[np.argsort(fitness)] = np.arange(len(fitness))
        r = r / (len(r) - 1) - 0.5
        g = ((r[:self.n_pairs] - r[self.n_pairs:])[:, None] * self.eps).mean(0) / self.sigma
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g**2
        mh = self.m / (1 - 0.9**self.t); vh = self.v / (1 - 0.999**self.t)
        self.x += self.lr * mh / (np.sqrt(vh) + 1e-8)
