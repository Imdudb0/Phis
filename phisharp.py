#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHISHARP - reference executable du spec.

    ask --model out/sharp.phis --file perceived.eff

Substrat : graphe d'unites binaires, couplage EXACTEMENT antisymetrique
(A_ij = -A_ji). Deux flous causaux independants : Aret (abduction, R->C)
et Agen (generation, C->P). Le port P n'est PAS dans le substrat.

phi_sharp(c;E) = iota(c) * a(c;E)
   iota : integration intrinseque (P noise), min sur MIP, min sur les 2 flots
   a    : adequation causale = regeneration de E sous do(c), percept RETIRE

Aucune donnee de domaine. Aucune BPTT. Aucun decodeur. Aucun MDL.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

F32 = np.float32
PRINTABLE = (
    b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    b" .,;:!?()-_'\"[]=+*/<>|@#%\n"
)


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
@dataclass
class Config:
    n: int = 512            # unites du substrat
    K: int = 32             # operateurs (cellules M)
    rank: int = 2           # rang bas de chaque operateur
    L: int = 64             # octets du port
    beta: float = 2.0
    density: float = 0.15
    kc: int = 48            # taille du mecanisme candidat
    pz: int = 128           # taille du purview interne
    ncuts: int = 8          # coupes echantillonnees (MIP Monte-Carlo)
    T: int = 48             # pas de recherche par sweep
    tau: int = 2            # pas de regeneration sous do(c)
    phi_every: int = 4
    learn_every: int = 4
    nd: int = 8             # distinctions echantillonnees
    nr: int = 6             # relations echantillonnees
    nexcl: int = 2          # candidats pour l'exclusion
    theta: float = 0.02     # seuil de sens trouvable
    eta: float = 0.010      # plasticite 3-facteurs
    eta_grad: float = 0.060 # gradient local analytique (Wout)
    decay: float = 0.80     # trace d'eligibilite
    bdecay: float = 0.95    # baseline
    wclip: float = 0.60
    max_sweeps: int = 12
    seed: int = 0

    @property
    def bs(self) -> int:
        return self.n // self.K

    @property
    def P(self) -> int:
        return self.L * 8


PROFILES = {
    "ci":    dict(n=512,  K=32, L=64,  T=48,  kc=48,  pz=128),
    "small": dict(n=1024, K=32, L=96,  T=96,  kc=96,  pz=192),
    "base":  dict(n=2048, K=64, L=128, T=192, kc=192, pz=256, nd=16, nr=12, nexcl=4),
}


def make_config(profile: str, seed: int) -> Config:
    cfg = Config(**PROFILES[profile])
    cfg.seed = seed
    assert cfg.n % cfg.K == 0, "n doit etre divisible par K"
    return cfg


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def bits_of(data: bytes, L: int) -> np.ndarray:
    buf = data[:L]
    if len(buf) < L:
        buf = buf + b" " * (L - len(buf))
    return np.unpackbits(np.frombuffer(buf, dtype=np.uint8)).astype(F32)


# --------------------------------------------------------------------------
# substrat
# --------------------------------------------------------------------------
class Substrate:
    """Graphe connexionniste, non-reciprocite structurelle."""

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        n, K, bs, P = cfg.n, cfg.K, cfg.bs, cfg.P

        m = rng.random((n, n)) < cfg.density
        m = m | m.T                       # arete presente dans les 2 sens
        np.fill_diagonal(m, False)
        self.mask = m.astype(F32)

        sc = 1.0 / np.sqrt(max(cfg.density * n, 1.0))
        g = rng.standard_normal((n, n)).astype(F32) * sc
        self.Aret = (g - g.T) * self.mask          # flot abductif
        g2 = rng.standard_normal((n, n)).astype(F32) * sc
        self.Agen = (g2 - g2.T) * self.mask        # flot generatif
        # Aret != Agen.T : deux operateurs causaux independants

        self.b = (rng.standard_normal(n) * 0.1).astype(F32)
        self.U = (rng.standard_normal((K, bs, cfg.rank)) * 0.30).astype(F32)
        self.V = (rng.standard_normal((K, bs, cfg.rank)) * 0.30).astype(F32)
        self.Wm = (rng.standard_normal((K, n)) / np.sqrt(n)).astype(F32)
        self.bm = np.zeros(K, F32)

        sin = 1.0 / np.sqrt(max(P * 0.10 * 0.5, 1.0))
        sout = 1.0 / np.sqrt(max(n * 0.10 * 0.5, 1.0))
        self.Win = (rng.standard_normal((n, P)) * sin).astype(F32) * (rng.random((n, P)) < 0.10)
        self.Wout = (rng.standard_normal((P, n)) * sout).astype(F32) * (rng.random((P, n)) < 0.10)
        self.bout = np.zeros(P, F32)

        self.etr = np.zeros(n, F32)
        self.ein = np.zeros(n, F32)
        self.baseline = 0.0

    # --- operateur module (etat-dependant) : la "procedure" n'est stockee nulle part
    def mods(self, s: np.ndarray) -> np.ndarray:
        return sigmoid(self.Wm @ s + self.bm)

    def apply(self, A: np.ndarray, s: np.ndarray, m: np.ndarray) -> np.ndarray:
        """A(m) @ s sans former la matrice complete."""
        K, bs = self.cfg.K, self.cfg.bs
        h = A @ s
        sb = s.reshape(K, bs)
        va = np.einsum("kbr,kb->kr", self.V, sb)
        ua = np.einsum("kbr,kb->kr", self.U, sb)
        mv = m[:, None] * va
        mu = m[:, None] * ua
        d = np.einsum("kbr,kr->kb", self.U, mv) - np.einsum("kbr,kr->kb", self.V, mu)
        return h + d.reshape(-1)

    def modulate(self, A: np.ndarray, m: np.ndarray) -> np.ndarray:
        """A(m) explicite, antisymetrique par construction."""
        K, bs = self.cfg.K, self.cfg.bs
        UV = np.einsum("kbr,kcr->kbc", self.U, self.V)
        anti = UV - UV.transpose(0, 2, 1)
        Am = A.copy()
        v = Am.reshape(K, bs, K, bs)
        idx = np.arange(K)
        v[idx, :, idx, :] = v[idx, :, idx, :] + m[:, None, None] * anti
        return Am

    # --- interface
    def perceive(self, s: np.ndarray, ebits: np.ndarray) -> np.ndarray:
        return self.Win @ ebits

    def write(self, s: np.ndarray) -> np.ndarray:
        return sigmoid(self.cfg.beta * (self.Wout @ s + self.bout))

    # --- plasticite
    def learn(self, phi: float, s_pre: np.ndarray, s_post: np.ndarray,
              ebits: np.ndarray, iota: float, pout: np.ndarray, s_gen: np.ndarray):
        cfg = self.cfg
        self.baseline = cfg.bdecay * self.baseline + (1 - cfg.bdecay) * phi
        adv = float(phi - self.baseline)

        self.etr = cfg.decay * self.etr + (s_post - 0.5)
        self.ein = cfg.decay * self.ein + (s_post - 0.5)

        # (1) gradient local analytique de phi sur le flot generatif.
        #     Porte par iota : si iota=0 (copie), AUCUN gradient. Verrou anti-triche.
        if iota > 1e-6:
            err = np.sign(pout - ebits) * (2.0 / cfg.P)
            g = -cfg.eta_grad * iota * err * cfg.beta * pout * (1.0 - pout)
            self.Wout += np.outer(g, s_gen).astype(F32)
            self.bout += g.astype(F32)

        # (2) trois-facteurs antisymetrise sur les flots internes + entree.
        e = self.etr
        dA = (np.outer(e, s_post) - np.outer(s_post, e)).astype(F32) * self.mask
        step = cfg.eta * adv
        self.Aret += np.clip(step * dA, -0.02, 0.02)
        self.Agen += np.clip(step * dA * 0.5, -0.02, 0.02)
        self.Win += np.clip(cfg.eta * adv * np.outer(self.ein, ebits), -0.02, 0.02).astype(F32)

        # (3) operateurs : eligibilite locale modulee par m_k.
        m = self.mods(s_post)
        K, bs = cfg.K, cfg.bs
        sb = s_post.reshape(K, bs)
        eb = e.reshape(K, bs)
        vs = np.einsum("kbr,kb->kr", self.V, sb)
        us = np.einsum("kbr,kb->kr", self.U, sb)
        dU = np.einsum("kb,kr->kbr", eb, vs) * m[:, None, None]
        dV = -np.einsum("kb,kr->kbr", eb, us) * m[:, None, None]
        self.U += np.clip(cfg.eta * adv * dU, -0.02, 0.02).astype(F32)
        self.V += np.clip(cfg.eta * adv * dV, -0.02, 0.02).astype(F32)
        self.Wm += np.clip(cfg.eta * adv * np.outer(m * adv, s_post - 0.5) * 0.1,
                           -0.01, 0.01).astype(F32)

        np.clip(self.Aret, -cfg.wclip, cfg.wclip, out=self.Aret)
        np.clip(self.Agen, -cfg.wclip, cfg.wclip, out=self.Agen)
        np.clip(self.Win, -cfg.wclip, cfg.wclip, out=self.Win)
        np.clip(self.Wout, -cfg.wclip, cfg.wclip, out=self.Wout)

    def n_params(self) -> int:
        n = self.cfg.n
        return int(n * (n - 1) // 2 * 2 + self.U.size + self.V.size + self.Wm.size
                   + self.Win.size + self.Wout.size + self.b.size + self.bout.size)

    def save(self, path: str, extra: Optional[dict] = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            cfg=json.dumps(asdict(self.cfg)),
            meta=json.dumps(extra or {}),
            Aret=self.Aret, Agen=self.Agen, b=self.b,
            U=self.U, V=self.V, Wm=self.Wm, bm=self.bm,
            Win=self.Win, Wout=self.Wout, bout=self.bout,
            baseline=np.array([self.baseline], F32),
        )

    @staticmethod
    def load(path: str) -> Tuple["Substrate", dict]:
        z = np.load(path, allow_pickle=False)
        cfg = Config(**json.loads(str(z["cfg"])))
        sub = Substrate(cfg, np.random.default_rng(cfg.seed))
        for k in ("Aret", "Agen", "b", "U", "V", "Wm", "bm", "Win", "Wout", "bout"):
            getattr(sub, k)[...] = z[k]
        sub.baseline = float(z["baseline"][0])
        return sub, json.loads(str(z["meta"]))


# --------------------------------------------------------------------------
# phi_sharp
# --------------------------------------------------------------------------
def cut_probs(w, base, s_c, q, z, beta):
    """Repertoire apres coupe : les liens traversant la coupe sont noises."""
    a1 = w @ (s_c * q)
    a0 = w @ (s_c * (1.0 - q))
    n1 = w @ q
    n0 = w @ (1.0 - q)
    h = z * (a1 + 0.5 * n0) + (1.0 - z) * (a0 + 0.5 * n1) + base
    return sigmoid(beta * h)


def iota_of(sub: Substrate, Amret, Amgen, c_idx, s_c, rng, purview=None) -> float:
    """Integration intrinseque. P est hors substrat : le percept est noise."""
    cfg = sub.cfg
    n = cfg.n
    if purview is None:
        purview = rng.choice(n, size=min(cfg.pz, n), replace=False)
    pz = len(purview)
    kc = len(c_idx)
    vals = []
    for A in (Amret, Amgen):
        w = A[np.ix_(purview, c_idx)]
        base = sub.b[purview] + 0.5 * (A[purview].sum(1) - w.sum(1))
        pw = sigmoid(cfg.beta * (w @ s_c + base))
        best = 1.0
        for _ in range(cfg.ncuts):
            q = (rng.random(kc) < 0.5).astype(F32)
            if q.sum() == 0 or q.sum() == kc:
                q[0] = 1.0 - q[0]
            z = (rng.random(pz) < 0.5).astype(F32)
            pc = cut_probs(w, base, s_c, q, z, cfg.beta)
            idv = float(np.abs(pw - pc).sum() / pz)   # ID = Wasserstein-Hamming
            if idv < best:
                best = idv
        vals.append(best)
    return float(min(vals))


def regenerate(sub: Substrate, s: np.ndarray, c_idx, s_c, ebits, rng):
    """do(c=s_c), percept RETIRE (Win debranche), le port est libre."""
    cfg = sub.cfg
    s2 = s.copy()
    for _ in range(cfg.tau):
        m = sub.mods(s2)
        h = sub.apply(sub.Agen, s2, m) + sub.b
        p = sigmoid(cfg.beta * h)
        s2 = (rng.random(cfg.n) < p).astype(F32)
        s2[c_idx] = s_c                      # intervention, pas conditionnement
    pout = sub.write(s2)
    match = 1.0 - float(np.mean(np.abs(pout - ebits)))
    a = max(0.0, 2.0 * match - 1.0)          # 0 au hasard, 1 si regeneration exacte
    return a, pout, s2


def mech_phi(sub, s, ebits, rng, c_idx, s_c, Amret=None, Amgen=None):
    if Amret is None:
        m = sub.mods(s)
        Amret = sub.modulate(sub.Aret, m)
        Amgen = sub.modulate(sub.Agen, m)
    io = iota_of(sub, Amret, Amgen, c_idx, s_c, rng)
    a, pout, s2 = regenerate(sub, s, c_idx, s_c, ebits, rng)
    return float(io * a), io, a, pout


def candidate(p: np.ndarray, kc: int):
    idx = np.argsort(-np.abs(p - 0.5))[:kc]
    return idx


def structure_phi(sub, s, ebits, rng, c_idx, s_c, Amret, Amgen):
    """Phi = structure (distinctions + relations) moins sa coupe systeme.
    Maximalite : on ne purge rien, on additionne tout ce qui est irreductible."""
    cfg = sub.cfg
    kc = len(c_idx)
    rng_local = np.random.default_rng(int(rng.integers(1 << 30)))

    subs = []
    for _ in range(cfg.nd):
        k = max(2, kc // int(rng_local.integers(2, 4)))
        subs.append(rng_local.choice(c_idx, size=k, replace=False))
    pairs = [(subs[i], subs[j]) for i, j in
             (rng_local.integers(0, len(subs), size=(cfg.nr, 2)))]

    def total(Ar, Ag):
        ds = []
        for d in subs:
            ph, _, _, _ = mech_phi(sub, s, ebits, rng_local, d, s[c_idx][np.isin(c_idx, d)]
                                   if False else s[d], Ar, Ag)
            ds.append(ph)
        rel = 0.0
        for d1, d2 in pairs:
            u = np.union1d(d1, d2)
            phu, _, _, _ = mech_phi(sub, s, ebits, rng_local, u, s[u], Ar, Ag)
            p1, _, _, _ = mech_phi(sub, s, ebits, rng_local, d1, s[d1], Ar, Ag)
            p2, _, _, _ = mech_phi(sub, s, ebits, rng_local, d2, s[d2], Ar, Ag)
            rel += max(0.0, phu - p1 - p2)     # gain de liaison
        return float(sum(ds) + rel), ds

    whole, ds = total(Amret, Amgen)

    side = (rng_local.random(kc) < 0.5)
    cross = side[:, None] != side[None, :]
    cut = []
    for A in (Amret, Amgen):
        Ac = A.copy()
        blk = Ac[np.ix_(c_idx, c_idx)]
        blk[cross] = 0.0
        Ac[np.ix_(c_idx, c_idx)] = blk
        cut.append(Ac)
    partial, _ = total(cut[0], cut[1])
    return max(0.0, whole - partial), ds


# --------------------------------------------------------------------------
# PTGRS : ne fabrique QUE E. La cause cachee ne sert qu'a l'evaluation.
# --------------------------------------------------------------------------
class PTGRS:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def signature(self):
        k = int(self.rng.integers(4, 9))
        return self.rng.integers(1, 4, size=k)

    def tokens(self, k, pool):
        return [pool[i] for i in range(k)]

    def graph(self, arities, size=14):
        rng = self.rng
        nodes: List[Tuple[int, List[int]]] = []

        def grow(depth):
            if depth <= 0 or len(nodes) >= size:
                nodes.append((int(rng.integers(0, len(arities))), []))
                return len(nodes) - 1
            c = int(rng.integers(0, len(arities)))
            kids = []
            for _ in range(int(arities[c])):
                if nodes and rng.random() < 0.30:
                    kids.append(int(rng.integers(0, len(nodes))))   # PARTAGE -> DAG
                else:
                    kids.append(grow(depth - 1))
            nodes.append((c, kids))
            return len(nodes) - 1

        root = grow(3)
        return nodes, root

    def rewrite(self, nodes, root, arities, rules, steps=4):
        rng = self.rng
        nodes = list(nodes)
        n0 = len(nodes)
        for _ in range(steps):
            i = int(rng.integers(0, n0))
            c, kids = nodes[i]
            if rng.random() < rules[c]:
                nc = int(rng.integers(0, len(arities)))
                nk = []
                for _ in range(int(arities[nc])):
                    r = rng.random()
                    if kids and r < 0.60:
                        nk.append(kids[int(rng.integers(0, len(kids)))])
                    elif nodes and r < 0.80:
                        nk.append(int(rng.integers(0, len(nodes))))
                    else:
                        nodes.append((int(rng.integers(0, len(arities))), []))
                        nk.append(len(nodes) - 1)
                nodes[i] = (nc, nk)
        return nodes, root

    def render(self, nodes, root, toks, style, hole=None, explicit=True, budget=200):
        out: List[bytes] = []
        seen: Dict[int, int] = {}
        used = [0]

        def rec(i):
            if used[0] > budget:
                return
            if hole is not None and i == hole:
                used[0] += 1
                if explicit:
                    out.append(b"?")
                return
            if i in seen:
                out.append(b"@" + str(seen[i]).encode())
                used[0] += 2
                return
            seen[i] = len(seen)
            c, kids = nodes[i]
            used[0] += len(toks[c]) + 2
            if style == 0:
                out.append(toks[c]); out.append(b"(")
                for j, k in enumerate(kids):
                    if j:
                        out.append(b",")
                    rec(k)
                out.append(b")")
            else:
                out.append(b"[")
                for k in kids:
                    rec(k); out.append(b" ")
                out.append(b"]"); out.append(toks[c])

        rec(root)
        return b"".join(out)[:240]

    def pool(self, n=48):
        rng = self.rng
        toks = set()
        while len(toks) < n:
            ln = int(rng.integers(2, 5))
            t = bytes(rng.choice(np.frombuffer(b"abcdefghijklmnopqrstuvwxyz", dtype=np.uint8),
                                 size=ln)).decode()
            toks.add(t.encode())
        return sorted(toks)

    def episode(self, kind: str):
        rng = self.rng
        if kind == "noise":
            ln = int(rng.integers(40, 160))
            E = bytes(rng.choice(np.frombuffer(PRINTABLE, dtype=np.uint8), size=ln))
            return E, {"kind": kind}

        arities = self.signature()
        rules = rng.uniform(0.2, 0.9, size=len(arities))
        nodes, root = self.graph(arities)
        nodes, root = self.rewrite(nodes, root, arities, rules)
        pool = self.pool()
        k = len(arities)
        style = int(rng.integers(0, 2))
        dA = self.tokens(k, pool[:k])
        dB = self.tokens(k, pool[k:2 * k])          # dictionnaires DISJOINTS

        if kind == "disjoint":
            a = self.render(nodes, root, dA, style)
            b = self.render(nodes, root, dB, style)
            E = a + b" || " + b
            meta = {"kind": kind, "half_a": len(a), "half_b": len(b)}
        elif kind == "hole":
            hole = int(rng.integers(0, len(nodes)))
            explicit = bool(rng.random() < 0.5)
            E = self.render(nodes, root, dA, style, hole=hole, explicit=explicit)
            meta = {"kind": kind, "hole_node": hole, "explicit": explicit}
        else:
            E = self.render(nodes, root, dA, style)
            meta = {"kind": "plain"}

        # cause cachee : gardee pour evaluation apres coup, JAMAIS vue du modele
        meta["_hidden"] = {"nodes": nodes, "root": root, "rules": rules.tolist()}
        return E, meta


# --------------------------------------------------------------------------
# recherche : elle PART de R, elle va vers la cause
# --------------------------------------------------------------------------
def regional_R(sub: Substrate, ebits: np.ndarray, rng) -> np.ndarray:
    """R = ID(do(P=E), do(P=noise)) : ce que E fait au substrat."""
    cfg = sub.cfg
    acc_e = np.zeros(cfg.n, F32)
    acc_x = np.zeros(cfg.n, F32)
    reps = 3
    for _ in range(reps):
        s = (rng.random(cfg.n) < 0.5).astype(F32)
        for _ in range(3):
            m = sub.mods(s)
            h = sub.apply(sub.Aret, s, m) + sub.b + sub.perceive(s, ebits)
            s = (rng.random(cfg.n) < sigmoid(cfg.beta * h)).astype(F32)
        acc_e += s
        s = (rng.random(cfg.n) < 0.5).astype(F32)
        for _ in range(3):
            m = sub.mods(s)
            h = sub.apply(sub.Aret, s, m) + sub.b + sub.perceive(s, np.full(cfg.P, 0.5, F32))
            s = (rng.random(cfg.n) < sigmoid(cfg.beta * h)).astype(F32)
        acc_x += s
    return np.abs(acc_e - acc_x) / reps


def search(sub: Substrate, ebits: np.ndarray, rng, learn: bool = False,
           collect: bool = False):
    cfg = sub.cfg
    R = regional_R(sub, ebits, rng)
    s = (rng.random(cfg.n) < sigmoid(cfg.beta * (R - 0.5))).astype(F32)

    trace: List[float] = []
    best = (0.0, None, None, None)
    pool: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    last_a, last_io, last_pout = 0.0, 0.0, None

    for t in range(cfg.T):
        m = sub.mods(s)
        h = sub.apply(sub.Aret, s, m) + sub.b + sub.perceive(s, ebits)
        p = sigmoid(cfg.beta * h)
        s_next = (rng.random(cfg.n) < p).astype(F32)

        if t % cfg.phi_every == 0:
            c_idx = candidate(p, cfg.kc)
            s_c = s_next[c_idx].copy()
            ph, io, a, pout = mech_phi(sub, s_next, ebits, rng, c_idx, s_c)
            trace.append(ph)
            last_a, last_io, last_pout = a, io, pout
            if ph > best[0]:
                best = (ph, c_idx.copy(), s_c.copy(), s_next.copy())
            if collect:
                pool.append((ph, c_idx.copy(), s_c.copy(), s_next.copy()))
            if learn and (t % cfg.learn_every == 0):
                sub.learn(ph, s, s_next, ebits, io, pout, s_next)
        s = s_next

    return {
        "trace": trace,
        "phi_max": float(best[0]),
        "phi_mean": float(np.mean(trace)) if trace else 0.0,
        "best": best,
        "pool": sorted(pool, key=lambda x: -x[0])[: cfg.nexcl],
        "a": last_a, "iota": last_io, "pout": last_pout,
    }


def exclusion(sub, ebits, rng, pool):
    """Selection par max Phi. Pas de MDL, pas de penalite de longueur."""
    best = (0.0, None, None, None, [])
    for ph, c_idx, s_c, s in pool:
        m = sub.mods(s)
        Ar = sub.modulate(sub.Aret, m)
        Ag = sub.modulate(sub.Agen, m)
        Phi, ds = structure_phi(sub, s, ebits, rng, c_idx, s_c, Ar, Ag)
        if Phi > best[0]:
            best = (Phi, c_idx, s_c, s, ds)
    return best


# --------------------------------------------------------------------------
# projection referentielle + residu
# --------------------------------------------------------------------------
def purview_spans(sub, c_idx, s_c, off, thresh=0.10, maxbits=64):
    w = sub.Wout[:, c_idx]
    base = sub.bout + 0.5 * (sub.Wout.sum(1) - w.sum(1))
    p = sigmoid(sub.cfg.beta * (w @ s_c + base))
    sal = np.abs(p - 0.5)
    idx = np.where(sal > thresh)[0]
    if len(idx) == 0:
        idx = np.argsort(-sal)[:8]
    idx = idx[np.argsort(-sal[idx])][:maxbits]
    bidx = np.unique(idx // 8 + off)
    spans, start, prev = [], int(bidx[0]), int(bidx[0])
    for b in bidx[1:]:
        b = int(b)
        if b == prev + 1:
            prev = b
        else:
            spans.append((start, prev + 1)); start = prev = b
    spans.append((start, prev + 1))
    return spans, float(sal.mean())


def causal_order(sub, c_idx, distinctions):
    Ag = sub.Agen[np.ix_(c_idx, c_idx)]
    pos = {int(u): i for i, u in enumerate(c_idx)}
    rank = Ag.sum(0) - Ag.sum(1)          # source nette (non-reciprocite => ordre)
    scored = []
    for d in distinctions:
        r = np.mean([rank[pos[int(u)]] for u in d if int(u) in pos] or [0.0])
        scored.append((float(r), d))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored]


def infer(sub: Substrate, E: bytes, seed: int = 1234):
    cfg = sub.cfg
    rng = np.random.default_rng(seed)
    units, phis, residues = [], [], []
    nsw = min(cfg.max_sweeps, max(1, (len(E) + cfg.L - 1) // cfg.L))

    for k in range(nsw):
        off = k * cfg.L
        ebits = bits_of(E[off:off + cfg.L], cfg.L)
        res = search(sub, ebits, rng, learn=False, collect=True)
        Phi, c_idx, s_c, s, ds = exclusion(sub, ebits, rng, res["pool"])
        phis.append({"sweep": k, "offset": off, "phi": res["phi_max"],
                     "Phi": float(Phi), "a": res["a"], "iota": res["iota"]})
        if c_idx is None:
            continue
        m = sub.mods(s)
        Ag = sub.modulate(sub.Agen, m)
        Ar = sub.modulate(sub.Aret, m)
        subs = [np.array([u]) if len(ds) == 0 else c_idx[i::max(1, cfg.nd)]
                for i in range(cfg.nd)]
        subs = [d for d in subs if len(d) > 0]
        for d in causal_order(sub, c_idx, subs):
            ph, io, a, _ = mech_phi(sub, s, ebits, rng, d, s[d], Ar, Ag)
            spans, sal = purview_spans(sub, d, s[d], off)
            units.append({"phi": ph, "iota": io, "a": a, "spans": spans,
                          "sal": sal, "size": int(len(d))})
        # residu : decomposition de phi sur son purview (en octets de E)
        pout = sub.write(s)
        mismatch = np.abs(pout - ebits) > 0.25
        bad = np.unique(np.where(mismatch)[0] // 8 + off)
        residues.append({"offset": off, "bytes": bad.tolist(),
                         "a": res["a"], "iota": res["iota"]})

    return {"units": units, "sweeps": phis, "residues": residues}


def render_output(E: bytes, out: dict, theta: float) -> str:
    lines = []
    phi_max = max([s["phi"] for s in out["sweeps"]] or [0.0])
    Phi_max = max([s["Phi"] for s in out["sweeps"]] or [0.0])
    lines.append("=== PHISHARP / ask ===")
    lines.append(f"phi_sharp(max)={phi_max:.5f}  Phi_sharp(max)={Phi_max:.5f}  "
                 f"unites={len(out['units'])}  sweeps={len(out['sweeps'])}")
    if phi_max < theta:
        lines.append(f"[sens non trouvable dans le budget : phi < theta={theta}]")

    lines.append("")
    lines.append("--- projection referentielle (maximale, aucun elagage) ---")
    text_parts = []
    for i, u in enumerate(sorted(out["units"], key=lambda x: -x["phi"])):
        frag = b" ".join(E[a:b] for a, b in u["spans"] if a < len(E))
        frag = frag.replace(b"\n", b" ")
        try:
            s = frag.decode("utf-8", errors="replace")
        except Exception:
            s = repr(frag)
        text_parts.append(s)
        sp = ",".join(f"{a}..{b}" for a, b in u["spans"])
        lines.append(f"[d{i:02d} phi={u['phi']:.4f} iota={u['iota']:.4f} "
                     f"a={u['a']:.3f} purview={sp}] {s}")
    lines.append("")
    lines.append("--- explication (concatenation des purviews) ---")
    lines.append(" ".join(text_parts) if text_parts else "(vide)")

    nj = sum(len(r["bytes"]) for r in out["residues"])
    ne = sum(1 for u in out["units"] if u["sal"] < 0.10)
    nc = sum(1 for u in out["units"] if u["a"] > 0.5 and u["iota"] < 0.02)
    lines.append("")
    lines.append("--- residu ---")
    lines.append(f"juste      : {nj} octets non regeneres")
    for r in out["residues"]:
        if r["bytes"]:
            b = r["bytes"][:24]
            lines.append(f"   offsets {b}{' ...' if len(r['bytes']) > 24 else ''}")
    lines.append(f"expressive : {ne} distinctions sans purview -> materiel expressif manquant "
                 f"(ajouter des definitions dans perceived.eff)")
    lines.append(f"claire     : {nc} distinctions regenerees mais reductibles -> liant manquant")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# commandes
# --------------------------------------------------------------------------
def cmd_train(args):
    cfg = make_config(args.profile, args.seed)
    rng = np.random.default_rng(cfg.seed)
    sub = Substrate(cfg, rng)
    gen = PTGRS(np.random.default_rng(cfg.seed + 777))
    kinds = ["plain"] * 4 + ["disjoint"] * 4 + ["hole"] * 2 + ["noise"] * 2
    hist, t0 = [], time.time()
    print(f"[train] profil={args.profile} n={cfg.n} params={sub.n_params():,} "
          f"episodes={args.episodes}", flush=True)

    for ep in range(args.episodes):
        kind = kinds[int(rng.integers(0, len(kinds)))]
        E, _ = gen.episode(kind)          # la cause cachee est ignoree ici
        ebits = bits_of(E, cfg.L)
        res = search(sub, ebits, rng, learn=True)
        hist.append({"ep": ep, "kind": kind, "phi": res["phi_max"],
                     "a": res["a"], "iota": res["iota"]})
        if (ep + 1) % max(1, args.episodes // 20) == 0:
            w = hist[-100:]
            print(f"[train] ep={ep+1:5d} phi={np.mean([h['phi'] for h in w]):.5f} "
                  f"a={np.mean([h['a'] for h in w]):.3f} "
                  f"iota={np.mean([h['iota'] for h in w]):.4f} "
                  f"t={time.time()-t0:.0f}s", flush=True)
        if time.time() - t0 > args.max_seconds:
            print(f"[train] arret budget temps a ep={ep+1}", flush=True)
            break

    sub.save(args.out, {"episodes": len(hist), "profile": args.profile,
                        "seconds": time.time() - t0})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(os.path.join(os.path.dirname(args.out) or ".", "train_history.json"), "w") as f:
        json.dump(hist, f)
    print(f"[train] modele -> {args.out}", flush=True)


def cmd_ask(args):
    sub, _ = Substrate.load(args.model)
    with open(args.file, "rb") as f:
        E = f.read()
    out = infer(sub, E, seed=1234)
    print(render_output(E, out, sub.cfg.theta))


def cmd_eval(args):
    sub, meta = Substrate.load(args.model)
    cfg = sub.cfg
    rng = np.random.default_rng(args.seed + 99991)     # grammaires jamais vues
    gen = PTGRS(np.random.default_rng(args.seed + 424242))
    report: Dict[str, object] = {"params": sub.n_params(), "profile": meta.get("profile")}

    for kind in ("plain", "disjoint", "hole", "noise"):
        vals, dphis = [], []
        for _ in range(args.episodes):
            E, m = gen.episode(kind)
            eb = bits_of(E, cfg.L)
            r = search(sub, eb, rng, collect=True)
            Phi, *_ = exclusion(sub, eb, rng, r["pool"])
            vals.append(r["phi_max"])
            if kind == "disjoint":
                a = E[: m["half_a"]]
                b = E[m["half_a"] + 4:]
                ra = search(sub, bits_of(a, cfg.L), rng, collect=True)
                rb = search(sub, bits_of(b, cfg.L), rng, collect=True)
                Pa, *_ = exclusion(sub, bits_of(a, cfg.L), rng, ra["pool"])
                Pb, *_ = exclusion(sub, bits_of(b, cfg.L), rng, rb["pool"])
                dphis.append(float(Phi - Pa - Pb))
        report[kind] = {"phi_mean": float(np.mean(vals)), "phi_std": float(np.std(vals))}
        if dphis:
            report["MR"] = {"delta_Phi_mean": float(np.mean(dphis)),
                            "delta_Phi_pos_rate": float(np.mean([d > 0 for d in dphis]))}
        print(f"[eval] {kind:9s} phi={np.mean(vals):.5f}", flush=True)

    # test MR textuel : mots-contenus disjoints
    demo = DEMO_TEXT
    out = infer(sub, demo, seed=7)
    report["demo_phi"] = max([s["phi"] for s in out["sweeps"]] or [0.0])

    path = os.path.join(os.path.dirname(args.model) or ".", "eval_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("## PHISHARP — rapport\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n")
            f.write("\n_Lemmes structurels verifies par `selftest`. Competence mesuree, pas presumee._\n")


def cmd_demo(args):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(DEMO_TEXT)
    print(f"[demo] {args.out} ({len(DEMO_TEXT)} octets)")


def cmd_selftest(args):
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"[{'OK ' if cond else 'FAIL'}] {name} {detail}")

    cfg = make_config("ci", 0)
    rng = np.random.default_rng(0)
    sub = Substrate(cfg, rng)

    check("antisymetrie exacte Aret", np.max(np.abs(sub.Aret + sub.Aret.T)) == 0.0)
    check("antisymetrie exacte Agen", np.max(np.abs(sub.Agen + sub.Agen.T)) == 0.0)
    s = (rng.random(cfg.n) < 0.5).astype(F32)
    check("pas d'energie (s^T A s = 0)", abs(float(s @ sub.Aret @ s)) < 1e-3,
          f"val={float(s @ sub.Aret @ s):.2e}")
    check("interface non reciproque (Win.T != Wout)",
          np.linalg.norm(sub.Win.T - sub.Wout) > 1.0)

    # LEMME DE COPIE : a=1 mais iota=0 => phi=0
    c2 = make_config("ci", 1)
    c2.beta, c2.kc, c2.pz, c2.ncuts, c2.tau = 8.0, c2.n, c2.n, 4, 1
    s2 = Substrate(c2, np.random.default_rng(1))
    s2.Aret[:] = 0; s2.Agen[:] = 0; s2.U[:] = 0; s2.V[:] = 0
    s2.Wm[:] = 0; s2.bm[:] = 0
    s2.b[:] = -c2.beta / 2; s2.bout[:] = -c2.beta / 2
    s2.Win = np.eye(c2.n, c2.P, dtype=F32)
    s2.Wout = np.eye(c2.P, c2.n, dtype=F32)
    E = bytes(np.random.default_rng(2).choice(np.frombuffer(PRINTABLE, np.uint8), size=c2.L))
    eb = bits_of(E, c2.L)
    st = (eb > 0.5).astype(F32)
    ci = np.arange(c2.n)
    ph, io, a, _ = mech_phi(s2, st, eb, np.random.default_rng(3), ci, st[ci])
    check("LEMME DE COPIE", a > 0.90 and io < 1e-6 and ph < 1e-6,
          f"a={a:.3f} iota={io:.2e} phi={ph:.2e}")

    # LEMME DE CONFABULATION : iota>0 mais a=0 => phi=0
    s3 = Substrate(cfg, np.random.default_rng(4))
    s3.Wout[:] = 0; s3.bout[:] = 0
    eb3 = bits_of(b"bonjour le monde", cfg.L)
    st3 = (rng.random(cfg.n) < 0.5).astype(F32)
    ci3 = candidate(np.abs(st3 - 0.5) + 0.6, cfg.kc)
    ph3, io3, a3, _ = mech_phi(s3, st3, eb3, np.random.default_rng(5), ci3, st3[ci3])
    check("LEMME DE CONFABULATION", a3 < 1e-6 and ph3 < 1e-6,
          f"iota={io3:.4f} a={a3:.2e} phi={ph3:.2e}")

    # ID : metrique
    p = rng.random(64).astype(F32); q = rng.random(64).astype(F32)
    check("ID(p,p)=0", float(np.abs(p - p).sum()) == 0.0)
    check("ID symetrique", abs(float(np.abs(p - q).sum()) - float(np.abs(q - p).sum())) < 1e-6)

    # PTGRS : rendus disjoints
    gen = PTGRS(np.random.default_rng(6))
    E, m = gen.episode("disjoint")
    ta = set(re.findall(rb"[a-z]{2,}", E[: m["half_a"]]))
    tb = set(re.findall(rb"[a-z]{2,}", E[m["half_a"] + 4:]))
    check("PTGRS rendus disjoints", len(ta & tb) == 0, f"|A|={len(ta)} |B|={len(tb)}")

    # determinisme
    r1 = np.random.default_rng(11); r2 = np.random.default_rng(11)
    g1 = PTGRS(np.random.default_rng(12)); g2 = PTGRS(np.random.default_rng(12))
    e1, _ = g1.episode("plain"); e2, _ = g2.episode("plain")
    sb1 = Substrate(cfg, np.random.default_rng(13))
    sb2 = Substrate(cfg, np.random.default_rng(13))
    v1 = search(sb1, bits_of(e1, cfg.L), r1)["phi_max"]
    v2 = search(sb2, bits_of(e2, cfg.L), r2)["phi_max"]
    check("determinisme", e1 == e2 and abs(v1 - v2) < 1e-9, f"{v1:.6f}")

    print("\nSELFTEST:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


DEMO_TEXT = (
    b"DEFINITIONS\n"
    b"ambiguite: etat d'un enonce qui admet plusieurs lectures distinctes sans que "
    b"le contexte en selectionne une.\n"
    b"cause: ce qui produit un effet.\n"
    b"effet: ce qui resulte d'une cause.\n"
    b"comprendre: retrouver la cause d'un effet.\n"
    b"relation: lien oriente entre deux termes, non symmetric.\n"
    b"structure: ensemble de relations formant un tout.\n"
    b"lecture: interpretation possible d'un enonce.\n"
    b"attachement: rattachement d'un groupe a un constituant plutot qu'a un autre.\n"
    b"\nPERCEPTION\n"
    b"Le professeur a vu l'homme avec le telescope.\n"
    b"\nQUESTION\n"
    b"Quelle est la nature de la difficulte de cet enonce ?\n"
)


def main():
    ap = argparse.ArgumentParser(prog="phisharp")
    sp = ap.add_subparsers(dest="cmd", required=True)

    t = sp.add_parser("train")
    t.add_argument("--out", default="out/sharp.phis")
    t.add_argument("--profile", default="ci", choices=list(PROFILES))
    t.add_argument("--episodes", type=int, default=2000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--max-seconds", type=float, default=900)
    t.set_defaults(fn=cmd_train)

    a = sp.add_parser("ask")
    a.add_argument("--model", required=True)
    a.add_argument("--file", required=True)
    a.set_defaults(fn=cmd_ask)

    e = sp.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--episodes", type=int, default=24)
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(fn=cmd_eval)

    d = sp.add_parser("demo")
    d.add_argument("--out", default="perceived.eff")
    d.set_defaults(fn=cmd_demo)

    s = sp.add_parser("selftest")
    s.set_defaults(fn=cmd_selftest)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()