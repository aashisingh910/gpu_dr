"""A5 - Bidirectional Pathology-Anatomy Cross-Attention / Graph Fusion.

The graph, the anatomy tokens and the edge types are unchanged.  The direction
of information flow is not.

Previously the stream was one-way: pathology queried anatomy, so anatomy could
inform lesions but lesions could never re-shape the anatomical representation.
That asymmetry costs exactly the cases DR grading turns on - the model could
learn "this is the macula" and "there is a haemorrhage", but the macula token
never learned that it *contains* one.

Now both directions run:

    Z_{P->A} = CA(Z_P, Z_A)        pathology attends to anatomy
    Z_{A->P} = CA(Z_A, Z_P)        anatomy attends to pathology
    Z_F      = Fusion(Z_P, Z_A, Z_{P->A}, Z_{A->P}, G)

and a shared probe P(.) ties the two views together:

    L_PA = || P(Z_P) - P(Z_A) ||_2^2

The consistency term is what makes the bidirectionality mean something: it
forces the pathology and anatomy streams to agree about severity, so a
haemorrhage seen by the lesion branch has to show up as an abnormal macula in
the anatomy branch as well.

    Z_G (global retinal tokens)   ---.
    Z_L (lesion expert tokens)    ---+--> X-Attn(G<->L) --> bidirectional P<->A
    Z_A (anatomical region tokens)---'          |
                                                v
                                    graph construction -> GNN -> Z_F

Anatomy tokens are not learned from nothing: they are the backbone's own patch
tokens **pooled over the real anatomical region maps** (optic disc, macula,
superior/inferior arcades, periphery) produced in `data/lesion_priors.py`.

The graph carries three edge types - feature kNN, spatial adjacency on the
patch grid, and explicit lesion->anatomy "is located in" edges - which is what
lets the model reason about *where* a lesion sits (a haemorrhage in the macula
matters more than one in the periphery).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import FusionCfg


@dataclass
class FusionOutput:
    z_fused: torch.Tensor         # (B, fused_dim)
    node_feats: torch.Tensor      # (B, n_nodes, fused_dim)
    adjacency: torch.Tensor       # (B, n_nodes, n_nodes)
    attn_g2l: torch.Tensor        # (B, heads, N_g, N_l) global<->lesion attention
    attn_p2a: torch.Tensor        # (B, heads, N_p, N_a) pathology -> anatomy
    attn_a2p: torch.Tensor        # (B, heads, N_a, N_p) anatomy -> pathology
    probe_p: torch.Tensor         # (B, probe_dim) P(Z_P)
    probe_a: torch.Tensor         # (B, probe_dim) P(Z_A)
    z_pathology: torch.Tensor     # (B, fused_dim) pooled pathology stream
    z_anatomy: torch.Tensor       # (B, fused_dim) pooled anatomy stream


def pathology_anatomy_consistency(out: "FusionOutput") -> torch.Tensor:
    """L_PA = || P(Z_P) - P(Z_A) ||_2^2, averaged over the batch.

    Both probes are L2-normalised first so the loss cannot be trivially
    minimised by shrinking both streams toward zero.
    """
    p = F.normalize(out.probe_p, dim=-1)
    a = F.normalize(out.probe_a, dim=-1)
    return (p - a).pow(2).sum(-1).mean()


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.h = heads
        self.dk = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, q_in: torch.Tensor, kv: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        B, Nq, D = q_in.shape
        Nk = kv.shape[1]
        x = self.n1(q_in)
        c = self.n1(kv)
        q = self.q(x).view(B, Nq, self.h, self.dk).transpose(1, 2)
        k = self.k(c).view(B, Nk, self.h, self.dk).transpose(1, 2)
        v = self.v(c).view(B, Nk, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        att = att.softmax(-1)
        out = (self.drop(att) @ v).transpose(1, 2).reshape(B, Nq, D)
        h = q_in + self.drop(self.o(out))
        h = h + self.ff(self.n2(h))
        return h, att


class GraphLayer(nn.Module):
    """Dense-adjacency graph attention layer (no torch-geometric dependency)."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.h = heads
        self.dk = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.o = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, N, self.h, self.dk).transpose(1, 2)
        k = k.view(B, N, self.h, self.dk).transpose(1, 2)
        v = v.view(B, N, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        mask = (adj <= 0).unsqueeze(1)                       # (B,1,N,N)
        att = att.masked_fill(mask, float("-inf"))
        att = att.softmax(-1).nan_to_num(0.0)
        out = (self.drop(att) @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.drop(self.o(out))
        return x + self.ff(self.norm2(x))


def build_adjacency(nodes: torch.Tensor, grid: int, n_global: int,
                    n_lesion: int, n_anat: int, knn: int,
                    lesion_in_region: torch.Tensor | None = None,
                    n_spatial: int | None = None) -> torch.Tensor:
    """Construct the fusion graph adjacency (B, N, N), binary, self-looped.

    `n_spatial` is the count of leading nodes that live on the pooled patch
    grid.  With the local branch enabled the pathology stream is
    `grid*grid` global tokens followed by one token per lesion crop, and the
    crop tokens have no grid position - so the 4-neighbour spatial edges must
    be built over the first `n_spatial` nodes only, not over all of them.
    """
    B, N, _ = nodes.shape
    dev = nodes.device
    adj = torch.zeros(B, N, N, device=dev)

    # 1) feature-similarity kNN over every node
    z = F.normalize(nodes, dim=-1)
    sim = z @ z.transpose(1, 2)
    k = min(knn + 1, N)
    idx = sim.topk(k, dim=-1).indices
    adj.scatter_(2, idx, 1.0)

    # 2) 4-neighbour spatial adjacency among the pooled global tokens
    ns = n_global if n_spatial is None else n_spatial
    if ns == grid * grid:
        ar = torch.arange(ns, device=dev)
        r, c = ar // grid, ar % grid
        dr = (r[:, None] - r[None, :]).abs()
        dc = (c[:, None] - c[None, :]).abs()
        spatial = ((dr + dc) == 1).float()
        adj[:, :ns, :ns] = torch.maximum(adj[:, :ns, :ns], spatial.unsqueeze(0))

    # 3) lesion <-> anatomy "is located in" edges from the real region overlap
    ls, le = n_global, n_global + n_lesion
    as_, ae = le, le + n_anat
    if lesion_in_region is not None:
        rel = (lesion_in_region > 0.02).float()              # (B, n_lesion, n_anat)
        adj[:, ls:le, as_:ae] = torch.maximum(adj[:, ls:le, as_:ae], rel)
        adj[:, as_:ae, ls:le] = torch.maximum(adj[:, as_:ae, ls:le], rel.transpose(1, 2))
    else:
        adj[:, ls:le, as_:ae] = 1.0
        adj[:, as_:ae, ls:le] = 1.0

    # lesion and anatomy nodes are hubs: connect them to all global tokens
    adj[:, ls:ae, :n_global] = 1.0
    adj[:, :n_global, ls:ae] = 1.0

    eye = torch.eye(N, device=dev).unsqueeze(0)
    return torch.maximum(adj, eye)


class PathologyAnatomyFusion(nn.Module):
    def __init__(self, cfg: FusionCfg, global_dim: int, lesion_dim: int,
                 expert_dim: int, n_experts: int, pool_grid: int = 7):
        super().__init__()
        self.cfg = cfg
        self.pool_grid = pool_grid
        d = cfg.fused_dim
        self.n_anat = len(cfg.anatomy_regions)
        self.n_lesion = n_experts

        self.proj_g = nn.Linear(global_dim, d)
        # local lesion crops enter the pathology stream as one token each, with
        # their own type embedding so the graph can tell a high-resolution
        # window apart from a pooled global patch
        self.proj_local = nn.Linear(global_dim, d)
        self.local_type = nn.Parameter(torch.zeros(1, 1, d))
        self.proj_l = nn.Linear(expert_dim, d)
        self.proj_a = nn.Linear(global_dim, d)
        self.lesion_ctx = nn.Linear(lesion_dim, d)

        self.xattn_gl = CrossAttention(d, cfg.n_heads)
        # the two directions get their own parameters: "which anatomy does this
        # lesion sit in" and "which lesions does this region contain" are
        # different questions and should not share a projection
        self.xattn_p2a = CrossAttention(d, cfg.n_heads)
        self.xattn_a2p = CrossAttention(d, cfg.n_heads)
        self.gnn = nn.ModuleList([GraphLayer(d, heads=4) for _ in range(cfg.gnn_layers)])
        self.readout = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())
        # Fusion(Z_P, Z_A, Z_P->A, Z_A->P, G): four streams + the graph readout
        self.stream_mix = nn.Sequential(
            nn.LayerNorm(5 * d), nn.Linear(5 * d, d), nn.GELU(), nn.Linear(d, d))
        # shared probe P(.) used by the L_PA consistency term - deliberately one
        # module applied to both streams, so agreement is measured in a common space
        self.probe = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.probe_dim))
        self.type_emb = nn.Parameter(torch.zeros(3, d))
        nn.init.trunc_normal_(self.type_emb, std=0.02)

    def _anatomy_tokens(self, tokens: torch.Tensor, grid: int,
                        anat_maps: torch.Tensor) -> torch.Tensor:
        """Pool backbone patch tokens inside each anatomical region map."""
        B, N, D = tokens.shape
        maps = F.adaptive_avg_pool2d(anat_maps.float(), (grid, grid))   # (B,R,g,g)
        w = maps.flatten(2)                                            # (B,R,N)
        w = w / (w.sum(-1, keepdim=True) + 1e-6)
        return torch.bmm(w, tokens)                                    # (B,R,D)

    def _pool_global(self, tokens: torch.Tensor, grid: int) -> torch.Tensor:
        B, N, D = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, D, grid, grid)
        x = F.adaptive_avg_pool2d(x, (self.pool_grid, self.pool_grid))
        return x.flatten(2).transpose(1, 2)                            # (B,P*P,D)

    def forward(self, tokens: torch.Tensor, grid: int, expert_tokens: torch.Tensor,
                z_lesion: torch.Tensor, anat_maps: torch.Tensor,
                lesion_evidence: torch.Tensor | None = None,
                local_cls: torch.Tensor | None = None) -> FusionOutput:
        B = tokens.shape[0]
        anat_tok = self.proj_a(self._anatomy_tokens(tokens, grid, anat_maps))
        g_tok = self.proj_g(self._pool_global(tokens, grid))
        n_spatial = g_tok.shape[1]
        if local_cls is not None:
            g_tok = torch.cat(
                [g_tok, self.proj_local(local_cls) + self.local_type], dim=1)
        l_tok = self.proj_l(expert_tokens) + self.lesion_ctx(z_lesion).unsqueeze(1)

        g_tok = g_tok + self.type_emb[0]
        l_tok = l_tok + self.type_emb[1]
        anat_tok = anat_tok + self.type_emb[2]

        # cross-attention 1: global <-> lesion  (the pathology stream)
        path_tok, attn_gl = self.xattn_gl(g_tok, l_tok)

        # cross-attention 2+3: pathology <-> anatomy, both directions
        p2a, attn_p2a = self.xattn_p2a(path_tok, anat_tok)      # Z_{P->A}
        if self.cfg.bidirectional:
            a2p, attn_a2p = self.xattn_a2p(anat_tok, path_tok)  # Z_{A->P}
        else:
            a2p, attn_a2p = anat_tok, attn_p2a.transpose(-2, -1)

        # lesion-in-region overlap drives the semantic edges
        lir = None
        if lesion_evidence is not None:
            le = torch.sigmoid(lesion_evidence)                         # (B,L,S,S)
            am = F.interpolate(anat_maps.float(), size=le.shape[-2:],
                               mode="bilinear", align_corners=False)    # (B,R,S,S)
            lir = torch.einsum("blhw,brhw->blr", le, am) / (le.shape[-1] * le.shape[-2])

        nodes = torch.cat([p2a, l_tok, a2p], dim=1)
        adj = build_adjacency(nodes, self.pool_grid, p2a.shape[1], self.n_lesion,
                              self.n_anat, self.cfg.knn, lir, n_spatial=n_spatial)
        for layer in self.gnn:
            nodes = layer(nodes, adj)
        g_read = self.readout(nodes.mean(1))                            # G

        # Fusion(Z_P, Z_A, Z_P->A, Z_A->P, G)
        z_p = path_tok.mean(1)
        z_a = anat_tok.mean(1)
        z_f = self.stream_mix(torch.cat(
            [z_p, z_a, p2a.mean(1), a2p.mean(1), g_read], dim=-1))

        return FusionOutput(z_fused=z_f, node_feats=nodes, adjacency=adj,
                            attn_g2l=attn_gl, attn_p2a=attn_p2a, attn_a2p=attn_a2p,
                            probe_p=self.probe(z_p), probe_a=self.probe(z_a),
                            z_pathology=z_p, z_anatomy=z_a)
