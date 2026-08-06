"""Object-scoped derived head (REFERENCE — activate on reader rebuild for the new dataset).

ONE query-conditioned head for ALL tier-2 derived attributes, SECTION- or OBJECT-scoped. The only
difference between the two scopes is the CONTEXT vector fed in: the section pool (section-scoped:
intersect, mode, nclosure, salt) vs. the reader's per-object hidden state h_i (object-scoped: closure
fluid, intersects_*). Same head, same non-differentiable copy rail (categorical → word token, scalar →
digit token) — adding an attribute is a registry row + a query-embedding index, never a new head.

DerivedHead IS wired into the live reader (reader.py imports + instantiates it; the reader runs the
registry over both scopes inline). Schema nests fluid/intersects_* under region.derive.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DerivedHead(nn.Module):
    """context (h_i for OBJECT scope, or section pool for SECTION scope) + per-attribute query → value.

    n_names = total derived attributes across BOTH scopes (each gets a query-embedding row).
    max_cat = widest categorical label set. Output branch is picked per attribute by its `kind`.
    """

    def __init__(self, d, n_names, max_cat):
        super().__init__()
        self.query = nn.Embedding(n_names, d)                 # a learned query vector per registry name
        self.trunk = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
        self.cat = nn.Linear(d, max_cat)                     # categorical (fluid, mode)
        self.scalar = nn.Linear(d, 1)                        # scalar / count (intersect, nclosure)
        self.boolean = nn.Linear(d, 1)                       # bool (intersects_*, salt)

    def forward(self, ctx, name_id, kind):
        """ctx: (B, d) context. name_id: (B,) registry index. kind: 'cat' | 'scalar' | 'bool'.
        Returns logits (cat), value (scalar), or logit (bool)."""
        h = self.trunk(torch.cat([ctx, self.query(name_id)], dim=-1))
        if kind == "cat":
            return self.cat(h)                               # (B, max_cat) — argmax → label word
        if kind == "bool":
            return self.boolean(h).squeeze(-1)               # (B,) — sigmoid > .5 → yes/no
        return self.scalar(h).squeeze(-1)                    # (B,) — round → count/scalar


# NOTE: the reader runs this registry INLINE (reader._decode + the object-derived loop) rather than calling
# a module-level helper here — the standalone decode()/derive_all() reference impls were unused and removed.
