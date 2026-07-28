"""Object-scoped derived head (REFERENCE — activate on reader rebuild for the new dataset).

ONE query-conditioned head for ALL tier-2 derived attributes, SECTION- or OBJECT-scoped. The only
difference between the two scopes is the CONTEXT vector fed in: the section pool (section-scoped:
intersect, mode, nclosure, salt) vs. the reader's per-object hidden state h_i (object-scoped: closure
fluid, intersects_*). Same head, same non-differentiable copy rail (categorical → word token, scalar →
digit token) — adding an attribute is a registry row + a query-embedding index, never a new head.

NOT wired into the live reader yet: instantiating it with the new attribute count changes the reader
state-dict, so it needs a retrain on the new dataset (schema v3 already nests fluid/intersects_* under
region.derive). This module is the drop-in the rebuild uses.
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


@torch.no_grad()
def decode(out, kind, labels=None):
    """Head output → the LM fact form the narration copies. cat → label WORD, bool → yes/no, scalar → int."""
    if kind == "cat":
        return labels[int(out.argmax(-1))]
    if kind == "bool":
        return "yes" if float(torch.sigmoid(out)) > 0.5 else "no"
    return int(round(float(out)))


def derive_all(head, section_ctx, obj_h, obj_classes, section_reg, object_reg):
    """Run the registry over both scopes with the ONE head. Returns section-derived dict + a per-object
    derived dict (object-scoped attrs attached only to objects of their class).

    section_reg = [(name_id, marker, kind, labels)]      (SECTION-scoped entries)
    object_reg  = [(name_id, marker, kind, labels, klass)] (OBJECT-scoped entries)
    """
    section = {}
    for name_id, marker, kind, labels in section_reg:
        nid = torch.tensor([name_id], device=section_ctx.device)
        section[marker] = decode(head(section_ctx.unsqueeze(0), nid, kind)[0], kind, labels)
    objects = [dict() for _ in obj_classes]
    for i, cls in enumerate(obj_classes):
        for name_id, marker, kind, labels, klass in object_reg:
            if cls == klass:                                  # object-scoped attr only for its class
                nid = torch.tensor([name_id], device=obj_h.device)
                objects[i][marker] = decode(head(obj_h[i].unsqueeze(0), nid, kind)[0], kind, labels)
    return section, objects
