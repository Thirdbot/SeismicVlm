"""Attribute registry — the single source of truth for object CLASSES and TIER-2 derived attributes.

Tier-1 primitives (dip/throw/area/count) are read directly by their own reader heads. Tier-2
attributes are DERIVED after that read by ONE query-conditioned head (derived_head.DerivedHead),
SECTION- or OBJECT-scoped: the only difference is the context fed in — the section pool
(section-scoped: intersect/mode/nclosure/nonlap/salt) vs. the reader's per-object hidden state h_i
(object-scoped: closure fluid/intersects_*). The list POSITION is the head's query index (shared
across both scopes), so adding a derived attribute is ONE row here — never a new head.

Add a derived attribute = add ONE row to DERIVED. scenes.py routes its GT, reader.py supervises that
query index, narrator.py injects/states it via the marker word.
"""
# ---- Object classes — keyed off the region's `object_type` (the consistent semantic name), NOT the
# numeric class_id. name -> reader class id (reader.py: NO_OBJ=0). Add a class = one row here + widen
# reader.class_head/obj_cls. The name is also the word the narrator states ("fault"/"closure"/…).
CLASS_ID = {"fault": 1, "closure": 2, "salt": 3, "onlap": 4}
ID_CLASS = {v: k for k, v in CLASS_ID.items()}          # reader id -> class word (narration)
N_CLASS = 1 + len(CLASS_ID)                             # incl. NO_OBJ=0 -> reader.class_head width (=5)
AREA_CLASSES = {2, 3, 4}                                # classes whose tier-1 measure is area (%)

# ACTIVE_CLASSES — restrict this multiclass model to a subset of object classes for a focused pass
# (e.g. ACTIVE_CLASSES=fault for fault-only interpretation, or dropping 'salt' on a corpus that has none).
# Default = ALL classes (None → no restriction). Consumed by reader.detect (argmax is masked to these
# classes) and reader.add_real_adapter (only these class-head rows receive gradient). NO_OBJ (∅) is
# ALWAYS active. Accepts class words or ids, comma/space separated.
import os as _os
def _resolve_active_classes():
    env = _os.environ.get("ACTIVE_CLASSES", "").strip()
    if not env:
        return None                                    # all classes active — no restriction
    ids = set()
    for tok in env.replace(",", " ").split():
        if tok in CLASS_ID:
            ids.add(CLASS_ID[tok])
        elif tok.isdigit() and int(tok) in ID_CLASS:
            ids.add(int(tok))
    return ids or None
ACTIVE_CLASS_IDS = _resolve_active_classes()           # None = all; else set of active object-class ids

# ============================================================================================
# TIER-1 MEASURE attributes — the class-driven per-object measurements. name -> (input kind, scale):
#   "spatial" reads the footprint's 2nd-moment stats (angle-preserving → dip);
#   "pooled"  reads the pooled feature (magnitudes → throw, area).
# The reader builds ONE head per measure from this dict (nn.ModuleDict), shared across every class that
# declares the measure. Add a measure = one row here (+ its dataset key + which classes carry it).
# ============================================================================================
MEASURE = {
    "dip":   ("spatial", 90.0),
    "throw": ("pooled",  500.0),
    "area":  ("spatial", 100.0),   # extent: reads the occupancy-fraction stat (a POOLED convex combination cannot encode size)
}
MEASURE_KEY = {"dip": "dip_deg", "throw": "throw", "area": "area_pct"}   # head name -> values.measure key
MEASURE_SLOTS = list(MEASURE.keys())                                    # fixed slot order for the meas/mmask vector
MEASURE_SCALE = [MEASURE[n][1] for n in MEASURE_SLOTS]                  # per-slot scale (dip 90 · throw 500 · area 100)

# CLASS SCHEMA — the mapping the model architecture is built from: object_type -> the tier-1 measures it
# carries. Class 0 = "none"/background (no attributes). Presence is still data-gated (a class may declare
# a measure the dataset doesn't fill → simply not supervised), so this is safe for ANY dataset
# composition (fault-only, fault+closure, …). Regenerate from a CSV with schema_from_csv() below.
CLASS_SCHEMA = {
    "fault":   ["dip", "throw"],
    "closure": ["area"],
    "salt":    ["area"],
    "onlap":   ["area"],
}
NONE_CLASS = 0                                                # preserved no-object class (no attributes)


def measures_for_id(cid):
    """reader class id -> its tier-1 measures (empty for NONE_CLASS / unknown)."""
    return CLASS_SCHEMA.get(ID_CLASS.get(int(cid)), [])


def schema_from_csv(csv_path, max_rows=None):
    """Scan a dataset CSV and return {object_type: {"measure": set, "derive": set}} — the DYNAMIC map
    from the data. Use it to (re)generate CLASS_SCHEMA / DERIVED when the dataset changes; the model
    itself reads the static dicts above (design-time mapping), this just derives them from real data."""
    import ast
    import collections
    import json
    import pandas as pd
    out = collections.defaultdict(lambda: {"measure": set(), "derive": set()})

    def parse(v):
        if isinstance(v, list):
            return v
        if not isinstance(v, str):
            return []
        try:
            return json.loads(v)
        except Exception:
            try:
                return ast.literal_eval(v)
            except Exception:
                return []
    df = pd.read_csv(csv_path)
    for cell in df["regions"].dropna().head(max_rows) if max_rows else df["regions"].dropna():
        for reg in parse(cell):
            if not isinstance(reg, dict):
                continue
            ot = reg.get("object_type")
            v = reg.get("values") or {}
            out[ot]["measure"].update((v.get("measure") or {}).keys())
            out[ot]["derive"].update((v.get("derive") or {}).keys())
    return {k: {"measure": sorted(v["measure"]), "derive": sorted(v["derive"])} for k, v in out.items()}


# ---- Categorical label sets ----
FAULT_MODES = ["horst_and_graben", "self_branching", "relay_ramp", "stair_case"]
MODE_WORD = {"horst_and_graben": "horst and graben", "self_branching": "self-branching",
             "relay_ramp": "relay ramp", "stair_case": "staircase"}
FLUID_LABELS = ["gas", "oil", "brine"]
BOOL_WORD = {True: "yes", False: "no"}

SECTION, OBJECT = "section", "object"

# ---- Unified tier-2 DERIVED registry (ONE head, both scopes). Query index = position in this list.
# (values.derive key · LM marker word · kind · labels · scope · klass[object-scoped only]) ----
DERIVED = [
    ("number_fault_intersections", "intersect",        "scalar", None,         SECTION, None),
    ("fault_mode",                 "mode",             "cat",    FAULT_MODES,  SECTION, None),
    ("number_hc_closures",         "nclosure",         "scalar", None,         SECTION, None),
    ("number_onlap_episodes",      "nonlap",           "scalar", None,         SECTION, None),
    ("salt_inserted",              "salt",             "bool",   None,         SECTION, None),
    ("fluid",                      "fluid",             "cat",    FLUID_LABELS, OBJECT,  "closure"),
    ("intersects_fault",           "intersects_fault",  "bool",   None,         OBJECT,  "closure"),
    ("intersects_salt",            "intersects_salt",   "bool",   None,         OBJECT,  "closure"),
    ("intersects_onlap",           "intersects_onlap",  "bool",   None,         OBJECT,  "closure"),  # dataset key: intersects_onlap (NOT onlap)
]
NUM_DERIVED = len(DERIVED)
MAX_CAT = max((len(r[3]) for r in DERIVED if r[3]), default=1)   # widest categorical label set

# Scalar regression scales (keep the target ~O(1); value/scale in). Markers absent here use 1.0.
SCALAR_SCALE = {"intersect": 50.0, "nclosure": 10.0, "nonlap": 10.0}

# Scope views for reader/derived_head: (query_idx, key, marker, kind, labels[, klass]).
SECTION_DERIVED = [(i, key, marker, kind, labels)
                   for i, (key, marker, kind, labels, sc, kl) in enumerate(DERIVED) if sc == SECTION]
OBJECT_DERIVED = [(i, key, marker, kind, labels, kl)
                  for i, (key, marker, kind, labels, sc, kl) in enumerate(DERIVED) if sc == OBJECT]


def derived_facts(der):
    """SECTION-scoped scene derived {intersect,mode,nclosure,nonlap,salt} (reader/GT numeric form) ->
    LM fact form {marker: readable value}. The categorical index becomes the word the narration copies;
    the bool becomes yes/no; scalars stay ints. Present-only (a missing attribute is skipped)."""
    if not der:
        return {}
    out = {}
    if der.get("intersect") is not None:
        out["intersect"] = int(der["intersect"])
    if der.get("mode") is not None:
        out["mode"] = MODE_WORD[FAULT_MODES[int(der["mode"])]]
    if der.get("nclosure") is not None:
        out["nclosure"] = int(der["nclosure"])
    if der.get("nonlap") is not None:
        out["nonlap"] = int(der["nonlap"])
    if der.get("salt") is not None:
        out["salt"] = BOOL_WORD[bool(der["salt"])]
    return out


def object_derived_facts(raw):
    """Per-OBJECT (closure) derived — RAW dataset values.derive dict (keys = dataset keys, e.g.
    {'fluid':'gas','intersects_fault':True}) -> LM fact form {marker: readable value}. Categoricals
    stay their label word, bools become yes/no. Present-only. Mirrors the reader's decoded output so
    the GT-injection path (region_metadata) and the reader path (detect) inject identical marker words."""
    if not raw:
        return {}
    out = {}
    for _i, key, marker, kind, labels, _kl in OBJECT_DERIVED:
        v = raw.get(key)
        if v is None:
            continue
        if kind == "cat":
            out[marker] = v if isinstance(v, str) else labels[int(v)]
        elif kind == "bool":
            out[marker] = BOOL_WORD[bool(v)]
        else:
            out[marker] = int(v)
    return out
