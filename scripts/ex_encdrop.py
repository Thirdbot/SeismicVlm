# exp_encoder_dropin.py
import os, json, torch
os.environ.setdefault("SFM_CKPT", "hybrid/checkpoints/SFM-Base-512.pth")
from hybrid.model.reader import RegionReader
from hybrid.stages.stage2_reader import _build_encoder
from hybrid.eval.benchmark import bench                       # EXISTING benchmark fn — no new metric code
device = torch.device("cuda")
ENC, READER, DATASET = os.environ.get("ENCODER","sfm").lower(), os.environ["READER"], os.environ["DATASET"]
r = RegionReader().to(device); sd = torch.load(READER, map_location=device)
if any(k.startswith("real_adapter") for k in sd): r.add_real_adapter()
r.load_state_dict(sd); r.eval()
if ENC == "ncs":
    from hybrid.model.encoder import NcsEncoder
    r.set_encoder(NcsEncoder().to(device).eval())             # SWAP: different pretrained 768-dim encoder
else:
    r.set_encoder(_build_encoder())                           # baseline: SFM-512
try:
    m = bench(r, DATASET)                                     # runs end-to-end → contract holds if this returns
    print(f"[{ENC}] RUNS OK · pooled_iou {m['pooled_iou']:.3f} · detF1 {m['det'][2]:.3f} · "
          f"deploy-Dice {m['ddice']:.3f} · class {m['cls'][0]}/{m['cls'][1]}")
    print(f"[{ENC}] FULL {json.dumps({k:(list(v) if isinstance(v,tuple) else v) for k,v in m.items()})}")
except Exception as e:
    import traceback; traceback.print_exc(); print(f"[{ENC}] FAILED end-to-end: {type(e).__name__}: {e}")