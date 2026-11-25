"""Precompute real network outputs for the site's interactive explorer.

Scenes
  * standard map  p' = p + K sin x, x' = x + p'  (not in the training set; no ground truth), on a grid
    K x k (kick strength x number of orbits), the same nested initial conditions for every K;
  * a few validation maps of the clean bank set, k = 5..200 nested orbits (fixed permutation), with ground truth.

Every frame is painted exactly as in training (bank_paint.paint_inst_map + make_encoding, T_w = 1000) and run
through the paper's headline networks (instance + MeanShift decode as in the tables, binary, semantic on the
torus only).  Output per scene, packed for the browser:
  x/<scene>.png      atlas, frames on a grid (rows = K values, cols = k values), 256x256 each, RGB:
                     R = orbit id of the input pixel (0 = empty), G = predicted instance (0 = chaotic),
                     B = predicted semantic class (bits 0-6) | predicted chaotic (bit 7)
  x/<scene>_gt.png   ground truth (bank scenes): R = instance, G = semantic class, B = chaotic | separatrix<<1
  x/<scene>.json     frame metadata (scores, counts, 2-D PCA of the instance embeddings)
  x/scenes.json      index
Run (needs MLX, i.e. Metal access):  /opt/anaconda3/envs/poincare_ia/bin/python website/tools/build_explorer.py
"""
import json, os, sys, time
import numpy as np
from PIL import Image

ROOT = "/Users/vpeano/ownCloud/2026_poincare/poincare_ia"
sys.path.insert(0, ROOT)
import instance_metrics as IM
from bank_io import load_bank
from bank_paint import paint_inst_map, unpack_valid, make_encoding
from poincare_ia.model_io import load_unetr

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "x"); os.makedirs(OUT, exist_ok=True)
W = os.path.join(ROOT, "paper_checkpoints", "safetensors")
M, NE, BW, T_W, R = 30, 10, 1.4, 1000, 256
BANK_K = [5, 10, 20, 30, 50, 100, 150, 200]
STD_K = [10, 20, 30, 50, 100, 150, 200]
STD_KICK = [0.3, 0.6, 0.8, 0.97, 1.2, 1.5, 2.0, 3.0]
BANK = [("46295", "Oblique map on the torus"), ("1089", "Double-kick map on the torus"),
        ("50224", "Oblique map on the torus"), ("13176", "Double-kick map on the torus, many small chains"),
        ("4492", "Double-kick map on the plane"), ("63098", "Oblique map on the plane"),
        ("30301014", "Kerr map on the plane"), ("59826", "Oblique map on the cylinder")]

t0 = time.time()
net_i = load_unetr(f"{W}/headline_instance.safetensors", backend="mlx")
net_b = load_unetr(f"{W}/headline_binary.safetensors", backend="mlx")
net_s = load_unetr(f"{W}/headline_semantic.safetensors", backend="mlx")
print(f"models loaded {time.time() - t0:.0f}s", flush=True)


def run(inst, torus, seed):
    """inst: (R,R) orbit-id painting -> (pred instance, pred semantic or None, pred chaotic, embedding sample)."""
    img = make_encoding(int(inst.max()), M)[inst].astype(np.float32)
    o = net_i.predict(img)
    emb, fg = o[..., :NE], 1 / (1 + np.exp(-o[..., NE].astype(np.float64))) > 0.5
    pi = IM.decode(emb, fg, BW, decoder="meanshift", min_pixels=1, rng=np.random.default_rng(seed))
    ch = net_b.predict(img)[..., 0] > 0          # sigmoid > 1/2  <=> logit > 0 ; foreground = chaotic | separatrix
    sem = np.argmax(net_s.predict(img), -1) if torus else None
    # 2-D view of the embedding: PCA of the foreground pixels, at most 900 of them
    ys, xs = np.nonzero(pi > 0); pts = []
    if len(ys) >= 3:
        sel = np.random.default_rng(seed).choice(len(ys), min(900, len(ys)), replace=False)
        E = emb[ys[sel], xs[sel]].astype(np.float64); E -= E.mean(0)
        _, _, vt = np.linalg.svd(E, full_matrices=False); P = E @ vt[:2].T
        P = np.round(127.5 + 120 * P / max(1e-9, np.abs(P).max())).astype(int)
        lab = pi[ys[sel], xs[sel]]
        pts = np.stack([P[:, 0], P[:, 1], np.minimum(lab, 255)], 1).ravel().tolist()
    return pi, sem, ch, pts


def pack(inst, pi, sem, ch):
    sem = np.zeros_like(pi) if sem is None else np.minimum(sem, 127)
    return np.stack([np.minimum(inst, 255), np.minimum(pi, 255), sem | (ch.astype(np.int64) << 7)], -1).astype(np.uint8)


def counts(a):
    return int(len(np.unique(a[a > 0])))


index = []

# ── standard map ──
rng = np.random.default_rng(2026)
x0, p0 = rng.uniform(0, 2 * np.pi, max(STD_K)), rng.uniform(-np.pi, np.pi, max(STD_K))
atlas = np.zeros((len(STD_KICK) * R, len(STD_K) * R, 3), np.uint8)
frames, lyap = [], []
for a, K in enumerate(STD_KICK):
    x, p = x0.copy(), p0.copy(); dx, dp = np.ones_like(x), np.zeros_like(x); ls = np.zeros_like(x)
    X = np.empty((len(x), T_W)); P = np.empty_like(X)
    for n in range(T_W):
        c = np.cos(x)
        p = np.mod(p + K * np.sin(x) + np.pi, 2 * np.pi) - np.pi; x = np.mod(x + p, 2 * np.pi)
        dp = dp + K * c * dx; dx = dx + dp; nr = np.hypot(dx, dp); ls += np.log(nr); dx /= nr; dp /= nr
        X[:, n], P[:, n] = x, p
    lyap.append(np.round(ls / T_W, 4).tolist())
    bank = np.stack([np.minimum((X / (2 * np.pi) * R).astype(int), R - 1),
                     np.minimum(((np.pi - P) / (2 * np.pi) * R).astype(int), R - 1)], -1).astype(np.uint8)  # p up
    for b, k in enumerate(STD_K):
        inst = paint_inst_map(bank, np.arange(k), 0, T_W).astype(np.int64)
        pi, sem, ch, pts = run(inst, True, 1000 * a + b)
        atlas[a * R:(a + 1) * R, b * R:(b + 1) * R] = pack(inst, pi, sem, ch)
        frames.append(dict(a=a, b=b, n_pred=counts(pi), chaotic_frac=round(float(ch.mean()), 4), emb=pts))
    print(f"standard map K={K}  {time.time() - t0:.0f}s", flush=True)
Image.fromarray(atlas).save(f"{OUT}/standard.png", optimize=True)
json.dump(dict(id="standard", title="Standard map", sub="not in the training set · no ground truth",
               torus=True, gt=False, axisA=dict(name="K", values=STD_KICK), axisB=dict(name="k", values=STD_K),
               lyap=lyap, frames=frames), open(f"{OUT}/standard.json", "w"), separators=(",", ":"))
index.append(dict(id="standard", title="Standard map", sub="not in the training set", topo="pbc", family="standard"))

# ── validation maps ──
for sid, title in BANK:
    d = load_bank(f"{ROOT}/validation_dataset_general_bank_clean/dict_bank_{sid}.pkl")
    bank, valid = d["bank"], unpack_valid(d)
    pbc = tuple(bool(v) for v in np.asarray(d["params"]["pbc"]).tolist())
    topo = {(True, True): "pbc", (False, False): "nopbc"}.get(pbc, "cyl"); torus = topo == "pbc"
    fam = "kerr" if "n_period" in d["params"] else ("twist" if "modes_c" in d["params"] else "golden")
    g = d["ground_truth"]
    gi, gs = np.asarray(g[2]).astype(np.int64), np.minimum(np.asarray(g[3]).astype(np.int64), 29)
    gch, gsep = np.asarray(g[0]).astype(bool), np.asarray(g[1]).astype(bool)
    Image.fromarray(np.stack([np.minimum(gi, 255), gs, gch | (gsep.astype(np.int64) << 1)], -1).astype(np.uint8)
                    ).save(f"{OUT}/{sid}_gt.png", optimize=True)
    gt_reg = ~(gch | gsep)
    order = np.random.default_rng(int(sid)).permutation(bank.shape[0])
    atlas = np.zeros((R, len(BANK_K) * R, 3), np.uint8); frames = []
    for b, k in enumerate(BANK_K):
        inst = paint_inst_map(bank, order[:k], 0, T_W, valid=valid).astype(np.int64)
        pi, sem, ch, pts = run(inst, torus, b)
        atlas[:, b * R:(b + 1) * R] = pack(inst, pi, sem, ch)
        sc = IM.score_map(gi, pi); ag = IM.aggregate([sc])
        pr = ~ch; un = (pr | gt_reg).sum()
        f = dict(a=0, b=b, n_pred=counts(pi), tp=sc["tp"], fp=sc["fp"], fn=sc["fn"],
                 pq=round(ag["PQ"], 4) if counts(gi) else None, rq=round(ag["RQ"], 4), sq=round(ag["SQ"], 4),
                 reg_iou=round(float((pr & gt_reg).sum() / un), 4) if un else None, emb=pts)
        if torus:   # per-map mIoU over the classes present in the ground truth or the prediction
            cl = np.union1d(np.unique(gs), np.unique(sem))
            f["sem_miou"] = round(float(np.mean([((gs == c) & (sem == c)).sum() / ((gs == c) | (sem == c)).sum()
                                                  for c in cl])), 4)
        frames.append(f)
    Image.fromarray(atlas).save(f"{OUT}/{sid}.png", optimize=True)
    json.dump(dict(id=sid, title=title, sub=f"validation map #{sid}", torus=torus, topo=topo, family=fam, gt=True,
                   n_gt=counts(gi), axisA=None, axisB=dict(name="k", values=BANK_K), frames=frames),
              open(f"{OUT}/{sid}.json", "w"), separators=(",", ":"))
    index.append(dict(id=sid, title=title, sub=f"#{sid} · {counts(gi)} GT instances", topo=topo, family=fam))
    print(f"{sid} {topo} {fam}: PQ by k", [f["pq"] for f in frames], f"{time.time() - t0:.0f}s", flush=True)

json.dump(dict(t_w=T_W, train_k=[30, 200], train_tw=[300, 2000], scenes=index),
          open(f"{OUT}/scenes.json", "w"), indent=1)
print("done", time.time() - t0)
