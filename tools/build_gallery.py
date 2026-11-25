"""Export the full validation galleries for the project website.

Reads the gallery folders at the repository root (the same ones packed into the Edmond deposit) and writes
  website/g/{instance,semantic,binary}/{dense,sparse}/<sid>.webp
  website/g/completion/<sid>.webp
  website/g/pretraining/<sid>.webp
  website/gallery.json
PNG panels are stored as lossless WebP (pixel-exact, about 30 KB each); the pretraining-comparison panels,
which are JPEG at the source, as lossy WebP (quality 88).  Existing outputs are skipped, so reruns are cheap.

Run from anywhere:  /opt/anaconda3/envs/poincare_ia/bin/python website/tools/build_gallery.py
"""
import json, os
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

WEB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(WEB)
DENSE, SPARSE = "dense (150 orbits, T_w = 1000)", "sparse (30 orbits, T_w = 500)"
DOI = "10.17617/3.NBUIL8"


def convert(job):
    src, dst = job
    p = os.path.join(WEB, dst)
    if os.path.exists(p):
        return
    os.makedirs(os.path.dirname(p), exist_ok=True)
    im = Image.open(src)
    if src.lower().endswith(".png"):
        im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB").save(p, "WEBP", lossless=True, method=4)
    else:
        im.convert("RGB").save(p, "WEBP", quality=88, method=6)


def main():
    load = lambda g: {d["sid"]: d for d in json.load(open(f"{ROOT}/{g}/maps.json"))}
    inst, sem, binr, comp = (load(g) for g in ("gallery_instance", "gallery_semantic", "gallery_binary", "gallery_completion"))
    arms = {d["sid"]: d for d in json.load(open(f"{ROOT}/instance_gallery_pq_nofilter/maps.json"))["maps"]}
    r3 = lambda x: None if x is None else round(float(x), 3)
    jobs, maps = [], []
    for s in sorted(inst, key=lambda s: int(s)):
        d = inst[s]
        e = {"sid": s, "family": d["family"], "topo": d["topo"]}
        for key, src, gname, keys in (("instance", inst, "gallery_instance", ["pq", "rq", "sq", "n_gt", "n_pred", "tp", "fp", "fn"]),
                                      ("semantic", sem, "gallery_semantic", ["miou", "n_gt", "n_pred"]),
                                      ("binary", binr, "gallery_binary", ["reg_iou", "sep_recall"])):
            if s not in src:
                continue
            e[key] = {}
            for k, lab in (("dense", DENSE), ("sparse", SPARSE)):
                v = src[s]["v"][lab]
                dst = f"g/{key}/{k}/{s}.webp"
                jobs.append((f"{ROOT}/{gname}/{v['img']}", dst))
                e[key][k] = {"img": dst, "m": {kk: r3(v["m"].get(kk)) for kk in keys}}
        if s in comp:
            c = comp[s]; dst = f"g/completion/{s}.webp"
            jobs.append((f"{ROOT}/gallery_completion/{c['img']}", dst))
            e["completion"] = {"img": dst, "m": {k: r3(c["m"][k]) for k in ("fg_iou", "reg_mse", "chaos_mse", "chaotic")}}
        if s in arms:
            a = arms[s]; dst = f"g/pretraining/{s}.webp"
            jobs.append((f"{ROOT}/instance_gallery_pq_nofilter/img/{s}.jpg", dst))
            e["pretraining"] = {"img": dst, "n_gt": a["n_gt"],
                                "arms": {k: {"PQ": r3(v["PQ"]), "n_pred": v["n_pred"]} for k, v in a["arms"].items()}}
        maps.append(e)
    with ThreadPoolExecutor(8) as ex:   # threads: PIL releases the GIL while encoding
        list(ex.map(convert, jobs))
    json.dump({"source": f"Edmond deposit galleries, doi:{DOI} (all validation maps)", "maps": maps},
              open(os.path.join(WEB, "gallery.json"), "w"), separators=(",", ":"))
    print(len(maps), "maps,", len(jobs), "images")


if __name__ == "__main__":
    main()
