import glob, os, time, zlib
import numpy as np

import prahari_config as cfg
from logmel import log_mel, quantise, dequantise

HOP = 12
DEAD_S = 3.0
TARGET_FA_H = 0.5
GRID = np.round(np.arange(0.50, 0.995, 0.01), 2)


def half_of(path):
    return zlib.crc32(("fa-half:" + os.path.basename(path)).encode("utf-8")) & 1


def windows(pcm):
    f = dequantise(quantise(log_mel(pcm))).astype(np.float16)
    idx = range(0, f.shape[0] - cfg.CTX_FRAMES + 1, HOP)
    return np.stack([f[s:s + cfg.CTX_FRAMES] for s in idx]) if len(idx) else \
        np.zeros((0, cfg.CTX_FRAMES, cfg.MEL_BINS), np.float16)


def build_select(root, minutes=15.0):
    import fa_eval as fe
    path = os.path.join(root, "features", f"fa_select_{int(minutes)}m.npz")
    xpath = path[:-4] + "_X.npy"
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        if not os.path.exists(xpath):
            np.save(xpath, z["X"])
        return (np.load(xpath, mmap_mode="r"), z["src"], [str(s) for s in z["names"]], z["hours"])
    t0 = time.time()
    noise = fe.board_noise_tails()
    nz = np.concatenate(noise).astype(np.float32) if noise else None
    Xs, src, names, hours = [], [], [], []
    for name, paths in fe.eval_lists(root).items():
        sel = [p for p in paths if half_of(p) == 0]
        x = fe.stream(sel, fe.LEVEL.get(name, fe.BOARD_SPEECH_RMS), minutes * 60)
        if not len(x):
            continue
        if nz is not None:
            x = np.clip(x.astype(np.float32) + np.resize(nz, len(x)), -32768, 32767).astype(np.int16)
        w = windows(x)
        Xs.append(w); src.append(np.full(len(w), len(names), np.int16))
        names.append(name); hours.append(len(x) / 16000 / 3600)
        print(f"  fa-select {name:<14} {hours[-1]*60:5.1f} min  {len(w):7,} windows", flush=True)
    if not Xs:
        return None
    X, src, hours = np.concatenate(Xs), np.concatenate(src), np.array(hours)
    np.save(xpath, X)
    np.savez(path, X=np.zeros((0,), np.float16), src=src, names=np.array(names), hours=hours)
    X = np.load(xpath, mmap_mode="r")
    print(f"  fa-select stream: {hours.sum():.2f} h, built in {time.time()-t0:.0f} s -> {path}")
    return X, src, names, hours


def predict(model, X, bs=4096):
    out = []
    for i in range(0, len(X), bs):
        out.append(np.asarray(model(X[i:i + bs].astype(np.float32)[..., None], training=False)).ravel())
    return np.concatenate(out) if out else np.zeros(0)


def fires_at(a, thr, rule="peak2"):
    dead = int(DEAD_S / (HOP * 0.01))
    v = np.maximum(a[1:], a[:-1]) if rule == "peak2" else np.minimum(a[1:], a[:-1])
    hit = np.flatnonzero(v >= thr)
    n, nxt = 0, -1
    for i in hit:
        if i >= nxt:
            n += 1; nxt = i + dead
    return n


def fa_per_hour(p, src, hours, thr, rule="peak2"):
    tot = sum(fires_at(p[src == k], thr, rule) for k in range(len(hours)))
    return tot / float(np.sum(hours))


def threshold_for_target(p, src, hours, rule="peak2", target=TARGET_FA_H):
    best, best_f = None, None
    for t in GRID[::-1]:
        f = fa_per_hour(p, src, hours, float(t), rule)
        if f > target:
            break
        best, best_f = float(t), f
    if best is None:
        return None, fa_per_hour(p, src, hours, float(GRID[-1]), rule)
    return best, best_f
