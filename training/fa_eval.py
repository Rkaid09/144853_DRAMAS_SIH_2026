import csv, glob, json, os, sys, wave
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prahari_config as cfg
from logmel import log_mel, quantise, dequantise

ROOT = os.path.join("..")
CAP = os.path.join(ROOT, "server", "captures")
CV_EVAL = os.path.join(ROOT, "tools", "dataset_cv_full", "eval_clips")
FW_LIST = os.path.join(ROOT, "tools", "board_false_wakes.txt")
HOP, DEAD_S, EVAL_FRAC = 12, 3.0, 0.25
THRS = (0.60, 0.70, 0.80, 0.90)
BOARD_SPEECH_RMS = 1000.0


def wav(p):
    with wave.open(p) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


class Model:
    def __init__(self, path):
        import tensorflow as tf
        self.it = tf.lite.Interpreter(model_path=path); self.it.allocate_tensors()
        self.i, self.o = self.it.get_input_details()[0], self.it.get_output_details()[0]
        self.so, self.zo = self.o["quantization"]
        ap = os.path.join(os.path.dirname(path), "input_affine.json")
        aff = json.load(open(ap if os.path.exists(ap) else os.path.join("artifacts", "input_affine.json")))
        self.sc, self.zp = aff["eff_scale"], aff["eff_zp"]

    def scores(self, pcm):
        f = dequantise(quantise(log_mel(pcm)))
        out = []
        for s in range(0, f.shape[0] - cfg.CTX_FRAMES + 1, HOP):
            q = np.clip(np.round(f[s:s + cfg.CTX_FRAMES][None, ..., None] / self.sc + self.zp),
                        -128, 127).astype(np.int8)
            self.it.set_tensor(self.i["index"], q); self.it.invoke()
            out.append((float(np.ravel(self.it.get_tensor(self.o["index"]))[0]) - self.zo) * self.so)
        return np.array(out)


def fires(a, thr, rule):
    dead = int(DEAD_S / (HOP * 0.01)); i = n = 0
    i = 1
    while i < len(a):
        v = max(a[i], a[i - 1]) if rule == "peak2" else min(a[i], a[i - 1])
        if v >= thr:
            n += 1; i += dead
        else:
            i += 1
    return n


def level(x, target):
    x = x.astype(np.float32)
    r = np.sqrt(np.mean(x * x)) or 1.0
    return np.clip(x * (target / r), -32768, 32767).astype(np.int16)


def board_noise_tails():
    out = []
    for f in sorted(glob.glob(os.path.join(CAP, "*-room*.wav"))):
        if os.path.getsize(f) > 20 * 16000 * 2:
            x = wav(f); out.append(x[int(len(x) * (1 - EVAL_FRAC)):])
    return out


def table(title, a, hours):
    print(f"\n{title}  ({hours*60:.1f} min)")
    print(f"  {'thr':>5} {'peak-of-2 (now)':>18} {'2-in-a-row':>16}")
    for t in THRS:
        p, r = fires(a, t, "peak2"), fires(a, t, "row2")
        print(f"  {t:5.2f} {p:6d} = {p/hours:6.1f}/h {r:6d} = {r/hours:6.1f}/h")


def eval_lists(root):
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "features", "eval_*.txt"))):
        name = os.path.basename(f)[5:-4]
        if name == "board":
            continue
        paths = [l.strip() for l in open(f, encoding="utf-8") if l.strip()]
        if REPORT_HALF_ONLY:
            from fa_stream import half_of
            paths = [p for p in paths if half_of(p) == 1]
        out[name] = paths
    return out


LEVEL = {"musan_noise": 300.0, "fsd50k": 600.0}
REPORT_HALF_ONLY = False


def stream(paths, target_rms, max_s):
    from prep_corpus import decode_file
    rng = np.random.default_rng(0)
    order = rng.permutation(len(paths))
    parts, got = [], 0
    gap = np.zeros(4000, np.int16)
    for i in order:
        x = decode_file(paths[i])
        if x is None or len(x) < 8000:
            continue
        x = level((x * 32767).astype(np.int16), target_rms)
        parts += [x, gap]; got += len(x) + len(gap)
        if got >= max_s * 16000:
            break
    return np.concatenate(parts) if parts else np.zeros(0, np.int16)


def main():
    ap_root = os.environ.get("PRAHARI_DATA", r"E:\PRAHARI_DATA")
    mpath = sys.argv[1] if len(sys.argv) > 1 else os.path.join("artifacts", "prahari_int8.tflite")
    max_min = float(os.environ.get("PRAHARI_FA_MINUTES", "40"))
    global REPORT_HALF_ONLY, THRS
    REPORT_HALF_ONLY = True
    tj = os.path.join(os.path.dirname(mpath), "threshold.json")
    if os.path.exists(tj):
        t = round(float(json.load(open(tj))["threshold"]), 2)
        THRS = tuple(sorted(set(THRS) | {t}))
        print(f"(threshold picked in training: {t:.2f} - from {tj})")
    m = Model(mpath)
    print("=" * 66); print(f"FALSE WAKES - {mpath}"); print("=" * 66)
    noise = board_noise_tails()
    nz = np.concatenate(noise).astype(np.float32) if noise else None
    tot_fires = {(r, t): 0 for r in ("peak2", "row2") for t in THRS}
    tot_h = 0.0

    lists = eval_lists(ap_root)
    if not lists and os.path.isdir(CV_EVAL):
        lists = {"cv": sorted(glob.glob(os.path.join(CV_EVAL, "*.wav")))}
    for name, paths in lists.items():
        x = stream(paths, LEVEL.get(name, BOARD_SPEECH_RMS), max_min * 60)
        if not len(x):
            continue
        if nz is not None:
            x = np.clip(x.astype(np.float32) + np.resize(nz, len(x)), -32768, 32767).astype(np.int16)
        hours = len(x) / 16000 / 3600
        a = m.scores(x)
        table(f"A. {name}: held-out audio{' + XIAO noise' if nz is not None else ''}", a, hours)
        for r in ("peak2", "row2"):
            for t in THRS:
                tot_fires[(r, t)] += fires(a, t, r)
        tot_h += hours

    if tot_h:
        print(f"\nALL HELD-OUT AUDIO COMBINED  ({tot_h:.2f} h)   target: < 0.5 false wakes / hour")
        print(f"  {'thr':>5} {'peak-of-2 (now)':>18} {'2-in-a-row':>16}")
        for t in THRS:
            p, r = tot_fires[("peak2", t)], tot_fires[("row2", t)]
            print(f"  {t:5.2f} {p:6d} = {p/tot_h:6.2f}/h {r:6d} = {r/tot_h:6.2f}/h")

    names = []
    if os.path.exists(FW_LIST):
        names = [l.strip() for l in open(FW_LIST, encoding="utf-8")
                 if l.strip() and not l.startswith("#")]
    if names:
        have = [os.path.join(CAP, n) for n in names if os.path.exists(os.path.join(CAP, n))]
        print(f"\nB. the user's real false wakes on the board (never trained on): {len(have)} of {len(names)} files found")
        sc = [m.scores(wav(p)[: int(2.5 * 16000)]) for p in have]
        for t in THRS:
            hp = sum(fires(a, t, "peak2") > 0 for a in sc)
            hr = sum(fires(a, t, "row2") > 0 for a in sc)
            print(f"  thr {t:.2f}: still fire  peak-of-2 {hp}/{len(have)}   2-in-a-row {hr}/{len(have)}")

    if nz is not None:
        table("C. the XIAO's own held-out noise, no speech", m.scores(nz.astype(np.int16)), len(nz) / 16000 / 3600)


if __name__ == "__main__":
    main()
