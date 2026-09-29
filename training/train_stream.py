import argparse, glob, json, os, sys, time
import numpy as np


def full_speed():
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class PPTS(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]
        k32 = ctypes.windll.kernel32
        h = k32.GetCurrentProcess()
        st = PPTS(1, 0x1, 0x0)
        ok = k32.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st))
        k32.SetPriorityClass(h, 0x00008000)
        print(f"full speed: power throttling off ({'ok' if ok else 'refused'}), priority above normal")
    except Exception as e:
        print(f"full speed: could not change power settings ({e})")


full_speed()
N_CPU = os.cpu_count() or 4
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(N_CPU)
tf.config.threading.set_inter_op_parallelism_threads(max(2, N_CPU // 4))
from tensorflow.keras import callbacks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prahari_config as cfg
from model import build_model
from train import make_augmenter

DATA, OUT, SEED = "data", "artifacts", 1337

SHARES = {
    "keyword": 0.05, "keyword2": 0.03, "voices_kw": 0.12,
    "confusable": 0.05, "negative2": 0.04, "voices_neg": 0.09,
    "core": 0.03, "esc": 0.05, "board": 0.04,
    "indicvoices": 0.11, "slr104": 0.08, "kathbath": 0.07, "cv": 0.05, "shrutilipi": 0.02,
    "musan_speech": 0.03,
    "musan_music": 0.05, "musan_noise": 0.03, "fsd50k": 0.01, "fsd50k_dev": 0.04,
}


class Source:
    def __init__(self, d):
        self.name = os.path.basename(d)
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.label = int(self.meta["label"])
        self.shards = [np.load(f, mmap_mode="r") for f in sorted(glob.glob(os.path.join(d, "shard_*.npy")))]
        self.shards = [s for s in self.shards if len(s)]
        self.n = sum(len(s) for s in self.shards)

    BUF = 1024

    def _refill(self, rng):
        s = self.shards[rng.integers(len(self.shards))]
        m = min(self.BUF, len(s))
        a = int(rng.integers(0, len(s) - m + 1))
        self.buf = np.array(s[a:a + m])
        self.served = 0

    def draw(self, k, rng):
        if getattr(self, "buf", None) is None or self.served >= max(len(self.buf), 1):
            self._refill(rng)
        self.served += k
        return self.buf[rng.integers(0, len(self.buf), size=k)]


def batches(sources, shares, batch, seed):
    rng = np.random.default_rng(seed)
    names = list(shares)
    p = np.array([shares[n] for n in names])
    while True:
        counts = rng.multinomial(batch, p)
        xs, ys = [], []
        for n, c in zip(names, counts):
            if c:
                xs.append(sources[n].draw(int(c), rng))
                ys.append(np.full(int(c), sources[n].label, np.float32))
        x = np.concatenate(xs); y = np.concatenate(ys)
        idx = rng.permutation(len(y))
        yield x[idx], y[idx]


def recall_at_far(p, y, far=0.005):
    neg = np.sort(p[y == 0])[::-1]
    if not len(neg):
        return 0.0, 1.0
    k = max(int(np.floor(far * len(neg))), 0)
    thr = neg[k] if k < len(neg) else 0.0
    return float((p[y == 1] > thr).mean()), float(thr)


def batch_augmenter(fill_value):
    T, F = cfg.CTX_FRAMES, cfg.MEL_BINS

    def aug(x, y):
        x = tf.cast(x, tf.float32)
        B = tf.shape(x)[0]
        shift = tf.random.uniform([B], -8, 9, dtype=tf.int32)
        idx = tf.clip_by_value(tf.range(T)[None, :] - shift[:, None], 0, T - 1)
        x = tf.gather(x, idx, batch_dims=1)
        t = tf.random.uniform([B], 0, 5, dtype=tf.int32)
        t0 = tf.cast(tf.random.uniform([B]) * tf.cast(T - t + 1, tf.float32), tf.int32)
        r = tf.range(T)[None, :]
        tm = tf.cast(~((r >= t0[:, None]) & (r < (t0 + t)[:, None])), tf.float32)[:, :, None]
        f = tf.random.uniform([B], 0, 5, dtype=tf.int32)
        f0 = tf.cast(tf.random.uniform([B]) * tf.cast(F - f + 1, tf.float32), tf.int32)
        c = tf.range(F)[None, :]
        fm = tf.cast(~((c >= f0[:, None]) & (c < (f0 + f)[:, None])), tf.float32)[:, None, :]
        m = tm * fm
        x = x * m + fill_value * (1.0 - m)
        a = tf.random.uniform([B], 0.88, 1.12)
        a = 1.0 + (a - 1.0) * tf.cast(tf.random.uniform([B]) < 0.6, tf.float32)
        pos = tf.clip_by_value(tf.range(F, dtype=tf.float32)[None, :] * a[:, None], 0.0, F - 1.0)
        lo = tf.floor(pos)
        w = (pos - lo)[:, None, :]
        lo = tf.cast(lo, tf.int32)
        hi = tf.minimum(lo + 1, F - 1)
        x = tf.gather(x, lo, axis=2, batch_dims=1) * (1.0 - w) + tf.gather(x, hi, axis=2, batch_dims=1) * w
        x = x + tf.random.normal(tf.shape(x), stddev=0.15)
        return x[..., None], y

    return aug


class Pick(callbacks.Callback):
    def __init__(self, Xva, yva, path, fa=None, human=None):
        super().__init__(); self.X, self.y, self.path, self.fa, self.human = Xva, yva, path, fa, human
        self.best, self.log, self.best_row = None, [], None

    def on_epoch_end(self, epoch, logs=None):
        import fa_stream as fs
        t_cb = time.time()
        p = self.model.predict(self.X, verbose=0, batch_size=512).ravel()
        r05, t05 = recall_at_far(p, self.y, 0.005)
        r1, _ = recall_at_far(p, self.y, 0.01)
        row = {"epoch": epoch + 1, "recall_at_0.5pct": r05, "thr": t05, "recall_at_1pct": r1,
               "loss": float(logs.get("loss", 0))}
        line = f"  val: detection {r05*100:5.1f}% at 0.5% false accepts (thr {t05:.3f}), {r1*100:5.1f}% at 1%"
        if self.fa is not None:
            Xs, src, names, hours = self.fa
            ps = fs.predict(self.model, Xs)
            thr, fah = fs.threshold_for_target(ps, src, hours)
            f90 = fs.fa_per_hour(ps, src, hours, 0.90)
            per = {n: fs.fires_at(ps[src == k], 0.90) / hours[k] for k, n in enumerate(names)}
            det = float((p[self.y == 1] >= thr).mean()) if thr is not None else 0.0
            row.update(stream_thr=thr, stream_fa_h=fah, stream_detect=det, fa_h_at_090=f90,
                       fa_h_at_090_by_source=per)
            key = (det, -f90)
            if self.human is not None:
                import human_sets as hs
                Xh, ih, rows = self.human
                ph = fs.predict(self.model, Xh)
                sc = hs.clip_scores(ph, ih, len(rows))
                sb = hs.board_scores(ph, ih, len(rows))
                hs_t = hs.summary(sb, rows, thr if thr is not None else 0.99)
                hs_7 = hs.summary(sb, rows, 0.70)
                hs_40 = hs.summary(sc, rows, thr if thr is not None else 0.99)
                row.update(real_detect=hs_t["speaker_mean"], real_by_gender=hs_t["by_gender"],
                           real_per_speaker=hs_t["per_speaker"], real_confusable_fa=hs_t["confusable_fa"],
                           real_detect_070=hs_7["speaker_mean"])
                key = (hs_t["speaker_mean"], det, -f90)
            line += (f"\n       stream ({hours.sum():.1f} h): " +
                     (f"<= {fs.TARGET_FA_H}/h from thr {thr:.2f} ({fah:.2f}/h) -> detects {det*100:5.1f}% of val keywords"
                      if thr is not None else f"never <= {fs.TARGET_FA_H}/h (at 0.99: {fah:.2f}/h)") +
                     f"   | at 0.90: {f90:.2f}/h  worst: " +
                     ", ".join(f"{n} {v:.1f}" for n, v in sorted(per.items(), key=lambda kv: -kv[1])[:3]))
            if self.human is not None:
                g = hs_t["by_gender"]
                line += (f"\n       REAL voices, BOARD scan, thr {thr if thr is not None else 0.99:.2f}: "
                         f"{hs_t['speaker_mean']*100:5.1f}% (speaker mean) | "
                         + " ".join(f"{k} {v*100:.0f}%" for k, v in g.items())
                         + " | " + " ".join(f"{k} {v*100:.0f}%" for k, v in hs_t["per_speaker"].items())
                         + " | length " + " ".join(f"{k} {v*100:.0f}%" for k, v in hs_t["by_length"].items())
                         + f" | real sound-alikes fire {hs_t['confusable_fa']*100:.0f}%"
                         + f"   (at 0.70: {hs_7['speaker_mean']*100:.0f}%; best-window {hs_40['speaker_mean']*100:.0f}%)")
        else:
            key = (r05, 0.0)
        self.log.append(row)
        if self.best is None or key > self.best:
            self.best, self.best_row = key, row
            self.model.save(self.path); line += "   <- best, saved"
        print(line + f"   [check took {time.time() - t_cb:.0f} s]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"E:\PRAHARI_DATA")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--fa-minutes", type=float, default=30.0,
                    help="minutes per held-out source in the in-loop false-wake stream (0 = off)")
    a = ap.parse_args()
    tf.random.set_seed(SEED); np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)

    feat = os.path.join(a.root, "features")
    sources = {}
    for d in sorted(glob.glob(os.path.join(feat, "*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "meta.json")):
            s = Source(d)
            if s.n:
                sources[s.name] = s
    if "keyword" not in sources:
        sys.exit("no keyword shards - run prep_corpus.py first")
    shares = {n: w for n, w in SHARES.items() if n in sources}
    tot = sum(shares.values())
    shares = {n: w / tot for n, w in shares.items()}
    print(f"{'source':<14}{'windows':>10}{'label':>7}{'share':>8}")
    for n, w in shares.items():
        print(f"{n:<14}{sources[n].n:>10,}{sources[n].label:>7}{w*100:>7.1f}%")
    print(f"total windows on disk: {sum(s.n for s in sources.values()):,}")

    Xva = np.load(os.path.join(DATA, "X_val.npy")).astype("float32")[..., None]
    yva = np.load(os.path.join(DATA, "y_val.npy")).astype("float32")
    Xte = np.load(os.path.join(DATA, "X_test.npy")).astype("float32")[..., None]
    yte = np.load(os.path.join(DATA, "y_test.npy")).astype("int32")
    ve = os.path.join(feat, "val_extra.npz")
    if os.path.exists(ve):
        z = np.load(ve)
        Xva = np.concatenate([Xva, z["X"].astype("float32")[..., None]])
        yva = np.concatenate([yva, z["y"].astype("float32")])
        print(f"val + {len(z['y'])} new clips from the validation voices "
              f"({int(z['y'].sum())} keyword)")
    print(f"val {Xva.shape}  test {Xte.shape}")

    fill = float(np.mean([sources[n].shards[0][:512].astype(np.float64).mean() for n in shares]))
    aug = batch_augmenter(fill)
    sig = (tf.TensorSpec((None, cfg.CTX_FRAMES, cfg.MEL_BINS), tf.float16),
           tf.TensorSpec((None,), tf.float32))
    ds = (tf.data.Dataset.from_generator(lambda: batches(sources, shares, a.batch, SEED),
                                         output_signature=sig)
          .map(aug, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
          .prefetch(8))
    t_in = time.time()
    it = iter(ds)
    for _ in range(30):
        next(it)
    print(f"input pipeline alone: {(time.time() - t_in) / 30 * 1000:.0f} ms per batch of {a.batch}"
          f"  ({N_CPU} CPU threads)", flush=True)

    model = build_model()
    lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3, decay_steps=a.epochs * a.steps, warmup_target=3e-3,
        warmup_steps=a.steps)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.03),
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    import fa_stream as fs
    fa = fs.build_select(a.root, a.fa_minutes) if a.fa_minutes > 0 else None
    if fa is None:
        print("WARNING: no held-out stream (features/eval_*.txt) - picking epochs by val only")
    import human_sets as hs
    human = hs.load("select")
    if human is None:
        print("WARNING: no real 'select' recordings found - epochs are picked on synthetic val only")
    else:
        rs = human[2]
        print(f"real voices in the loop (select): {len({r['speaker'] for r in rs})} speakers "
              f"({', '.join(sorted({r['speaker'] + '/' + r['gender'][0] for r in rs}))}), "
              f"{sum(r['label'] == 'keyword' for r in rs)} keyword clips, {len(human[0]):,} windows")
    pick = Pick(Xva, yva, os.path.join(OUT, "best.keras"), fa, human)
    t0 = time.time()
    model.fit(ds, epochs=a.epochs, steps_per_epoch=a.steps, verbose=2, callbacks=[pick])
    print(f"training took {(time.time()-t0)/60:.0f} min")

    model = tf.keras.models.load_model(os.path.join(OUT, "best.keras"))
    model.save(os.path.join(OUT, "prahari_dscnn.keras"))
    best = pick.best_row or {}
    if best.get("stream_thr") is not None:
        json.dump({"threshold": best["stream_thr"], "fa_per_hour_select": best["stream_fa_h"],
                   "val_detect": best["stream_detect"], "epoch": best["epoch"],
                   "real_detect_select": best.get("real_detect"), "real_by_gender_select": best.get("real_by_gender"),
                   "note": "float model on the SELECT half; fa_eval.py re-checks int8 on the REPORT half"},
                  open(os.path.join(OUT, "threshold.json"), "w"), indent=1)
        print(f"\nkept epoch {best['epoch']}: threshold {best['stream_thr']:.2f} -> "
              f"{best['stream_fa_h']:.2f} false wakes/h on the select stream, "
              f"{best['stream_detect']*100:.1f}% val keywords detected  (artifacts/threshold.json)")
    json.dump({"picks": pick.log, "shares": shares,
               "windows": {n: s.n for n, s in sources.items()}},
              open(os.path.join(OUT, "history.json"), "w"), indent=1)

    p = model.predict(Xte, verbose=0, batch_size=512).ravel()
    print("\n" + "=" * 62 + "\nTEST SET (speaker-disjoint, never trained on)\n" + "=" * 62)
    for far in (0.001, 0.005, 0.01, 0.02):
        r, t = recall_at_far(p, yte, far)
        print(f"  detection {r*100:5.1f}% at {far*100:.1f}% false accepts  (threshold {t:.3f})")
    kp = os.path.join(DATA, "kind_test.npy")
    if os.path.exists(kp):
        kinds = np.load(kp)
        for thr in (0.70, 0.90):
            print(f"\n  at threshold {thr:.2f}:")
            for kd in sorted(set(kinds.tolist())):
                m = (kinds == kd)
                if m.any():
                    rate = (p[m] >= thr).mean() * 100
                    lab = "DETECTED" if kd == "keyword" else "false accepts"
                    print(f"    {kd:<12}{int(m.sum()):>6}  {rate:5.1f}% {lab}")
    te = os.path.join(feat, "test_extra.npz")
    if os.path.exists(te):
        z = np.load(te)
        pe = model.predict(z["X"].astype("float32")[..., None], verbose=0, batch_size=512).ravel()
        ye, ge = z["y"].astype(int), z["gender"]
        thr = (pick.best_row or {}).get("stream_thr") or 0.9
        print(f"\nSECOND TEST SET (new clips, test voices only) at the chosen threshold {thr:.2f}:")
        for gname in sorted(set(ge.tolist())):
            m = (ye == 1) & (ge == gname)
            if m.any():
                print(f"  keyword, {gname:<8} {int(m.sum()):>5}  {(pe[m] >= thr).mean()*100:5.1f}% DETECTED")
        m = ye == 0
        if m.any():
            print(f"  sound-alikes     {int(m.sum()):>5}  {(pe[m] >= thr).mean()*100:5.1f}% false accepts")
    tv = os.path.join(feat, "test_voices.npz")
    if os.path.exists(tv):
        z = np.load(tv)
        pv = model.predict(z["X"].astype("float32")[..., None], verbose=0, batch_size=512).ravel()
        yv, gv, bv = z["y"].astype(int), z["gender"], z["backend"]
        thr = (pick.best_row or {}).get("stream_thr") or 0.9
        print(f"\nHELD-OUT SYNTHETIC VOICES (dataset_voices test split) at thr {thr:.2f}:")
        for bn in sorted(set(bv.tolist())):
            for gname in sorted(set(gv.tolist())):
                m = (yv == 1) & (gv == gname) & (bv == bn)
                if m.any():
                    print(f"  keyword {bn:<8} {gname:<7} {int(m.sum()):>5}  {(pv[m] >= thr).mean()*100:5.1f}% DETECTED")
            m = (yv == 0) & (bv == bn)
            if m.any():
                print(f"  sound-alikes {bn:<8}     {int(m.sum()):>5}  {(pv[m] >= thr).mean()*100:5.1f}% false accepts")
    thr = (pick.best_row or {}).get("stream_thr") or 0.9
    for split in ("select", "report"):
        hv = hs.load(split)
        if hv is None:
            print(f"\nREAL VOICES ({split}): none recorded yet")
            continue
        ph = model.predict(hv[0].astype("float32")[..., None], verbose=0, batch_size=1024).ravel()
        sc = hs.board_scores(ph, hv[1], len(hv[2]))
        print(f"\nREAL VOICES - {split.upper()} speakers"
              + (" (helped choose the epoch)" if split == "select" else " (never influenced any choice)"))
        for t in sorted({round(thr, 2), 0.70, 0.90}):
            r = hs.summary(sc, hv[2], t)
            print(f"  thr {t:.2f}: detected {r['speaker_mean']*100:5.1f}% (speaker mean)  "
                  + "  ".join(f"{k} {v*100:.0f}%" for k, v in r["by_gender"].items())
                  + "  | length " + " ".join(f"{k} {v*100:.0f}%" for k, v in r["by_length"].items())
                  + f"  | sound-alikes fire {r['confusable_fa']*100:.0f}%, other speech {r['speech_fa']*100:.0f}%")
        r = hs.summary(sc, hv[2], thr)
        print("  per speaker at chosen thr: " + ", ".join(f"{k} {v*100:.0f}%" for k, v in r["per_speaker"].items()))
    print(f"\nsaved {OUT}/prahari_dscnn.keras (best epoch: most REAL-voice detection at <= 0.5 false wakes/h)")
    print("next: python export_tflite.py && python gen_test_clip.py && python eval_human.py && python fa_eval.py")


if __name__ == "__main__":
    main()
