import os, json, argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
import prahari_config as cfg
from model_v8 import build_v8, macs, est_ms, VARIANTS

DATA, OUT = "data", "artifacts"
SEED, BATCH = 1337, 96
tf.random.set_seed(SEED); np.random.seed(SEED)


def load(split):
    X = np.load(f"{DATA}/X_{split}.npy").astype("float32")[..., None]
    y = np.load(f"{DATA}/y_{split}.npy").astype("float32")
    return X, y


def make_augmenter(fill_value):
    T_MASK, F_MASK, N_MASKS = 5, 5, 1

    def aug(x, y):
        shift = tf.random.uniform([], -8, 9, dtype=tf.int32)
        x = tf.roll(x, shift, axis=0)
        for _ in range(N_MASKS):
            t = tf.random.uniform([], 0, T_MASK, dtype=tf.int32)
            t0 = tf.random.uniform([], 0, cfg.CTX_FRAMES - t + 1, dtype=tf.int32)
            tm = tf.concat([tf.ones([t0, cfg.MEL_BINS, 1]),
                            tf.zeros([t, cfg.MEL_BINS, 1]),
                            tf.ones([cfg.CTX_FRAMES - t0 - t, cfg.MEL_BINS, 1])], axis=0)
            f = tf.random.uniform([], 0, F_MASK, dtype=tf.int32)
            f0 = tf.random.uniform([], 0, cfg.MEL_BINS - f + 1, dtype=tf.int32)
            fm = tf.concat([tf.ones([cfg.CTX_FRAMES, f0, 1]),
                            tf.zeros([cfg.CTX_FRAMES, f, 1]),
                            tf.ones([cfg.CTX_FRAMES, cfg.MEL_BINS - f0 - f, 1])], axis=1)
            m = tm * fm
            x = x * m + fill_value * (1.0 - m)
        x = x + tf.random.normal(tf.shape(x), stddev=0.15)
        return x, y
    return aug


def evaluate(p, yte, kinds):
    a = tf.keras.metrics.AUC(); a.update_state(yte, p)
    auc = float(a.result().numpy())
    best = None
    for t in np.arange(0.05, 1.00, 0.01):
        pred = (p >= t)
        tp = int((pred & (yte == 1)).sum()); fp = int((pred & (yte == 0)).sum())
        fn = int((~pred & (yte == 1)).sum()); tn = int((~pred & (yte == 0)).sum())
        far = fp / max(fp + tn, 1) * 100
        rec = tp / max(tp + fn, 1) * 100
        if far <= 1.0 and (best is None or rec > best[1]):
            best = (float(t), rec, far)
    if best is None:
        best = (0.5, float((p[yte == 1] >= 0.5).mean() * 100), 100.0)
    thr, rec, far = best
    conf = None
    if kinds is not None:
        m = (kinds == "confusable") & (yte == 0)
        if m.any():
            conf = float((p[m] >= thr).mean() * 100)
    return auc, thr, rec, far, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=25)
    args = ap.parse_args()

    Xtr, ytr = load("train"); Xva, yva = load("val"); Xte, yte = load("test")
    kp = f"{DATA}/kind_test.npy"
    kinds = np.load(kp, allow_pickle=False) if os.path.exists(kp) else None
    print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")

    fill = float(Xtr.mean())
    n_pos, n_neg = int((ytr == 1).sum()), int((ytr == 0).sum())
    cw = {0: len(ytr) / (2.0 * n_neg), 1: len(ytr) / (2.0 * n_pos)}
    ds_tr = (tf.data.Dataset.from_tensor_slices((Xtr, ytr))
             .shuffle(len(ytr), seed=SEED)
             .map(make_augmenter(fill), num_parallel_calls=tf.data.AUTOTUNE)
             .batch(BATCH).prefetch(tf.data.AUTOTUNE))
    ds_va = tf.data.Dataset.from_tensor_slices((Xva, yva)).batch(BATCH)
    steps = int(np.ceil(len(ytr) / BATCH))

    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name, kw in VARIANTS.items():
        tag = name.split()[0]
        m = macs(**kw)
        print("\n" + "=" * 72)
        print(f"{name}    {m:,} MACs   est {est_ms(m):.1f} ms")
        print("=" * 72)
        tf.keras.backend.clear_session()
        tf.random.set_seed(SEED); np.random.seed(SEED)

        model = build_v8(**kw)
        lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-3, decay_steps=args.epochs * steps,
            warmup_target=3e-3, warmup_steps=5 * steps)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.03),
                      metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
        model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, verbose=2,
                  class_weight=cw,
                  callbacks=[callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                     patience=args.patience,
                                                     restore_best_weights=True)])

        p = model.predict(Xte, verbose=0).ravel()
        auc, thr, rec, far, conf = evaluate(p, yte, kinds)
        ms = est_ms(m)
        rows.append(dict(tag=tag, name=name, macs=m, ms=ms, auc=auc, thr=thr,
                         rec=rec, far=far, conf=conf, params=int(model.count_params())))
        model.save(os.path.join(OUT, f"v8_{tag}.keras"))
        print(f"  AUC {auc:.4f}   detection {rec:.1f}% @ FAR {far:.1f}% (thr {thr:.2f})"
              + (f"   confusable FAR {conf:.1f}%" if conf is not None else ""))

    base = next(r for r in rows if r["tag"] == "v5")
    print("\n" + "=" * 96)
    print("SWEEP RESULT   (idle CPU = 5.0% measured front-end + inference at a 960 ms idle scan)")
    print("=" * 96)
    print(f"{'variant':<28}{'MACs':>10}{'est ms':>8}{'idle%':>8}{'AUC':>9}{'dAUC':>9}"
          f"{'det@1%FAR':>11}{'confFAR':>9}")
    print("-" * 96)
    for r in rows:
        idle = 5.0 + 100 * r["ms"] / 960
        flag = "" if idle >= 10 else "  <10%"
        print(f"{r['name']:<28}{r['macs']:>10,}{r['ms']:>8.1f}{idle:>7.1f}%"
              f"{r['auc']:>9.4f}{r['auc']-base['auc']:>+9.4f}{r['rec']:>10.1f}%"
              f"{(r['conf'] if r['conf'] is not None else float('nan')):>8.1f}%{flag}")
    print("-" * 96)
    ok = [r for r in rows if 5.0 + 100 * r["ms"] / 960 < 10.0]
    if ok:
        pick = max(ok, key=lambda r: r["auc"])
        print(f"\nBEST UNDER 10% IDLE: {pick['name']}")
        print(f"  AUC {pick['auc']:.4f} ({pick['auc']-base['auc']:+.4f} vs baseline), "
              f"detection {pick['rec']:.1f}% at <=1% FAR, est {pick['ms']:.1f} ms")
        print(f"\n  next: retrain it as the main model, export int8, confirm on device.")
        print(f"        The ms figure is ESTIMATED from MAC count - the board decides.")
    else:
        print("\nNo variant reaches <10% idle. The remaining levers are a slower idle")
        print("scan, a cheaper stem, or accepting the gap and accounting for it.")
    json.dump(rows, open(os.path.join(OUT, "v8_sweep.json"), "w"), indent=2)
    print(f"\nsaved: {OUT}/v8_sweep.json and v8_<tag>.keras for each variant")


if __name__ == "__main__":
    main()
