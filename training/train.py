import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
import prahari_config as cfg
from model import build_model

DATA   = "data"
OUT    = "artifacts"
SEED   = 1337
EPOCHS = 60
BATCH  = 96

tf.random.set_seed(SEED)
np.random.seed(SEED)


def load(split):
    X = np.load(os.path.join(DATA, f"X_{split}.npy"))
    if X.dtype != np.float16:
        X = X.astype("float32")
    y = np.load(os.path.join(DATA, f"y_{split}.npy")).astype("float32")
    return X[..., None], y


def make_augmenter(fill_value):
    T_MASK, F_MASK, N_MASKS = 5, 5, 1

    def aug(x, y, w=None):
        x = tf.cast(x, tf.float32)
        shift = tf.random.uniform([], -8, 9, dtype=tf.int32)
        x = tf.roll(x, shift, axis=0)

        for _ in range(N_MASKS):
            t = tf.random.uniform([], 0, T_MASK, dtype=tf.int32)
            t0 = tf.random.uniform([], 0, cfg.CTX_FRAMES - t + 1, dtype=tf.int32)
            tm = tf.concat([
                tf.ones([t0, cfg.MEL_BINS, 1]),
                tf.zeros([t, cfg.MEL_BINS, 1]),
                tf.ones([cfg.CTX_FRAMES - t0 - t, cfg.MEL_BINS, 1])], axis=0)
            f = tf.random.uniform([], 0, F_MASK, dtype=tf.int32)
            f0 = tf.random.uniform([], 0, cfg.MEL_BINS - f + 1, dtype=tf.int32)
            fm = tf.concat([
                tf.ones([cfg.CTX_FRAMES, f0, 1]),
                tf.zeros([cfg.CTX_FRAMES, f, 1]),
                tf.ones([cfg.CTX_FRAMES, cfg.MEL_BINS - f0 - f, 1])], axis=1)
            m = tm * fm
            x = x * m + fill_value * (1.0 - m)

        x = x + tf.random.normal(tf.shape(x), stddev=0.15)
        return (x, y) if w is None else (x, y, w)

    return aug


def main():
    os.makedirs(OUT, exist_ok=True)
    Xtr, ytr = load("train")
    Xva, yva = load("val")
    Xte, yte = load("test")
    print(f"train {Xtr.shape}   val {Xva.shape}   test {Xte.shape}")

    fill = float(Xtr.mean(dtype=np.float64))
    print(f"augment fill value (corpus mean): {fill:.3f}")

    kpath = os.path.join(DATA, "kind_train.npy")
    ktr = np.load(kpath, allow_pickle=False) if os.path.exists(kpath) else np.array([""] * len(ytr))
    board = (ktr == "boardnoise")
    HARDNEG_W = float(os.environ.get("PRAHARI_HARDNEG_W", "2"))
    POS_FRAC = float(os.environ.get("PRAHARI_POS_FRAC", "0.5"))
    n_pos, n_neg = int((ytr == 1).sum()), int((ytr == 0).sum())
    neg_total = float(n_neg - board.sum()) + float(board.sum()) * HARDNEG_W
    POS_W = POS_FRAC * neg_total / max(n_pos, 1)
    wtr = np.where(ytr == 1, POS_W, 1.0).astype("float32")
    wtr[board] = HARDNEG_W
    print(f"train balance   : {n_pos} pos / {n_neg} neg  ({n_neg/max(n_pos,1):.1f}:1)"
          f"  incl. {int(board.sum())} board-noise negatives")
    print(f"sample weights  : pos {POS_W:.2f}, neg 1, board {HARDNEG_W:g}"
          f"   -> total pos:neg weight {n_pos*POS_W/max(n_neg,1):.2f}")

    ds_tr = (tf.data.Dataset.from_tensor_slices((Xtr, ytr, wtr))
             .shuffle(len(ytr), seed=SEED)
             .map(make_augmenter(fill), num_parallel_calls=tf.data.AUTOTUNE)
             .batch(BATCH).prefetch(tf.data.AUTOTUNE))
    ds_va = tf.data.Dataset.from_tensor_slices((Xva, yva)).batch(BATCH)

    model = build_model()
    model.summary()

    steps = int(np.ceil(len(ytr) / BATCH))
    lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3, decay_steps=EPOCHS * steps, warmup_target=3e-3,
        warmup_steps=5 * steps)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.03),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

    hist = model.fit(
        ds_tr, validation_data=ds_va, epochs=EPOCHS, verbose=2,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                    patience=25, restore_best_weights=True),
            callbacks.ModelCheckpoint(os.path.join(OUT, "best.keras"),
                                      monitor="val_auc", mode="max",
                                      save_best_only=True, verbose=0),
        ])

    print("\n" + "=" * 62)
    print("TEST SET  (groups never seen in training)")
    print("=" * 62)
    p = model.predict(Xte, verbose=0).ravel()

    auc = tf.keras.metrics.AUC()
    auc.update_state(yte, p)
    print(f"AUC                    : {auc.result().numpy():.4f}")
    print(f"accuracy @ 0.5         : {((p > 0.5).astype(int) == yte).mean():.4f}\n")

    print("threshold sweep - pick the operating point from here, not 0.5")
    print(f"{'thresh':>7}{'FRR %':>9}{'FAR %':>9}{'recall %':>10}{'precision %':>13}")
    best = None
    for t in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (yte == 1)).sum())
        fp = int(((pred == 1) & (yte == 0)).sum())
        fn = int(((pred == 0) & (yte == 1)).sum())
        tn = int(((pred == 0) & (yte == 0)).sum())
        frr = fn / max(tp + fn, 1) * 100
        far = fp / max(fp + tn, 1) * 100
        rec = tp / max(tp + fn, 1) * 100
        pre = tp / max(tp + fp, 1) * 100
        print(f"{t:>7.2f}{frr:>9.1f}{far:>9.1f}{rec:>10.1f}{pre:>13.1f}")
        if far <= 1.0 and (best is None or frr < best[1]):
            best = (t, frr, far)

    if best:
        print(f"\nlowest false-reject at <=1% false-accept: "
              f"threshold {best[0]:.2f}  (FRR {best[1]:.1f}%, FAR {best[2]:.1f}%)")
        op = best[0]
    else:
        print("\nno threshold reaches <=1% false-accept on this test set")
        op = 0.5

    kpath = os.path.join(DATA, "kind_test.npy")
    if os.path.exists(kpath):
        kinds = np.load(kpath, allow_pickle=False)
        print(f"\nfalse accepts by kind, at threshold {op:.2f}")
        print(f"{'kind':<14}{'clips':>7}{'fired':>7}{'FAR %':>9}")
        for kd in sorted(set(kinds.tolist())):
            m = (kinds == kd) & (yte == 0)
            if not m.any():
                continue
            fired = int((p[m] >= op).sum())
            print(f"{kd:<14}{int(m.sum()):>7}{fired:>7}{fired/m.sum()*100:>8.1f}%")
        mp = (yte == 1)
        print(f"{'KEYWORD':<14}{int(mp.sum()):>7}{int((p[mp]>=op).sum()):>7}"
              f"{(p[mp]>=op).mean()*100:>8.1f}%  <- detection rate")

    model.save(os.path.join(OUT, "prahari_dscnn.keras"))
    with open(os.path.join(OUT, "history.json"), "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in hist.history.items()}, f)
    print(f"\nsaved: {OUT}/prahari_dscnn.keras")
    print(f"epochs run: {len(hist.history['loss'])}")


if __name__ == "__main__":
    main()
