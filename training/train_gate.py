import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
import prahari_config as cfg
from gate_model import build_gate, mac_count

DATA, OUT = "data", "artifacts"
SEED, EPOCHS, BATCH = 1337, 40, 96
RECALL_TARGET = 0.99
SCAN_MS       = 160.0
MAIN_MS       = 64.3
FRONT_PCT     = 5.0

tf.random.set_seed(SEED); np.random.seed(SEED)


def load(split):
    X = np.load(f"{DATA}/X_{split}.npy").astype("float32")[..., None]
    y = np.load(f"{DATA}/y_{split}.npy").astype("float32")
    kp = f"{DATA}/kind_{split}.npy"
    k = np.load(kp, allow_pickle=True) if os.path.exists(kp) else None
    return X, y, k


def augment(fill):
    def aug(x, y):
        shift = tf.random.uniform([], -6, 7, dtype=tf.int32)
        x = tf.roll(x, shift, axis=0)
        t0 = tf.random.uniform([], 0, cfg.CTX_FRAMES - 5, dtype=tf.int32)
        mask = tf.concat([tf.ones([t0, cfg.MEL_BINS, 1]),
                          tf.zeros([5, cfg.MEL_BINS, 1]),
                          tf.ones([cfg.CTX_FRAMES - t0 - 5, cfg.MEL_BINS, 1])], 0)
        x = x * mask + fill * (1.0 - mask)
        x = x + tf.random.normal(tf.shape(x), stddev=0.05)
        return x, y
    return aug


def threshold_for_recall(y, p, target):
    pos = np.sort(p[y == 1])
    if len(pos) == 0:
        raise SystemExit("no positives in split - check the data splits")
    idx = int(np.floor((1.0 - target) * len(pos)))
    return float(pos[max(0, min(idx, len(pos) - 1))])


def main():
    Xtr, ytr, _ = load("train")
    Xva, yva, _ = load("val")
    Xte, yte, kte = load("test")
    print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")
    print(f"positives: train {int(ytr.sum())}  val {int(yva.sum())}  test {int(yte.sum())}")

    fill = float(np.percentile(Xtr, 1))
    ds = (tf.data.Dataset.from_tensor_slices((Xtr, ytr))
          .shuffle(4096, seed=SEED)
          .map(augment(fill), num_parallel_calls=tf.data.AUTOTUNE)
          .batch(BATCH).prefetch(tf.data.AUTOTUNE))

    model = build_gate()
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-3),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
    os.makedirs(OUT, exist_ok=True)
    ckpt = os.path.join(OUT, "gate_best.keras")
    model.fit(ds, validation_data=(Xva, yva), epochs=EPOCHS, verbose=2,
              callbacks=[
                  callbacks.ModelCheckpoint(ckpt, monitor="val_auc", mode="max",
                                            save_best_only=True, verbose=0),
                  callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                          patience=10, restore_best_weights=True),
                  callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max",
                                              factor=0.5, patience=5, min_lr=1e-5),
              ])

    pva = model.predict(Xva, verbose=0).ravel()
    thr = threshold_for_recall(yva, pva, RECALL_TARGET)
    pte = model.predict(Xte, verbose=0).ravel()

    recall = float((pte[yte == 1] >= thr).mean())
    neg = pte[yte == 0]
    pass_rate = float((neg >= thr).mean())

    print("\n" + "=" * 68)
    print(f"gate threshold (from validation, {RECALL_TARGET:.0%} recall): {thr:.4f}")
    print(f"TEST recall on keywords : {recall:6.2%}   <- must stay near {RECALL_TARGET:.0%}")
    print(f"TEST pass rate on non-keywords : {pass_rate:6.2%}   <- THE NUMBER")
    if kte is not None:
        print("\npass rate by kind (how often each wakes stage 2):")
        for kind in sorted(set(str(k) for k in kte)):
            m = np.array([str(k) == kind for k in kte]) & (yte == 0)
            if m.sum():
                print(f"   {kind:<12} {float((pte[m] >= thr).mean()):6.2%}   (n={int(m.sum())})")

    macs = mac_count()
    est_ms = MAIN_MS * macs / 1_460_000
    gate_pct = 100 * est_ms / SCAN_MS
    s2_pct = pass_rate * 100 * MAIN_MS / SCAN_MS
    total = gate_pct + s2_pct + FRONT_PCT
    print("\n" + "=" * 68)
    print(f"CPU projection, gate every {SCAN_MS:.0f} ms")
    print(f"  front-end (measured)      {FRONT_PCT:5.1f}%")
    print(f"  gate  {est_ms:.1f} ms (ESTIMATED)   {gate_pct:5.1f}%")
    print(f"  stage 2 at {pass_rate:.1%} pass     {s2_pct:5.1f}%")
    print(f"  TOTAL                     {total:5.1f}%    "
          f"{'PASS - under 10%' if total < 10 else 'FAIL - over 10%'}")
    print("\nThe gate time is an ESTIMATE scaled by MAC count. Small models pay")
    print("proportionally more per-operator overhead, so expect it to be worse.")
    print("Measure arena_used_bytes() and real ms on the device before quoting.")

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    idx = np.random.RandomState(0).choice(len(Xtr), min(500, len(Xtr)), replace=False)
    conv.representative_dataset = lambda: ([Xtr[i:i+1].astype(np.float32)] for i in idx)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    blob = conv.convert()
    path = os.path.join(OUT, "prahari_gate_int8.tflite")
    open(path, "wb").write(blob)
    print(f"\ngate model file : {len(blob):,} B  -> {path}")

    it = tf.lite.Interpreter(model_content=blob); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    print(f"input  scale={i['quantization'][0]:.7f} zp={i['quantization'][1]}")
    print(f"output scale={o['quantization'][0]:.7f} zp={o['quantization'][1]}")
    print("\nIMPORTANT: if the gate's input scale/zero-point differ from the main")
    print("model's (0.0803922 / 46), the firmware must NOT reuse g_feat directly.")
    print("Check this before writing the integration code.")

    json.dump({"threshold": thr, "test_recall": recall, "test_pass_rate": pass_rate,
               "macs": macs, "scan_ms": SCAN_MS},
              open(os.path.join(OUT, "gate_report.json"), "w"), indent=2)
    print("\nnext:  python arena_calc.py artifacts/prahari_gate_int8.tflite")


if __name__ == "__main__":
    main()
