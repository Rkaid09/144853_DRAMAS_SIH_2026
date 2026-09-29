import os, json, argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

import prahari_config as cfg
from model import build_model
from train_v8 import load, make_augmenter, evaluate, DATA, OUT, SEED, BATCH

PIX          = cfg.CTX_FRAMES * (cfg.MEL_BINS // 2) * 8
NS_GENERAL   = 19.6
NS_IM2COL    = 39.6
PAD_MS_WIDE  = 2.0
REST_MS      = 30.07 - 14.73
MEASURED     = {(3, 5): 14.73, (6, 5): 19.86, (2, 16): 11.88}

VARIANTS = [
    ((3, 5),  "3x5  baseline, shipping"),
    ((2, 16), "2x16 timed on board"),
    ((2, 5),  "2x5  fewest slots, narrow"),
    ((1, 16), "1x16 one time tap"),
]


def stem_ms(t, m, in_ch=1):
    if (t, m) in MEASURED:
        return MEASURED[(t, m)], True
    row, window = m * in_ch, m * t * in_ch
    if row < 16 and window >= 16:
        slots, ns = (window + 15) & ~15, NS_IM2COL
    else:
        slots, ns = ((row + 15) // 16) * 16 * t, NS_GENERAL
    return PIX * slots * ns / 1e6 + (PAD_MS_WIDE if m >= 16 else 0.0), False


def useful_pct(t, m, in_ch=1):
    row, window = m * in_ch, m * t * in_ch
    slots = ((window + 15) & ~15) if (row < 16 and window >= 16) \
            else ((row + 15) // 16) * 16 * t
    return 100.0 * window / slots


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
    for (t, m), label in VARIANTS:
        ms, measured = stem_ms(t, m)
        tag = f"{t}x{m}"
        print("\n" + "=" * 72)
        print(f"stem {label}")
        print(f"  stem {ms:.2f} ms {'(MEASURED on board)' if measured else '(predicted)'}"
              f"   total ~{REST_MS + ms:.2f} ms   {useful_pct(t, m):.0f}% of slots useful")
        print("=" * 72)
        tf.keras.backend.clear_session()
        tf.random.set_seed(SEED); np.random.seed(SEED)

        model = build_model(stem=(t, m))
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
        rows.append(dict(tag=tag, label=label, time_taps=t, mel_taps=m,
                         stem_ms=ms, measured=measured, total_ms=REST_MS + ms,
                         auc=auc, thr=thr, rec=rec, far=far, conf=conf,
                         params=int(model.count_params())))
        model.save(os.path.join(OUT, f"v9_{tag}.keras"))
        print(f"  AUC {auc:.4f}   detection {rec:.1f}% @ FAR {far:.1f}% (thr {thr:.2f})"
              + (f"   confusable FAR {conf:.1f}%" if conf is not None else ""))

    base = next(r for r in rows if r["tag"] == "3x5")
    print("\n" + "=" * 100)
    print("V9 STEM SWEEP")
    print("=" * 100)
    print(f"{'stem':<8}{'stem ms':>9}{'total ms':>10}{'AUC':>9}{'dAUC':>9}"
          f"{'det@1%FAR':>11}{'confFAR':>9}  note")
    print("-" * 100)
    for r in rows:
        d = r["auc"] - base["auc"]
        note = "baseline" if r["tag"] == "3x5" else (
            "ADOPT" if d >= -0.004 else "loses accuracy")
        print(f"{r['tag']:<8}{r['stem_ms']:>9.2f}{r['total_ms']:>10.2f}"
              f"{r['auc']:>9.4f}{d:>+9.4f}{r['rec']:>10.1f}%"
              f"{(r['conf'] if r['conf'] is not None else float('nan')):>8.1f}%  {note}")
    print("-" * 100)

    ok = [r for r in rows if r["tag"] != "3x5"
          and r["auc"] - base["auc"] >= -0.004
          and (r["conf"] is None or base["conf"] is None
               or r["conf"] <= base["conf"] + 2.0)]
    if ok:
        pick = min(ok, key=lambda r: r["total_ms"])
        print(f"\nFASTEST VARIANT THAT HOLDS ACCURACY: {pick['tag']}")
        print(f"  AUC {pick['auc']:.4f} ({pick['auc']-base['auc']:+.4f}), "
              f"total ~{pick['total_ms']:.2f} ms vs {base['total_ms']:.2f} ms")
        print(f"\n  To adopt it:")
        print(f"    set PRAHARI_STEM={pick['tag']}")
        print(f"    copy artifacts/v9_{pick['tag']}.keras over "
              f"artifacts/prahari_dscnn.keras")
        print(f"    python export_tflite.py")
        print(f"    python gen_test_clip.py, rebuild, flash, p50")
        print(f"  Then CONFIRM on the board. Every ms figure above is predicted")
        print(f"  except 3x5 and 2x16, and predictions have been wrong twice today.")
    else:
        print("\nNO CHEAPER STEM HOLDS ACCURACY.")
        print("  That is a result, not a failure: 3x5's frequency width is doing")
        print("  real work. Ship 30.07 ms and stop optimising the stem.")

    json.dump(rows, open(os.path.join(OUT, "v9_sweep.json"), "w"), indent=2)
    print(f"\nsaved: {OUT}/v9_sweep.json and v9_<stem>.keras for each variant")


if __name__ == "__main__":
    main()
