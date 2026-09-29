import os, sys
import numpy as np
import tensorflow as tf

DATA = "data"
MODEL = os.path.join("artifacts", "prahari_dscnn.keras")
THRESHOLDS = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]


def load(split):
    X = np.load(f"{DATA}/X_{split}.npy").astype("float32")[..., None]
    y = np.load(f"{DATA}/y_{split}.npy").astype("int32")
    k = np.load(f"{DATA}/kind_{split}.npy")
    rp = f"{DATA}/rep_{split}.npy"
    r = np.load(rp) if os.path.exists(rp) else np.zeros(len(y), dtype=np.int16)
    return X, y, k, r


def main():
    if not os.path.exists(MODEL):
        sys.exit(f"no model at {MODEL} - run train.py first")
    X, y, kind, rep = load("test")
    if not os.path.exists(f"{DATA}/rep_test.npy"):
        print("WARNING: rep_test.npy missing - rerun build_dataset.py to get")
        print("         the clean-vs-noisy breakdown.\n")

    model = tf.keras.models.load_model(MODEL)
    p = model.predict(X, verbose=0).ravel()

    auc = tf.keras.metrics.AUC(); auc.update_state(y, p)
    print(f"test clips      : {len(y)}   positives {int((y==1).sum())}")
    print(f"AUC             : {auc.result().numpy():.4f}\n")

    pos, neg = (y == 1), (y == 0)
    clean, aug = (rep == 0), (rep > 0)

    print("DETECTION RATE - clean room vs augmented (noise, gain, speed)")
    print("-" * 62)
    print(f"{'thresh':>7}{'overall':>10}{'clean':>10}{'augmented':>12}{'gap':>8}")
    for t in THRESHOLDS:
        d_all = (p[pos] >= t).mean() * 100
        mc, ma = pos & clean, pos & aug
        d_c = (p[mc] >= t).mean() * 100 if mc.any() else float("nan")
        d_a = (p[ma] >= t).mean() * 100 if ma.any() else float("nan")
        print(f"{t:>7.2f}{d_all:>9.1f}%{d_c:>9.1f}%{d_a:>11.1f}%{d_c-d_a:>7.1f}")

    print("\nFALSE ACCEPTS BY KIND")
    print("-" * 62)
    kinds = sorted({k for k in kind[neg].tolist()})
    hdr = "".join(f"{t:>9.2f}" for t in THRESHOLDS)
    print(f"{'kind':<13}{'clips':>7}{hdr}")
    for kd in kinds:
        m = neg & (kind == kd)
        row = "".join(f"{(p[m]>=t).mean()*100:>8.1f}%" for t in THRESHOLDS)
        print(f"{kd:<13}{int(m.sum()):>7}{row}")

    m = neg & (kind != "confusable")
    row = "".join(f"{(p[m]>=t).mean()*100:>8.1f}%" for t in THRESHOLDS)
    print(f"{'-'*62}")
    print(f"{'EXCL confus.':<13}{int(m.sum()):>7}{row}   <- deployment-like")

    print("\nOPERATING POINTS")
    print("-" * 62)
    print(f"{'thresh':>7}{'detect %':>10}{'FAR all %':>11}{'FAR real %':>12}")
    for t in THRESHOLDS:
        d = (p[pos] >= t).mean() * 100
        f_all = (p[neg] >= t).mean() * 100
        mr = neg & (kind != "confusable")
        f_real = (p[mr] >= t).mean() * 100
        print(f"{t:>7.2f}{d:>9.1f}%{f_all:>10.1f}%{f_real:>11.1f}%")

    print("\n'FAR real' excludes confusables. Both numbers belong on the")
    print("slide: the strict one shows rigour, the realistic one shows")
    print("what a device in a room would actually do.")


if __name__ == "__main__":
    main()
