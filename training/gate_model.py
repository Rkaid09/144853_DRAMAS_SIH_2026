import tensorflow as tf
from tensorflow.keras import layers, Model
import prahari_config as cfg

POOL    = (4, 4)
C_STEM  = 8
C_OUT   = 16


def build_gate(dropout=0.10):
    inp = layers.Input(shape=(cfg.CTX_FRAMES, cfg.MEL_BINS, 1), name="logmel")

    x = layers.BatchNormalization(name="gate_in_bn")(inp)

    x = layers.AveragePooling2D(POOL, name="gate_pool")(x)

    x = layers.Conv2D(C_STEM, (3, 3), padding="same", use_bias=False,
                      name="gate_conv")(x)
    x = layers.BatchNormalization(name="gate_conv_bn")(x)
    x = layers.ReLU(name="gate_conv_relu")(x)

    x = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False,
                               name="gate_dw")(x)
    x = layers.BatchNormalization(name="gate_dw_bn")(x)
    x = layers.ReLU(name="gate_dw_relu")(x)

    x = layers.Conv2D(C_OUT, (1, 1), use_bias=False,
                      name="gate_pw")(x)
    x = layers.BatchNormalization(name="gate_pw_bn")(x)
    x = layers.ReLU(name="gate_pw_relu")(x)

    x = layers.GlobalAveragePooling2D(name="gate_gap")(x)
    x = layers.Dropout(dropout, name="gate_drop")(x)
    out = layers.Dense(1, activation="sigmoid", name="gate_score")(x)

    return Model(inp, out, name="prahari_gate")


def mac_count():
    ph = cfg.CTX_FRAMES // POOL[0]
    pw = cfg.MEL_BINS   // POOL[1]
    conv = ph * pw * C_STEM * 3 * 3 * 1
    dw   = ph * pw * C_STEM * 3 * 3
    pwc  = ph * pw * C_OUT  * C_STEM
    dense = C_OUT
    return conv + dw + pwc + dense


if __name__ == "__main__":
    m = build_gate()
    m.summary()
    macs = mac_count()
    MAIN_MACS, MAIN_MS = 1_460_000, 64.3
    est = MAIN_MS * macs / MAIN_MACS
    print(f"\nparameters       : {m.count_params():,}")
    print(f"MACs per run     : {macs:,}   (main model ~{MAIN_MACS:,})")
    print(f"reduction        : {MAIN_MACS/macs:.1f}x")
    print(f"est. inference   : {est:.1f} ms   "
          f"(scaled from the main model's measured {MAIN_MS} ms)")
    print("\nCAUTION: that scaling assumes cost is proportional to MACs. For a")
    print("model this small, per-operator overhead is a larger share, so the")
    print("real figure will be WORSE. Measure on device before believing it.")
    print("\nCPU budget, scanning every 160 ms:")
    print(f"  gate        {est:.1f} ms / 160 ms          = {100*est/160:.1f}%")
    print(f"  front-end                              = 5.0%")
    for p in (0.02, 0.04, 0.08, 0.15):
        total = 100*est/160 + 5.0 + p * 100 * 64.3/160
        print(f"  pass-rate {p*100:4.0f}%  -> stage 2 {p*100*64.3/160:4.1f}%"
              f"   TOTAL {total:5.1f}%   {'PASS' if total < 10 else 'FAIL'}")
