import os
import tensorflow as tf
from tensorflow.keras import layers, Model
import prahari_config as cfg

def _parse_stem(spec):
    try:
        t, m = spec.lower().split("x")
        return (int(t), int(m))
    except Exception:
        raise SystemExit(f"PRAHARI_STEM must look like 4x5, got {spec!r}")

STEM_KERNEL = _parse_stem(os.environ.get("PRAHARI_STEM", "2x16"))

C_STEM = 8
C_DS1 = 16
C_DS2 = 32


def ds_block(x, channels, name, kernel=(5, 3)):
    shortcut = x
    x = layers.DepthwiseConv2D(kernel, padding="same", use_bias=False,
                               name=f"{name}_dw")(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = layers.ReLU(name=f"{name}_dw_relu")(x)

    x = layers.Conv2D(channels, (1, 1), use_bias=False, name=f"{name}_pw")(x)
    x = layers.BatchNormalization(name=f"{name}_pw_bn")(x)
    x = layers.ReLU(name=f"{name}_pw_relu")(x)

    if shortcut.shape[-1] == channels:
        x = layers.Add(name=f"{name}_add")([shortcut, x])
    return x


def build_model(dropout=0.15, input_bn=True, fast_head=False, stem=None):
    inp = layers.Input(shape=(cfg.CTX_FRAMES, cfg.MEL_BINS, 1), name="logmel")

    x = layers.BatchNormalization(name="input_bn")(inp) if input_bn else inp

    x = layers.Conv2D(C_STEM, stem or STEM_KERNEL, strides=(1, 2),
                      padding="same", use_bias=False, name="stem")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    x = layers.MaxPooling2D((2, 1), name="pool1")(x)
    x = ds_block(x, C_DS1, "ds1")

    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = ds_block(x, C_DS2, "ds2")

    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = ds_block(x, C_DS2, "ds3")
    x = ds_block(x, C_DS2, "ds4")

    if fast_head:
        h, w = int(x.shape[1]), int(x.shape[2])
        x = layers.AveragePooling2D((h, w), name="gap_pool")(x)
        x = layers.Dropout(dropout, name="drop")(x)
        out = layers.Conv2D(1, (1, 1), activation="sigmoid", name="score")(x)
    else:
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.Dropout(dropout, name="drop")(x)
        out = layers.Dense(1, activation="sigmoid", name="score")(x)

    return Model(inp, out, name="prahari_dscnn_v5")


if __name__ == "__main__":
    m = build_model()
    m.summary()
    print(f"\nparameters      : {m.count_params():,}   (v4 had 7,501)")

    print("\nactivation tensors (int8):")
    shapes = [("input", 98*40*1), ("stem out", 98*20*C_STEM),
              ("pool1", 49*20*C_STEM), ("ds1 out", 49*20*C_DS1),
              ("pool2", 24*10*C_DS1), ("ds2-4", 24*10*C_DS2)]
    for n, s in shapes:
        print(f"  {n:<12} {s:>7,} B  ({s/1024:5.1f} KB)")

    macs = (98*20*C_STEM * 3*5
            + 49*20*C_STEM * 5*3 + 49*20*C_DS1 * C_STEM
            + 24*10*C_DS1 * 5*3 + 24*10*C_DS2 * C_DS1
            + 2 * (24*10*C_DS2 * 5*3 + 24*10*C_DS2 * C_DS2))
    print(f"\nMACs            : {macs:,}   (v4 had 2,180,000)")
    print(f"  v4 measured 105.6 ms on device; scaling by MACs alone")
    print(f"  suggests roughly {105.6 * macs / 2180000:.0f} ms, likely better")
    print(f"  because 16/32 channels hit ESP-NN paths that 12/24/40 missed.")

    print("\nPredicted tensor arena: 44,332 B  (v4 measured 76,188 B)")
    print("Confirm after export with:  python arena_calc.py <model>.tflite")
