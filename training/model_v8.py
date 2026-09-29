import tensorflow as tf
from tensorflow.keras import layers, Model
import prahari_config as cfg

C_STEM, C_DS1, C_DS2 = 8, 16, 32


def ds_block(x, channels, name, kernel=(5, 3)):
    shortcut = x
    x = layers.DepthwiseConv2D(kernel, padding="same", use_bias=False, name=f"{name}_dw")(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = layers.ReLU(name=f"{name}_dw_relu")(x)
    x = layers.Conv2D(channels, (1, 1), use_bias=False, name=f"{name}_pw")(x)
    x = layers.BatchNormalization(name=f"{name}_pw_bn")(x)
    x = layers.ReLU(name=f"{name}_pw_relu")(x)
    if shortcut.shape[-1] == channels:
        x = layers.Add(name=f"{name}_add")([shortcut, x])
    return x


VARIANTS = {
    "v5  baseline 4 blocks":      dict(tail=2, pool_tail=False),
    "A   3 blocks (no ds4)":      dict(tail=1, pool_tail=False),
    "B   2 blocks (no ds3/ds4)":  dict(tail=0, pool_tail=False),
    "C   4 blocks, pooled tail":  dict(tail=2, pool_tail=True),
    "D   3 blocks, pooled tail":  dict(tail=1, pool_tail=True),
}


def build_v8(tail=2, pool_tail=False, dropout=0.15):
    inp = layers.Input(shape=(cfg.CTX_FRAMES, cfg.MEL_BINS, 1), name="logmel")
    x = layers.BatchNormalization(name="input_bn")(inp)

    x = layers.Conv2D(C_STEM, (3, 5), strides=(1, 2), padding="same",
                      use_bias=False, name="stem")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    x = layers.MaxPooling2D((2, 1), name="pool1")(x)
    x = ds_block(x, C_DS1, "ds1")
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = ds_block(x, C_DS2, "ds2")

    if pool_tail and tail > 0:
        x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    for i in range(tail):
        x = ds_block(x, C_DS2, f"ds{3+i}")

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="drop")(x)
    out = layers.Dense(1, activation="sigmoid", name="score")(x)
    return Model(inp, out, name="prahari_v8")


def macs(tail=2, pool_tail=False):
    k = 5 * 3
    stem = 98 * 20 * C_STEM * 3 * 5
    ds1 = 49 * 20 * C_STEM * k + 49 * 20 * C_DS1 * C_STEM
    ds2 = 24 * 10 * C_DS1 * k + 24 * 10 * C_DS2 * C_DS1
    h, w = (12, 5) if pool_tail else (24, 10)
    one = h * w * C_DS2 * k + h * w * C_DS2 * C_DS2
    return stem + ds1 + ds2 + tail * one


V5_MACS, V5_MS = macs(), 64.3


def est_ms(m):
    return V5_MS * m / V5_MACS


if __name__ == "__main__":
    print(f"{'variant':<28}{'MACs':>10}{'vs v5':>8}{'est ms':>9}"
          f"{'idle @640':>11}{'idle @960':>11}")
    print("-" * 77)
    for name, kw in VARIANTS.items():
        m = macs(**kw); ms = est_ms(m)
        i640 = 5.0 + 100 * ms / 640
        i960 = 5.0 + 100 * ms / 960
        print(f"{name:<28}{m:>10,}{m/V5_MACS:>7.0%}{ms:>9.1f}"
              f"{i640:>10.1f}%{i960:>10.1f}%")
    print("\nidle columns = 5.0% measured front-end + inference at that scan period.")
    print("Under 10% is the target. These are ESTIMATES - the board decides.")
