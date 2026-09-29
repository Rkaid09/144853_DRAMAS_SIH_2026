import os, sys
import numpy as np
import tensorflow as tf

import prahari_config as cfg
from model import build_model, STEM_KERNEL, C_STEM

OUT = os.path.join("artifacts", "probe.keras")
BASELINE = (3, 5)


NS_GENERAL = 19.6
NS_IM2COL  = 39.6
STEM_PIXELS = cfg.CTX_FRAMES * (cfg.MEL_BINS // 2) * 8


def esp_nn_path(time_taps, mel_taps, in_ch=1):
    filter_ht, filter_wd = time_taps, mel_taps
    row = filter_wd * in_ch
    window = filter_wd * filter_ht * in_ch

    if filter_wd == 3 and filter_ht == 3 and in_ch >= 16 and in_ch % 16 == 0:
        return "3x3 optimised", window, window, NS_GENERAL

    if row < 16 and window >= 16:
        return "im2col (2x slower per slot)", window, (window + 15) & ~15, NS_IM2COL

    padded_row = ((row + 15) // 16) * 16
    return "general (assembly)", window, padded_row * filter_ht, NS_GENERAL


def report(time_taps, mel_taps, label=""):
    name, window, slots, ns = esp_nn_path(time_taps, mel_taps)
    us = STEM_PIXELS * slots * ns / 1000.0
    print(f"  {time_taps:>2}x{mel_taps:<2}{label:<12} slots {slots:>3}  "
          f"useful {100.0*window/slots:5.1f}%  predicted {us/1000.0:6.2f} ms   {name}")
    return us


def main():
    t, m = STEM_KERNEL
    print(f"stem kernel     : {t}x{m} (time x mel), from "
          f"PRAHARI_STEM={os.environ.get('PRAHARI_STEM', '<unset, default 3x5>')}")
    print(f"output channels : {C_STEM}")

    print("\nESP-NN dispatch and cost, from constants measured on the board:")
    base_us = report(*BASELINE, label=" baseline")
    report(6, 5, label=" probed")
    this_us = report(t, m, label=" <- THIS")
    print("  reference points:")
    for cand in [(3, 16), (2, 16), (1, 16), (2, 8)]:
        report(*cand)

    if (t, m) != BASELINE:
        print(f"\n  predicted stem  : {this_us/1000.0:.2f} ms "
              f"(baseline measured 14.73 ms)")
        print(f"  predicted total : "
              f"{(30.07 - 14.73 + this_us/1000.0):.2f} ms  (now 30.07 ms)")
    print("\n  Predictions. The board is the authority - 6x5 was predicted")
    print("  9.8 ms and measured 19.86 ms, which is why this model is now")
    print("  fitted to measurements instead of to the dispatcher's comments.")

    rf = t + (5 - 1) * 2 + (5 - 1) * 4 + (5 - 1) * 8 + (5 - 1) * 8
    print(f"\ntime receptive field: {rf} frames = {rf * 10} ms "
          f"(baseline {BASELINE[0] + (5-1)*2 + (5-1)*4 + (5-1)*8 + (5-1)*8} frames). "
          f"A spoken प्रहरी is 600-700 ms.")

    model = build_model()
    os.makedirs("artifacts", exist_ok=True)
    model.save(OUT)
    n = model.count_params()
    print(f"\nwrote {OUT}   ({n:,} parameters, UNTRAINED - random weights)")
    print("\nTHE SCORES THIS MODEL PRODUCES ARE MEANINGLESS. It has never seen")
    print("audio. The device self-test will still say MATCH, because both")
    print("sides compute the same random function - that check is verifying")
    print("the C chain, not the model. Live detection will be nonsense.")
    print("Read operator 0's time from p50. That is the only number here.")
    print("\nnext:")
    print("  python export_tflite.py artifacts/probe.keras")
    print("  python gen_test_clip.py")
    print("  cd C:\\PRAHARI  &&  pio run -e xiao_esp32s3 -t upload")


if __name__ == "__main__":
    main()
