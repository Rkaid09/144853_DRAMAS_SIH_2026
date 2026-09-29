#!/usr/bin/env python3
import csv, glob, os, sys, wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(HERE, "..", "server", "captures")
OUT = os.path.join(HERE, "dataset_board")
POOL = os.path.join(HERE, "dataset_board_noise")
SR, WIN, HOP = 16000, 16000, 8000
EVAL_FRAC = 0.25


def room_recordings():
    return [f for f in sorted(glob.glob(os.path.join(CAP, "*-room*.wav")))
            if os.path.getsize(f) > 20 * SR * 2]


def main():
    files = room_recordings()
    if not files:
        sys.exit("no 60 s room recordings in server/captures - record some with `n`")
    for d in (os.path.join(OUT, "train"), POOL):
        os.makedirs(d, exist_ok=True)
    man = open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8")
    wr = csv.writer(man)
    wr.writerow(["file", "label", "kind", "speaker", "source", "split"])
    n_win, secs = 0, 0.0
    for f in files:
        with wave.open(f) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        secs += len(x) / SR
        seg = x[: int(len(x) * (1 - EVAL_FRAC))]
        name = os.path.splitext(os.path.basename(f))[0]
        k = 0
        for s in range(0, len(seg) - WIN + 1, HOP):
            fn = f"{name}_{k:03d}.wav"
            for d in (os.path.join(OUT, "train"), POOL):
                with wave.open(os.path.join(d, fn), "wb") as o:
                    o.setnchannels(1); o.setsampwidth(2); o.setframerate(SR)
                    o.writeframes(seg[s:s + WIN].tobytes())
            wr.writerow([f"train/{fn}", "negative", "boardnoise", "board_" + name,
                         os.path.basename(f), "train"])
            k += 1
        n_win += k
        print(f"  {os.path.basename(f):40} {len(x)/SR:5.1f} s -> {k} windows "
              f"(last {EVAL_FRAC:.0%} held out)")
    man.close()
    print(f"\n{len(files)} recordings, {secs/60:.1f} min of board noise -> {n_win} windows")
    print(f"noise pool for augment.py : {POOL}")
    print(f"negatives for build_dataset: {OUT}/manifest.csv")


if __name__ == "__main__":
    main()
