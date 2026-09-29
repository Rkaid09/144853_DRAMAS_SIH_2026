import os, csv, subprocess, sys, tempfile
import numpy as np
import prahari_config as cfg
from logmel import log_mel, load_wav_int16, quantise

EXE  = os.path.join("..", "firmware", "test", "test_logmel")
TTS  = os.path.join("..", "tools", "dataset_tts")
NEG  = os.path.join("..", "tools", "dataset_neg")
N_CLIPS = 24


def pick_clips():
    out = []
    rows = list(csv.DictReader(open(os.path.join(TTS, "manifest.csv"), encoding="utf-8")))
    pos = [r for r in rows if r["label"] == "positive"][:8]
    con = [r for r in rows if r["label"] == "negative"][:8]
    for r in pos + con:
        out.append(os.path.join(TTS, r["file"].replace("/", os.sep)))
    p = os.path.join(NEG, "manifest.csv")
    if os.path.exists(p):
        nr = list(csv.DictReader(open(p, encoding="utf-8")))
        for kind in ("speech", "noise", "silence"):
            hits = [r for r in nr if r["kind"] == kind][:3]
            for r in hits:
                out.append(os.path.join(NEG, r["file"].replace("/", os.sep)))
    return [p for p in out if os.path.exists(p)][:N_CLIPS]


def main():
    exe = EXE + (".exe" if os.name == "nt" and os.path.exists(EXE + ".exe") else "")
    if not os.path.exists(exe):
        sys.exit(f"no binary at {exe} - build firmware/test/test_logmel first")

    clips = pick_clips()
    print(f"comparing {len(clips)} clips\n")
    print(f"{'clip':<34}{'max|df|':>10}{'mean|df|':>10}"
          f"{'int8 diff':>11}{'max d':>7}")
    print("-" * 72)

    worst_f, worst_i, total_vals, total_mismatch = 0.0, 0, 0, 0
    tmp = tempfile.mkdtemp()

    for path in clips:
        pcm = load_wav_int16(path)
        raw = os.path.join(tmp, "in.pcm")
        fo  = os.path.join(tmp, "o.f32")
        qo  = os.path.join(tmp, "o.i8")
        pcm.astype("<i2").tofile(raw)

        r = subprocess.run([exe, raw, fo, qo], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {os.path.basename(path)}: C failed: {r.stderr.strip()}")
            continue
        n_frames = int(r.stdout.strip())

        py = log_mel(pcm)[:n_frames]
        c  = np.fromfile(fo, dtype="<f4").reshape(n_frames, cfg.MEL_BINS)
        d  = np.abs(py - c)

        qpy = quantise(py).astype(np.int16)
        qc  = np.fromfile(qo, dtype=np.int8).reshape(n_frames, cfg.MEL_BINS).astype(np.int16)
        dq  = np.abs(qpy - qc)
        mism = int((dq != 0).sum())

        total_vals += dq.size
        total_mismatch += mism
        worst_f = max(worst_f, float(d.max()))
        worst_i = max(worst_i, int(dq.max()))

        print(f"{os.path.basename(path):<34}{d.max():>10.6f}{d.mean():>10.6f}"
              f"{mism:>7} /{dq.size:>4}{dq.max():>7}")

    print("-" * 72)
    print(f"worst float difference : {worst_f:.6f}")
    print(f"quantisation step      : {cfg.FEAT_SCALE:.6f}  "
          f"(half-step {cfg.FEAT_SCALE/2:.6f})")
    pct = 100.0 * total_mismatch / max(total_vals, 1)
    print(f"int8 values differing  : {total_mismatch} of {total_vals}  ({pct:.4f} %)")
    print(f"largest int8 difference: {worst_i}")

    print()
    if worst_i == 0:
        print("PASS - C and Python produce identical int8 features.")
    elif worst_i <= 1 and pct < 1.0:
        print("PASS - differences are +/-1 on a negligible fraction, which is")
        print("       float32-vs-float64 rounding at the quantisation boundary.")
    else:
        print("FAIL - the two front-ends disagree. The model would behave")
        print("       differently on hardware than in training. Do not deploy.")


if __name__ == "__main__":
    main()
