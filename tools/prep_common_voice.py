import os, sys, csv, wave, argparse, random, collections

import numpy as np

SRC_DIR = "cv-corpus-27.0-2026-09-11/hi"
OUT     = "dataset_cv"
SR      = 16000
WIN     = SR


def load_any(path):
    try:
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        return x.mean(axis=1), sr
    except ImportError:
        pass
    except Exception:
        pass
    import shutil, subprocess
    if shutil.which("ffmpeg"):
        p = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-i", path, "-f", "s16le",
             "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SR), "-"],
            capture_output=True)
        if p.returncode == 0 and p.stdout:
            return np.frombuffer(p.stdout, "<i2").astype(np.float32) / 32768.0, SR
    raise SystemExit(
        "Cannot decode mp3.\n"
        "  Easiest fix:   pip install soundfile\n"
        "  (libsndfile 1.2 ships with it and reads mp3 directly.)\n"
        "  Alternative:   install ffmpeg and put it on PATH")


def to_16k(x, sr):
    if sr == SR:
        return x
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sr), SR)
        return resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    except ImportError:
        n = int(round(len(x) * SR / float(sr)))
        return np.interp(np.linspace(0, len(x) - 1, n),
                         np.arange(len(x)), x).astype(np.float32)


def write_wav(path, x):
    y = np.clip(x * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(y.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-speaker", type=int, default=25,
                    help="max CLIPS per speaker, to fight the 62%% skew")
    ap.add_argument("--windows-per-speaker", type=int, default=20,
                    help="max 1 s windows kept per speaker")
    ap.add_argument("--eval-speakers", type=int, default=80,
                    help="speakers held out ENTIRELY for evaluation")
    ap.add_argument("--min-rms", type=float, default=0.004,
                    help="drop near-silent windows; they teach nothing and "
                         "room tone already measures 0%% false accepts")
    args = ap.parse_args()

    tsv = os.path.join(SRC_DIR, "validated.tsv")
    if not os.path.exists(tsv):
        sys.exit(f"no {tsv}\n"
                 f"  extract the archive first:\n"
                 f"    tar xzf 1789492443485-cv-corpus-27.0-2026-09-11-hi.tar.gz")

    rows = list(csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t"))
    by_spk = collections.defaultdict(list)
    for r in rows:
        by_spk[r["client_id"]].append(r["path"])
    speakers = sorted(by_spk)
    rng = random.Random(1337)
    rng.shuffle(speakers)

    n_eval = min(args.eval_speakers, len(speakers) // 3)
    eval_spk = set(speakers[:n_eval])
    print(f"corpus     : {len(rows):,} clips, {len(speakers)} speakers")
    print(f"held out   : {n_eval} speakers for EVALUATION (never trained on)")
    print(f"training   : {len(speakers) - n_eval} speakers")
    print(f"caps       : {args.per_speaker} clips and "
          f"{args.windows_per_speaker} windows per speaker\n")

    for sub in ("train", "eval"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    man = open(os.path.join(OUT, "manifest.csv"), "w",
               newline="", encoding="utf-8")
    wr = csv.writer(man)
    wr.writerow(["file", "label", "kind", "speaker", "source", "split"])

    kept = collections.Counter()
    dropped_quiet = 0
    for si, spk in enumerate(speakers):
        split = "eval" if spk in eval_spk else "train"
        short = "cv_" + spk[:12]
        clips = by_spk[spk][:args.per_speaker]
        made = 0
        for rel in clips:
            if made >= args.windows_per_speaker:
                break
            path = os.path.join(SRC_DIR, "clips", rel)
            if not os.path.exists(path):
                continue
            try:
                x, sr = load_any(path)
                x = to_16k(x, sr)
            except SystemExit:
                raise
            except Exception:
                continue
            for s in range(0, max(len(x) - WIN + 1, 0), WIN):
                if made >= args.windows_per_speaker:
                    break
                w = x[s:s + WIN]
                if float(np.sqrt(np.mean(w * w))) < args.min_rms:
                    dropped_quiet += 1
                    continue
                name = f"{short}_{made:03d}.wav"
                write_wav(os.path.join(OUT, split, name), w)
                wr.writerow([f"{split}/{name}", "negative", "cvspeech",
                             short, f"common_voice/{rel}", split])
                made += 1
        kept[split] += made
        if si % 25 == 0:
            print(f"  {si:>3}/{len(speakers)} speakers   "
                  f"train {kept['train']:,}  eval {kept['eval']:,}", flush=True)
        man.flush()
    man.close()

    print(f"\ntrain windows : {kept['train']:,}")
    print(f"eval  windows : {kept['eval']:,}   "
          f"({n_eval} speakers, none of them in training)")
    print(f"dropped quiet : {dropped_quiet:,}")
    print(f"\nwrote {OUT}/manifest.csv")
    print("\nNEXT")
    print("  1. build_dataset.py: add 'cvspeech' to DOMAIN_OF and give it a")
    print("     REPLICAS count. Watch the ratio - this many negatives next")
    print("     to ~1,200 positives will teach the model to always say no.")
    print("  2. The eval split is for measuring speech false-accepts with")
    print("     enough samples to be believable. The four human speakers")
    print("     stay frozen as the channel-truth check; this is a different")
    print("     microphone and a different room, and it cannot replace them.")


if __name__ == "__main__":
    main()
