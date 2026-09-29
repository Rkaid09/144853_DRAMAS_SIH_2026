#!/usr/bin/env python3
import argparse, collections, csv, os, random, sys, wave
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_common_voice import load_any, to_16k, SRC_DIR, SR

OUT = "dataset_cv_full"
WIN = SR


def decode(path):
    try:
        x, sr = load_any(path)
        return to_16k(x, sr)
    except SystemExit:
        raise
    except Exception:
        return None


def write_wav(path, x):
    y = np.clip(x * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(y.tobytes())


def work(job):
    split, short, rel, base = job
    x = decode(os.path.join(SRC_DIR, "clips", rel))
    if x is None or len(x) < WIN // 2:
        return []
    rows = []
    stem = os.path.splitext(rel)[0]
    if split == "eval":
        fn = f"{short}_{stem}.wav"
        write_wav(os.path.join(OUT, "eval_clips", fn), x)
        rows.append([f"eval_clips/{fn}", "negative", "cvspeech", short, rel, "eval"])
        return rows
    k = 0
    for s in range(0, max(len(x) - WIN + 1, 0), WIN):
        w = x[s:s + WIN]
        if float(np.sqrt(np.mean(w * w))) < 0.004:
            continue
        fn = f"{short}_{stem}_{k}.wav"
        write_wav(os.path.join(OUT, "train", fn), w)
        rows.append([f"train/{fn}", "negative", "cvspeech", short, rel, "train"])
        k += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-speaker", type=int, default=400,
                    help="max TRAIN windows per speaker (diversity over volume)")
    ap.add_argument("--max-train", type=int, default=45000,
                    help="total cap on TRAIN windows (RAM: ~15.7 KB each as features)")
    ap.add_argument("--eval-speakers", type=int, default=80)
    ap.add_argument("--eval-clips", type=int, default=1500,
                    help="max whole clips kept for the held-out FA/hour test")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    val = list(csv.DictReader(open(os.path.join(SRC_DIR, "validated.tsv"), encoding="utf-8"), delimiter="\t"))
    oth = []
    p = os.path.join(SRC_DIR, "other.tsv")
    if os.path.exists(p):
        oth = list(csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"))

    spk_val = collections.defaultdict(list)
    for r in val:
        spk_val[r["client_id"]].append(r["path"])
    speakers = sorted(spk_val)
    random.Random(1337).shuffle(speakers)
    n_eval = min(a.eval_speakers, len(speakers) // 3)
    eval_spk = set(speakers[:n_eval])

    by_spk = collections.defaultdict(list)
    for r in val + oth:
        by_spk[r["client_id"]].append(r["path"])

    rng = random.Random(7)
    jobs, eval_jobs = [], []
    for spk, clips in sorted(by_spk.items()):
        short = "cv_" + spk[:12]
        clips = sorted(set(clips))
        rng.shuffle(clips)
        if spk in eval_spk:
            eval_jobs += [("eval", short, c, 0) for c in clips]
        else:
            jobs += [("train", short, c, 0) for c in clips[: max(1, a.per_speaker // 3)]]
    rng.shuffle(eval_jobs)
    eval_jobs = eval_jobs[: a.eval_clips]

    for sub in ("train", "eval_clips"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    print(f"clips: {len(val)} validated + {len(oth)} other, {len(by_spk)} speakers")
    print(f"held out: {n_eval} speakers ({len(eval_jobs)} whole clips kept for FA/hour)")
    print(f"decoding {len(jobs)} train clips + {len(eval_jobs)} eval clips on {a.workers} workers ...")

    per_spk = collections.Counter()
    rows, done, total_train = [], 0, 0
    with Pool(a.workers) as pool:
        for res in pool.imap_unordered(work, jobs + eval_jobs, chunksize=8):
            done += 1
            for r in res:
                if r[5] == "train":
                    if per_spk[r[3]] >= a.per_speaker or total_train >= a.max_train:
                        continue
                    per_spk[r[3]] += 1
                    total_train += 1
                rows.append(r)
            if done % 500 == 0:
                print(f"  {done}/{len(jobs) + len(eval_jobs)} clips   train windows {total_train}", flush=True)

    with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "label", "kind", "speaker", "source", "split"])
        wr.writerows(rows)
    n_eval_rows = sum(1 for r in rows if r[5] == "eval")
    print(f"\ntrain windows : {total_train:,} from {len(per_spk)} speakers "
          f"(~{total_train/3600:.1f} h of speech; was 4,513 = 1.25 h)")
    print(f"eval clips    : {n_eval_rows:,} whole clips from {n_eval} unseen speakers")
    print(f"wrote {OUT}/manifest.csv")
    print("note: train windows beyond the caps were written to disk but are not in the manifest")


if __name__ == "__main__":
    main()
