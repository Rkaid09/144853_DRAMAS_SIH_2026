import os, sys, csv, json, collections, zlib
import numpy as np
import prahari_config as cfg
from logmel import log_mel, load_wav_int16, quantise, dequantise
from augment import augment

REPLICAS = {
    "train": {"keyword": 2, "confusable": 2, "speech": 2, "noise": 2,
              "silence": 2, "hardneg": 1, "cvspeech": 1, "boardnoise": 3, "event": 3},
    "val":   {"keyword": 2, "confusable": 2, "speech": 2, "noise": 2,
              "silence": 2, "hardneg": 0, "cvspeech": 1, "boardnoise": 1, "event": 1},
    "test":  {"keyword": 2, "confusable": 2, "speech": 2, "noise": 2,
              "silence": 2, "hardneg": 0, "cvspeech": 1, "boardnoise": 1, "event": 1},
}

TTS_ROOT = os.path.join("..", "tools", "dataset_tts")
NEG_ROOT = os.path.join("..", "tools", "dataset_neg")
HARD_ROOT = os.path.join("..", "tools", "dataset_hardneg")
CV_ROOT   = os.path.join("..", "tools", "dataset_cv")
_full = os.path.join("..", "tools", "dataset_cv_full")
if os.environ.get("PRAHARI_CV_ROOT"):
    CV_ROOT = os.environ["PRAHARI_CV_ROOT"]
BOARD_ROOT = os.path.join("..", "tools", "dataset_board")
ESC_EV_ROOT = os.path.join("..", "tools", "dataset_esc_events")
OUTDIR   = "data"
VALTEST_ONLY = os.environ.get("PRAHARI_VALTEST_ONLY") == "1"
SEED     = 1337
TARGET   = {"train": 0.70, "val": 0.15, "test": 0.15}


def read_rows():
    out = []

    p = os.path.join(TTS_ROOT, "manifest.csv")
    for r in csv.DictReader(open(p, encoding="utf-8")):
        out.append({
            "path": os.path.join(TTS_ROOT, r["file"].replace("/", os.sep)),
            "label": 1 if r["label"] == "positive" else 0,
            "group": f"tts_{r['voice_id']}",
            "kind": "keyword" if r["label"] == "positive" else "confusable",
        })

    p = os.path.join(NEG_ROOT, "manifest.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            grp = r["file"] if r["kind"] == "silence" else r["speaker"]
            out.append({
                "path": os.path.join(NEG_ROOT, r["file"].replace("/", os.sep)),
                "label": 0,
                "group": grp,
                "kind": r["kind"],
            })
    else:
        print(f"WARNING: {p} missing - negatives not merged")

    p = os.path.join(HARD_ROOT, "manifest.csv")
    if os.path.exists(p):
        n = 0
        for r in csv.DictReader(open(p, encoding="utf-8")):
            out.append({
                "path": os.path.join(HARD_ROOT, r["file"].replace("/", os.sep)),
                "label": 0,
                "group": r["speaker"],
                "kind": "hardneg",
            })
            n += 1
        print(f"merged {n} mined hard negatives (train only)")

    p = os.path.join(CV_ROOT, "manifest.csv")
    if os.path.exists(p):
        n = skipped = 0
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r.get("split") != "train":
                skipped += 1
                continue
            out.append({
                "path": os.path.join(CV_ROOT, r["file"].replace("/", os.sep)),
                "label": 0,
                "group": r["speaker"],
                "kind": "cvspeech",
            })
            n += 1
        print(f"Common Voice source: {CV_ROOT}")
        print(f"merged {n} Common Voice Hindi windows "
              f"({skipped} held-out-speaker windows deliberately skipped)")
    else:
        print(f"note: {p} missing - run tools/prep_common_voice.py to add "
              f"real Hindi speech negatives")
    p = os.path.join(ESC_EV_ROOT, "manifest.csv")
    if os.path.exists(p):
        n = 0
        for r in csv.DictReader(open(p, encoding="utf-8")):
            out.append({
                "path": os.path.join(ESC_EV_ROOT, r["file"]),
                "label": 0,
                "group": r["speaker"],
                "kind": "event",
            })
            n += 1
        print(f"merged {n} event-centred ESC-50 windows (coughs, claps, taps, ...)")
    else:
        print(f"note: {p} missing - run tools/prep_esc_events.py")

    p = os.path.join(BOARD_ROOT, "manifest.csv")
    if os.path.exists(p):
        n = 0
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r.get("split") != "train":
                continue
            out.append({
                "path": os.path.join(BOARD_ROOT, r["file"].replace("/", os.sep)),
                "label": 0,
                "group": r["speaker"],
                "kind": "boardnoise",
            })
            n += 1
        print(f"merged {n} board room-recording windows (held-out tails skipped)")
    return out


DOMAIN_OF = {"keyword": "tts", "confusable": "tts",
             "speech": "speech", "noise": "noise", "silence": "silence",
             "hardneg": "hardneg", "cvspeech": "cvspeech",
             "boardnoise": "board", "event": "events"}


def assign_groups(rows, domain, seed):
    sub = [r for r in rows if DOMAIN_OF[r["kind"]] == domain]
    if not sub:
        return {}
    counts = collections.Counter(r["group"] for r in sub)
    total = sum(counts.values())

    groups = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    rng = np.random.RandomState(seed + hash(domain) % 1000)
    rng.shuffle(groups)
    groups.sort(key=lambda kv: -kv[1])

    assigned, got = {}, {k: 0 for k in TARGET}
    for g, n in groups:
        deficit = {k: TARGET[k] * total - got[k] for k in TARGET}
        pick = max(deficit, key=deficit.get)
        assigned[g] = pick
        got[pick] += n
    return assigned


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = read_rows()
    print(f"rows            : {len(rows)}")
    print("by kind         :", dict(collections.Counter(r["kind"] for r in rows)))
    print(f"positives       : {sum(r['label'] for r in rows)}")
    print(f"distinct groups : {len({r['group'] for r in rows})}\n")

    amap = {}
    for dom in sorted(set(DOMAIN_OF.values())):
        m = assign_groups(rows, dom, SEED)
        amap.update(m)
        print(f"  domain {dom:<9} {len(m):>5} groups assigned")
    print()

    buckets = {k: {"X": [], "y": [], "g": set(), "kinds": collections.Counter()}
               for k in TARGET}
    missing = 0
    for i, r in enumerate(rows, 1):
        if not os.path.exists(r["path"]):
            missing += 1
            continue
        try:
            pcm = load_wav_int16(r["path"])
        except Exception:
            missing += 1
            continue

        if r["group"] not in amap:
            sys.exit(f"group {r['group']!r} (kind {r['kind']!r}) was never "
                     f"assigned a split.\n"
                     f"  Its domain is {DOMAIN_OF.get(r['kind'])!r}. Every kind "
                     f"needs an entry in DOMAIN_OF, and every domain is "
                     f"assigned automatically from it.")
        s = amap[r["group"]]
        if s == "train" and VALTEST_ONLY:
            continue
        n_rep = REPLICAS[s].get(r["kind"], 1)

        for rep in range(n_rep):
            if rep == 0 and r["kind"] == "cvspeech" and zlib.crc32(r["path"].encode()) % 2:
                audio = augment(pcm, zlib.crc32(r["path"].encode()))
            elif rep == 0:
                audio = pcm
            else:
                seed = zlib.crc32(r["path"].encode()) + rep * 7919
                audio = augment(pcm, seed)

            lm = dequantise(quantise(log_mel(audio)))
            if lm.shape != (cfg.CTX_FRAMES, cfg.MEL_BINS):
                continue
            buckets[s]["X"].append(lm.astype(np.float16))
            buckets[s]["y"].append(r["label"])
            buckets[s]["kinds"][r["kind"]] += 1
            buckets[s].setdefault("k", []).append(r["kind"])
            buckets[s].setdefault("r", []).append(rep)

        buckets[s]["g"].add(r["group"])
        if i % 500 == 0:
            print(f"  {i}/{len(rows)} source clips")
    if missing:
        print(f"  ({missing} files missing)")

    print("\nSPLIT SUMMARY")
    print("-" * 66)
    print(f"{'split':<8}{'files':>7}{'groups':>8}{'pos':>7}{'neg':>7}{'pos%':>8}{'neg:pos':>10}")
    meta = {"seed": SEED, "splits": {}}
    for k in (("val", "test") if VALTEST_ONLY else ("train", "val", "test")):
        X = np.stack(buckets[k]["X"])
        X = X if k == "train" else X.astype(np.float32)
        y = np.array(buckets[k]["y"], dtype=np.int8)
        np.save(os.path.join(OUTDIR, f"X_{k}.npy"), X)
        np.save(os.path.join(OUTDIR, f"y_{k}.npy"), y)
        np.save(os.path.join(OUTDIR, f"kind_{k}.npy"),
                np.array(buckets[k].get("k", []), dtype="<U16"))
        np.save(os.path.join(OUTDIR, f"rep_{k}.npy"),
                np.array(buckets[k].get("r", []), dtype=np.int16))
        pos, neg = int((y == 1).sum()), int((y == 0).sum())
        print(f"{k:<8}{len(y):>7}{len(buckets[k]['g']):>8}{pos:>7}{neg:>7}"
              f"{pos/len(y)*100:>7.1f}%{neg/max(pos,1):>9.1f}:1")
        meta["splits"][k] = {"files": len(y), "positives": pos, "negatives": neg,
                             "groups": len(buckets[k]["g"]),
                             "kinds": dict(buckets[k]["kinds"])}

    print("\ncomposition by kind")
    for k in ("train", "val", "test"):
        print(f"  {k:<6}", dict(buckets[k]["kinds"]))

    with open(os.path.join(OUTDIR, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    leak = False
    for a in TARGET:
        for b in TARGET:
            if a < b and buckets[a]["g"] & buckets[b]["g"]:
                print(f"\n!! LEAK between {a} and {b}")
                leak = True
    print("\nspeaker-disjoint:", "FAILED" if leak else
          "PASSED - no group appears in more than one split")

    if not VALTEST_ONLY:
        X = np.load(os.path.join(OUTDIR, "X_train.npy"))
        print(f"X_train         : {X.shape}  range {X.min():.2f}..{X.max():.2f}")


if __name__ == "__main__":
    main()
