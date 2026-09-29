import os, sys, csv, glob, wave, random, argparse
import numpy as np

SR          = 16000
WIN         = SR
SKIP_CAPTURE_S = 1.6
OUT_ROOT    = "dataset_hardneg"
SEED        = 1337


def read_wav(p):
    with wave.open(p, "rb") as w:
        if w.getframerate() != SR or w.getsampwidth() != 2:
            return None
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)
    return x


def write_wav(p, x):
    with wave.open(p, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(np.clip(x, -32768, 32767).astype(np.int16).tobytes())


def build_streams(rng, n_streams, clips_per_stream, sources):
    out = []
    for _ in range(n_streams):
        parts, tag = [], None
        for _ in range(clips_per_stream):
            src, pool = rng.choice(sources)
            if not pool:
                continue
            x = read_wav(rng.choice(pool))
            if x is None or len(x) < 800:
                continue
            tag = tag or src
            parts.append(x)
            g = rng.choice([0, 0, int(0.05*SR), int(0.12*SR)])
            if g:
                parts.append(np.zeros(g, dtype=np.int16))
        if parts:
            s = np.concatenate(parts)
            if len(s) > WIN + SR:
                out.append((tag, s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=3000, help="hardest windows to keep")
    ap.add_argument("--stratified", action="store_true", default=True,
                    help="keep the hardest N from EACH source, not N overall")
    ap.add_argument("--no-stratified", dest="stratified", action="store_false")
    ap.add_argument("--candidates", type=int, default=30000)
    ap.add_argument("--model", default=os.path.join("..", "training", "artifacts",
                                                    "prahari_int8.tflite"))
    ap.add_argument("--no-score", action="store_true",
                    help="skip mining, keep random windows (needs no TensorFlow)")
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    rng = random.Random(SEED)

    captures = sorted(glob.glob(os.path.join("..", "server", "captures", "*.wav")))
    sc = sorted(glob.glob(os.path.join("speech_commands", "**", "*.wav"), recursive=True))
    rng.shuffle(sc); sc = sc[:8000]
    conf = sorted(glob.glob(os.path.join("dataset_tts", "negative", "*.wav")))
    print(f"sources: {len(captures)} device captures, {len(sc)} speech-commands, "
          f"{len(conf)} confusables")
    if not (captures or sc or conf):
        sys.exit("no source audio found")

    cands = []

    for p in captures:
        x = read_wav(p)
        if x is None:
            continue
        start = int(SKIP_CAPTURE_S * SR)
        x = x[start:]
        for _ in range(max(1, (len(x) - WIN) // (SR // 2))):
            if len(x) <= WIN:
                break
            o = rng.randint(0, len(x) - WIN)
            cands.append(("capture", x[o:o + WIN]))

    per = max(1, (args.candidates - len(cands)) // 2)
    for src, pool, n_clip, share in (("speech_stream", sc, 6, 0.70),
                                     ("confusable_stream", conf, 5, 0.30)):
        streams = build_streams(rng, max(1, int(per * 2 * share) // 12), n_clip, [(src, pool)])
        for tag, s in streams:
            for _ in range(12):
                o = rng.randint(0, len(s) - WIN)
                cands.append((tag, s[o:o + WIN]))

    rng.shuffle(cands)
    cands = cands[:args.candidates]
    print(f"{len(cands)} candidate windows built")

    if args.no_score:
        keep = [(0.0, t, w) for t, w in cands[:args.keep]]
        print("scoring skipped (--no-score): keeping a random subset")
    else:
        sys.path.insert(0, os.path.join("..", "training"))
        import tensorflow as tf
        import prahari_config as cfg
        from logmel import log_mel, quantise, dequantise
        it = tf.lite.Interpreter(model_path=args.model); it.allocate_tensors()
        inp, out = it.get_input_details()[0], it.get_output_details()[0]
        s_in, z_in = inp["quantization"]; s_out, z_out = out["quantization"]
        scored = []
        for i, (tag, w) in enumerate(cands):
            f = dequantise(quantise(log_mel(w)))
            if f.shape[0] < cfg.CTX_FRAMES:
                continue
            f = f[:cfg.CTX_FRAMES][None, ..., None]
            q = np.clip(np.round(f / s_in) + z_in, -128, 127).astype(np.int8)
            it.set_tensor(inp["index"], q); it.invoke()
            v = (it.get_tensor(out["index"])[0][0].astype(np.float32) - z_out) * s_out
            scored.append((float(v), tag, w))
            if (i + 1) % 2000 == 0:
                print(f"   scored {i+1}/{len(cands)}")
        scored.sort(key=lambda r: -r[0])
        hi = sum(1 for s, _, _ in scored if s >= 0.70)
        print(f"\ncandidates scoring >=0.70 (would false-fire): {hi} "
              f"of {len(scored)}  = {100*hi/max(len(scored),1):.1f}%")
        print("\nper-source false-fire rate (>=0.70):")
        import collections as _c
        tot_s = _c.Counter(t for _, t, _ in scored)
        hi_s  = _c.Counter(t for v, t, _ in scored if v >= 0.70)
        for t in sorted(tot_s):
            print(f"   {t:<20}{hi_s[t]:>6} / {tot_s[t]:<6} = {100*hi_s[t]/tot_s[t]:>5.1f}%")

        if args.stratified:
            per = args.keep // max(len(tot_s), 1)
            keep, bys = [], _c.defaultdict(list)
            for v, t, w in scored:
                if len(bys[t]) < per:
                    bys[t].append((v, t, w))
            for t in sorted(bys):
                keep += bys[t]
                print(f"   kept {len(bys[t]):>5} from {t:<20} "
                      f"scores {bys[t][0][0]:.3f} .. {bys[t][-1][0]:.3f}")
            rng.shuffle(keep)
        else:
            keep = scored[:args.keep]
            print(f"keeping the {len(keep)} hardest overall, scores "
                  f"{keep[0][0]:.3f} down to {keep[-1][0]:.3f}")

    os.makedirs(os.path.join(OUT_ROOT, "hardneg"), exist_ok=True)
    man = open(os.path.join(OUT_ROOT, "manifest.csv"), "w", newline="", encoding="utf-8")
    wr = csv.writer(man); wr.writerow(["file", "label", "kind", "speaker", "score", "source"])
    import collections
    bysrc = collections.Counter()
    for i, (sc_, tag, w) in enumerate(keep):
        fn = f"hardneg_{i:05d}.wav"
        write_wav(os.path.join(OUT_ROOT, "hardneg", fn), w)
        wr.writerow([f"hardneg/{fn}", "negative", "hardneg",
                     f"hardneg_{tag}_{i % 40}", f"{sc_:.4f}", tag])
        bysrc[tag] += 1
    man.close()
    print(f"\nwrote {len(keep)} clips -> {os.path.join(os.getcwd(), OUT_ROOT)}")
    print("by source:", dict(bysrc))
    print("\nNEXT: add dataset_hardneg to build_dataset.py, rebuild, retrain.")
    print("Then re-mine - the model's failures move as it improves, so one")
    print("round of mining only fixes the failures it had at the time.")


if __name__ == "__main__":
    main()
