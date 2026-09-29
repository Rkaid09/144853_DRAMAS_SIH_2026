import os, csv, glob, json, sys, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prahari_config as cfg
from logmel import log_mel, load_wav_int16, quantise, dequantise

MDIR    = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else "artifacts"
MODEL   = os.path.join(MDIR, "prahari_int8.tflite")
WIN_HOP = 4
THRESHOLDS = (0.50, 0.70, 0.80, 0.90, 0.95)


EXCLUDE = ("aryaman",)


def find_speakers(root):
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*prahari_record*"))
                    + glob.glob(os.path.join(root, "*mohisha*"))):
        if os.path.isfile(os.path.join(d, "manifest.csv")):
            out.append(d)
        for sub in sorted(glob.glob(os.path.join(d, "*"))):
            if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "manifest.csv")):
                out.append(sub)
    return sorted(set(out))


def main():
    import tensorflow as tf
    if not os.path.exists(MODEL):
        sys.exit(f"no model at {MODEL} - run export_tflite.py first")

    print(f"model  : {MODEL}")
    it = tf.lite.Interpreter(model_path=MODEL)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    s_in, z_in = inp["quantization"]
    s_out, z_out = out["quantization"]

    import json as _json
    _aff_p = os.path.join(MDIR, "input_affine.json")
    _aff = _json.load(open(_aff_p)) if os.path.exists(_aff_p) else {}
    eff_scale = _aff.get("eff_scale", s_in)
    eff_zp    = _aff.get("eff_zp", float(z_in))
    print(f"input  : device uses scale {eff_scale:.7f} zp {eff_zp:.4f}"
          f"   (raw tflite {s_in:.7f} / {z_in})"
          f"   {'FOLDED' if _aff.get('folded') else 'not folded'}")

    import human_sets as hs
    clips = hs.clips()
    if not clips:
        sys.exit("no recording folders found next to the project root")
    print(f"model  : {MODEL}")
    print(f"speakers: {len({c['speaker'] for c in clips})}  (invalid early recordings excluded)\n")

    rows = []
    for m in clips:
        if True:
            p = m["path"]
            try:
                pcm = load_wav_int16(p)
            except Exception as e:
                print(f"  skip {m['file']}: {e}"); continue
            feat = dequantise(quantise(log_mel(pcm)))
            n = feat.shape[0]
            if n < cfg.CTX_FRAMES:
                pad = np.full((cfg.CTX_FRAMES - n, cfg.MEL_BINS),
                              feat.min(), dtype=feat.dtype)
                feat = np.concatenate([feat, pad], axis=0); n = feat.shape[0]
            best = 0.0
            best_at = 0
            for s in range(0, n - cfg.CTX_FRAMES + 1, WIN_HOP):
                w = feat[s:s + cfg.CTX_FRAMES][None, ..., None]
                q = np.clip(np.round(w / eff_scale + eff_zp), -128, 127).astype(np.int8)
                it.set_tensor(inp["index"], q)
                it.invoke()
                v = (float(np.ravel(it.get_tensor(out["index"]))[0]) - z_out) * s_out
                if v > best:
                    best = float(v)
                    best_at = s
            rows.append(dict(at=best_at * cfg.HOP_SAMPLES / float(cfg.SAMPLE_RATE),
                             speaker=m["speaker"], gender=m["gender"], split=m["split"], label=m["label"],
                             cond=m["cond"], rms=float(m.get("rms", 0) or 0),
                             score=best, file=m["file"]))

    kw = [r for r in rows if r["label"] == "keyword"]
    cf = [r for r in rows if r["label"] == "confusable"]
    sp = [r for r in rows if r["label"] == "speech"]
    rm = [r for r in rows if r["label"] == "room"]
    print(f"\n{len(rows)} clips: {len(kw)} keyword, {len(cf)} confusable, "
          f"{len(sp)} speech, {len(rm)} room tone")
    mp = [r for r in rows if r["label"] == "mispronounced"]
    if mp:
        print(f"   + {len(mp)} takes judged mispronounced by ear, left out of detection: "
              + ", ".join(f"{s} {sum(r['speaker'] == s for r in mp)}" for s in sorted({r['speaker'] for r in mp}))
              + f"  (the model fires on {100*sum(r['score'] >= 0.70 for r in mp)/len(mp):.0f}% of them at 0.70)")

    print("\n" + "=" * 78)
    print("HUMAN TEST SET - none of these speakers appear in any training data")
    print("=" * 78)
    print(f"{'thresh':>7}{'detect':>9}{'missed':>8}{'confFAR':>9}{'speechFAR':>11}{'roomFAR':>9}")
    print("-" * 78)
    for t in THRESHOLDS:
        d = sum(1 for r in kw if r["score"] >= t)
        print(f"{t:>7.2f}{100*d/max(len(kw),1):>8.1f}%{len(kw)-d:>8}"
              f"{100*sum(1 for r in cf if r['score']>=t)/max(len(cf),1):>8.1f}%"
              f"{100*sum(1 for r in sp if r['score']>=t)/max(len(sp),1):>10.1f}%"
              f"{100*sum(1 for r in rm if r['score']>=t)/max(len(rm),1):>8.1f}%")

    for t in (0.70, 0.90):
        print(f"\n--- detection by CONDITION at threshold {t:.2f} "
              f"(this is the distance question) ---")
        print(f"{'condition':<12}{'clips':>7}{'detected':>10}{'rate':>8}{'medRMS':>9}")
        for c in ("near", "far", "soft", "fast"):
            g = [r for r in kw if r["cond"] == c]
            if not g: continue
            d = sum(1 for r in g if r["score"] >= t)
            med = np.median([r["rms"] for r in g]) if g else 0
            print(f"{c:<12}{len(g):>7}{d:>10}{100*d/len(g):>7.1f}%{med:>9.0f}")

    print(f"\n--- detection by SPEAKER at threshold 0.70 (generalisation spread) ---")
    print(f"{'speaker':<24}{'clips':>7}{'detected':>10}{'rate':>8}")
    for s in sorted({r["speaker"] for r in kw}):
        g = [r for r in kw if r["speaker"] == s]
        d = sum(1 for r in g if r["score"] >= 0.70)
        print(f"{s:<24}{len(g):>7}{d:>10}{100*d/len(g):>7.1f}%")

    print(f"\n--- detection by GENDER and by HALF (select = helped choose the epoch; report = never did) ---")
    print(f"{'group':<24}{'clips':>7}{'@0.70':>8}{'@0.90':>8}")
    groups = [(f"gender {g}", [r for r in kw if r["gender"] == g]) for g in sorted({r["gender"] for r in kw})]
    groups += [(f"half {h}", [r for r in kw if r["split"] == h]) for h in sorted({r["split"] for r in kw})]
    for name, g in groups:
        print(f"{name:<24}{len(g):>7}" + "".join(f"{100*sum(r['score'] >= t for r in g)/len(g):>7.1f}%" for t in (0.70, 0.90)))

    print("\n--- the 8 worst-scoring keyword clips (listen to these) ---")
    for r in sorted(kw, key=lambda r: r["score"])[:8]:
        print(f"   {r['score']:.3f}  rms {r['rms']:>6.0f}  {r['speaker']:<22} "
              f"{r['cond']:<6} {r['file']}")

    for thr in (0.70, 0.90):
        fa = [r for r in (cf + sp + rm) if r["score"] >= thr]
        fa.sort(key=lambda r: -r["score"])
        print(f"\n--- FALSE ACCEPTS at {thr:.2f}: {len(fa)} of "
              f"{len(cf)+len(sp)+len(rm)} non-keyword clips  "
              f"(LISTEN TO THESE) ---")
        if not fa:
            print("   none")
            continue
        for r in fa[:20]:
            print(f"   {r['score']:.3f}  at {r.get('at',0):4.2f}-{r.get('at',0)+0.98:4.2f}s"
                  f"  rms {r['rms']:>6.0f}  "
                  f"{r['label']:<11} {r['speaker']:<22} {r['file']}")
        if len(fa) > 20:
            print(f"   ... and {len(fa)-20} more")

    try:
        a = tf.keras.metrics.AUC()
        y = np.array([1]*len(kw) + [0]*(len(cf)+len(sp)+len(rm)), dtype=np.float32)
        s_ = np.array([r["score"] for r in kw+cf+sp+rm], dtype=np.float32)
        a.update_state(y, s_)
        print(f"\nHUMAN AUC: {float(a.result().numpy()):.4f}"
              f"    (synthetic test set was 0.9916)")
    except Exception:
        pass

    json.dump(rows, open(os.path.join("artifacts", "human_eval.json"), "w"), indent=1)
    print("\nsaved: artifacts/human_eval.json")
    print("\nWhatever this says, it is the number. It was not tuned on and it")
    print("does not get re-run until something real changes.")


if __name__ == "__main__":
    main()
