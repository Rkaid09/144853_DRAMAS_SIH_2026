import csv, glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

GENDER = {"avani": "female", "divyanshi": "female", "mohisha": "female",
          "rajani": "male", "aryaman": "male"}
EXCLUDE_V1 = {"aryaman"}
DEFAULT_V1_SPLIT = "select"
DEFAULT_V2_SPLIT = "report"
LABEL = {"keyword": "keyword", "confusable": "confusable", "speech": "speech",
         "talk": "speech", "room": "room"}


def _override():
    p = os.path.join(HERE, "human_split.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def _v1_dirs():
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, "*prahari_record*")) + glob.glob(os.path.join(ROOT, "*mohisha*"))):
        for m in glob.glob(os.path.join(d, "manifest.csv")) + glob.glob(os.path.join(d, "*", "manifest.csv")):
            out.append(os.path.dirname(m))
    return sorted(set(out))


def _v2_dirs():
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, "prahari_*"))) + sorted(glob.glob(os.path.join(ROOT, "recordings", "prahari_*"))):
        if "prahari_record" in os.path.basename(d):
            continue
        for m in glob.glob(os.path.join(d, "manifest.csv")) + glob.glob(os.path.join(d, "*", "manifest.csv")):
            if os.path.exists(os.path.join(os.path.dirname(m), "info.json")):
                out.append(os.path.dirname(m))
    return sorted(set(out))


REVIEW = os.path.join(HERE, "human_review.csv")


def _review():
    if not os.path.exists(REVIEW):
        return {}
    return {r["file"]: r["verdict"] for r in csv.DictReader(open(REVIEW, encoding="utf-8"))}


def clips(raw=False):
    rows = _clips()
    if not raw:
        rv = _review()
        for r in rows:
            if r["label"] == "keyword" and rv.get(r["file"]) == "n":
                r["label"] = "mispronounced"
    return rows


def _clips():
    ov = _override()
    rows = []
    for d in _v1_dirs():
        for m in csv.DictReader(open(os.path.join(d, "manifest.csv"), encoding="utf-8")):
            spk = (m.get("speaker_id") or "").split("_")[0].lower()
            if not spk or spk in EXCLUDE_V1:
                continue
            p = os.path.join(d, m["file"])
            if os.path.exists(p):
                rows.append(dict(path=p, file=m["file"], speaker=spk, gender=GENDER.get(spk, "?"),
                                 split=ov.get(spk, DEFAULT_V1_SPLIT), label=m["label"],
                                 cond=m.get("condition", ""), rms=float(m.get("rms") or 0)))
    for d in _v2_dirs():
        info = json.load(open(os.path.join(d, "info.json"), encoding="utf-8"))
        spk = str(info.get("name", os.path.basename(d).replace("prahari_", ""))).lower()
        g = info.get("voice", "?")
        for m in csv.DictReader(open(os.path.join(d, "manifest.csv"), encoding="utf-8")):
            p = os.path.join(d, m["file"])
            lab = LABEL.get(m["label"])
            if lab and os.path.exists(p):
                rows.append(dict(path=p, file=m["file"], speaker=spk, gender=g,
                                 split=ov.get(spk, DEFAULT_V2_SPLIT), label=lab,
                                 cond=m.get("condition", ""), rms=float(m.get("rms") or 0)))
    return rows


def voiced_seconds(pcm):
    import numpy as np
    x = pcm.astype(np.float64)
    n = len(x) // 160
    if n < 3:
        return 0.0
    e = np.sqrt((x[: n * 160].reshape(n, 160) ** 2).mean(axis=1))
    a = np.flatnonzero(e > max((np.percentile(e, 20) + 1) * 4, e.max() * 0.06))
    return float((a[-1] - a[0] + 1) / 100) if len(a) else 0.0


LENGTH_BUCKETS = ((0.0, 0.5, "<0.5s"), (0.5, 0.7, "0.5-0.7s"), (0.7, 9.0, ">=0.7s"))


def windows(path, hop=4):
    import numpy as np
    import prahari_config as cfg
    from logmel import log_mel, load_wav_int16, quantise, dequantise
    f = dequantise(quantise(log_mel(load_wav_int16(path))))
    if f.shape[0] < cfg.CTX_FRAMES:
        f = np.concatenate([f, np.full((cfg.CTX_FRAMES - f.shape[0], cfg.MEL_BINS), f.min(), f.dtype)])
    idx = range(0, f.shape[0] - cfg.CTX_FRAMES + 1, hop)
    return np.stack([f[s:s + cfg.CTX_FRAMES] for s in idx]).astype(np.float16)


def load(split=None):
    import numpy as np
    rows = [r for r in clips() if split is None or r["split"] == split]
    Xs, idx, keep = [], [], []
    from logmel import load_wav_int16
    for r in rows:
        try:
            w = windows(r["path"])
            r["voiced"] = voiced_seconds(load_wav_int16(r["path"]))
        except Exception:
            continue
        Xs.append(w); idx.append(np.full(len(w), len(keep), np.int32)); keep.append(r)
    if not keep:
        return None
    return np.concatenate(Xs), np.concatenate(idx), keep


BURST_ARM, BURST_STEPS = 0.50, 13


def board_scores(p, idx, n, arm=BURST_ARM):
    import numpy as np
    out = np.zeros((n, 3), np.float32)
    order = np.argsort(idx, kind="stable")
    p, idx = np.asarray(p, np.float32)[order], np.asarray(idx)[order]
    starts = np.searchsorted(idx, np.arange(n))
    ends = np.searchsorted(idx, np.arange(n), side="right")
    for c in range(n):
        s = p[starts[c]:ends[c]]
        for ph in range(3):
            best, t, until = 0.0, ph, -1
            while t < len(s):
                best = max(best, s[t])
                if (t - ph) % 3 == 0 and s[t] >= arm:
                    until = t + BURST_STEPS
                if t < until:
                    t += 1
                else:
                    t += 3 - ((t - ph) % 3) if (t - ph) % 3 else 3
            out[c, ph] = best
    return out


def clip_scores(p, idx, n):
    import numpy as np
    out = np.zeros(n, np.float32)
    np.maximum.at(out, idx, p.astype(np.float32))
    return out


def summary(scores, rows, thr):
    import numpy as np
    kw = [i for i, r in enumerate(rows) if r["label"] == "keyword"]
    spk = sorted({rows[i]["speaker"] for i in kw})
    per = {s: float(np.mean([scores[i] >= thr for i in kw if rows[i]["speaker"] == s])) for s in spk}
    by_g = {}
    for g in sorted({rows[i]["gender"] for i in kw}):
        m = [i for i in kw if rows[i]["gender"] == g]
        by_g[g] = float(np.mean([scores[i] >= thr for i in m]))
    bad = [i for i, r in enumerate(rows) if r["label"] == "mispronounced"]
    conf = [i for i, r in enumerate(rows) if r["label"] == "confusable"]
    other = [i for i, r in enumerate(rows) if r["label"] in ("speech", "room")]
    by_len = {}
    for lo, hi, name in LENGTH_BUCKETS:
        m = [i for i in kw if lo <= rows[i].get("voiced", 0.5) < hi]
        if m:
            by_len[name] = float(np.mean([scores[i] >= thr for i in m]))
    return {"speaker_mean": float(np.mean(list(per.values()))) if per else 0.0, "per_speaker": per,
            "by_length": by_len,
            "mispronounced_fire": float(np.mean([scores[i] >= thr for i in bad])) if bad else None,
            "n_mispronounced": len(bad),
            "by_gender": by_g,
            "confusable_fa": float(np.mean([scores[i] >= thr for i in conf])) if conf else 0.0,
            "speech_fa": float(np.mean([scores[i] >= thr for i in other])) if other else 0.0,
            "n_keyword": len(kw)}
