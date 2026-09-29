import argparse, csv, glob, io, json, os, random, subprocess, sys, time, wave, zlib
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import prahari_config as cfg
from logmel import log_mel, quantise, dequantise

SR, WIN = 16000, 16000
SHARD = 8192
AUG_P = 0.5
TOOLS = os.path.join(HERE, "..", "tools")


def to16k(x, sr):
    if sr == SR:
        return x.astype(np.float32)
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sr), SR)
        return resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    except ImportError:
        n = int(round(len(x) * SR / float(sr)))
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def decode_bytes(b):
    try:
        import soundfile as sf
        x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
        return to16k(x.mean(axis=1), sr)
    except Exception:
        pass
    p = subprocess.run(["ffmpeg", "-v", "quiet", "-i", "-", "-f", "s16le", "-ac", "1",
                        "-ar", str(SR), "-"], input=b, capture_output=True)
    if p.returncode == 0 and p.stdout:
        return np.frombuffer(p.stdout, "<i2").astype(np.float32) / 32768.0
    return None


def decode_file(path):
    try:
        if path.lower().endswith(".wav"):
            with wave.open(path) as w:
                sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
                raw = w.readframes(w.getnframes())
            if sw == 2:
                x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
                if ch > 1:
                    x = x.reshape(-1, ch).mean(axis=1)
                return to16k(x, sr)
        with open(path, "rb") as f:
            return decode_bytes(f.read())
    except Exception:
        return None


def features(x_float, seed, aug):
    from augment import augment
    pcm = np.clip(x_float * 32767.0, -32768, 32767).astype(np.int16)
    if aug:
        pcm = augment(pcm, seed)
    lm = dequantise(quantise(log_mel(pcm)))
    if lm.shape != (cfg.CTX_FRAMES, cfg.MEL_BINS):
        return None
    return lm.astype(np.float16)


def speech_windows(x, rng, max_n, min_rms=0.003):
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    off = int(rng.integers(0, max(1, min(WIN, len(x) - WIN + 1))))
    starts = list(range(off, len(x) - WIN + 1, WIN)) or [0]
    rng.shuffle(starts)
    out = []
    for s in starts:
        w = x[s:s + WIN]
        if float(np.sqrt(np.mean(w * w))) >= min_rms:
            out.append(w)
        if len(out) >= max_n:
            break
    return out


def random_windows(x, rng, max_n):
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    n = min(max_n, max(1, (len(x) - WIN) // (WIN // 2) + 1))
    return [x[s:s + WIN] for s in rng.integers(0, len(x) - WIN + 1, size=n)]


def event_windows(x, rng, max_n):
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    hop = SR // 100
    n = len(x) // hop
    e = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    if e.max() < 0.003:
        return []
    taken, out = [], []
    for k in np.argsort(e)[::-1]:
        if len(out) >= max_n or e[k] < 0.25 * e.max():
            break
        c = int(k) * hop
        if any(abs(c - t) < WIN // 2 for t in taken):
            continue
        taken.append(c)
        s = int(np.clip(c - WIN // 2 + rng.integers(-3200, 3201), 0, len(x) - WIN))
        out.append(x[s:s + WIN])
    return out


STRETCH_P = 0.5
WORD_S = (0.35, 0.90)
STRETCH_SOURCES = ("keyword", "keyword2", "voices_kw", "confusable", "negative2", "voices_neg")


def stretch_word(w, rng):
    from augment import time_stretch
    hop = SR // 100
    n = len(w) // hop
    e = np.sqrt((w[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    act = np.flatnonzero(e > max(e.max() * 0.06, 1e-4))
    if len(act) < 5:
        return w
    cur = (act[-1] - act[0] + 1) / 100.0
    seg = w[max(0, act[0] * hop - 480): min(len(w), (act[-1] + 1) * hop + 480)]
    f = float(np.clip(rng.uniform(*WORD_S) / cur, 0.75, 2.8))
    y = time_stretch(seg, f)
    if len(y) >= WIN - 160:
        f = (WIN - 800) / len(seg)
        y = time_stretch(seg, f)
    out = np.zeros(WIN, np.float32)
    s = int(rng.integers(0, WIN - len(y) + 1))
    out[s:s + len(y)] = y[: WIN - s]
    return out


def work(task):
    mode, item, max_n, copies, seed = task
    rng = np.random.default_rng(seed)
    stretch = mode.endswith("_s")
    mode = mode[:-2] if stretch else mode
    if isinstance(item, tuple) and item[0] == "bytes":
        x = decode_bytes(item[1])
    else:
        x = decode_file(item)
    if x is None or len(x) < SR // 4:
        return np.zeros((0, cfg.CTX_FRAMES, cfg.MEL_BINS), np.float16)
    if stretch and mode == "speech" and rng.random() < STRETCH_P:
        from augment import time_stretch
        x = time_stretch(x, float(rng.uniform(1.0, 1.8)))
    if isinstance(max_n, float):
        want = len(x) / SR * max_n
        max_n = int(want) + (1 if rng.random() < want - int(want) else 0)
        if max_n == 0:
            return np.zeros((0, cfg.CTX_FRAMES, cfg.MEL_BINS), np.float16)
    if mode == "speech":
        ws = speech_windows(x, rng, max_n)
    elif mode == "event":
        ws = event_windows(x, rng, max_n)
    elif mode == "random":
        ws = random_windows(x, rng, max_n)
    elif mode == "random75":
        ws = random_windows(x[: int(len(x) * 0.75)], rng, max_n)
    else:
        ws = [x[:WIN] if len(x) >= WIN else np.pad(x, (0, WIN - len(x)))]
    out = []
    for i, w in enumerate(ws):
        if copies:
            f = features(w, seed * 31 + i, False)
            if f is not None:
                out.append(f)
            for c in range(copies):
                src = stretch_word(w, rng) if (stretch and mode == "whole" and rng.random() < STRETCH_P) else w
                f = features(src, seed * 131 + i * 17 + c + 1, True)
                if f is not None:
                    out.append(f)
        else:
            f = features(w, seed * 31 + i, rng.random() < AUG_P)
            if f is not None:
                out.append(f)
    if not out:
        return np.zeros((0, cfg.CTX_FRAMES, cfg.MEL_BINS), np.float16)
    return np.stack(out)


def h(s):
    return zlib.crc32(s.encode("utf-8", "replace"))


_SPLIT = None


def core_split():
    global _SPLIT
    if _SPLIT is None:
        import build_dataset as bd
        rows = bd.read_rows()
        amap = {}
        for dom in sorted(set(bd.DOMAIN_OF.values())):
            amap.update(bd.assign_groups(rows, dom, bd.SEED))
        _SPLIT = (rows, amap)
    return _SPLIT


def src_core_rows():
    rows, amap = core_split()
    return [r for r in rows if amap.get(r["group"]) == "train"]


TTS2 = os.path.join(TOOLS, "dataset_tts2")


def tts2_rows():
    man = os.path.join(TTS2, "manifest.csv")
    if not os.path.exists(man):
        return []
    _, amap = core_split()
    out = []
    for r in csv.DictReader(open(man, encoding="utf-8")):
        p = os.path.join(TTS2, r["file"].replace("/", os.sep))
        if os.path.exists(p):
            r = dict(r, path=p, split=amap.get(f"tts_{r['voice_id']}", "train"))
            out.append(r)
    return out


VOICES = os.path.join(TOOLS, "dataset_voices")
PER_VOICE_WINDOWS = 240


def voices_rows():
    man = os.path.join(VOICES, "manifest.csv")
    if not os.path.exists(man):
        return []
    out = []
    for r in csv.DictReader(open(man, encoding="utf-8")):
        p = os.path.join(VOICES, r["file"].replace("/", os.sep))
        if os.path.exists(p):
            out.append(dict(r, path=p))
    return out


def voices_tasks(label_names):
    rows = [r for r in voices_rows() if r["split"] == "train" and r["label"] in label_names]
    n = {}
    for r in rows:
        k = (r["voice"], r["label"])
        n[k] = n.get(k, 0) + 1
    T = []
    for r in rows:
        if r["label"] == "sentence":
            T.append(("speech", r["path"], 4, 3, h(r["path"])))
        else:
            copies = int(np.clip(round(PER_VOICE_WINDOWS / n[(r["voice"], r["label"])]) - 1, 2, 30))
            T.append(("whole", r["path"], 1, copies, h(r["path"])))
    return T


def build_voices_test(root):
    path = os.path.join(root, "features", "test_voices.npz")
    rows = [r for r in voices_rows() if r["split"] == "test" and r["label"] != "sentence"]
    if os.path.exists(path) or not rows:
        return
    X, y, g, b = [], [], [], []
    for r in rows:
        x = decode_file(r["path"])
        if x is None:
            continue
        w = x[:WIN] if len(x) >= WIN else np.pad(x, (0, WIN - len(x)))
        f = features(w, h(r["path"]), False)
        if f is not None:
            X.append(f); y.append(1 if r["label"] == "positive" else 0)
            g.append(r.get("gender", "?")); b.append(r["backend"])
    if X:
        np.savez(path, X=np.stack(X), y=np.array(y, np.int8), gender=np.array(g), backend=np.array(b))
        print(f"== test_voices: {len(X)} clips ({sum(y)} keyword) from held-out voices -> {path}")


def build_extra(root):
    rows = tts2_rows()
    for split in ("val", "test"):
        path = os.path.join(root, "features", f"{split}_extra.npz")
        if os.path.exists(path) or not rows:
            continue
        X, y, g = [], [], []
        for r in rows:
            if r["split"] != split or r["label"] == "sentence":
                continue
            x = decode_file(r["path"])
            if x is None:
                continue
            w = x[:WIN] if len(x) >= WIN else np.pad(x, (0, WIN - len(x)))
            f = features(w, h(r["path"]), False)
            if f is not None:
                X.append(f); y.append(1 if r["label"] == "positive" else 0); g.append(r.get("gender", "?"))
        if X:
            np.savez(path, X=np.stack(X), y=np.array(y, np.int8), gender=np.array(g))
            print(f"== {split}_extra: {len(X)} clips ({sum(y)} keyword, "
                  f"{sum(1 for a, b in zip(y, g) if a and b == 'female')} of them female) -> {path}")


def tasks_for(name, root, cap):
    ext = os.path.join(root, "extracted")
    T, E = [], []
    if name == "keyword":
        for r in src_core_rows():
            if r["kind"] == "keyword":
                T.append(("whole", r["path"], 1, 20, h(r["path"])))
        return T, E, 1
    if name == "confusable":
        for r in src_core_rows():
            if r["kind"] == "confusable":
                T.append(("whole", r["path"], 1, 7, h(r["path"])))
        return T, E, 0
    if name == "core":
        for r in src_core_rows():
            if r["kind"] in ("speech", "noise", "silence", "hardneg"):
                T.append(("whole", r["path"], 1, 1, h(r["path"])))
        return T, E, 0
    if name in ("keyword2", "negative2"):
        for r in tts2_rows():
            if r["split"] != "train":
                continue
            if name == "keyword2" and r["label"] == "positive":
                T.append(("whole", r["path"], 1, 20, h(r["path"])))
            elif name == "negative2" and r["label"] == "negative":
                T.append(("whole", r["path"], 1, 7, h(r["path"])))
            elif name == "negative2" and r["label"] == "sentence":
                T.append(("speech", r["path"], 4, 3, h(r["path"])))
        return T, E, (1 if name == "keyword2" else 0)
    if name == "voices_kw":
        return voices_tasks({"positive"}), E, 1
    if name == "voices_neg":
        return voices_tasks({"negative", "sentence"}), E, 0
    if name == "esc":
        man = os.path.join(TOOLS, "dataset_esc_events", "manifest.csv")
        if os.path.exists(man):
            for r in csv.DictReader(open(man, encoding="utf-8")):
                p = os.path.join(TOOLS, "dataset_esc_events", r["file"])
                T.append(("whole", p, 1, 2, h(p)))
        return T, E, 0
    if name == "board":
        cap_dir = os.path.join(HERE, "..", "server", "captures")
        for f in sorted(glob.glob(os.path.join(cap_dir, "*-room*.wav"))):
            if os.path.getsize(f) > 20 * SR * 2:
                T.append(("random75", f, 60, 0, h(f)))
                E.append(f)
        return T, E, 0
    if name == "cv":
        src = os.path.join(TOOLS, "cv-corpus-27.0-2026-09-11", "hi")
        val = list(csv.DictReader(open(os.path.join(src, "validated.tsv"), encoding="utf-8"), delimiter="\t"))
        oth_p = os.path.join(src, "other.tsv")
        oth = list(csv.DictReader(open(oth_p, encoding="utf-8"), delimiter="\t")) if os.path.exists(oth_p) else []
        spk = sorted({r["client_id"] for r in val})
        random.Random(1337).shuffle(spk)
        held = set(spk[: min(80, len(spk) // 3)])
        per = {}
        for r in val + oth:
            p = os.path.join(src, "clips", r["path"])
            if r["client_id"] in held:
                E.append(p); continue
            per[r["client_id"]] = per.get(r["client_id"], 0) + 1
            if per[r["client_id"]] <= 400:
                T.append(("speech", p, 4, 0, h(p)))
        return T, E, 0
    if name == "slr104":
        base = os.path.join(ext, "slr104")
        files = []
        for f in glob.glob(os.path.join(base, "**", "*.wav"), recursive=True):
            dirs = os.path.relpath(os.path.dirname(f), base).replace("\\", "/").lower().split("/")
            (E if any("test" in d for d in dirs) else files).append(f)
        d = density(files, cap)
        return [("speech", f, d, 0, h(f)) for f in files], E, 0
    if name.startswith("musan_"):
        kind = name.split("_")[1]
        files = glob.glob(os.path.join(ext, "musan", "**", kind, "**", "*.wav"), recursive=True)
        mode = "speech" if kind == "speech" else "random"
        tr = []
        for f in files:
            (E if h(f) % 10 == 0 else tr).append(f)
        d = density(tr, cap)
        return [(mode, f, d, 0, h(f)) for f in tr], E, 0
    if name in ("fsd50k", "fsd50k_dev"):
        sub = "FSD50K.eval_audio" if name == "fsd50k" else "FSD50K.dev_audio"
        files = glob.glob(os.path.join(ext, "fsd50k", sub, "**", "*.wav"), recursive=True)
        tr = []
        for f in files:
            (E if h(f) % 10 == 0 else tr).append(f)
        d = density(tr, cap)
        return [("event", f, d, 0, h(f)) for f in tr], E, 0
    if name in PARQUET:
        allp = glob.glob(os.path.join(ext, name, "**", "*.parquet"), recursive=True)
        files = sorted({f for f in allp if "hindi" in os.path.relpath(f, ext).lower()})
        return [("parquet_file", f) for f in files], [], 0
    raise SystemExit(f"unknown source {name}")


def wav_seconds(f):
    try:
        with wave.open(f) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return os.path.getsize(f) / (2.0 * SR)


def density(files, cap):
    tot = sum(wav_seconds(f) for f in files)
    d = min(1.0, cap / max(tot, 1.0))
    print(f"  {len(files):,} files, {tot/3600:.1f} h -> {d:.3f} windows per second", flush=True)
    return float(d)


PARQUET = ("kathbath", "indicvoices", "shrutilipi")
GENDER_BALANCED = ("kathbath", "indicvoices")
EVAL_MINUTES = 90


def _audio_col(tbl):
    return next((k for k, v in tbl.items() if v and isinstance(v[0], dict) and "bytes" in v[0]), None)


def parquet_seconds(f):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(f)
    for c in ("duration", "audio_duration", "duration_s", "length"):
        if c in pf.schema_arrow.names:
            try:
                return float(sum(x for x in pq.read_table(f, columns=[c]).column(c).to_pylist() if x))
            except Exception:
                break
    return os.path.getsize(f) / 17000.0


def parquet_split(files):
    ev = [f for f in files if "valid" in os.path.basename(f).lower() or "/valid" in f.replace("\\", "/").lower()]
    if ev:
        return [f for f in files if f not in ev], ev
    ranked = sorted(files, key=h)
    k = max(1, len(files) // 20) if len(files) > 1 else 0
    return [f for f in files if f not in ranked[:k]], ranked[:k]


def density_parquet(files, cap):
    tot = 0.0
    for f in files:
        try:
            tot += parquet_seconds(f)
        except Exception:
            tot += os.path.getsize(f) / 17000.0
    d = min(1.0, cap / max(tot, 1.0))
    print(f"  {len(files)} train files, {tot/3600:.1f} h -> {d:.3f} windows per second", flush=True)
    return float(d)


def gender_keep(files):
    import pyarrow.parquet as pq
    hours = {}
    for f in files:
        try:
            names = pq.ParquetFile(f).schema_arrow.names
            if "gender" not in names:
                return None, None
            cols = ["gender"] + (["duration"] if "duration" in names else [])
            t = pq.read_table(f, columns=cols).to_pydict()
            for i, g in enumerate(t["gender"]):
                g = str(g).lower()
                hours[g] = hours.get(g, 0.0) + float((t.get("duration") or [1.0] * len(t["gender"]))[i] or 0) / 3600
        except Exception:
            continue
    main = {g: v for g, v in hours.items() if g in ("female", "male")}
    if len(main) < 2:
        return None, None
    lo = min(main.values())
    keep = {g: (lo / v if g in main else 0.0) for g, v in hours.items()}
    kept_s = sum(hours[g] * keep[g] for g in hours) * 3600
    print("  gender balance: " + ", ".join(f"{g} {v:.1f} h -> keep {keep[g]*100:.0f}%" for g, v in hours.items()),
          flush=True)
    return keep, kept_s


def parquet_tasks(files, d, keep=None):
    import pyarrow.parquet as pq
    n = 0
    for f in files:
        try:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=64):
                tbl = batch.to_pydict()
                col = _audio_col(tbl)
                if col is None:
                    raise SystemExit(f"no audio column in {f}: {list(tbl)}")
                gs = tbl.get("gender") or [None] * len(tbl[col])
                for a, g in zip(tbl[col], gs):
                    if a and a.get("bytes"):
                        n += 1
                        if keep is not None and (h(f"keep:{f}:{n}") % 10000) / 10000.0 >= keep.get(str(g).lower(), 1.0):
                            continue
                        yield ("speech", ("bytes", a["bytes"]), d, 0, h(f"{f}:{n}"))
        except SystemExit:
            raise
        except Exception as e:
            print(f"  !! {os.path.basename(f)} unreadable ({str(e)[:100]}) - skipped", flush=True)


def write_eval_wavs(files, out_dir, minutes):
    import pyarrow.parquet as pq
    os.makedirs(out_dir, exist_ok=True)
    paths, got = [], 0.0
    for f in files:
        try:
            batches = pq.ParquetFile(f).iter_batches(batch_size=64)
            for batch in batches:
                tbl = batch.to_pydict()
                col = _audio_col(tbl)
                for a in (tbl[col] if col else []):
                    x = decode_bytes(a["bytes"]) if a and a.get("bytes") else None
                    if x is None:
                        continue
                    p = os.path.join(out_dir, f"{len(paths):05d}.wav")
                    with wave.open(p, "wb") as w:
                        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
                        w.writeframes(np.clip(x * 32767, -32768, 32767).astype("<i2").tobytes())
                    paths.append(p); got += len(x) / SR
                    if got >= minutes * 60:
                        return paths
        except Exception as e:
            print(f"  !! held-out {os.path.basename(f)} unreadable ({str(e)[:100]}) - skipped", flush=True)
    return paths


READY = {"slr104": ["Hindi-English_train.tar.gz.extracted", "Hindi-English_test.tar.gz.extracted"],
         "musan_speech": ["musan.tar.gz.extracted"], "musan_music": ["musan.tar.gz.extracted"],
         "musan_noise": ["musan.tar.gz.extracted"], "fsd50k": ["FSD50K.extracted"],
         "fsd50k_dev": ["FSD50K_dev.extracted"], "kathbath": ["kathbath.complete"],
         "indicvoices": ["indicvoices.complete"], "shrutilipi": ["shrutilipi.complete"]}


def run_source(name, root, workers, cap):
    out = os.path.join(root, "features", name)
    meta_p = os.path.join(out, "meta.json")
    if os.path.exists(meta_p):
        print(f"== {name}: already done ({json.load(open(meta_p))['windows']:,} windows) - skipping")
        return
    missing = [m for m in READY.get(name, []) if not os.path.exists(os.path.join(root, "raw", m))]
    if missing:
        print(f"== {name}: download/unpack not finished ({', '.join(missing)} missing) - skipping for now")
        return
    if os.path.isdir(out) and os.listdir(out):
        os.rename(out, f"{out}_partial_{int(time.time())}")
    tasks, evals, label = tasks_for(name, root, cap)
    if name in STRETCH_SOURCES:
        tasks = [(t[0] + "_s",) + tuple(t[1:]) for t in tasks]
    if not tasks:
        print(f"== {name}: nothing on disk - skipping"); return
    os.makedirs(out, exist_ok=True)
    if name in PARQUET:
        print(f"== {name}: {len(tasks)} parquet files", flush=True)
        tr, ev = parquet_split([t[1] for t in tasks])
        evals = write_eval_wavs(ev, os.path.join(root, "features", "eval_audio", name), EVAL_MINUTES)
        keep, kept_s = gender_keep(tr) if name in GENDER_BALANCED else (None, None)
        if keep is not None:
            d = min(1.0, cap / max(kept_s, 1.0))
            print(f"  {len(tr)} train files, {kept_s/3600:.1f} h after gender balance -> {d:.3f} windows per second",
                  flush=True)
        else:
            d = density_parquet(tr, cap)
        stream = parquet_tasks(random.Random(1).sample(tr, len(tr)), d, keep)
        n_items = "?"
    else:
        random.Random(1).shuffle(tasks)
        stream, n_items = iter(tasks), len(tasks)
    if evals:
        with open(os.path.join(root, "features", f"eval_{name}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(evals))
    print(f"== {name}: {n_items} items, {len(evals)} held out, label {label}, cap {cap:,} windows", flush=True)
    buf, shard, total, t0, done = [], 0, 0, time.time(), 0

    def flush(final=False):
        nonlocal buf, shard
        while buf and (final or sum(len(b) for b in buf) >= SHARD):
            X = np.concatenate(buf)
            take = X[:SHARD] if len(X) > SHARD else X
            rest = X[len(take):]
            take = take[np.random.default_rng(shard).permutation(len(take))]
            np.save(os.path.join(out, f"shard_{shard:04d}.npy"), take)
            shard += 1
            buf = [rest] if len(rest) else []
            if not final and len(rest) < SHARD:
                break

    with Pool(workers) as pool:
        for arr in pool.imap_unordered(work, stream, chunksize=4):
            done += 1
            if len(arr):
                buf.append(arr); total += len(arr)
            if sum(len(b) for b in buf) >= SHARD:
                flush()
            if done % 500 == 0:
                print(f"  {done:,} items  {total:,} windows  {time.time()-t0:5.0f} s", flush=True)
            if total >= cap:
                pool.terminate(); break
    flush(final=True)
    json.dump({"source": name, "label": label, "windows": total, "shards": shard,
               "items": done, "held_out": len(evals), "seconds": round(time.time() - t0)},
              open(meta_p, "w"), indent=1)
    print(f"  {name}: {total:,} windows in {shard} shards ({time.time()-t0:.0f} s)")


ORDER = ["keyword", "keyword2", "voices_kw", "confusable", "negative2", "voices_neg", "core", "esc", "board", "cv", "slr104", "kathbath",
         "indicvoices", "shrutilipi", "musan_speech", "musan_music", "musan_noise",
         "fsd50k", "fsd50k_dev"]
CAPS = {"keyword": 10**9, "keyword2": 10**9, "voices_kw": 10**9, "voices_neg": 10**9, "negative2": 10**9, "confusable": 10**9, "core": 10**9, "esc": 10**9, "board": 10**9,
        "cv": 80000, "slr104": 300000, "kathbath": 400000, "indicvoices": 500000,
        "shrutilipi": 400000, "musan_speech": 200000, "musan_music": 150000,
        "musan_noise": 40000, "fsd50k": 40000, "fsd50k_dev": 120000}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"E:\PRAHARI_DATA")
    ap.add_argument("--only", choices=ORDER)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    os.chdir(HERE)
    failed = []
    for name in ([a.only] if a.only else ORDER):
        try:
            run_source(name, a.root, a.workers, CAPS[name])
        except ImportError as e:
            print(f"\n!! {name}: missing Python package ({e}). Fix:  pip install pyarrow soundfile  - then rerun")
            failed.append(name)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"\n!! {name} FAILED ({e}) - the other sources continue; rerun: python prep_corpus.py --only {name}")
            failed.append(name)
    if not a.only:
        try:
            build_extra(a.root)
            build_voices_test(a.root)
        except Exception as e:
            print(f"\n!! val/test extra sets FAILED ({e})"); failed.append("extra")
    if failed:
        print(f"\nprep_corpus: {len(failed)} source(s) failed: {', '.join(failed)}")
        sys.exit(1)
    print("\nprep_corpus finished. Next: python train_stream.py")
