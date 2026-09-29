import os, csv, glob, random, hashlib, wave, sys
import numpy as np

SR      = 16000
WIN     = SR
OUT     = "dataset_neg"
SEED    = 4242

SC_ROOT   = os.path.join("speech_commands")
ESC_ROOT  = os.path.join("ESC-50-master")

N_SPEECH  = 2000
N_NOISE   = 1000
N_SILENCE = 300

rng = random.Random(SEED)
nprng = np.random.RandomState(SEED)


def write_wav(path, x_int16):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(x_int16.tobytes())


def read_wav_any(path):
    with wave.open(path, "rb") as w:
        n, sw, ch, fr = w.getnframes(), w.getsampwidth(), w.getnchannels(), w.getframerate()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ValueError(f"{path}: {sw*8}-bit not supported")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if fr != SR:
        idx = np.linspace(0, len(x) - 1, int(len(x) * SR / fr))
        x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    return x


def to_int16(x):
    peak = np.max(np.abs(x)) if len(x) else 0.0
    if peak > 0.99:
        x = x * (0.99 / peak)
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)


def take_window(x, rng):
    if len(x) >= WIN:
        s = rng.randint(0, len(x) - WIN)
        return x[s:s + WIN]
    out = np.zeros(WIN, dtype=np.float32)
    s = rng.randint(0, WIN - len(x))
    out[s:s + len(x)] = x
    return out


rows = []


def do_speech():
    if not os.path.isdir(SC_ROOT):
        print(f"  SKIP speech: {SC_ROOT}/ not found")
        return
    words = [d for d in sorted(os.listdir(SC_ROOT))
             if os.path.isdir(os.path.join(SC_ROOT, d)) and not d.startswith("_")]
    files = []
    for wd in words:
        files += glob.glob(os.path.join(SC_ROOT, wd, "*.wav"))
    print(f"  speech: {len(files)} candidate files across {len(words)} words")
    rng.shuffle(files)
    os.makedirs(os.path.join(OUT, "speech"), exist_ok=True)
    n = 0
    for f in files:
        if n >= N_SPEECH:
            break
        try:
            x = read_wav_any(f)
        except Exception:
            continue
        spk = os.path.basename(f).split("_")[0]
        name = f"speech_{n:05d}.wav"
        write_wav(os.path.join(OUT, "speech", name), to_int16(take_window(x, rng)))
        rows.append({"file": f"speech/{name}", "label": "negative",
                     "kind": "speech", "speaker": f"sc_{spk}",
                     "source": os.path.relpath(f)})
        n += 1
    print(f"  speech: wrote {n}")


def do_noise():
    esc = os.path.join(ESC_ROOT, "audio")
    pool = []
    if os.path.isdir(esc):
        pool = [("esc", f) for f in glob.glob(os.path.join(esc, "*.wav"))]
        print(f"  noise: {len(pool)} ESC-50 clips")
    bg = os.path.join(SC_ROOT, "_background_noise_")
    if os.path.isdir(bg):
        extra = glob.glob(os.path.join(bg, "*.wav"))
        pool += [("bg", f) for f in extra]
        print(f"  noise: + {len(extra)} background files from Speech Commands")
    if not pool:
        print("  SKIP noise: no ESC-50 and no _background_noise_")
        return
    os.makedirs(os.path.join(OUT, "noise"), exist_ok=True)
    n = 0
    while n < N_NOISE:
        kind, f = pool[n % len(pool)]
        try:
            x = read_wav_any(f)
        except Exception:
            n += 1
            continue
        base = os.path.basename(f)
        grp = f"esc_fold{base.split('-')[0]}" if kind == "esc" else f"bgnoise_{base[:-4]}"
        name = f"noise_{n:05d}.wav"
        write_wav(os.path.join(OUT, "noise", name), to_int16(take_window(x, rng)))
        rows.append({"file": f"noise/{name}", "label": "negative",
                     "kind": "noise", "speaker": grp,
                     "source": os.path.relpath(f)})
        n += 1
    print(f"  noise: wrote {n}")


def do_silence():
    os.makedirs(os.path.join(OUT, "silence"), exist_ok=True)
    for i in range(N_SILENCE):
        if i < N_SILENCE // 5:
            x = np.zeros(WIN, dtype=np.float32)
            grp = "silence_digital"
        else:
            amp = 10 ** (nprng.uniform(-60, -35) / 20)
            x = nprng.normal(0, amp, WIN).astype(np.float32)
            grp = "silence_hiss"
        name = f"silence_{i:05d}.wav"
        write_wav(os.path.join(OUT, "silence", name), to_int16(x))
        rows.append({"file": f"silence/{name}", "label": "negative",
                     "kind": "silence", "speaker": grp, "source": "generated"})
    print(f"  silence: wrote {N_SILENCE}")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("building negative corpus...")
    do_speech(); do_noise(); do_silence()

    if not rows:
        sys.exit("nothing produced - check the extracted folder names")

    with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["file", "label", "kind", "speaker", "source"])
        wr.writeheader(); wr.writerows(rows)

    import collections
    print(f"\ntotal negatives : {len(rows)}")
    print("by kind         :", dict(collections.Counter(r["kind"] for r in rows)))
    print("distinct groups :", len({r['speaker'] for r in rows}))
    print(f"manifest        : {OUT}/manifest.csv")


if __name__ == "__main__":
    main()
