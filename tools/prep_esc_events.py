#!/usr/bin/env python3
import csv, os, sys, wave
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_common_voice import to_16k

HERE = os.path.dirname(os.path.abspath(__file__))
ESC = os.path.join(HERE, "ESC-50-master")
OUT = os.path.join(HERE, "dataset_esc_events")
SR, WIN = 16000, 16000
PER_CLIP = 3


def read(path):
    with wave.open(path) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        return None, None
    x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def main():
    meta = list(csv.DictReader(open(os.path.join(ESC, "meta", "esc50.csv"), encoding="utf-8")))
    os.makedirs(OUT, exist_ok=True)
    rows, per_cat = [], {}
    for i, m in enumerate(meta):
        x, sr = read(os.path.join(ESC, "audio", m["filename"]))
        if x is None:
            continue
        x = to_16k(x, sr)
        hop = SR // 100
        n = len(x) // hop
        if n < 10:
            continue
        e = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
        if e.max() < 0.003:
            continue
        taken = []
        for k in np.argsort(e)[::-1]:
            if len(taken) >= PER_CLIP or e[k] < 0.25 * e.max():
                break
            c = int(k) * hop
            if any(abs(c - t) < WIN // 2 for t in taken):
                continue
            taken.append(c)
        base = os.path.splitext(m["filename"])[0]
        for j, c in enumerate(taken):
            s = int(np.clip(c - WIN // 2 + np.random.default_rng(i * 7 + j).integers(-3200, 3201),
                            0, max(len(x) - WIN, 0)))
            w = x[s:s + WIN]
            if len(w) < WIN:
                w = np.pad(w, (0, WIN - len(w)))
            fn = f"{m['category']}_{base}_{j}.wav"
            y = np.clip(w * 32767.0, -32768, 32767).astype("<i2")
            with wave.open(os.path.join(OUT, fn), "wb") as o:
                o.setnchannels(1); o.setsampwidth(2); o.setframerate(SR)
                o.writeframes(y.tobytes())
            rows.append([fn, "negative", "event", f"esc_fold{m['fold']}", m["category"], "all"])
            per_cat[m["category"]] = per_cat.get(m["category"], 0) + 1
        if i % 400 == 0:
            print(f"  {i}/{len(meta)} clips, {len(rows)} windows", flush=True)
    with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "label", "kind", "speaker", "source", "split"])
        wr.writerows(rows)
    hum = ["coughing", "sneezing", "laughing", "breathing", "clapping", "snoring",
           "footsteps", "keyboard_typing", "mouse_click", "door_wood_knock", "drinking_sipping"]
    print(f"\n{len(rows)} event-centred windows from {len(meta)} ESC-50 clips")
    print("human / indoor sounds: " + ", ".join(f"{h} {per_cat.get(h, 0)}" for h in hum))
    print(f"wrote {OUT}/manifest.csv")


if __name__ == "__main__":
    main()
