import glob, io, os, random, sys, wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dataset_babble")
DATA = os.environ.get("PRAHARI_DATA", r"E:\PRAHARI_DATA")
SR, N_CLIPS, N_UTTS = 16000, 3000, 2500


def decode(b):
    import soundfile as sf
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = np.interp(np.linspace(0, len(x) - 1, int(len(x) * SR / sr)), np.arange(len(x)), x)
    return x.astype(np.float32)


def main():
    import pyarrow.parquet as pq
    files = []
    for src in ("kathbath", "indicvoices"):
        fs = sorted(glob.glob(os.path.join(DATA, "extracted", src, "**", "train-*.parquet"), recursive=True))
        files += fs[:6]
    if not files:
        sys.exit(f"no Kathbath/IndicVoices train parquet under {DATA}\\extracted")
    rng = random.Random(2025)
    utts = []
    for f in files:
        pf = pq.ParquetFile(f)
        names = pf.schema_arrow.names
        acol = "audio_filepath" if "audio_filepath" in names else names[0]
        for batch in pf.iter_batches(batch_size=64, columns=[acol] + (["text"] if "text" in names else [])):
            d = batch.to_pydict()
            for i, a in enumerate(d[acol]):
                t = (d.get("text") or [""] * len(d[acol]))[i] or ""
                if "प्रहर" in t or not a or not a.get("bytes") or rng.random() > 0.15:
                    continue
                try:
                    x = decode(a["bytes"])
                except Exception:
                    continue
                if len(x) >= SR:
                    utts.append(x)
            if len(utts) >= N_UTTS // len(files) * (files.index(f) + 1):
                break
        print(f"  {os.path.basename(os.path.dirname(os.path.dirname(f)))}: {len(utts)} utterances so far")
    os.makedirs(OUT, exist_ok=True)
    for k in range(N_CLIPS):
        mix = np.zeros(SR, np.float32)
        for _ in range(rng.randint(3, 6)):
            u = rng.choice(utts)
            s = rng.randint(0, len(u) - SR)
            seg = u[s:s + SR]
            r = float(np.sqrt(np.mean(seg ** 2))) or 1.0
            mix += seg / r * 10 ** (rng.uniform(-12, 0) / 20)
        mix *= 0.1 / (float(np.sqrt(np.mean(mix ** 2))) or 1.0)
        with wave.open(os.path.join(OUT, f"babble_{k:04d}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes(np.clip(mix * 32767, -32768, 32767).astype("<i2").tobytes())
    print(f"wrote {N_CLIPS} crowd-chatter clips from {len(utts)} utterances -> {OUT}")


if __name__ == "__main__":
    main()
