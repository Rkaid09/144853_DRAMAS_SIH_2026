import argparse, csv, os, random, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_voices as gv

TTS2 = os.path.join(HERE, "dataset_tts2")
F5_LANGS = {"hi", "mr", "ne", "gu", "pa", "bn", "or", "ta", "te", "kn", "ml"}
PER_VOICE = {"positive": 20, "negative": 20, "sentence": 2}
MIN_VOICED = {"positive": 0.32, "negative": 0.22}
EL_HELD_OUT = {"Lily", "Liam", "George", "Laura", "Jessica", "Will"}
SLOWER = [1.0, 0.8, 0.65]


def references():
    refs = {}
    rows = []
    if os.path.exists(gv.MAN):
        for r in csv.DictReader(open(gv.MAN, encoding="utf-8")):
            if r["backend"] != "indicf5":
                rows.append((r["backend"], r["voice"], r["gender"], r["lang"],
                             os.path.join(gv.OUT, r["file"]), r["text"], r["label"]))
    m2 = os.path.join(TTS2, "manifest.csv")
    if os.path.exists(m2):
        for r in csv.DictReader(open(m2, encoding="utf-8")):
            rows.append(("eleven", r["voice_name"].split(" - ")[0], r["gender"], "hi",
                         os.path.join(TTS2, r["file"].replace("/", os.sep)), r["text"], r["label"]))
    for backend, voice, gender, lang, path, text, label in rows:
        if label != "sentence" or lang not in F5_LANGS or not os.path.exists(path):
            continue
        secs = os.path.getsize(path) / (2 * gv.SR)
        if not 1.5 <= secs <= 8:
            continue
        k = (backend, voice, lang)
        if k not in refs or abs(secs - 4) < abs(refs[k]["secs"] - 4):
            refs[k] = {"backend": backend, "voice": voice, "gender": gender, "lang": lang,
                       "path": path, "text": text, "secs": secs}
    return list(refs.values())


def slowed(path, rate):
    if rate == 1.0:
        return path
    out = os.path.join(gv.OUT, "_refs", os.path.basename(path)[:-4] + f"_slow{rate:.2f}.wav")
    if not os.path.exists(out):
        import librosa
        x, _ = librosa.load(path, sr=gv.SR)
        y = librosa.effects.time_stretch(x, rate=rate)
        gv.save(out, (y / max(1e-6, float(np.abs(y).max())) * 0.8 * 32767).astype("int16"))
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--max-voices", type=int, default=0, help="0 = every reference voice")
    ap.add_argument("--shifts", default="0,-2.5,2.5", help="semitone shifts applied to each reference")
    ap.add_argument("--per-voice", type=int, default=0,
                    help="keyword AND sound-alike clips per voice (default 20). Pass 1 can use 12 to cover every "
                         "voice sooner; a later run without it tops each voice up to 20 - nothing is redone.")
    a = ap.parse_args()

    refs = references()
    rng = random.Random(int(time.time()))
    rng.shuffle(refs)
    if a.max_voices:
        refs = refs[: a.max_voices]
    if a.test:
        refs = refs[:2]
    shifts = [float(x) for x in a.shifts.split(",")]
    refdir = os.path.join(gv.OUT, "_refs")
    os.makedirs(refdir, exist_ok=True)
    jobs = []
    for ref in refs:
        for sh in shifts:
            r = dict(ref, shift=sh)
            if sh:
                fn = f"{ref['backend']}_{ref['voice']}_{ref['lang']}_{sh:+.1f}.wav"
                r["path"] = os.path.join(refdir, fn.replace(":", "-").replace(" ", "_"))
                r["src"] = ref["path"]
            jobs.append(r)
    refs = jobs
    if not refs:
        sys.exit("no reference sentences found - run gen_voices.py first (and/or keep dataset_tts2)")
    print(f"{len(refs)} reference voices ({sum(r['gender'] == 'female' for r in refs)} female) from "
          + ", ".join(f"{b} {sum(r['backend'] == b for r in refs)}" for b in sorted({r['backend'] for r in refs})))

    import torch
    from transformers import AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev} {torch.cuda.get_device_name(0) if dev == 'cuda' else '(no GPU - slow)'}")
    model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
    try:
        model = model.to(dev)
    except Exception:
        pass

    man = gv.Manifest()
    sents = gv.sentences()
    t0, made, bad = time.time(), 0, 0
    start_slow = {}
    for ri, ref in enumerate(refs):
        voice = f"f5:{ref['backend']}:{ref['voice']}" + (f":{ref['shift']:+.1f}" if ref["shift"] else "")
        if ref["shift"] and not os.path.exists(ref["path"]):
            import librosa
            x, _ = librosa.load(ref["src"], sr=gv.SR)
            y = librosa.effects.pitch_shift(x, sr=gv.SR, n_steps=ref["shift"])
            gv.save(ref["path"], (y / max(1e-6, abs(y).max()) * 0.8 * 32767).astype("int16"))
        if ref["backend"] == "eleven":
            split = "test" if ref["voice"] in EL_HELD_OUT else "train"
        else:
            split = gv.split_of(ref["backend"], ref["voice"])
        todo = []
        for lab, n in PER_VOICE.items():
            if a.per_voice and lab != "sentence":
                n = a.per_voice
            if a.test:
                n = {"positive": 1, "negative": 1, "sentence": 0}[lab]
            for _ in range(max(0, n - man.done("indicf5", voice, ref["lang"], lab))):
                dev_text = (rng.choice(gv.KEYWORD) if lab == "positive" else rng.choice(gv.CONFUSABLES)
                            if lab == "negative" else rng.choice(sents))
                todo.append((lab, gv.to_script(dev_text, ref["lang"])))
        for lab, text in todo:
            torch.manual_seed(rng.randrange(1 << 30))
            clip, ok, why, used = None, False, "failed", 1.0
            tries = SLOWER[SLOWER.index(start_slow.get(ref["path"], 1.0)):] if lab in MIN_VOICED else [1.0]
            for slow in tries:
                path = slowed(ref["path"], slow)
                try:
                    y = model(text, ref_audio_path=path, ref_text=ref["text"])
                except Exception as e:
                    why = f"failed ({str(e)[:100]})"; break
                y = np.asarray(y)
                if y.dtype == np.int16:
                    y = y.astype(np.float32) / 32768.0
                y = y.astype(np.float32).ravel()
                clip, ok, why = gv.finish(y, 24000, lab != "sentence", rng)
                used = slow
                if ok and lab in MIN_VOICED:
                    _, voiced = gv.trim(gv.resample(y / max(1e-6, float(np.abs(y).max())), 24000))
                    if voiced < MIN_VOICED[lab]:
                        ok, why = False, f"too quick ({voiced:.2f}s voiced)"
                if ok:
                    if lab in MIN_VOICED and slow < start_slow.get(ref["path"], 1.0):
                        start_slow[ref["path"]] = slow
                    break
            if not ok:
                bad += 1
                if a.test:
                    print(f"   rejected '{text}': {why}")
                continue
            f = man.next_name("indicf5", lab)
            gv.save(os.path.join(gv.OUT, f), clip)
            man.add({"file": f, "label": lab, "text": text, "backend": "indicf5", "voice": voice,
                     "gender": ref["gender"], "lang": ref["lang"],
                     "variant": f"ref={os.path.basename(ref['path'])},slow={used}",
                     "split": split, "seconds": round(len(clip) / gv.SR, 2)})
            made += 1
            if a.test:
                print(f"   {f}  {voice:<30} {ref['lang']} '{text}'")
        rate = made / max(time.time() - t0, 1) * 60
        print(f"   voice {ri + 1}/{len(refs)} {voice:<34} {made} made, {bad} rejected, {rate:.0f} clips/min",
              flush=True)
    gv.summary(man)
    if a.test:
        print(f"\nLISTEN to {gv.OUT}\\indicf5\\positive: each must say प्रहरी clearly, in the reference voice.")


if __name__ == "__main__":
    main()
