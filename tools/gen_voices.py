import argparse, asyncio, base64, csv, glob, hashlib, io, json, os, random, sys, time, wave
import urllib.request, urllib.error
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dataset_voices")
MAN = os.path.join(OUT, "manifest.csv")
FIELDS = ["file", "label", "text", "backend", "voice", "gender", "lang", "variant", "split", "seconds"]
DATA = os.environ.get("PRAHARI_DATA", r"E:\PRAHARI_DATA")
SR, WIN = 16000, 16000
TEST_SHARE = 15

KEYWORD = ["प्रहरी", "प्रहरी", "प्रहरी", "प्रहरी।", "प्रहरी!", "प्रहरी?"]
CONFUSABLES = ["प्रभारी", "प्रहर", "पहाड़ी", "प्रचार", "पुजारी", "भिखारी", "बिहारी", "प्रकृति",
               "हरी", "प्रतिहारी", "प्रयाग", "परिहार", "प्रहार", "पहरा", "प्रहलाद", "सहारी",
               "प्यारी", "प्रहरियों", "हमारी", "तुम्हारी", "पहरी", "प्रहरा", "बहरी", "कचहरी"]
FALLBACK_SENTENCES = [
    "आज मौसम बहुत अच्छा है", "मुझे एक गिलास पानी चाहिए", "कल हम बाज़ार जाएंगे",
    "दरवाज़ा बंद कर दीजिए", "यह किताब बहुत दिलचस्प है", "क्या आपने खाना खा लिया",
    "मैं थोड़ी देर में वापस आती हूँ", "कैंटीन में आज बहुत भीड़ है", "चलो क्लास के लिए देर हो रही है",
    "उसने मुझे फ़ोन पर सब बता दिया", "बस स्टॉप यहाँ से ज़्यादा दूर नहीं है", "मेरी चाय ठंडी हो गई",
]

LANGS = {
    "hi": (None, False), "mr": (None, False), "ne": (None, False),
    "gu": ("GUJARATI", False), "pa": ("GURMUKHI", False), "bn": ("BENGALI", False),
    "or": ("ORIYA", True), "ta": ("TAMIL", True), "te": ("TELUGU", True),
    "kn": ("KANNADA", True), "ml": ("MALAYALAM", True), "en": ("ITRANS", False),
}
SKIP_LANGS = {"bn", "or"}
EN_KEYWORD = ["Prahree", "Prehri"]
MULTI_OK = {"en-AU-WilliamMultilingualNeural", "en-US-AndrewMultilingualNeural",
            "en-US-BrianMultilingualNeural", "en-US-EmmaMultilingualNeural",
            "fr-FR-VivienneMultilingualNeural", "fr-FR-RemyMultilingualNeural",
            "de-DE-SeraphinaMultilingualNeural", "de-DE-FlorianMultilingualNeural",
            "it-IT-GiuseppeMultilingualNeural", "pt-BR-ThalitaMultilingualNeural"}
MULTI_BAD = {"en-US-AvaMultilingualNeural", "ko-KR-HyunsuMultilingualNeural"}
EN_CANDIDATES = ["प्रहरी", "Prahari", "Pruhuree", "Prahree", "Pruh-huh-ree", "Prehri"]
AUDITION = os.path.join(OUT, "_audition")

_CONS = set(chr(c) for c in list(range(0x0915, 0x093A)) + list(range(0x0958, 0x0960)))
VIRAMA = "\u094d"


def to_script(text, lang):
    if lang == "en" and EN_KEYWORD and text.strip("।!? ") == "प्रहरी":
        return random.choice(EN_KEYWORD) + text.replace("प्रहरी", "").replace("।", ".")
    scheme, virama = LANGS[lang]
    if scheme is None:
        return text
    from indic_transliteration import sanscript
    if virama:
        text = " ".join(w + VIRAMA if w and w[-1] in _CONS else w for w in text.split(" "))
    out = sanscript.transliterate(text, sanscript.DEVANAGARI, getattr(sanscript, scheme))
    if lang == "ta":
        out = "".join(ch for ch in out if ch not in "¹²³⁴₁₂₃₄")
    if lang == "en":
        for a, b in (("A", "aa"), ("I", "ee"), ("U", "oo"), ("Ch", "chh"), ("~N", "n"), ("M", "n"),
                     (".D", "d"), (".N", "n"), ("RRi", "ri"), ("|", "."), ("!", "!")):
            out = out.replace(a, b)
        words = []
        for w in out.lower().split(" "):
            core = w.rstrip(".!?")
            if len(core) > 2 and core.endswith("a") and not core.endswith("aa"):
                w = core[:-1] + w[len(core):]
            words.append(w)
        out = " ".join(words)
        out = out[:1].upper() + out[1:]
    return out


def decode_any(raw):
    if raw[:4] == b"RIFF":
        with wave.open(io.BytesIO(raw)) as w:
            sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
            x = np.frombuffer(w.readframes(n), dtype={2: "<i2", 4: "<i4"}[sw]).astype(np.float32)
            x /= 32768.0 if sw == 2 else 2147483648.0
            return x.reshape(-1, ch).mean(axis=1), sr
    try:
        import soundfile as sf
        x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        return x.mean(axis=1), sr
    except Exception:
        from pydub import AudioSegment
        a = AudioSegment.from_file(io.BytesIO(raw)).set_channels(1)
        x = np.array(a.get_array_of_samples(), dtype=np.float32) / (1 << (8 * a.sample_width - 1))
        return x, a.frame_rate


def resample(x, sr):
    if sr == SR:
        return x
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, SR)
        return resample_poly(x, SR // g, sr // g).astype(np.float32)
    except ImportError:
        return np.interp(np.linspace(0, len(x) - 1, int(len(x) * SR / sr)), np.arange(len(x)), x).astype(np.float32)


def trim(x):
    hop = SR // 100
    n = len(x) // hop
    if n < 3:
        return x, 0.0
    e = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    act = np.flatnonzero(e > max(e.max() * 0.06, 1e-4))
    if not len(act):
        return x, 0.0
    y = x[max(0, act[0] * hop - 480): min(len(x), (act[-1] + 1) * hop + 480)]
    return y, (act[-1] - act[0] + 1) / 100


MIN_KEYWORD_S = 0.30


def finish(x, sr, word, rng, min_voiced=0.2):
    x = resample(x, sr)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak < 1e-3:
        return None, False, "silent"
    x = x / peak * rng.uniform(0.3, 0.9)
    y, voiced = trim(x)
    if word:
        if not min_voiced <= voiced <= 0.95 or len(y) > WIN:
            return None, False, f"word length {voiced:.2f}s"
        out = np.zeros(WIN, np.float32)
        s = rng.randint(0, WIN - len(y))
        out[s:s + len(y)] = y
        y = out
    elif not 0.8 <= voiced <= 12:
        return None, False, f"sentence length {voiced:.2f}s"
    return np.clip(y * 32767, -32768, 32767).astype(np.int16), True, "ok"


def save(path, y):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(y.astype("<i2").tobytes())


def split_of(backend, voice):
    h = int(hashlib.md5(f"{backend}:{voice}".encode()).hexdigest(), 16) % 100
    return "test" if h < TEST_SHARE else "train"


class Manifest:
    def __init__(self):
        os.makedirs(OUT, exist_ok=True)
        self.rows = []
        if os.path.exists(MAN) and os.path.getsize(MAN):
            self.rows = [r for r in csv.DictReader(open(MAN, encoding="utf-8"))
                         if os.path.exists(os.path.join(OUT, r["file"]))]
        self.fh = open(MAN, "a", newline="", encoding="utf-8")
        self.wr = csv.DictWriter(self.fh, fieldnames=FIELDS)
        if not os.path.getsize(MAN):
            self.wr.writeheader()

    def done(self, backend, voice, lang, label, slow=False):
        return sum(1 for r in self.rows if r["backend"] == backend and r["voice"] == voice
                   and r["lang"] == lang and r["label"] == label
                   and r.get("variant", "").startswith("slow") == slow)

    def next_name(self, backend, label):
        n = sum(1 for r in self.rows if r["backend"] == backend and r["label"] == label)
        while True:
            n += 1
            f = f"{backend}/{label}/{backend}_{label}_{n:05d}.wav"
            if not os.path.exists(os.path.join(OUT, f)):
                return f

    def add(self, row):
        self.wr.writerow(row); self.fh.flush(); self.rows.append(row)


def sentences():
    out = []
    try:
        import pyarrow.parquet as pq
        for f in sorted(glob.glob(os.path.join(DATA, "extracted", "kathbath", "**", "train-*.parquet"),
                                  recursive=True))[10:14]:
            for s in pq.read_table(f, columns=["text"]).column("text").to_pylist():
                if s and 15 <= len(s) <= 50 and "प्रहर" not in s and "पहर" not in s:
                    out.append(s.strip())
    except Exception as e:
        print(f"  (Kathbath sentences unavailable: {e}; using the built-in list)")
    random.Random(17).shuffle(out)
    return out or FALLBACK_SENTENCES


def plan_for(lang, backend="sarvam"):
    if backend == "edge":
        return (30, 30, 6)
    if backend == "sarvam":
        return (24, 24, 5)
    return (12, 12, 3) if lang in ("hi", "mr", "ne") else (5, 5, 2)


def jobs_for(voices, man, backend, sents, rng, test, plan=None, slow=False):
    per = []
    for v in voices:
        k, n, s = plan or plan_for(v["lang"], backend)
        if test:
            k, n, s = 1, 0, 0
        want = {"positive": k, "negative": n, "sentence": s}
        todo = []
        for lab, cnt in want.items():
            for _ in range(max(0, cnt - man.done(backend, v["voice"], v["lang"], lab, slow))):
                dev = (rng.choice(KEYWORD) if lab == "positive" else rng.choice(CONFUSABLES)
                       if lab == "negative" else rng.choice(sents))
                todo.append((lab, dev))
        rng.shuffle(todo)
        per.append([(v, lab, dev) for lab, dev in todo])
    out = []
    while any(per):
        for q in per:
            if q:
                out.append(q.pop())
    return out[:5] if test else out


SARVAM_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_V3 = {
    "female": ["ritu", "priya", "neha", "pooja", "simran", "kavya", "ishita", "shreya", "roopa", "tanya",
               "shruti", "suhani", "kavitha", "rupali"],
    "male": ["shubh", "aditya", "rahul", "rohan", "amit", "dev", "ratan", "varun", "manan", "sumit", "kabir",
             "aayan", "ashutosh", "advait", "anand", "tarun", "sunny", "mani", "gokul", "vijay", "mohit",
             "rehan", "soham"],
}
SARVAM_V2 = {}
SARVAM_USE_LANGS = {"hi"}
SARVAM_BAD = {"ritu", "neha", "kavya", "ishita", "suhani", "amit", "ashutosh", "anand", "soham"}
SARVAM_LANGS = {"hi": "hi-IN", "mr": "mr-IN", "gu": "gu-IN", "pa": "pa-IN", "bn": "bn-IN", "or": "od-IN",
                "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN", "ml": "ml-IN", "en": "en-IN"}


def sarvam_keys():
    p = os.path.join(HERE, "sarvam_key.txt")
    keys = [l.strip() for l in open(p, encoding="utf-8")] if os.path.exists(p) else []
    keys = [k for k in keys if k and not k.startswith("#")]
    if not keys:
        sys.exit("put your Sarvam API key(s) in tools/sarvam_key.txt, one per line, and SAVE the file")
    return keys


def sarvam_call(key, payload):
    req = urllib.request.Request(SARVAM_URL, data=json.dumps(payload).encode(),
                                 headers={"api-subscription-key": key, "Content-Type": "application/json"})
    err = "rate-limited 5 times"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return base64.b64decode(json.load(r)["audios"][0]), None
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:200]
            if e.code == 429:
                time.sleep(3 * (attempt + 1)); continue
            return None, f"HTTP {e.code}: {body}"
        except Exception as e:
            time.sleep(2); err = str(e)[:120]
    return None, err


def run_sarvam(a, man, sents, rng):
    voices = []
    for model, table in (("bulbul:v3", SARVAM_V3), ("bulbul:v2", SARVAM_V2)):
        for g, names in table.items():
            for n in names:
                for lang in SARVAM_LANGS:
                    if lang in SKIP_LANGS or lang not in SARVAM_USE_LANGS or n in SARVAM_BAD:
                        continue
                    voices.append({"voice": f"{model.split(':')[1]}:{n}", "speaker": n, "model": model,
                                   "gender": g, "lang": lang})
    if a.list:
        print(f"sarvam: {len({v['voice'] for v in voices})} voices x {len(SARVAM_LANGS)} languages")
        return
    keys = sarvam_keys()
    ki = 0
    key = keys[ki]
    if a.audition:
        return sarvam_audition(voices, keys, rng)
    jobs = jobs_for(voices, man, "sarvam", sents, rng, a.test)
    est = sum(len(to_script(d, v["lang"])) for v, _, d in jobs)
    print(f"sarvam: {len(jobs)} clips to make, ~{est:,} characters (~Rs {est * 3 / 1000:.0f}); "
          f"budget --max-chars {a.max_chars:,}")
    spent, bad, dead, t0 = 0, 0, set(), time.time()
    for i, (v, lab, dev) in enumerate(jobs):
        if v["voice"] in dead or (v["voice"], v["lang"]) in dead:
            continue
        text = to_script(dev, v["lang"])
        if spent + len(text) > a.max_chars:
            print(f"budget reached ({spent:,} characters)"); break
        pay = {"text": text, "target_language_code": SARVAM_LANGS[v["lang"]], "speaker": v["speaker"],
               "model": v["model"], "speech_sample_rate": 16000,
               "pace": round(rng.uniform(0.85, 1.3) if lab != "sentence" else rng.uniform(0.9, 1.2), 2)}
        if v["model"] == "bulbul:v3":
            pay["temperature"] = round(rng.uniform(0.2, 0.5), 2)
        else:
            pay["pitch"] = round(rng.uniform(-0.5, 0.5), 2); pay["loudness"] = round(rng.uniform(0.8, 1.4), 2)
        raw, err = sarvam_call(key, pay)
        if raw is None and err and "od-IN" in json.dumps(pay) and "language" in err.lower():
            SARVAM_LANGS["or"] = "or-IN"; pay["target_language_code"] = "or-IN"
            raw, err = sarvam_call(key, pay)
        if raw is None:
            print(f"   {v['voice']} {v['lang']}: {err}")
            low = (err or "").lower()
            if "credit" in low or "quota" in low or "insufficient" in low or "HTTP 402" in (err or "") \
                    or "HTTP 401" in (err or "") or "HTTP 403" in (err or ""):
                ki += 1
                if ki >= len(keys):
                    print("   every key is out of credits - stopping"); break
                key = keys[ki]
                print(f"   key {ki} used up/refused - switching to key {ki + 1} of {len(keys)}")
                continue
            if "speaker" in low:
                dead.add(v["voice"])
            elif "language" in low:
                dead.add((v["voice"], v["lang"]))
            continue
        spent += len(text)
        x, sr = decode_any(raw)
        y, ok, why = finish(x, sr, lab != "sentence", rng,
                                MIN_KEYWORD_S if lab == "positive" else 0.2)
        if not ok:
            bad += 1
            if a.test:
                print(f"   rejected: {why}")
            continue
        f = man.next_name("sarvam", lab)
        save(os.path.join(OUT, f), y)
        man.add({"file": f, "label": lab, "text": text, "backend": "sarvam", "voice": v["voice"],
                 "gender": v["gender"], "lang": v["lang"],
                 "variant": f"pace={pay['pace']}" + (f",temp={pay['temperature']}" if "temperature" in pay
                                                      else f",pitch={pay['pitch']}"),
                 "split": split_of("sarvam", v["voice"]), "seconds": round(len(y) / SR, 2)})
        if a.test or (i + 1) % 50 == 0:
            print(f"   {i + 1}/{len(jobs)} {f}  {v['voice']:<16} {v['lang']} '{text}'  "
                  f"({spent:,} chars, {bad} rejected, {(time.time() - t0) / 60:.1f} min)", flush=True)
        time.sleep(0.15)
    print(f"sarvam done: {spent:,} characters spent, {bad} clips rejected (bad length)")


def sarvam_audition(voices, keys, rng):
    aud = os.path.join(OUT, "_audition_sarvam_hi")
    os.makedirs(aud, exist_ok=True)
    picks = [v for v in voices if v["lang"] == "hi"]
    lines, key = [], keys[0]
    for n, v in enumerate(picks, 1):
        text = to_script("प्रहरी", v["lang"])
        pay = {"text": text, "target_language_code": SARVAM_LANGS[v["lang"]], "speaker": v["speaker"],
               "model": v["model"], "speech_sample_rate": 16000, "pace": 1.0}
        if v["model"] == "bulbul:v3":
            pay["temperature"] = 0.35
        raw, err = sarvam_call(key, pay)
        if raw is None and len(keys) > 1 and err and ("401" in err or "credit" in err.lower() or "402" in err):
            key = keys[1]; raw, err = sarvam_call(key, pay)
        ok = raw is not None
        if ok:
            x, sr = decode_any(raw)
            y, ok, why = finish(x, sr, True, rng)
            err = why
        if ok:
            save(os.path.join(aud, f"{n:02d}_{v['lang']}_{v['speaker']}.wav"), y)
        line = f"{n:2d}  {v['lang']:<3} {v['gender']:<6} {v['voice']:<16} '{text}'" + ("" if ok else f"  FAILED: {err}")
        print(line); lines.append(line)
    open(os.path.join(aud, "list.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\nListen to {aud}\\NN_*.wav and tell me the NUMBERS that do NOT sound like pra-ha-ree.")


EDGE_LOCALES = {"hi-IN": "hi", "mr-IN": "mr", "ne-NP": "ne", "gu-IN": "gu", "pa-IN": "pa", "bn-IN": "bn",
                "or-IN": "or", "ta-IN": "ta", "te-IN": "te", "kn-IN": "kn", "ml-IN": "ml", "en-IN": "en"}


async def edge_voices(all_multi=False):
    import edge_tts
    vs = await edge_tts.list_voices()
    out = []
    for v in vs:
        g = v.get("Gender", "").lower()
        if v["Locale"] in EDGE_LOCALES and EDGE_LOCALES[v["Locale"]] not in SKIP_LANGS:
            out.append({"voice": v["ShortName"], "gender": g, "lang": EDGE_LOCALES[v["Locale"]]})
        elif "Multilingual" in v["ShortName"] and (all_multi or v["ShortName"] in MULTI_OK):
            out.append({"voice": v["ShortName"], "gender": g, "lang": "hi"})
    return out


async def edge_one(text, voice, rate, pitch):
    import edge_tts
    buf = bytearray()
    async for ch in edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).stream():
        if ch["type"] == "audio":
            buf += ch["data"]
    return bytes(buf)


async def run_edge_async(a, man, sents, rng):
    voices = await edge_voices(all_multi=a.audition_multi)
    if a.audition_multi:
        await edge_audition([v for v in voices if "Multilingual" in v["voice"]
                             and v["voice"] not in MULTI_OK | MULTI_BAD],
                            rng, multi_only=True)
        return
    if a.list:
        print(f"edge: {len(voices)} voices ({sum(v['gender'] == 'female' for v in voices)} female)")
        for v in voices:
            print(f"   {v['voice']:<40} {v['gender']:<7} {v['lang']}")
        return
    if a.audition:
        await edge_audition(voices, rng)
        return
    jobs = jobs_for(voices, man, "edge", sents, rng, a.test, plan=(12, 12, 2) if a.slow else None, slow=a.slow)
    print(f"edge{' (slow takes)' if a.slow else ''}: {len(voices)} voices, {len(jobs)} clips to make")
    bad, t0 = 0, time.time()
    for i, (v, lab, dev) in enumerate(jobs):
        text = to_script(dev, v["lang"])
        rate = f"{rng.randint(-55, -15):+d}%" if a.slow else f"{rng.randint(-15, 25):+d}%"
        pitch = f"{rng.randint(-25, 35):+d}Hz"
        raw = None
        for attempt in range(3):
            try:
                raw = await edge_one(text, v["voice"], rate, pitch); break
            except Exception as e:
                err = str(e)[:100]; await asyncio.sleep(2 * (attempt + 1))
        if not raw:
            print(f"   {v['voice']}: failed ({err})"); continue
        x, sr = decode_any(raw)
        y, ok, why = finish(x, sr, lab != "sentence", rng,
                                MIN_KEYWORD_S if lab == "positive" else 0.2)
        if not ok:
            bad += 1
            if a.test:
                print(f"   rejected: {why}")
            continue
        f = man.next_name("edge", lab)
        save(os.path.join(OUT, f), y)
        man.add({"file": f, "label": lab, "text": text, "backend": "edge", "voice": v["voice"],
                 "gender": v["gender"], "lang": v["lang"],
                 "variant": ("slow," if a.slow else "") + f"rate={rate},pitch={pitch}",
                 "split": split_of("edge", v["voice"]), "seconds": round(len(y) / SR, 2)})
        if a.test or (i + 1) % 50 == 0:
            print(f"   {i + 1}/{len(jobs)} {f}  {v['voice']:<34} '{text}'  "
                  f"({bad} rejected, {(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"edge done: {bad} clips rejected (bad length)")


async def edge_audition(voices, rng, multi_only=False):
    global AUDITION
    if multi_only:
        AUDITION = os.path.join(OUT, "_audition_multi")
    os.makedirs(AUDITION, exist_ok=True)
    picks = [dict(v, lang="multi") for v in voices] if multi_only else []
    for lang in [] if multi_only else sorted({v["lang"] for v in voices if "Multilingual" not in v["voice"]}):
        for g in ("female", "male"):
            vs = [v for v in voices if v["lang"] == lang and v["gender"] == g and "Multilingual" not in v["voice"]]
            if vs:
                picks.append(vs[0])
    multi = [] if multi_only else [v for v in voices if "Multilingual" in v["voice"]]
    for g in ("female", "male"):
        vs = [v for v in multi if v["gender"] == g]
        if vs:
            picks.append(dict(vs[0], lang="multi"))
    lines, n = [], 0
    for v in picks:
        texts = EN_CANDIDATES if v["lang"] == "en" else ["प्रहरी"] if v["lang"] == "multi" else \
            [to_script("प्रहरी", v["lang"])]
        for t in texts:
            n += 1
            try:
                raw = await edge_one(t, v["voice"], "+0%", "+0Hz")
                x, sr = decode_any(raw)
                y, ok, why = finish(x, sr, True, rng)
            except Exception as e:
                ok, why = False, str(e)[:60]
            name = f"{n:02d}_{v['lang']}_{v['voice'].split('-')[-1].replace('Neural', '')}.wav"
            if multi_only:
                name = f"{n:02d}_{v['voice'].replace('Neural', '')}.wav"
            if ok:
                save(os.path.join(AUDITION, name), y)
            line = f"{n:2d}  {v['lang']:<5} {v['gender']:<6} {v['voice']:<38} '{t}'" + ("" if ok else f"  FAILED: {why}")
            print(line); lines.append(line)
    open(os.path.join(AUDITION, "list.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\nListen to {AUDITION}\\NN_*.wav and tell me the NUMBERS that do NOT sound like pra-ha-ree.")


def summary(man):
    rows = man.rows
    if not rows:
        return
    print("\n== dataset_voices so far")
    for b in sorted({r["backend"] for r in rows}):
        rb = [r for r in rows if r["backend"] == b]
        vs = {r["voice"] for r in rb}
        fv = {r["voice"] for r in rb if r["gender"] == "female"}
        pos = [r for r in rb if r["label"] == "positive"]
        print(f"   {b:<8} {len(vs):4d} voices ({len(fv)} female) | keyword {len(pos):5d} "
              f"({sum(r['gender'] == 'female' for r in pos)} female) | sound-alike "
              f"{sum(r['label'] == 'negative' for r in rb):5d} | sentence {sum(r['label'] == 'sentence' for r in rb):4d}"
              f" | languages {','.join(sorted({r['lang'] for r in rb}))}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", nargs="?", choices=["sarvam", "edge"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--audition", action="store_true", help="edge: one keyword per language/spelling, to listen")
    ap.add_argument("--audition-multi", action="store_true", help="edge: every not-yet-approved multilingual voice")
    ap.add_argument("--slow", action="store_true", help="edge: 12+12+2 SLOW takes per voice (rate -55%%..-15%%)")
    ap.add_argument("--max-chars", type=int, default=60000, help="Sarvam: stop after this many characters")
    a = ap.parse_args()
    man = Manifest()
    rng = random.Random(int(time.time()))
    sents = [] if a.list else sentences()
    backends = [a.backend] if a.backend else (["sarvam", "edge"] if a.list else [])
    if not backends:
        ap.print_help(); summary(man); return
    for b in backends:
        if b == "sarvam":
            run_sarvam(a, man, sents, rng)
        else:
            asyncio.run(run_edge_async(a, man, sents, rng))
    summary(man)
    if a.test:
        print(f"\nLISTEN to the newest files in {OUT}\\{a.backend}\\positive: each must say प्रहरी clearly.")


if __name__ == "__main__":
    main()
