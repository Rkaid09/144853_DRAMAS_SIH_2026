import argparse, asyncio, csv, glob, io, os, random, sys, time

import aiohttp
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dataset_tts2")
MAN = os.path.join(OUT, "manifest.csv")
FIELDS = ["file", "label", "text", "voice_id", "voice_name", "gender", "accent",
          "voice_source", "key_index", "seed", "stability", "style", "offset_ms"]
API = "https://api.elevenlabs.io/v1"
MODEL_ID = "eleven_multilingual_v2"
SR, WINDOW_MS = 16000, 1000
DATA = os.environ.get("PRAHARI_DATA", r"E:\PRAHARI_DATA")

POSITIVE_TEXTS = [("प्रहरी", 6), ("प्रहरी।", 2), ("प्रहरी!", 1), ("प्रहरी?", 1)]
CONFUSABLES = ["प्रभारी", "प्रहर", "पहाड़ी", "प्रचार", "पुजारी", "भिखारी", "बिहारी",
               "प्रकृति", "हरी", "प्रतिहारी", "प्रयाग", "परिहार", "प्रहार", "पहरा",
               "प्रहलाद", "सहारी", "प्यारी", "प्रहरियों", "हमारी", "तुम्हारी"]
FALLBACK_SENTENCES = [
    "आज मौसम बहुत अच्छा है", "मुझे एक गिलास पानी चाहिए", "कल हम बाज़ार जाएंगे",
    "दरवाज़ा बंद कर दीजिए", "यह किताब बहुत दिलचस्प है", "क्या आपने खाना खा लिया",
    "मैं थोड़ी देर में वापस आती हूँ", "कैंटीन में आज बहुत भीड़ है", "चलो क्लास के लिए देर हो रही है",
    "उसने मुझे फ़ोन पर सब बता दिया", "बस स्टॉप यहाँ से ज़्यादा दूर नहीं है", "मेरी चाय ठंडी हो गई",
]
SPLIT = {"positive": 0.60, "negative": 0.25, "sentence": 0.15}
COST = {"positive": 7, "negative": 6, "sentence": 40}


def load_keys():
    kf = os.path.join(HERE, "elevenlabs_keys.txt")
    keys = [l.strip().split()[-1] for l in open(kf, encoding="utf-8")
            if l.strip() and not l.startswith("#")] if os.path.exists(kf) else []
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k); out.append(k)
    return out


def sentences():
    out = []
    try:
        import pyarrow.parquet as pq
        for f in sorted(glob.glob(os.path.join(DATA, "extracted", "kathbath", "**", "train-*.parquet"),
                                  recursive=True))[:4]:
            for s in pq.read_table(f, columns=["text"]).column("text").to_pylist():
                if s and 15 <= len(s) <= 55 and "प्रहर" not in s and "पहर" not in s:
                    out.append(s.strip())
    except Exception as e:
        print(f"  (Kathbath sentences unavailable: {e}; using the built-in list)")
    random.Random(7).shuffle(out)
    return out or FALLBACK_SENTENCES


def to_window(audio, rng):
    audio = audio.set_frame_rate(SR).set_channels(1).set_sample_width(2)
    parts = detect_nonsilent(audio, min_silence_len=40, silence_thresh=audio.dBFS - 18)
    if parts:
        audio = audio[parts[0][0]:parts[-1][1]]
    if len(audio) >= WINDOW_MS:
        s = (len(audio) - WINDOW_MS) // 2
        return audio[s:s + WINDOW_MS], 0
    slack = WINDOW_MS - len(audio)
    off = rng.randint(0, slack)
    return (AudioSegment.silent(off, SR) + audio + AudioSegment.silent(slack - off, SR)), off


class Manifest:
    def __init__(self):
        for sub in ("positive", "negative", "sentence"):
            os.makedirs(os.path.join(OUT, sub), exist_ok=True)
        fresh = not os.path.exists(MAN) or os.path.getsize(MAN) == 0
        self.done = {}
        if not fresh:
            for r in csv.DictReader(open(MAN, encoding="utf-8")):
                if os.path.exists(os.path.join(OUT, r["file"])):
                    self.done[r["file"]] = r
        self.fh = open(MAN, "a", newline="", encoding="utf-8")
        self.wr = csv.DictWriter(self.fh, fieldnames=FIELDS)
        if fresh:
            self.wr.writeheader(); self.fh.flush()

    def add(self, row):
        self.wr.writerow({k: row.get(k, "") for k in FIELDS}); self.fh.flush()
        self.done[row["file"]] = row

    def count(self, label):
        return sum(1 for r in self.done.values() if r["label"] == label)


async def get_json(s, url, key, **kw):
    async with s.get(url, headers={"xi-api-key": key}, **kw) as r:
        return r.status, (await r.json() if r.content_type == "application/json" else await r.text())


async def remaining_chars(s, key):
    st, d = await get_json(s, f"{API}/user/subscription", key)
    if st != 200 or not isinstance(d, dict):
        msg = d if isinstance(d, str) else str(d.get("detail", d))
        return None, f"HTTP {st}: {msg[:160]}"
    return int(d.get("character_limit", 0)) - int(d.get("character_count", 0)), d.get("tier", "?")


async def account_voices(s, key, n_library, dry):
    st, d = await get_json(s, f"{API}/voices", key)
    voices = []
    if st == 200:
        for v in d.get("voices", []):
            lab = v.get("labels") or {}
            voices.append({"voice_id": v["voice_id"], "voice_name": v.get("name", ""),
                           "gender": (lab.get("gender") or "?").lower(),
                           "accent": (lab.get("accent") or "?").lower(),
                           "voice_source": "account"})
    have = {v["voice_id"] for v in voices}
    added = 0
    for gender in ("female", "female", "male"):
        if added >= n_library:
            break
        params = {"page_size": "30", "language": "hi", "gender": gender}
        st, d = await get_json(s, f"{API}/shared-voices", key, params=params)
        if st != 200 or not isinstance(d, dict):
            print(f"    library search refused ({st}) - stock voices only")
            break
        for v in d.get("voices", []):
            if added >= n_library or v.get("voice_id") in have:
                continue
            if "indian" not in (v.get("accent") or "").lower() and (v.get("language") or "") != "hi":
                continue
            entry = {"voice_id": v["voice_id"], "voice_name": v.get("name", ""), "gender": gender,
                     "accent": (v.get("accent") or "indian").lower(), "voice_source": "library"}
            if dry:
                print(f"    would add library voice: {entry['voice_name']} ({gender}, {entry['accent']})")
                added += 1; have.add(v["voice_id"]); continue
            url = f"{API}/voices/add/{v.get('public_owner_id')}/{v['voice_id']}"
            async with s.post(url, headers={"xi-api-key": key},
                              json={"new_name": f"prahari-{gender}-{v['voice_id'][:6]}"}) as r:
                body = await r.text()
                if r.status == 200:
                    import json as _j
                    entry["voice_id"] = _j.loads(body).get("voice_id", v["voice_id"])
                    voices.append(entry); have.add(entry["voice_id"]); added += 1
                    print(f"    added library voice: {entry['voice_name']} ({gender})")
                else:
                    print(f"    could not add {v.get('name')} ({r.status}: {body[:90]})")
                    if r.status in (401, 403) or "limit" in body.lower():
                        added = n_library
                        break
    return voices


async def synth(s, key, voice, text, rng):
    url = f"{API}/text-to-speech/{voice['voice_id']}?output_format=pcm_16000"
    payload = {"text": text, "model_id": MODEL_ID, "seed": rng.randint(1, 10**6),
               "voice_settings": {"stability": round(rng.uniform(0.15, 0.6), 2),
                                  "similarity_boost": round(rng.uniform(0.5, 0.85), 2),
                                  "style": round(rng.uniform(0.0, 0.45), 2)}}
    for attempt in range(4):
        async with s.post(url, json=payload, headers={"xi-api-key": key}) as r:
            if r.status == 200:
                return await r.read(), payload
            body = (await r.text())[:200]
            if r.status == 429:
                await asyncio.sleep(5 * (attempt + 1)); continue
            if "quota" in body.lower():
                return "QUOTA", payload
            print(f"    HTTP {r.status}: {body[:120]}")
            return None, payload
    return None, payload


def pick_voice(voices, rng):
    fem = [v for v in voices if v["gender"] == "female"]
    pool = fem if fem and rng.random() < 0.70 else voices
    weights = [2.0 if v["voice_source"] == "library" else 1.0 for v in pool]
    return rng.choices(pool, weights=weights, k=1)[0]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="show quotas and voices, spend nothing")
    ap.add_argument("--library", type=int, default=3, help="library voices to add per account")
    ap.add_argument("--reserve", type=int, default=150, help="characters left unspent per key")
    a = ap.parse_args()
    keys = load_keys()
    if not keys:
        sys.exit("no keys in tools/elevenlabs_keys.txt")
    man = Manifest()
    sents = sentences()
    rng = random.Random(int(time.time()))
    idx = {lab: 1 + max([int(os.path.splitext(f)[0].rsplit("_", 1)[1])
                         for f in man.done if f.startswith(lab + "/")] or [0])
           for lab in ("positive", "negative", "sentence")}
    print(f"{len(keys)} keys | already made: " +
          ", ".join(f"{l} {man.count(l)}" for l in ("positive", "negative", "sentence")))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90)) as s:
        for ki, key in enumerate(keys):
            left, tier = await remaining_chars(s, key)
            print(f"\n== key {ki + 1}/{len(keys)}: {tier}, {left} characters left")
            if left is None:
                st, d = await get_json(s, f"{API}/voices", key)
                print(f"   voices list: HTTP {st}" + ("" if st == 200 else f" {str(d)[:160]}"))
                if st != 200:
                    continue
                raw, _ = await synth(s, key, {"voice_id": d["voices"][0]["voice_id"]}, "प्रहरी", random.Random(1))
                if raw in (None, "QUOTA"):
                    print("   test speech: FAILED (" + ("out of characters" if raw == "QUOTA" else "refused") + ")")
                    continue
                print("   test speech: OK - quota unknown, will generate until the key runs out")
                left = 10000
            if left < a.reserve + 50:
                continue
            voices = await account_voices(s, key, a.library, a.plan)
            fem = sum(v["gender"] == "female" for v in voices)
            print(f"   voices: {len(voices)} ({fem} female, "
                  f"{sum(v['voice_source'] == 'library' for v in voices)} from the Indian library)")
            budget = left - a.reserve
            plan = {lab: int(budget * frac / COST[lab]) for lab, frac in SPLIT.items()}
            print(f"   this key can make ~{plan['positive']} keyword, {plan['negative']} sound-alike, "
                  f"{plan['sentence']} sentence clips")
            if a.plan or not voices:
                continue
            spent = 0
            order = [lab for lab in plan for _ in range(plan[lab])]
            rng.shuffle(order)
            for lab in order:
                text = (rng.choices([t for t, _ in POSITIVE_TEXTS], [w for _, w in POSITIVE_TEXTS])[0]
                        if lab == "positive" else rng.choice(CONFUSABLES) if lab == "negative"
                        else rng.choice(sents))
                if spent + len(text) > budget:
                    continue
                v = pick_voice(voices, rng)
                raw, pay = await synth(s, key, v, text, rng)
                if raw == "QUOTA":
                    print("   key out of characters - next key"); break
                if not raw:
                    continue
                spent += len(text)
                audio = AudioSegment(data=raw, sample_width=2, frame_rate=SR, channels=1)
                off = 0
                if lab != "sentence":
                    audio, off = to_window(audio, rng)
                name = f"{lab}/{lab}_{idx[lab]:05d}.wav"; idx[lab] += 1
                audio.export(os.path.join(OUT, name), format="wav")
                man.add({"file": name, "label": lab, "text": text, **v, "key_index": ki,
                         "seed": pay["seed"], "stability": pay["voice_settings"]["stability"],
                         "style": pay["voice_settings"]["style"], "offset_ms": off})
                if sum(idx.values()) % 25 == 0:
                    print(f"   {name}  {v['voice_name'][:24]:<24} {v['gender']:<7} '{text[:30]}'")
                await asyncio.sleep(0.4)
    rows = list(man.done.values())
    pos = [r for r in rows if r["label"] == "positive"]
    print(f"\ncorpus v2: {len(pos)} keyword ({sum(r['gender'] == 'female' for r in pos)} female, "
          f"{sum(r['voice_source'] == 'library' for r in pos)} Indian-library), "
          f"{man.count('negative')} sound-alikes, {man.count('sentence')} sentences, "
          f"{len({r['voice_id'] for r in rows})} voices -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
