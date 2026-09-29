import asyncio, aiohttp, random, os, io, csv, sys, argparse
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def _load_keys():
    keys = []
    here = os.path.dirname(os.path.abspath(__file__))
    kf = os.path.join(here, "elevenlabs_keys.txt")
    if os.path.exists(kf):
        for line in open(kf, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                keys.append(line)
    env = os.environ.get("ELEVENLABS_API_KEYS") or os.environ.get("ELEVENLABS_API_KEY", "")
    keys += [k.strip() for k in env.split(",") if k.strip()]
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k); out.append(k)
    return out

KEYS = _load_keys()
if not KEYS:
    raise SystemExit(
        "No API keys found.\n"
        "  Put one key per line in tools/elevenlabs_keys.txt   (gitignored)\n"
        "  or set ELEVENLABS_API_KEYS=key1,key2,key3\n"
        "Keys are deliberately NOT stored in this file - this repo is shared.")

_key_i = 0
def current_key():
    return KEYS[_key_i]

def rotate_key(reason=""):
    global _key_i
    if _key_i + 1 >= len(KEYS):
        return False
    _key_i += 1
    print(f"  >>> key {_key_i} exhausted ({reason}); switching to key {_key_i+1}/{len(KEYS)}")
    return True

API_KEY = KEYS[0]

OUT_ROOT   = "dataset_tts"
MODEL_ID   = "eleven_multilingual_v2"
SAMPLE_RATE = 16000
WINDOW_MS   = 1000

N_POSITIVES = 400
N_NEGATIVES = 400

MAX_CONCURRENT = 2
RETRY_ON_429   = 4

POSITIVE_TEXTS = [
    ("प्रहरी",     5),
    ("प्रहरी।",    2),
    ("pra-ha-ree", 2),
    ("prahari",    1),
]

NEGATIVE_TEXTS = [
    "प्रभारी", "प्रहर", "पहाड़ी", "प्रचार", "पुजारी", "भिखारी",
    "बिहारी", "प्रकृति", "हरी", "प्रतिहारी", "प्रयाग", "परिहार",
    "prabhari", "pahari", "prachar", "pujari", "sahara", "hari",
    "prayer", "priority", "pray", "harry", "prahlad", "bihari",
]


def weighted_choice(pairs):
    population = [p for p, w in pairs for _ in range(w)]
    return random.choice(population)


async def fetch_voices(session):
    url = "https://api.elevenlabs.io/v1/voices"
    async with session.get(url, headers={"xi-api-key": current_key()}) as r:
        if r.status != 200:
            print(f"Could not list voices: {r.status} {await r.text()}")
            return []
        data = await r.json()
        voices = [(v["voice_id"], v["name"]) for v in data.get("voices", [])]
        print(f"Loaded {len(voices)} voices.")
        return voices


def to_window(audio: AudioSegment, rng: random.Random):
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)

    parts = detect_nonsilent(audio, min_silence_len=40, silence_thresh=audio.dBFS - 18)
    if parts:
        audio = audio[parts[0][0]:parts[-1][1]]

    if len(audio) >= WINDOW_MS:
        start = (len(audio) - WINDOW_MS) // 2
        placed = audio[start:start + WINDOW_MS]
        offset = 0
    else:
        slack  = WINDOW_MS - len(audio)
        offset = rng.randint(0, slack)
        placed = (AudioSegment.silent(duration=offset, frame_rate=SAMPLE_RATE)
                  + audio
                  + AudioSegment.silent(duration=slack - offset, frame_rate=SAMPLE_RATE))

    return placed.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2), offset


async def synth(session, sem, rows, idx, label, text, voice_id, voice_name):
    async with sem:
        rng   = random.Random(f"{label}{idx}")
        seed  = rng.randint(1, 1_000_000)
        stab  = round(rng.uniform(0.15, 0.55), 2)
        style = round(rng.uniform(0.0, 0.45), 2)

        url = (f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
               f"?output_format=pcm_16000")
        payload = {
            "text": text,
            "model_id": MODEL_ID,
            "seed": seed,
            "voice_settings": {
                "stability": stab,
                "similarity_boost": round(rng.uniform(0.5, 0.85), 2),
                "style": style,
            },
        }
        headers = {"xi-api-key": current_key(), "Content-Type": "application/json"}

        raw, is_pcm = None, True
        for attempt in range(RETRY_ON_429):
            async with session.post(url, json=payload, headers=headers) as r:
                if r.status == 200:
                    raw = await r.read()
                    break
                if r.status == 429:
                    wait = 5 * (attempt + 1)
                    print(f"  rate limited, waiting {wait}s")
                    await asyncio.sleep(wait)
                    continue
                body = (await r.text())[:300]
                if r.status == 401 and "quota" in body.lower():
                    if rotate_key("quota"):
                        headers["xi-api-key"] = current_key()
                        continue
                    print("  ALL KEYS EXHAUSTED - stopping. Re-run after adding more.")
                    return
                if r.status in (400, 401, 403) and is_pcm:
                    is_pcm = False
                    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                    continue
                print(f"  [{label} {idx}] HTTP {r.status}: {body[:160]}")
                return
        if raw is None:
            print(f"  [{label} {idx}] gave up")
            return

        try:
            if is_pcm:
                audio = AudioSegment(data=raw, sample_width=2,
                                     frame_rate=16000, channels=1)
            else:
                audio = AudioSegment.from_file(io.BytesIO(raw), format="mp3")

            placed, offset = to_window(audio, rng)
            fname = f"{label}_{idx:04d}.wav"
            path  = os.path.join(OUT_ROOT, label, fname)
            placed.export(path, format="wav")

            row = {
                "file": f"{label}/{fname}",
                "label": label,
                "text": text,
                "voice_id": voice_id,
                "voice_name": voice_name,
                "seed": seed,
                "stability": stab,
                "style": style,
                "offset_ms": offset,
                "source": "elevenlabs_pcm" if is_pcm else "elevenlabs_mp3",
            }
            rows.append(row)
            record(row)
            print(f"  [{label} {idx}] {voice_name:<22} '{text}'  offset={offset}ms")
        except Exception as e:
            print(f"  [{label} {idx}] audio error: {e}")

        await asyncio.sleep(1.2)


MAN_PATH  = os.path.join(OUT_ROOT, "manifest.csv")
MAN_FIELDS = ["file", "label", "text", "voice_id", "voice_name",
              "seed", "stability", "style", "offset_ms", "source"]
_man_fh = None
_man_wr = None


def open_manifest():
    global _man_fh, _man_wr
    fresh = not os.path.exists(MAN_PATH) or os.path.getsize(MAN_PATH) == 0
    _man_fh = open(MAN_PATH, "a", newline="", encoding="utf-8")
    _man_wr = csv.DictWriter(_man_fh, fieldnames=MAN_FIELDS)
    if fresh:
        _man_wr.writeheader(); _man_fh.flush()


def record(row):
    _man_wr.writerow({k: row.get(k, "") for k in MAN_FIELDS})
    _man_fh.flush()


def already_done():
    done = set()
    if not os.path.exists(MAN_PATH):
        return done
    try:
        for r in csv.DictReader(open(MAN_PATH, encoding="utf-8")):
            f = r.get("file", "")
            if "/" not in f:
                continue
            lab, name = f.split("/", 1)
            if not os.path.exists(os.path.join(OUT_ROOT, lab, name)):
                continue
            stem = os.path.splitext(name)[0]
            try:
                done.add((lab, int(stem.rsplit("_", 1)[1])))
            except (IndexError, ValueError):
                pass
    except Exception as e:
        print(f"  (could not read existing manifest: {e})")
    return done


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=N_POSITIVES)
    ap.add_argument("--neg", type=int, default=N_NEGATIVES)
    args = ap.parse_args()

    for sub in ("positive", "negative"):
        os.makedirs(os.path.join(OUT_ROOT, sub), exist_ok=True)
    open_manifest()
    done = already_done()
    if done:
        print(f"Resuming: {len(done)} clips already present, will not regenerate.")
    print(f"{len(KEYS)} API key(s) loaded. Target {args.pos} positives + "
          f"{args.neg} negatives.\n")

    sem  = asyncio.Semaphore(MAX_CONCURRENT)
    rows = []

    conn = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=conn) as session:
        voices = await fetch_voices(session)
        if not voices:
            sys.exit("No voices available - check the API key.")

        tasks = []
        for i in range(1, args.pos + 1):
            if ("positive", i) in done:
                continue
            vid, vname = random.choice(voices)
            tasks.append(synth(session, sem, rows, i, "positive",
                               weighted_choice(POSITIVE_TEXTS), vid, vname))
        for i in range(1, args.neg + 1):
            if ("negative", i) in done:
                continue
            vid, vname = random.choice(voices)
            tasks.append(synth(session, sem, rows, i, "negative",
                               random.choice(NEGATIVE_TEXTS), vid, vname))
        print(f"{len(tasks)} clips to generate this run.\n")

        random.shuffle(tasks)
        await asyncio.gather(*tasks)

    _man_fh.close()
    allrows = list(csv.DictReader(open(MAN_PATH, encoding="utf-8")))
    pos = sum(1 for r in allrows if r["label"] == "positive")
    neg = len(allrows) - pos
    voices_used = len({r["voice_id"] for r in allrows})
    print(f"\nThis run added {len(rows)} clips.")
    print(f"Corpus now: {pos} positives, {neg} negatives, {voices_used} distinct voices.")
    print(f"Manifest: {MAN_PATH}")
    print("\nIf a key ran out, add more to tools/elevenlabs_keys.txt and re-run -")
    print("finished clips are skipped, so nothing is paid for twice.")
    print("\nNEXT: these are TRAINING data only. The evaluation set must be")
    print("real human recordings, or your numbers will not survive contact")
    print("with a live microphone.")


if __name__ == "__main__":
    asyncio.run(main())
