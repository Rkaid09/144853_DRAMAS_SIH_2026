#!/usr/bin/env python3
"""
PRAHARI - voice recording tool
==============================
Thank you for helping! This records you saying one Hindi word, प्रहरी
(pra-ha-ree), plus a few other words and sentences. It takes about
10-12 minutes. Everything stays on your laptop until you send the zip.
These recordings are used ONLY to TEST the detector - never to train it.

WHAT YOU NEED TO DO
    1.  Install Python from python.org (tick "Add python.exe to PATH").
    2.  Open Command Prompt and run:   pip install sounddevice numpy
    3.  Put this file on your Desktop, then run:
            cd Desktop
            python record_prahari.py
    4.  Follow the prompts. When it says DONE, send the .zip file it made
        (it is on your Desktop too) back to Rajani.

If it stops halfway (battery, a call, anything), just run it again - it
continues where you left off and never re-records finished clips.

TIPS FOR GOOD RECORDINGS
    * A normal room is perfect - we WANT some real-life sound. But no one
      else should be saying the word while you record.
    * Say the word the way you naturally would to call someone. Do not
      read it robotically - variety is what the model needs.
    * Watch the screen: 3 ... 2 ... 1 ... BEEP ... ">>>>> SPEAK NOW <<<<<".
      Speak right after that line appears. Each take is 2 seconds.
    * After each block you see how many takes need redoing. Redo them.
"""
import csv, datetime, json, os, sys, time, wave, zipfile

SR = 16000
TAKE_S = 2.0
SENT_S = 5.0
TALK_S = 60.0
ROOM_S = 10.0

KW = "प्रहरी"
KW_SAY = 'प्रहरी  (pra-ha-ree)'

PLAN = [
    ("k1", "keyword", "near",   f"Laptop in front of you, arm's length. Say {KW_SAY} normally.", 15),
    ("k2", "keyword", "1m",     f"Move back so you are about 1 METRE from the laptop. Say {KW_SAY} normally.", 15),
    ("k3", "keyword", "far",    f"Go 2-3 METRES away (across the room). Say {KW_SAY} a bit louder, like calling someone.", 10),
    ("k4", "keyword", "soft",   f"Back at arm's length. Say {KW_SAY} SOFTLY, like someone is sleeping nearby.", 10),
    ("k5", "keyword", "fast",   f"Say {KW_SAY} QUICKLY and casually - but still all three syllables.", 10),
    ("k6", "keyword", "noisy",  f"Turn ON a fan, TV or music (not too loud), 1 metre away. Say {KW_SAY} normally.", 10),
    ("c1", "confusable", "near", "Say the word shown on screen (NOT prahari). Arm's length.", 20),
    ("s1", "speech", "near",    "Read the sentence shown on screen, normally.", 8),
    ("t1", "talk", "near",      "Talk freely for 60 seconds - your day, a movie, anything. Do NOT say prahari.", 1),
    ("r1", "room", "near",      "SILENCE for 10 seconds. Don't speak, just let the room be.", 1),
]
CONFUSABLES = [
    ("प्रहर", "pra-har"), ("पहरी", "pah-ree"), ("प्रहार", "pra-haar"), ("प्रभारी", "pra-bhaa-ree"),
    ("पहाड़ी", "pa-haa-ree"), ("बिहारी", "bi-haa-ree"), ("हरी", "ha-ree"), ("प्यारी", "pyaa-ree"),
    ("पुजारी", "pu-jaa-ree"), ("प्रचार", "pra-chaar"),
]
SENTENCES = [
    "आज मौसम बहुत अच्छा है", "मुझे एक कप चाय चाहिए", "कल हम बाज़ार जाएंगे",
    "दरवाज़ा बंद कर दो प्लीज़", "मेरी क्लास दस बजे शुरू होती है", "तुमने खाना खा लिया क्या",
    "Can you send me the notes", "Let's meet at the canteen", "फ़ोन की बैटरी खत्म हो गई",
    "बस स्टॉप यहाँ से पास है", "What time is the lab tomorrow", "मैं थोड़ी देर में आती हूँ",
]

RMS_QUIET, CLIP_LEVEL = 150, 32000

HOW_TO_SAY = """
HOW TO SAY THE WORD  -  प्रहरी  (sentinel, guard)
    pra - ha - REE        three syllables, ends in a LONG "ee"
    NOT  प्रहर  (pra-har)       - no "ee" at the end
    NOT  प्रहारी (pra-HAA-ree)  - the middle "ha" is short
    NOT  पहरी  (pah-ree)       - keep the "r" in "pra"
Say it the way you would call out to a guard: clear, natural, not robotic.
"""


def need(mod):
    try:
        return __import__(mod)
    except ImportError:
        print(f"\nMissing package '{mod}'. Run:   pip install sounddevice numpy\n")
        sys.exit(1)


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
np = need("numpy")
sd = need("sounddevice")


def beep():
    try:
        if os.name == "nt":
            import winsound
            winsound.Beep(1000, 200)
            return
    except Exception:
        pass
    try:
        sr = 44100
        t = np.arange(int(0.2 * sr)) / sr
        tone = 0.3 * np.sin(2 * np.pi * 1000 * t)
        x = np.concatenate([np.zeros(int(0.3 * sr)), tone]).astype(np.float32)
        sd.play(x, sr); sd.wait()
    except Exception:
        pass


def go():
    for n in (3, 2, 1):
        print(f"      {n} ...", end="\r", flush=True)
        time.sleep(0.35)
    beep()


def done():
    print("      ...... recorded", flush=True)


LEAD_S = 0.35


def record(seconds, what="SPEAK NOW"):
    x = sd.rec(int((seconds + LEAD_S) * SR), samplerate=SR, channels=1, dtype="int16")
    time.sleep(LEAD_S)
    print(f"      >>>>>   {what}   ({seconds:g} s)   <<<<<      ", flush=True)
    sd.wait()
    return x[int(LEAD_S * SR):, 0]


def playback(x):
    try:
        pad = np.zeros(int(0.3 * SR), np.float32)
        sd.play(np.concatenate([pad, x.astype(np.float32) / 32768.0]), SR); sd.wait()
    except Exception as e:
        print(f"     (could not play it back: {e})")


def save(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(x.astype("<i2").tobytes())


def check_word(x):
    xf = x.astype(np.float64)
    hop = SR // 100
    n = len(xf) // hop
    e = np.sqrt((xf[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    floor = np.percentile(e, 20) + 1
    loud = e > max(floor * 4, RMS_QUIET)
    rms = float(e[loud].mean()) if loud.any() else float(e.mean())
    peak = int(np.abs(x).max())
    if int((np.abs(x.astype(np.int32)) >= CLIP_LEVEL).sum()) > 20:
        return False, "too loud (distorted) - step back a little", rms, peak
    if not loud.any():
        return False, "no voice heard - speak up or come closer", rms, peak
    on = np.flatnonzero(loud)
    if on[0] < 5:
        return False, "started too early - wait for the beep", rms, peak
    if on[-1] > n - 5:
        return False, "cut off at the end - speak right after the beep", rms, peak
    dur = (on[-1] - on[0]) / 100
    if dur > 1.6:
        return False, "too long - just the one word", rms, peak
    if dur < 0.2:
        return False, "too short - no whole word heard (a click or a cut-off word?)", rms, peak
    return True, "ok", rms, peak


def ask(prompt, choices):
    while True:
        a = input(prompt).strip().lower()
        if a in choices:
            return a


def main():
    print(__doc__.split("TIPS")[0].split("WHAT YOU NEED")[0])
    name = ""
    while not name.replace("_", "").isalnum():
        name = input("Your first name (letters only): ").strip().lower().replace(" ", "_")
    voice = ask("Your voice - type f (female) or m (male): ", {"f", "m"})
    print("\nYour recordings will be used ONLY to TEST the PRAHARI keyword detector")
    print("(a student project) - never to train it, never published or shared.")
    if ask("Do you agree? (y/n): ", {"y", "n"}) != "y":
        print("No problem - nothing was recorded. Thank you!"); return

    home = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(home, f"prahari_{name}")
    os.makedirs(out, exist_ok=True)
    info = os.path.join(out, "info.json")
    if not os.path.exists(info):
        json.dump({"name": name, "voice": "female" if voice == "f" else "male", "consent": True,
                   "started": datetime.datetime.now().isoformat(timespec="seconds"),
                   "tool": "record_prahari.py v3", "purpose": "evaluation only", "sample_rate": SR}, open(info, "w"), indent=1)
    man_p = os.path.join(out, "manifest.csv")
    fields = ["file", "label", "condition", "text", "rms", "peak", "ok", "note"]
    have = set()
    if os.path.exists(man_p):
        have = {r["file"] for r in csv.DictReader(open(man_p, encoding="utf-8"))}
    fh = open(man_p, "a", newline="", encoding="utf-8")
    wr = csv.DictWriter(fh, fieldnames=fields)
    if not have:
        wr.writeheader()

    print("\n--- MIC CHECK: say 'hello, testing' after the beep ---")
    input("Press Enter when ready...")
    go(); x = record(3.0); done()
    ok, why, rms, peak = check_word(x[: int(2.0 * SR)])
    level = "GOOD" if 400 < rms < 12000 else ("TOO QUIET - move closer / raise mic volume" if rms <= 400 else "TOO LOUD")
    print(f"   level {rms:.0f} -> {level}\n")

    if not have:
        print(HOW_TO_SAY)
        while True:
            input("PRACTICE (not saved): press Enter, wait for the beep, say प्रहरी ...")
            go(); x = record(TAKE_S); done()
            print("   playing it back ...")
            playback(x)
            if ask("   Did that sound like  pra-ha-REE ?  y = yes, start   r = practise again : ", {"y", "r"}) == "y":
                break

    total = sum(b[4] for b in PLAN)
    done_n = len(have)
    for bid, label, cond, instr, count in PLAN:
        names = [f"{bid}_{label}_{cond}_{i:03d}.wav" for i in range(count)]
        todo = [i for i, f in enumerate(names) if f not in have]
        if not todo:
            continue
        print("=" * 64)
        print(f"BLOCK {bid.upper()}  ({len(todo)} takes)   progress {done_n}/{total}")
        print(f">> {instr}")
        if label == "keyword":
            print("   (pra - ha - REE : three syllables, long 'ee' at the end)")
        print("=" * 64)
        input("Get into position, then press Enter to start the block...")
        for i in todo:
            text, show, secs = KW, KW_SAY, TAKE_S
            if label == "confusable":
                w, r = CONFUSABLES[i % len(CONFUSABLES)]
                text, show = w, f"{w}  ({r})"
            elif label == "speech":
                text = show = SENTENCES[i % len(SENTENCES)]; secs = SENT_S
            elif label == "talk":
                text, show, secs = "", "talk freely (NOT the keyword)", TALK_S
            elif label == "room":
                text, show, secs = "", "(silence)", ROOM_S
            print(f"\n  [{i + 1}/{count}]  say:  {show}")
            while True:
                go()
                x = record(secs, "STAY SILENT" if label == "room" else "SPEAK NOW")
                done()
                if label in ("keyword", "confusable"):
                    ok, why, rms, peak = check_word(x)
                else:
                    ok, why = True, "ok"
                    rms = float(np.sqrt((x.astype(np.float64) ** 2).mean())); peak = int(np.abs(x).max())
                if ok:
                    break
                print(f"     !! the checker thinks: {why}.  Listen and decide:")
                playback(x)
                a = ask("     k = keep it (sounded fine)   r = redo   p = play again : ", {"r", "k", "p"})
                while a == "p":
                    playback(x)
                    a = ask("     k = keep   r = redo   p = play again : ", {"r", "k", "p"})
                if a == "k":
                    ok, why = True, "kept by speaker after listening"
                    break
                print(f"  [{i + 1}/{count}]  again:  {show}")
            f = names[i]
            save(os.path.join(out, f), x)
            wr.writerow({"file": f, "label": label, "condition": cond, "text": text,
                         "rms": round(rms), "peak": peak, "ok": int(ok), "note": "" if why == "ok" else why})
            fh.flush()
            done_n += 1
            print("     saved" if ok else "     kept")
    fh.close()

    zpath = os.path.join(home, f"prahari_{name}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(out)):
            z.write(os.path.join(out, fn), os.path.join(f"prahari_{name}", fn))
    print("\n" + "=" * 64)
    print(f"DONE - thank you! Please send this file to Rajani:\n   {zpath}")
    print("=" * 64)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped. Run the script again any time - it continues where you left off.")
