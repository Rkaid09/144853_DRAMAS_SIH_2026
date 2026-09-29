import csv, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import human_sets as hs

OUT = os.path.join(HERE, "human_review.csv")


def play(path):
    if os.name == "nt":
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME)
    else:
        os.system(f'aplay -q "{path}" 2>/dev/null || afplay "{path}"')


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    done = {}
    if os.path.exists(OUT) and "--redo" not in sys.argv:
        done = {r["file"]: r for r in csv.DictReader(open(OUT, encoding="utf-8"))}
    clips = [c for c in hs.clips(raw=True) if c["label"] == "keyword"]
    random.Random(2709).shuffle(clips)
    todo = [c for c in clips if c["file"] not in done]
    print(f"{len(clips)} keyword clips, {len(clips) - len(todo)} already judged, {len(todo)} to go\n")
    fresh = not done
    fh = open(OUT, "w" if fresh else "a", newline="", encoding="utf-8")
    wr = csv.DictWriter(fh, fieldnames=["file", "speaker", "verdict"])
    if fresh:
        wr.writeheader()
    for k, c in enumerate(todo, 1):
        while True:
            print(f"[{len(clips) - len(todo) + k}/{len(clips)}] playing...", flush=True)
            play(c["path"])
            a = input("   प्रहरी?  y = yes   n = no   u = unsure   r = replay   q = quit : ").strip().lower()
            if a in ("y", "n", "u"):
                wr.writerow({"file": c["file"], "speaker": c["speaker"], "verdict": a}); fh.flush()
                break
            if a == "q":
                fh.close(); print(f"saved -> {OUT}. Run again to continue."); return
    fh.close()
    rows = list(csv.DictReader(open(OUT, encoding="utf-8")))
    print("\nDONE. Per speaker (y / n / u):")
    for s in sorted({r["speaker"] for r in rows}):
        g = [r["verdict"] for r in rows if r["speaker"] == s]
        print(f"   {s:<12} {g.count('y'):>3} / {g.count('n'):>3} / {g.count('u'):>3}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
