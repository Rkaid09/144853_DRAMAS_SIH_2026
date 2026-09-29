import glob, json, os, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FW = os.path.join(ROOT, "firmware", "core")
SHIP = os.path.join(HERE, "artifacts", "ship_0918")
DATA = os.environ.setdefault("PRAHARI_DATA", r"E:\PRAHARI_DATA")
STAMP = time.strftime("%m%d_%H%M")
LOGDIR = os.path.join(ROOT, "logs")
LOG = os.path.join(LOGDIR, f"retrain_{STAMP}.log")
PY = sys.executable
HEADERS = ("prahari_model.h", "prahari_test_clip.h")
KEEP = ("prahari_int8.tflite", "prahari_dscnn.keras", "input_affine.json", "threshold.json",
        "history.json", "human_eval.json", "prahari_model.h", "prahari_test_clip.h")


def say(msg, fh):
    print(msg, flush=True)
    fh.write(msg + "\n"); fh.flush()


def run(name, args, fh, env_extra=None):
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", **(env_extra or {}))
    say(f"\n{'#' * 70}\n# {name}: {' '.join(args)}\n# {time.strftime('%H:%M:%S')}\n{'#' * 70}", fh)
    t0, out = time.time(), []
    p = subprocess.Popen([PY, "-u"] + args, cwd=HERE, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        if "oneDNN" in line or "cpu_feature_guard" in line or line.startswith(("I0000", "W0000")):
            continue
        sys.stdout.write(line); fh.write(line); out.append(line)
    p.wait()
    say(f"# {name}: exit {p.returncode} after {(time.time()-t0)/60:.1f} min", fh)
    return p.returncode == 0, "".join(out)


def block(text, start, n=8):
    i = text.rfind(start)
    return "\n".join(text[i:].splitlines()[:n]) if i >= 0 else f"({start!r} not found)"


def install(src):
    src = os.path.abspath(src)
    for h in HEADERS:
        shutil.copyfile(os.path.join(src, h), os.path.join(FW, h))
    for f in ("prahari_int8.tflite", "input_affine.json", "prahari_dscnn.keras"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copyfile(os.path.join(src, f), os.path.join(HERE, "artifacts", f))
    tj = os.path.join(src, "threshold.json")
    if os.path.exists(tj):
        t = float(json.load(open(tj))["threshold"])
        mp = os.path.join(ROOT, "src", "main.cpp")
        s = open(mp, encoding="utf-8").read()
        s2 = re.sub(r"#define TRIGGER_THRESHOLD\s+[0-9.]+f",
                    f"#define TRIGGER_THRESHOLD      {t:.2f}f", s, count=1)
        if s2 != s:
            shutil.copyfile(mp, mp + f".pre-{os.path.basename(src)}")
            open(mp, "w", encoding="utf-8").write(s2)
        print(f"TRIGGER_THRESHOLD set to {t:.2f} in src/main.cpp (backup kept)")
    print(f"firmware now carries {src}. Rebuild and flash as usual.")


def main():
    argv = sys.argv[1:]
    if argv[:1] == ["--install"]:
        return install(argv[1])
    start = "board"
    if "--from" in argv:
        i = argv.index("--from"); start = argv[i + 1]; del argv[i:i + 2]
    order = ["board", "valtest", "corpus", "train", "export", "clip", "human", "fa"]
    if start not in order:
        sys.exit(f"--from must be one of {order}")
    todo = order[order.index(start):]
    os.makedirs(LOGDIR, exist_ok=True)
    fh = open(LOG, "w", encoding="utf-8")
    say(f"PRAHARI retrain {STAMP}   data {DATA}   log {LOG}", fh)
    outs = {}

    def step(name, args, env=None):
        if name not in todo:
            return True
        ok, outs[name] = run(name, args, fh, env)
        if not ok:
            say(f"\nSTOPPED at '{name}'. Fix it, then: python retrain.py --from {name}", fh)
        return ok

    if "board" in todo:
        if glob.glob(os.path.join(ROOT, "server", "captures", "*-room*.wav")):
            if not step("board", [os.path.join("..", "tools", "prep_board_noise.py")]):
                return 1
        else:
            say("\n# board: no `n` room recordings in server/captures yet - skipped", fh)
    if "corpus" in todo and not glob.glob(os.path.join(ROOT, "tools", "dataset_babble", "*.wav")):
        if not step("corpus", [os.path.join("..", "tools", "prep_babble.py")]):
            return 1
    if not step("valtest", ["build_dataset.py"], {"PRAHARI_VALTEST_ONLY": "1"}):
        return 1
    if not step("corpus", ["prep_corpus.py", "--root", DATA]):
        return 1
    if not step("train", ["train_stream.py", "--root", DATA] + argv):
        return 1
    before = {h: open(os.path.join(FW, h), "rb").read() for h in HEADERS
              if os.path.exists(os.path.join(FW, h))}
    for name, args in (("export", ["export_tflite.py"]), ("clip", ["gen_test_clip.py"])):
        if not step(name, args):
            return 1
    if "clip" in todo:
        for h in HEADERS:
            shutil.copyfile(os.path.join(FW, h), os.path.join(HERE, "artifacts", h))
            if h in before:
                open(os.path.join(FW, h), "wb").write(before[h])
            else:
                shutil.copyfile(os.path.join(SHIP, h), os.path.join(FW, h))
    if not step("human", ["eval_human.py"]):
        return 1
    if "fa" in todo:
        ok1, outs["fa_ship"] = run("fa (shipping model ship_0918)",
                                   ["fa_eval.py", os.path.join("artifacts", "ship_0918", "prahari_int8.tflite")], fh)
        ok2, outs["fa"] = run("fa (new model)", ["fa_eval.py"], fh)
        if not (ok1 and ok2):
            say("\nSTOPPED at 'fa'. Fix it, then: python retrain.py --from fa", fh)
            return 1

    dst = os.path.join(HERE, "artifacts", f"retrain3_{STAMP}")
    os.makedirs(dst, exist_ok=True)
    for f in KEEP:
        src = os.path.join(HERE, "artifacts", f)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(dst, f))

    summ = [f"PRAHARI retrain {STAMP}", f"log: {LOG}", ""]
    if "train" in outs:
        summ += ["TRAINING (evaluation inside the loop)", block(outs["train"], "kept epoch", 2),
                 block(outs["train"], "TEST SET", 8), ""]
    if "human" in outs:
        summ += ["HUMAN RECORDINGS (eval_human.py)", "\n".join(outs["human"].strip().splitlines()[-14:]), ""]
    if "fa" in outs:
        summ += ["FALSE WAKES / HOUR - held-out REPORT half, same audio for both models",
                 "--- shipping model (ship_0918) ---", block(outs["fa_ship"], "ALL HELD-OUT AUDIO COMBINED", 8),
                 block(outs["fa_ship"], "B. the user's", 7),
                 "--- new model ---", block(outs["fa"], "ALL HELD-OUT AUDIO COMBINED", 9),
                 block(outs["fa"], "B. the user's", 8), ""]
    summ += [f"new model kept in {dst}",
             "firmware headers are unchanged (whatever was installed before this run).",
             f"to put the new one on the board:  python retrain.py --install {os.path.relpath(dst, HERE)}"]
    text = "\n".join(summ)
    open(os.path.join(dst, "SUMMARY.txt"), "w", encoding="utf-8").write(text)
    say("\n" + "=" * 70 + "\n" + text + "\n" + "=" * 70, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
