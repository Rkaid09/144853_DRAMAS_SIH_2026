#!/usr/bin/env python3
import argparse, os, shutil, subprocess, sys, tarfile, time

SOURCES = {
    "slr104": [
        ("https://openslr.trmal.net/resources/104/Hindi-English_train.tar.gz",
         "https://openslr.elda.org/resources/104/Hindi-English_train.tar.gz"),
        ("https://openslr.trmal.net/resources/104/Hindi-English_test.tar.gz",
         "https://openslr.elda.org/resources/104/Hindi-English_test.tar.gz"),
    ],
    "musan": [
        ("https://openslr.trmal.net/resources/17/musan.tar.gz",
         "https://openslr.elda.org/resources/17/musan.tar.gz"),
    ],
    "fsd50k": [
        ("https://zenodo.org/api/records/4060432/files/FSD50K.eval_audio.z01/content", None, "FSD50K.eval_audio.z01"),
        ("https://zenodo.org/api/records/4060432/files/FSD50K.eval_audio.zip/content", None, "FSD50K.eval_audio.zip"),
        ("https://zenodo.org/api/records/4060432/files/FSD50K.ground_truth.zip/content", None, "FSD50K.ground_truth.zip"),
    ],
    "fsd50k_dev": [
        *[(f"https://zenodo.org/api/records/4060432/files/FSD50K.dev_audio.z0{i}/content", None,
           f"FSD50K.dev_audio.z0{i}") for i in range(1, 6)],
        ("https://zenodo.org/api/records/4060432/files/FSD50K.dev_audio.zip/content", None, "FSD50K.dev_audio.zip"),
    ],
}


CHUNK = 128 * 2**20
WORKERS = 8


def _size(u):
    import requests
    r = requests.head(u, allow_redirects=True, timeout=30)
    r.raise_for_status()
    n = int(r.headers.get("Content-Length", 0))
    if not n or r.headers.get("Accept-Ranges", "bytes") == "none":
        raise RuntimeError("no size / no range support")
    return r.url, n


def _probe(u, nbytes=3 * 2**20):
    import requests
    t = time.time()
    with requests.get(u, headers={"Range": f"bytes=0-{nbytes - 1}"}, stream=True, timeout=30) as r:
        r.raise_for_status()
        for _ in r.iter_content(1 << 20):
            pass
    return time.time() - t


def _piece(u, dest, k, start, end):
    import requests
    part = f"{dest}.part{k:04d}"
    want = end - start + 1
    for attempt in range(8):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have >= want:
            return have
        try:
            h = {"Range": f"bytes={start + have}-{end}"}
            with requests.get(u, headers=h, stream=True, timeout=60) as r:
                if r.status_code != 206:
                    raise RuntimeError(f"HTTP {r.status_code} (range not honoured)")
                with open(part, "ab") as f:
                    for c in r.iter_content(1 << 20):
                        f.write(c)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            time.sleep(5 + 5 * attempt)
    return os.path.getsize(part) if os.path.exists(part) else 0


def fetch(url, dest, mirror=None, extra_mirrors=()):
    from concurrent.futures import ThreadPoolExecutor
    if os.path.exists(dest + ".complete"):
        print(f"  {os.path.basename(dest)} already complete"); return True
    cands = [u for u in (url, mirror, *extra_mirrors) if u]
    best, best_t, size = None, 1e9, 0
    for u in cands:
        try:
            ru, n = _size(u)
            t = _probe(ru)
            print(f"  mirror {u.split('/')[2]:28} first 3 MB in {t:5.1f} s")
            if t < best_t:
                best, best_t, size = ru, t, n
        except Exception as e:
            print(f"  mirror {u.split('/')[2]:28} unavailable ({e})")
    if not best:
        return False
    pieces = [(k, s, min(s + CHUNK, size) - 1) for k, s in enumerate(range(0, size, CHUNK))]
    print(f"  {os.path.basename(dest)}: {size/2**30:.2f} GB in {len(pieces)} pieces x {WORKERS} connections")
    stop = [False]

    def progress():
        while not stop[0]:
            got = sum(os.path.getsize(f"{dest}.part{k:04d}") for k, _, _ in pieces
                      if os.path.exists(f"{dest}.part{k:04d}"))
            print(f"\r  {got/2**30:6.2f} / {size/2**30:.2f} GB", end="", flush=True)
            time.sleep(10)

    import threading
    th = threading.Thread(target=progress, daemon=True); th.start()
    with ThreadPoolExecutor(WORKERS) as ex:
        got = list(ex.map(lambda p: _piece(best, dest, *p), pieces))
    stop[0] = True
    if any(g < e - s + 1 for g, (k, s, e) in zip(got, pieces)):
        print(f"\n  incomplete - run the script again, it resumes")
        return False
    with open(dest, "wb") as out:
        for k, _, _ in pieces:
            with open(f"{dest}.part{k:04d}", "rb") as f:
                shutil.copyfileobj(f, out, 16 * 2**20)
    if os.path.getsize(dest) != size:
        print("\n  size mismatch after joining - delete the file and rerun"); return False
    for k, _, _ in pieces:
        try: os.remove(f"{dest}.part{k:04d}")
        except OSError: pass
    open(dest + ".complete", "w").write(str(size))
    print(f"\r  {os.path.basename(dest)} done ({size/2**30:.2f} GB)          ")
    return True


def extract_tar(path, out):
    mark = path + ".extracted"
    if os.path.exists(mark):
        return
    print(f"  extracting {os.path.basename(path)} ...")
    with tarfile.open(path) as t:
        t.extractall(out)
    open(mark, "w").write("ok")


HF = {
    "kathbath": ("ai4bharat/Kathbath", ["*hindi*"]),
    "indicvoices": ("ai4bharat/IndicVoices", ["hindi/*"]),
    "shrutilipi": ("ai4bharat/Shrutilipi",
                   [f"hindi/train-{i:05d}-of-00196.parquet" for i in range(0, 196, 5)]),
}
ZIPS = {
    "fsd50k": (("FSD50K.eval_audio.zip", "FSD50K.ground_truth.zip"), "FSD50K.extracted"),
    "fsd50k_dev": (("FSD50K.dev_audio.zip",), "FSD50K_dev.extracted"),
}


def hf_get(name, raw, ext):
    mark = os.path.join(raw, f"{name}.complete")
    if os.path.exists(mark):
        print(f"  {name}: already complete"); return True
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import snapshot_download
    repo, pats = HF[name]
    for attempt in range(1, 61):
        try:
            snapshot_download(repo, repo_type="dataset", allow_patterns=pats, max_workers=8,
                              local_dir=os.path.join(ext, name))
            break
        except Exception as e:
            msg = str(e)
            if "401" in msg or "403" in msg or "gated" in msg.lower():
                print(f"\n  {name}: no access ({msg[:120]}) - accept the terms on "
                      f"huggingface.co/datasets/{repo}, then rerun with --only {name}")
                return False
            print(f"\n  {name}: attempt {attempt} failed ({msg[:120]}) - retrying in 60 s", flush=True)
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            try:
                from huggingface_hub import constants
                constants.HF_HUB_DISABLE_XET = True
            except Exception:
                pass
            time.sleep(60)
    else:
        print(f"\n  {name}: gave up after 60 attempts - rerun with --only {name}")
        return False
    open(mark, "w").write("ok")
    print(f"\n  {name} hindi parquet done")
    return True


def unzip(name, raw, ext):
    zips, markname = ZIPS[name]
    mark = os.path.join(raw, markname)
    if os.path.exists(mark):
        return
    z = shutil.which("7z") or r"C:\Program Files\7-Zip\7z.exe"
    if not (os.path.exists(z) or shutil.which("7z")):
        print("  7-Zip not found: install it (winget install 7zip.7zip), then run with --only " + name)
        return
    if not all(os.path.exists(os.path.join(raw, zp + ".complete")) for zp in zips):
        print(f"  {name}: download incomplete - rerun --only {name}"); return
    rc = [subprocess.run([z, "x", "-y", os.path.join(raw, zp), f"-o{os.path.join(ext, 'fsd50k')}"]).returncode
          for zp in zips]
    if not any(rc):
        open(mark, "w").write("ok")
    else:
        print(f"  {name}: 7-Zip error {rc} - rerun --only {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"E:\PRAHARI_DATA")
    ap.add_argument("--only", nargs="+", choices=list(SOURCES) + list(HF),
                    help="one or more sources, e.g. --only indicvoices shrutilipi fsd50k_dev")
    ap.add_argument("--no-extract", action="store_true")
    a = ap.parse_args()
    raw, ext = os.path.join(a.root, "raw"), os.path.join(a.root, "extracted")
    os.makedirs(raw, exist_ok=True); os.makedirs(ext, exist_ok=True)
    free = shutil.disk_usage(a.root).free / 2**30
    print(f"target {a.root}  ({free:.0f} GB free)")

    names = a.only or (list(SOURCES) + list(HF))
    hf_names = [n for n in names if n in HF]
    http_names = [n for n in names if n in SOURCES]
    import threading
    th = None
    if hf_names:
        th = threading.Thread(target=lambda: [hf_get(n, raw, ext) for n in hf_names])
        th.start()
    for name in http_names:
        print(f"\n== {name}")
        for item in SOURCES[name]:
            url, mirror = item[0], item[1]
            fn = item[2] if len(item) > 2 else os.path.basename(url)
            dest = os.path.join(raw, fn)
            extra = ()
            if "openslr" in url:
                path = url.split("/resources/", 1)[1]
                extra = (f"https://openslr.magicdatatech.com/resources/{path}",
                         f"https://us.openslr.org/resources/{path}")
            ok = False
            for attempt in range(1, 31):
                if fetch(url, dest, mirror, extra):
                    ok = True; break
                print(f"  {fn}: attempt {attempt} incomplete - retrying in 60 s", flush=True)
                time.sleep(60)
            if not ok:
                print(f"  FAILED: {fn} - run the script again later, it resumes")
                continue
            if not a.no_extract and fn.endswith(".tar.gz"):
                extract_tar(dest, os.path.join(ext, name))
        if name in ZIPS and not a.no_extract:
            unzip(name, raw, ext)
    if th:
        th.join()
    print("\nall requested sources processed.")

if __name__ == "__main__":
    main()
