#!/usr/bin/env python3
import json, os, socket, struct, sys, threading, time, wave
from datetime import datetime
try:
    import ima_adpcm
except ImportError:
    ima_adpcm = None

PORT = int(os.environ.get("PRAHARI_TCP_PORT", 5005))
CAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
os.makedirs(CAP, exist_ok=True)
n_events = 0
lock = threading.Lock()
ACKS = []


IDLE_TIMEOUT = 45


def read_blocks(rf, bb):
    chunks = []
    while True:
        blk = rf.read(bb)
        if len(blk) < bb:
            return b"".join(chunks) + blk, {}, False
        if blk[2] == 0xFF:
            raw = blk[4:].split(b"\0", 1)[0]
            try:
                trailer = json.loads(raw.decode("utf-8", "replace")) if raw else {}
            except Exception:
                trailer = {"raw": raw[:80].decode("utf-8", "replace")}
            return b"".join(chunks), trailer, True
        chunks.append(blk)


def save_event(hdr, wire, trailer, ended, addr, t0):
    global n_events
    dt = time.time() - t0
    with lock:
        n_events += 1
        k = n_events
    pcm = bytearray(wire)
    codec = (hdr or {}).get("codec")
    if codec == "ima-adpcm" and ima_adpcm is not None:
        bb = int((hdr or {}).get("block_bytes", 244))
        pcm, nblk, ragged, gaps = ima_adpcm.decode_stream(bytes(wire), bb)
        pcm = bytearray(pcm)
        note = f" [adpcm: {nblk} blocks" + (f", {len(gaps)} GAPS {gaps[:5]}]" if gaps else ", no gaps]")
    elif codec == "ima-adpcm":
        note = " [ADPCM, NOT DECODED - ima_adpcm.py missing, wav will be noise]"
    else:
        note = ""
    secs = len(pcm) / 2 / 16000.0
    name = os.path.join(CAP, datetime.now().strftime("%Y%m%d-%H%M%S-") + f"wake{k}.wav")
    if len(pcm) >= 2:
        with wave.open(name, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(bytes(pcm))
    print(f"\n[{k}] from {addr[0]}")
    print(f"     header   : {json.dumps(hdr) if hdr else '(none)'}")
    print(f"     audio    : {len(pcm):,} B = {secs:.2f} s{note}")
    if trailer:
        print(f"     device   : ACK {trailer.get('ack_ms')} ms after the keyword "
              f"(round trip), {trailer.get('dropped', 0)} blocks dropped")
        if isinstance(trailer.get('ack_ms'), (int, float)) and trailer['ack_ms'] >= 0:
            with lock:
                ACKS.append(trailer['ack_ms'])
                a = sorted(ACKS)
            p90 = a[min(len(a) - 1, int(round(0.9 * (len(a) - 1))))]
            med = a[len(a) // 2] if len(a) % 2 else (a[len(a) // 2 - 1] + a[len(a) // 2]) / 2
            print(f"     SO FAR   : ACK round trip median {med:g} ms, p90 {p90} ms, "
                  f"range {a[0]}-{a[-1]} ms  (n={len(a)})")
    if not ended:
        print("     WARNING  : link died before the END block - audio is partial")
    print(f"     utterance: {dt*1000:.0f} ms from header to END")
    print(f"     saved    : {os.path.basename(name)}", flush=True)


def handle(conn, addr):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.settimeout(IDLE_TIMEOUT)
    rf = conn.makefile("rb")
    warm = False
    try:
        while True:
            line = rf.readline(4096)
            if not line:
                break
            t0 = time.time()
            try:
                hdr = json.loads(line.decode("utf-8", "replace"))
            except Exception:
                print(f"[{addr[0]}] bad line {line[:120]!r} - closing")
                break
            ev = hdr.get("event")
            if ev == "ping":
                conn.sendall(b'{"event":"pong"}\n')
                continue
            if ev == "hello":
                warm = True
                print(f"\n[link] {addr[0]} UP - warm socket, protocol {hdr.get('proto')} "
                      f"at {datetime.now():%H:%M:%S}", flush=True)
                continue
            if ev != "wake":
                continue
            framed = hdr.get("framing") == "sentinel"
            if framed:
                conn.sendall(json.dumps({"event": "ack", "seq": hdr.get("seq")})
                             .encode() + b"\n")
                wire, trailer, ended = read_blocks(rf, int(hdr.get("block_bytes", 244)))
            else:
                wire, trailer, ended = rf.read(), {}, True
            threading.Thread(target=save_event,
                             args=(hdr, wire, trailer, ended, addr, t0),
                             daemon=True).start()
            if not framed:
                break
    except socket.timeout:
        print(f"[link] {addr[0]} silent for {IDLE_TIMEOUT} s - closing", flush=True)
    except OSError as e:
        print(f"[link] {addr[0]} {e}", flush=True)
    finally:
        conn.close()
        if warm:
            print(f"[link] {addr[0]} DOWN", flush=True)


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", PORT))
    s.listen(8)
    print("=" * 60)
    print(f"PRAHARI receiver listening on 0.0.0.0:{PORT}")
    print("If the board says 'connect FAILED', the EC2 security group is")
    print(f"not allowing inbound TCP {PORT}. That is the usual cause.")
    print("=" * 60, flush=True)
    while True:
        conn, addr = s.accept()
        threading.Thread(target=handle, args=(conn, addr), daemon=True).start()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
