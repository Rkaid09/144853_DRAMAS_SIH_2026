#!/usr/bin/env python3
import array, json, math, os, queue, socket, struct, sys, threading, time, wave
import ima_adpcm
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE        = os.path.dirname(os.path.abspath(__file__))
TCP_PORT    = int(os.environ.get("PRAHARI_TCP_PORT", 5005))
HTTP_PORT   = int(os.environ.get("PRAHARI_HTTP_PORT", 8000))
DISCOVERY_PORT = int(os.environ.get("PRAHARI_DISCOVERY_PORT", 5006))
SAMPLE_RATE = 16000
CAPTURES    = os.path.join(HERE, "captures")
MODEL_PATH  = os.environ.get("PRAHARI_VOSK_MODEL", "")

os.makedirs(CAPTURES, exist_ok=True)

recognizer_model = None
try:
    from vosk import Model, KaldiRecognizer, SetLogLevel
    SetLogLevel(-1)
    if not MODEL_PATH:
        cands = [d for d in os.listdir(HERE)
                 if d.startswith("vosk-model") and os.path.isdir(os.path.join(HERE, d))]
        MODEL_PATH = os.path.join(HERE, sorted(cands)[0]) if cands else ""
    for _ in range(3):
        if not MODEL_PATH or not os.path.isdir(MODEL_PATH):
            break
        if os.path.isdir(os.path.join(MODEL_PATH, "am")) or \
           os.path.isfile(os.path.join(MODEL_PATH, "final.mdl")) or \
           os.path.isdir(os.path.join(MODEL_PATH, "graph")):
            break
        subs = [d for d in os.listdir(MODEL_PATH)
                if os.path.isdir(os.path.join(MODEL_PATH, d))]
        if len(subs) == 1:
            MODEL_PATH = os.path.join(MODEL_PATH, subs[0])
            print(f"[vosk] descending into {subs[0]}")
        else:
            break

    if MODEL_PATH and os.path.isdir(MODEL_PATH):
        print(f"[vosk] loading {os.path.basename(MODEL_PATH)} ...", flush=True)
        recognizer_model = Model(MODEL_PATH)
        print("[vosk] ready", flush=True)
    else:
        print("[vosk] NO MODEL FOUND - audio will still be captured and the",
              "dashboard will still work, but there will be no transcript.",
              flush=True)
except ImportError:
    print("[vosk] not installed (pip install vosk) - capture only", flush=True)

subscribers = []
sub_lock = threading.Lock()

def publish(evt):
    line = json.dumps(evt)
    with sub_lock:
        dead = []
        for q in subscribers:
            try:
                q.put_nowait(line)
            except queue.Full:
                dead.append(q)
        for q in dead:
            subscribers.remove(q)

def normalise(pcm, target_rms=3000.0, max_gain=20.0):
    if not pcm:
        return pcm, 1.0
    a = array.array("h")
    a.frombytes(pcm[:len(pcm) // 2 * 2])
    if not len(a):
        return pcm, 1.0
    acc = 0
    for v in a:
        acc += v * v
    rms = math.sqrt(acc / len(a))
    if rms < 1.0:
        return pcm, 1.0
    gain = min(target_rms / rms, max_gain)
    peak = max(abs(min(a)), abs(max(a))) or 1
    gain = min(gain, 30000.0 / peak)
    if gain <= 1.05:
        return pcm, 1.0
    for i, v in enumerate(a):
        a[i] = int(max(-32768, min(32767, v * gain)))
    return a.tobytes(), gain


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


def process_utterance(meta, wire, trailer, ended, peer, t_hdr):
    try:
        total = len(wire)
        if meta.get("codec") == "ima-adpcm":
            bb = int(meta.get("block_bytes", 244))
            pcm, nblk, ragged, gaps = ima_adpcm.decode_stream(wire, bb)
            secs = len(pcm) / (2 * SAMPLE_RATE)
            print(f"[dev] {total} B adpcm -> {len(pcm)} B pcm = {secs:.2f} s "
                  f"({nblk} blocks"
                  + (f", {ragged} B TRUNCATED TAIL DROPPED" if ragged else "")
                  + (f", {len(gaps)} GAPS {gaps[:5]}" if gaps else ", no gaps")
                  + f", {total/max(len(pcm),1):.2f}x compression)")
        else:
            pcm = wire
            secs = total / (2 * SAMPLE_RATE)
            print(f"[dev] {total} B = {secs:.2f} s of audio (raw pcm)")
        if trailer:
            print(f"[dev] device says: ack {trailer.get('ack_ms')} ms after the "
                  f"keyword, {trailer.get('dropped', 0)} pre-roll blocks dropped, "
                  f"peak backlog {trailer.get('backlog_max')} blocks")
        if not ended:
            print("[dev] WARNING: link died before the END block - audio is partial")

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        kind = "room" if not meta.get("score") else "wake"
        wav_path = os.path.join(CAPTURES, f"{stamp}-{kind}{meta.get('seq','')}.wav")
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)

        text, t_asr = "", 0.0
        if recognizer_model is not None and pcm:
            pcm_asr, gain = normalise(pcm)
            if gain > 1.05:
                print(f"[asr] level was low - normalised x{gain:.1f}")
            t0 = time.time()
            rec = KaldiRecognizer(recognizer_model, SAMPLE_RATE)
            rec.SetWords(False)
            rec.AcceptWaveform(pcm_asr)
            text = json.loads(rec.FinalResult()).get("text", "")
            t_asr = (time.time() - t0) * 1000.0
            print(f"[asr] \"{text}\"   ({t_asr:.0f} ms)")
        else:
            print("[asr] skipped - no model")

        publish({"type": "transcript", "seq": meta.get("seq"), "text": text,
                 "audio_ms": round(secs * 1000),
                 "asr_ms": round(t_asr),
                 "total_ms": round((time.time() - t_hdr) * 1000),
                 "ack_ms": trailer.get("ack_ms") if trailer else None,
                 "dropped": trailer.get("dropped") if trailer else None,
                 "wav": os.path.basename(wav_path)})
    except Exception as e:
        print(f"[dev] processing error: {e}")


def handle_device(sock, addr):
    peer = addr[0]
    warm = False
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(IDLE_TIMEOUT)
        rf = sock.makefile("rb")
        while True:
            line = rf.readline(4096)
            if not line:
                break
            t_hdr = time.time()
            try:
                meta = json.loads(line.decode("utf-8", "replace"))
            except Exception:
                print(f"[dev] {peer} sent a bad line: {line[:120]!r} - closing")
                break
            ev = meta.get("event")
            if ev == "ping":
                sock.sendall(b'{"event":"pong"}\n')
                continue
            if ev == "hello":
                warm = True
                print(f"\n[dev] {peer} link UP - warm socket, protocol "
                      f"{meta.get('proto')}, fw {meta.get('fw')}")
                publish({"type": "link", "state": "up", "peer": peer, **meta})
                continue
            if ev != "wake":
                print(f"[dev] {peer} unknown event {ev!r} - ignored")
                continue

            framed = meta.get("framing") == "sentinel"
            if framed:
                sock.sendall(json.dumps({"event": "ack", "seq": meta.get("seq")})
                             .encode() + b"\n")
            print(f"\n[dev] wake #{meta.get('seq','?')} from {peer} "
                  f"score={meta.get('score')} "
                  + (f"warm link (ready {meta.get('connect_ms')} ms)" if meta.get("warm")
                     else f"connect={meta.get('connect_ms')}ms"))
            publish({"type": "wake", **meta})

            if framed:
                bb = int(meta.get("block_bytes", 244))
                wire, trailer, ended = read_blocks(rf, bb)
            else:
                wire, trailer, ended = rf.read(), {}, True
            threading.Thread(target=process_utterance,
                             args=(meta, wire, trailer, ended, peer, t_hdr),
                             daemon=True).start()
            if not framed:
                break
    except socket.timeout:
        print(f"[dev] {peer} silent for {IDLE_TIMEOUT} s - closing")
    except OSError as e:
        print(f"[dev] {peer} connection error: {e}")
    except Exception as e:
        print(f"[dev] error: {e}")
    finally:
        try: sock.close()
        except Exception: pass
        if warm:
            print(f"[dev] {peer} link DOWN")
            publish({"type": "link", "state": "down", "peer": peer})

def discovery_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.bind(("0.0.0.0", DISCOVERY_PORT))
    print(f"[udp] discovery on 0.0.0.0:{DISCOVERY_PORT} "
          f"(the device finds us by itself)")
    while True:
        try:
            data, addr = s.recvfrom(64)
            if data.startswith(b"PRAHARI?"):
                s.sendto(f"PRAHARI!{TCP_PORT}".encode(), addr)
                print(f"[udp] device at {addr[0]} asked; told it port {TCP_PORT}")
        except Exception as e:
            print(f"[udp] {e}")
            time.sleep(0.5)


def tcp_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", TCP_PORT))
    s.listen(8)
    print(f"[tcp] listening on 0.0.0.0:{TCP_PORT} (point SERVER_IP here)")
    while True:
        c, a = s.accept()
        threading.Thread(target=handle_device, args=(c, a), daemon=True).start()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path.startswith("/events"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = queue.Queue(maxsize=64)
            with sub_lock:
                subscribers.append(q)
            try:
                self.wfile.write(b": connected\n\n"); self.wfile.flush()
                while True:
                    try:
                        line = q.get(timeout=15)
                        self.wfile.write(f"data: {line}\n\n".encode()); self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n"); self.wfile.flush()
            except Exception:
                pass
            finally:
                with sub_lock:
                    if q in subscribers: subscribers.remove(q)
            return

        if self.path.startswith("/captures/"):
            fn = os.path.join(CAPTURES, os.path.basename(self.path))
            if not os.path.isfile(fn):
                self.send_error(404); return
            body = open(fn, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "none")
            self.end_headers()
            self.wfile.write(body)
            return

        path = "/index.html" if self.path in ("/", "") else self.path
        fn = os.path.join(HERE, "static", os.path.basename(path))
        if not os.path.isfile(fn):
            self.send_error(404); return
        body = open(fn, "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def http_server():
    srv = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), Handler)
    print(f"[web] dashboard on http://localhost:{HTTP_PORT}")
    srv.serve_forever()

if __name__ == "__main__":
    ip = "?"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
    except Exception:
        pass
    print("=" * 62)
    print("PRAHARI demo server")
    print(f"  this machine:               {ip}")
    print(f"  the device DISCOVERS us over UDP {DISCOVERY_PORT} - you should")
    print(f"  not need to put an IP in the firmware at all.")
    print(f"  dashboard:                  http://localhost:{HTTP_PORT}")
    print("=" * 62)
    threading.Thread(target=http_server, daemon=True).start()
    threading.Thread(target=discovery_server, daemon=True).start()
    try:
        tcp_server()
    except KeyboardInterrupt:
        print("\nbye")
