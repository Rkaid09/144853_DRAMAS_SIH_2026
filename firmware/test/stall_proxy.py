import socket, sys, threading, time
LP, TP, STALL = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
ls = socket.socket(); ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
ls.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
ls.bind(("127.0.0.1", LP)); ls.listen(4)
def pipe(a, b, stall):
    try:
        while True:
            d = a.recv(1024)
            if not d: break
            b.sendall(d)
            if stall and b'"wake"' in d:
                print(f"[proxy] wake header seen - freezing uplink {STALL}s", flush=True)
                time.sleep(STALL)
    except OSError: pass
    finally:
        for s in (a, b):
            try: s.shutdown(socket.SHUT_RDWR)
            except OSError: pass
while True:
    c, _ = ls.accept()
    u = socket.create_connection(("127.0.0.1", TP))
    threading.Thread(target=pipe, args=(c, u, True), daemon=True).start()
    threading.Thread(target=pipe, args=(u, c, False), daemon=True).start()
