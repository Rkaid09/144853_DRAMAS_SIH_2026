#!/usr/bin/env python3
import socket, statistics, sys, time

host = sys.argv[1] if len(sys.argv) > 1 else "3.110.168.217"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 5005
n = int(sys.argv[3]) if len(sys.argv) > 3 else 20

ms, fails = [], 0
print(f"TCP handshake to {host}:{port}, {n} tries")
for i in range(n):
    t = time.perf_counter()
    try:
        s = socket.create_connection((host, port), timeout=3)
        dt = (time.perf_counter() - t) * 1000
        s.close()
        ms.append(dt)
        print(f"  {i + 1:>2}: {dt:6.1f} ms")
    except OSError as e:
        fails += 1
        print(f"  {i + 1:>2}: FAILED ({e})")
    time.sleep(0.5)
if ms:
    s = sorted(ms)
    p90 = s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]
    print(f"\nlaptop -> {host}  median {statistics.median(s):.1f} ms   p90 {p90:.1f}   "
          f"range {s[0]:.1f}-{s[-1]:.1f}   (n={len(s)}, {fails} failed)")
else:
    print("\nevery try failed: is prahari_recv.py running, and does the EC2 security group allow inbound TCP", port, "?")
