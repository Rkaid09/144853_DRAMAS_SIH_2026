import sys, wave, numpy as np
for path in sys.argv[1:]:
    w = wave.open(path); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float)
    n = len(x) // 480
    ks = []
    for i in range(n):
        seg = x[i*480:(i+1)*480] * np.hanning(480)
        spec = np.abs(np.fft.rfft(seg, 16000))
        f = np.argmax(spec)
        ks.append(int(round((f - 400) / 100.0)) % 20)
    jumps = [(i, ks[i-1], ks[i]) for i in range(1, n) if ks[i] != (ks[i-1] + 1) % 20]
    print(f"{path.split('/')[-1]}: {n} blocks = {n*0.03:.2f} s, "
          + ("CONTINUOUS - nothing missing" if not jumps else f"{len(jumps)} JUMPS {jumps[:6]}"))
