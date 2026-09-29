import os, glob, wave
import numpy as np
import prahari_config as cfg

SR   = cfg.SAMPLE_RATE
WIN  = SR
NOISE_DIR = os.path.join("..", "tools", "dataset_neg", "noise")
BOARD_NOISE_DIR = os.path.join("..", "tools", "dataset_board_noise")
BOARD_NOISE_P = 0.5
BABBLE_DIR = os.path.join("..", "tools", "dataset_babble")
BABBLE_P = 0.30
PITCH_P = 0.5
CLIP_P = 0.08
PITCH_SEMITONES = (-4.0, 7.0)

_noise_cache = None
_board_cache = None
_babble_cache = None


def load_babble_pool(limit=3000):
    global _babble_cache
    if _babble_cache is not None:
        return _babble_cache
    pool = []
    for f in sorted(glob.glob(os.path.join(BABBLE_DIR, "*.wav")))[:limit]:
        try:
            with wave.open(f, "rb") as w:
                if w.getframerate() != SR or w.getnchannels() != 1:
                    continue
                x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            pool.append(x.astype(np.float32) / 32768.0)
        except Exception:
            continue
    _babble_cache = pool
    return pool


def _stft(x, n=512, hop=128):
    win = np.hanning(n).astype(np.float32)
    pad = np.pad(x, (n // 2, n // 2))
    frames = 1 + (len(pad) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(frames)[:, None]
    return np.fft.rfft(pad[idx] * win, axis=1), win


def _istft(S, win, length, n=512, hop=128):
    frames = S.shape[0]
    out = np.zeros(n + hop * (frames - 1), dtype=np.float64)
    norm = np.zeros_like(out)
    y = np.fft.irfft(S, n=n, axis=1)
    for i in range(frames):
        out[i * hop:i * hop + n] += y[i] * win
        norm[i * hop:i * hop + n] += win * win
    out = out / np.maximum(norm, 1e-8)
    out = out[n // 2:n // 2 + length]
    return np.pad(out, (0, max(0, length - len(out)))).astype(np.float32)


def _pv(x, r):
    n, hop = 512, 128
    S, win = _stft(x, n, hop)
    steps = np.arange(0, S.shape[0] - 1, 1.0 / r)
    mag, ph = np.abs(S), np.angle(S)
    omega = 2 * np.pi * hop * np.arange(S.shape[1]) / n
    acc = ph[0].copy()
    out = np.empty((len(steps), S.shape[1]), dtype=np.complex64)
    for k, t in enumerate(steps):
        i = int(t); f = t - i
        m = (1 - f) * mag[i] + f * mag[i + 1]
        out[k] = m * np.exp(1j * acc)
        dp = ph[i + 1] - ph[i] - omega
        dp -= 2 * np.pi * np.round(dp / (2 * np.pi))
        acc += omega + dp
    return _istft(out, win, int(round(len(x) * r)), n, hop)


def time_stretch(x, factor):
    if abs(factor - 1.0) < 1e-3 or len(x) < 1024:
        return x.astype(np.float32)
    return _pv(x.astype(np.float32), float(factor))


def pitch_shift(x, semitones):
    r = 2.0 ** (semitones / 12.0)
    if abs(r - 1.0) < 1e-3 or len(x) < 1024:
        return x
    y = _pv(x, r)
    idx = np.linspace(0, len(y) - 1, len(x))
    return np.interp(idx, np.arange(len(y)), y).astype(np.float32)


def load_board_pool(limit=2000):
    global _board_cache
    if _board_cache is not None:
        return _board_cache
    pool = []
    for f in sorted(glob.glob(os.path.join(BOARD_NOISE_DIR, "*.wav")))[:limit]:
        try:
            with wave.open(f, "rb") as w:
                if w.getframerate() != SR or w.getnchannels() != 1:
                    continue
                x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            pool.append(x.astype(np.float32) / 32768.0)
        except Exception:
            continue
    _board_cache = pool
    return pool


def load_noise_pool(limit=400):
    global _noise_cache
    if _noise_cache is not None:
        return _noise_cache
    files = sorted(glob.glob(os.path.join(NOISE_DIR, "*.wav")))[:limit]
    pool = []
    for f in files:
        try:
            with wave.open(f, "rb") as w:
                if w.getframerate() != SR or w.getnchannels() != 1:
                    continue
                x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            pool.append(x.astype(np.float32) / 32768.0)
        except Exception:
            continue
    _noise_cache = pool
    return pool


def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def _fit(x, rng):
    if len(x) == WIN:
        return x
    if len(x) > WIN:
        s = rng.integers(0, len(x) - WIN + 1)
        return x[s:s + WIN]
    out = np.zeros(WIN, dtype=np.float32)
    s = rng.integers(0, WIN - len(x) + 1)
    out[s:s + len(x)] = x
    return out


def _synth_rir(rng, sr=16000):
    rt60 = rng.uniform(0.12, 0.35)
    n = max(8, int(rt60 * sr))
    t = np.arange(n) / float(sr)
    h = rng.standard_normal(n) * np.exp(-6.9078 * t / rt60)
    h[1:] *= rng.uniform(0.10, 0.55)
    h[0] = 1.0
    e = float(np.sqrt(np.sum(h * h)))
    return h / e if e > 0 else h


def _reverb(x, rng):
    h = _synth_rir(rng)
    L = len(x) + len(h) - 1
    nfft = 1 << (L - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(h, nfft), nfft)
    return y[:len(x)]


def augment(x_int16, seed):
    rng = np.random.default_rng(seed)
    x = x_int16.astype(np.float32) / 32768.0

    rate = rng.uniform(0.85, 1.18)
    if abs(rate - 1.0) > 0.005:
        n_out = max(8, int(round(len(x) / rate)))
        idx = np.linspace(0, len(x) - 1, n_out)
        x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    x = _fit(x, rng)

    if rng.random() < PITCH_P:
        x = pitch_shift(x, float(rng.uniform(*PITCH_SEMITONES)))

    if rng.random() < 0.7:
        a = float(rng.uniform(-0.4, 0.4))
        before = _rms(x)
        y = np.empty_like(x)
        y[0] = x[0]
        y[1:] = x[1:] - a * x[:-1]
        after = _rms(y)
        if after > 1e-6 and before > 1e-6:
            x = (y * (before / after)).astype(np.float32)

    sh = int(rng.integers(-2400, 2401))
    hop = 160
    nf = len(x) // hop
    if nf > 2:
        e = np.sqrt((x[: nf * hop].reshape(nf, hop).astype(np.float64) ** 2).mean(axis=1))
        act = np.flatnonzero(e > max(e.max() * 0.06, 1e-5))
        if len(act):
            lo, hi = -int(act[0] * hop), int(len(x) - (act[-1] + 1) * hop)
            sh = int(np.clip(sh, min(lo, 0), max(hi, 0)))
    x = np.roll(x, sh)

    if rng.random() < 0.35:
        x = _reverb(x, rng)

    pool = load_noise_pool()
    bpool = load_board_pool()
    cpool = load_babble_pool()
    snr_lo, snr_hi = -5.0, 20.0
    u = rng.random()
    if cpool and u < BABBLE_P:
        pool, snr_lo, snr_hi = cpool, 0.0, 15.0
    elif bpool and rng.random() < BOARD_NOISE_P:
        pool = bpool
    if pool and rng.random() < 0.85:
        n = pool[int(rng.integers(len(pool)))]
        n = _fit(n, rng)
        sx, sn = _rms(x), _rms(n)
        if sx > 1e-5 and sn > 1e-5:
            snr_db = rng.uniform(snr_lo, snr_hi)
            target = sx / (10.0 ** (snr_db / 20.0))
            x = x + n * (target / sn)

    if rng.random() < CLIP_P:
        pk = float(np.max(np.abs(x))) if len(x) else 0.0
        if pk > 1e-4:
            x = (np.clip(x / pk * rng.uniform(1.3, 3.0), -1.0, 1.0) * pk).astype(np.float32)

    x = x * (10.0 ** (rng.uniform(-20.0, 6.0) / 20.0))

    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 0.99:
        x = x * (0.99 / peak)

    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)


if __name__ == "__main__":
    pool = load_noise_pool()
    print(f"noise pool      : {len(pool)} clips from {NOISE_DIR}")
    demo = (np.random.default_rng(0).normal(0, 0.05, WIN) * 32767).astype(np.int16)
    for s in range(3):
        y = augment(demo, s)
        print(f"  seed {s}: len {len(y)}, rms {_rms(y/32768.0):.4f}, "
              f"peak {np.max(np.abs(y))}")
