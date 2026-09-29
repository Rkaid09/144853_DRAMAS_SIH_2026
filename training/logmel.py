import numpy as np
import prahari_config as cfg


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sample_rate=cfg.SAMPLE_RATE,
                   n_fft=cfg.FFT_SIZE,
                   n_mels=cfg.MEL_BINS,
                   fmin=cfg.MEL_FMIN_HZ,
                   fmax=cfg.MEL_FMAX_HZ):
    n_bins = n_fft // 2 + 1

    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sample_rate).astype(int)
    bin_pts = np.clip(bin_pts, 0, n_bins - 1)

    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        left, centre, right = bin_pts[m], bin_pts[m + 1], bin_pts[m + 2]
        for k in range(left, centre):
            if centre > left:
                fb[m, k] = (k - left) / (centre - left)
        for k in range(centre, right):
            if right > centre:
                fb[m, k] = (right - k) / (right - centre)
        if centre < n_bins:
            fb[m, centre] = 1.0
    return fb


_FB = None


def get_filterbank():
    global _FB
    if _FB is None:
        _FB = mel_filterbank()
    return _FB


def hann_periodic(n):
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


_WIN = hann_periodic(cfg.WIN_SAMPLES)


def frame_signal(x, win=cfg.WIN_SAMPLES, hop=cfg.HOP_SAMPLES):
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def log_mel(x_int16):
    x = np.asarray(x_int16, dtype=np.float64) / 32768.0

    frames = frame_signal(x)
    frames = frames * _WIN

    padded = np.zeros((frames.shape[0], cfg.FFT_SIZE))
    padded[:, :cfg.WIN_SAMPLES] = frames
    spec = np.fft.rfft(padded, n=cfg.FFT_SIZE, axis=1)

    power = (spec.real ** 2 + spec.imag ** 2)

    mel = power @ get_filterbank().T

    return np.log(mel + cfg.LOG_EPSILON).astype(np.float32)


def quantise(logmel):
    q = np.round(logmel / cfg.FEAT_SCALE) + cfg.FEAT_ZERO_POINT
    return np.clip(q, -128, 127).astype(np.int8)


def dequantise(q):
    return (q.astype(np.float32) - cfg.FEAT_ZERO_POINT) * cfg.FEAT_SCALE


def load_wav_int16(path):
    import wave
    with wave.open(path, "rb") as w:
        assert w.getframerate() == cfg.SAMPLE_RATE, f"{path}: wrong rate"
        assert w.getnchannels() == 1, f"{path}: not mono"
        assert w.getsampwidth() == 2, f"{path}: not 16-bit"
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


if __name__ == "__main__":
    fb = get_filterbank()
    print(f"filterbank        : {fb.shape}  (mel bins x fft bins)")
    print(f"non-zero weights  : {int((fb > 0).sum())}")
    print(f"window            : Hann periodic, {cfg.WIN_SAMPLES} samples")
    dummy = np.zeros(cfg.SAMPLE_RATE, dtype=np.int16)
    print(f"frames for 1.0 s  : {log_mel(dummy).shape[0]}  "
          f"(contract says {cfg.CTX_FRAMES})")
