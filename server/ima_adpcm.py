import struct

STEP = [
      7,     8,     9,    10,    11,    12,    13,    14,    16,    17,
     19,    21,    23,    25,    28,    31,    34,    37,    41,    45,
     50,    55,    60,    66,    73,    80,    88,    97,   107,   118,
    130,   143,   157,   173,   190,   209,   230,   253,   279,   307,
    337,   371,   408,   449,   494,   544,   598,   658,   724,   796,
    876,   963,  1060,  1166,  1282,  1411,  1552,  1707,  1878,  2066,
   2272,  2499,  2749,  3024,  3327,  3660,  4026,  4428,  4871,  5358,
   5894,  6484,  7132,  7845,  8630,  9493, 10442, 11487, 12635, 13899,
  15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767]

IDX_ADJ = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8]


def decode_block(blk):
    pred = struct.unpack_from("<h", blk, 0)[0]
    idx = blk[2]
    out = bytearray()
    for byte in blk[4:]:
        for code in (byte & 0x0F, byte >> 4):
            step = STEP[idx]
            diffq = step >> 3
            if code & 4:
                diffq += step
            if code & 2:
                diffq += step >> 1
            if code & 1:
                diffq += step >> 2
            pred = pred - diffq if (code & 8) else pred + diffq
            if pred > 32767:
                pred = 32767
            elif pred < -32768:
                pred = -32768
            idx += IDX_ADJ[code]
            if idx < 0:
                idx = 0
            elif idx > 88:
                idx = 88
            out += struct.pack("<h", pred)
    return bytes(out)


def decode_stream(data, block_bytes=244):
    pcm = bytearray()
    n = len(data) // block_bytes
    gaps, prev = [], None
    for i in range(n):
        blk = data[i * block_bytes:(i + 1) * block_bytes]
        seq = blk[3]
        if prev is not None and seq != (prev + 1) % 256:
            gaps.append((i, prev, seq))
        prev = seq
        pcm += decode_block(blk)
    return bytes(pcm), n, len(data) - n * block_bytes, gaps


if __name__ == "__main__":
    import math, random
    def encode(samples):
        pred, idx = 0, 0
        blk = bytearray(struct.pack("<hBB", 0, 0, 0))
        nib = []
        for sm in samples:
            step = STEP[idx]
            diff = sm - pred
            code = 0
            if diff < 0:
                code, diff = 8, -diff
            diffq = step >> 3
            if diff >= step:
                code |= 4; diff -= step; diffq += step
            step >>= 1
            if diff >= step:
                code |= 2; diff -= step; diffq += step
            step >>= 1
            if diff >= step:
                code |= 1; diffq += step
            pred = pred - diffq if (code & 8) else pred + diffq
            pred = max(-32768, min(32767, pred))
            idx = max(0, min(88, idx + IDX_ADJ[code]))
            nib.append(code)
        for i in range(0, len(nib), 2):
            blk.append(nib[i] | (nib[i + 1] << 4))
        return bytes(blk)

    random.seed(1)
    sig = [int(12000 * math.sin(i / 9.0) + random.randint(-600, 600))
           for i in range(480)]
    back = struct.unpack("<480h", decode_block(encode(sig)))
    err = [abs(a - b) for a, b in zip(sig, back)]
    rms = math.sqrt(sum(e * e for e in err) / len(err))
    peak = math.sqrt(sum(v * v for v in sig) / len(sig))
    print(f"round trip over {len(sig)} samples")
    print(f"  max error  {max(err)}")
    print(f"  rms error  {rms:.1f}   signal rms {peak:.0f}"
          f"   SNR {20*math.log10(peak/max(rms,1e-9)):.1f} dB")
    assert rms < peak * 0.1, "ADPCM round trip is worse than 10% - check the tables"
    print("  OK")
