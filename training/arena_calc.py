#!/usr/bin/env python3
import sys
from flatbuffers.table import Table
from flatbuffers import encode, packer, number_types as nt

ADD, CONV, DWCONV, FC, LOGISTIC, MAXPOOL, MUL, MEAN = 0, 3, 4, 9, 14, 17, 18, 40
NAMES = {ADD: "ADD", CONV: "CONV_2D", DWCONV: "DEPTHWISE_CONV_2D", FC: "FULLY_CONNECTED",
         LOGISTIC: "LOGISTIC", MAXPOOL: "MAX_POOL_2D", MUL: "MUL", MEAN: "MEAN"}
ALIGN = 16


def _align(x, a=ALIGN):
    return (x + a - 1) // a * a


class Model:
    def __init__(self, path):
        buf = bytearray(open(path, "rb").read())
        self.root = Table(buf, encode.Get(packer.uoffset, buf, 0))
        self.buf = buf

    def _sub(self, tbl, slot):
        o = tbl.Offset(slot)
        if o == 0:
            return []
        st, ln = tbl.Vector(o), tbl.VectorLen(o)
        return [Table(tbl.Bytes, encode.Get(packer.uoffset, tbl.Bytes, st + i * 4) + st + i * 4)
                for i in range(ln)]

    def _ivec(self, tbl, slot):
        o = tbl.Offset(slot)
        if o == 0:
            return []
        st, ln = tbl.Vector(o), tbl.VectorLen(o)
        return [tbl.Get(nt.Int32Flags, st + i * 4) for i in range(ln)]

    def _u32(self, tbl, slot, d=0):
        o = tbl.Offset(slot)
        return tbl.Get(nt.Uint32Flags, o + tbl.Pos) if o else d

    def _i8(self, tbl, slot, d=0):
        o = tbl.Offset(slot)
        return tbl.Get(nt.Int8Flags, o + tbl.Pos) if o else d

    def _i32(self, tbl, slot, d=0):
        o = tbl.Offset(slot)
        return tbl.Get(nt.Int32Flags, o + tbl.Pos) if o else d

    def parse(self):
        codes = []
        for oc in self._sub(self.root, 6):
            dep = self._i8(oc, 4)
            bi = self._i32(oc, 10, dep)
            codes.append(bi if bi else dep)
        buffers = self._sub(self.root, 12)
        self.buf_has_data = [b.Offset(4) != 0 and b.VectorLen(b.Offset(4)) > 0
                             for b in buffers]
        sg = self._sub(self.root, 8)[0]
        self.tensors = []
        for t in self._sub(sg, 4):
            shape = self._ivec(t, 4)
            n = 1
            for d in shape:
                n *= d
            self.tensors.append({"shape": shape, "bytes": n,
                                 "const": self.buf_has_data[self._u32(t, 8)]})
        self.inputs = self._ivec(sg, 6)
        self.outputs = self._ivec(sg, 8)
        self.ops = []
        for op in self._sub(sg, 10):
            self.ops.append({"code": codes[self._u32(op, 4)],
                             "in": self._ivec(op, 6),
                             "out": self._ivec(op, 8)})
        return self


def conv_scratch(in_h, in_w, in_ch, f_h, f_w, out_ch, out_h, out_w,
                 stride_h, stride_w, pad_h, pad_w):
    align_buf = 64
    if f_w == 1 and f_h == 1 and pad_w == 0 and pad_h == 0 and stride_w == 1 and stride_h == 1:
        new_ch = (in_ch + 7) & ~7
        transpose = 2 * (8 * new_ch) if in_w * in_h >= 8 else 0
        return transpose + align_buf

    filter_row = f_w * in_ch
    window_len = f_w * f_h * in_ch
    if filter_row < 16 and window_len >= 16:
        wl = (window_len + 15) & ~15
        return out_ch * 4 + 16 + out_ch * wl + 16 + wl + align_buf

    pad_right = max(0, (out_w - 1) * stride_w + f_w - pad_w - in_w)
    pad_bottom = max(0, (out_h - 1) * stride_h + f_h - pad_h - in_h)
    if pad_w == 0 and pad_h == 0 and pad_right == 0 and pad_bottom == 0:
        input_scratch = 0
    else:
        input_scratch = (in_w + pad_w + pad_right) * (in_h + pad_h + pad_bottom) * in_ch
    aligned_row = ((filter_row + 15) // 16) * 16
    filter_scratch = max(aligned_row * f_h * out_ch, f_w * f_h * in_ch * out_ch)
    return input_scratch + filter_scratch + align_buf + out_ch * 4


def dw_scratch(in_h, in_w, ch, f_h, f_w, out_h, out_w,
               stride_h, stride_w, pad_h, pad_w, ch_mult=1):
    filter_size = f_w * f_h * ch * ch_mult

    if ch_mult == 1 and ch % 8 == 0:
        if f_w == 3 and f_h == 3:
            if ch % 16 == 0 and ((pad_w == 1 and pad_h == 1) or (pad_w == 0 and pad_h == 0)):
                if pad_w or pad_h:
                    pw, ph = pad_w * 2, pad_h * 2
                else:
                    pw = (out_w * stride_w + f_w - 1) - in_w
                    ph = (out_h * stride_h + f_h - 1) - in_h
                if pw or ph:
                    full = (in_w + pw) * (in_h + ph) * ch
                    if full <= 40 * 1024:
                        return filter_size + full + 16
                    strip = (in_w + pw) * f_h * ch
                    return filter_size + strip + 16
                return filter_size + 16
            elif ch >= 12:
                new_ch = (ch + 15) & ~15
                new_filter = 9 * new_ch
                tot_w = pad_w * 2 + max(0, (out_w * stride_w + 2) - in_w)
                tot_h = pad_h * 2 + max(0, (out_h * stride_h + 2) - in_h)
                new_in = (in_w + tot_w) * (in_h + tot_h) * new_ch
                return new_filter + new_in + out_w * out_h * new_ch + 64
            else:
                return 2 * (filter_size + in_w * in_h * ch) + 32
        else:
            total_s16 = 2 * (filter_size + in_w * in_h * ch)
            if total_s16 <= 48 * 1024:
                return total_s16 + 32
            return 2 * filter_size + 2 * in_w * f_h * ch + 32

    if ch_mult == 1 and ch > 3:
        pch = (ch + 7) & ~7
        filter_bytes = f_w * f_h * pch * 2
        input_start = _align(filter_bytes)
        input_bytes = in_w * in_h * pch * 2
        out_start = _align(input_start + input_bytes)
        out_bytes = out_w * out_h * pch
        bias_start = _align(out_start + out_bytes)
        return bias_start + pch * 4 * 3 + 16

    if ch_mult % 4 == 0:
        return 2 * (filter_size + in_w * in_h * ch) + 32
    return 32


def greedy_plan(buffers):
    order = sorted(range(len(buffers)), key=lambda i: -buffers[i]["size"])
    placed = []
    for i in order:
        b = buffers[i]
        conflicts = [p for p in placed
                     if not (p["last"] < b["first"] or p["first"] > b["last"])]
        conflicts.sort(key=lambda p: p["offset"])
        offset = 0
        for c in conflicts:
            if c["offset"] >= offset + b["size"]:
                break
            offset = max(offset, _align(c["offset"] + c["size"]))
        b = dict(b, offset=offset)
        placed.append(b)
    return max((p["offset"] + p["size"] for p in placed), default=0), placed


def analyse(path, esp_nn=True, verbose=True):
    m = Model(path).parse()
    n_ops = len(m.ops)

    first = {}
    last = {}
    for i in m.inputs:
        first[i] = 0
    for oi, op in enumerate(m.ops):
        for t in op["in"]:
            if t >= 0 and not m.tensors[t]["const"]:
                first.setdefault(t, oi)
                last[t] = oi
        for t in op["out"]:
            first.setdefault(t, oi)
            last[t] = max(last.get(t, oi), oi)
    for t in m.outputs:
        last[t] = n_ops - 1

    buffers = []
    for t, f in first.items():
        if m.tensors[t]["const"]:
            continue
        buffers.append({"name": f"t{t}", "size": _align(m.tensors[t]["bytes"]),
                        "first": f, "last": last.get(t, f)})

    scratch_rows = []
    if esp_nn:
        for oi, op in enumerate(m.ops):
            if op["code"] not in (CONV, DWCONV):
                continue
            it = m.tensors[op["in"][0]]["shape"]
            ft = m.tensors[op["in"][1]]["shape"]
            ot = m.tensors[op["out"][0]]["shape"]
            in_h, in_w, in_c = it[1], it[2], it[3]
            out_h, out_w, out_c = ot[1], ot[2], ot[3]
            if op["code"] == CONV:
                f_h, f_w = ft[1], ft[2]
            else:
                f_h, f_w = ft[1], ft[2]
            s_h = max(1, round(in_h / out_h))
            s_w = max(1, round(in_w / out_w))
            p_h = max(0, ((out_h - 1) * s_h + f_h - in_h) // 2)
            p_w = max(0, ((out_w - 1) * s_w + f_w - in_w) // 2)
            if op["code"] == CONV:
                sz = conv_scratch(in_h, in_w, in_c, f_h, f_w, out_c,
                                  out_h, out_w, s_h, s_w, p_h, p_w)
            else:
                sz = dw_scratch(in_h, in_w, in_c, f_h, f_w,
                                out_h, out_w, s_h, s_w, p_h, p_w)
            buffers.append({"name": f"scratch@{oi}", "size": _align(sz),
                            "first": oi, "last": oi})
            scratch_rows.append((oi, NAMES[op["code"]], in_h, in_w, in_c,
                                 f_h, f_w, out_c, sz))

    head, placed = greedy_plan(buffers)

    if verbose:
        print(f"\n=== {path}   ESP-NN {'ON' if esp_nn else 'OFF'} ===")
        peak = 0
        for oi in range(n_ops):
            live = sum(b["size"] for b in buffers
                       if b["first"] <= oi <= b["last"] and b["name"].startswith("t"))
            peak = max(peak, live)
        print(f"peak live activations : {peak:,} B")
        if scratch_rows:
            print("\nESP-NN scratch per layer:")
            for oi, nm, ih, iw, ic, fh, fw, oc, sz in sorted(
                    scratch_rows, key=lambda r: -r[-1]):
                flag = "  <<< dominates" if sz > 20000 else ""
                print(f"  op{oi:<3} {nm:<18} in {ih}x{iw}x{ic:<3} "
                      f"k{fh}x{fw} out_ch {oc:<3} scratch {sz:>7,} B{flag}")
        print(f"\nplanner head          : {head:,} B")
    return head


CTX, MEL = 98, 40

def build(c0, c1, c2, dw_k=(5, 3), pool1=(2, 1), pool2=(2, 2), stem_k=(3, 5), stem_stride=(1, 2)):
    bufs, notes, t = [], [], 0
    def act(h, w, c, first, last, tag):
        bufs.append({"name": tag, "size": _align(h * w * c), "first": first, "last": last})

    h, w = CTX, MEL
    act(h, w, 1, 0, 0, "in"); act(h, w, 1, 0, 1, "bn_mul"); act(h, w, 1, 1, 2, "bn_add")
    op = 2
    sh, sw = stem_stride
    oh, ow = h // sh, w // sw
    ph = max(0, ((oh - 1) * sh + stem_k[0] - h) // 2)
    pw = max(0, ((ow - 1) * sw + stem_k[1] - w) // 2)
    bufs.append({"name": "sc_stem",
                 "size": _align(conv_scratch(h, w, 1, stem_k[0], stem_k[1], c0,
                                             oh, ow, sh, sw, ph, pw)),
                 "first": op, "last": op})
    act(oh, ow, c0, op, op + 1, "stem_out"); h, w = oh, ow; op += 1
    h, w = h // pool1[0], w // pool1[1]
    act(h, w, c0, op, op + 1, "pool1"); op += 1

    def ds(cin, cout, first_op, shortcut_from=None):
        nonlocal h, w, op
        ph_ = dw_k[0] // 2; pw_ = dw_k[1] // 2
        bufs.append({"name": f"sc_dw{first_op}",
                     "size": _align(dw_scratch(h, w, cin, dw_k[0], dw_k[1],
                                               h, w, 1, 1, ph_, pw_)),
                     "first": op, "last": op})
        act(h, w, cin, op, op + 1, f"dw{op}"); op += 1
        bufs.append({"name": f"sc_pw{op}",
                     "size": _align(conv_scratch(h, w, cin, 1, 1, cout,
                                                 h, w, 1, 1, 0, 0)),
                     "first": op, "last": op})
        last = op + 3 if shortcut_from else op + 1
        act(h, w, cout, op, last, f"pw{op}"); op += 1
        if shortcut_from:
            act(h, w, cout, op, op + 2, f"add{op}"); op += 1

    ds(c0, c1, op)
    h, w = h // pool2[0], w // pool2[1]
    act(h, w, c1, op, op + 1, "pool2"); op += 1
    ds(c1, c2, op)
    ds(c2, c2, op, shortcut_from=True)
    ds(c2, c2, op, shortcut_from=True)
    act(1, 1, c2, op, op + 1, "gap"); op += 1
    return bufs

def report(label, **kw):
    bufs = build(**kw)
    head, _ = greedy_plan(bufs)
    acts = [b for b in bufs if not b["name"].startswith("sc_")]
    scr  = [b for b in bufs if b["name"].startswith("sc_")]
    worst = max(scr, key=lambda b: b["size"])
    peak = max(sum(b["size"] for b in acts if b["first"] <= t <= b["last"])
               for t in range(20))
    total = head + 4940
    print(f"{label:<34} head {head:>7,}  arena {total:>7,}  "
          f"peak-act {peak:>6,}  worst-scratch {worst['size']:>6,} ({worst['name']})")
    return total


CANDIDATES = {
    "v4   12/24/40 k5x3 [ON DEVICE]": dict(c0=12, c1=24, c2=40, dw_k=(5, 3)),
    "v5m   8/16/32 k3x3":             dict(c0=8,  c1=16, c2=32, dw_k=(3, 3)),
    "v5p   8/16/32 k5x3":             dict(c0=8,  c1=16, c2=32, dw_k=(5, 3)),
    "v5q   8/32/32 k3x3":             dict(c0=8,  c1=32, c2=32, dw_k=(3, 3)),
    "v5n   8/16/40 k3x3 (40ch trap)": dict(c0=8,  c1=16, c2=40, dw_k=(3, 3)),
    "v5a  16/16/32 k3x3":             dict(c0=16, c1=16, c2=32, dw_k=(3, 3)),
    "v5d  16/16/32 k3x3 pool1(2,2)":  dict(c0=16, c1=16, c2=32, dw_k=(3, 3), pool1=(2, 2)),
    "v5i   8/16/32 k3x3 stem s(2,2)": dict(c0=8,  c1=16, c2=32, dw_k=(3, 3),
                                           stem_stride=(2, 2), pool1=(1, 1)),
}

USAGE = """usage:
  python arena_calc.py <model.tflite> [--no-esp-nn]   analyse a real model
  python arena_calc.py --sweep                        compare v5 candidates
"""

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(USAGE); sys.exit(1)
    if sys.argv[1] == "--sweep":
        print("Predicted tensor arena, ESP-NN on, ESP32-S3.")
        print("Validated: the v4 row must print 76,188 - that is the number")
        print("the board reported on 2026-09-07.\n")
        for label, kw in CANDIDATES.items():
            report(label, **kw)
        print("\nRule of thumb learned the hard way: keep every channel count a")
        print("multiple of 16. ESP-NN's depthwise kernel falls back to an int16")
        print("path otherwise, and that fallback IS the arena (12 ch cost 47.7 KB).")
    else:
        analyse(sys.argv[1], "--no-esp-nn" not in sys.argv)
