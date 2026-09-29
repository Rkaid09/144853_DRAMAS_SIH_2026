#!/usr/bin/env python3
import os, struct, sys

ELF = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    ".pio", "build", "xiao_esp32s3", "firmware.elf")

d = open(ELF, "rb").read()
assert d[:4] == b"\x7fELF", "not an ELF file"
shoff, = struct.unpack_from("<I", d, 0x20)
shent, = struct.unpack_from("<H", d, 0x2E)
shnum, = struct.unpack_from("<H", d, 0x30)
shstr, = struct.unpack_from("<H", d, 0x32)

def sec(i):
    o = shoff + i * shent
    return struct.unpack_from("<IIIIII", d, o)

_, _, _, _, stro, strs = sec(shstr)
st = d[stro:stro + strs]
name = lambda x: st[x:st.index(b"\0", x)].decode()

dram = iram = 0
print(f"{'section':<22}{'addr':>11}{'size':>10}")
for i in range(shnum):
    n, _, _, addr, _, size = sec(i)
    if not size:
        continue
    nm = name(n)
    if nm == ".dram0.dummy":
        print(f"{nm:<22}{addr:>11x}{size:>10,}   (padding - excluded)")
        continue
    if 0x3FC80000 <= addr < 0x3FD00000:
        dram += size
        print(f"{nm:<22}{addr:>11x}{size:>10,}   DRAM")
    elif 0x40370000 <= addr < 0x403E0000:
        iram += size

print()
print(f"STATIC DRAM  {dram:,} B ({dram/1024:.1f} KB)  <- put this in STATIC_DRAM_BYTES")
print(f"static IRAM  {iram:,} B ({iram/1024:.1f} KB)  (code in RAM; not the data budget)")
