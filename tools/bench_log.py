#!/usr/bin/env python3
import glob, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

R_WAKE = re.compile(r"\*\*\* PRAHARI \*\*\*\s+#(\d+)\s+score ([\d.]+).*?RMS (\d+)")
R_UPL = re.compile(r"uplink: (WARM|cold) link \| (\d+) ms pre-roll \+ (\d+) ms live \| (\d+) blocks sent.*?stream #(\d+) (.*)$")
R_LAT = re.compile(r"latency: keyword -> link ready (\d+) ms .*?header in TCP (-?\d+) ms \| server ACK (-?\d+) ms.*?END sent (\d+) ms")
R_LAT_OLD = re.compile(r"latency: keyword -> cloud connected (\d+) ms, keyword -> audio uploaded (\d+) ms")
R_CAP = re.compile(r"capture: (\d+) ms audio in (\d+) ms wall \| backlog peak (\d+) blocks.*?mic held (\d+) ms \| (.*)$")
R_DROP = re.compile(r"capture: (\d+) ms of the oldest unsent audio was dropped")
R_RAM = re.compile(r"TOTAL INTERNAL SRAM\s+(\d+) B\s+([\d.]+) KB\s+=\s+([\d.]+)%")
R_HEAP = re.compile(r"heap PEAK ever used\s+(\d+) B")
R_LIVE = re.compile(r"\[live\].*?gate\s+([\d.]+)%.*?front ([\d.]+) infer ([\d.]+) \| CPU (\d+)% = (\d+) front \+ (\d+) net(?:.*?link (\w+) (\d+)us)?")
R_RESET = re.compile(r">> hit counter( and CPU average)? reset")
R_LINK_UP = re.compile(r"link: UP to (\S+) in (\d+) ms")
R_LINK_DOWN = re.compile(r"link: DOWN \((.*?)\)")
R_BOOT = re.compile(r"PRAHARI - LIVE")
R_VAD = re.compile(r"^\s*vad\s+([\d.]+)s")
R_QUIET = re.compile(r">> QUIET RESULT: (\d+) s \| gate opened (\d+) times \| scan duty ([\d.]+)% \| CPU ([\d.]+)%")
QUIET = []


def med(xs):
    return round(statistics.median(xs), 2) if xs else None


def pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def parse(paths):
    ev, cur = [], None
    ram, live, idle_live = [], [], []
    link_up, link_down, boots, vad_lines = [], [], 0, 0
    in_idle = False
    for path in paths:
        for raw in open(path, encoding="utf-8", errors="replace"):
            line = re.sub(r"\x1b\[[0-9;]*m", "", raw).rstrip("\r\n")
            if R_BOOT.search(line):
                boots += 1; in_idle = False
            m = R_WAKE.search(line)
            if m:
                cur = {"n": int(m[1]), "score": float(m[2]), "rms": int(m[3]),
                       "log": os.path.basename(path)}
                ev.append(cur); in_idle = False
                continue
            if R_VAD.search(line):
                vad_lines += 1
            mq = R_QUIET.search(line)
            if mq:
                QUIET.append({"seconds": int(mq[1]), "gate_opens": int(mq[2]),
                              "scan_duty_pct": float(mq[3]), "cpu_pct": float(mq[4])})
            if cur is not None:
                m = R_UPL.search(line)
                if m:
                    cur.update(warm=m[1] == "WARM", live_ms=int(m[3]),
                               blocks=int(m[4]), status=m[6].strip())
                m = R_LAT.search(line)
                if m:
                    cur.update(ready_ms=int(m[1]), hdr_ms=int(m[2]),
                               ack_ms=int(m[3]), end_ms=int(m[4]))
                m = R_LAT_OLD.search(line)
                if m:
                    cur.update(warm=False, ready_ms=int(m[1]), end_ms=int(m[2]))
                m = R_CAP.search(line)
                if m:
                    cur.update(backlog_blocks=int(m[3]), held_ms=int(m[4]),
                               clean=m[5].startswith("clean"))
                m = R_DROP.search(line)
                if m:
                    cur["dropped_ms"] = int(m[1])
            m = R_RAM.search(line)
            if m:
                ram.append({"bytes": int(m[1]), "kb": float(m[2]), "pct": float(m[3])})
            m = R_LIVE.search(line)
            if m:
                row = {"gate": float(m[1]), "front_ms": float(m[2]),
                       "infer_ms": float(m[3]), "cpu": int(m[4]),
                       "link": m[7], "link_us": int(m[8]) if m[8] else None}
                live.append(row)
                if in_idle:
                    idle_live.append(row)
            if R_RESET.search(line):
                in_idle = True; idle_live = []
            m = R_LINK_UP.search(line)
            if m:
                link_up.append(int(m[2]))
            m = R_LINK_DOWN.search(line)
            if m:
                link_down.append(m[1])
    return ev, ram, live, idle_live, link_up, link_down, boots, vad_lines


def summarise(paths):
    ev, ram, live, idle_live, up, down, boots, vad = parse(paths)
    acks = [e["ack_ms"] for e in ev if e.get("ack_ms", -1) >= 0]
    warm = [e for e in ev if e.get("warm")]
    ready_warm = [e["ready_ms"] for e in warm if "ready_ms" in e]
    ready_cold = [e["ready_ms"] for e in ev if not e.get("warm") and "ready_ms" in e]
    with_cap = [e for e in ev if "clean" in e]
    out = {
        "logs": [os.path.basename(p) for p in paths],
        "boots": boots,
        "wake_events": len(ev),
        "latency_ms": {
            "note": "server ACK = round trip measured on the device, 10 ms resolution",
            "server_ack_n": len(acks),
            "server_ack_median": med(acks),
            "server_ack_p90": pct(acks, 90),
            "server_ack_min": min(acks) if acks else None,
            "server_ack_max": max(acks) if acks else None,
            "link_ready_warm_median": med(ready_warm),
            "link_ready_cold_median": med(ready_cold),
        },
        "capture": {
            "events_checked": len(with_cap),
            "clean": sum(1 for e in with_cap if e["clean"]),
            "with_dropped_audio": sum(1 for e in with_cap if not e["clean"]),
            "dropped_ms_total": sum(e.get("dropped_ms", 0) for e in ev),
            "endpointed_on_silence": sum(1 for e in ev if "endpointed" in e.get("status", "")),
            "hit_ceiling": sum(1 for e in ev if "ceiling" in e.get("status", "")),
            "truncated": sum(1 for e in ev if "TRUNCATED" in e.get("status", "")),
            "live_ms_median": med([e["live_ms"] for e in ev if "live_ms" in e]),
        },
        "ram": {
            "peak_kb": max(r["kb"] for r in ram) if ram else None,
            "peak_pct_of_256kb": max(r["pct"] for r in ram) if ram else None,
            "under_200kb_claim": (max(r["kb"] for r in ram) < 200.0) if ram else None,
        },
        "cpu": {
            "front_ms_median": med([r["front_ms"] for r in live]),
            "infer_ms_median": med([r["infer_ms"] for r in live]),
            "idle_after_reset_pct": med([r["cpu"] for r in idle_live[len(idle_live)//2:]]) if idle_live else None,
            "idle_gate_pct": med([r["gate"] for r in idle_live[len(idle_live)//2:]]) if idle_live else None,
            "idle_samples": len(idle_live),
            "link_service_us_median": med([r["link_us"] for r in live if r["link_us"] is not None]),
        },
        "warm_link": {
            "connects": len(up),
            "connect_ms_median": med(up),
            "drops": len(down),
            "drop_reasons": down,
        },
        "vad_trace_lines": vad,
        "quiet_tests": QUIET,
        "events": ev,
    }
    return out


def show(s):
    L, C, R, P, W = s["latency_ms"], s["capture"], s["ram"], s["cpu"], s["warm_link"]
    print("=" * 66)
    print(f"PRAHARI bench - {', '.join(s['logs'])}")
    print("=" * 66)
    print(f"wake events            {s['wake_events']}   (boots in log: {s['boots']})")
    if L["server_ack_n"]:
        print(f"server ACK, round trip median {L['server_ack_median']} ms   "
              f"p90 {L['server_ack_p90']}   range {L['server_ack_min']}-{L['server_ack_max']}"
              f"   (n={L['server_ack_n']})")
    if L["link_ready_warm_median"] is not None:
        print(f"keyword -> link ready  {L['link_ready_warm_median']} ms warm"
              + (f"   vs {L['link_ready_cold_median']} ms cold" if L['link_ready_cold_median'] is not None else ""))
    if C["events_checked"]:
        print(f"captures               {C['clean']}/{C['events_checked']} clean, "
              f"{C['dropped_ms_total']} ms dropped in total | "
              f"{C['endpointed_on_silence']} endpointed, {C['hit_ceiling']} hit the ceiling, "
              f"{C['truncated']} truncated")
    if R["peak_kb"] is not None:
        print(f"RAM peak               {R['peak_kb']} KB = {R['peak_pct_of_256kb']}% of 256 KB   "
              f"-> {'UNDER' if R['under_200kb_claim'] else 'OVER'} the 200 KB claim")
    print(f"front-end / inference  {P['front_ms_median']} ms per 10 ms hop / {P['infer_ms_median']} ms")
    if P["idle_after_reset_pct"] is not None:
        print(f"IDLE CPU (after 'r')   {P['idle_after_reset_pct']}%   gate open {P['idle_gate_pct']}%   "
              f"({P['idle_samples']} status lines)")
    else:
        print("IDLE CPU               not measured - type r, then stay quiet for 2 minutes")
    if P["link_service_us_median"] is not None:
        print(f"warm-link upkeep       {P['link_service_us_median']} us per 10 ms hop")
    print(f"warm link              {W['connects']} connects (median {W['connect_ms_median']} ms), "
          f"{W['drops']} drops {W['drop_reasons'][:3]}")
    for q in s.get("quiet_tests", []):
        print(f"QUIET TEST             {q['seconds']} s silent, serial muted: gate opened "
              f"{q['gate_opens']}x, scan duty {q['scan_duty_pct']}%, CPU {q['cpu_pct']}%")
    print("=" * 66)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_json = None
    if "--json" in sys.argv:
        i = sys.argv.index("--json")
        out_json = sys.argv[i + 1] if i + 1 < len(sys.argv) else "bench.json"
        args = [a for a in args if a != out_json]
    if not args:
        logs = sorted(glob.glob(os.path.join(ROOT, "*.log")) +
                      glob.glob(os.path.join(ROOT, "logs", "*.log")),
                      key=os.path.getmtime)
        if not logs:
            sys.exit("no .log files found - run the monitor with -f log2file")
        args = [logs[-1]]
    s = summarise(args)
    show(s)
    if out_json:
        json.dump(s, open(out_json, "w"), indent=2)
        print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
