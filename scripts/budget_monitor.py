#!/usr/bin/env python3
"""Watch Codex rate limits (from the newest session rollout file) and optionally stop codex.

Codex records `rate_limits` (primary = 5-hour window, secondary = weekly) inside token_count
events of ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl. Each tick we take the newest rollout
file, read its tail, use the LAST rate_limits found, print a one-line reading and append a JSON
line to the log. With --kill, crossing --stop-at sends SIGINT to every `codex exec` process, writes the
--marker file (if given), keeps monitoring (no further kills for KILL_COOLDOWN seconds) and sends SIGTERM to
codex processes still alive TERM_AFTER seconds after the SIGINT.
"""
import argparse, glob, json, os, signal, subprocess, sys, time

TAIL_BYTES = 4 * 1024 * 1024
TAIL_LINES = 200
KILL_COOLDOWN = 120  # no further SIGINT within this many seconds of the last one
TERM_AFTER = 90      # SIGTERM codex processes still alive this long after the SIGINT


def find_key(obj, key):
    """Recursively find the first dict value stored under `key` in obj."""
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            v = o.get(key)
            if isinstance(v, dict):
                return v
            stack.extend(o.values())
        elif isinstance(o, list):
            stack.extend(o)
    return None


def newest_rollout():
    best, best_m = None, -1.0
    for p in glob.glob(os.path.expanduser("~/.codex/sessions/*/*/*/rollout-*.jsonl")):
        try:
            m = os.stat(p).st_mtime
        except OSError:
            continue
        if m > best_m:
            best, best_m = p, m
    return best


def read_tail_lines(path):
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - TAIL_BYTES))
        data = f.read()
    return data.split(b"\n")[-TAIL_LINES:]


def last_rate_limits(path):
    for raw in reversed(read_tail_lines(path)):
        if b"rate_limits" not in raw:
            continue
        try:  # partial / mid-write lines fail to parse: skip them
            ev = json.loads(raw)
        except ValueError:
            continue
        rl = find_key(ev, "rate_limits")
        if rl and isinstance(rl.get("primary"), dict):
            return rl
    return None


def window(rl, name):
    w = rl.get(name)
    if not isinstance(w, dict) or not isinstance(w.get("used_percent"), (int, float)):
        return None
    return w


def read_budget():
    """Return a reading dict, or None if there is no rollout file at all."""
    path = newest_rollout()
    if not path:
        return None
    rl = last_rate_limits(path)
    if not rl:
        return {"ts": time.time(), "file": os.path.basename(path), "available": False}
    now = time.time()
    r = {"ts": now, "file": os.path.basename(path), "available": True,
         "plan_type": rl.get("plan_type")}
    for name in ("primary", "secondary"):
        w = window(rl, name)
        if w:
            r[name] = {"used_percent": w["used_percent"], "window_minutes": w.get("window_minutes"),
                       "resets_at": w.get("resets_at"),
                       "stale": bool(w.get("resets_at")) and w["resets_at"] < now}
    if "primary" not in r:
        r["available"] = False
    return r


def fmt_reset(resets_at, now, long=False):
    if not resets_at:
        return "reset unknown"
    s = resets_at - now
    if s < 0:
        return "window already reset; stale"
    m = round(s / 60)
    return "resets in %dd %dh" % (m // 1440, (m % 1440) // 60) if long else "resets in %dm" % m


def format_line(r):
    now = r.get("ts", time.time())
    stamp = time.strftime("%H:%M:%S", time.localtime(now))
    if not r.get("available"):
        return "[budget] %s no rate_limits reading available | file: %s" % (stamp, r.get("file", "none"))
    p, s = r["primary"], r.get("secondary")
    weekly = "n/a"
    if s:
        weekly = "%g%% (%s)" % (s["used_percent"], fmt_reset(s.get("resets_at"), now, long=True))
    return "[budget] %s 5h: %g%% used (%s) | weekly: %s | file: %s" % (
        stamp, p["used_percent"], fmt_reset(p.get("resets_at"), now), weekly, r["file"])


def codex_pids():
    try:
        out = subprocess.run(["pgrep", "-f", "codex exec"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    me = {os.getpid(), os.getppid()}
    return [int(x) for x in out.split() if x.isdigit() and int(x) not in me]


def send_all(sig, pids):
    sent = []
    for pid in pids:
        try:
            os.kill(pid, sig)
            sent.append(pid)
        except OSError:
            pass
    return sent


def kill_codex(reading, stop_at, marker=None):
    sent = send_all(signal.SIGINT, codex_pids())
    print("!!! budget stop: primary %g%% >= %g%%, sent SIGINT to pids %s" % (
        reading["primary"]["used_percent"], stop_at, ",".join(map(str, sent)) or "(none found)"), flush=True)
    if marker:  # tells run_driver.sh unambiguously that the monitor stopped this run
        try:
            os.makedirs(os.path.dirname(os.path.abspath(marker)), exist_ok=True)
            with open(marker, "w") as f:
                json.dump({"ts": time.time(), "used_percent": reading["primary"]["used_percent"],
                           "stop_at": stop_at, "sigint_pids": sent}, f)
                f.write("\n")
        except OSError as e:
            print("[budget] marker write failed: %s" % e, flush=True)


def tick(args, state):
    r = read_budget()
    if r is None:
        print("[budget] %s no codex rollout file found" % time.strftime("%H:%M:%S"), flush=True)
        r = {"ts": time.time(), "available": False, "file": None}
    else:
        print(format_line(r), flush=True)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
        with open(args.log, "a") as f:
            f.write(json.dumps(r) + "\n")
    except OSError as e:
        print("[budget] log write failed: %s" % e, flush=True)
    if (args.kill and r.get("available") and not r["primary"].get("stale")
            and r["primary"]["used_percent"] >= args.stop_at
            and time.time() - state["last_kill"] >= KILL_COOLDOWN):
        state["last_kill"] = time.time()
        state["term_pending"] = True
        kill_codex(r, args.stop_at, args.marker)
    # escalate: codex still alive TERM_AFTER s after the SIGINT -> SIGTERM (once per kill)
    if state.get("term_pending") and time.time() - state["last_kill"] >= TERM_AFTER:
        state["term_pending"] = False
        pids = codex_pids()
        if pids:
            sent = send_all(signal.SIGTERM, pids)
            print("!!! budget stop: codex still alive %ds after SIGINT, sent SIGTERM to pids %s" % (
                TERM_AFTER, ",".join(map(str, sent)) or "(none)"), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--interval", type=float, default=15, help="seconds between readings (default 15)")
    ap.add_argument("--stop-at", type=float, default=90, help="percent of the 5h window at which --kill fires (default 90)")
    ap.add_argument("--log", default="out/budget.jsonl", help="JSONL log (default out/budget.jsonl)")
    ap.add_argument("--marker", help="file to create when --kill stops a run (read by run_driver.sh)")
    ap.add_argument("--once", action="store_true", help="print the current reading and exit")
    ap.add_argument("--kill", action="store_true", help="SIGINT every `codex exec` process when --stop-at is crossed")
    args = ap.parse_args()

    if args.once:
        try:
            r = read_budget()
            print(format_line(r) if r else "[budget] no codex rollout file found")
        except Exception as e:
            print("[budget] reading failed: %r" % (e,))
        return 0

    state = {"last_kill": 0.0, "term_pending": False}
    while True:
        try:
            tick(args, state)
        except Exception as e:  # never crash on a tick
            print("[budget] tick error: %r" % (e,), flush=True)
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
