#!/usr/bin/env python3
"""Judge watcher: polls classify.py and has Haiku audit it against the Tier A rules."""
import argparse, asyncio, hashlib, json, re, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROMPT = (HERE / "prompt.txt").read_text()


def parse_verdict(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    cands = [t]
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        cands.append(m.group(0))
    for c in cands:
        try:
            d = json.loads(c)
            if isinstance(d, dict) and "verdict" in d:
                return d
        except Exception:
            pass
    return None


async def _sdk_call(code):
    from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage, TextBlock
    opts = ClaudeAgentOptions(model="haiku", system_prompt=PROMPT, allowed_tools=[],
                              tools=[], max_turns=1, setting_sources=[])
    parts, result = [], None
    async for msg in query(prompt=code, options=opts):
        if isinstance(msg, AssistantMessage):
            for b in msg.content:
                if isinstance(b, TextBlock):
                    parts.append(b.text)
        elif isinstance(msg, ResultMessage):
            result = msg
            if msg.is_error:
                raise RuntimeError(f"sdk error: {msg.result}")
    text = "".join(parts) or (result.result if result and result.result else "")
    if not text:
        raise RuntimeError("empty SDK response")
    return text


def cli_call(code):
    p = subprocess.run(["claude", "-p", "--model", "haiku", "--system-prompt", PROMPT,
                        "--output-format", "json", "--tools", "", code],
                       capture_output=True, text=True, timeout=180)
    if p.returncode != 0:
        raise RuntimeError(f"cli rc={p.returncode}: {p.stderr[:300]}")
    return json.loads(p.stdout).get("result", "")


def judge(code):
    t0 = time.time()
    try:
        text, backend = asyncio.run(_sdk_call(code)), "sdk"
    except Exception as e:
        print(f"[judge] sdk failed ({e!r}); falling back to cli", file=sys.stderr)
        text, backend = cli_call(code), "cli"
    lat = round(time.time() - t0, 2)
    d = parse_verdict(text)
    if d is None:
        d = {"verdict": "parse_error", "evidence": [text[:2000]], "notes": "could not parse judge reply"}
    return d, lat, backend


def run_once(path, log):
    raw = path.read_bytes()
    code = raw.decode("utf-8")  # may raise UnicodeDecodeError (mid-write)
    sha = hashlib.sha256(raw).hexdigest()
    d, lat, backend = judge(code)
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sha256": sha,
           "bytes": len(raw), "verdict": d.get("verdict"), "evidence": d.get("evidence", []),
           "notes": d.get("notes", ""), "latency_s": lat, "backend": backend}
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    flag = "!!! " if rec["verdict"] in ("suspicious", "violation") else ""
    print(f"{flag}[judge] {datetime.now().strftime('%H:%M:%S')} {rec['verdict']} ({rec['bytes']} B) {rec['notes']}", flush=True)
    return sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "workspace" / "classify.py"))
    ap.add_argument("--log", default=str(ROOT / "out" / "judge.jsonl"))
    ap.add_argument("--interval", type=float, default=10)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    path, log = Path(a.file), Path(a.log)

    if a.once:
        run_once(path, log)
        return
    last = None
    while True:
        try:
            if path.exists():
                raw = path.read_bytes()
                sha = hashlib.sha256(raw).hexdigest()
                if sha != last:
                    try:
                        raw.decode("utf-8")
                    except UnicodeDecodeError:
                        time.sleep(1)
                        continue
                    last = run_once(path, log)
        except Exception as e:
            print(f"[judge] error: {e!r}", file=sys.stderr, flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
