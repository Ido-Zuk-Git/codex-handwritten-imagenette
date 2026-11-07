#!/usr/bin/env python3
"""Viewer daemon: serves workspace/.looker/req/*.json via a vision LLM backend.
Views are unlimited by default (--cap / --val-cap 0 = no limit). Every val image that is successfully
described is moved to data/train/<class>/ before the reply is written ("look and it moves")."""
import argparse, json, os, re, shutil, subprocess, sys, time, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
OPENCODE = os.environ.get("OPENCODE_BIN") or shutil.which("opencode") or "opencode"
CLAUDE = os.environ.get("CLAUDE_BIN") or shutil.which("claude") or "claude"
DEFAULT_Q = "Describe this image using the structured format."
# phrases meaning the model did not get the image (a plain "attached" in a valid description must not match)
BAD = re.compile(r"don't see an image|do not see an image|no image (was )?(attached|provided)|"
                 r"cannot see (the|an|any) image|unable to (view|see) the image|re-?upload|please attach", re.I)
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def clean_primary(out):
    out = ANSI.sub("", out)
    lines = out.splitlines()
    while lines and (not lines[0].strip() or lines[0].lstrip().startswith(">")):
        lines.pop(0)  # header like "> build · model"
    return "\n".join(lines).strip()


def run(cmd, timeout):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       stdin=subprocess.DEVNULL, cwd="/tmp")
    if p.returncode != 0:
        raise RuntimeError("exit %d: %s" % (p.returncode, ANSI.sub("", p.stderr or p.stdout).strip()[-200:]))
    return p.stdout


def check(text):
    if not text:
        raise RuntimeError("empty output")
    if BAD.search(text):
        raise RuntimeError("model could not see image: " + text[:100])
    return text


def call_primary(a, system, img, q):
    prompt = system + "\n\n" + q
    out = run([OPENCODE, "run", "-m", a.primary, "-f", img, prompt], a.primary_timeout)
    return check(clean_primary(out))


def call_fallback(a, system, img, q):
    user = "Use the Read tool to view the image file at %s, then answer.\n\n%s" % (img, q)
    # prompt first: --allowedTools is variadic and would swallow a trailing prompt
    out = run([CLAUDE, "-p", user, "--model", a.fallback_model, "--system-prompt", system,
               "--allowedTools", "Read"], a.fallback_timeout)
    return check(ANSI.sub("", out).strip())


def val_cap_now(a):
    """Val view budget; 0 = unlimited. out/looker_val_cap.json {"val_cap": N} overrides --val-cap when it
    holds a non-zero integer (absent file, null or 0 are ignored)."""
    try:
        v = int(json.load(open(a.val_cap_path))["val_cap"])
        return v if v else a.val_cap
    except Exception:
        return a.val_cap


def move_to_train(a, val_rel):
    """Move data/val/<class>/<name> to data/train/<class>/<name>. Return (new workspace-relative path or None,
    conflict message or None). If the target exists the val file stays in place (conflict)."""
    ws = a.workspace
    parts = val_rel.split(os.sep)  # data / val / <class> / ... / <name>
    src = os.path.join(ws, val_rel)
    dst_rel = os.path.join("data", "train", *parts[2:])
    dst = os.path.join(ws, dst_rel)
    if os.path.exists(dst):
        return None, "target exists, val file kept: %s" % dst_rel
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.replace(src, dst)
    return dst_rel, None


def load_viewed(path):
    try:
        v = json.load(open(path))
        return [str(x) for x in v] if isinstance(v, list) else []
    except Exception:
        return []


def classify_path(a, img):
    """Return (kind, workspace-relative path) with kind in 'train', 'val', None. img is a realpath."""
    ws = os.path.realpath(a.workspace)
    for kind in ("train", "val"):
        if img.startswith(os.path.join(ws, "data", kind) + os.sep) and os.path.isfile(img):
            return kind, os.path.relpath(img, ws)
    return None, None


def handle(a, req, state, system):
    """Return (text, backend, attempts, err, t0, is_val, val_rel, primary_err). Counting is done by the caller.
    primary_err is the last primary-backend error (first 200 chars) when the fallback was used, else None."""
    t0 = time.time()
    img = os.path.realpath(str(req.get("image", "")))
    kind, rel = classify_path(a, img)
    is_val = kind == "val"
    if kind is None:
        return ("ERROR: path not allowed (only data/train and data/val images can be viewed)", None, 0,
                "path not allowed", t0, False, None, None)
    if a.cap and state["views_used"] >= a.cap:
        e = "view budget exhausted (%d views)" % a.cap
        return ("ERROR: " + e, None, 0, e, t0, is_val, rel, None)
    if (is_val and val_cap_now(a) and rel not in state["viewed_val"]
            and state["val_views_used"] >= val_cap_now(a)):
        e = "val view budget exhausted (%d val views)" % val_cap_now(a)
        return ("ERROR: " + e, None, 0, e, t0, True, rel, None)
    q = req.get("question") or DEFAULT_Q
    attempts, errs = 0, []
    for _ in range(3):  # 1 try + 2 retries
        attempts += 1
        try:
            return (call_primary(a, system, img, q), "deepseek", attempts, None, t0, is_val, rel, None)
        except Exception as e:
            errs.append("primary: %s" % e)
    primary_err = errs[-1][:200]  # shown in looker.jsonl so an exhausted free tier is visible
    attempts += 1
    try:
        return (call_fallback(a, system, img, q), "haiku", attempts, None, t0, is_val, rel, primary_err)
    except Exception as e:
        errs.append("fallback: %s" % e)
    reason = errs[-2][-150:] + " | " + errs[-1][-150:] if len(errs) > 1 else errs[-1]
    return ("ERROR: viewer failed: " + reason, None, attempts, "viewer failed: " + reason, t0, is_val, rel, primary_err)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=os.path.join(ROOT, "workspace"))
    ap.add_argument("--log", default=os.path.join(ROOT, "out", "looker.jsonl"))
    ap.add_argument("--cap", type=int, default=0, help="max total views; 0 = unlimited (default)")
    ap.add_argument("--val-cap", type=int, default=0,
                    help="max distinct val images viewed; 0 = unlimited (default); a non-zero "
                         "out/looker_val_cap.json {\"val_cap\": N} overrides it")
    ap.add_argument("--primary", default="opencode/deepseek-v4.1-flash-free")
    ap.add_argument("--fallback-model", default="haiku")
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--primary-timeout", type=float, default=60)
    ap.add_argument("--fallback-timeout", type=float, default=90)
    ap.add_argument("--reset-cap", action="store_true")
    a = ap.parse_args()
    a.workspace = os.path.abspath(a.workspace)
    a.log = os.path.abspath(a.log)
    reqd = os.path.join(a.workspace, ".looker", "req")
    resd = os.path.join(a.workspace, ".looker", "res")
    for d in (reqd, resd, os.path.dirname(a.log)):
        os.makedirs(d, exist_ok=True)
    outd = os.path.dirname(a.log)
    state_path = os.path.join(outd, "looker_state.json")
    viewed_path = os.path.join(outd, "looker_viewed_val.json")
    a.val_cap_path = os.path.join(outd, "looker_val_cap.json")
    state = {"views_used": 0, "val_views_used": 0, "viewed_val": []}
    if a.reset_cap:
        write_json(viewed_path, [])
    elif os.path.exists(state_path):
        try:
            st = json.load(open(state_path))
            state["views_used"] = int(st["views_used"])
            state["val_views_used"] = int(st.get("val_views_used", 0))
        except Exception:
            pass
    state["viewed_val"] = load_viewed(viewed_path)
    if not os.path.exists(viewed_path):
        write_json(viewed_path, state["viewed_val"])
    save_state = lambda: write_json(state_path, {"views_used": state["views_used"],
                                                 "val_views_used": state["val_views_used"]})
    save_state()
    system = open(os.path.join(HERE, "prompt.txt")).read().strip()
    print("[looker] started pid=%d workspace=%s cap=%s used=%d val_cap=%s val_used=%d primary=%s (0 = unlimited; "
          "viewed val images are moved to data/train)"
          % (os.getpid(), a.workspace, a.cap or "unlimited", state["views_used"], val_cap_now(a) or "unlimited",
             state["val_views_used"], a.primary), flush=True)

    print("[looker] primary model: %s (fallback: claude %s). Reminder: the primary is a free tier; if "
          "looker.jsonl shows backend=haiku with a primary_error, it is exhausted or failing." % (a.primary, a.fallback_model),
          flush=True)

    while True:
        try:
            files = sorted((os.path.join(reqd, f) for f in os.listdir(reqd) if f.endswith(".json")),
                           key=lambda p: os.path.getmtime(p))
        except Exception:
            files = []
        if not files:
            time.sleep(a.poll)
            continue
        fp = files[0]
        rid = os.path.basename(fp)[:-5]
        req = {}
        try:
            try:
                req = json.load(open(fp))
            except ValueError:
                time.sleep(0.2); req = json.load(open(fp))
            try:  # pick up an external reset (run_driver.sh --reset-views) of the state file
                st = json.load(open(state_path))
                state["views_used"] = int(st["views_used"])
                state["val_views_used"] = int(st.get("val_views_used", 0))
            except Exception:
                pass
            state["viewed_val"] = load_viewed(viewed_path)
            text, backend, attempts, err, t0, is_val, val_rel, primary_err = handle(a, req, state, system)
        except Exception as e:
            traceback.print_exc()
            text, backend, attempts, err, t0 = "ERROR: viewer failed: internal %r" % e, None, 0, "internal %r" % e, time.time()
            is_val, val_rel, primary_err = False, None, None
        moved, conflict = None, None
        if backend and is_val and val_rel:  # successful val view: move the file to train before replying
            try:
                moved, conflict = move_to_train(a, val_rel)
                if moved:
                    text = text.rstrip("\n") + "\n[note: %s has been moved to %s/]" % (val_rel, os.path.dirname(moved))
                if conflict:
                    print("[looker] CONFLICT %s: %s" % (val_rel, conflict), flush=True)
            except Exception as e:
                conflict = "move failed: %r" % e
                traceback.print_exc()
        lat = round(time.time() - t0, 2)
        if backend:
            state["views_used"] += 1
            if is_val:
                if val_rel not in state["viewed_val"]:
                    state["viewed_val"].append(val_rel)
                    write_json(viewed_path, state["viewed_val"])
                    state["val_views_used"] += 1
            save_state()
        try:
            write_json(os.path.join(resd, rid + ".json"),
                       {"text": text, "backend": backend, "latency_s": lat, "error": err})
            os.remove(fp)
        except Exception:
            traceback.print_exc()
        try:
            img = str(req.get("image", ""))
            rel = os.path.relpath(img, a.workspace) if img else ""
            with open(a.log, "a") as f:
                f.write(json.dumps({"ts": time.time(), "id": rid, "image": rel, "question": req.get("question"),
                                    "backend": backend, "latency_s": lat, "attempts": attempts, "error": err,
                                    "views_used": state["views_used"], "is_val": bool(is_val),
                                    "val_views_used": state["val_views_used"],
                                    "text_len": len(text), "moved_to_train": moved,
                                    "move_conflict": conflict,
                                    "primary_error": primary_err}) + "\n")
            print("[looker] %s #%d #v%d %s%s %ss %s%s%s" % (time.strftime("%H:%M:%S"), state["views_used"],
                  state["val_views_used"], "val " if is_val else "", backend, lat, rel,
                  " -> " + moved if moved else "",
                  "  [primary failed: %s]" % primary_err[:80] if primary_err else ""),
                  flush=True)
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
