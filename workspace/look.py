#!/usr/bin/env python3
"""Ask the viewer service to describe an image.
Usage: python3 look.py <image> ["question"]   |   python3 look.py --batch <image1> <image2> ... ["question"]"""
import json, os, sys, time, uuid

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".looker")
REQ, RES = os.path.join(D, "req"), os.path.join(D, "res")
TIMEOUT = 300


def look(path, question=None):
    os.makedirs(REQ, exist_ok=True); os.makedirs(RES, exist_ok=True)
    rid = uuid.uuid4().hex
    body = {"image": os.path.abspath(path), "question": question, "ts": time.time()}
    tmp = os.path.join(REQ, rid + ".tmp")
    with open(tmp, "w") as f:
        json.dump(body, f)
    os.rename(tmp, os.path.join(REQ, rid + ".json"))  # atomic publish
    res = os.path.join(RES, rid + ".json")
    end = time.time() + TIMEOUT
    while time.time() < end:
        if os.path.exists(res):
            try:
                with open(res) as f:
                    r = json.load(f)
            except ValueError:
                time.sleep(0.2); continue
            try: os.remove(res)
            except OSError: pass
            return r.get("text") or ("ERROR: " + str(r.get("error")))
        time.sleep(0.5)
    try: os.remove(os.path.join(REQ, rid + ".json"))
    except OSError: pass
    return None


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"):
        print(__doc__); sys.exit(2)
    if a[0] == "--batch":
        rest = a[1:]
        q = None
        if rest and not os.path.isfile(rest[-1]):
            q, rest = rest[-1], rest[:-1]
        paths = rest
        if not paths:
            print("ERROR: --batch needs at least one existing image path"); sys.exit(2)
    else:
        paths, q = [a[0]], (a[1] if len(a) > 1 else None)
    for p in paths:
        t = look(p, q)
        if t is None:
            print("ERROR: timed out after %ds waiting for the viewer; the viewer service may be down." % TIMEOUT)
            sys.exit(1)
        if a[0] == "--batch":
            print("### " + p)
        print(t)


if __name__ == "__main__":
    main()
