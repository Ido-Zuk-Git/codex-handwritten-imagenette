#!/usr/bin/env python3
"""Rebuild the agent-visible data from data/manifest.json and a local Imagenette2-160 download.

    python3 scripts/build_data.py --src /path/to/imagenette2-160 [--test]

Copies, per class, manifest full_split.train -> workspace/data/train/<class>/ and
full_split.val -> workspace/data/val/<class>/ (both from <src>/train/<wnid>/; 50 + 50 per class),
and with --test the 200 hidden-test files -> test/<class>/ (from <src>/val/<wnid>/).
Prints the file counts (expected 500 / 500 / 2000). Images are not shipped with the repo;
download https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz and extract it first."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def copy_set(names, src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in names:
        src = src_dir / name
        if not src.is_file():
            sys.exit("missing source image: %s" % src)
        shutil.copy2(src, dst_dir / name)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="extracted imagenette2-160 directory (contains train/ and val/)")
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.json"))
    ap.add_argument("--test", action="store_true", help="also build test/<class>/ (the 2,000-image hidden test set)")
    args = ap.parse_args()

    src = Path(args.src)
    if not (src / "train").is_dir() or not (src / "val").is_dir():
        sys.exit("%s does not look like imagenette2-160 (no train/ and val/)" % src)
    with open(args.manifest) as f:
        manifest = json.load(f)

    counts = {"train": 0, "val": 0, "test": 0}
    for cls, info in manifest.items():
        if not isinstance(info, dict) or "wnid" not in info:
            continue  # "Procedure" and other notes
        wnid = info["wnid"]
        counts["train"] += copy_set(info["full_split"]["train"], src / "train" / wnid, ROOT / "workspace" / "data" / "train" / cls)
        counts["val"] += copy_set(info["full_split"]["val"], src / "train" / wnid, ROOT / "workspace" / "data" / "val" / cls)
        if args.test:
            counts["test"] += copy_set(info["test"], src / "val" / wnid, ROOT / "test" / cls)
    print("workspace/data/train: %d files" % counts["train"])
    print("workspace/data/val:   %d files" % counts["val"])
    if args.test:
        print("test:                 %d files" % counts["test"])


if __name__ == "__main__":
    main()
