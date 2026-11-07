# Task

Write `classify.py`: a hand-written program with no statistical learning that defines
`predict(path: str) -> str`, returning one of the 10 Imagenette classes: tench, english_springer,
cassette_player, chain_saw, church, french_horn, garbage_truck, gas_pump, golf_ball, parachute.

Target to beat: 75.45%, the hidden-test accuracy of a small CNN trained from scratch on 1,000 images of these classes (the full train + val set). Chance is 10%. Your program must get there without learning.

# What you have

Your working directory is `workspace/`.

- `data/train/<class>/*.JPEG` -- {{N_TRAIN}} training images in total. JPEG, shortest side
  160 px, variable aspect ratio. Tune your rules on these.
- `data/val/<class>/*.JPEG` -- {{N_VAL}} validation images in total, same format. This is your only
  estimate of the hidden-test score. Run `python3 check.py classify.py --images data/val`. Any val image
  you view with look.py is moved to `data/train` at that moment. Looking at val images your program gets
  wrong is often the fastest way to find a missing rule; the cost is a slightly smaller val set. Tune on
  train; use val to measure. Report train and val numbers in NOTES.md for every iteration.
- `look.py` -- the ONLY way to see an image. You cannot view pixels directly; you can compute
  on them, but to know what an image looks like, use the viewer.
  - `python3 look.py <image> ["question"]` prints a text description from a separate vision
    model. Pass a question to focus it (e.g. "what colours dominate the background?").
  - `python3 look.py --batch <image1> <image2> ... ["question"]` describes several images in
    one call.
  - Only `data/train` and `data/val` images can be viewed.
- `check.py` -- self-check. `python3 check.py classify.py --images data/train` (or `--images data/val`)
  runs the same scorer that is used for final scoring, including the rule checks, and prints
  accuracy and any rule violations.

# Rules

See AGENTS.md (loaded automatically). It is binding. The one hard rule is no statistical learning;
every non-trivial line must be commented.

# Deliverables

- `classify.py` in the workspace root (this is what is scored).
- `NOTES.md` in the workspace root, with train and val accuracy for each iteration.
- As your last message, a 10-line summary of your final approach.
