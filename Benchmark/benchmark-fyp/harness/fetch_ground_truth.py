"""
Download DataSciBench's ground truth from Hugging Face into data/<task_id>/gt/.

DataSciBench ships prompts but not answers: every data/<task_id>/ folder in the
clone contains only prompt.json. experiments.evaluate therefore scores nothing
until the ground truth is fetched from huggingface.co/datasets/zd21/DataSciBench
and placed at data/<task_id>/gt/, as the repo README instructs.

That dataset is GATED. Before this script can work you must, once, with the
same account your HF_TOKEN belongs to:
    1. open https://huggingface.co/datasets/zd21/DataSciBench
    2. accept the "share your contact information" gate
    3. put the token in .env as HF_TOKEN

The archive layout is not documented upstream, so this script downloads first,
reports what it actually found, and only then places files -- rather than
assuming a structure and silently mis-filing everything.

    venvs/datascibench/Scripts/python.exe harness/fetch_ground_truth.py
    venvs/datascibench/Scripts/python.exe harness/fetch_ground_truth.py --inspect-only
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from harness.common import REPOS, load_env  # noqa: E402

REPO_ID = "zd21/DataSciBench"
DSB = REPOS / "DataSciBench"
CACHE = DSB / "_gt_download"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect-only", action="store_true",
                    help="download and report the layout without placing files")
    ap.add_argument("--force", action="store_true",
                    help="overwrite gt/ folders that already exist")
    args = ap.parse_args()

    cfg = load_env()
    token = cfg["hf_token"]
    if not token or token.startswith("your-"):
        raise SystemExit(
            "HF_TOKEN is not set in .env.\n"
            "This dataset is gated: accept the terms at\n"
            f"  https://huggingface.co/datasets/{REPO_ID}\n"
            "then put your token in .env as HF_TOKEN.")

    from huggingface_hub import snapshot_download

    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] snapshot_download({REPO_ID}) -> {CACHE}")
    try:
        local = snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                                  token=token, local_dir=str(CACHE))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Download failed: {type(exc).__name__}: {exc}\n\n"
            "If this is a 401/403, the gate has not been accepted for this "
            f"token's account. Open https://huggingface.co/datasets/{REPO_ID} "
            "and accept it, then retry.")

    local = pathlib.Path(local)

    # Unpack any archives so we can see the real structure.
    for zpath in list(local.rglob("*.zip")):
        target = zpath.with_suffix("")
        if target.exists() and not args.force:
            continue
        print(f"[fetch] extracting {zpath.name}")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(target)

    print("\n[fetch] top-level contents of the download:")
    for p in sorted(local.iterdir()):
        kind = "dir " if p.is_dir() else "file"
        print(f"    {kind} {p.name}")

    # Task ids we need ground truth for (non-BCB; BCB is scored separately by
    # experiments.evaluate_tmc against BigCodeBench functions).
    task_ids = sorted(d.name for d in (DSB / "data").iterdir()
                      if d.is_dir() and not d.name.startswith("bcb"))
    print(f"\n[fetch] looking for ground truth for {len(task_ids)} non-BCB tasks")

    # The archive has TWO trees for each task id and only one is the ground
    # truth:
    #   DataSciBench-data/<tid>/          -- prompts and input data
    #   DataSciBench_GroundTruth_Data/gt_data/<tid>/gt/   -- the real answers
    # experiments.evaluate looks for data/<tid>/gt/output.csv, so copying the
    # prompt tree produces a run that completes cleanly and scores 0.0 on
    # everything with "No such file or directory: .../gt/output.csv" buried in
    # the log. Prefer a path containing gt_data, and require a 'gt' directory.
    # __MACOSX is the zip's resource-fork junk and must never be selected.
    def rank(path: pathlib.Path) -> tuple:
        parts = set(path.parts)
        return (
            "__MACOSX" in parts,             # False sorts first
            "gt_data" not in parts,          # prefer the ground-truth tree
            not (path / "gt").is_dir(),      # prefer one that has gt/
            len(path.parts),
        )

    placed, not_found = [], []
    for tid in task_ids:
        candidates = sorted((p for p in local.rglob(tid) if p.is_dir()), key=rank)
        candidates = [p for p in candidates if "__MACOSX" not in p.parts]
        if not candidates:
            not_found.append(tid)
            continue
        src = candidates[0]
        inner = src / "gt"
        if inner.is_dir():
            src = inner
        dest = DSB / "data" / tid / "gt"
        if dest.exists() and not args.force:
            placed.append((tid, "already present"))
            continue
        if args.inspect_only:
            placed.append((tid, f"would copy {src}"))
            continue
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        placed.append((tid, f"copied {len(list(dest.iterdir()))} files"))

    print(f"\n[fetch] placed/found: {len(placed)}   missing: {len(not_found)}")
    for tid, note in placed[:10]:
        print(f"    {tid}: {note}")
    if len(placed) > 10:
        print(f"    ... and {len(placed) - 10} more")
    if not_found:
        print(f"\n[fetch] NO ground truth located for {len(not_found)} tasks:")
        print("    " + ", ".join(not_found[:15])
              + (" ..." if len(not_found) > 15 else ""))
        print("\n    The archive layout may differ from what this script "
              "expects. Inspect\n"
              f"    {CACHE}\n"
              "    and place each task's files at data/<task_id>/gt/ manually, "
              "as the\n    DataSciBench README describes.")

    if args.inspect_only:
        print("\n[fetch] --inspect-only: nothing was copied.")


if __name__ == "__main__":
    main()
