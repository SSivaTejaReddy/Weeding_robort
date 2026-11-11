"""Normalize Roboflow-style filenames by stripping long hash suffixes.

Example: onion_mov-0089_jpg.rf.be72ff15e85b3dbb3d637ab6582a9b29.jpg -> onion_mov-0089.jpg

The script creates a backup copy of each file it will rename under
`backups/normalize_<timestamp>/...` so changes can be restored.

Usage:
  python -m scripts.normalize_filenames --paths datasets/crop_non_crop/current_dataset datasets/crop_non_crop/crops_data datasets/crop_non_crop/images
  python -m scripts.normalize_filenames --dry-run --paths ...
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from shutil import copy2
from datetime import datetime
import json
import sys


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_new_stem(stem: str) -> str | None:
    # If stem contains the Roboflow `.rf.` pattern, strip after it, then drop trailing _jpg/_png markers
    if ".rf." in stem:
        base = stem.split(".rf.")[0]
        base = re.sub(r"_(?:jpg|jpeg|png|bmp|webp)$", "", base, flags=re.I)
        return base
    # Also handle cases with multiple dots where last dot preceded by hash-like suffix
    m = re.match(r"(?P<base>.+?)(?:_[a-z0-9]{8,}|\.[a-f0-9]{8,})$", stem, flags=re.I)
    if m:
        return m.group("base")
    return None


def plan_renames(root: Path) -> list[tuple[Path, Path]]:
    actions = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        stem = p.stem
        new_stem = find_new_stem(stem)
        if not new_stem:
            continue
        new_name = new_stem + ext
        target = p.with_name(new_name)
        if target.resolve() == p.resolve():
            continue
        # avoid collisions by adding numeric suffix
        idx = 1
        while target.exists() and target.resolve() != p.resolve():
            target = p.with_name(f"{new_stem}_{idx}{ext}")
            idx += 1
        actions.append((p, target))
    return actions


def backup_and_apply(actions: list[tuple[Path, Path]], backup_root: Path) -> None:
    mapping = []
    for src, dst in actions:
        rel = src.resolve().relative_to(Path.cwd().resolve())
        backup_path = backup_root / rel
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        copy2(src, backup_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        mapping.append({"src": str(src), "dst": str(dst), "backup": str(backup_path)})
    # write mapping
    with (backup_root / "mapping.json").open("w", encoding="utf-8") as fh:
        json.dump({"generated_at": datetime.utcnow().isoformat(), "mapping": mapping}, fh, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", nargs="+", required=True, help="Root paths to process")
    p.add_argument("--dry-run", action="store_true", help="Show planned renames but do not apply")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    roots = [Path(p) for p in args.paths]
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_root = Path("backups") / f"normalize_{timestamp}"
    all_actions: list[tuple[Path, Path]] = []
    for r in roots:
        if not r.exists():
            print(f"Path not found, skipping: {r}")
            continue
        actions = plan_renames(r)
        print(f"Planned {len(actions)} renames under {r}")
        all_actions.extend(actions)

    if not all_actions:
        print("No files to rename.")
        return 0

    for src, dst in all_actions:
        print(f"{src} -> {dst}")

    if args.dry_run:
        print("Dry run: no changes made. To apply, re-run without --dry-run.")
        return 0

    backup_root.mkdir(parents=True, exist_ok=True)
    backup_and_apply(all_actions, backup_root)
    print(f"Applied {len(all_actions)} renames. Backups stored under {backup_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
