"""Rename the current extension project to a new project name."""

import os
import sys
from pathlib import Path


TEMPLATE_NAME = "blood_absorption"
EXCLUDE_DIRS = {".git", "__pycache__"}


def detect_current_name(root_dir: Path) -> str:
    """Detect the current extension/package name from the exts directory."""
    ext_dirs = [path for path in (root_dir / "exts").iterdir() if path.is_dir()]
    candidates = []
    for ext_dir in ext_dirs:
        package_dir = ext_dir / ext_dir.name
        if package_dir.is_dir() and (ext_dir / "setup.py").exists():
            candidates.append(ext_dir.name)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one extension under {root_dir / 'exts'}, found: {candidates}")
    return candidates[0]


def should_skip_dir(path: Path) -> bool:
    """Check if a directory should be excluded from traversal."""
    return any(part in EXCLUDE_DIRS for part in path.parts)


def replace_names(text: str, source_names: list[str], new_name: str) -> str:
    """Replace all source names with the new name."""
    for source_name in source_names:
        text = text.replace(source_name, new_name)
    return text


def rename_file_contents(root_dir: Path, source_names: list[str], new_name: str) -> int:
    """Rename all source names to the new keyword in text files under the root directory."""
    updated_files = 0
    for dirpath, dirnames, files in os.walk(root_dir):
        current_dir = Path(dirpath)
        dirnames[:] = [dirname for dirname in dirnames if not should_skip_dir(current_dir / dirname)]
        if should_skip_dir(current_dir):
            continue
        for file_name in files:
            file_path = current_dir / file_name
            try:
                file_contents = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            renamed_contents = replace_names(file_contents, source_names, new_name)
            if renamed_contents == file_contents:
                continue

            file_path.write_text(renamed_contents, encoding="utf-8")
            updated_files += 1
    return updated_files


def rename_paths(root_dir: Path, source_names: list[str], new_name: str) -> int:
    """Rename files and directories containing any of the source names."""
    renamed_paths = 0
    all_paths = sorted(root_dir.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in all_paths:
        if should_skip_dir(path):
            continue
        renamed_name = replace_names(path.name, source_names, new_name)
        if renamed_name == path.name:
            continue
        path.rename(path.with_name(renamed_name))
        renamed_paths += 1
    return renamed_paths


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_template.py <new_name>")
        sys.exit(1)

    root_dir = Path(__file__).resolve().parent.parent
    new_name = sys.argv[1]
    current_name = detect_current_name(root_dir)
    source_names = []
    for source_name in [current_name, TEMPLATE_NAME]:
        if source_name != new_name and source_name not in source_names:
            source_names.append(source_name)

    print(
        f"Warning, this script will rename all instances of {source_names or [current_name]} "
        f"to '{new_name}' in {root_dir}."
    )
    proceed = input("Proceed? (y/n): ")

    if proceed.lower() == "y":
        updated_files = rename_file_contents(root_dir, source_names, new_name)
        renamed_paths = rename_paths(root_dir / "exts", source_names, new_name)
        print(f"Updated {updated_files} text files and renamed {renamed_paths} paths.")
    else:
        print("Aborting.")
