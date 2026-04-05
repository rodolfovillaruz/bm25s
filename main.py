#!/usr/bin/env python
"""
CLI script to recursively search for JSON files matching a specific chat format.
Uses BFS for traversal and progress reporting.
"""

import json
import os
import sys
from collections import deque


def validate_chat_format(data):
    """
    Validate that the JSON data matches the expected chat format:
    A list of objects with 'role' and 'content' string fields,
    where role is either 'user' or 'assistant'.
    """
    if not isinstance(data, list):
        return False

    if len(data) == 0:
        return False

    valid_roles = {"user", "assistant"}

    for item in data:
        if not isinstance(item, dict):
            return False

        if set(item.keys()) != {"role", "content"}:
            return False

        if not isinstance(item.get("role"), str):
            return False

        if not isinstance(item.get("content"), str):
            return False

        if item["role"] not in valid_roles:
            return False

    return True


def is_valid_chat_json(filepath):
    """
    Check if a file is a valid JSON file matching the chat format.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return validate_chat_format(data)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError, PermissionError):
        return False


def bfs_search(root_path):
    """
    Perform BFS traversal of the directory tree.
    First pass: count total items for progress.
    Second pass: process files with progress reporting.
    """
    root_path = os.path.abspath(root_path)

    if not os.path.isdir(root_path):
        print(f"Error: '{root_path}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    # --- Phase 1: BFS to count total entries for progress ---
    print(f"[Phase 1] Counting entries in '{root_path}'...", file=sys.stderr)

    total_files = 0
    total_dirs = 0
    all_files = []
    queue = deque()
    queue.append(root_path)

    while queue:
        current_dir = queue.popleft()
        total_dirs += 1

        try:
            entries = sorted(os.listdir(current_dir))
        except PermissionError:
            print(f"  [WARN] Permission denied: {current_dir}", file=sys.stderr)
            continue
        except OSError as e:
            print(f"  [WARN] OS error accessing {current_dir}: {e}", file=sys.stderr)
            continue

        for entry in entries:
            full_path = os.path.join(current_dir, entry)

            if os.path.isdir(full_path):
                queue.append(full_path)
            elif os.path.isfile(full_path):
                total_files += 1
                all_files.append(full_path)

    total_items = total_files
    print(
        f"[Phase 1] Complete. Found {total_files} files in {total_dirs} directories.",
        file=sys.stderr,
    )

    if total_items == 0:
        print("[Phase 2] No files to process.", file=sys.stderr)
        return

    # --- Phase 2: BFS process files with progress ---
    print(f"[Phase 2] Processing {total_items} files...", file=sys.stderr)

    processed = 0
    matched = 0
    json_files_found = 0
    last_percent = -1

    for filepath in all_files:
        processed += 1

        # Progress reporting
        percent = (processed * 100) // total_items
        if percent != last_percent:
            last_percent = percent
            bar_width = 40
            filled = (percent * bar_width) // 100
            bar = "█" * filled + "░" * (bar_width - filled)
            print(
                f"\r  [{bar}] {percent:3d}% ({processed}/{total_items}) "
                f"| Matched: {matched} | JSON: {json_files_found}",
                end="",
                file=sys.stderr,
            )

        # Only consider files with .json extension
        if not filepath.lower().endswith(".json"):
            continue

        json_files_found += 1

        if is_valid_chat_json(filepath):
            matched += 1
            # Print matching file path to stdout
            print(filepath, file=sys.stdout, flush=True)

    # Final progress update
    bar = "█" * 40
    print(
        f"\r  [{bar}] 100% ({processed}/{total_items}) "
        f"| Matched: {matched} | JSON: {json_files_found}",
        file=sys.stderr,
    )
    print(file=sys.stderr)  # newline after progress bar

    # Summary
    print(f"[Done] Summary:", file=sys.stderr)
    print(f"  Total files scanned:    {total_items}", file=sys.stderr)
    print(f"  JSON files found:       {json_files_found}", file=sys.stderr)
    print(f"  Matching chat format:   {matched}", file=sys.stderr)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <directory>", file=sys.stderr)
        print(
            f"  Recursively searches <directory> for JSON files matching chat format.",
            file=sys.stderr,
        )
        print(f"  Matching file paths are printed to stdout.", file=sys.stderr)
        print(f"  Progress is reported on stderr.", file=sys.stderr)
        sys.exit(1)

    root_dir = sys.argv[1]
    bfs_search(root_dir)


if __name__ == "__main__":
    main()
