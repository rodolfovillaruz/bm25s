#!/usr/bin/env python
"""
CLI tool to index JSON chat files in the current directory using BM25S,
and search them with a query.

Usage:
    ./main.py "your search query"       # Index (if needed) and search
    ./main.py --index-only              # Only build/rebuild the index
    ./main.py --reindex "your query"    # Force reindex, then search
"""

import json
import os
import sys
from collections import deque

import bm25s

INDEX_DIR = ".bm25"
CORPUS_JSONL = os.path.join(INDEX_DIR, "corpus.jsonl")


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


def load_chat_json(filepath):
    """
    Load a valid chat JSON file and return (data, text_representation).
    Returns None if invalid.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not validate_chat_format(data):
            return None
        return data
    except (json.JSONDecodeError, UnicodeDecodeError, OSError, PermissionError):
        return None


def chat_to_text(chat_data):
    """
    Convert a chat data list into a single searchable text string.
    """
    parts = []
    for msg in chat_data:
        parts.append(f"{msg['role']}: {msg['content']}")
    return "\n".join(parts)


def discover_chat_files(root_path):
    """
    BFS traversal to find all valid chat JSON files.
    Returns list of (filepath, text_representation).
    """
    root_path = os.path.abspath(root_path)

    if not os.path.isdir(root_path):
        print(f"Error: '{root_path}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    # --- Phase 1: BFS to collect all files ---
    print(f"[Phase 1] Scanning '{root_path}' for files...", file=sys.stderr)

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

            # Skip the index directory itself
            if os.path.isdir(full_path) and os.path.abspath(full_path) == os.path.abspath(
                os.path.join(root_path, INDEX_DIR)
            ):
                continue

            if os.path.isdir(full_path):
                queue.append(full_path)
            elif os.path.isfile(full_path):
                all_files.append(full_path)

    total_files = len(all_files)
    print(
        f"[Phase 1] Complete. Found {total_files} files in {total_dirs} directories.",
        file=sys.stderr,
    )

    if total_files == 0:
        return []

    # --- Phase 2: Process JSON files ---
    print(f"[Phase 2] Checking {total_files} files for chat JSON format...", file=sys.stderr)

    documents = []
    processed = 0
    json_checked = 0
    matched = 0
    last_percent = -1

    for filepath in all_files:
        processed += 1

        # Progress reporting
        percent = (processed * 100) // total_files
        if percent != last_percent:
            last_percent = percent
            bar_width = 40
            filled = (percent * bar_width) // 100
            bar = "█" * filled + "░" * (bar_width - filled)
            print(
                f"\r  [{bar}] {percent:3d}% ({processed}/{total_files}) "
                f"| Matched: {matched} | JSON checked: {json_checked}",
                end="",
                file=sys.stderr,
            )

        if not filepath.lower().endswith(".json"):
            continue

        json_checked += 1
        chat_data = load_chat_json(filepath)

        if chat_data is not None:
            matched += 1
            text = chat_to_text(chat_data)
            documents.append({"path": filepath, "text": text})

    # Final progress
    bar = "█" * 40
    print(
        f"\r  [{bar}] 100% ({processed}/{total_files}) "
        f"| Matched: {matched} | JSON checked: {json_checked}",
        file=sys.stderr,
    )
    print(file=sys.stderr)

    print(f"[Phase 2] Found {matched} valid chat JSON files.", file=sys.stderr)
    return documents


def build_index(root_path="."):
    """
    Discover chat JSON files, index them with BM25S, and save to .bm25 folder.
    """
    documents = discover_chat_files(root_path)

    if not documents:
        print("[Index] No valid chat JSON files found. Nothing to index.", file=sys.stderr)
        return None, None

    corpus_texts = [doc["text"] for doc in documents]
    corpus_paths = [doc["path"] for doc in documents]

    print(f"[Index] Tokenizing {len(corpus_texts)} documents...", file=sys.stderr)
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")

    print(f"[Index] Building BM25 index...", file=sys.stderr)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    # Save index
    os.makedirs(INDEX_DIR, exist_ok=True)
    print(f"[Index] Saving index to '{INDEX_DIR}'...", file=sys.stderr)
    retriever.save(INDEX_DIR, corpus=corpus_texts)

    # Save file paths as corpus metadata (one JSON line per doc)
    with open(CORPUS_JSONL, "w", encoding="utf-8") as f:
        for doc in documents:
            json.dump({"path": doc["path"]}, f, ensure_ascii=False)
            f.write("\n")

    print(f"[Index] Done. Indexed {len(documents)} documents.", file=sys.stderr)
    return retriever, documents


def load_index():
    """
    Load a previously saved BM25 index and corpus metadata.
    Returns (retriever, metadata_list) or (None, None) if not found.
    """
    if not os.path.isdir(INDEX_DIR):
        return None, None

    try:
        print(f"[Load] Loading index from '{INDEX_DIR}'...", file=sys.stderr)
        retriever = bm25s.BM25.load(INDEX_DIR, load_corpus=True)

        metadata = []
        if os.path.exists(CORPUS_JSONL):
            with open(CORPUS_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        metadata.append(json.loads(line))

        print(f"[Load] Index loaded. {len(metadata)} documents.", file=sys.stderr)
        return retriever, metadata
    except Exception as e:
        print(f"[Load] Failed to load index: {e}", file=sys.stderr)
        return None, None


def search(query, k=10, reindex=False):
    """
    Search the BM25 index. Build it first if it doesn't exist or reindex is requested.
    """
    retriever = None
    metadata = None

    if reindex:
        print("[Search] Reindexing requested.", file=sys.stderr)
        retriever, documents = build_index(".")
        if documents:
            metadata = [{"path": doc["path"]} for doc in documents]
    else:
        retriever, metadata = load_index()

    if retriever is None:
        print("[Search] No index found. Building index...", file=sys.stderr)
        retriever, documents = build_index(".")
        if documents:
            metadata = [{"path": doc["path"]} for doc in documents]

    if retriever is None:
        print("[Search] No documents indexed. Nothing to search.", file=sys.stderr)
        sys.exit(1)

    print(f'[Search] Querying: "{query}"', file=sys.stderr)
    query_tokens = bm25s.tokenize(query, stopwords="en")

    n_docs = retriever.corpus.shape[0] if hasattr(retriever.corpus, "shape") else len(retriever.corpus)
    effective_k = min(k, n_docs)

    results, scores = retriever.retrieve(query_tokens, k=effective_k)

    print(f"[Search] Top {effective_k} results:\n", file=sys.stderr)

    for i in range(results.shape[1]):
        score = scores[0, i]
        doc_text = results[0, i]

        # Try to find the file path from metadata
        # We need to match the doc text back to the metadata index
        filepath = "unknown"
        if metadata:
            # Find which corpus entry this is by matching text
            if hasattr(retriever, "corpus") and retriever.corpus is not None:
                corpus_list = retriever.corpus
                if hasattr(corpus_list, "tolist"):
                    corpus_list = corpus_list.tolist()
                for idx, corpus_doc in enumerate(corpus_list):
                    if corpus_doc == doc_text and idx < len(metadata):
                        filepath = metadata[idx].get("path", "unknown")
                        break

        # Print result to stdout
        print(f"--- Result {i + 1} (score: {score:.4f}) ---")
        print(f"File: {filepath}")

        # Show a preview of the content (first 300 chars)
        preview = str(doc_text)
        if len(preview) > 300:
            preview = preview[:300] + "..."
        print(f"Preview:\n{preview}")
        print()


def print_usage():
    prog = sys.argv[0]
    print(f"Usage:", file=sys.stderr)
    print(f"  {prog} \"your search query\"         Search (auto-index if needed)", file=sys.stderr)
    print(f"  {prog} --index-only                 Build/rebuild the index only", file=sys.stderr)
    print(f"  {prog} --reindex \"your query\"       Force reindex, then search", file=sys.stderr)
    print(f"  {prog} -k 20 \"your query\"           Return top 20 results", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"Searches current directory recursively for JSON chat files.", file=sys.stderr)
    print(f"Index is stored in '{INDEX_DIR}/' folder.", file=sys.stderr)


def main():
    args = sys.argv[1:]

    if not args:
        print_usage()
        sys.exit(1)

    index_only = False
    reindex = False
    k = 10
    query_parts = []

    i = 0
    while i < len(args):
        arg = args[i]

        if arg == "--index-only":
            index_only = True
        elif arg == "--reindex":
            reindex = True
        elif arg == "-k" and i + 1 < len(args):
            i += 1
            try:
                k = int(args[i])
            except ValueError:
                print(f"Error: -k requires an integer argument.", file=sys.stderr)
                sys.exit(1)
        elif arg in ("-h", "--help"):
            print_usage()
            sys.exit(0)
        elif arg.startswith("-"):
            print(f"Error: Unknown option '{arg}'", file=sys.stderr)
            print_usage()
            sys.exit(1)
        else:
            query_parts.append(arg)

        i += 1

    if index_only:
        build_index(".")
        return

    query = " ".join(query_parts)
    if not query.strip():
        print("Error: No query provided.", file=sys.stderr)
        print_usage()
        sys.exit(1)

    search(query, k=k, reindex=reindex)


if __name__ == "__main__":
    main()
