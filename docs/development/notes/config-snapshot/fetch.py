#!/usr/bin/env python3
"""Fetch config.json + directory listings from both HF repos."""
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent

REPOS = {
    "avatar-1.5": "meituan-longcat/LongCat-Video-Avatar-1.5",
    "longcat-video": "meituan-longcat/LongCat-Video",
}

# (repo_key, path_in_repo)
FILES = [
    ("avatar-1.5", "config.json"),
    ("avatar-1.5", "base_model/config.json"),
    ("avatar-1.5", "base_model_int8/config.json"),
    ("avatar-1.5", "scheduler/scheduler_config.json"),
    ("avatar-1.5", "whisper-large-v3/config.json"),
    ("longcat-video", "config.json"),
    ("longcat-video", "model_index.json"),
    ("longcat-video", "dit/config.json"),
    ("longcat-video", "text_encoder/config.json"),
    ("longcat-video", "vae/config.json"),
    ("longcat-video", "scheduler/scheduler_config.json"),
    ("longcat-video", "tokenizer/tokenizer_config.json"),
]

DIRS = [
    ("avatar-1.5", "base_model"),
    ("avatar-1.5", "base_model_int8"),
    ("avatar-1.5", "lora"),
    ("avatar-1.5", "whisper-large-v3"),
    ("avatar-1.5", "scheduler"),
    ("longcat-video", "dit"),
    ("longcat-video", "vae"),
    ("longcat-video", "text_encoder"),
]

def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()
    except Exception as e:
        return f"ERROR: {e}".encode()

print("="*70, "\nFETCHING CONFIG FILES\n", "="*70, sep="")
for repo_key, path in FILES:
    url = f"https://huggingface.co/{REPOS[repo_key]}/resolve/main/{path}"
    out = HERE / f"{repo_key}--{path.replace('/', '-')}"
    data = fetch(url)
    out.write_bytes(data)
    print(f"\n--- {repo_key}/{path}  ({len(data)} bytes) ---")
    try:
        print(json.dumps(json.loads(data), indent=2)[:2500])
    except Exception:
        print(data[:500].decode("utf-8", errors="replace"))

print("\n", "="*70, "\nFETCHING DIRECTORY LISTINGS\n", "="*70, sep="")
for repo_key, dir_path in DIRS:
    url = f"https://huggingface.co/api/models/{REPOS[repo_key]}/tree/main/{dir_path}"
    out = HERE / f"{repo_key}--{dir_path.replace('/', '-')}-listing.json"
    data = fetch(url)
    out.write_bytes(data)
    print(f"\n--- {repo_key}/{dir_path} ---")
    try:
        listing = json.loads(data)
        for f in listing:
            size = f.get("size", "-")
            print(f"  {f['type']:5s} {size!s:>14s}  {f['path']}")
    except Exception as e:
        print(f"  parse error: {e}, raw[:300]: {data[:300]}")
