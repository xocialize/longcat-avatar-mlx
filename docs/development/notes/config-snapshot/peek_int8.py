#!/usr/bin/env python3
"""Peek at the safetensors header of base_model_int8 to determine quant format."""
import json
import struct
import urllib.request

URL = "https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5/resolve/main/base_model_int8/model.safetensors"

# Try a few candidate paths (single file vs. sharded)
CANDIDATES = [
    "base_model_int8/quantized_model-00001-of-00004.safetensors",
    "base_model_int8/quantized_model.safetensors.index.json",
]


def fetch_range(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read(), r.status
    except urllib.error.HTTPError as e:
        return None, e.code


def head(url):
    """HEAD request to learn the file size."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.headers.get("Content-Length"), r.status, r.headers.get("Location")
    except Exception as e:
        return None, str(e), None


print("=" * 70, "\nProbing base_model_int8 file layout\n", "=" * 70, sep="")
for path in CANDIDATES:
    url = f"https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5/resolve/main/{path}"
    size, status, _ = head(url)
    print(f"  {status!s:5s}  size={size}  {path}")

# Pull the index.json if it exists
for path in CANDIDATES:
    if "index.json" in path:
        url = f"https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5/resolve/main/{path}"
        data, status = fetch_range(url, 0, 65535)
        if data and status == 206:
            print(f"\n--- {path} (first 64 KB) ---")
            try:
                obj = json.loads(data.decode("utf-8"))
                # Just show first ~30 keys of the weight_map
                wmap = obj.get("weight_map", {})
                keys = list(wmap.keys())
                print(f"  total_size: {obj.get('metadata', {}).get('total_size')}")
                print(f"  weight_map entries: {len(wmap)}")
                for k in keys[:25]:
                    print(f"    {k} -> {wmap[k]}")
                if len(keys) > 25:
                    print(f"    ... {len(keys) - 25} more")
            except Exception as e:
                print(f"  parse error: {e}\n  raw[:1000]: {data[:1000]}")

# Pick a shard or single file and read its safetensors header
header_candidates = [
    "base_model_int8/quantized_model-00001-of-00004.safetensors",
]
for path in header_candidates:
    url = f"https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5/resolve/main/{path}"
    # First 8 bytes = uint64 LE header size
    data, status = fetch_range(url, 0, 7)
    if data and status == 206 and len(data) == 8:
        header_size = struct.unpack("<Q", data)[0]
        print(f"\n--- {path} header_size = {header_size} ---")
        if header_size < 10 * 1024 * 1024:  # sanity
            header_data, _ = fetch_range(url, 8, 8 + header_size - 1)
            if header_data:
                hdr = json.loads(header_data.decode("utf-8"))
                # Show a few representative tensors
                tensor_keys = [k for k in hdr.keys() if k != "__metadata__"]
                print(f"  total tensors: {len(tensor_keys)}")
                print(f"  __metadata__: {hdr.get('__metadata__')}")
                # Sample a few tensors
                for k in tensor_keys[:8]:
                    print(f"    {k}: {hdr[k]}")
                # Look for any quant-suspicious tensor names
                quant_keys = [k for k in tensor_keys if any(s in k.lower() for s in ["scale", "zero", "quant", "_q", "_s"])]
                print(f"\n  quant-like tensor names ({len(quant_keys)} found, showing 15):")
                for k in quant_keys[:15]:
                    print(f"    {k}: {hdr[k]}")
                # Count dtypes
                from collections import Counter
                dtypes = Counter(v["dtype"] for k, v in hdr.items() if k != "__metadata__")
                print(f"\n  dtype distribution: {dict(dtypes)}")
        break  # only need one
