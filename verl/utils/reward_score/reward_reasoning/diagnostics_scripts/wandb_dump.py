# SPDX-License-Identifier: Apache-2.0
"""Extract the full metric history from a local ``.wandb`` datastore file.

Motivation: these runs log to wandb in online mode, so ``wandb/<run>/files/``
never gets a ``wandb-summary.json`` and the console log carries no reward
numbers (``ray_trainer.py:1266`` only calls ``logger.log``, it does not print).
When a run dies at SLURM wall-time the dashboard is the only copy of the curve.
The ``.wandb`` binary next to it, however, holds every logged step -- this reads
it with no wandb install required.

Usage:
    python3 wandb_dump.py run-*/run-*.wandb              # JSONL to stdout
    python3 wandb_dump.py --csv out.csv run-*/run-*.wandb

Format (wandb leveldb-log framing + protobuf):
    file header : 7 bytes  b':W&B' + uint16 magic + uint8 version
    block       : 32768 bytes, records packed, zero-padded tail
    record hdr  : crc32(4 LE) + length(2 LE) + type(1)  [1=FULL 2=FIRST 3=MID 4=LAST]
    Record        field 2 -> HistoryRecord
    HistoryRecord field 1 -> repeated HistoryItem
    HistoryItem   field 2 -> key, field 16 -> value_json

Note the HistoryItem key is field *2* in the wandb version these runs used
(the proto has ``key = 1`` in some releases -- verified empirically here).
"""

import argparse
import csv
import json

BLOCK = 32768
HDR = 7


def _varint(buf, i):
    shift = val = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, i
        shift += 7


def _fields(buf):
    """Yield (field_number, wire_type, payload) for one protobuf message."""
    i, n = 0, len(buf)
    while i < n:
        try:
            tag, i = _varint(buf, i)
            fnum, wt = tag >> 3, tag & 7
            if wt == 0:
                val, i = _varint(buf, i)
                yield fnum, wt, val
            elif wt == 1:
                yield fnum, wt, buf[i : i + 8]
                i += 8
            elif wt == 2:
                ln, i = _varint(buf, i)
                yield fnum, wt, buf[i : i + ln]
                i += ln
            elif wt == 5:
                yield fnum, wt, buf[i : i + 4]
                i += 4
            else:
                return
        except IndexError:
            return


def _records(path):
    """Yield reassembled Record payloads from the leveldb-log framing."""
    data = open(path, "rb").read()
    pos, pending = HDR, b""
    while pos + HDR <= len(data):
        if BLOCK - (pos % BLOCK) < HDR:  # tail padding
            pos += BLOCK - (pos % BLOCK)
            continue
        ln = int.from_bytes(data[pos + 4 : pos + 6], "little")
        rtype = data[pos + 6]
        payload = data[pos + HDR : pos + HDR + ln]
        pos += HDR + ln
        if rtype == 1:
            yield payload
        elif rtype == 2:
            pending = payload
        elif rtype == 3:
            pending += payload
        elif rtype == 4:
            yield pending + payload
            pending = b""
        elif pos % BLOCK:  # zero padding -> skip to next block
            pos += BLOCK - (pos % BLOCK)


def history(path):
    """Yield one dict per logged step."""
    for rec in _records(path):
        for fnum, wt, payload in _fields(rec):
            if fnum != 2 or wt != 2:  # not a HistoryRecord
                continue
            row = {}
            for hf, hwt, item in _fields(payload):
                if hf != 1 or hwt != 2:
                    continue
                key = value = None
                for inum, iwt, ipay in _fields(item):
                    if iwt != 2:
                        continue
                    if inum == 2:
                        key = ipay.decode("utf-8", "replace")
                    elif inum == 16:
                        value = ipay.decode("utf-8", "replace")
                if key is not None and value is not None:
                    try:
                        row[key] = json.loads(value)
                    except json.JSONDecodeError:
                        row[key] = value
            if row:
                yield row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help=".wandb files, in chronological order")
    ap.add_argument("--csv", help="write a merged CSV here instead of JSONL stdout")
    args = ap.parse_args()

    rows = [r for p in args.paths for r in history(p)]
    if not args.csv:
        for r in rows:
            print(json.dumps(r))
        return

    # Later files win on overlapping steps (a resume replays its start step).
    merged = {}
    for r in rows:
        if "_step" in r:
            merged.setdefault(r["_step"], {}).update(r)
    keys = sorted({k for r in merged.values() for k in r})
    keys = ["_step"] + [k for k in keys if k != "_step"]
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for s in sorted(merged):
            w.writerow(merged[s])
    print(f"wrote {args.csv}: {len(merged)} steps x {len(keys)} metrics")


if __name__ == "__main__":
    main()
