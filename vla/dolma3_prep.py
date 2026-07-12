#!/usr/bin/env python
"""Convert a downloaded dolma3 repo into globally-shuffled, text-only parquet shards.

dolma3 ships 6081 .jsonl.zst shards partitioned *by topic* (one shard is all
common_crawl-adult_content, the next all software_development), and its `metadata` column
has a different struct per source, so `datasets` cannot even load the raw shards
non-streaming (CastError), and a streaming buffer shuffle only mixes within the one shard
it happens to be reading. Both problems go away by rewriting the corpus once:

  pass 1 (scatter)  each worker reads a random subset of the raw shards, keeps only `text`,
                    and appends each document to a randomly chosen output part file. A part
                    therefore draws from shards all over the corpus, not from one topic.
  pass 2 (shuffle)  each part file is read into memory, its rows permuted, and rewritten,
                    so the order *inside* a part is shuffled too (pass 1 appends in shard
                    order). A part is ~1-2 GB, so this is bounded RAM.

The result is a set of parquet shards that are each a uniform random sample of the whole
corpus: any read order after this — streaming or not — is already globally shuffled.

    hf download allenai/dolma3_mix-150B-1025 --repo-type dataset --local-dir ./dolma3
    python dolma3_prep.py --src ./dolma3 --dst ./dolma3_text

Then train with `--dataset dolma3` (vla.py picks up ./dolma3_text automatically).
"""
import argparse
import glob
import io
import json
import multiprocessing as mp
import os
import random
import time
import zlib

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

parser = argparse.ArgumentParser()
parser.add_argument('--src',       default='./dolma3',      help='downloaded repo (jsonl.zst)')
parser.add_argument('--dst',       default='./dolma3_text', help='output dir (parquet)')
parser.add_argument('--workers',   default=16, type=int, help='parallel reader processes')
parser.add_argument('--parts',     default=64, type=int,
                    help='output parts per worker. Total parts = workers*parts; keep the part '
                         'size (corpus_text / total_parts) well under the pass-2 RAM budget')
parser.add_argument('--budget_mb', default=256, type=int,
                    help='pass 1: scatter buffer budget PER WORKER. Documents vary from a few '
                         'kB to many MB (olmocr PDFs), so buffers are bounded in bytes, not '
                         'rows -- peak pass-1 RSS is roughly workers*budget_mb')
parser.add_argument('--procs',     default=4, type=int,
                    help='pass 2: parts shuffled in parallel. Each holds a whole part plus its '
                         'permuted copy, so peak RSS is roughly procs*2*part_size')
parser.add_argument('--seed',      default=1234, type=int)
args = parser.parse_args()

SCHEMA = pa.schema([('text', pa.string())])


def scatter(job):
    """Read this worker's raw shards; append each doc to a random one of its part files."""
    wid, shards = job
    rng     = random.Random(args.seed + wid)
    dctx    = zstd.ZstdDecompressor()
    paths   = [os.path.join(args.dst, f'part-{wid:03d}-{b:03d}.parquet') for b in range(args.parts)]
    writers = [pq.ParquetWriter(p, SCHEMA, compression='zstd') for p in paths]
    buf     = [[] for _ in range(args.parts)]
    nbytes  = [0] * args.parts                    # bytes held per part buffer
    budget  = args.budget_mb * 1024 * 1024
    held    = 0                                   # total bytes buffered by this worker
    docs    = 0

    def flush(b):
        nonlocal held
        if buf[b]:
            writers[b].write_table(pa.table({'text': buf[b]}, schema=SCHEMA))
            buf[b].clear()
            held     -= nbytes[b]
            nbytes[b] = 0

    for i, shard in enumerate(shards):
        with open(shard, 'rb') as fh:
            for line in io.TextIOWrapper(dctx.stream_reader(fh), encoding='utf-8'):
                try:
                    text = json.loads(line).get('text') or ''
                except Exception:
                    continue                      # tolerate a truncated/bad line
                if not text:
                    continue
                b = rng.randrange(args.parts)     # scatter across parts
                buf[b].append(text)
                sz         = len(text)
                nbytes[b] += sz
                held      += sz
                docs      += 1
                # Bound memory by bytes: one worker never holds more than `budget`, however
                # large individual documents happen to be. Flush the fattest buffer, which
                # keeps parquet row groups reasonably sized.
                while held > budget:
                    flush(max(range(args.parts), key=lambda k: nbytes[k]))
        if wid == 0:
            print(f'  worker0: {i + 1}/{len(shards)} shards, {docs / 1e6:.2f}M docs, '
                  f'{held / 1e6:.0f} MB buffered', flush=True)
    for b in range(args.parts):
        flush(b)
        writers[b].close()
    return docs


def shuffle_part(path):
    """Permute the rows of one part file in place (pass 1 wrote them in shard order)."""
    if pq.read_metadata(path).num_rows == 0:      # a worker can run out of docs to scatter
        os.remove(path)
        return 0
    tbl = pq.read_table(path, schema=SCHEMA)
    perm = list(range(tbl.num_rows))
    random.Random(args.seed + zlib.crc32(os.path.basename(path).encode())).shuffle(perm)
    pq.write_table(tbl.take(pa.array(perm, type=pa.int64())), path + '.tmp', compression='zstd')
    os.replace(path + '.tmp', path)
    return tbl.num_rows


if __name__ == '__main__':
    raw = sorted(glob.glob(os.path.join(args.src, '**', '*.jsonl.zst'), recursive=True))
    if not raw:
        raise SystemExit(f'no .jsonl.zst shards under {args.src}')
    os.makedirs(args.dst, exist_ok=True)

    # A previous (or crashed) run leaves parts behind. Re-running with different
    # --workers/--parts writes a different set of names, so the leftovers would survive and
    # be read as training data -- duplicated documents, and pass-2-unshuffled ones at that.
    stale = glob.glob(os.path.join(args.dst, 'part-*.parquet'))
    if stale:
        raise SystemExit(f'{len(stale)} part file(s) already in {args.dst} -- from an earlier '
                         f'run. Remove them first:\n    rm {args.dst}/part-*.parquet')

    # Shuffle the shard list *before* splitting it across workers: each worker then reads a
    # random subset of topics, so every part file it writes mixes topics from all over.
    random.Random(args.seed).shuffle(raw)
    jobs = [(w, raw[w::args.workers]) for w in range(args.workers)]

    # dolma3's text decompresses ~8.5x, so estimate the corpus (and hence the part size and
    # the pass-2 footprint) from the compressed bytes on disk rather than guessing.
    n_parts = args.workers * args.parts
    text_gb = 8.5 * sum(os.path.getsize(f) for f in raw) / 1e9
    part_gb = text_gb / n_parts
    print(f'{len(raw)} raw shards ({text_gb:.0f} GB of text) -> {n_parts} parts '
          f'({args.workers} workers x {args.parts}), ~{part_gb * 1e3:.0f} MB of text per part')
    print(f'peak RSS ~{args.workers * args.budget_mb / 1e3:.1f} GB (pass 1: workers x budget) '
          f'then ~{args.procs * 2 * part_gb:.1f} GB (pass 2: procs x 2 x part)')

    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        docs = sum(pool.map(scatter, jobs))
    print(f'pass 1 (scatter): {docs / 1e6:.1f}M docs in {(time.time() - t0) / 60:.1f} min')

    parts = sorted(glob.glob(os.path.join(args.dst, 'part-*.parquet')))
    t1 = time.time()
    with mp.Pool(args.procs) as pool:             # each holds a whole part in RAM
        rows = sum(pool.map(shuffle_part, parts, chunksize=1))
    print(f'pass 2 (shuffle): {rows / 1e6:.1f}M rows in {len(parts)} parts, '
          f'{(time.time() - t1) / 60:.1f} min')
    print(f'done: {args.dst} ({(time.time() - t0) / 60:.1f} min total)')
