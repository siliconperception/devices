# vla

Stateful recurrent CNN language model. `vla.py` trains it; `chart.py` plots the loss log;
`doit` is the experiment history.

## HellaSwag evaluation

`--evaluate` scores a checkpoint zero-shot on
[HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag) and exits — no training
dataset, no training. Each example is a context plus four candidate endings; the context is
fed to the model, each ending is then scored by the summed log-probability of its UTF-8
bytes, and the most likely one is the model's answer. Compare against the leaderboard at
[rowanzellers.com/hellaswag](https://rowanzellers.com/hellaswag/) — random is 25%.

```bash
python vla.py --evaluate --load checkpoint.pt                 # validation split (10042 examples)
python vla.py --evaluate --load checkpoint.pt --limit 500     # quick estimate
```

Two accuracies are printed, the usual pair for zero-shot LM eval. **acc_norm** is the
headline: it normalizes by ending length (mean log-prob per byte), so long endings are not
penalized. **acc** uses the un-normalized total.

`--batch` is examples per batch (each becomes four sequences once the endings fork off) and
`--monitor` is the progress-print interval. `--split train` scores the train split instead;
the test split is unlabeled and cannot be scored.

## Datasets

`--dataset tiny | c4 | web | brt | dolma3 | s1k`, or `--mix "brt:0.5,web:0.5"` for a
token-proportional mix. All except `dolma3` stream straight from the hub:

```bash
python vla.py --dataset web --streaming ...
```

`s1k` ([simplescaling/s1K](https://huggingface.co/datasets/simplescaling/s1K)) is the 1000
curated reasoning examples from the s1 paper. It has no plain text column, so each example
is rendered to `question<think>\n…\n</think>\n<answer>\n…\n</answer>` — the same shape
`brt` already uses, so both can be mixed without teaching two formats:

```bash
python vla.py --mix "brt:0.45,web:0.45,s1k:0.10" ...
```

It is tiny (~16 MB, ~13 K bytes of trace per example), so on its own it loops over the same
1000 examples within a few hundred steps — use it as a small slice of a `--mix`.

`dolma3` ([allenai/dolma3_mix-150B-1025](https://huggingface.co/datasets/allenai/dolma3_mix-150B-1025))
must be downloaded and prepared first — see below.

## dolma3: download and prep

The hub copy cannot be used as-is, in either mode:

- **Streaming** yields one topic at a time. The 6081 shards are partitioned *by topic*
  (one shard is all `common_crawl-adult_content`, the next all `software_development`), and
  a streaming dataset reads shards sequentially, so `shuffle(buffer_size=...)` only mixes
  *within the shard currently being read*. A run trains on a single topic for tens of
  thousands of documents.
- **Non-streaming** fails outright. Each source writes a different `metadata` struct
  (`cc_dump`/`weborganizer` for common_crawl, `pdf-total-pages`/`olmocr-version` for
  olmocr, which also adds `source`/`added`/`created`), so the json builder cannot cast the
  shards to one schema. Restricting to `features={'text': ...}` does not help: the builder
  has no column selection, and the cast requires the file's columns to be a subset of the
  schema.

`dolma3_prep.py` fixes both by rewriting the corpus once into text-only parquet parts that
are each a uniform random sample of the whole corpus — so any later read order, streaming
or not, is already globally shuffled.

```bash
# 1. download: 110 GB, 6081 .jsonl.zst shards
hf download allenai/dolma3_mix-150B-1025 --repo-type dataset \
    --local-dir ./dolma3 --max-workers 16

# 2. prep: ~110 GB of parquet parts (./dolma3 can be deleted afterwards)
python dolma3_prep.py --src ./dolma3 --dst ./dolma3_text

# 3. train: vla.py finds ./dolma3_text automatically
python vla.py --dataset dolma3 --streaming ...
```

Use `--streaming` at step 3. The parts are pre-shuffled, so streaming reads them directly
and skips building an Arrow cache — which for this corpus would be ~945 GB, since 85% of
the jsonl bytes are `text` (~40M documents). Non-streaming also works, and adds a row-level
shuffle across all parts, if the disk is worth it to you.

### How the prep works

1. **scatter** — each worker reads a *random* subset of the raw shards (the shard list is
   shuffled before being split across workers), keeps only `text`, and appends each document
   to a randomly chosen one of its output parts. No part is tied to a topic.
2. **shuffle** — pass 1 appends in shard order, so each part is then read into memory, its
   rows permuted, and rewritten.

### Memory

Documents run from a few kB to several MB (olmocr PDF text), so pass 1 bounds its buffers
in **bytes**, not rows: peak RSS is about `--workers × --budget_mb`. Pass 2 holds a whole
part plus its permuted copy per process: about `--procs × 2 × part_size`, where
`part_size ≈ 945 GB / (--workers × --parts)`. The script prints both estimates before it
starts — check them against `free -g` first. Defaults (`--workers 16 --parts 64
--budget_mb 256 --procs 4`) come to roughly 4 GB then 7.4 GB.

If a training run is using the machine at the same time, `--procs 2 --budget_mb 128` halves
both. `--workers 8` halves pass 1 again, at roughly double the wall time.

Re-running refuses to start if parts from an earlier run are still in `--dst`; a rerun with
different `--workers`/`--parts` would otherwise leave them behind to be read as duplicated,
unshuffled training data. Clear them first:

```bash
rm -f ./dolma3_text/part-*.parquet
```
