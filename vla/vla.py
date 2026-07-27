#!/usr/bin/env python3
# vla.py -- Vision-Language-Audio CNN model (text-only core)
# Silicon Perception Inc.
#
# Recurrent CNN core: a per-token ContextCNN DFF loop + DecoderCNN. The current
# token (raw byte) is embedded, broadcast over an S×S grid, projected and summed
# into the recurrent DFF state; the decoder reads the state out to next-token
# logits. Image and audio conditioning will be added back incrementally.
#
# Vocabulary: 256 (raw bytes). No attention, no positional encoding.

import math
import torch ; print('torch', torch.__version__)
import torch.nn as nn
from torch.nn import functional as F
import torchinfo
import numpy as np
import argparse
import glob
import os
import queue
import threading
import subprocess
import datetime
import time
import random
import re
import matplotlib.pyplot as plt

NULL      = 0x00
START     = 0x02
END       = 0x03

STX_TEXT  = '<STX>'   # how START renders in displayed text
ETX_TEXT  = '<ETX>'   # how END renders in displayed text


def _printable(s):
    """Render model text for display: START/END as <STX>/<ETX>, whitespace escaped,
    any remaining unprintable bytes dropped."""
    s = (s.replace(chr(START), STX_TEXT).replace(chr(END), ETX_TEXT)
          .replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r'))
    return ''.join(c for c in s if c.isprintable())


# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# architecture
parser.add_argument('--context',       default=7,     type=int,
                    help='spatial size S of the DFF grid (S×S)')
parser.add_argument('--n_hidden',      default=512,   type=int,
                    help='ContextCNN channel depth')
parser.add_argument('--c_text',        default=1,     type=int,
                    help='token embedding channels (None → n_hidden)')
parser.add_argument('--depth',         default=7,     type=int,
                    help='ContextCNN conv layer count')
parser.add_argument('--n_layers',      default=3,     type=int,
                    help='stack N (ContextCNN + feedback) recurrent layers in series between '
                         'the thin encoder and decoder; each layer keeps its own DFF state. '
                         '1 = original single-layer behavior')
parser.add_argument('--kernel',        default=3,     type=int,
                    help='conv kernel size (odd integer)')
# training
parser.add_argument('--dataset',       default='tiny',
                    help='tiny | c4 | web | brt | dolma3')
parser.add_argument('--mix',           default=None,
                    help='joint multi-dataset training, "name:weight,..." e.g. '
                         '"brt:0.5,web:0.5"; overrides --dataset, samples one dataset per clip '
                         '(token-proportional)')
parser.add_argument('--streaming',     default=False, action='store_true')
parser.add_argument('--shards',        default=None,  type=int,
                    help='non-streaming: load only the first N parquet shards of the dataset '
                         '(skips most of the slow "Loading dataset shards" step for large '
                         'corpora like web/openwebtext); ignored with --streaming or if the '
                         'repo is not parquet')
parser.add_argument('--batch',         default=100,   type=int)
parser.add_argument('--workers',       default=12,    type=int,
                    help='parallel clip-builder threads (decode/download); raise for cc3m image streaming')
parser.add_argument('--run_steps',     default=None,  type=int,
                    help='stop the run after this many gradient steps (None = run indefinitely); '
                         'handy for fixed-length sweeps')
parser.add_argument('--opt',           default='sgd',
                    choices=['sgd', 'rmsprop', 'rprop', 'adagrad', 'adamw'],
                    help='training optimizer')
parser.add_argument('--momentum',      default=0.0,   type=float,
                    help='momentum (--opt sgd/rmsprop; wired to beta2 for adamw; '
                         'ignored by rprop/adagrad)')
parser.add_argument('--weight_decay',  default=0.0,   type=float,
                    help='L2 weight decay (--opt sgd/rmsprop/adagrad; ignored by rprop)')
# learning rate: all --schedule modes are driven by this one set. lr_max is the peak
# (and the optimizer base lr); lr_min is the floor; lr_warmup / lr_period are step counts.
parser.add_argument('--schedule',      default='const',
                    choices=['const', 'linear', 'triangle', 'cosine'],
                    help='lr schedule: "const" (fixed --lr_max), "linear" (ramp --lr_min->'
                         '--lr_max over --lr_period, then hold), "triangle" (ramp --lr_min->'
                         '--lr_max over --lr_warmup, then --lr_max->--lr_min over --lr_period) '
                         'or "cosine" (cyclic --lr_min<->--lr_max, --lr_period steps/cycle)')
parser.add_argument('--lr_min',        default=1e-6,  type=float,
                    help='minimum / floor learning rate (linear|triangle|cosine)')
parser.add_argument('--lr_max',        default=0.001, type=float,
                    help='peak learning rate; also the constant lr for --schedule const '
                         'and the optimizer base lr')
parser.add_argument('--lr_mult',       default=1.0,   type=float,
                    help='per-layer lr multiplier: layer k (1-based) gets base_lr * '
                         '--lr_mult**(k-1), so with --lr_mult 10 --n_layers 3 the three '
                         'layers use lr, 10*lr, 100*lr. Default 1.0 (all layers equal)')
parser.add_argument('--lr_warmup',     default=1000,  type=int,
                    help='ramp-up length in steps (triangle)')
parser.add_argument('--lr_period',     default=10000, type=int,
                    help='schedule length in steps: linear ramp / triangle ramp-down / cosine cycle')
# I/O
parser.add_argument('--load',          default=None)
parser.add_argument('--save',          default='checkpoint.pt')
parser.add_argument('--checkpoint',    default=1000,  type=int)
parser.add_argument('--log',           default=None)
parser.add_argument('--monitor',       default=100,   type=int)
parser.add_argument('--generate',      default=False, action='store_true',
                    help='one-shot: load checkpoint, generate from START+prompt, print, exit (no dataset/training)')
parser.add_argument('--n',             default=200,   type=int,
                    help='tokens to generate per sample')
parser.add_argument('--prompt',        default='',
                    help='generation prompt; START (0x02) is always prepended, so leave '
                         'empty to generate from START alone')
parser.add_argument('--evaluate',      default=False, action='store_true',
                    help='one-shot: load checkpoint, score HellaSwag zero-shot, print accuracy, '
                         'exit (no training dataset/training)')
parser.add_argument('--split',         default='validation', choices=['train', 'validation'],
                    help='--evaluate: HellaSwag split (its test split is unlabeled, so it '
                         'cannot be scored)')
parser.add_argument('--limit',         default=None,  type=int,
                    help='--evaluate: score only the first N examples (None = the whole split)')
parser.add_argument('--seed',          default=None,  type=int)
parser.add_argument('--device',        default=None)
parser.add_argument('--verbose',       default=False, action='store_true',
                    help='print the torchinfo model summaries in --generate/--vis (training '
                         'and --evaluate always print them, since they are logged)')
parser.add_argument('--vis',           default=False, action='store_true',
                    help='live text-generation viz (DFF std + P(next token)) from the '
                         'prompt token; no training. Read-only. Press x to exit')
parser.add_argument('--delay',         default=0.1,   type=float,
                    help='seconds to pause per vis frame')
parser.add_argument('--cmap',          default='viridis',
                    help='matplotlib colormap for --vis')
parser.add_argument('--temperature',   default=1.0,   type=float,
                    help='sampling temperature for --vis/--generate and the training-loop '
                         'text samples (0 = argmax)')
parser.add_argument('--push',          default=False, action='store_true',
                    help='push the --load checkpoint to the HuggingFace hub as a '
                         'PyTorchModelHubMixin model (repo --hub_repo), then exit. '
                         'No training. Requires a write token (huggingface-cli login)')
parser.add_argument('--pretrained',    default=False, action='store_true',
                    help='load weights + architecture from the HuggingFace hub repo '
                         '--hub_repo instead of a local checkpoint. Overrides --load')
parser.add_argument('--revision',      default=None,
                    help='--pretrained: hub branch, tag or commit (default: main)')
parser.add_argument('--hub_repo',      default='siliconperception/VLA',
                    help='--push target / --pretrained source hub repo id')
parser.add_argument('--private',       default=False, action='store_true',
                    help='--push: create the hub repo private (default: the org/user default '
                         'visibility). Ignored if the repo already exists')
args = parser.parse_args()

# ── multi-dataset mix ─────────────────────────────────────────────────────────

def _parse_mix(spec):
    """'brt:0.5,web:0.5' -> ([names], [normalized weights]); None if unset."""
    if not spec:
        return None
    names, weights = [], []
    for part in spec.split(','):
        nm, _, w = part.partition(':')
        names.append(nm.strip())
        weights.append(float(w) if w else 1.0)
    s = sum(weights) or 1.0
    return names, [w / s for w in weights]

_mix = _parse_mix(args.mix)

# ── restore architecture args from checkpoint ─────────────────────────────────

_ARCH_ARGS = ('context', 'n_hidden', 'c_text', 'n_layers', 'depth', 'kernel')

_loaded_ckpt = None
_start_step  = 0                 # resumed global step count (0 for a fresh run)
if args.pretrained and args.load is not None:
    # --pretrained overrides --load: the hub repo carries both weights and config.json,
    # so a local checkpoint would only be half-used (its saved_args then foreign weights).
    print(f'--pretrained overrides --load {args.load}: loading {args.hub_repo} from the hub')
    args.load = None
if args.load is not None:
    _loaded_ckpt = torch.load(args.load, map_location='cpu', weights_only=True)
    for _k in _ARCH_ARGS:        # rebuild the model with the architecture it was trained with
        # older checkpoints may predate an arch arg → keep the current default for it
        setattr(args, _k, _loaded_ckpt['saved_args'].get(_k, getattr(args, _k)))
    _start_step = int(_loaded_ckpt.get('step', 0))   # continue the step sequence

if args.c_text is None:          # default token embedding width to n_hidden
    args.c_text = args.n_hidden

# ── log / device / seed ───────────────────────────────────────────────────────

if args.log is None:
    if args.generate or args.vis or args.push:   # one-shot modes: don't create a log file
        args.log = os.devnull
    else:
        # training -> log/log.<date>, --evaluate -> log/eval.<date>, so eval scores are kept
        # and can be compared across checkpoints. Never reuse a name: two runs started in the
        # same second would otherwise interleave their lines in one file.
        os.makedirs('log', exist_ok=True)
        date = subprocess.check_output(['/usr/bin/date', '+%Y.%m.%d-%H.%M.%S']).decode().strip()
        stem = f'log/{"eval" if args.evaluate else "log"}.{date}'
        args.log, n = stem, 1
        while os.path.exists(args.log):
            args.log, n = f'{stem}.{n}', n + 1
if args.device is None:
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.seed is None:
    args.seed = int.from_bytes(os.urandom(4), byteorder='big')
    print('seed', args.seed)

torch.manual_seed(args.seed)
print(args)
# Prepend the resumed checkpoint's log so chart.py sees one uninterrupted sequence.
# Only for real training log files (skip devnull / generate / vis / evaluate).
if (_loaded_ckpt is not None and not args.evaluate
        and args.log and args.log != os.devnull):
    _prev_log = _loaded_ckpt.get('log') or ''
    if _prev_log:
        with open(args.log, 'a') as f:
            f.write(_prev_log)
            if not _prev_log.endswith('\n'):
                f.write('\n')
with open(args.log, 'a') as f:
    print('ARGS', args, file=f)

# ── model ─────────────────────────────────────────────────────────────────────

class ContextCNN(nn.Module):
    """DFF state update: a plain chain of conv+batchnorm+ReLU blocks.
    Input:  [B, n_hidden, S, S]
    Output: [B, n_hidden, S, S]
    """
    def __init__(self, n_hidden, depth, kernel):
        super().__init__()
        pad = kernel // 2
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Conv2d(n_hidden, n_hidden, kernel, padding=pad),
                          nn.BatchNorm2d(n_hidden, affine=False), nn.ReLU())
            for _ in range(depth))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderCNN(nn.Module):
    """1×1 conv n_hidden→256, then global average pool → logits.
    Input:  [B, n_hidden, S, S]
    Output: [B, 256]
    """
    def __init__(self, n_hidden, S):
        super().__init__()
        self.proj = nn.Conv2d(n_hidden, 256, 1)
        self.pool = nn.AvgPool2d(S)

    def forward(self, x):
        return self.pool(self.proj(x)).squeeze(-1).squeeze(-1)


# PyTorchModelHubMixin gives VLAModel save_pretrained()/push_to_hub()/from_pretrained()
# (see --push). It records the __init__ kwargs as config.json, so a hub checkpoint rebuilds
# with the architecture it was trained with. huggingface_hub is optional: without it the
# model is a plain nn.Module and only --push is unavailable.
try:
    from huggingface_hub import PyTorchModelHubMixin as _HubMixin
except ImportError:                                  # pragma: no cover
    class _HubMixin:                                 # no-op base
        pass


class VLAModel(nn.Module, _HubMixin):
    """Recurrent CNN language model over raw bytes. Each step the current token is
    embedded, broadcast to an S×S grid, projected to n_hidden and summed into the
    per-layer DFF state; ContextCNN updates the state and the decoder reads it out
    to next-token logits. n_layers stacks (ContextCNN + feedback) units in series,
    each carrying its own DFF state."""
    def __init__(self, n_hidden, depth, kernel, S,
                 c_text=None, n_layers=1):
        super().__init__()
        self.S          = S
        self.n_hidden   = n_hidden
        self.c_text     = c_text if c_text is not None else n_hidden
        self.n_layers   = max(1, n_layers)    # stacked (ContextCNN + feedback) layers (see --n_layers)

        self.tok_embed = nn.Embedding(256, self.c_text)   # current token → c_text channels
        self.context   = ContextCNN(n_hidden, depth, kernel)
        self.decoder   = DecoderCNN(n_hidden, S)
        # token grid (c_text) → n_hidden, summed into the DFF state
        self.text_proj = nn.Conv2d(self.c_text, n_hidden, 1)

        # Deep layers 2..N: each is its own (ContextCNN + feedback) unit. Layer 1 takes
        # the token injection; deep layers take the previous layer's output combined with
        # their own DFF state. Only built when n_layers>1, so the n_layers=1 state_dict is
        # byte-identical to the single-layer model.
        if self.n_layers > 1:
            self.deep_context = nn.ModuleList(
                ContextCNN(n_hidden, depth, kernel) for _ in range(self.n_layers - 1))
            self.deep_proj = nn.ModuleList(
                nn.Conv2d(n_hidden, n_hidden, 1) for _ in range(self.n_layers - 1))
        else:
            self.deep_context = self.deep_proj = None

        self.dff = None   # list of n_layers tensors [B, n_hidden, S, S], detached; None until first step

    def _init_dff(self, B, device):
        dev = torch.device(device)
        if (self.dff is None or len(self.dff) != self.n_layers
                or self.dff[0].shape[0] != B or self.dff[0].device != dev):
            self.dff = [torch.zeros(B, self.n_hidden, self.S, self.S, device=device)
                        for _ in range(self.n_layers)]

    def _tok_grid(self, byte_idx):
        B   = byte_idx.shape[0]
        tok = self.tok_embed(byte_idx)                             # [B, c_text]
        return tok.view(B, self.c_text, 1, 1).expand(-1, -1, self.S, self.S)

    def _layers(self, dff, tok_grid):
        """Run the stack of n_layers (ContextCNN + feedback) units once.
        dff: list of n_layers tensors [B, n_hidden, S, S]. Layer 0 takes the token injection;
        each deeper layer takes the previous layer's output summed with its own DFF state.
        Returns (new_dff_list, last_layer_output [B, n_hidden, S, S]). No detach here — the
        caller detaches the returned list to bound the time horizon."""
        o = self.context(dff[0] + self.text_proj(tok_grid))
        outs = [o]
        for li in range(self.n_layers - 1):
            o = self.deep_context[li](dff[li + 1] + self.deep_proj[li](o))
            outs.append(o)
        return outs, o

    def forward(self, byte_idx, targets=None, flag=None):
        """
        byte_idx  [B] long — current token (teacher-forced: ground truth at t)
        targets   [B] long — next token (CE target), or None
        flag      [B] bool — reset DFF for these batch elements
        """
        B, dev = byte_idx.shape[0], byte_idx.device
        self._init_dff(B, dev)

        if flag is not None and flag.any():
            self.dff = [d.clone() for d in self.dff]   # clone before in-place
            for d in self.dff:
                d[flag] = 0.0

        stack, new_ctx = self._layers(self.dff, self._tok_grid(byte_idx))
        self.dff = [d.detach().clone() for d in stack]   # store for next step (gradient horizon = 1)

        logits = self.decoder(new_ctx)
        loss   = F.cross_entropy(logits, targets.long()) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt_bytes, n_tokens, temperature=1.0):
        """Autoregressive generation seeded by prompt_bytes (temperature 0 = argmax)."""
        self.eval()
        dev = next(self.parameters()).device
        dff = [torch.zeros(1, self.n_hidden, self.S, self.S, device=dev)
               for _ in range(self.n_layers)]

        def _step(b):
            nonlocal dff
            bi   = torch.tensor([b], dtype=torch.long, device=dev)
            tok  = self.tok_embed(bi).view(1, self.c_text, 1, 1).expand(-1, -1, self.S, self.S)
            stack, new_ctx = self._layers(dff, tok)
            dff     = [d.detach().clone() for d in stack]
            if not all(d.isfinite().all() for d in dff):
                print('generate: dff NaN/Inf')
                return None
            logits = self.decoder(new_ctx)
            if not logits.isfinite().all():
                print('generate: logits NaN/Inf')
                return None
            return logits

        # feed each prompt byte exactly once; logits after the last byte seed generation
        logits = None
        for b in prompt_bytes:
            logits = _step(b)
            if logits is None:
                return []

        if logits is None:
            return []

        out = []
        for _ in range(n_tokens):
            prob = F.softmax(logits[0] / max(temperature, 1e-6), dim=-1)
            bval = torch.multinomial(prob, 1).item()
            if bval == END:
                break
            if bval != NULL:
                out.append(bval)
            logits = _step(bval)
            if logits is None:
                break
        return out


# ── DFF state stats ─────────────────────────────────────────────────────────────
# The DFF state is a list of one tensor per recurrent layer. These flatten across all
# layers to report a single aggregate (element-weighted over the whole state).

def _dff_absmax(dff):
    """Max |value| across the whole per-layer DFF list (0.0 if empty/None)."""
    return max((d.abs().max().item() for d in dff), default=0.0) if dff else 0.0


def _dff_stats(dff):
    """(mean, std, absmax, zero_frac) flattened across the whole per-layer DFF list."""
    if not dff:
        return 0.0, 0.0, 0.0, 0.0
    flat = torch.cat([d.reshape(-1) for d in dff])
    return (flat.mean().item(), flat.std().item(),
            flat.abs().max().item(), (flat == 0).float().mean().item())


# ── clip builders ──────────────────────────────────────────────────────────────

# Thread-safe tally of text examples. Builder threads
# count examples/tokens and stash the most recent raw sample; the main loop drains
# both each checkpoint to report throughput and show a training-text sample.
_text_lock  = threading.Lock()
_text_stats = {'examples': 0, 'tokens': 0, 'sample': ''}

def _text_tally(n_tokens, sample=''):
    with _text_lock:
        _text_stats['examples'] += 1
        _text_stats['tokens']   += n_tokens
        if sample:
            _text_stats['sample'] = sample

def _text_drain():
    """Return (examples, tokens, sample) since the last drain; reset the counters.
    The sample is kept (not cleared) so a recent one is always available."""
    with _text_lock:
        e, t, s = _text_stats['examples'], _text_stats['tokens'], _text_stats['sample']
        _text_stats['examples'] = _text_stats['tokens'] = 0
    return e, t, s
# Each builder returns (steps, caption). A step is a 4-tuple
#   (x_byte, y_byte, img_or_None, aud_or_None)
# one entry per 100 Hz frame.

def _text_to_bytes(text):
    return list(text.encode('utf-8', errors='replace'))


def clip_text(text):
    """One step per byte: input/target are consecutive bytes of [START]+text+[END]."""
    full = [START] + _text_to_bytes(text) + [END]
    steps = [(full[i], full[i + 1]) for i in range(len(full) - 1)]
    return steps, text


# ── dataset loading ────────────────────────────────────────────────────────────

_TEXT_DATASETS = {
    'tiny':   ('roneneldan/TinyStories', None, 'train', 'text'),
    'c4':     ('allenai/c4',            'en', 'train', 'text'),
    'web':    ('Skylion007/openwebtext', None, 'train', 'text'),
    'brt':    ('allenai/big-reasoning-traces', 'DeepSeek', 'train', 'text'),
    'dolma3': ('allenai/dolma3_mix-150B-1025', None, 'train', 'text'),
}

# A dataset prepared on local disk takes precedence over the hub repo. dolma3 must be
# prepared this way (see dolma3_prep.py): the hub shards are topic-partitioned and
# per-source schemas differ, so streaming them trains on one topic at a time and loading
# them non-streaming fails to cast. dolma3_prep.py rewrites the corpus into text-only
# parquet parts that are each a uniform random sample of the whole corpus.
_LOCAL_DIRS = {'dolma3': './dolma3_text'}


def _local_files(name):
    d = _LOCAL_DIRS.get(name)
    return sorted(glob.glob(os.path.join(d, '*.parquet'))) if d else []


def _split_for_name(name):
    return _TEXT_DATASETS[name][2] if name in _TEXT_DATASETS else 'train'


def _shard_files(hf, cfg, n):
    """First n train parquet shard paths in dataset repo `hf` (config subdir `cfg` if the
    repo is multi-config). Empty list if the repo isn't parquet-based (e.g. json/script)."""
    from huggingface_hub import HfApi
    sib = [s.rfilename for s in HfApi().repo_info(hf, repo_type='dataset').siblings]
    par = sorted(f for f in sib if f.endswith('.parquet') and 'train' in f)
    if cfg:                                       # multi-config repo: keep only this config
        par = [f for f in par if f'/{cfg}/' in f or f.startswith(f'{cfg}/')]
    return par[:n]


def _init_one_dataset(name, streaming, shards=None):
    from datasets import load_dataset
    if name not in _TEXT_DATASETS:
        raise ValueError(f'unknown dataset: {name}')
    hf, cfg, _, _ = _TEXT_DATASETS[name]

    # prepared local copy (dolma3_prep.py) wins over the hub repo: no download, and its
    # parts are pre-shuffled across the whole corpus.
    local = _local_files(name)
    if local:
        if shards:
            local = local[:shards]
        print(f'{name}: loading {len(local)} local parquet part(s) from {_LOCAL_DIRS[name]}')
        return load_dataset('parquet', data_files={'train': local}, streaming=streaming)
    if name in _LOCAL_DIRS:
        print(f'{name}: no prepared copy at {_LOCAL_DIRS[name]} -- falling back to the hub '
              f'repo. Hub shards are topic-partitioned: streaming trains on one topic at a '
              f'time and non-streaming fails to cast. See dolma3_prep.py')

    # --shards: restrict a non-streaming load to the first N parquet shards via data_files,
    # so datasets only resolves/reads those files instead of the whole repo.
    data_files = None
    if not streaming and shards:
        files = _shard_files(hf, cfg, shards)
        if files:
            print(f'{name}: --shards {shards} -> loading {len(files)} parquet shard(s)')
            data_files = {'train': files}
        else:
            print(f'{name}: --shards set but no parquet shards found; loading full dataset')

    if data_files is not None:
        # no_checks: skip the split-size verification, which would fail since we load only
        # a subset of the repo's shards (recorded rows != the repo's recorded full split).
        loader = lambda dlc: load_dataset(hf, data_files=data_files, download_config=dlc,
                                          verification_mode='no_checks')
    else:
        loader = lambda dlc: load_dataset(hf, cfg, streaming=streaming, download_config=dlc)

    # Non-streaming eagerly resolves every shard (large corpora have thousands), which
    # can blow past HF's 5000-requests/5-min limit -> 429. Throttle the download workers
    # and retry across rate-limit windows; cached shards persist between tries so each
    # attempt makes forward progress until the cache is complete.
    if streaming:
        return loader(None)
    from datasets import DownloadConfig
    dlc = DownloadConfig(num_proc=4, max_retries=8)
    for attempt in range(1, 1000):
        try:
            return loader(dlc)
        except Exception as e:
            if '429' not in str(e) and 'Too Many Requests' not in str(e):
                raise
            print(f'HF 429 rate-limit on {name} (attempt {attempt}); cached shards kept, '
                  f'cooling down 300s then resuming...')
            time.sleep(300)


def _make_iter(ds, split, streaming, seed, buffer_size=10000):
    split_ds = ds[split] if split in ds else ds['train']
    if streaming:
        return iter(split_ds.shuffle(buffer_size=buffer_size, seed=seed))
    return iter(split_ds.shuffle(seed=seed))


# ── worker ─────────────────────────────────────────────────────────────────────

class _ClipSlot:
    """One batch slot: independent clip stream."""
    __slots__ = ['steps', 'pos', 'caption']

    def __init__(self):
        self.steps   = []
        self.pos     = 0
        self.caption = ''

    def load(self, steps, caption=''):
        self.steps   = steps
        self.pos     = 0
        self.caption = caption

    def done(self):       return self.pos >= len(self.steps)
    def is_start(self):   return self.pos == 0

    def advance(self):
        item     = self.steps[self.pos]
        self.pos += 1
        return item   # (x_byte, y_byte)


def worker(stop, q, datasets, args, mix=None):
    # datasets: {name: hf_dataset}. mix: (names, weights) or None (single dataset).
    if mix is None:
        names, weights = [args.dataset], [1.0]
    else:
        names, weights = mix
    _iters  = {}
    _locks  = {}
    _epochs = {}
    for n in set(names):
        _iters[n]  = _make_iter(datasets[n], _split_for_name(n), args.streaming, args.seed)
        _locks[n]  = threading.Lock()   # HF streaming iterator is not thread-safe
        _epochs[n] = [1]

    def _next_example(name):
        """Thread-safe pull of the next raw example for one dataset (loops on exhaustion)."""
        with _locks[name]:
            try:
                return next(_iters[name])
            except StopIteration:
                _epochs[name][0] += 1
                _iters[name] = _make_iter(datasets[name], _split_for_name(name),
                                          args.streaming, args.seed + _epochs[name][0])
                return next(_iters[name])

    def _build_clip(name, example):
        col = _TEXT_DATASETS[name][3]
        return clip_text(example.get(col, example.get('text', '')))

    slots = [_ClipSlot() for _ in range(args.batch)]

    # Pool of builder threads runs the dataset pull/encode in parallel and parks
    # finished clips in clip_q, so the stepping loop below never blocks on one clip.
    n_builders = max(1, args.workers)
    clip_q     = queue.Queue(maxsize=2 * n_builders)

    # token-proportional dataset sampling: `weights` is the desired *token* mix, but
    # datasets have very different tokens-per-example, so choosing examples directly by
    # `weights` skews the token mix. Track a running mean clip length per dataset and
    # sample examples with weight w/mean_len, so expected tokens per dataset ∝ w.
    _len_lock = threading.Lock()
    _len_sum  = {n: 0.0 for n in set(names)}
    _len_cnt  = {n: 0   for n in set(names)}

    def _mix_weights():
        with _len_lock:
            # bootstrap unseen datasets with mean_len=1 so they still get sampled
            return [w / max(_len_sum[n] / _len_cnt[n] if _len_cnt[n] else 1.0, 1e-9)
                    for n, w in zip(names, weights)]

    def builder():
        rng = random.Random()   # per-clip dataset choice (thread-local instance)
        while not stop.is_set():
            name = rng.choices(names, weights=_mix_weights(), k=1)[0]
            try:
                steps, caption = _build_clip(name, _next_example(name))
            except Exception as e:
                print(f'builder: {e}')
                continue
            if not steps:
                continue
            with _len_lock:                      # update token-length stats for the mix
                _len_sum[name] += len(steps)
                _len_cnt[name] += 1
            _text_tally(len(steps), sample=caption)
            while not stop.is_set():
                try:
                    clip_q.put((steps, caption), timeout=0.5)
                    break
                except queue.Full:
                    pass

    builders = [threading.Thread(target=builder, daemon=True) for _ in range(n_builders)]
    for b in builders:
        b.start()

    while not stop.is_set():
        for slot in slots:
            while slot.done():
                try:
                    steps, caption = clip_q.get(timeout=0.5)
                except queue.Empty:
                    if stop.is_set():
                        return
                    continue
                slot.load(steps, caption)

        x_list, y_list, flag_list = [], [], []
        for slot in slots:
            flag_list.append(slot.is_start())
            x, y = slot.advance()
            x_list.append(x)
            y_list.append(y)

        q.put((x_list, y_list, flag_list))


# ── dataset loading (before CUDA init) ───────────────────────────────────────

if args.vis or args.generate or args.evaluate or args.push:
    hf_dataset = None                 # one-shot modes: --vis/--generate run from the prompt
                                      # token, --evaluate loads HellaSwag itself, --push only
                                      # uploads weights — none of them needs a training dataset
else:
    print('loading dataset...')
    if _mix is not None:
        _ds_names  = sorted(set(_mix[0]))
        hf_dataset = {n: _init_one_dataset(n, args.streaming, args.shards) for n in _ds_names}
        print('mix:', ', '.join(f'{n}:{w:.3f}' for n, w in zip(_mix[0], _mix[1])))
    else:
        hf_dataset = {args.dataset: _init_one_dataset(args.dataset, args.streaming, args.shards)}
    print('dataset ready')

# ── model instantiation ───────────────────────────────────────────────────────

if args.pretrained:
    # Hub checkpoint: config.json (written by save_pretrained) supplies the architecture,
    # so the model is built from it rather than from the --context/--n_hidden/... args.
    # Those args are then synced to what was actually built, since the rest of the script
    # reads them (DFF shapes, summaries, logged ARGS line).
    if not hasattr(VLAModel, 'from_pretrained'):
        print('ERROR: --pretrained requires huggingface_hub (pip install huggingface_hub)')
        raise SystemExit(1)
    print(f'loading {args.hub_repo} from the hub...')
    model = VLAModel.from_pretrained(args.hub_repo, revision=args.revision)
    _cfg  = getattr(model, '_hub_mixin_config', None) or {}
    for _k, _a in (('context', 'S'), ('n_hidden', 'n_hidden'), ('c_text', 'c_text'),
                   ('n_layers', 'n_layers'), ('depth', 'depth'), ('kernel', 'kernel')):
        if _a in _cfg:
            setattr(args, _k, _cfg[_a])
    print('pretrained config', _cfg)
    # The ARGS line was logged before the hub config was known, so it still shows the CLI
    # arch defaults. Re-log the corrected args, or the log would misattribute the run.
    print('ARGS (pretrained)', args)
    with open(args.log, 'a') as f:
        print('ARGS', args, file=f)
else:
    model = VLAModel(
        n_hidden  = args.n_hidden,
        depth     = args.depth,
        kernel    = args.kernel,
        S         = args.context,
        c_text    = args.c_text,
        n_layers  = args.n_layers,
    )

    if _loaded_ckpt is not None:
        model.load_state_dict(_loaded_ckpt['state_dict'])
        del _loaded_ckpt

_dff_shape = (1, args.n_hidden, args.context, args.context)
def _summ(title, mod, **kw):
    # --generate/--vis have no log file, so their summaries would only be terminal noise in
    # front of the sample / the plot window. --verbose asks for them anyway.
    if (args.generate or args.vis or args.push) and not args.verbose:
        return
    s = str(torchinfo.summary(mod, col_names=['input_size', 'output_size', 'num_params'],
                              verbose=0, **kw))
    block = f'── {title} ──\n{s}'
    print(block)
    with open(args.log, 'a') as f:                  # mirror to log, each line prefixed INFO
        for line in block.splitlines():
            print(f'INFO {line}', file=f)

# Summaries follow the data flow: tok_embed → text_proj (token grid → n_hidden,
# summed into the DFF) → context (recurrent state update) → decoder → logits.
_summ('tok_embed', model.tok_embed, input_data=torch.zeros(1, dtype=torch.long))
_summ('text_proj', model.text_proj, input_size=(1, model.c_text, args.context, args.context))
_summ('context (layer 1)' if model.n_layers > 1 else 'context',
      model.context, input_size=_dff_shape)
if model.deep_context is not None:
    _summ(f'deep_context x{model.n_layers - 1} (layers 2..{model.n_layers})',
          model.deep_context[0], input_size=_dff_shape)
_summ('decoder', model.decoder, input_size=_dff_shape)

model = model.to(args.device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 'M trainable params')

if args.push:
    # one-shot: save the loaded checkpoint in HuggingFace format (model.safetensors +
    # config.json, from PyTorchModelHubMixin) and upload it to --hub_repo, then exit.
    # Needs a write token in the environment (huggingface-cli login, or HF_TOKEN).
    if args.load is None and not args.pretrained:
        print('ERROR: --push requires --load (nothing to push but random weights)')
        raise SystemExit(1)
    if not hasattr(model, 'push_to_hub'):
        print('ERROR: --push requires huggingface_hub (pip install huggingface_hub)')
        raise SystemExit(1)
    local = os.path.basename(args.hub_repo)
    print(f'saving {args.load or args.hub_repo} to ./{local} ...')
    model.to('cpu').save_pretrained(local)
    print(f'pushing to https://huggingface.co/{args.hub_repo} ...')
    # private=None (not False) when the flag is unset: that lets the hub apply the org's
    # default visibility for a new repo. Ignored if the repo already exists.
    model.push_to_hub(args.hub_repo, private=args.private or None)
    print('done')
    raise SystemExit(0)

if args.generate:
    # one-shot text generation: prompt = START + args.prompt, print the decoded
    # continuation, then exit. No dataset, worker, optimizer or training loop.
    # Same seeding as --vis and the training-loop sampler, and the same [START]+text
    # framing the model is trained on, so the output always opens with <STX>.
    if args.load is None and not args.pretrained:
        print('WARNING: --generate without --load/--pretrained is running an untrained model')
    prompt = [START] + list(args.prompt.encode('utf-8', errors='replace'))
    out    = model.generate(prompt, args.n, temperature=args.temperature)
    print(_printable(bytes(prompt + out).decode('utf-8', errors='replace')))
    raise SystemExit(0)

# ── hellaswag evaluation ──────────────────────────────────────────────────────
# HellaSwag (https://huggingface.co/datasets/Rowan/hellaswag) gives a context and four
# candidate endings, one of which is correct. A language model is scored zero-shot by
# completion likelihood: feed the context, then score each ending and pick the most likely.
# Accuracy compares against https://rowanzellers.com/hellaswag/ (random = 25%).
#
# This is a byte model, so "scoring an ending" means summing the log-probability of each of
# its UTF-8 bytes, teacher-forced, with the DFF state carried over from the context. Two
# scores are reported, the usual pair for zero-shot LM eval:
#
#   acc_norm -- mean log-prob per ending byte (length-normalized). The headline number: it
#               does not penalize long endings, and is what zero-shot results are quoted with.
#   acc      -- total log-prob of the ending (un-normalized).

def _hs_preprocess(text):
    """The standard HellaSwag text cleanup (EleutherAI lm-evaluation-harness,
    lm_eval/tasks/hellaswag/utils.py). The bracket tags -- [header] [title] [step]
    [substeps] -- are artifacts of the WikiHow portion of the dataset and are stripped
    before scoring; ~68% of contexts and ~65% of endings contain them."""
    text = text.strip()
    text = text.replace(' [title]', '. ')
    text = re.sub('\\[.*?\\]', '', text)
    text = text.replace('  ', ' ')
    return text


def _hs_render(example):
    """Rendered the way zero-shot HellaSwag is rendered for a left-to-right LM, matching
    the lm-evaluation-harness so acc_norm is comparable to published numbers:

        prompt     = activity_label + ': ' + ctx_a + ' ' + ctx_b.capitalize()
        completion = ' ' + ending

    both run through _hs_preprocess. Note the dataset's own 'ctx' field is just
    ctx_a + ' ' + ctx_b -- it carries no activity label and leaves ctx_b uncapitalized --
    so the parts are recombined here rather than used directly. Bytes, not tokens."""
    ctx     = example['ctx_a'] + ' ' + example['ctx_b'].capitalize()
    ctx     = _hs_preprocess(example['activity_label'] + ': ' + ctx)
    ctx     = [START] + list(ctx.encode('utf-8', errors='replace'))
    endings = [list((' ' + _hs_preprocess(e)).encode('utf-8', errors='replace'))
               for e in example['endings']]
    return ctx, endings


def _hs_pad(seqs, device):
    """Right-pad byte lists to a [B, Lmax] long tensor (+ their [B] lengths)."""
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long, device=device)
    out  = torch.full((len(seqs), int(lens.max())), NULL, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    return out, lens


@torch.no_grad()
def _hs_score(model, examples, device):
    """Log-prob of each ending under the model.
    Returns (sum_lp [E, 4], n_bytes [E, 4]): total ending log-prob and its byte count."""
    E        = len(examples)
    rendered = [_hs_render(e) for e in examples]

    # Context phase: run each context once, freezing the DFF state of sequences that have
    # already ended, so every example finishes on the step that consumes its own last
    # context byte. `logits` then predicts the first byte of the ending.
    ctx, ctx_len = _hs_pad([c for c, _ in rendered], device)
    dff    = [torch.zeros(E, model.n_hidden, model.S, model.S, device=device)
              for _ in range(model.n_layers)]
    logits = torch.zeros(E, 256, device=device)
    for t in range(int(ctx_len.max())):
        live = (t < ctx_len).view(E, 1, 1, 1)
        stack, out = model._layers(dff, model._tok_grid(ctx[:, t]))
        dff    = [torch.where(live, s, d) for s, d in zip(stack, dff)]
        logits = torch.where(live.view(E, 1), model.decoder(out), logits)

    # Ending phase: fork the context state to the four candidates, so the (long) context is
    # not recomputed four times. Byte e[t] is scored under the logits produced by everything
    # before it, then fed in to produce the logits for e[t+1]. Sequences past their end keep
    # stepping (their state is dead) but contribute nothing: `live` masks them out.
    end, end_len = _hs_pad([e for _, ends in rendered for e in ends], device)
    dff    = [d.repeat_interleave(4, dim=0) for d in dff]
    logits = logits.repeat_interleave(4, dim=0)
    sum_lp = torch.zeros(4 * E, device=device)
    for t in range(int(end_len.max())):
        live    = (t < end_len)
        logp    = F.log_softmax(logits, dim=-1).gather(1, end[:, t:t + 1]).squeeze(1)
        sum_lp += torch.where(live, logp, torch.zeros_like(logp))
        if t + 1 == end.shape[1]:
            break
        stack, out  = model._layers(dff, model._tok_grid(end[:, t]))
        dff, logits = stack, model.decoder(out)

    return sum_lp.view(E, 4), end_len.view(E, 4)


if args.evaluate:
    # one-shot: score the HellaSwag split, print accuracy, exit. --batch is examples per
    # batch (each becomes 4 sequences in the ending phase), --monitor the progress interval.
    # Everything printed also goes to log/eval.<date> (with the ARGS line already written
    # there), so a score stays attributable to the checkpoint and split that produced it and
    # runs can be compared after the fact.
    def _elog(s):
        print(s, flush=True)
        with open(args.log, 'a') as f:
            print(s, file=f)

    if args.load is None and not args.pretrained:
        _elog('WARNING: --evaluate without --load/--pretrained is scoring an untrained model')
    from datasets import load_dataset
    _elog(f'loading hellaswag ({args.split})...')
    hs = load_dataset('Rowan/hellaswag', split=args.split)
    if args.limit is not None:
        hs = hs.select(range(min(args.limit, len(hs))))
    _elog(f'EVAL checkpoint {args.load or (args.hub_repo + "@" + (args.revision or "main"))} '
          f'split {args.split} examples {len(hs)}')

    model.eval()
    n = hit = hit_norm = 0
    t0 = time.time()
    nxt = 0   # next example count to report at: progress every --monitor examples, rounded
              # up to the batch that crosses it (so any --batch/--monitor pairing reports)

    for bi in range(0, len(hs), args.batch):
        batch  = [hs[j] for j in range(bi, min(bi + args.batch, len(hs)))]
        labels = torch.tensor([int(e['label']) for e in batch], device=args.device)

        sum_lp, n_bytes = _hs_score(model, batch, args.device)
        avg_lp = sum_lp / n_bytes.clamp(min=1)

        n        += len(batch)
        hit      += (sum_lp.argmax(dim=1) == labels).sum().item()
        hit_norm += (avg_lp.argmax(dim=1) == labels).sum().item()

        if n >= nxt:
            _elog(f'{n:6d}/{len(hs)}  acc_norm {hit_norm / n:.4f}  acc {hit / n:.4f}  '
                  f'({n / (time.time() - t0):.1f} ex/s)')
            nxt = n + args.monitor

    _elog(f'\nHellaSwag {args.split}: {n} examples, {time.time() - t0:.0f}s')
    _elog(f'  acc_norm  {hit_norm / n:.4f}   ({hit_norm}/{n})   <- length-normalized (headline)')
    _elog(f'  acc       {hit / n:.4f}   ({hit}/{n})')
    _elog(f'  random    0.2500')
    print(f'\nlog {args.log}')
    raise SystemExit(0)

# ── vis setup ─────────────────────────────────────────────────────────────────

_vis_exit        = False
_vis_stats_line  = 'initializing...'
_VIS_WIN         = 80


def _vc(b):
    if b == START: return STX_TEXT
    if b == END:   return ETX_TEXT
    if b == NULL:  return '_'
    c = chr(b) if 32 <= b < 127 else '\xb7'
    return '\xb7' if c in '$\\' else c


if args.vis:
    plt.style.use('dark_background')
    plt.ion()
    _vis_fig, _vis_axes_grid = plt.subplots(1, 3, figsize=(12, 4))
    _vis_fig.subplots_adjust(top=0.78, bottom=0.10, left=0.04, right=0.98,
                              hspace=0.1, wspace=0.08)
    _vis_ax_dff, _vis_ax_zero, _vis_ax_prob = _vis_axes_grid
    _vis_ax_dff.set_title('DFF std / {:d} ch'.format(model.n_hidden), fontsize=8, pad=2)
    _vis_ax_zero.set_title('DFF zeros / {:d} ch'.format(model.n_hidden), fontsize=8, pad=2)
    _vis_ax_prob.set_title('P(next token)', fontsize=8, pad=2)
    # row/column dimension labels (e.g. "7"): DFF state is S×S, prob is 16×16 (256 bytes).
    _vis_ax_dff.set_xlabel(str(model.S),  fontsize=7); _vis_ax_dff.set_ylabel(str(model.S),  fontsize=7)
    _vis_ax_zero.set_xlabel(str(model.S), fontsize=7); _vis_ax_zero.set_ylabel(str(model.S), fontsize=7)
    _vis_ax_prob.set_xlabel('16',         fontsize=7); _vis_ax_prob.set_ylabel('16',         fontsize=7)
    _vis_img_dff  = None
    _vis_img_zero = None
    _vis_img_prob = None
    _vis_dff_vmax = None   # slow-tracking color scale for the DFF-std panel (see set_clim below)
    _vis_header   = _vis_fig.text(0.01, 0.99, _vis_stats_line,
                                   family='monospace', fontsize=7,
                                   va='top', ha='left')

    # header font is in points (fixed); scale it with window width so the monospace
    # ticker keeps filling the same fraction of the figure when the window resizes.
    _vis_hdr_base_pt = 7
    _vis_w0          = _vis_fig.get_size_inches()[0] * _vis_fig.dpi
    def _scale_header(event=None):
        w = event.width if event is not None else _vis_w0
        _vis_header.set_fontsize(max(_vis_hdr_base_pt * w / _vis_w0, 4.0))
    _vis_fig.canvas.mpl_connect('resize_event', _scale_header)
    _scale_header()

    def _on_key(event):
        global _vis_exit
        if event.key in ('x', 'X'):
            _vis_exit = True
    _vis_fig.canvas.mpl_connect('key_press_event', _on_key)

    # ── live text-generation visualization (no training) ───────────────────────
    # Feed the prompt (START, plus --prompt if given) then autoregressively sample,
    # rendering the DFF-std and P(next token) panels for every generated byte. The header
    # shows a stats line and the generated text. Read-only: no dataset, optimizer or worker.
    model.eval()
    _dev    = next(model.parameters()).device
    _prompt = [START] + list(args.prompt.encode('utf-8', errors='replace'))
    _ptext  = ''.join(_vc(b) for b in _prompt)  # rendered prompt, echoed at each episode start
    _queue  = list(_prompt)   # remaining prompt bytes to inject as new tokens
    _b      = _queue.pop(0)   # current token being injected
    _ticker = _ptext          # ticker-tape display buffer (trailing _VIS_WIN chars)
    _ep_len = 0               # chars generated in the current episode (length cap)
    _nstep  = 0
    dff     = [torch.zeros(1, model.n_hidden, model.S, model.S, device=_dev)
               for _ in range(model.n_layers)]

    while not _vis_exit:
        with torch.no_grad():
            bi   = torch.tensor([_b], dtype=torch.long, device=_dev)
            tok  = model.tok_embed(bi).view(1, model.c_text, 1, 1).expand(-1, -1, model.S, model.S)
            stack, new_ctx = model._layers(dff, tok)               # last-layer output
            dff     = [d.detach().clone() for d in stack]
            logits  = model.decoder(new_ctx)
        _nstep += 1

        prob = F.softmax(logits[0] / max(args.temperature, 1e-6), dim=-1)

        if _queue:
            _b = _queue.pop(0)             # still draining the prompt
        else:
            nxt = int(torch.multinomial(prob, 1).item())
            if nxt == END or _ep_len >= _VIS_WIN:
                if nxt == END:
                    _ticker += _vc(END)    # mark the episode break in the ticker
                dff    = [torch.zeros(1, model.n_hidden, model.S, model.S, device=_dev)
                          for _ in range(model.n_layers)]
                _queue = list(_prompt)     # episode over: reset state, restart prompt
                _b     = _queue.pop(0)
                _ep_len = 0
                _ticker += _ptext          # new episode opens with <STX> again
            else:
                if nxt != NULL:
                    _ticker += _vc(nxt)
                    _ep_len += 1
                _b = nxt
            _ticker = _ticker[-_VIS_WIN:]  # slide + bound the visible window (ticker tape)

        # ── render ──────────────────────────────────────────────────────────
        # DFF-std panel shows the last layer's state (the one feeding the decoder).
        dff_std_map = new_ctx[0].cpu().std(dim=0).numpy()           # [S, S]
        # per-location sparsity: how many of the n_hidden channels are exactly 0 here
        zero_map    = (new_ctx[0] == 0).sum(dim=0).cpu().numpy()     # [S, S], 0..n_hidden
        prob_img    = prob.detach().cpu().numpy().reshape(16, 16)    # 256 bytes → 16×16
        _dff_mean, _dff_std, _dff_max, _ = _dff_stats(dff)
        _vis_header.set_text(
            ('step {:8d}  layers {:d}  dff_std {:6.4f}  dff_max {:7.3f}  p_max {:5.3f}  T {:.2f}').format(
                _nstep, model.n_layers, _dff_std, _dff_max,
                float(prob.max()), args.temperature)
            + '\n' + 'gen: ' + _ticker
        )

        if _vis_img_dff is None:
            _vis_img_dff = _vis_ax_dff.matshow(dff_std_map, cmap=args.cmap)
            _vis_ax_dff.set_xticks([]); _vis_ax_dff.set_yticks([])
        else:
            _vis_img_dff.set_data(dff_std_map)
        _fmax = float(dff_std_map.max())
        _vis_dff_vmax = _fmax if _vis_dff_vmax is None else max(_fmax, 0.98 * _vis_dff_vmax)
        _vis_img_dff.set_clim(0.0, _vis_dff_vmax or 1e-9)

        # fixed 0..n_hidden scale: absolute zero-count is more readable than an autoscale.
        if _vis_img_zero is None:
            _vis_img_zero = _vis_ax_zero.matshow(zero_map, cmap=args.cmap,
                                                 vmin=0, vmax=model.n_hidden)
            _vis_ax_zero.set_xticks([]); _vis_ax_zero.set_yticks([])
        else:
            _vis_img_zero.set_data(zero_map)

        if _vis_img_prob is None:
            _vis_img_prob = _vis_ax_prob.matshow(prob_img, cmap=args.cmap, vmin=0, vmax=1)
            _vis_ax_prob.set_xticks([]); _vis_ax_prob.set_yticks([])
            # overlay printable-ASCII (32..126) glyphs in white at each byte's 16×16 cell;
            # byte b is row b//16, col b%16 (matches the prob_img reshape order).
            _vis_prob_txts = [
                _vis_ax_prob.text(b % 16, b // 16, chr(b), color='white',
                                  ha='center', va='center', family='monospace')
                for b in range(32, 127)
            ]
            # font size is in points (fixed), so rescale to the cell's pixel size on resize.
            def _scale_prob_text(event=None):
                bb = _vis_ax_prob.get_window_extent()
                cell_px = min(bb.width, bb.height) / 16.0   # one of 16×16 cells
                pts = max(cell_px * 72.0 / _vis_fig.dpi * 0.55, 1.0)
                for _t in _vis_prob_txts:
                    _t.set_fontsize(pts)
            _vis_fig.canvas.mpl_connect('resize_event', _scale_prob_text)
            _scale_prob_text()
        else:
            _vis_img_prob.set_data(prob_img)
        _vis_img_prob.set_clim(0, float(prob_img.max()) or 1e-9)

        _vis_fig.canvas.draw()        # synchronous: render every step (draw_idle coalesces
        _vis_fig.canvas.flush_events()  # frames when sleep blocks the GUI loop → skips steps)
        time.sleep(args.delay)

    raise SystemExit(0)

# ── optimizer ─────────────────────────────────────────────────────────────────

_trainable = [p for p in model.parameters() if p.requires_grad]

# --lr_mult scales the base lr per layer: layer 1 keeps base lr, each deeper layer is
# multiplied by an extra factor of --lr_mult. Layer 1 = {tok_embed, text_proj, context};
# deep layers 2..N = their (deep_context + deep_proj) unit; the decoder reads the last
# layer's output, so it rides the deepest lr. Groups carry a per-group 'lr' (its initial_lr)
# which LambdaLR then scales by the shared schedule factor. With --lr_mult 1.0 we pass the
# flat param list so the optimizer/checkpoint layout is byte-identical to before.
def _lr_scaled_groups(model, base_lr, mult):
    N      = model.n_layers
    layers = [[] for _ in range(N)]
    layers[0] += list(model.tok_embed.parameters())
    layers[0] += list(model.text_proj.parameters())
    layers[0] += list(model.context.parameters())
    if model.deep_context is not None:
        for li in range(N - 1):
            layers[li + 1] += list(model.deep_context[li].parameters())
            layers[li + 1] += list(model.deep_proj[li].parameters())
    layers[-1] += list(model.decoder.parameters())
    return [{'params': [p for p in g if p.requires_grad], 'lr': base_lr * (mult ** i)}
            for i, g in enumerate(layers)]

# Only build per-layer optimizer groups when lr_mult scaling is active; otherwise pass the
# flat param list so the optimizer/checkpoint layout is byte-identical to before.
_params = _lr_scaled_groups(model, args.lr_max, args.lr_mult) if args.lr_mult != 1.0 else _trainable

# Per-layer summary: trainable params and lr, shown when --lr_mult scales the layers. Param
# counts come from the same grouping as the lr scaling (layer 1 includes tok_embed/text_proj;
# the last includes the decoder), so the columns add up to the trainable total.
if args.lr_mult != 1.0:
    _groups = _lr_scaled_groups(model, args.lr_max, args.lr_mult)
    _pc  = [sum(p.numel() for p in g['params']) for g in _groups]
    _lrs = [g['lr'] for g in _groups]
    _yw  = max(len('layer'), 1 + len(str(len(_groups))))   # 'L' + index, vs header label
    _pw  = max(len('params'), *(len(f'{p:,}')   for p in _pc))
    _lw  = max(len('lr'),     *(len(f'{lr:.9f}') for lr in _lrs))
    print(f'per-layer (lr_mult {args.lr_mult}):')
    print(f'  {"layer":<{_yw}}  {"params":>{_pw}}  {"lr":>{_lw}}')
    for _i in range(len(_groups)):
        print(f'  {"L" + str(_i + 1):<{_yw}}  {_pc[_i]:>{_pw},}  {_lrs[_i]:>{_lw}.9f}')

# Route lr/weight_decay/momentum to each optimizer's supported knobs: Rprop takes
# neither weight_decay nor momentum; Adagrad takes weight_decay but no momentum.
if args.opt == 'sgd':
    optimizer = torch.optim.SGD(_params, lr=args.lr_max,
                                momentum=args.momentum, weight_decay=args.weight_decay)
elif args.opt == 'rmsprop':
    optimizer = torch.optim.RMSprop(_params, lr=args.lr_max,
                                    momentum=args.momentum, weight_decay=args.weight_decay, centered=True)
elif args.opt == 'rprop':
    optimizer = torch.optim.Rprop(_params, lr=args.lr_max)
elif args.opt == 'adagrad':
    optimizer = torch.optim.Adagrad(_params, lr=args.lr_max,
                                    weight_decay=args.weight_decay)
elif args.opt == 'adamw':
    # --momentum drives beta2 (second-moment decay); beta1 fixed at 0.9.
    optimizer = torch.optim.AdamW(_params, lr=args.lr_max,
                                  betas=(0.9, args.momentum),
                                  weight_decay=args.weight_decay)

# ── scheduler ─────────────────────────────────────────────────────────────────
# Every --schedule oscillates between --lr_min and --lr_max. LambdaLR multiplies the
# optimizer base lr (--lr_max), so we express the absolute lr as a factor relative to it.
_lo, _hi = args.lr_min, args.lr_max
_warm    = max(1, args.lr_warmup)
_period  = max(1, args.lr_period)

def _lr_factor(step):
    if args.schedule == 'const':                       # fixed lr_max
        lr = _hi
    elif args.schedule == 'linear':                    # ramp lr_min -> lr_max over lr_period, then hold
        lr = _lo + (_hi - _lo) * min(step / _period, 1.0)
    elif args.schedule == 'triangle':                  # up over lr_warmup, down over lr_period, then hold lr_min
        if step < _warm:
            lr = _lo + (_hi - _lo) * (step / _warm)
        elif step < _warm + _period:
            lr = _hi + (_lo - _hi) * ((step - _warm) / _period)
        else:
            lr = _lo
    else:                                              # cosine: cyclic lr_min <-> lr_max, lr_period per cycle
        phase = (1.0 - math.cos(2.0 * math.pi * (step % _period) / _period)) / 2.0   # 0->1->0
        lr = _lo + (_hi - _lo) * phase
    return lr / _hi

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_factor)

# On resume, fast-forward the schedule to the checkpoint's step so the lr at the first
# continued step matches where the previous run left off (LambdaLR is stateless in step).
# Update both the live optimizer lr and the scheduler's cached last_lr (what get_last_lr
# reports in the STEP line).
if _start_step:
    scheduler.last_epoch = _start_step
    _lrs = [_base * _lr_factor(_start_step) for _base in scheduler.base_lrs]
    for _g, _lr in zip(optimizer.param_groups, _lrs):
        _g['lr'] = _lr
    scheduler._last_lr = _lrs

# ── data thread ───────────────────────────────────────────────────────────────

stop  = threading.Event()
q     = queue.Queue(maxsize=args.batch)
w     = threading.Thread(target=worker,
                          args=[stop, q, hf_dataset, args, _mix],
                          daemon=False)
w.start()

# ── helpers ───────────────────────────────────────────────────────────────────

# ── gradient / explosion instrumentation ──────────────────────────────────────

def _gnorm(params):
    return sum(p.grad.detach().norm(2).item() ** 2
               for p in params if p.grad is not None) ** 0.5


def _wnorm(params):
    return sum(p.detach().norm(2).item() ** 2 for p in params) ** 0.5


def _gnorm_all(model):
    return sum(p.grad.detach().norm(2).item() ** 2
               for p in model.parameters()
               if p.grad is not None and p.requires_grad) ** 0.5


def _param_groups_named(model):
    """Trainable params split by submodule, for per-module norm reporting."""
    g = {'emb':  list(model.tok_embed.parameters()),
         'tprj': list(model.text_proj.parameters()),
         'ctx':  list(model.context.parameters()),
         'dec':  list(model.decoder.parameters())}
    if model.deep_context is not None:   # stacked layers 2..N (--n_layers)
        g['deep'] = list(model.deep_context.parameters()) + list(model.deep_proj.parameters())
    return g


def _diag(model):
    return ' '.join(f'{n}:g{_gnorm(ps):.1e}/w{_wnorm(ps):.1e}'
                    for n, ps in _param_groups_named(model).items())

# ── training loop ─────────────────────────────────────────────────────────────

larr, garr = [], []
i = _start_step        # continue the step count from the resumed checkpoint (0 if fresh)

try:
    while True:
        # ── checkpoint / generation sample ───────────────────────────────────
        if (i % args.checkpoint) == 0:
            _finite = all(torch.isfinite(p).all() for p in model.state_dict().values()
                          if p.is_floating_point())
            if _finite:
                # Embed the cumulative log so a later --load can prepend it and chart.py
                # sees one uninterrupted sequence across resumes. args.log already begins
                # with any log carried in from the checkpoint we resumed from.
                _log_text = ''
                if args.log and args.log != os.devnull and os.path.exists(args.log):
                    with open(args.log) as _lf:
                        _log_text = _lf.read()
                torch.save({'saved_args': vars(args), 'state_dict': model.state_dict(),
                            'log': _log_text, 'step': i}, args.save)
            else:
                print(f'WARNING: non-finite params at step {i}; skipping checkpoint save')

            # text throughput for this checkpoint interval (examples/tokens consumed)
            _te, _tt, _tsample = _text_drain()
            if _te:
                d = f'TEXT step {i:10} examples {_te} tokens {_tt}'
                print(d)
                with open(args.log, 'a') as f:
                    print(d, file=f)

            model.eval()
            prompt = [START] + list(args.prompt.encode('utf-8', errors='replace'))
            out    = model.generate(prompt, args.n, temperature=args.temperature)
            gen_text = _printable(bytes(prompt + out).decode('utf-8', errors='replace'))
            lines = [f'GEN: {gen_text}']
            if _tsample:                       # a recent training-text example
                lines.append(f'TXT-SAMPLE: {_printable(_tsample[:120])}')
            print('\n' + '\n'.join(lines) + '\n')
            with open(args.log, 'a') as f:
                print('\n' + '\n'.join(lines) + '\n', file=f)

        # ── fetch batch ───────────────────────────────────────────────────────
        x_list, y_list, flag_list = q.get()
        x    = torch.tensor(x_list,    dtype=torch.long).to(args.device)
        y    = torch.tensor(y_list,    dtype=torch.long).to(args.device)
        flag = torch.tensor(flag_list, dtype=torch.bool).to(args.device)

        # ── train step ────────────────────────────────────────────────────────
        dff_prev = model.dff                                    # save pre-step DFF
        model.train()
        logits, loss = model(x, targets=y, flag=flag)

        # halt immediately on non-finite loss — do NOT backward/step/save, so the
        # last good checkpoint on disk is preserved.
        if not torch.isfinite(loss):
            print(f'\n*** HALT: non-finite loss at step {i} ***')
            print(f'  loss     = {loss.item()}')
            print(f'  dff_max  = {_dff_absmax(model.dff)}')
            print(f'  logit_max= {logits.detach().abs().max().item()}')
            print(f'  grads(prev step): {_diag(model)}')
            break

        loss.backward()

        total_norm = _gnorm_all(model)                         # grad norm BEFORE the step
        garr.append(total_norm)

        # halt before the optimizer corrupts weights if grads went non-finite
        if not math.isfinite(total_norm):
            print(f'\n*** HALT: non-finite grad at step {i} ***')
            print(f'  dff_max  = {_dff_absmax(model.dff)}')
            print(f'  logit_max= {logits.detach().abs().max().item()}')
            print(f'  per-module g(rad)/w(eight): {_diag(model)}')
            break

        optimizer.step()
        model.dff = dff_prev                                    # restore so eval recomputes same step
        model.eval()
        with torch.no_grad():
            _, _ = model(x, targets=y, flag=flag)              # recompute DFF using updated model

        # ── monitor ───────────────────────────────────────────────────────────
        larr.append(loss.item())

        if (i % args.monitor) == 0:
            _dmean, _dstd, _dmax, _dzero = _dff_stats(model.dff)
            s = ('STEP {:10} wall {} loss {:12.9f} grad {:12.6f} '
                 'lr {:10.9f} dff_mean {:12.5f} dff_std {:12.5f} dff_max {:11.3f} '
                 'dff_zeros {:8.5f}').format(
                i, datetime.datetime.now(),
                np.mean(larr[-args.monitor:]),
                np.mean(garr[-args.monitor:]),
                scheduler.get_last_lr()[0],
                _dmean, _dstd, _dmax, _dzero,
            )
            print(s)
            with open(args.log, 'a') as f:
                print(s, file=f)

            # bound monitor history so host RAM stays flat over long runs (these are
            # only ever read as [-monitor:]; without this they grow once per step forever)
            for _a in (larr, garr):
                if len(_a) > args.monitor:
                    del _a[:-args.monitor]

        scheduler.step()
        optimizer.zero_grad()
        i += 1

        if args.run_steps is not None and (i - _start_step) >= args.run_steps:
            break   # --run_steps caps steps for this invocation, not the global count

except KeyboardInterrupt:
    pass

# ── shutdown ──────────────────────────────────────────────────────────────────

print('\nSTOPPING THREADS')
stop.set()
while not q.empty():
    q.get()
w.join()
print('EXIT MAIN')

# Hard-exit after our work is flushed. Streaming datasets keep native aiohttp
# threads alive whose teardown races Python finalization (PyGILState_Release
# crash); os._exit skips finalization entirely. Everything is already saved.
import sys
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)
