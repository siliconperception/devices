#!/usr/bin/env python3
# vla.py -- Vision-Language-Audio CNN model
# Silicon Perception Inc.
#
# Recurrent CNN core from lang.py (ContextCNN DFF loop + DecoderCNN) conditioned
# at 100 Hz by audio (log-mel → conv) and image (frozen ResNet-18 layer4) feature
# maps. The current token is the always-on 100 char/s text stream. Modalities are
# folded into the projector input via --cond {add | concat}.
#
#   M0  text-only scaffold (tiny|c4|web)
#   M1  audio (LibriSpeech): MelAudioEncoder + clip_librispeech            ← here
#   M2  image (CC3M): ImageEncoder wired; clip_image_caption pending
#   M3  video (WebVid): pending
#
# Vocabulary: 256 (raw bytes). No attention, no positional encoding.

import math
import torch ; print('torch', torch.__version__)
import torch.nn as nn
from torch.nn import functional as F
import torchaudio
import torchinfo
import numpy as np
import argparse
import os
import queue
import threading
import subprocess
import datetime
import time
import random
import matplotlib.pyplot as plt

NULL      = 0x00
START     = 0x02
END       = 0x03
IMAGE_CH  = 512   # ResNet-18 layer4 channel count

# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# architecture
parser.add_argument('--context',       default=7,     type=int,
                    help='spatial size S of the DFF grid (S×S)')
parser.add_argument('--n_hidden',      default=512,   type=int,
                    help='ContextCNN channel depth')
parser.add_argument('--depth',         default=7,     type=int,
                    help='ContextCNN conv layer count')
parser.add_argument('--kernel',        default=3,     type=int,
                    help='conv kernel size (odd integer)')
parser.add_argument('--residual',      default=False, action=argparse.BooleanOptionalAction,
                    help='per-layer residual x = 0.5*x + layer(x) in ContextCNN (default off; --residual to enable)')
parser.add_argument('--cond',          default='add', choices=['add', 'concat'],
                    help='modality conditioning: add (project→sum into DFF) | '
                         'concat (channel cat → 1×1 adapter)')
parser.add_argument('--norm',          default='none', choices=['none', 'group', 'batch'],
                    help='normalization in ContextCNN conv blocks (stabilizes deep/recurrent)')
parser.add_argument('--state_norm',    default='none', choices=['none', 'tanh', 'rms'],
                    help='bound the recurrent DFF state each step (tanh | rms)')
# audio / video
parser.add_argument('--no_image',      default=False, action='store_true',
                    help='disable the image encoder entirely')
parser.add_argument('--no_audio',      default=False, action='store_true',
                    help='disable the audio encoder entirely')
parser.add_argument('--c_audio',       default=64,    type=int,
                    help='audio encoder output channels')
parser.add_argument('--n_mels',        default=64,    type=int,
                    help='mel filterbank bins')
parser.add_argument('--fps',           default=100,   type=int,
                    help='master frame rate in Hz (step rate; window stride / frame selection)')
parser.add_argument('--mel_hop',       default=None,  type=int,
                    help='mel spectrogram hop_length in samples (intra-window audio resolution); '
                         'default = 10ms = work_sr/100, independent of --fps')
parser.add_argument('--audio_work_sr', default=16000, type=int,
                    help='internal working sample rate')
parser.add_argument('--audio_window',  default=1600,  type=int,
                    help='samples per frame fed to the audio encoder (sliding window)')
# training
parser.add_argument('--dataset',       default='tiny',
                    help='tiny | c4 | web | librispeech | cc3m | webvid')
parser.add_argument('--mix',           default=None,
                    help='joint multi-dataset training, "name:weight,..." e.g. '
                         '"webvid:0.5,web:0.5"; overrides --dataset, samples one dataset '
                         'per clip, auto-enables the encoders the mix needs')
parser.add_argument('--max_clip_steps', default=None, type=int,
                    help='truncate each clip to this many steps (token-balance long clips; '
                         'for webvid the caption is re-spread over the capped length)')
parser.add_argument('--streaming',     default=False, action='store_true')
parser.add_argument('--batch',         default=32,    type=int)
parser.add_argument('--workers',       default=12,    type=int,
                    help='parallel clip-builder threads (decode/download); raise for cc3m image streaming')
parser.add_argument('--steps',         default=None,  type=int)
parser.add_argument('--learning_rate', default=3e-4,  type=float)
parser.add_argument('--opt',           default='adamw')
parser.add_argument('--beta1',         default=0.9,   type=float)
parser.add_argument('--beta2',         default=0.95,  type=float)
parser.add_argument('--weight_decay',  default=0.01,  type=float)
parser.add_argument('--clip',          default=None,  type=float)
parser.add_argument('--grad_stop',     default=None,  type=float,
                    help='halt if total grad norm exceeds this (early explosion guard)')
parser.add_argument('--debug',         default=False, action='store_true',
                    help='print per-module grad/weight norms and activation maxes each monitor')
parser.add_argument('--schedule',      default='decay')
parser.add_argument('--warmup',        default=1000,  type=int)
parser.add_argument('--period',        default=10000, type=int)
parser.add_argument('--start_factor',  default=0.01,  type=float)
# I/O
parser.add_argument('--load',          default=None)
parser.add_argument('--save',          default='checkpoint.pt')
parser.add_argument('--checkpoint',    default=1000,  type=int)
parser.add_argument('--log',           default=None)
parser.add_argument('--monitor',       default=10,    type=int)
parser.add_argument('--generate',      default=False, action='store_true',
                    help='print generation sample at each checkpoint; skip file saves')
parser.add_argument('--n',             default=200,   type=int,
                    help='tokens to generate per sample')
parser.add_argument('--prompt',        default='\x02',
                    help='generation prompt (\\x02 = START)')
parser.add_argument('--seed',          default=None,  type=int)
parser.add_argument('--device',        default=None)
parser.add_argument('--verbose',       default=False, action='store_true')
parser.add_argument('--vis',           default=False, action='store_true',
                    help='live matshow of DFF std during training')
parser.add_argument('--delay',         default=0.1,   type=float,
                    help='seconds to pause per vis frame')
parser.add_argument('--cmap',          default='viridis',
                    help='matplotlib colormap for --vis')
# live demo
parser.add_argument('--demo',          default=False, action='store_true',
                    help='live webcam+mic -> text caption (use with --load); no training')
parser.add_argument('--camera',        default=0,     type=int,
                    help='--demo webcam device index (/dev/videoN)')
parser.add_argument('--audio_device',  default=None,
                    help='--demo mic: sounddevice index or name substring '
                         '(default = system default input; list with '
                         '`python -c "import sounddevice;print(sounddevice.query_devices())"`)')
parser.add_argument('--temperature',   default=1.0,   type=float,
                    help='--demo sampling temperature (0 = argmax)')
parser.add_argument('--no_preview',    default=False, action='store_true',
                    help='--demo: terminal caption only, no OpenCV preview window')
args = parser.parse_args()

# ── multi-dataset mix ─────────────────────────────────────────────────────────

def _dataset_modalities(name):
    """(uses_image, uses_audio) for a dataset name."""
    if name == 'webvid':      return (True, True)
    if name == 'cc3m':        return (True, False)
    if name == 'librispeech': return (False, True)
    return (False, False)   # text datasets (tiny/c4/web)

def _parse_mix(spec):
    """'webvid:0.5,web:0.5' -> ([names], [normalized weights]); None if unset."""
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
if _mix is not None:
    # both encoders are sized by the mix's union of modalities; per-example masks
    # zero out the contribution for clips that lack a modality (see _fuse)
    _mix_names = _mix[0]
    args.no_image = not any(_dataset_modalities(n)[0] for n in _mix_names)
    args.no_audio = not any(_dataset_modalities(n)[1] for n in _mix_names)

# ── restore architecture args from checkpoint ─────────────────────────────────

_ARCH_ARGS = ('context', 'n_hidden', 'depth', 'kernel', 'residual', 'cond',
              'norm', 'state_norm',
              'no_image', 'no_audio', 'c_audio', 'n_mels', 'fps', 'mel_hop',
              'audio_work_sr', 'audio_window')

_loaded_ckpt = None
if args.load is not None:
    _loaded_ckpt = torch.load(args.load, map_location='cpu', weights_only=True)
    if isinstance(_loaded_ckpt, dict) and 'saved_args' in _loaded_ckpt:
        _saved = _loaded_ckpt['saved_args']
        for _k in _ARCH_ARGS:
            if _k in _saved:
                setattr(args, _k, _saved[_k])

# ── log / device / seed ───────────────────────────────────────────────────────

if args.log is None:
    os.makedirs('log', exist_ok=True)
    date = subprocess.check_output(['/usr/bin/date', '+%Y.%m.%d-%H.%M.%S']).decode().strip()
    args.log = f'log/log.{date}'
if args.device is None:
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.seed is None:
    args.seed = int.from_bytes(os.urandom(4), byteorder='big')
    print('seed', args.seed)

torch.manual_seed(args.seed)
print(args)
with open(args.log, 'a') as f:
    print('ARGS', args, file=f)

# ── model ─────────────────────────────────────────────────────────────────────

class ContextCNN(nn.Module):
    """DFF state update. With --residual: x = 0.5*x + layer(x) per conv+ReLU
    block, then a final 1×1 linear projection. Without: plain layer chain.
    Input:  [B, n_hidden, S, S]
    Output: [B, n_hidden, S, S]
    """
    def __init__(self, n_hidden, depth, kernel, residual=True, norm='none'):
        super().__init__()
        self.residual = residual
        pad = kernel // 2

        def _norm():
            if norm == 'group':
                return nn.GroupNorm(8 if n_hidden % 8 == 0 else 1, n_hidden)
            if norm == 'batch':
                return nn.BatchNorm2d(n_hidden)
            return nn.Identity()

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.Sequential(
                nn.Conv2d(n_hidden, n_hidden, kernel, padding=pad),
                _norm(),
                nn.ReLU(),
            ))
        self.out = nn.Conv2d(n_hidden, n_hidden, 1)   # final 1×1 linear projection

    def forward(self, x):
        for layer in self.layers:
            x = 0.5 * x + layer(x) if self.residual else layer(x)
        return self.out(x)


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


class MelAudioEncoder(nn.Module):
    """Per-frame sliding window of raw samples → log-mel → small conv → [c_audio,S,S].
    The mel filterbank is fixed (not trained); only the conv stack learns."""
    def __init__(self, S, c_audio, n_mels=64, mel_hop=None, work_sr=16000):
        super().__init__()
        # hop_length is the intra-window mel resolution, independent of --fps;
        # default 10ms (work_sr/100) keeps audio detail fixed as the step rate varies
        hop = mel_hop if mel_hop else max(1, work_sr // 100)
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=work_sr, n_fft=512, win_length=400,
            hop_length=hop, n_mels=n_mels, power=2.0)
        self.net = nn.Sequential(
            nn.Conv2d(1,  32, 3, padding=1),  nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),  nn.ReLU(),
            nn.AdaptiveAvgPool2d((S, S)),     # mel/T grid → S×S, any window length
            nn.Conv2d(64, c_audio, 1),
        )

    def forward(self, win):                       # [B, 1, audio_window] mono
        x   = win[:, 0]                            # [B, audio_window]
        mel = self.melspec(x)                      # [B, n_mels, T]
        mel = torch.log(mel + 1e-6).unsqueeze(1)   # [B, 1, n_mels, T]
        return self.net(mel)                        # [B, c_audio, S, S]


class ImageEncoder(nn.Module):
    """Frozen ResNet-18 through layer4: [B,3,224,224] in [0,1] → [B,512,7,7]."""
    def __init__(self):
        super().__init__()
        import torchvision.models as tv_models
        rn = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            rn.conv1, rn.bn1, rn.relu, rn.maxpool,
            rn.layer1, rn.layer2, rn.layer3, rn.layer4,
        )
        for p in self.parameters():
            p.requires_grad_(False)
        # channels_last lets the frozen convs hit tensor cores; persists across .to(device)
        self.backbone = self.backbone.to(memory_format=torch.channels_last)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img):                         # [B, 3, 224, 224] in [0,1]
        x = ((img - self.mean) / self.std).to(memory_format=torch.channels_last)
        # frozen inference: fp16 autocast on CUDA for tensor-core speedup; cast the
        # feature map back to fp32 so downstream (image_proj, add) stays fp32
        with torch.autocast('cuda', dtype=torch.float16, enabled=x.is_cuda):
            out = self.backbone(x)
        return out.float().contiguous()


class VLAModel(nn.Module):
    """Recurrent CNN conditioned by token + optional audio/image @100 Hz.
    forward()/generate() accept img/aud; folded into the projector input per
    --cond. Absent modalities (None) contribute zeros so concat width is fixed."""
    def __init__(self, n_hidden, depth, kernel, S, residual, cond,
                 use_audio, use_image, c_audio, n_mels, mel_hop, work_sr,
                 norm='none', state_norm='none'):
        super().__init__()
        self.S          = S
        self.n_hidden   = n_hidden
        self.cond       = cond
        self.use_audio  = use_audio
        self.use_image  = use_image
        self.c_audio    = c_audio
        self.state_norm = state_norm

        self.tok_embed = nn.Embedding(256, n_hidden)   # current token → n_hidden
        self.context   = ContextCNN(n_hidden, depth, kernel, residual=residual, norm=norm)
        self.decoder   = DecoderCNN(n_hidden, S)

        self.audio_encoder = MelAudioEncoder(S, c_audio, n_mels, mel_hop, work_sr) if use_audio else None
        self.image_encoder = ImageEncoder() if use_image else None

        if cond == 'add':
            # project each modality to n_hidden, summed into the DFF + token grid
            self.audio_proj = nn.Conv2d(c_audio,  n_hidden, 1) if use_audio else None
            self.image_proj = nn.Conv2d(IMAGE_CH, n_hidden, 1) if use_image else None
            self.input_adapter = None
        else:  # concat
            in_ch = 2 * n_hidden \
                  + (c_audio  if use_audio else 0) \
                  + (IMAGE_CH if use_image else 0)
            self.input_adapter = nn.Conv2d(in_ch, n_hidden, 1)
            self.audio_proj = self.image_proj = None

        self.dff        = None   # [B, n_hidden, S, S] — DFF register, always detached
        self.last_a_enc = None
        self.last_i_enc = None
        self._img_cache = None   # (img_tensor, i_feat) — frozen encoder is deterministic,
                                 # so the post-step recompute reuses it instead of re-running ResNet

    def train(self, mode=True):
        super().train(mode)
        if self.image_encoder is not None:
            self.image_encoder.eval()   # keep frozen ResNet BN stats fixed
        return self

    def _init_dff(self, B, device):
        if self.dff is None or self.dff.shape[0] != B or self.dff.device != torch.device(device):
            self.dff = torch.zeros(B, self.n_hidden, self.S, self.S, device=device)

    def _tok_grid(self, byte_idx):
        B   = byte_idx.shape[0]
        tok = self.tok_embed(byte_idx)                             # [B, n_hidden]
        return tok.view(B, self.n_hidden, 1, 1).expand(-1, -1, self.S, self.S)

    def _norm_state(self, x):
        """Bound the recurrent state magnitude to prevent unbounded growth."""
        if self.state_norm == 'tanh':
            return torch.tanh(x)
        if self.state_norm == 'rms':
            rms = x.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt()
            return x / (rms + 1e-5)
        return x

    def _encode_modalities(self, B, dev, img, aud):
        a_feat = None
        if self.use_audio:
            a_feat = self.audio_encoder(aud) if aud is not None \
                     else torch.zeros(B, self.c_audio, self.S, self.S, device=dev)
        i_feat = None
        if self.use_image:
            if img is not None:
                # frozen ResNet is deterministic: cache by tensor identity so the
                # post-optimizer recompute (same img object) skips a second ResNet pass
                if self._img_cache is not None and self._img_cache[0] is img:
                    i_feat = self._img_cache[1]
                else:
                    i_feat = self.image_encoder(img)
                    self._img_cache = (img, i_feat)
            else:
                i_feat = torch.zeros(B, IMAGE_CH, self.S, self.S, device=dev)
        return i_feat, a_feat

    def _fuse(self, dff, tok_grid, i_feat, a_feat, i_mask=None, a_mask=None):
        """Combine DFF, token grid and modality maps into the ContextCNN input.
        i_mask/a_mask [B] (1=present, 0=absent) gate each modality per example, so
        text-only clips in a mixed batch contribute exactly zero — not the encoder's
        response to a black/silent input. None mask = all present (no gating)."""
        def _g(feat, mask):
            return feat if mask is None else feat * mask.view(-1, 1, 1, 1)
        if self.cond == 'add':
            out = dff + tok_grid
            if a_feat is not None: out = out + _g(self.audio_proj(a_feat), a_mask)
            if i_feat is not None: out = out + _g(self.image_proj(i_feat), i_mask)
            return out
        parts = [dff, tok_grid]                       # order must match in_ch
        if a_feat is not None: parts.append(_g(a_feat, a_mask))
        if i_feat is not None: parts.append(_g(i_feat, i_mask))
        return self.input_adapter(torch.cat(parts, dim=1))

    def forward(self, byte_idx, img=None, aud=None, targets=None, flag=None,
                img_mask=None, aud_mask=None):
        """
        byte_idx  [B] long — current token (teacher-forced: ground truth at t)
        img       [B,3,224,224] or None — per-frame image
        aud       [B,1,audio_window] or None — per-frame audio window
        targets   [B] long — next token (CE target), or None
        flag      [B] bool — reset DFF for these batch elements
        img_mask/aud_mask [B] float (1=present) — gate modality per example in a
                  mixed batch; None = all present
        """
        B, dev = byte_idx.shape[0], byte_idx.device
        self._init_dff(B, dev)

        if flag is not None and flag.any():
            self.dff = self.dff.clone()   # clone before in-place
            self.dff[flag] = 0.0

        tok_grid       = self._tok_grid(byte_idx)
        i_feat, a_feat = self._encode_modalities(B, dev, img, aud)
        self.last_a_enc = a_feat.detach() if a_feat is not None else None
        self.last_i_enc = i_feat.detach() if i_feat is not None else None

        new_ctx  = self.context(self._fuse(self.dff, tok_grid, i_feat, a_feat,
                                           img_mask, aud_mask))
        new_ctx  = self._norm_state(new_ctx)
        self.dff = new_ctx.detach().clone()             # store for next step

        logits = self.decoder(new_ctx)
        loss   = F.cross_entropy(logits, targets.long()) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt_bytes, n_tokens, img_feed=None, aud_feed=None):
        """Autoregressive generation. img_feed/aud_feed are optional per-step
        callbacks returning the current frame tensor (or None)."""
        self.eval()
        dev = next(self.parameters()).device
        dff = torch.zeros(1, self.n_hidden, self.S, self.S, device=dev)

        def _step(b):
            nonlocal dff
            bi   = torch.tensor([b], dtype=torch.long, device=dev)
            tok  = self.tok_embed(bi).view(1, self.n_hidden, 1, 1).expand(-1, -1, self.S, self.S)
            img  = img_feed() if img_feed else None
            aud  = aud_feed() if aud_feed else None
            i_feat, a_feat = self._encode_modalities(1, dev, img, aud)
            new_ctx = self.context(self._fuse(dff, tok, i_feat, a_feat))
            new_ctx = self._norm_state(new_ctx)
            dff     = new_ctx.detach().clone()
            if not dff.isfinite().all():
                print(f'generate: dff NaN/Inf  dff_std={dff[dff.isfinite()].std().item():.4f}')
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
            bval = torch.multinomial(F.softmax(logits[0], dim=-1), 1).item()
            if bval == END:
                break
            if bval != NULL:
                out.append(bval)
            logits = _step(bval)
            if logits is None:
                break
        return out


# ── caption alignment ─────────────────────────────────────────────────────────

def align_caption_proportional(caption_bytes, n_steps):
    """
    Spread caption words uniformly across n_steps (causal, word-boundary-aware).
    Words split on space (0x20), trailing space stays with the word. Groups =
    [START] + word_segments + [END] are placed at evenly-spaced steps; within a
    group bytes are emitted consecutively. All other steps carry NULL.
    Returns (inputs, targets), each length n_steps.

    OPEN ISSUE: this is a uniform-spread placeholder. True causal alignment to
    audio (forced aligner / CTC timestamps) is future work.
    """
    segments, cur = [], []
    for b in caption_bytes:
        cur.append(b)
        if b == 0x20:
            segments.append(cur)
            cur = []
    if cur:
        segments.append(cur)

    full   = [START] + list(caption_bytes) + [END]
    groups = [[START]] + segments + [[END]]
    G      = len(groups)

    inputs  = [NULL] * n_steps
    targets = [NULL] * n_steps
    fi      = 0
    for g, group in enumerate(groups):
        s = min(round(g * (n_steps - 1) / max(G - 1, 1)), n_steps - 1)
        for offset, val in enumerate(group):
            t          = min(s + offset, n_steps - 1)
            inputs[t]  = val
            targets[t] = full[fi + 1] if fi + 1 < len(full) else NULL
            fi        += 1
    return inputs, targets


# ── clip builders ──────────────────────────────────────────────────────────────

# Thread-safe decode tally (builder threads write, main loop drains each checkpoint).
# A FAIL is a clip that could not be decoded and fell back to text-only.
_decode_lock  = threading.Lock()
_decode_stats = {'pass': 0, 'fail': 0}

def _decode_tally(ok):
    with _decode_lock:
        _decode_stats['pass' if ok else 'fail'] += 1

def _decode_drain():
    """Return (pass, fail) since the last drain and reset the counters."""
    with _decode_lock:
        p, f = _decode_stats['pass'], _decode_stats['fail']
        _decode_stats['pass'] = _decode_stats['fail'] = 0
    return p, f

# Thread-safe tally of pure text-only examples (e.g. openwebtext). Builder threads
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


def clip_text(text, max_steps=None):
    """Text-only: one step per byte, no img/aud."""
    cap  = _text_to_bytes(text)
    full = [START] + cap + [END]
    if max_steps is not None:
        full = full[:max_steps + 1]   # n_steps = len(full) - 1
    steps = [(full[i], full[i + 1], None, None) for i in range(len(full) - 1)]
    return steps, text


def clip_librispeech(example, work_sr, fps, audio_window):
    """Audio + transcript. Audio defines duration; transcript spread across steps.
    Each step carries a sliding window of `audio_window` raw samples ending at
    that frame (left zero-padded at the clip start)."""
    import io, soundfile as sf
    aud_info = example['audio']
    if 'array' in aud_info and aud_info['array'] is not None:
        audio = np.asarray(aud_info['array'], dtype=np.float32)
        sr    = aud_info['sampling_rate']
    elif aud_info.get('bytes') is not None:
        arr, sr = sf.read(io.BytesIO(aud_info['bytes']))
        audio = (arr.mean(axis=1) if arr.ndim == 2 else arr).astype(np.float32)
    elif aud_info.get('path') is not None:
        arr, sr = sf.read(aud_info['path'])
        audio = (arr.mean(axis=1) if arr.ndim == 2 else arr).astype(np.float32)
    else:
        raise ValueError(f'audio example has no usable field: {list(aud_info.keys())}')
    text = example['text'].strip()

    if sr != work_sr:
        a_t   = torch.from_numpy(audio).unsqueeze(0)
        audio = torchaudio.functional.resample(a_t, sr, work_sr).squeeze(0).numpy()

    hop     = max(1, work_sr // fps)
    n_steps = max(1, len(audio) // hop)

    cap        = _text_to_bytes(text)
    ins, tgts  = align_caption_proportional(cap, n_steps)
    steps      = []
    for t in range(n_steps):
        end   = (t + 1) * hop
        start = max(0, end - audio_window)
        win   = audio[start:end]
        if len(win) < audio_window:
            win = np.pad(win, (audio_window - len(win), 0))   # left-pad at clip start
        steps.append((ins[t], tgts[t], None, win.reshape(1, -1).astype(np.float32)))
    return steps, text


def _img_to_chw(img):
    """PIL.Image | ndarray | URL str | None → [3,224,224] uint8, or None on failure."""
    import io
    import PIL.Image
    try:
        if img is None:
            return None
        if isinstance(img, str):                       # URL (cc3m image_url)
            import urllib.request
            req = urllib.request.Request(img, headers={'User-Agent': 'vla/1.0'})
            with urllib.request.urlopen(req, timeout=5) as r:
                img = PIL.Image.open(io.BytesIO(r.read()))
        if not isinstance(img, np.ndarray):
            img = np.array(img.convert('RGB')) if hasattr(img, 'convert') else np.array(img)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        pil = PIL.Image.fromarray(img).resize((224, 224))
        return np.array(pil).transpose(2, 0, 1).astype(np.uint8)   # [3,224,224]
    except Exception:
        return None


def clip_image_caption(img, caption_text):
    """Single image held constant; caption emitted one char per step (cc3m)."""
    frame = _img_to_chw(img)
    _decode_tally(frame is not None)   # None = image fetch/decode failed (black frame)
    cap   = _text_to_bytes(caption_text)
    full  = [START] + cap + [END]
    steps = [(full[i], full[i + 1], frame, None) for i in range(len(full) - 1)]
    return steps, caption_text


def clip_webvid(example, work_sr, fps, audio_window, no_image, no_audio, max_steps=None):
    """Decode a WebVid clip → image frames @fps + audio sliding windows, with the
    caption spread across steps. Falls back to text-only if decode is unavailable."""
    caption = example.get('name', example.get('caption', ''))
    container = None
    try:
        import io, av
        av.logging.set_level(av.logging.FATAL)   # silence per-frame NAL/pull noise; fail rate is tracked via _decode_tally
        vid = example.get('video', example.get('contentUrl', example.get('url')))
        if vid is None:
            raise ValueError('no video field')
        if isinstance(vid, dict):
            container = av.open(io.BytesIO(vid['bytes']))
        else:                                     # remote URL: bound I/O so a reset/stalled
            container = av.open(vid, timeout=15)  # connection fails fast instead of buffering

        has_v = len(container.streams.video) > 0 and not no_image
        has_a = len(container.streams.audio) > 0 and not no_audio
        if not has_v and not has_a:
            raise ValueError('no usable streams')

        streams = []
        if has_v: streams.append(container.streams.video[0])
        if has_a: streams.append(container.streams.audio[0])

        fps_src   = float(container.streams.video[0].average_rate) if has_v else fps
        a_sr      = container.streams.audio[0].sample_rate if has_a else work_sr
        hop       = max(1, work_sr // fps)

        # Hard caps on how much we pull into RAM. A flaky webvid stream that keeps
        # reconnecting ("Connection reset by peer") can otherwise feed frames
        # indefinitely, and decoding full-length clips at full resolution × many
        # builder threads exhausted host RAM (60GB OOM kill). When the output is
        # capped (--max_clip_steps) we only need enough source content to cover it;
        # otherwise fall back to a generous absolute cap (~5 min of source).
        max_v = int(fps_src * 300) if has_v else 0
        max_a = int(a_sr * 300)
        if max_steps is not None:
            if has_v:
                max_v = min(max_v, int(max_steps * fps_src / fps) + 2)
            max_a = min(max_a, int((max_steps + 1) * hop * (a_sr / work_sr)) + a_sr)

        v_frames  = []   # resized [3,224,224] uint8 (resize-on-decode, NOT full-res rgb24)
        a_chunks  = []
        a_count   = 0
        for frame in container.decode(*streams):
            if isinstance(frame, av.VideoFrame):
                if len(v_frames) < max_v:
                    v_frames.append(_img_to_chw(frame.to_ndarray(format='rgb24')))
            else:
                if a_count < max_a:
                    chunk = frame.to_ndarray().mean(0).astype(np.float32)
                    a_chunks.append(chunk)
                    a_count += len(chunk)
            if (not has_v or len(v_frames) >= max_v) and \
               (not has_a or a_count >= max_a):
                break                              # have all the source content we need

        if has_a and a_chunks:
            audio = np.concatenate(a_chunks)
            if a_sr != work_sr:
                a_t   = torch.from_numpy(audio).unsqueeze(0)
                audio = torchaudio.functional.resample(a_t, a_sr, work_sr).squeeze(0).numpy()
            n_steps = max(1, len(audio) // hop)
        else:
            audio   = None
            n_steps = max(1, round(len(v_frames) * fps / fps_src))

        if max_steps is not None:           # cap BEFORE alignment so the full caption
            n_steps = min(n_steps, max_steps)  # still spreads over the (shorter) clip
        cap        = _text_to_bytes(caption)
        ins, tgts  = align_caption_proportional(cap, n_steps)
        steps      = []
        for t in range(n_steps):
            frame = None
            if has_v and v_frames:
                src_idx = min(int(round(t * fps_src / fps)), len(v_frames) - 1)
                frame   = v_frames[src_idx]     # already resized to [3,224,224] during decode
            aud = None
            if audio is not None:
                end   = (t + 1) * hop
                start = max(0, end - audio_window)
                win   = audio[start:end]
                if len(win) < audio_window:
                    win = np.pad(win, (audio_window - len(win), 0))
                aud = win.reshape(1, -1).astype(np.float32)
            steps.append((ins[t], tgts[t], frame, aud))
        _decode_tally(True)
        return steps, caption

    except Exception:
        _decode_tally(False)   # summarized per checkpoint instead of printed per clip
        return clip_text(caption, max_steps=max_steps)
    finally:
        if container is not None:
            try:    container.close()   # always free the ffmpeg context (leak on failure otherwise)
            except Exception: pass


# ── dataset loading ────────────────────────────────────────────────────────────

_TEXT_DATASETS = {
    'tiny': ('roneneldan/TinyStories', None, 'train', 'text'),
    'c4':   ('allenai/c4',            'en', 'train', 'text'),
    'web':  ('Skylion007/openwebtext', None, 'train', 'text'),
}

_DS_SPEC = {   # name → (hf_name, hf_config, split)
    'librispeech': ('openslr/librispeech_asr',                      'clean', 'train.100'),
    'cc3m':        ('google-research-datasets/conceptual_captions', None,    'train'),
    'webvid':      ('TempoFunk/webvid-10M',                         None,    'train'),
}


def _split_for_name(name):
    if name in _TEXT_DATASETS:
        return _TEXT_DATASETS[name][2]
    if name in _DS_SPEC:
        return _DS_SPEC[name][2]
    return 'train'


def _init_one_dataset(name, streaming):
    from datasets import load_dataset
    if name in _TEXT_DATASETS:
        hf, cfg, _, _ = _TEXT_DATASETS[name]
        loader = lambda dlc: load_dataset(hf, cfg, streaming=streaming, download_config=dlc)
    elif name in _DS_SPEC:
        hf, cfg, _ = _DS_SPEC[name]
        def loader(dlc):
            ds = load_dataset(hf, cfg, streaming=streaming, download_config=dlc)
            if name == 'librispeech':
                from datasets import Audio
                try:   # decode audio manually with soundfile (avoids torchcodec dependency)
                    ds = ds.cast_column('audio', Audio(decode=False))
                except Exception:
                    pass
            return ds
    else:
        raise ValueError(f'unknown dataset: {name}')

    # Non-streaming eagerly resolves every shard (webvid-10M = 7152 metadata CSVs),
    # which blows past HF's 5000-requests/5-min limit -> 429. Throttle the download
    # workers and retry across rate-limit windows; cached shards persist between tries
    # so each attempt makes forward progress until the cache is complete.
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


def _shuffle_buffer_for(name):
    # webvid examples carry whole video blobs (MBs each); a 10k buffer would pin GBs
    # of host RAM, so keep the video reservoir small
    return 256 if name == 'webvid' else 10000

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
        return item   # (x_byte, y_byte, img_or_None, aud_or_None)


GEN_SAMPLE_STEPS = 400

def worker(stop, q, gen_q, datasets, args, mix=None):
    # datasets: {name: hf_dataset}. mix: (names, weights) or None (single dataset).
    if mix is None:
        names, weights = [args.dataset], [1.0]
    else:
        names, weights = mix
    M       = args.max_clip_steps
    _iters  = {}
    _locks  = {}
    _epochs = {}
    for n in set(names):
        _iters[n]  = _make_iter(datasets[n], _split_for_name(n), args.streaming,
                                args.seed, _shuffle_buffer_for(n))
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
                                          args.streaming, args.seed + _epochs[name][0],
                                          _shuffle_buffer_for(name))
                return next(_iters[name])

    def _build_clip(name, example):
        if name in _TEXT_DATASETS:
            col = _TEXT_DATASETS[name][3]
            return clip_text(example.get(col, example.get('text', '')), max_steps=M)
        elif name == 'librispeech':
            if args.no_audio:
                return clip_text(example['text'], max_steps=M)
            return clip_librispeech(example, args.audio_work_sr, args.fps, args.audio_window)
        elif name == 'cc3m':
            cap = example.get('caption', '')
            if args.no_image:
                return clip_text(cap, max_steps=M)
            img = example.get('image', example.get('image_url', example.get('url')))
            return clip_image_caption(img, cap)
        elif name == 'webvid':
            if args.no_image and args.no_audio:
                return clip_text(example.get('name', example.get('caption', '')), max_steps=M)
            return clip_webvid(example, args.audio_work_sr, args.fps, args.audio_window,
                               args.no_image, args.no_audio, max_steps=M)
        return [], ''

    slots = [_ClipSlot() for _ in range(args.batch)]

    # Pool of builder threads runs the expensive decode/download in parallel and
    # parks finished clips in clip_q, so the stepping loop below never blocks on a
    # single clip. The buffer is bounded to cap memory (webvid clips are large).
    n_builders = max(1, args.workers)
    clip_q     = queue.Queue(maxsize=2 * n_builders)

    def builder():
        rng = random.Random()   # per-clip dataset choice (thread-local instance)
        while not stop.is_set():
            name = rng.choices(names, weights=weights, k=1)[0]
            try:
                steps, caption = _build_clip(name, _next_example(name))
            except Exception as e:
                print(f'builder: {e}')
                continue
            if not steps:
                continue
            if name in _TEXT_DATASETS:   # pure text-only example (e.g. openwebtext)
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
                try:
                    gen_q.put_nowait((caption, steps[:GEN_SAMPLE_STEPS]))
                except queue.Full:
                    pass

        x_list, y_list, img_list, aud_list, flag_list = [], [], [], [], []
        for slot in slots:
            flag = slot.is_start()
            x, y, img, aud = slot.advance()
            x_list.append(x)
            y_list.append(y)
            img_list.append(img)
            aud_list.append(aud)
            flag_list.append(flag)

        q.put((x_list, y_list, img_list, aud_list, flag_list))


# ── dataset loading (before CUDA init) ───────────────────────────────────────

if args.demo:
    hf_dataset = None                 # --demo streams from the webcam, not a dataset
else:
    print('loading dataset...')
    if _mix is not None:
        _ds_names  = sorted(set(_mix[0]))
        hf_dataset = {n: _init_one_dataset(n, args.streaming) for n in _ds_names}
        print('mix:', ', '.join(f'{n}:{w:.3f}' for n, w in zip(_mix[0], _mix[1])),
              '| use_image', not args.no_image, '| use_audio', not args.no_audio)
    else:
        hf_dataset = {args.dataset: _init_one_dataset(args.dataset, args.streaming)}
    print('dataset ready')

# ── model instantiation ───────────────────────────────────────────────────────

model = VLAModel(
    n_hidden  = args.n_hidden,
    depth     = args.depth,
    kernel    = args.kernel,
    S         = args.context,
    residual  = args.residual,
    cond      = args.cond,
    use_audio = not args.no_audio,
    use_image = not args.no_image,
    c_audio   = args.c_audio,
    n_mels    = args.n_mels,
    mel_hop   = args.mel_hop,
    work_sr   = args.audio_work_sr,
    norm      = args.norm,
    state_norm= args.state_norm,
)

if _loaded_ckpt is not None:
    _sd = _loaded_ckpt['state_dict'] if isinstance(_loaded_ckpt, dict) and 'state_dict' in _loaded_ckpt \
          else _loaded_ckpt
    model.load_state_dict(_sd)
    del _loaded_ckpt

_dff_shape = (1, args.n_hidden, args.context, args.context)
print(torchinfo.summary(model.context, input_size=_dff_shape,
                         col_names=['input_size', 'output_size', 'num_params'], verbose=0))
print(torchinfo.summary(model.decoder, input_size=_dff_shape,
                         col_names=['input_size', 'output_size', 'num_params'], verbose=0))
if model.image_encoder is not None:
    print(torchinfo.summary(model.image_encoder, input_size=(1, 3, 224, 224),
                            col_names=['input_size', 'output_size', 'num_params'], verbose=0))

model = model.to(args.device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 'M trainable params')

# ── live demo ─────────────────────────────────────────────────────────────────

def run_demo(model, args):
    """Stream webcam frames + mic audio through the recurrent model one step per
    1/fps seconds, emitting a rolling text caption. self.dff carries context across
    steps exactly as in training; press 'r' to reset it, ESC/q (or Ctrl-C) to quit."""
    import time, sys, collections
    try:
        import cv2
    except ImportError:
        raise SystemExit('--demo needs OpenCV:  pip install opencv-python')

    model.eval()
    dev       = args.device
    SR, WIN   = args.audio_work_sr, args.audio_window
    period    = 1.0 / max(1, args.fps)            # seconds per step (match training fps)
    use_image = model.use_image
    use_audio = model.use_audio

    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        raise SystemExit(f'cannot open camera index {args.camera}')

    stream  = None
    dev_sr  = SR
    ring    = collections.deque([0.0] * WIN, maxlen=WIN)
    if use_audio:
        try:
            import sounddevice as sd
            # pick the mic: explicit --audio_device, else system default input
            adev = args.audio_device
            if adev is not None:
                try:    adev = int(adev)          # numeric index
                except (TypeError, ValueError):   pass   # else a name substring
            info = sd.query_devices(adev, 'input') if adev is not None \
                   else sd.query_devices(kind='input')
            # capture at the mic's native rate (the Logitech C270 mic is 48kHz, not 16kHz)
            # and resample each window to work_sr in _aud
            dev_sr  = int(info['default_samplerate'])
            ring_n  = max(WIN, int(round(args.audio_window * dev_sr / SR)))   # ~100ms at dev_sr
            ring    = collections.deque([0.0] * ring_n, maxlen=ring_n)
            def _acb(indata, frames, t, status):
                ring.extend(indata[:, 0].tolist())
            stream = sd.InputStream(device=adev, samplerate=dev_sr, channels=1,
                                    dtype='float32', callback=_acb)
            stream.start()
            print(f"mic '{info['name']}' @ {dev_sr} Hz -> resampled to {SR} Hz")
        except Exception as e:
            print(f'audio disabled ({e}); running video-only')
            use_audio = False

    def _frame():
        ok, bgr = cam.read()
        if not ok:
            return None, None
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (224, 224))
        t = torch.from_numpy(small).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(dev)
        return t, bgr

    def _aud():
        a = np.asarray(ring, dtype=np.float32)
        if dev_sr != SR:                              # mic rate -> work_sr (e.g. 48k -> 16k)
            a = torchaudio.functional.resample(torch.from_numpy(a).unsqueeze(0),
                                               dev_sr, SR).squeeze(0).numpy()
        a = a[-WIN:]
        if len(a) < WIN:
            a = np.pad(a, (WIN - len(a), 0))          # left-pad at startup
        return torch.from_numpy(a.astype(np.float32)).view(1, 1, -1).to(dev)

    one     = torch.ones(1, device=dev)
    preview = not args.no_preview
    model.dff = None                              # fresh context
    b, caption = START, []
    print(f'live demo @ {args.fps} steps/s — image={use_image} audio={use_audio} | '
          f"{'ESC/q quit, r reset; ' if preview else ''}Ctrl-C quits")
    try:
        with torch.no_grad():
            while True:
                t0       = time.time()
                img, bgr = _frame() if use_image else (None, None)
                aud      = _aud()   if use_audio else None
                bi       = torch.tensor([b], dtype=torch.long, device=dev)
                logits, _ = model(bi, img, aud,
                                  img_mask=(one if img is not None else None),
                                  aud_mask=(one if aud is not None else None))
                if args.temperature <= 0:
                    b = int(logits[0].argmax())
                else:
                    p = F.softmax(logits[0] / args.temperature, dim=-1)
                    b = int(torch.multinomial(p, 1))

                if b in (START, END):
                    caption = []                  # clip boundary → start fresh
                elif 32 <= b < 127:
                    caption.append(chr(b))        # NULL/padding skipped
                line = ''.join(caption[-args.n:])
                sys.stdout.write('\rCAPTION: ' + line + '\x1b[K'); sys.stdout.flush()

                if preview and bgr is not None:
                    try:
                        cv2.putText(bgr, line[-60:], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 0), 2)
                        cv2.imshow('vla demo', bgr)
                        k = cv2.waitKey(1) & 0xFF
                        if k in (27, ord('q')):
                            break
                        if k == ord('r'):
                            model.dff, b, caption = None, START, []
                    except Exception:
                        preview = False           # headless: fall back to terminal only

                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        if stream is not None:
            stream.stop(); stream.close()
        try:    cv2.destroyAllWindows()
        except Exception: pass
        print()

if args.demo:
    if args.load is None:
        print('WARNING: --demo without --load is running an untrained model')
    run_demo(model, args)
    raise SystemExit(0)

# ── vis setup ─────────────────────────────────────────────────────────────────

_vis_exit        = False
_vis_input_chars = []
_vis_pred_chars  = []
_vis_stats_line  = 'initializing...'
_VIS_WIN         = 80


def _vc(b):
    if b == 0x02: return '▶'
    if b == 0x03: return '◀'
    if b == 0x00: return '_'
    c = chr(b) if 32 <= b < 127 else '\xb7'
    return '\xb7' if c in '$\\' else c


if args.vis:
    plt.style.use('dark_background')
    plt.ion()
    _vis_fig, _vis_axes_grid = plt.subplots(1, 2, figsize=(8, 4))
    _vis_fig.subplots_adjust(top=0.78, bottom=0.04, left=0.02, right=0.98,
                              hspace=0.1, wspace=0.08)
    _vis_ax_dff, _vis_ax_prob = _vis_axes_grid
    _vis_ax_dff.set_title('DFF std',       fontsize=8, pad=2)
    _vis_ax_prob.set_title('P(next token)', fontsize=8, pad=2)
    _vis_ax_dff.axis('off')
    _vis_ax_prob.axis('off')
    _vis_img_dff  = None
    _vis_img_prob = None
    _vis_header   = _vis_fig.text(0.01, 0.99, _vis_stats_line,
                                   family='monospace', fontsize=7,
                                   va='top', ha='left')

    def _on_key(event):
        global _vis_exit
        if event.key in ('x', 'X'):
            _vis_exit = True
    _vis_fig.canvas.mpl_connect('key_press_event', _on_key)

# ── optimizer ─────────────────────────────────────────────────────────────────
# Only trainable params (frozen ResNet excluded).

_trainable = [p for p in model.parameters() if p.requires_grad]

if args.opt == 'adamw':
    optimizer = torch.optim.AdamW(_trainable, lr=args.learning_rate,
                                  betas=(args.beta1, args.beta2),
                                  weight_decay=args.weight_decay)
elif args.opt == 'sgd':
    optimizer = torch.optim.SGD(_trainable, lr=args.learning_rate,
                                momentum=args.beta1)
elif args.opt == 'rms':
    optimizer = torch.optim.RMSprop(_trainable, lr=args.learning_rate,
                                    weight_decay=args.weight_decay)

# ── scheduler ─────────────────────────────────────────────────────────────────

if args.schedule == 'linear':
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=args.start_factor, end_factor=1.0,
        total_iters=args.period)
elif args.schedule == 'warmup':
    scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=args.start_factor, total_iters=args.period)
elif args.schedule == 'cyclic':
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, args.learning_rate * args.start_factor, args.learning_rate,
        step_size_up=args.period, step_size_down=args.period, mode='triangular',
        cycle_momentum=False)
elif args.schedule == 'decay':
    warm  = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=args.start_factor, end_factor=1.0,
        total_iters=args.warmup)
    decay = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=args.start_factor,
        total_iters=args.period)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warm, decay], milestones=[args.warmup])
elif args.schedule == 'piecewise':
    warm   = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=args.start_factor, end_factor=1.0,
        total_iters=args.warmup)
    decay1 = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=10 * args.start_factor,
        total_iters=args.period)
    decay2 = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=10 * args.start_factor, end_factor=args.start_factor,
        total_iters=args.period)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warm, decay1, decay2],
        milestones=[args.warmup, args.warmup + args.period])

# ── data thread ───────────────────────────────────────────────────────────────

stop  = threading.Event()
q     = queue.Queue(maxsize=args.batch)
gen_q = queue.Queue(maxsize=2)
w     = threading.Thread(target=worker,
                          args=[stop, q, gen_q, hf_dataset, args, _mix],
                          daemon=False)
w.start()

# ── helpers ───────────────────────────────────────────────────────────────────

def _to_img_tensor(img_list):
    """list of [3,224,224] uint8 ndarray or None → [B,3,224,224] float or None."""
    if args.no_image or all(x is None for x in img_list):
        return None
    frames = [f if f is not None else np.zeros((3, 224, 224), dtype=np.uint8)
              for f in img_list]
    # transfer uint8 (4x smaller H2D) then normalize on the GPU
    t = torch.from_numpy(np.stack(frames))                       # [B,3,224,224] uint8
    return t.to(args.device, non_blocking=True).float().div(255.0)


def _to_aud_tensor(aud_list):
    """list of [ch,n] float32 ndarray or None → [B,1,n] mono tensor or None."""
    if args.no_audio or all(x is None for x in aud_list):
        return None
    n    = max(a.shape[-1] for a in aud_list if a is not None)
    bufs = []
    for a in aud_list:
        if a is None:
            bufs.append(np.zeros((1, n), dtype=np.float32))
        else:
            if a.shape[-1] < n:
                a = np.pad(a, ((0, 0), (0, n - a.shape[-1])))
            bufs.append(a[:, :n])
    t = torch.from_numpy(np.stack(bufs)).to(args.device)   # [B, ch, n]
    return t.mean(dim=1, keepdim=True)                      # [B, 1,  n]


def _printable(s):
    s = s.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
    return ''.join(c for c in s if c.isprintable())

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
    g = {'emb': list(model.tok_embed.parameters()),
         'ctx': list(model.context.parameters()),
         'dec': list(model.decoder.parameters())}
    if model.audio_encoder is not None:
        g['aud'] = list(model.audio_encoder.parameters())
    if model.cond == 'add':
        if model.audio_proj is not None: g['aprj'] = list(model.audio_proj.parameters())
        if model.image_proj is not None: g['iprj'] = list(model.image_proj.parameters())
    elif model.input_adapter is not None:
        g['adpt'] = list(model.input_adapter.parameters())
    return g


def _diag(model):
    return ' '.join(f'{n}:g{_gnorm(ps):.1e}/w{_wnorm(ps):.1e}'
                    for n, ps in _param_groups_named(model).items())

# ── training loop ─────────────────────────────────────────────────────────────

larr, garr, a_std_arr, i_std_arr = [], [], [], []
ltxt_arr, lav_arr = [], []   # per-modality loss split (mix mode): text-only vs visual rows
i = 0

try:
    while True:
        # ── checkpoint / generation sample ───────────────────────────────────
        if (i % args.checkpoint) == 0:
            no_side_effects = args.generate
            if not no_side_effects:
                _finite = all(torch.isfinite(p).all() for p in model.state_dict().values()
                              if p.is_floating_point())
                if _finite:
                    torch.save({'saved_args': vars(args), 'state_dict': model.state_dict()},
                               args.save)
                else:
                    print(f'WARNING: non-finite params at step {i}; skipping checkpoint save')

            # decode tally for this checkpoint interval (only when modalities are on)
            # plus pure text-only throughput (e.g. openwebtext examples/tokens)
            _dp, _df = _decode_drain()
            _te, _tt, _tsample = _text_drain()
            _parts = [f'DECODES step {i:10}']
            if _dp or _df:
                _rate = 100.0 * _df / (_dp + _df)
                _parts.append(f'PASS {_dp} FAIL {_df} ({_rate:.1f}% fail)')
            if _te:
                _parts.append(f'TEXT examples {_te} tokens {_tt}')
            if len(_parts) > 1:
                d = '  '.join(_parts)
                print(d)
                if not args.generate:
                    with open(args.log, 'a') as f:
                        print(d, file=f)

            model.eval()

            gen_caption, gen_steps = '', []
            while True:
                try:
                    gen_caption, gen_steps = gen_q.get_nowait()
                except queue.Empty:
                    break

            # build per-step img/aud feeds from the sample clip
            has_imgs = model.use_image and any(s[2] is not None for s in gen_steps)
            has_auds = model.use_audio and any(s[3] is not None for s in gen_steps)

            if has_imgs:
                _gi, _gi_idx = [s[2] for s in gen_steps], [0]
                def img_feed():
                    f = _gi[_gi_idx[0]]
                    _gi_idx[0] = min(_gi_idx[0] + 1, len(_gi) - 1)
                    if f is None:
                        return torch.zeros(1, 3, 224, 224, device=args.device)
                    return torch.from_numpy(f).float().div(255.0).unsqueeze(0).to(args.device)
            else:
                img_feed = None

            if has_auds:
                _ga, _ga_idx = [s[3] for s in gen_steps], [0]
                def aud_feed():
                    a = _ga[_ga_idx[0]]
                    _ga_idx[0] = min(_ga_idx[0] + 1, len(_ga) - 1)
                    if a is None:
                        return torch.zeros(1, 1, args.audio_window, device=args.device)
                    return torch.from_numpy(a).unsqueeze(0).to(args.device)
            else:
                aud_feed = None

            prompt = list(args.prompt.encode('utf-8', errors='replace'))
            out    = model.generate(prompt, args.n, img_feed=img_feed, aud_feed=aud_feed)
            gen_text = _printable(bytes(out).decode('utf-8', errors='replace'))
            cap_text = _printable(gen_caption[:120])

            lines = [f'GEN: {gen_text}', f'CAP: {cap_text}']

            # text-only generation (eyes closed, ears covered): no modality feeds.
            # Only worth a separate line when the sample clip actually drove a
            # modality — otherwise GEN above is already pure text.
            if has_imgs or has_auds:
                txt_out  = model.generate(prompt, args.n, img_feed=None, aud_feed=None)
                txt_text = _printable(bytes(txt_out).decode('utf-8', errors='replace'))
                lines.append(f'TXT-GEN: {txt_text}')
            if _tsample:                       # a recent pure text-only training example
                lines.append(f'TXT-SAMPLE: {_printable(_tsample[:120])}')
            print('\n' + '\n'.join(lines) + '\n')
            if not no_side_effects:
                with open(args.log, 'a') as f:
                    print('\n' + '\n'.join(lines) + '\n', file=f)

        # ── fetch batch ───────────────────────────────────────────────────────
        x_list, y_list, img_list, aud_list, flag_list = q.get()

        x    = torch.tensor(x_list,    dtype=torch.long).to(args.device)
        y    = torch.tensor(y_list,    dtype=torch.long).to(args.device)
        flag = torch.tensor(flag_list, dtype=torch.bool).to(args.device)
        img  = _to_img_tensor(img_list)
        aud  = _to_aud_tensor(aud_list)
        # per-example presence masks gate each modality (text-only rows → exactly 0)
        img_mask = torch.tensor([1.0 if v is not None else 0.0 for v in img_list],
                                device=args.device) if img is not None else None
        aud_mask = torch.tensor([1.0 if v is not None else 0.0 for v in aud_list],
                                device=args.device) if aud is not None else None

        # ── train step ────────────────────────────────────────────────────────
        dff_prev = model.dff                                    # save pre-step DFF
        model.train()
        logits, loss = model(x, img, aud, targets=y, flag=flag,
                             img_mask=img_mask, aud_mask=aud_mask)

        # halt immediately on non-finite loss — do NOT backward/step/save, so the
        # last good checkpoint on disk is preserved.
        if not torch.isfinite(loss):
            print(f'\n*** HALT: non-finite loss at step {i} ***')
            print(f'  loss     = {loss.item()}')
            print(f'  dff_max  = {model.dff.abs().max().item() if model.dff is not None else None}')
            print(f'  logit_max= {logits.detach().abs().max().item()}')
            print(f'  grads(prev step): {_diag(model)}')
            break

        loss.backward()

        total_norm = _gnorm_all(model)                         # grad norm BEFORE the step
        garr.append(total_norm)

        # halt before the optimizer corrupts weights if grads exploded
        if (not math.isfinite(total_norm)) or \
           (args.grad_stop is not None and total_norm > args.grad_stop):
            why = 'non-finite grad' if not math.isfinite(total_norm) \
                  else f'grad {total_norm:.3e} > --grad_stop {args.grad_stop:.3e}'
            print(f'\n*** HALT: {why} at step {i} ***')
            print(f'  dff_max  = {model.dff.abs().max().item() if model.dff is not None else None}')
            print(f'  logit_max= {logits.detach().abs().max().item()}')
            print(f'  per-module g(rad)/w(eight): {_diag(model)}')
            break

        if args.clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()
        model.dff = dff_prev                                    # restore so eval recomputes same step
        model.eval()
        with torch.no_grad():
            _, _ = model(x, img, aud, targets=y, flag=flag,    # recompute DFF using updated model
                         img_mask=img_mask, aud_mask=aud_mask)

        if args.vis:
            _vis_input_chars.append(_vc(y_list[0]))
            _vis_pred_chars.append(_vc(logits[0].argmax().item()))
            if len(_vis_input_chars) > _VIS_WIN:
                _vis_input_chars.pop(0)
                _vis_pred_chars.pop(0)

        # ── monitor ───────────────────────────────────────────────────────────
        larr.append(loss.item())
        a_std_arr.append(model.last_a_enc.std().item() if model.last_a_enc is not None else 0.0)
        i_std_arr.append(model.last_i_enc.std().item() if model.last_i_enc is not None else 0.0)
        if _mix is not None and img_mask is not None:   # split loss by visual vs text-only rows
            with torch.no_grad():
                _per = F.cross_entropy(logits.detach(), y.long(), reduction='none')
            _vm = img_mask > 0
            if _vm.any():     lav_arr.append(_per[_vm].mean().item())
            if (~_vm).any():  ltxt_arr.append(_per[~_vm].mean().item())

        if args.vis and model.dff is not None:
            dff_std_map = model.dff[0].cpu().detach().std(dim=0).numpy()  # [S, S]
            prob_map    = F.softmax(logits[0].detach().cpu(), dim=-1).numpy()
            prob_img    = prob_map.reshape(16, 16)                        # 256 bytes → 16×16

            _vis_header.set_text(
                _vis_stats_line + '\n'
                + 'in:   ' + ''.join(_vis_input_chars) + '\n'
                + 'pred: ' + ''.join(_vis_pred_chars)
            )

            if _vis_img_dff is None:
                _vis_img_dff = _vis_ax_dff.matshow(dff_std_map, cmap=args.cmap)
                _vis_ax_dff.set_xticks([]); _vis_ax_dff.set_yticks([])
            else:
                _vis_img_dff.set_data(dff_std_map)
            lo, hi = float(dff_std_map.min()), float(dff_std_map.max())
            _vis_img_dff.set_clim(lo, hi if hi > lo else lo + 1e-9)

            if _vis_img_prob is None:
                _vis_img_prob = _vis_ax_prob.matshow(prob_img, cmap=args.cmap,
                                                      vmin=0, vmax=1)
                _vis_ax_prob.set_xticks([]); _vis_ax_prob.set_yticks([])
            else:
                _vis_img_prob.set_data(prob_img)
            _vis_img_prob.set_clim(0, float(prob_img.max()) or 1e-9)

            _vis_fig.canvas.draw_idle()
            _vis_fig.canvas.flush_events()
            time.sleep(args.delay)
            if _vis_exit:
                break

        if (i % args.monitor) == 0:
            dff = model.dff
            s = ('STEP {:10} wall {} loss {:12.9f} grad {:12.6f} '
                 'lr {:10.9f} dff_mean {:12.5f} dff_std {:12.5f} dff_max {:11.3f} '
                 'a_std {:9.5f} i_std {:9.5f}').format(
                i, datetime.datetime.now(),
                np.mean(larr[-args.monitor:]),
                np.mean(garr[-args.monitor:]),
                scheduler.get_last_lr()[0],
                dff.mean().item() if dff is not None else 0.0,
                dff.std().item()  if dff is not None else 0.0,
                dff.abs().max().item() if dff is not None else 0.0,
                np.mean(a_std_arr[-args.monitor:]),
                np.mean(i_std_arr[-args.monitor:]),
            )
            if _mix is not None:
                _lt = np.mean(ltxt_arr[-args.monitor:]) if ltxt_arr else float('nan')
                _la = np.mean(lav_arr[-args.monitor:])  if lav_arr  else float('nan')
                s += f' loss_txt {_lt:12.9f} loss_av {_la:12.9f}'
            print(s)
            if not args.generate:
                with open(args.log, 'a') as f:
                    print(s, file=f)
            if args.debug:
                d = ('DEBUG step {:10} | {} | logit_max {:.3e}').format(
                    i, _diag(model), logits.detach().abs().max().item())
                print(d)
                if not args.generate:
                    with open(args.log, 'a') as f:
                        print(d, file=f)
            if args.vis:
                _vis_stats_line = (
                    'step {:8d}  loss {:7.4f}  lr {:8.2e}  dff_std {:6.4f}  a_std {:6.4f}  i_std {:6.4f}'
                ).format(
                    i,
                    np.mean(larr[-args.monitor:]),
                    scheduler.get_last_lr()[0],
                    dff.std().item() if dff is not None else 0.0,
                    np.mean(a_std_arr[-args.monitor:]),
                    np.mean(i_std_arr[-args.monitor:]),
                )

            # bound monitor history so host RAM stays flat over long runs (these are
            # only ever read as [-monitor:]; without this they grow once per step forever)
            for _a in (larr, garr, a_std_arr, i_std_arr, ltxt_arr, lav_arr):
                if len(_a) > args.monitor:
                    del _a[:-args.monitor]

        scheduler.step()
        optimizer.zero_grad()
        i += 1

        if args.steps is not None and i >= args.steps:
            break

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
