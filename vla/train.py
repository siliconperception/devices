#!/usr/bin/env python3
# train.py -- Multimodal CNN Language Model
# Silicon Perception Inc.

import torch ; print('torch', torch.__version__)
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as tv_models
import torchvision.transforms.functional as TVF
import torchaudio
import torchinfo
import numpy as np
import argparse
import os
import queue
import threading
import subprocess
import datetime
import random
import matplotlib.pyplot as plt

NULL  = 0x00   # no prediction this frame
START = 0x02   # start of caption
END   = 0x03   # end of caption
S     = 28     # fixed spatial size (ResNet-18 layer2)

# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# architecture
parser.add_argument('--c_text',        default=64,    type=int)
parser.add_argument('--c_audio',       default=32,    type=int)
parser.add_argument('--n_hidden',      default=128,   type=int)
parser.add_argument('--n_embd',        default=128,   type=int)
parser.add_argument('--proj_depth',    default=4,     type=int)
parser.add_argument('--kernel',        default=5,     type=int,
                    help='projection conv kernel size (odd integer)')
# audio
parser.add_argument('--audio_sr',      default=48000, type=int,
                    help='source audio sample rate for live input / demo')
parser.add_argument('--audio_work_sr', default=16000, type=int,
                    help='internal working sample rate')
# training
parser.add_argument('--dataset',       default='tiny',
                    help='tiny | c4 | librispeech | cc3m | webvid | mix')
parser.add_argument('--streaming',     default=False, action='store_true')
parser.add_argument('--no_image',      default=False, action='store_true')
parser.add_argument('--no_audio',      default=False, action='store_true')
parser.add_argument('--binary_audio',  default=False, action='store_true',
                    help='binary bit-plane audio encoder (24ch → ResNet-18 → 128ch)')
parser.add_argument('--native_fps',    default=False, action='store_true',
                    help='use source video fps; resample to 100 Hz')
parser.add_argument('--batch',         default=32,    type=int)
parser.add_argument('--steps',         default=None,  type=int)
parser.add_argument('--learning_rate', default=3e-4,  type=float)
parser.add_argument('--opt',           default='adamw')
parser.add_argument('--beta1',         default=0.9,   type=float)
parser.add_argument('--beta2',         default=0.95,  type=float)
parser.add_argument('--weight_decay',  default=0.01,  type=float)
parser.add_argument('--clip',          default=None,  type=float)
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
                    help='generate sample to stdout at each checkpoint (no file side effects)')
parser.add_argument('--n',             default=200,   type=int,
                    help='number of tokens for --generate samples')
parser.add_argument('--prompt',        default='\x02')
parser.add_argument('--seed',          default=None,  type=int)
parser.add_argument('--device',        default=None)
parser.add_argument('--verbose',       default=False, action='store_true')
parser.add_argument('--vis',           default=False, action='store_true',
                    help='live matshow of ctx std during training')
parser.add_argument('--delay',         default=0.1,   type=float,
                    help='seconds to pause per vis frame')
parser.add_argument('--cmap',          default='viridis',
                    help='matplotlib colormap for --vis')
args = parser.parse_args()

# ── restore architecture args from checkpoint ─────────────────────────────────
_ARCH_ARGS = ('c_text', 'c_audio', 'n_hidden', 'n_embd', 'proj_depth', 'kernel', 'audio_work_sr', 'binary_audio')
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

class TextEncoder(nn.Module):
    def __init__(self, c_text):
        super().__init__()
        self.embed = nn.Embedding(256, c_text)

    def forward(self, byte_idx):        # [B] int
        x = self.embed(byte_idx.long())                      # [B, c_text]
        return x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, S, S)  # [B, c_text, S, S]


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        rn = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            rn.conv1, rn.bn1, rn.relu, rn.maxpool, rn.layer1, rn.layer2,
        )
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, img):             # [B, 3, 224, 224] float in [0,1]
        return self.backbone((img - self.mean) / self.std)   # [B, 128, S, S]


class AudioEncoder(nn.Module):
    def __init__(self, c_audio, src_sr=48000, work_sr=16000):
        super().__init__()
        self.src_sr  = src_sr
        self.work_sr = work_sr
        self.c_audio = c_audio
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=work_sr, n_fft=512,
            win_length=work_sr // 100, hop_length=work_sr // 100,
            n_mels=S, f_min=60, f_max=work_sr // 2,
        )
        self.proj = nn.Conv2d(1, c_audio, kernel_size=1)

    def forward(self, audio_buf):       # [B, channels, n_samples] at src_sr
        x = audio_buf.mean(dim=1)      # [B, n_samples] mono
        if self.src_sr != self.work_sr:
            x = torchaudio.functional.resample(x, self.src_sr, self.work_sr)
        x = self.mel(x)                # [B, n_mels, n_frames]
        x = torch.log1p(x)
        t = x.shape[-1]
        x = x[..., -S:] if t >= S else F.pad(x, (S - t, 0))
        return self.proj(x.unsqueeze(1))   # [B, c_audio, S, S]


class BinaryAudioEncoder(nn.Module):
    """
    Maps a sliding window of 224×224 audio samples to a 24-channel binary
    "image", then encodes it through a frozen pretrained ResNet-18.

    Each sample occupies one pixel (x,y); each of N_BITS bit-planes becomes
    a channel.  Output: [B, 128, S, S] — same spatial format as ImageEncoder.
    """
    IMG    = 224
    N_BITS = 24

    def __init__(self, src_sr=16000, work_sr=16000):
        super().__init__()
        self.src_sr    = src_sr
        self.work_sr   = work_sr
        self.n_samples = self.IMG * self.IMG  # 50 176

        self.input_proj = nn.Conv2d(self.N_BITS, 3, kernel_size=1)

        rn = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            rn.conv1, rn.bn1, rn.relu, rn.maxpool, rn.layer1, rn.layer2,
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def _binary_image(self, x):   # [B, n_samples] float in [-1, 1]
        scale  = (1 << (self.N_BITS - 1)) - 1
        ints   = (x * scale).clamp(-scale, scale).long()
        shifts = torch.arange(self.N_BITS, device=x.device)       # [24]
        bits   = ((ints.unsqueeze(-1) >> shifts) & 1).float()     # [B, N, 24]
        B      = x.shape[0]
        return bits.permute(0, 2, 1).view(B, self.N_BITS, self.IMG, self.IMG)

    def forward(self, audio_buf):  # [B, ch, n_samples] at src_sr
        x = audio_buf.mean(dim=1)  # [B, n_samples] mono
        if self.src_sr != self.work_sr:
            x = torchaudio.functional.resample(x, self.src_sr, self.work_sr)
        N, n = self.n_samples, x.shape[-1]
        x = x[..., -N:] if n >= N else F.pad(x, (N - n, 0))

        img = self._binary_image(x)          # [B, 24, 224, 224]
        img = self.input_proj(img)            # [B,  3, 224, 224]
        img = (img - self.mean) / self.std
        return self.backbone(img)             # [B, 128, S, S]


class CNN_PROJECTOR(nn.Module):
    def __init__(self, in_ch, n_hidden, proj_depth=4, proj_kernel=5):
        super().__init__()
        pad = proj_kernel // 2
        layers, ch = [], in_ch
        for _ in range(proj_depth):
            layers += [nn.Conv2d(ch, n_hidden, proj_kernel, padding=pad), nn.ReLU()]
            ch = n_hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN_DECODER(nn.Module):
    """Pool variant: 1×1 conv → global average pool → linear."""
    def __init__(self, n_hidden, n_embd):
        super().__init__()
        self.conv   = nn.Conv2d(n_hidden, n_embd, 1)
        self.pool   = nn.AvgPool2d(S)
        self.lmhead = nn.Linear(n_embd, 256)

    def forward(self, x):
        x = self.pool(self.conv(x)).squeeze(-1).squeeze(-1)
        return self.lmhead(x)


class MultimodalCNN_LM(nn.Module):
    def __init__(self, c_text, c_audio, n_hidden, n_embd, proj_depth, proj_kernel=5,
                 src_sr=48000, work_sr=16000, binary_audio=False):
        super().__init__()
        self.n_hidden     = n_hidden
        self.c_audio      = c_audio
        self.binary_audio = binary_audio

        self.text_encoder  = TextEncoder(c_text)
        self.image_encoder = ImageEncoder()
        if binary_audio:
            self.audio_encoder = BinaryAudioEncoder(src_sr=work_sr, work_sr=work_sr)
            audio_ch = 128
        else:
            self.audio_encoder = AudioEncoder(c_audio, src_sr, work_sr)
            audio_ch = c_audio

        self.audio_ch  = audio_ch
        in_ch = c_text + 128 + audio_ch + n_hidden
        self.projector = CNN_PROJECTOR(in_ch, n_hidden, proj_depth, proj_kernel)
        self.decoder   = CNN_DECODER(n_hidden, n_embd)

        self.ctx = None   # [B, n_hidden, S, S]; allocated on first forward

    def _init_ctx(self, B, device):
        if self.ctx is None or self.ctx.shape[0] != B or self.ctx.device != device:
            self.ctx = torch.zeros(B, self.n_hidden, S, S, device=device)

    def forward(self, byte_idx, img, aud_buf, targets=None, flag=None):
        """
        byte_idx  [B] int
        img       [B,3,224,224] float in [0,1], or None → zeros
        aud_buf   [B,1,n_samples] float, or None → zeros
        flag      [B] bool — zero ctx for these batch elements before this step
        """
        B, dev = byte_idx.shape[0], byte_idx.device
        self._init_ctx(B, dev)

        if flag is not None and flag.any():
            self.ctx[flag] = 0.0

        t_enc = self.text_encoder(byte_idx)
        i_enc = self.image_encoder(img) if img is not None \
                else torch.zeros(B, 128, S, S, device=dev)
        a_enc = self.audio_encoder(aud_buf) if aud_buf is not None \
                else torch.zeros(B, self.audio_ch, S, S, device=dev)

        new_ctx = self.projector(torch.cat([t_enc, i_enc, a_enc, self.ctx], dim=1))
        self.ctx = new_ctx.detach().clone()

        logits = self.decoder(new_ctx)
        loss   = F.cross_entropy(logits, targets.long()) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt_bytes, n_tokens, img_feed=None, aud_feed=None):
        """Autoregressive generation. Uses a private ctx; does not touch self.ctx."""
        self.eval()
        dev = next(self.parameters()).device
        ctx = torch.zeros(1, self.n_hidden, S, S, device=dev)

        def _step(bval):
            nonlocal ctx
            bi = torch.tensor([bval], device=dev)
            t  = self.text_encoder(bi)
            ie = self.image_encoder(img_feed()) \
                 if img_feed else torch.zeros(1, 128, S, S, device=dev)
            ae = self.audio_encoder(aud_feed()) \
                 if aud_feed else torch.zeros(1, self.audio_ch, S, S, device=dev)
            ctx = self.projector(torch.cat([t, ie, ae, ctx], dim=1)).detach().clone()
            return self.decoder(ctx)

        bval = START
        for b in prompt_bytes:
            _step(bval)
            bval = b

        out = []
        for _ in range(n_tokens):
            logits = _step(bval)
            bval   = torch.multinomial(F.softmax(logits[0], dim=-1), 1).item()
            if bval == END:
                break
            if bval != NULL:
                out.append(bval)

        return out


# ── caption alignment ─────────────────────────────────────────────────────────

def align_caption(caption_bytes, n_steps):
    """
    Map caption bytes onto n_steps target slots at 1 char/step from step 0.
    Remaining steps (and steps beyond the caption) are NULL (\x00).
    full_seq = [START] + caption_bytes + [END]; returned as (inputs, targets)
    where inputs[t] feeds the model and targets[t] is the CE target.
    """
    full = [START] + list(caption_bytes) + [END]
    inputs  = []
    targets = []
    for t in range(n_steps):
        inputs.append( full[t]     if t     < len(full) else NULL)
        targets.append(full[t + 1] if t + 1 < len(full) else NULL)
    return inputs, targets


# ── clip builders ─────────────────────────────────────────────────────────────
# Each builder returns (steps, caption_str).
# steps: list of (x_byte, y_byte, img_ndarray_or_None, aud_ndarray_or_None)
# one tuple per 100 Hz step.

def _text_to_bytes(text):
    return list(text.encode('utf-8', errors='replace'))


def clip_text(text):
    """Text-only: one step per byte, no img/aud."""
    cap   = _text_to_bytes(text)
    n     = len(cap) + 1   # START + cap + END = len(cap)+2 bytes, len(cap)+1 steps
    ins, tgts = align_caption(cap, n)
    return [(x, y, None, None) for x, y in zip(ins, tgts)], text


def clip_librispeech(example, work_sr=16000, audio_window=None):
    """Audio + transcript. Audio defines duration; transcript emitted 1 char/step."""
    import io, soundfile as sf
    aud_info = example['audio']
    if 'array' in aud_info and aud_info['array'] is not None:
        audio = aud_info['array'].astype(np.float32)
        sr    = aud_info['sampling_rate']
    elif aud_info.get('bytes') is not None:
        arr, sr = sf.read(io.BytesIO(aud_info['bytes']))
        audio = (arr.mean(axis=1) if arr.ndim == 2 else arr).astype(np.float32)
    elif aud_info.get('path') is not None:
        arr, sr = sf.read(aud_info['path'])
        audio = (arr.mean(axis=1) if arr.ndim == 2 else arr).astype(np.float32)
    else:
        raise ValueError(f'audio example has no usable field: {list(aud_info.keys())}')
    text  = example['text'].strip()

    hop_src  = max(1, round(sr / 100))            # samples per step at src rate
    n_steps  = max(1, len(audio) // hop_src)
    hop_work = work_sr // 100

    # Resample to work_sr for building windows that AudioEncoder will receive
    if sr != work_sr:
        a_t   = torch.from_numpy(audio).unsqueeze(0)
        audio = torchaudio.functional.resample(a_t, sr, work_sr).squeeze(0).numpy()
        sr    = work_sr
        hop_src = hop_work

    win_size = audio_window if audio_window is not None else S * hop_src
    # Pad left so window at step 0 is all zeros before the signal
    pad   = np.zeros(win_size - hop_src, dtype=np.float32)
    audio = np.concatenate([pad, audio])

    cap          = _text_to_bytes(text)
    ins, tgts    = align_caption(cap, n_steps)
    steps        = []
    for t in range(n_steps):
        ws  = t * hop_src
        win = audio[ws:ws + win_size]
        if len(win) < win_size:
            win = np.pad(win, (0, win_size - len(win)))
        aud_buf = win.reshape(1, -1).astype(np.float32)   # [1, win_size]
        steps.append((ins[t], tgts[t], None, aud_buf))
    return steps, text


def clip_image_caption(img_arr, caption_text):
    """Single image, caption emitted one char per step. Used for cc3m / obelics."""
    cap         = _text_to_bytes(caption_text)
    n           = len(cap) + 1
    ins, tgts   = align_caption(cap, n)
    # Same image for all steps; resize to [3,224,224] uint8
    if img_arr is None:
        frame = np.zeros((3, 224, 224), dtype=np.uint8)
    else:
        import PIL.Image
        if not isinstance(img_arr, np.ndarray):
            img_arr = np.array(img_arr)
        if img_arr.ndim == 2:
            img_arr = np.stack([img_arr] * 3, axis=-1)
        pil = PIL.Image.fromarray(img_arr).resize((224, 224))
        frame = np.array(pil).transpose(2, 0, 1)   # [3,224,224]
    return [(x, y, frame, None) for x, y in zip(ins, tgts)], caption_text


def clip_webvid(example, work_sr=16000, audio_window=None, **kwargs):
    """
    TODO: decode video + audio from a webvid HF dataset example.
    Expected example keys: 'video' (bytes or path), 'name' (caption).
    Returns list of (x_byte, y_byte, frame_ndarray, aud_ndarray) at 100 Hz.
    Falls back to text-only if video decoding is unavailable.
    """
    caption = example.get('name', example.get('caption', ''))
    try:
        import av
        # example['video'] may be a dict with 'bytes' or a local path
        vid_data = example.get('video', None)
        if vid_data is None:
            raise ValueError('no video field')
        if isinstance(vid_data, dict):
            vid_bytes = vid_data.get('bytes')
            import io
            container = av.open(io.BytesIO(vid_bytes))
        else:
            container = av.open(vid_data)

        # --- video frames at 100 Hz ---
        vstream  = container.streams.video[0]
        fps_src  = float(vstream.average_rate)
        frames_v = []
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format='rgb24')   # [H,W,3] uint8
            import PIL.Image
            pil = PIL.Image.fromarray(img).resize((224, 224))
            frames_v.append(np.array(pil).transpose(2, 0, 1))  # [3,224,224]

        # Resample frames from fps_src to 100 Hz
        n_steps = max(1, round(len(frames_v) * 100.0 / fps_src))
        frames_100hz = []
        for t in range(n_steps):
            src_idx = min(int(round(t * fps_src / 100.0)), len(frames_v) - 1)
            frames_100hz.append(frames_v[src_idx])

        # --- audio at work_sr ---
        hop_work = work_sr // 100
        astream  = container.streams.audio[0] if container.streams.audio else None
        if astream is not None:
            chunks = []
            for aframe in container.decode(audio=0):
                a = aframe.to_ndarray()           # [channels, samples]
                chunks.append(a.mean(0).astype(np.float32))
            audio_raw = np.concatenate(chunks) if chunks else np.zeros(n_steps * hop_work, dtype=np.float32)
            a_sr = astream.sample_rate
            if a_sr != work_sr:
                a_t      = torch.from_numpy(audio_raw).unsqueeze(0)
                audio_raw = torchaudio.functional.resample(a_t, a_sr, work_sr).squeeze(0).numpy()
        else:
            audio_raw = np.zeros(n_steps * hop_work, dtype=np.float32)

        win_size = audio_window if audio_window is not None else S * hop_work
        pad   = np.zeros(win_size - hop_work, dtype=np.float32)
        audio = np.concatenate([pad, audio_raw])

        cap       = _text_to_bytes(caption)
        ins, tgts = align_caption(cap, n_steps)
        steps     = []
        for t in range(n_steps):
            ws  = t * hop_work
            win = audio[ws:ws + win_size]
            if len(win) < win_size:
                win = np.pad(win, (0, win_size - len(win)))
            steps.append((ins[t], tgts[t], frames_100hz[t], win.reshape(1, -1)))
        return steps, caption

    except Exception as e:
        print(f'webvid decode failed ({e}), falling back to text-only')
        return clip_text(caption)


# ── dataset loading ────────────────────────────────────────────────────────────
# Datasets are loaded in the main thread BEFORE model.to(device) to avoid
# SIGABRT from CUDA + fork: HF load_dataset spawns child processes to load
# shards; forking after CUDA init corrupts GPU state.

_TEXT_DATASETS = {
    'tiny': ('roneneldan/TinyStories', None,  'train',     'text'),
    'c4':   ('allenai/c4',             'en',  'train',     'text'),
    'web':  ('Skylion007/openwebtext', None,  'train',     'text'),
}

_MIX_SOURCES = [
    ('roneneldan/TinyStories', None, 'train', 0.1),
    ('allenai/c4',             'en', 'train', 0.9),
]

_DS_SPEC = {   # dataset name → (hf_name, hf_config, default_split)
    'librispeech': ('openslr/librispeech_asr',                       'clean', 'train.100'),
    'cc3m':        ('google-research-datasets/conceptual_captions',  None,    'train'),
    'webvid':      ('TempoFunk/webvid-10M',                          None,    'train'),
}


def _init_datasets(args):
    """Load HF dataset objects (no iteration). Returns list of loaded datasets."""
    from datasets import load_dataset, Audio
    if args.dataset in _TEXT_DATASETS:
        name, cfg, _, _ = _TEXT_DATASETS[args.dataset]
        return [load_dataset(name, cfg, streaming=args.streaming)]
    elif args.dataset == 'mix':
        return [load_dataset(n, c, streaming=args.streaming)
                for n, c, *_ in _MIX_SOURCES]
    elif args.dataset in _DS_SPEC:
        name, cfg, _ = _DS_SPEC[args.dataset]
        ds = load_dataset(name, cfg, streaming=args.streaming)
        if args.dataset in ('librispeech', 'cc3m', 'webvid'):
            # Disable HF's auto-decode (requires torchcodec); decode manually with soundfile
            if hasattr(ds, 'cast_column'):
                try:
                    ds = ds.cast_column('audio', Audio(decode=False))
                except Exception:
                    pass
        return [ds]
    else:
        raise ValueError(f'unknown dataset: {args.dataset}')


def _make_iter(ds, split, streaming, seed):
    """Create a (possibly reshuffled) iterator from an already-loaded dataset."""
    split_ds = ds[split] if split in ds else ds['train']
    if streaming:
        return iter(split_ds.shuffle(buffer_size=10000, seed=seed))
    else:
        return iter(split_ds.shuffle(seed=seed))


def _prebuild_initial_iters(hf_datasets, args):
    """Pre-build initial iterators before CUDA init to avoid fork-after-CUDA SIGABRT."""
    if args.dataset == 'mix':
        return [_make_iter(hf_datasets[i], _MIX_SOURCES[i][2],
                           args.streaming, args.seed)
                for i in range(len(_MIX_SOURCES))]
    elif args.dataset in _TEXT_DATASETS:
        split = _TEXT_DATASETS[args.dataset][2]
        return [_make_iter(hf_datasets[0], split, args.streaming, args.seed)]
    elif args.dataset in _DS_SPEC:
        split = _DS_SPEC[args.dataset][2]
        return [_make_iter(hf_datasets[0], split, args.streaming, args.seed)]
    return [None]


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

    def done(self):
        return self.pos >= len(self.steps)

    def is_start(self):
        return self.pos == 0

    def advance(self):
        item     = self.steps[self.pos]
        self.pos += 1
        return item   # (x_byte, y_byte, img_or_None, aud_or_None)


GEN_SAMPLE_STEPS = 200   # max steps stored per generation sample

def worker(stop, q, gen_q, hf_datasets, initial_iters, args):
    epoch = [1]   # mutable so nested fns can increment

    # resolve split name for the primary dataset
    if args.dataset in _TEXT_DATASETS:
        _split = _TEXT_DATASETS[args.dataset][2]
    elif args.dataset in _DS_SPEC:
        _split = _DS_SPEC[args.dataset][2]
    else:
        _split = 'train'

    def _mk_iter(ds_idx=0):
        epoch[0] += 1
        return _make_iter(hf_datasets[ds_idx], _split,
                          args.streaming, args.seed + epoch[0])

    if args.dataset == 'mix':
        mix_weights = [w for *_, w in _MIX_SOURCES]
        mix_iters   = initial_iters   # pre-built before CUDA init
        data_iter   = None
    else:
        data_iter = initial_iters[0]   # pre-built before CUDA init

    def _next_example():
        nonlocal data_iter
        if args.dataset == 'mix':
            idx = random.choices(range(len(mix_iters)), weights=mix_weights, k=1)[0]
            try:
                return next(mix_iters[idx])
            except StopIteration:
                epoch[0] += 1
                mix_iters[idx] = _make_iter(hf_datasets[idx], _MIX_SOURCES[idx][2],
                                             args.streaming, args.seed + epoch[0])
                return next(mix_iters[idx])
        else:
            try:
                return next(data_iter)
            except StopIteration:
                data_iter = _mk_iter()
                return next(data_iter)

    _audio_window = (BinaryAudioEncoder.IMG ** 2) if args.binary_audio else None

    def _build_clip(example):
        # returns (steps, caption)
        col = _TEXT_DATASETS.get(args.dataset, (None, None, 'text'))[2]
        if args.dataset in _TEXT_DATASETS or args.dataset == 'mix':
            text = example.get('text', example.get(col, ''))
            return clip_text(text)
        elif args.dataset == 'librispeech':
            if args.no_audio:
                return clip_text(example['text'])
            return clip_librispeech(example, work_sr=args.audio_work_sr,
                                    audio_window=_audio_window)
        elif args.dataset == 'cc3m':
            if args.no_image:
                return clip_text(example.get('caption', ''))
            img = example.get('image', None)
            cap = example.get('caption', '')
            return clip_image_caption(img, cap)
        elif args.dataset == 'webvid':
            if args.no_image and args.no_audio:
                return clip_text(example.get('name', example.get('caption', '')))
            return clip_webvid(example, work_sr=args.audio_work_sr,
                               audio_window=_audio_window)
        return [], ''

    slots = [_ClipSlot() for _ in range(args.batch)]

    while not stop.is_set():
        # fill any exhausted slots
        for slot in slots:
            while slot.done():
                ex            = _next_example()
                steps, caption = _build_clip(ex)
                if steps:
                    slot.load(steps, caption)
                    # offer this clip as a generation sample (drop if queue full)
                    try:
                        gen_q.put_nowait((caption, steps[:GEN_SAMPLE_STEPS]))
                    except queue.Full:
                        pass

        # gather one step from every slot
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

print('loading datasets...')
hf_datasets = _init_datasets(args)
# Pre-build initial iterators before CUDA init to avoid fork-after-CUDA SIGABRT
initial_iters = _prebuild_initial_iters(hf_datasets, args)
print('datasets ready')

# ── model instantiation ───────────────────────────────────────────────────────

model = MultimodalCNN_LM(
    c_text       = args.c_text,
    c_audio      = args.c_audio,
    n_hidden     = args.n_hidden,
    n_embd       = args.n_embd,
    proj_depth   = args.proj_depth,
    proj_kernel  = args.kernel,
    src_sr       = args.audio_work_sr,   # worker provides buffers at work_sr
    work_sr      = args.audio_work_sr,
    binary_audio = args.binary_audio,
)

if _loaded_ckpt is not None:
    _sd = _loaded_ckpt['state_dict'] if isinstance(_loaded_ckpt, dict) and 'state_dict' in _loaded_ckpt else _loaded_ckpt
    model.load_state_dict(_sd)
    del _loaded_ckpt

_B1       = 1
_hop      = args.audio_work_sr // 100
_aud_win  = (BinaryAudioEncoder.IMG ** 2) if args.binary_audio else S * _hop
print(torchinfo.summary(
    model,
    col_names=['input_size', 'output_size', 'num_params'],
    input_data=[
        torch.zeros([_B1], dtype=torch.int32),
        torch.zeros([_B1, 3, 224, 224]),
        torch.zeros([_B1, 1, _aud_win]),
    ],
))

model = model.to(args.device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 'M trainable params')

_vis_exit = False
if args.vis:
    args.batch = 1
    plt.ion()
    _vis_fig = plt.figure()
    _vis_ax  = _vis_fig.add_subplot(1, 1, 1)
    _vis_img = None
    def _on_key(event):
        global _vis_exit
        if event.key in ('x', 'X'):
            _vis_exit = True
    _vis_fig.canvas.mpl_connect('key_press_event', _on_key)

# ── optimizer ─────────────────────────────────────────────────────────────────

_audio_trained = (model.audio_encoder.input_proj.parameters()
                  if args.binary_audio
                  else model.audio_encoder.proj.parameters())
param_groups = [
    {'params': model.projector.parameters()},
    {'params': model.decoder.parameters()},
    {'params': model.text_encoder.parameters()},
    {'params': _audio_trained},
]

if args.opt == 'adamw':
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate,
                                  betas=(args.beta1, args.beta2),
                                  weight_decay=args.weight_decay)
elif args.opt == 'sgd':
    optimizer = torch.optim.SGD(param_groups, lr=args.learning_rate, momentum=args.beta1)
elif args.opt == 'rms':
    optimizer = torch.optim.RMSprop(param_groups, lr=args.learning_rate,
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
gen_q = queue.Queue(maxsize=2)   # generation samples from worker
w     = threading.Thread(target=worker, args=[stop, q, gen_q, hf_datasets, initial_iters, args], daemon=False)
w.start()

# ── helpers ───────────────────────────────────────────────────────────────────

def _to_img_tensor(img_list):
    """img_list: list of [3,224,224] uint8 ndarray or None. Returns tensor or None."""
    if args.no_image or all(x is None for x in img_list):
        return None
    frames = []
    for f in img_list:
        if f is None:
            frames.append(np.zeros((3, 224, 224), dtype=np.uint8))
        else:
            frames.append(f)
    return torch.from_numpy(np.stack(frames)).float().div(255.0).to(args.device)


def _to_aud_tensor(aud_list):
    """aud_list: list of [1, n_samples] float32 ndarray or None. Returns tensor or None."""
    if args.no_audio or all(x is None for x in aud_list):
        return None
    bufs = []
    n = max(a.shape[-1] for a in aud_list if a is not None)
    for a in aud_list:
        if a is None:
            bufs.append(np.zeros((1, n), dtype=np.float32))
        else:
            if a.shape[-1] < n:
                a = np.pad(a, ((0, 0), (0, n - a.shape[-1])))
            bufs.append(a[:, :n])
    return torch.from_numpy(np.stack(bufs)).to(args.device)


def _printable(s):
    s = s.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
    return ''.join(c for c in s if c.isprintable())

# ── training loop ─────────────────────────────────────────────────────────────

larr, garr = [], []
i = 0

try:
    while True:
        if (i % args.checkpoint) == 0:
            no_side_effects = args.vis or args.generate
            if not no_side_effects:
                torch.save({'saved_args': vars(args), 'state_dict': model.state_dict()}, args.save)

            # generation sample conditioned on a training clip
            model.eval()

            # drain gen_q, keep the most recent sample
            gen_caption, gen_steps = '', []
            while True:
                try:
                    gen_caption, gen_steps = gen_q.get_nowait()
                except queue.Empty:
                    break

            # build img_feed / aud_feed from the sample clip
            has_imgs = not args.no_image and any(s[2] is not None for s in gen_steps)
            has_auds = not args.no_audio and any(s[3] is not None for s in gen_steps)

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
                        return torch.zeros(1, 1, args.audio_work_sr // 100 * S,
                                           device=args.device)
                    return torch.from_numpy(a).unsqueeze(0).to(args.device)
            else:
                aud_feed = None

            prompt = list(args.prompt.encode('utf-8', errors='replace'))
            out    = model.generate(prompt, args.n, img_feed=img_feed, aud_feed=aud_feed)
            gen_text = _printable(bytes(out).decode('utf-8', errors='replace'))
            cap_text = _printable(gen_caption)

            lines = [f'GEN: {gen_text}', f'CAP: {cap_text}']
            print('\n' + '\n'.join(lines) + '\n')
            if not no_side_effects:
                with open(args.log, 'a') as f:
                    print('\n' + '\n'.join(lines) + '\n', file=f)

        # fetch batch
        x_list, y_list, img_list, aud_list, flag_list = q.get()

        x    = torch.tensor(x_list,    dtype=torch.long).to(args.device)
        y    = torch.tensor(y_list,    dtype=torch.long).to(args.device)
        flag = torch.tensor(flag_list, dtype=torch.bool).to(args.device)
        img  = _to_img_tensor(img_list)
        aud  = _to_aud_tensor(aud_list)

        # train step
        model.train()
        logits, loss = model(x, img, aud, targets=y, flag=flag)
        loss.backward()
        if args.clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # recompute ctx under updated weights (same input, no re-reset)
        model.eval()
        with torch.no_grad():
            model(x, img, aud)

        # monitor
        larr.append(loss.item())
        total_norm = sum(p.grad.detach().norm(2).item() ** 2
                         for p in model.parameters()
                         if p.grad is not None and p.requires_grad) ** 0.5
        garr.append(total_norm)

        if args.vis and model.ctx is not None:
            f = model.ctx[0].cpu().detach().std(dim=0).numpy()
            if _vis_img is None:
                _vis_img = _vis_ax.matshow(f, cmap=args.cmap)
            else:
                _vis_img.set_data(f)
            _vis_ax.set_title('step {:d}  loss {:.4f}'.format(i, loss.item()))
            plt.draw()
            plt.pause(args.delay)
            if _vis_exit:
                break

        if (i % args.monitor) == 0:
            ctx_t = model.ctx
            s = ('STEP {:10} wall {} loss {:12.9f} grad {:12.6f} '
                 'lr {:10.9f} ctx_mean {:12.5f} ctx_std {:12.5f}').format(
                i, datetime.datetime.now(),
                np.mean(larr[-args.monitor:]),
                np.mean(garr[-args.monitor:]),
                scheduler.get_last_lr()[0],
                ctx_t.mean().item() if ctx_t is not None else 0.0,
                ctx_t.std().item()  if ctx_t is not None else 0.0,
            )
            print(s)
            if not args.generate:
                with open(args.log, 'a') as f:
                    print(s, file=f)

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
