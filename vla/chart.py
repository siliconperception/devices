# Copyright (c) 2024 Silicon Perception Inc (www.siliconperception.com)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import numpy as np ; print('numpy ' + np.__version__)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#from matplotlib.ticker import MultipleLocator
#from matplotlib.ticker import FuncFormatter
import re
import datetime

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--head', help='remove first head lines from log',default=0, type=int)
parser.add_argument('--log',help='log file name',default='log')
parser.add_argument('--grad_lim', default=1000, type=int,
                    help='scale the gradient plot y-axis so the last N grad samples fill ~80%% of it (0 = autoscale)')
parser.add_argument('--verbose', default=False, action='store_true')
args = parser.parse_args()
print(args)
batch_size=0

# STEP lines look like:
#   STEP 100 wall <date> <time> loss <v> grad <v> lr <v> dff_mean <v> dff_std <v> \
#        dff_max <v> dff_zeros <v> a_std <v> i_std <v>  [loss_txt <v> loss_av <v>]
# Parse each numeric field by name so the layout is robust to the date/time tokens
# and to optional mix-mode fields.
_FIELDS = ['loss', 'grad', 'lr', 'dff_mean', 'dff_std', 'dff_max', 'dff_zeros', 'a_std', 'i_std']
_NUM    = r'(-?[\d.]+(?:[eE][-+]?\d+)?)'

def parselog(fn):
    global batch_size
    step_pat = re.compile(r'^STEP\s+(\d+)')
    wall_pat = re.compile(r'wall\s+(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)')
    pats     = {k: re.compile(r'\b' + k + r'\s+' + _NUM) for k in _FIELDS}
    data     = {k: [] for k in ['step'] + _FIELDS}
    data['wall'] = []
    with open(fn, 'r') as f:
        for l in f:
            ms = step_pat.match(l)
            if ms:
                data['step'].append(float(ms.group(1)))
                mw = wall_pat.search(l)
                data['wall'].append(mw.group(1) if mw else '')
                for k in _FIELDS:
                    m = pats[k].search(l)
                    data[k].append(float(m.group(1)) if m else 0.0)
            elif l[0:4] == 'ARGS':
                match = re.search(r"batch=(\d+)", l)
                if match:
                    batch_size = int(match.group(1))
    return {k: np.array(v)[args.head:] for k, v in data.items()}

print('loading log file')
d = parselog(args.log)
print('steps', d['step'].shape[0])
print('batch_size', batch_size)

step = d['step']
loss = d['loss']
grad = d['grad']
lr   = d['lr']
mean = d['dff_mean']
std  = d['dff_std']
dmax = d['dff_max']
zero = d['dff_zeros']

# total wall-clock elapsed and average training steps/sec (from the STEP wall timestamps)
title = None
walls = d['wall']
if walls.size >= 2 and walls[0] and walls[-1]:
    fmt = '%Y-%m-%d %H:%M:%S.%f'
    try:
        t0 = datetime.datetime.strptime(str(walls[0]),  fmt)
        t1 = datetime.datetime.strptime(str(walls[-1]), fmt)
        elapsed = (t1 - t0).total_seconds()
        nsteps  = float(step[-1] - step[0])
        sps     = nsteps / elapsed if elapsed > 0 else 0.0
        title   = f'elapsed {datetime.timedelta(seconds=int(elapsed))}   {sps:.2f} steps/s'
    except ValueError:
        title = None
print('title:', title)

#grad = np.clip(grad, 0, 10)

window_size = 10
weights = np.ones(window_size) / window_size
loss_mean = np.convolve(loss, weights, mode='same')

#fig = plt.figure(figsize=(10,40))
plt.style.use('dark_background')
fig = plt.figure(figsize=(12, 14))
if title:
    fig.suptitle(title)
nplots=6
ax1 = fig.add_subplot(nplots,1,1)
ax2 = fig.add_subplot(nplots,1,2, sharex=ax1)
ax3 = fig.add_subplot(nplots,1,3, sharex=ax1)
ax3b = ax3.twinx()
ax4 = fig.add_subplot(nplots,1,4, sharex=ax1)
ax5 = fig.add_subplot(nplots,1,5, sharex=ax1)
ax6 = fig.add_subplot(nplots,1,6, sharex=ax1)
#ax1.set_ylim(bottom=0, top=np.log(50257))
ax1.set_ylim(bottom=0, top=np.log(256))
ax1.plot(step, loss, '.w', linewidth=0.1,alpha=1.0, markersize=1)
#ax1.plot(step, loss_mean, '-w', linewidth=1,alpha=0.8)
ax1.axhline(y=np.min(loss), color='g', linestyle='-',linewidth=1,label='min')
ax2.plot(step, grad, '-y', linewidth=2.0,alpha=0.5)
# scale the grad y-axis so the last --grad_lim samples' peak sits at ~80% of the axis
if args.grad_lim > 0 and grad.size:
    _gtop = float(np.max(grad[-args.grad_lim:]))
    if _gtop > 0:
        ax2.set_ylim(bottom=0, top=_gtop / 0.8)
#ax3.set_ylim(bottom=0, top=20)
#ax3b.set_ylim(bottom=-1, top=1)
ax3.plot(step, std, '-r', linewidth=1.0,alpha=0.5)
ax3b.plot(step, mean, '-b', linewidth=1.0,alpha=0.5)
ax4.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.7f'))
ax4.plot(step, lr, '-c', linewidth=2.0,alpha=0.5)
ax5.plot(step, dmax, '-m', linewidth=2.0,alpha=0.5)
ax6.set_ylim(bottom=0, top=1)
ax6.plot(step, zero, '-g', linewidth=2.0,alpha=0.5)

ax1.set_ylabel('loss', color='w')
ax2.set_ylabel('grad', color='y')
ax3.set_ylabel('std', color='r')
ax3b.set_ylabel('mean', color='b')
ax4.set_ylabel('lr', color='c')
ax5.set_ylabel('dff_max', color='m')
ax6.set_ylabel('sparsity', color='g')

# only the bottom subplot carries the shared x-axis label and tick labels
for ax in (ax1, ax2, ax3, ax3b, ax4, ax5):
    ax.tick_params(labelbottom=False)
ax6.set_xlabel('batch')

fig.tight_layout()
plt.show()
