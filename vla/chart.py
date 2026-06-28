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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--head', help='remove first head lines from log',default=0, type=int)
parser.add_argument('--log',help='log file name',default='log')
parser.add_argument('--verbose', default=False, action='store_true')
args = parser.parse_args()
print(args)
batch_size=0

# STEP lines look like:
#   STEP 100 wall <date> <time> loss <v> grad <v> lr <v> dff_mean <v> dff_std <v> \
#        dff_max <v> a_std <v> i_std <v>  [loss_txt <v> loss_av <v>]
# Parse each numeric field by name so the layout is robust to the date/time tokens
# and to optional mix-mode fields.
_FIELDS = ['loss', 'grad', 'lr', 'dff_mean', 'dff_std', 'dff_max', 'a_std', 'i_std']
_NUM    = r'(-?[\d.]+(?:[eE][-+]?\d+)?)'

def parselog(fn):
    global batch_size
    step_pat = re.compile(r'^STEP\s+(\d+)')
    pats     = {k: re.compile(r'\b' + k + r'\s+' + _NUM) for k in _FIELDS}
    data     = {k: [] for k in ['step'] + _FIELDS}
    with open(fn, 'r') as f:
        for l in f:
            ms = step_pat.match(l)
            if ms:
                data['step'].append(float(ms.group(1)))
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

#grad = np.clip(grad, 0, 10)

window_size = 10
weights = np.ones(window_size) / window_size
loss_mean = np.convolve(loss, weights, mode='same')

#fig = plt.figure(figsize=(10,40))
plt.style.use('dark_background')
fig = plt.figure()
nplots=5
ax1 = fig.add_subplot(nplots,1,1)
ax2 = fig.add_subplot(nplots,1,2, sharex=ax1)
ax3 = fig.add_subplot(nplots,1,3, sharex=ax1)
ax3b = ax3.twinx()
ax4 = fig.add_subplot(nplots,1,4, sharex=ax1)
ax5 = fig.add_subplot(nplots,1,5, sharex=ax1)
#ax1.set_ylim(bottom=0, top=np.log(50257))
ax1.set_ylim(bottom=0, top=np.log(256))
ax1.plot(step, loss, '.w', linewidth=0.1,alpha=1.0, markersize=1)
#ax1.plot(step, loss_mean, '-w', linewidth=1,alpha=0.8)
ax1.axhline(y=np.min(loss), color='g', linestyle='-',linewidth=1,label='min')
#ax2.set_ylim(bottom=0, top=20)
ax2.plot(step, grad, '-y', linewidth=2.0,alpha=0.5)
#ax3.set_ylim(bottom=0, top=20)
#ax3b.set_ylim(bottom=-1, top=1)
ax3.plot(step, std, '-r', linewidth=1.0,alpha=0.5)
ax3b.plot(step, mean, '-b', linewidth=1.0,alpha=0.5)
ax4.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.7f'))
ax4.plot(step, lr, '-c', linewidth=2.0,alpha=0.5)
ax5.plot(step, dmax, '-m', linewidth=2.0,alpha=0.5)

ax1.set_xlabel('batch')
ax1.set_ylabel('loss', color='w')
ax2.set_xlabel('batch')
ax2.set_ylabel('grad', color='y')
ax3.set_xlabel('batch')
ax3.set_ylabel('std', color='r')
ax3b.set_ylabel('mean', color='b')
ax4.set_xlabel('batch')
ax4.set_ylabel('lr', color='c')
ax5.set_xlabel('batch')
ax5.set_ylabel('dff_max', color='m')
plt.show()
