#step            0 train_loss 4.1681 val_loss 4.1682 sample NlasVZbSlXFQJzqjPmHmV.EkJ eEhLa?.Cu.eUPOeK&A. DMIkvbPXzQuDx&SohayA-.CdhaSEHiUWp-akH,pIYLvxpuHpAPgqT'ihQcVB?d-nb-v?-OzO3dtJSCiVsS-zyxA&JxxeXV!u.ErkK
import argparse
import numpy as np ; print('numpy ' + np.__version__)
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log',help='log file name',default='log')
parser.add_argument('--verbose', default=False, action='store_true')
args = parser.parse_args()
print(args)

def parselog(fn):
    f = open(fn,'r')
    a=[]
    while True:
        l = f.readline()
        if not l:
            break
        if l[0:4] == 'STEP':
            b=[]
            for r in l[4:].split():
                try:
                    b.append(float(r))
                except:
                    b.append(0.0)
                if len(b)==18:
                    break
            a.append(b)
    return np.transpose(np.array(a))

print('loading log file')
arr = parselog(args.log)
print('arr',arr.shape)

step=arr[1]
loss=arr[6]
grad=arr[8]
lr=arr[10]
mean=arr[12]
std=arr[14]
ex=arr[16]

#grad = np.clip(grad, 0, 10)

window_size = 10
weights = np.ones(window_size) / window_size
loss_mean = np.convolve(loss, weights, mode='same')

#fig = plt.figure(figsize=(10,40))
plt.style.use('dark_background')
fig = plt.figure()
nplots=4
ax1 = fig.add_subplot(nplots,1,1)
ax2 = fig.add_subplot(nplots,1,2, sharex=ax1)
ax3 = fig.add_subplot(nplots,1,3, sharex=ax1)
ax4 = fig.add_subplot(nplots,1,4, sharex=ax1)
ax1.plot(step, loss, '.w', linewidth=0.1,alpha=1.0, markersize=1)
#ax1.plot(step, loss_mean, '-w', linewidth=1,alpha=0.8)
ax1.axhline(y=np.min(loss), color='g', linestyle=':',linewidth=1,label='min')
ax2.plot(step, grad, '-y', linewidth=0.5,alpha=0.5)
ax3.plot(step, std, '-r', linewidth=0.5,alpha=0.5)
ax4.plot(step, lr, '-c', linewidth=0.5,alpha=0.5)

ax1.set_ylim(bottom=0)
ax1.set_xlabel('batch')
ax1.set_ylabel('loss')
ax2.set_xlabel('batch')
ax2.set_ylabel('grad')
ax3.set_xlabel('batch')
ax3.set_ylabel('std')
ax4.set_xlabel('batch')
ax4.set_ylabel('lr')
plt.show()
