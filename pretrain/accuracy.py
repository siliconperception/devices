#BATCH            1 wall 2024-08-05 15:47:48.848148 lr 0.0001000000 wd 0.0100000000 batch     32 loss     1.675336     1.675336 grad     9.148475     9.148475
# generate nbatch vs. accuracy matplotlib chart
import argparse
import numpy as np ; print('numpy ' + np.__version__)
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--sigma', help='ylim',default=6.0, type=float)
parser.add_argument('--end', help='end batch',default=10000, type=int)
parser.add_argument('--log_resnet',help='log file name',default='log/log.2024.11.10-15.35.03')
parser.add_argument('--log_small',help='log file name',default='log/log.2024.11.11-07.51.44')
parser.add_argument('--log_medium',help='log file name',default='log/log.2024.11.14-08.18.35')
parser.add_argument('--log_large',help='log file name',default='log/log.2024.11.10-15.49.17')
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
        if l[0:5] == 'BATCH':
            b=[]
            for r in l[5:].split():
                try:
                    b.append(float(r))
                except:
                    b.append(0.0)
            a.append(b)
    return np.transpose(np.array(a))

print('loading log files')
arr_r = parselog(args.log_resnet)
print('arr_r',arr_r.shape)
arr_s = parselog(args.log_small)
print('arr_s',arr_s.shape)
arr_m = parselog(args.log_medium)
print('arr_m',arr_m.shape)
arr_l = parselog(args.log_large)
print('arr_l',arr_l.shape)

n_col=0
lr_col=5
bloss_col=11
loss_col=12
bgrad_col=14
grad_col=15
mse_col=19
fmapm_col=21
fmaps_col=22
refm_col=24
refs_col=25

fig = plt.figure(figsize=(10,40))
ax1 = fig.add_subplot(1,1,1)

ax1.plot(arr_r[n_col],arr_r[loss_col],'-k', linewidth=1,alpha=0.5,label='Resnet-18 (11.8M weights, 1.81G MAC)')
ax1.plot(arr_s[n_col],arr_s[loss_col],'-r', linewidth=1,alpha=0.5,label='IE120R-small (8.34M weights, 4.66G MAC)')
ax1.plot(arr_m[n_col],arr_m[loss_col],'-m', linewidth=1,alpha=0.5,label='IE120R-medium (13.12M weights, 6.54G MAC)')
ax1.plot(arr_l[n_col],arr_l[loss_col],'-g', linewidth=1,alpha=0.5,label='IE120R-large (32.17M weights, 8.12G MAC)')
ax1.set_ylabel('loss', color='k')
plt.legend(loc="upper right")


#ax1.set_yticks(np.arange(0,5))
#ax1.set_yscale('log')
#ax1b = ax1.twinx()
#ax2 = fig.add_subplot(3,1,2,sharex=ax1)
#ax2b = ax2.twinx()
#ax3 = fig.add_subplot(3,1,3,sharex=ax1)
#ax3b = ax3.twinx()
#ax1.set_ylim(np.mean(bloss)-args.sigma*np.std(bloss),np.mean(bloss)+args.sigma*np.std(bloss))
#ax1.set_ylim(0,np.mean(bloss)+args.sigma*np.std(bloss))
#ax1.set_ylim(bottom=0.0,top=10000.0)
#ax1.scatter(n,bloss[0:len(n)],marker='.',color='k',s=0.1,alpha=0.1)
#ax1.plot(n, loss[0:len(n)], '-g', linewidth=1,alpha=0.5)
#ax1.axhline(y=np.min(bloss[0:len(n)]), color='green', linestyle=':',linewidth=1,label='min')
##ax1b.plot(n, mse[0:len(n)], '-c', linewidth=1,alpha=0.5)
#ax1b.set_ylim(bottom=0.0,top=1.0)
#ax1b.scatter(n,mse[0:len(n)],marker='.',color='purple',s=0.1,alpha=0.1)
##ax2.set_ylim(0,grad[args.start])
#ax2.set_ylim(0,np.mean(bgrad)+args.sigma*np.std(bgrad))
#ax2.scatter(n,bgrad[0:len(n)],marker='.',color='k',s=0.1,alpha=0.1)
#ax2.plot(n, grad[0:len(n)], '-b', linewidth=1,alpha=0.5)
#ax2b.plot(n, lr[0:len(n)], '-k', linewidth=0.1,alpha=0.5)
#ax3.scatter(n,refm[0:len(n)],marker='.',color='k',s=0.1,alpha=0.1)
#ax3.scatter(n,refs[0:len(n)],marker='.',color='k',s=0.1,alpha=0.1)
#ax3.scatter(n,fmapm[0:len(n)],marker='.',color='r',s=0.1,alpha=0.1)
#ax3.scatter(n,fmaps[0:len(n)],marker='.',color='r',s=0.1,alpha=0.1)
##ax3.scatter(n,lr[0:len(n)],marker='.',color='r',s=0.2,alpha=1)
##ax3.plot(n,lr[0:len(n)], '-r', linewidth=1)
##ax3b.plot(n,mom[0:len(n)], '-m', linewidth=1)
##ax2.scatter(n,grad[0:len(n)],marker='.',color='b',s=0.1,alpha=0.8)
#ax1.set_ylabel('loss', color='g')
#ax1b.set_ylabel('mse', color='purple')
#ax2.set_ylabel('grad', color='b')
#ax2b.set_ylabel('lr', color='k')
#ax3.set_ylabel('mean,std', color='r')
#ax3.set_xlabel('batch')

plt.show()
exit()
