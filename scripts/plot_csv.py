import csv, statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

o,e=[],[]
with open('/home/hulushen/dh_lipsync_run/openness_energy.csv') as f:
    r=csv.DictReader(f)
    for row in r:
        o.append(float(row['openness'])); e.append(float(row['energy']))

n=len(o)
t=[i/25.0 for i in range(n)]

# Smooth
def smooth(v,w=5):
    out=[]
    for i in range(len(v)):
        s=0;c=0
        for k in range(-w//2+1, w//2+1):
            j=i+k
            if 0<=j<len(v): s+=v[j]; c+=1
        out.append(s/c)
    return out
os=smooth(o); es=smooth(e)

fig,ax=plt.subplots(2,1,figsize=(14,7),sharex=True)
ax[0].plot(t,o,color='C0',alpha=0.35,label='openness (raw)')
ax[0].plot(t,os,color='C0',lw=2,label='openness (smooth 5)')
ax[0].plot(t,[min(o)+(max(o)-min(o))*(ev-min(e))/(max(e)-min(e)) for ev in es],
           color='C1',lw=1.5,alpha=0.7,label='energy (scaled)')
ax[0].set_ylabel('mouth change (0-255)')
ax[0].set_title('30s fitting: mouth openness vs audio energy')
ax[0].legend(loc='upper right')
ax[0].grid(alpha=0.3)

ax[1].plot(t,e,color='C1',lw=1.5,label='energy (raw)')
ax[1].plot(t,es,color='C1',lw=2.5,alpha=0.6,label='energy (smooth)')
ax[1].set_ylabel('audio RMS energy')
ax[1].set_xlabel('time (s)')
ax[1].legend(loc='upper right')
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/hulushen/dh_lipsync_run/openness_energy_plot.png', dpi=120)
print('Saved plot')

# Also: scatter openness vs energy
fig,ax=plt.subplots(1,1,figsize=(7,5))
ax.scatter(e,o,s=8,alpha=0.4,color='C0')
ax.set_xlabel('audio RMS energy'); ax.set_ylabel('mouth openness (0-255)')
ax.set_title('openness vs energy scatter (r=-0.18)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/hulushen/dh_lipsync_run/scatter.png', dpi=120)
print('Saved scatter')
