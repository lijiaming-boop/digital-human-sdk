import csv, statistics as st, sys

o,e=[],[]
with open('/home/hulushen/dh_lipsync_run/openness_energy.csv') as f:
    r=csv.DictReader(f)
    for row in r:
        o.append(float(row['openness'])); e.append(float(row['energy']))

n=len(o)
print('N=', n)
print('openness: min=%.2f max=%.2f mean=%.2f std=%.2f'%(min(o),max(o),st.mean(o),st.stdev(o)))
print('energy:   min=%.3f max=%.3f mean=%.3f std=%.3f'%(min(e),max(e),st.mean(e),st.stdev(e)))

mo=sum(o)/n; me=sum(e)/n
sxy=sum((o[i]-mo)*(e[i]-me) for i in range(n))
sxx=sum((o[i]-mo)**2 for i in range(n)); syy=sum((e[i]-me)**2 for i in range(n))
print('Pearson r (frame-level) = %.4f'%(sxy/(sxx*syy)**0.5))

# Sliding mean with window 5
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
mo=sum(os)/n; me=sum(es)/n
sxy=sum((os[i]-mo)*(es[i]-me) for i in range(n))
sxx=sum((os[i]-mo)**2 for i in range(n)); syy=sum((es[i]-me)**2 for i in range(n))
print('Pearson r (200ms smoothed) = %.4f'%(sxy/(sxx*syy)**0.5))

# Lag sweep
best=(0,0)
for lag in range(-30,31):
    if lag>=0:
        o2=o[lag:]; e2=e[:n-lag]
    else:
        o2=o[:n+lag]; e2=e[-lag:]
    L=min(len(o2),len(e2))
    if L<10: continue
    mo=sum(o2[:L])/L; me=sum(e2[:L])/L
    sxy=sum((o2[i]-mo)*(e2[i]-me) for i in range(L))
    sxx=sum((o2[i]-mo)**2 for i in range(L)); syy=sum((e2[i]-me)**2 for i in range(L))
    r=sxy/((sxx*syy)**0.5) if sxx>0 and syy>0 else 0
    if abs(r)>abs(best[1]): best=(lag,r)
print('Best |r| at lag=%d frames = %.4f (= %dms at 25fps)'%(best[0],best[1],best[0]*40))

# Per-segment: 6 segments of 5s each
print('Per-segment (5s each, frame-level r):')
for seg in range(6):
    a=seg*125; b=a+125
    if b>n: b=n
    if a>=n: break
    o3=o[a:b]; e3=e[a:b]
    L=len(o3)
    mo=sum(o3)/L; me=sum(e3)/L
    sxy=sum((o3[i]-mo)*(e3[i]-me) for i in range(L))
    sxx=sum((o3[i]-mo)**2 for i in range(L)); syy=sum((e3[i]-me)**2 for i in range(L))
    r=sxy/((sxx*syy)**0.5) if sxx>0 and syy>0 else 0
    print('  seg%d [%d-%d s]: openness mean=%.2f, energy mean=%.3f, r=%.3f'%(
        seg, seg*5, seg*5+5, mo, me, r))
