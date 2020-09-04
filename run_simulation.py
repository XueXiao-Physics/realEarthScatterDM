#!/usr/bin/env python
# coding: utf-8

# In[1]:


from realES_methods import *
import matplotlib.pyplot as plt
import scipy.interpolate
import tqdm
import io


# In[2]:


mdm = 1e10
sige = 1e-28
rawN = 2**21
N = 10000
v0 = 0.02
r_init = 6371.

def init_sample(N,r):
    stheta = np.random.rand(N)*2*np.pi
    sr2 = np.random.rand(N)*r**2
    sr = np.sqrt(sr2)
    sx = sr*np.cos(stheta)
    sy = sr*np.sin(stheta)
    sz = -np.sqrt(r**2 - sx**2 - sy**2)
    return sx,sy,sz

sx,sy,sz = init_sample(N,r_init-1e-3)


# In[4]:


s = EarthEvents(mdm,sige)


s.load_Ktot()
s.calc_ndsigv2dlogEdlogq2rho()
s.cut_ndsigv2()
s.inSIG2rhos()

file = h5py.File('simulation1','w-')
for i in range(N):
    x0 = [sx[i],sy[i],sz[i]]
    
    s.direct_sample(N=2**21) 
    s.run_one(v0,x0)
    
    print(i+1,'\r',end='')
    file.create_dataset('path'+str(i),data = np.vstack([s.x.T,s.v]))
    file.flush()
file.close()

