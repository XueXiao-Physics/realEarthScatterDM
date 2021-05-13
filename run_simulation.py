from realES import *
import matplotlib.pyplot as plt
import scipy.interpolate
import io


#1e9,1e-30,1e-3,100
input_params = input(' > mdm,sige,v0,N:\n > ')
input_vals = input_params.split(',')

mdm = float(input_vals[0])
sige = float(input_vals[1])
rawN = 2**21
N = int(input_vals[3])
v0 = float(input_vals[2])
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
s.calc_sum_ndsig2rho_v2dlnEdlnq()
s.cut_ndsigv2()
s.inSIG2rhos()

f = h5py.File('results/'+input_vals[0]+'_'+input_vals[1]+'_'+input_vals[2],'w-')
f.create_dataset('mdm',data = mdm)
f.create_dataset('sige',data = sige)
for i in range(N):
    x0 = [sx[i],sy[i],sz[i]]
    
    s.direct_sample(N=2**23) 
    s.run_one(v0,x0)
    
    print(i+1,'\r',end='')
    f.create_dataset('path'+str(i),data = np.vstack([s.x.T,s.v]))
    f.flush()
f.close()

