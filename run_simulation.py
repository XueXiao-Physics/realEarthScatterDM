from realES import *
import matplotlib.pyplot as plt
import scipy.interpolate
import io
import os
try:
    import multiprocess
except:
    import multiprocessing as multiprocess
import math

import json
with io.open('settings.txt') as f:
    settings = json.load(f)
ncores = int(settings['ncores'])


#5e7,1e-28,1e-3,10
input_params = input(' > mdm,sige,v0,N:\n > ')
input_vals = input_params.split(',')

mdm = float(input_vals[0])
sige = float(input_vals[1])
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




s = EarthEvents(mdm,sige)


s.load_Ktot()
s.calc_sum_ndsig2rho_v2dlnEdlnq(0)
s.inSIG2rhos()

input(' >>> Press Enter to Start Simulaton. <<<')
# create file if not exist

Filename = 'results/'+input_vals[0]+'_'+input_vals[1]+'_'+input_vals[2]
try:
    f = h5py.File(Filename,'w-')
    f.create_dataset('mdm',data = mdm)
    f.create_dataset('sige',data = sige)
    N0 = 0
    f.close()
except OSError:
    pass

# open file
# get N0
with  h5py.File(Filename,'a') as f:
    keynames = list(filter(lambda i:i[:4]=='path',f.keys()) )
    if len(keynames) != 0:
        nums = [int(k[4:]) for k in keynames]
        N0 = max(nums)
    else:
        N0 = 0



# multiprocess

def single_job(_Nstart,_N,_icore):	
    with  h5py.File(Filename+'_'+str(icore),'w') as f:
        for i in range(_Nstart,_Nstart+_N):
            x0 = [sx[i],sy[i],sz[i]]
            
            s.run_one(v0,x0)
            
            print(i+1+N0,'\r',end='')
            
            f.create_dataset('path'+str(i+1+N0),data = np.vstack([s.x.T,s.v,s.ss]))
            #f.flush()
        print('\n')



ncores = ncores # cpu cores 
jobs = []
job_add = np.zeros(ncores,dtype=int)
job_add[:N%ncores] =1 
job_assign = np.ones(ncores,dtype=int)*(N//ncores) + job_add
job_starts =  np.concatenate( [[0],np.cumsum(job_assign)])

for icore in range(ncores):
    job = multiprocess.Process(target = single_job,args=(job_starts[icore],job_assign[icore],icore))
    jobs.append(job)
    job.start()
for job in jobs:
    job.join()
    
# combine data
f = h5py.File(Filename,'a')
for icore in range(ncores):
    with h5py.File(Filename+'_'+str(icore),'r') as _f:
        for ipath in range(job_starts[icore],job_assign[icore]+job_starts[icore]):
            f['path'+str(ipath+1+N0)] = _f['path'+str(ipath+1+N0)][()]
    os.remove(Filename+'_'+str(icore))
f.close()





