from realES import *
import matplotlib.pyplot as plt
import scipy.interpolate
import io
import os
import json
import math
try:
    import multiprocess
except:
    import multiprocessing as multiprocess



with io.open('settings.txt') as f:
    settings = json.load(f)
ncores = int(settings['ncores'])


mdm = float(settings['mdm'])
sige = float(settings['sige'])
N = int(settings['N'])
v0 = float(settings['v0'])

print('**********************************')
print('******  INITIAL CONDITIONS  ******')
print('**********************************')
print('{:<30}'.format('Dark matter mass'), ': %.2e'%mdm,'(eV)')
print('{:<30}'.format('DM-e cross section'), ': %.2e'%sige,'(1/cm^2)')
print('{:<30}'.format('Initial velocity of DM'), ': %.2e'%v0,'*c')
print('{:<30}'.format('Number of simulations'), ': %i'%N)
print('You can change above parameters in settings.txt')
input('Press Enter...')



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
print('\n\n*******************************')
print('******  ADDITIONAL INFO  ******')
print('*******************************')
s.load_Ktot()
s.calc_sum_ndsig2rho_v2dlnEdlnq(0)
s.inSIG2rhos()
# create file if not exist
Filename = settings['mdm']+'_'+settings['sige']+'_'+settings['v0']
input('Press Enter...')

print('\n\n************************************')
print('******  READY FOR SIMULATION  ******')
print('************************************')
print('The file is created with the root name : ',Filename)
print('If you use ctrl+c to cancel the process, you will lose the results of this run.')

try:
    os.mkdir('results/'+Filename)    
except OSError:
    pass

try:
    f = h5py.File('results/'+Filename+'/'+Filename,'w-')
    f.create_dataset('mdm',data = mdm)
    f.create_dataset('sige',data = sige)
    N0 = 0
    f.close()
except OSError:
    pass
    
input('Press Enter to start simulation...')

# open file
# get N0
with  h5py.File('results/'+Filename+'/'+Filename,'a') as f:
    keynames = list(filter(lambda i:i[:4]=='path',f.keys()) )
    if len(keynames) != 0:
        nums = [int(k[4:]) for k in keynames]
        N0 = max(nums)
    else:
        N0 = 0



# multiprocess

def single_job(_Nstart,_N,_icore):	
    with  h5py.File('results/'+Filename+'/'+Filename+'_'+str(icore),'w') as f:
        for i in range(_Nstart,_Nstart+_N):
            x0 = [sx[i],sy[i],sz[i]]
            
            s.run_one(v0,x0)
            
            print(i+1+N0,'\r',end='')
            
            f.create_dataset('path'+str(i+1+N0),data = np.vstack([s.x.T,s.v,s.ss]))
            f.flush()
        print('\n')



ncores = ncores # cpu cores 
jobs = []
job_add = np.zeros(ncores,dtype=int)
job_add[:N%ncores] = 1 
job_assign = np.ones(ncores,dtype=int)*(N//ncores) + job_add
job_starts =  np.concatenate( [[0],np.cumsum(job_assign)])

for icore in range(ncores):
    job = multiprocess.Process(target = single_job,args=(job_starts[icore],job_assign[icore],icore))
    jobs.append(job)
    job.start()
for job in jobs:
    job.join()
    
# combine data
f = h5py.File('results/'+Filename+'/'+Filename,'a')
for icore in range(ncores):
    with h5py.File('results/'+Filename+'/'+Filename+'_'+str(icore),'r') as _f:
        for ipath in range(job_starts[icore],job_assign[icore]+job_starts[icore]):
            f['path'+str(ipath+1+N0)] = _f['path'+str(ipath+1+N0)][()]
    os.remove('results/'+Filename+'/'+Filename+'_'+str(icore))
f.close()



print('\n\n***********************************')
print('******  SIMULATION FINISHES  ******')
print('***********************************')
print('try:')
print('"cd results"')
print('"python PathAnalysis.py '+Filename+'/'+Filename+'"'+'\n')
print('Good Luck. If you encounter any problem, email xxueitp@gmail.com and I will try to help you. Note that you can rerun the code with the same dark matter parameters, and the results will be added to the previous file (if you don\'t use ctrl+c to cancel it in the middle of the run).')


