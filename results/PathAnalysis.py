import numpy as np
import h5py
import os







class PathAnalysis:


    def __init__(self,filename):

        self.filename = filename
        if os.path.exists(filename)==False:
            print('error, file doesn\'t exist.')


    def load_paths(self):
        
        f = h5py.File(filename,'r')
        Npaths = len(f.keys()) - 2
        self.mdm = np.asarray(f['mdm'])
        self.sige = np.asarray(f['sige'])
        rawpaths = []
        for i in range(Npaths):    
            path = np.asarray( f['path'+str(i)] )
            rawpaths.append( path )
        MaxSteps = max([len(p[0]) for p in rawpaths])

        Paths = np.zeros((Npaths,4,MaxSteps))

        for i in range(Npaths):
            path = rawpaths[i]
            Paths[i,:,:len(path[0])] = path
            Paths[i,:,len(path[0]):] = path[:,-1][:,None]
        self.Paths = Paths

        

    def cut_sphere(self):

        detector_depth = 6371. - 1.6
        d2r0 = np.linalg.norm(self.Paths[:,:3],axis = 1)
        velo = self.Paths[:,3,:]
        insphere = d2r0 <= detector_depth
        ifhit = np.where(np.diff(insphere)!=0)

        # get where exactly the particle hit the sphere
        hitpos_before = self.Paths[ifhit[0],:,ifhit[1]][:,:3]
        hitpos_after = self.Paths[ifhit[0],:,ifhit[1]+1][:,:3]
        dhitpos = hitpos_after - hitpos_before
        hitvelo = self.Paths[ifhit[0],:,ifhit[1]][:,3]

        # the math
        a = np.sum(dhitpos*dhitpos , axis=1)
        b = 2*np.sum(dhitpos*hitpos_before , axis=1)
        c = np.sum(hitpos_before*hitpos_before , axis=1) - detector_depth**2
        y1 = ( -b+np.sqrt(b**2 - 4*a*c) )/2/a
        y2 = ( -b-np.sqrt(b**2 - 4*a*c) )/2/a
        y = np.where(y1 < y2 , y1 , y2)
        hitpos = y[:,None]*dhitpos + hitpos_before

        self.hitvelo = hitvelo
        self.hitpos = hitpos
        
        self.ctheta = hitpos[:,2]/detector_depth
        

        

if __name__=='__main__':   
    import matplotlib.pyplot as plt
    filename = input(' > filename: \n > ')
    s = PathAnalysis(filename)
    s.load_paths()
    s.cut_sphere()
        
