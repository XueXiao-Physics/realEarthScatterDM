import numpy as np
import h5py
import os
import sys
import tqdm


class PathAnalysis:


    def __init__(self,filename):

        self.filename = filename
        if os.path.exists(filename)==False:
            print('error, file doesn\'t exist.')


    def load_paths(self):
        
        with h5py.File(self.filename,'r') as f:
            keynames = list(filter(lambda i:i[:4]=='path',f.keys()) )
            self.Npaths = len(keynames)
            self.mdm = np.asarray(f['mdm'])
            self.sige = np.asarray(f['sige'])
            rawpaths = []
            for i in tqdm.tqdm(range(len(keynames))):    
                path = np.asarray( f[keynames[i]] )
                rawpaths.append( path )
            MaxSteps = max([len(p[0]) for p in rawpaths])

            Paths = np.zeros((self.Npaths,4,MaxSteps))

            for i in range(self.Npaths):
                path = rawpaths[i]
                Paths[i,:4,:len(path[0])] = path[:4,:]
                Paths[i,:4,len(path[0]):] = path[:4,-1][:,None]
            self.Paths = Paths

        

    def cut_sphere(self):
        Earth_radius = 6371.
        detector_depth = 1.6
        detector_pos = Earth_radius-detector_depth
        print('set Earth\'s radius', '%.1f'%Earth_radius,'km')
        print('set detector depth at', '%.1f'%detector_depth,'km')
        
        # B minus A
        A = self.Paths[:,:3,:-1]
        B = self.Paths[:,:3,1:]
        BminusA = B-A
        absBminusA = np.linalg.norm(BminusA,axis=1)
        absBminusA[absBminusA==0]=np.nan
        absA = np.linalg.norm(A,axis=1)

        #test = np.argsort([absA,absB,ib],axis=0)
        
        # helping find the point(s)
        
        a = absBminusA**2 + 0j
        b = 2*np.sum(A*(B-A),axis=1) + 0j
        c = absA**2 - detector_pos**2 + 0j
        
        solution_1 = (-b-np.sqrt(b**2-4*a*c))/2/a
        solution_2 = (-b+np.sqrt(b**2-4*a*c))/2/a
        solution_list = np.array([solution_1,solution_2])
        mask1 =  (solution_list>=0)*(solution_list<1)
        mask2 = np.isreal(solution_list)
        Mask = mask1*mask2
        
        
        

        print('{:<44}'.format('N of micro paths that inwardly hit border once'),np.sum(  (Mask[0,:]==1)*(Mask[1,:]==0)  ))
        print('{:<44}'.format('N of micro paths that outwardly hit border once'),np.sum(  (Mask[0,:]==0)*(Mask[1,:]==1)  ))
        print('{:<44}'.format('N of micro paths that hit border twice'),np.sum( np.sum( Mask ,axis=0)==2 ))
        
        ihit1 = np.where(Mask[0])
        ihit2 = np.where(Mask[1])
        
        hitpos1 = A[ihit1[0],:,ihit1[1]] + BminusA[ihit1[0],:,ihit1[1]]*np.real(solution_1[ihit1][:,None])
        hitpos2 = A[ihit2[0],:,ihit2[1]] + BminusA[ihit2[0],:,ihit2[1]]*np.real(solution_2[ihit2][:,None])
        
        hitvelo1 = self.Paths[ihit1[0],3,ihit1[1]]
        hitvelo2 = self.Paths[ihit2[0],3,ihit2[1]]
        
        nvelo1 = BminusA[ihit1[0],:,ihit1[1]]/absBminusA[ihit1[0],None,ihit1[1]]
        nvelo2 = BminusA[ihit2[0],:,ihit2[1]]/absBminusA[ihit2[0],None,ihit2[1]]
        
        weight1 = 1/np.abs(np.sum(hitpos1*nvelo1,axis=1)/detector_pos)
        weight2 = 1/np.abs(np.sum(hitpos2*nvelo2,axis=1)/detector_pos)
        
        
        
        self.hitpos = np.concatenate([hitpos1,hitpos2])
        self.hitvelo = np.concatenate([hitvelo1,hitvelo2])
        self.hitctheta = self.hitpos[:,2]/detector_pos
        self.weight = np.concatenate([weight1,weight2])

       
        
        


        

if __name__=='__main__':   
    import matplotlib.pyplot as plt
    filename = sys.argv[1]
    s = PathAnalysis(filename)
    s.load_paths()
    s.cut_sphere()
    np.savetxt(filename+'.txt',np.vstack([s.hitctheta,s.hitvelo]).T)
    plt.hist2d(s.hitvelo,s.hitctheta,weights = s.weight,bins=40,cmap='Reds')
    plt.xlabel('velo')
    plt.ylabel('ctheta')
    plt.ylim(-1,1)
    plt.savefig(filename+'.jpg')
