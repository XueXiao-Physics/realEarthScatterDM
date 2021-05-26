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
        absBcrossA = np.linalg.norm(np.cross(A,B,axis=1),axis=1)
        # impact factor b
        ib = absBcrossA/absBminusA
        
        #Check if the line is inward/in & out/outward
        absA = np.linalg.norm(A,axis=1)
        absB = np.linalg.norm(B,axis=1)
        ifAin = absA<detector_pos
        ifBin = absB<detector_pos
        ifbin = ib<detector_pos
        #test = np.argsort([absA,absB,ib],axis=0)
        
        # helping find the point(s)
        a = absBminusA**2
        b = 2*np.sum(A*(B-A),axis=1)
        c = absA**2 - detector_pos**2
        
        
        msk_h1i = np.where( ifBin*(1-ifAin) ) 
        msk_h1o = np.where( (1-ifBin)*ifAin )
        msk_h2  = np.where( (b<=0)*(c>0)*((2*a+b)>0)*((a+b+c)>0)*ifbin )
        print('{:<40}'.format('N of micro paths that inwardly hit once'),msk_h1i[0].shape[0])
        print('{:<40}'.format('N of micro paths that outwardly hit once'),msk_h1o[0].shape[0])
        print('{:<40}'.format('N of micro paths that hit twice'),msk_h2[0].shape[0])
        

       
        
        a1i = a[msk_h1i[0],msk_h1i[1]]
        b1i = b[msk_h1i[0],msk_h1i[1]]
        c1i = c[msk_h1i[0],msk_h1i[1]]
        
        a1o = a[msk_h1o[0],msk_h1o[1]]
        b1o = b[msk_h1o[0],msk_h1o[1]]
        c1o = c[msk_h1o[0],msk_h1o[1]]
        
        a2 = a[msk_h2[0],msk_h2[1]]
        b2 = b[msk_h2[0],msk_h2[1]]
        c2 = c[msk_h2[0],msk_h2[1]]        
        
        
        x_1 = (-b1i-np.sqrt(b1i**2-4*a1i*c1i))/2/a1i 
        x_2 = (-b1o+np.sqrt(b1o**2-4*a1o*c1o))/2/a1o
        x_3 = (-b2-np.sqrt(b2**2-4*a2*c2))/2/a2 
        x_4 = (-b2+np.sqrt(b2**2-4*a2*c2))/2/a2
        
        hitpos_1 = x_1[:,None]*B[msk_h1i[0],:,msk_h1i[1]] + (1-x_1)[:,None]*A[msk_h1i[0],:,msk_h1i[1]]
        hitvelo_1 = self.Paths[msk_h1i[0],3,msk_h1i[1]]
        nvelo_1 = BminusA[msk_h1i[0],:,msk_h1i[1]]/absBminusA[msk_h1i[0],None,msk_h1i[1]]
        weight_1 = 1/np.abs(np.sum(hitpos_1*nvelo_1,axis=1)/detector_pos)
        
        hitpos_2 = x_2[:,None]*B[msk_h1o[0],:,msk_h1o[1]] + (1-x_2)[:,None]*A[msk_h1o[0],:,msk_h1o[1]]
        hitvelo_2 = self.Paths[msk_h1o[0],3,msk_h1o[1]]
        nvelo_2 = BminusA[msk_h1o[0],:,msk_h1o[1]]/absBminusA[msk_h1o[0],None,msk_h1o[1]]
        weight_2 = 1/np.abs(np.sum(hitpos_2*nvelo_2,axis=1)/detector_pos)
        
        hitpos_3 = x_3[:,None]*B[msk_h2[0],:,msk_h2[1]] + (1-x_3)[:,None]*A[msk_h2[0],:,msk_h2[1]]
        hitvelo_3 = self.Paths[msk_h2[0],3,msk_h2[1]]  
        nvelo_3 = BminusA[msk_h2[0],:,msk_h2[1]]/absBminusA[msk_h2[0],None,msk_h2[1]]
        weight_3 = 1/np.abs(np.sum(hitpos_3*nvelo_3,axis=1)/detector_pos)
        
        hitpos_4 = x_4[:,None]*B[msk_h2[0],:,msk_h2[1]] + (1-x_4)[:,None]*A[msk_h2[0],:,msk_h2[1]]
        hitvelo_4 = self.Paths[msk_h2[0],3,msk_h2[1]]     
        nvelo_4 = BminusA[msk_h2[0],:,msk_h2[1]]/absBminusA[msk_h2[0],None,msk_h2[1]]
        weight_4 = 1/np.abs(np.sum(hitpos_4*nvelo_4,axis=1)/detector_pos)
        
        self.hitpos = np.concatenate([hitpos_1,hitpos_2,hitpos_3,hitpos_4])
        self.hitvelo = np.concatenate([hitvelo_1,hitvelo_2,hitvelo_3,hitvelo_4])
        self.hitctheta = self.hitpos[:,2]/detector_pos
        self.weight = np.concatenate([weight_1,weight_2,weight_3,weight_4])

       
        
        


        

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
