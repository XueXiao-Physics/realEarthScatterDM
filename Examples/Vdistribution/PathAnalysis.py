import numpy as np
import h5py
import os
import sys
import tqdm


class PathAnalysis:


    def __init__(self,filename):

        self.filename = filename
        self.Earth_radius = 6371.
        self.detector_depth = 1.6
        self.detector_pos = self.Earth_radius-self.detector_depth
        print('set Earth\'s radius', '%.1f'%self.Earth_radius,'km')
        print('set detector depth at', '%.1f'%self.detector_depth,'km')
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
                #print('%i'%(i+1)+' / '+'%i'%len(keynames),'\r',end='')    
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
        c = absA**2 - self.detector_pos**2 + 0j
        
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
        
        # the unit-velocity n_dm of the dark matter particle when hitting the border
        nvelo1 = BminusA[ihit1[0],:,ihit1[1]]/absBminusA[ihit1[0],None,ihit1[1]]
        nvelo2 = BminusA[ihit2[0],:,ihit2[1]]/absBminusA[ihit2[0],None,ihit2[1]]
        
        # n_dm \dot n_sphere 
        weight1 = 1/np.abs(np.sum(hitpos1*nvelo1,axis=1)/self.detector_pos)
        weight2 = 1/np.abs(np.sum(hitpos2*nvelo2,axis=1)/self.detector_pos)
        
        
        
        self.hitpos = np.concatenate([hitpos1,hitpos2])
        self.hitvelo = np.concatenate([hitvelo1,hitvelo2])
        self.hitctheta = self.hitpos[:,2]/self.detector_pos
        self.weight = np.concatenate([weight1,weight2])

       
    def cut_disc(self,ctheta):
    
        stheta = np.sqrt(1-ctheta**2)
        max_width = np.sqrt(self.Earth_radius**2 - (self.detector_pos*ctheta)**2)
        
        z_plane = ctheta*self.detector_pos
        
        if_above = self.Paths[:,2,:] > z_plane
        if_hit = np.where(np.diff(if_above))
        
        A = self.Paths[if_hit[0],:,if_hit[1]]
        B = self.Paths[if_hit[0],:,if_hit[1]+1]
        D_AB = (B-A)[:,:3]
        
        x = D_AB[:,2]
        y = z_plane - A[:,2]
        hit_pos = (y/x)[:,None]*D_AB + A[:,:3]
        hit_cphi = D_AB[:,2]/np.linalg.norm(D_AB,axis=1)
        hit_velo = A[:,3]
        
        
        hit_dis2core = np.linalg.norm(hit_pos[:,:2],axis=1)
        if_border = (hit_dis2core>(self.detector_pos*stheta-0.3)) * (hit_dis2core<(self.detector_pos*stheta+0.3))
        hitborder_pos = hit_pos[if_border]
        hitborder_cphi = hit_cphi[if_border]
        hitborder_velo = hit_velo[if_border]
        count = len(hitborder_velo)
        
        area = np.pi*(min(max_width,self.detector_pos*stheta+0.3)**2 - max(0,self.detector_pos*stheta-0.3)**2)
          
        return hitborder_pos , hitborder_velo , hitborder_cphi , ctheta*np.ones(count) , area
        
        
            
        


        

if __name__=='__main__':   
    import matplotlib.pyplot as plt
    filename = sys.argv[1]
    s = PathAnalysis(filename)
    print("Loading Paths")
    s.load_paths()
    

    
    print("\nUsing single sphere to cut the DM Paths.")
    s.cut_sphere()
    np.savetxt(filename+'_hitpos_sphere.txt',s.hitpos)

   

    print("\nUsing discs to cut the DM Paths.")
    _velo = [] 
    _ctheta = [] 
    _weight = [] 
    _hitpos = [] 
    _area_list = []
    
    
    cbins = 41
    vbins = 41
    cbins_interv = 50
    bin_ctheta = np.linspace(-1,1,cbins)
    bin_velo = np.logspace(-3,-1,vbins)

    diff_bin_ctheta = bin_ctheta[1] - bin_ctheta[0]
    diff_bin_velo = np.diff(bin_velo)
    
    cthetas = diff_bin_ctheta/cbins_interv * (np.arange(0,(cbins_interv*(cbins-1)))+0.5) - 1
    
    for i in tqdm.tqdm(range(len(cthetas))):
        result = s.cut_disc(cthetas[i]) 
        _area_list.append(result[4])
        try: 
            _velo.extend(result[1]) 
            _ctheta.extend(result[3]) 
            _weight.extend(1/result[2]) 
            _hitpos.extend(result[0])
        except: 
            pass 
    np.savetxt(filename+'_ctheta_velo_weight_disc.txt',np.vstack([_ctheta,_velo,_weight]).T)
    np.savetxt(filename+'_hitpos_disc',_hitpos)
    np.savetxt(filename+'_binarea_disc',np.vstack([cthetas,_area_list]).T)
    
    bin_area = np.array(_area_list).reshape(cbins-1,cbins_interv).sum(axis=1)
    
    
    
    phi0 = s.Paths.shape[0]/np.pi/s.Earth_radius**2
    
    darea = 4*np.pi*s.detector_pos**2/cbins
    plt.figure(figsize=(12,4))
    plt.subplot(121)    
    h = np.histogram2d(s.hitvelo,s.hitctheta,weights = s.weight,bins=[bin_velo,bin_ctheta])
    plt.pcolormesh(bin_velo,bin_ctheta,h[0].T/darea/phi0,cmap='afmhot',vmax=1)
    plt.xscale('log')
    plt.ylabel(r'$\cos(\theta)$')
    plt.ylim(-1,1)
    plt.colorbar()
    plt.title('sphere')
    
     
    plt.subplot(122)
    h = np.histogram2d(_velo,_ctheta,weights = _weight,bins=[ bin_velo,bin_ctheta])
    plt.pcolormesh(bin_velo,bin_ctheta,(h[0]/bin_area).T/phi0,cmap='afmhot',vmax=1)
    plt.xscale('log')
    plt.xlabel('velo')
    plt.ylabel(r'$\cos(\theta)$')
    plt.ylim(-1,1)
    plt.colorbar()
    plt.title('discs')
    
    plt.savefig(filename+'.jpg')
    print('\n Figure saved.')
