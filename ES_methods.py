import numpy as np
import h5py
import sys

###################################################
#           Contact:                              #                      
#           Xiao Xue                              #
#           xuexiao@mail.itp.ac.cn                #
#           xxueitp@gmail.com                     #
#           Xiao Xue                              #
#           xuexiao@mail.itp.ac.cn                #
#           xxueitp@gmail.com                     # 
#           September 2020                        # 
###################################################

###################################################
#           - fun    Load_Events                  #                      
#           - class  Events                       #
#           - class  Analysis                     #
###################################################


me = 511e3 # eV
a0 = 1./(me*1./137.)    


def Load_Events(name):

    f = h5py.File('results/'+name+'.hdf5','r')
    Paths = np.array(f['Paths'])
    Vs = np.array(f['Vs'])
    mdm = np.array(f['mdm'])
    sige = np.array(f['sige'])
    f.close()

    return Analysis(Paths,Vs,mdm,sige)

    

###################################################
#           Main Class                            #
###################################################


class Events:
    def __init__(self,mdm,sige):

        self.mdm = mdm
        self.sige = sige
        self.ct0 = 1.
        self.phi0 = 0. 
        self.nO = 3.45e22 # cm^-3


    
###################################################
#           Load K                                #
###################################################

    def load_Ktot(self):

        f = h5py.File('EarthAtomicResponse/O_Ktot.hdf5','r')

        self.Ktot = np.array(f['Ktot']) # ER,q
        self.q = np.array(f['q'])
        self.ER = np.array(f['ER'])

        f.close()



###################################################
#           Get dsig                              #
###################################################

#    def get_dsigv2dEdq(self):
#
#        mu = me*self.mdm/(me+self.mdm)    
#
#        self.dsigv2dEdq = self.sige*me*a0**2/(2*mu**2) * self.q[None,:] * self.Ktot    




    def calc_dsigv2dlogEdlogq(self):

        mu = me*self.mdm/(me+self.mdm)    

        self.dsigv2dlogEdlogq = self.sige*me*a0**2/(2*mu**2) * self.q[None,:] \
                        * self.Ktot * self.q[None,:] * self.ER[:,None] * np.log(10)**2


###################################################
#           Cut it smaller                        #
###################################################

    def cut_dsigv2(self):
        
        # first cut is based on the value of dsigv2
        dsigv2 = self.dsigv2dlogEdlogq
        check = np.where(dsigv2 > dsigv2.max()*1e-3)
        pos = [check[1].min(),check[1].max(),check[0].min(),check[0].max()]
        
        dsigv2_cut = dsigv2[pos[2]:pos[3]+1,pos[0]:pos[1]+1]
        q_cut = self.q[pos[0]:pos[1]+1]
        ER_cut = self.ER[pos[2]:pos[3]+1]       

        # second cut is based on 'mask', which is the physical constrain on ER and q which relies on v.

        self.dsigv2_cut = dsigv2_cut
        self.q_cut = q_cut
        self.ER_cut = ER_cut 


###################################################
#           Sampling methods                      #
###################################################

    def direct_sample(self,N=2**22):


        np.random.seed()
        a,b =  self.dsigv2_cut.shape
        x = np.random.randint(0,a,size=N)
        y = np.random.randint(0,b,size=N)
        dice = np.random.rand(N)*self.dsigv2_cut.max()
        msk = dice < self.dsigv2_cut[x,y]
        x_sel = x[msk]
        y_sel = y[msk]
        #print('efficiency_1:' , '%.3f'%(len(x_sel)/N) )

        self.q_sample = self.q_cut[y_sel]
        self.ER_sample = self.ER_cut[x_sel]

    

###################################################
#           Physical things                       #
###################################################

    def _scatter_info(self,v,qs,ERs):
        
        v2 = v**2
        vf2_ll = (v-qs/self.mdm)**2 
        vf2  = v2 - 2.*ERs/self.mdm 
        
        if vf2 < vf2_ll:
            vf = v
            ca = 1.1 # this is a effective way to tell ca is ineffective
        else:
            vf = np.sqrt(vf2)
            ca = (v2 + vf**2 - qs**2/self.mdm**2 )/2./v/vf 
        
        # v_final and cos(theta)
        return vf,ca


###################################################
#           Run                                   #
###################################################        

    def run_one(self,v0):
        

        N = len(self.q_sample)
        beta_pool = np.random.rand(N)*2*np.pi

        vf_raw = np.zeros(N+1)
        ca_raw = np.zeros(N+1)


        # iteration starts  
 
        vf = v0
        ca = 1.
        vf_raw[0] = vf
        ca_raw[0] = ca
        for i in range(N):
            vf,ca = self._scatter_info(vf,self.q_sample[i],self.ER_sample[i])
            vf_raw[i+1] = vf 
            ca_raw[i+1] = ca
           

        vf_select = vf_raw[np.where(ca_raw!=1.1)[0]]
        ca_select = ca_raw[np.where(ca_raw!=1.1)[0]]

        #print('efficiency_2:','%.3f'%(len(vf_select)/N))

        self.size = len(vf_select)
        self.pool = np.vstack([vf_select,ca_select])
        self.v0 = v0    

        word = '|                       | effective collisions: %.0f | final velocity: %.3f | \r'%(self.size,vf)
        print(word,end="") 

###################################################
#           V and Path Recovery                   #
################################################### 

    def v_recovery(self):

        # alpha's pool
        ca_save = self.pool[1]
        sa_save = np.sqrt(1.-ca_save**2)

        # beta's pool
        beta_save = np.random.rand(self.size)*2*np.pi - np.pi
        cb_save = np.cos(beta_save)
        sb_save = np.sin(beta_save)

        # initiate Earth's coordinates
        ct_save = np.zeros(self.size)
        st_save = np.zeros(self.size)
        phi_save = np.zeros(self.size)
        
        phi = self.phi0
        ct = self.ct0
        st = np.sqrt(1-ct**2)
        
        # iteration
        for i in range(self.size):

            ca = ca_save[i]
            sa = sa_save[i] 
            cb = cb_save[i]
            sb = sb_save[i]   
        
            phi += np.arctan2(sa*sb,st*ca + ct*sa*cb)
            ct = ct*ca - st*sa*cb # be careful for the rank
            st = np.sqrt(1.-ct**2)

            phi_save[i] = phi
            ct_save[i] = ct
            st_save[i] = st


        self.vpool = np.vstack([phi_save,ct_save,st_save,self.pool[0]])
            


    
    def path_recovery(self,x0_start):

        n = self.nO #cm^-3

        # because we also have to calculate the path before the first collision
        
        phi_save = self.vpool[0]
        ct_save = self.vpool[1]
        st_save = self.vpool[2]
        vf_save = self.vpool[3]

        sphi_save = np.sin(phi_save)
        cphi_save = np.cos(phi_save) 
        
        # sample the distances
        sigs_pool = self.isig(vf_save) # cm^2
        mfps_pool = 1e-5/(n*sigs_pool) # kms
        exp_pool = np.random.exponential(self.size)
        distance_pool =  exp_pool*mfps_pool 

        # distance each step
        dx = distance_pool*st_save*cphi_save
        dy = distance_pool*st_save*sphi_save
        dz = distance_pool*ct_save

        # create a mask to do integration
        path = np.zeros((3,self.size+1))
        Dx = 0.+x0_start[0]
        Dy = 0.+x0_start[1]
        Dz = 0.+x0_start[2]

        path[0,0] = Dx
        path[1,0] = Dy
        path[2,0] = Dz
        for i in range(self.size):

            Dx += dx[i]
            Dy += dy[i]
            Dz += dz[i]
            path[0,i+1] = Dx
            path[1,i+1] = Dy
            path[2,i+1] = Dz

        self.path = path
        d = np.linalg.norm(path[:,-1]-path[:,0])
  
        print('|        |d: %.0f km\r'%d,end='')





###################################################
#           Sig and Calculation                   #
###################################################

#    def _SIG1(self,v):
#
#        insqrt = self.mdm**2*self.v**2 - 2*self.mdm*self.ER
#        insqrt = np.where(insqrt<0,0,insqrt)
#        qmin = self.mdm*v - np.sqrt(insqrt)
#        qmax = self.mdm*v + np.sqrt(insqrt)
#        msk = (self.q[:,None]>qmin[None,:])*(self.q[:,None]<qmax[None,:])

#        dsig = msk[:-1,:-1] * self.dsigv2dEdq[:-1,:-1].T * np.diff(self.q)[:,None] * np.diff(self.ER)[None,:]/v**2
#        sig1 = np.sum( dsig )
#        return sig1


    def _SIG(self,v):
        
        logq = np.log10(self.q)
        logER = np.log10(self.ER)
        insqrt = self.mdm**2*v**2 - 2*self.mdm*10**logER
        insqrt = np.where(insqrt<0,0,insqrt)
        logqmin = np.log10(self.mdm*v - np.sqrt(insqrt))
        logqmax = np.log10(self.mdm*v + np.sqrt(insqrt))
        msk = (logq[:,None]>logqmin[None,:])*(logq[:,None]<logqmax[None,:])

        dsig =  msk[:-1,:-1] * self.dsigv2dlogEdlogq[:-1,:-1].T * np.diff(logq)[:,None] * np.diff(logER)[None,:]/v**2
        sig2 = np.sum( dsig )        
        return sig2


    def iSIGs(self):
        
        v_vec = np.linspace(1e-20,0.5,100)
        sigs=[]
        for v in v_vec:
            sig = self._SIG(v)
            sigs.append(sig)
        self.isig = lambda v:np.interp(v,v_vec,sigs)

        str1 = '| maximal sigma'
        print('{:<26}'.format(str1),':%.2e cm^2'%max(sigs))
        
#TODO#Run    
   
#TODO#Analysis


###################################################
#           Class of Analysis                     #
###################################################

class Analysis():

    def __init__(self,Paths,Vs,mdm,sige):

        self.mdm = mdm
        self.sige = sige

        N = len(Paths)
        MaxLen = max([len(p[0]) for p in Paths])
        Paths_ = np.ndarray((N,3,MaxLen))
        Vs_ = np.ndarray((N,4,MaxLen-1))

        for i in range(N):
            p = Paths[i]
            v = Vs[i]
            Paths_[i,:,:p.shape[1]] = p
            Paths_[i,:,p.shape[1]:] = p[:,-1][:,None]
            Vs_[i,:,:v.shape[1]] = v
            Vs_[i,:,v.shape[1]:] = v[:,-1][:,None]

        self.Paths = Paths_
        self.Vs = Vs_
        
        # output
        str1 = '| initial mean free path '
        str2 = '| mean collisions '
        print('{:<26}'.format(str1),':%.0f km'%np.mean( [p[2,1]-p[2,0] for p in Paths] ) )
        print('{:<26}'.format(str2),':%.0f'%np.mean([len(p[0]) for p in Paths]))



###################################################
#           Saving the data                       #
###################################################

    def save_data(self,name):

        try:
            filedir = 'results/'+name+'.hdf5'
            f = h5py.File(filedir,'w-')
        except OSError:
            print('Error, Unable to save the data.')
            
        else:
            f.create_dataset('mdm',data=self.mdm)
            f.create_dataset('sige',data=self.sige)
            f.create_dataset('Paths',data=self.Paths)
            f.create_dataset('Vs',data=self.Vs)
            f.close()

            print(filedir)



###################################################
#           Cut the paths with a sphere           #
###################################################

    def cut_sphere(self , r0_pos, rsphere , rbound ):

        # rbound is to include all the DM particles at their initial position.
        cPaths = self.Paths.copy()
        Vs = self.Vs

        r0_pos = np.array(r0_pos)

        # decide if the path is out of the bound
        d2r0 = np.linalg.norm(cPaths - r0_pos[None,:,None],axis=1)
        if_inbound = d2r0<=rbound
        if_hitbound = np.diff(if_inbound.astype(int))
        out_bound = np.where(if_hitbound==-1) # only the outward direction is considered

        # cut the path
        for i in range(len(out_bound[0])):
            which = out_bound[0][i]
            where = out_bound[1][i]
            # it is compatible with the case where the path hit the bound multiple times.
            cPaths[which,:,where+1:] = cPaths[which,:,where+1][:,None]


        # with the cut path, decide when the path hit the sphere (for statistics)
        dcut2r0 = np.linalg.norm(cPaths - r0_pos[None,:,None],axis=1)
        if_insphere = dcut2r0<=rsphere
        if_hitsphere = np.diff(if_insphere,axis=1) # both inward and outward are taken into consideration
        which_hit = np.where(if_hitsphere) # find out where the DM hit the surface
        
        # to recover the hit point
        # v = x*v1 + (1-x)*v2
        # |v| = rsphere

        pos1 = cPaths[which_hit[0],:,which_hit[1]] - r0_pos
        pos2 = cPaths[which_hit[0],:,which_hit[1]+1] - r0_pos
        velo = Vs[which_hit[0],:,which_hit[1]]

        v1 = np.linalg.norm(pos1,axis=1)
        v2 = np.linalg.norm(pos2,axis=1)
        v1dotv2 = np.sum(pos1*pos2,axis=1)

        a = v1**2 + v2**2 - 2.*v1dotv2
        b = 2.*v1dotv2 - 2.*v2**2
        c = v2**2 - rsphere**2
        
        # must differentiate between the outward and inward direction.
        x1 = ( -b-np.sqrt(b**2-4.*a*c) )/2./a 
        x2 = ( -b+np.sqrt(b**2-4.*a*c) )/2./a         
        x = np.where(v1>v2,x2,x1)
                
        pos = pos1*x[:,None] + pos2*(1.-x)[:,None] + r0_pos

        self.r0_pos = r0_pos
        self.rsphere = rsphere
        self.rbound = rbound
        self.cPaths = cPaths
        self.pos = pos
        self.velo = velo

###################################################
#           Statistical Analysis                  #
###################################################
                
    def sphere_statistics(self):

        normpos = self.pos/np.linalg.norm(self.pos,axis=1)[:,None]
        ctheta = normpos[:,2]
        phi = np.arctan2(normpos[:,1],normpos[:,0])
        T = self.velo[:,3]**2*self.mdm

        self.T = T
        self.ctheta = ctheta
        self.phi = phi
            
            
