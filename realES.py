import numpy as np
import h5py
import sys
import EarthProfile as EP

###################################################
#           Contact:                              #                      
#                                                 #
#           Xiao Xue                              #
#           xxueitp@gmail.com                     #
#                                                 #  
#           September 2020                        # 
###################################################


###################################################
#           - fun    Load_Events                  #                      
#           - class  EarthEvents                  #
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


class EarthEvents:

    def __init__(self,mdm,sige):

        self.mdm = mdm
        self.sige = sige
        self.ct0 = 1.
        self.phi0 = 0. 

        self.rEarth = EP.rEarth # km
        self.rCore = EP.rCore  # km


    
###################################################
#           Load K                                #
###################################################

    def load_Ktot(self):

        out1 = EP.get_K()
        out2 = EP.get_irho()
        self.K_core,self.K_mantle,self.q,self.ER = out1[:4]
        self.irho_c,self.irho_m = out2[:2]
        
        self.__mean_rho_core = out2[2]
        self.__mean_rho_mantle = out2[3]
        
        str1 ='<ne> * sige (core)'
        str2 ='<ne> * sige (mantle)'
        ER_vmin = np.sqrt(2*self.ER[1]/self.mdm)
        hard_vmin = 1e-3
        self.vmin=max(ER_vmin,hard_vmin)
        
        
        print('{:<36}'.format(str1), '= %.2e'%(self.sige*out1[6]*out2[2]),'(1/cm)|')
        print('{:<36}'.format(str2), '= %.2e'%(self.sige*out1[7]*out2[3]),'(1/cm)|')
        
        print('{:<36}'.format('v_min from min(ER)'),': %.2e'%ER_vmin )      
        print('{:<36}'.format('v_min hard cut'),': %.2e'%hard_vmin )


        


###################################################
#           Get dsig                              #
###################################################

    def calc_sum_ndsig2rho_v2dlnEdlnq(self,n):

        mu = me*self.mdm/(me+self.mdm)    

        # v^2*dsig/dlogq/dlogE * n/d(rho)
        self.sum_ndsig2rho_v2dlnEdlnq_mantle = self.sige*me*a0**2/(2*mu**2) * self.q[None,:] \
                * self.K_mantle * self.q[None,:] * self.ER[:,None]* (self.q[None,:]*a0)**n

        self.sum_ndsig2rho_v2dlnEdlnq_core = self.sige*me*a0**2/(2*mu**2) * self.q[None,:] \
                * self.K_core * self.q[None,:] * self.ER[:,None]* (self.q[None,:]*a0)**n





###################################################
#           Sampling methods                      #
###################################################

    def direct_sample(self,v,N=100000):

        ERmax = self.mdm*v**2/2
        cut = np.max(np.where(self.ER <= ERmax))
               
        np.random.seed()

        # for Earth Mantle
        x = np.random.randint(0,cut,size=N)
        y = np.random.randint(0,len(self.q),size=N)
        dice = np.random.rand(N)*self.sum_ndsig2rho_v2dlnEdlnq_mantle[:cut,:].max()
        msk = dice < self.sum_ndsig2rho_v2dlnEdlnq_mantle[x,y]
        x_sel = x[msk]
        y_sel = y[msk]
        
        ERs = self.ER[x_sel]
        qs = self.q[y_sel]        
        msk2 = ERs<(qs*v-qs**2/2/self.mdm)
        
        self.ER_sample_mantle = ERs[msk2]
        self.q_sample_mantle = qs[msk2]


        # for Earth Core
        x = np.random.randint(0,cut,size=N)
        y = np.random.randint(0,len(self.q),size=N)
        dice = np.random.rand(N)*self.sum_ndsig2rho_v2dlnEdlnq_core[:cut,:].max()
        msk = dice < self.sum_ndsig2rho_v2dlnEdlnq_core[x,y]
        x_sel = x[msk]
        y_sel = y[msk]
        
        
        ERs = self.ER[x_sel]
        qs = self.q[y_sel]
        msk2 = ERs<(qs*v-qs**2/2/self.mdm)
        
        self.ER_sample_core = ERs[msk2]
        self.q_sample_core = qs[msk2]
        
        return len(self.ER_sample_core),len(self.ER_sample_mantle)

    


###################################################
#           Run                                   #
###################################################        

    def run_one(self,v0,x0):
        


        # relative info ->
        v = v0
        ca = 1.
        sa = 0. # this will not be saved.
        v_vec = [] # +1 because of the inital values are added
        ca_vec = []
 
        
        # absolute info about velocity ->
        phi = self.phi0
        ct = self.ct0
        st = np.sqrt(1 - ct**2)
        phi_vec = []
        ct_vec = []
        st_vec = []
        ss_vec = []

        # absolute info about path ->
        x = np.array(x0)
        x_vec = []
        
              

        borders = [ None , self.rEarth , self.rCore ]

        def get_status(x):
            d = np.linalg.norm(x)
            inEarth = (d<=(self.rEarth)).astype(int)
            inCore = (d<=(self.rCore)).astype(int)
            inMantle = inEarth*(1-inCore)
            status = [inEarth,inMantle,inCore]
            if status == [0,0,0]:
                ss = 0
            elif status == [1,1,0]:
                ss = 1
            elif status == [1,0,1]:
                ss = 2
            return ss

            
        def save_data(v,ca,phi,ct,st,x,ss):
            v_vec.append(v)
            ca_vec.append(ca)
            phi_vec.append(phi)
            ct_vec.append(ct)
            st_vec.append(st)
            x_vec.append(x)
            ss_vec.append(ss)
            
        ss = get_status(x0)
        save_data(v,ca,phi,ct,st,x,ss)
        
        count = 0 # from 1 
        while ss!=0 and v>self.vmin:
            N1,N2 = self.direct_sample(v)
            while N1==0 or N2 ==0:
                N1,N2 = self.direct_sample(v)   

            r = np.linalg.norm(x)
            # choose what to sample (or break the loop directly)
            if ss == 1:
                qs,ERs = self.q_sample_mantle[0],self.ER_sample_mantle[0]
                rawdis = 1e-5/self.insig2rho_mantle(v)/self.irho_m(r)*np.random.exponential()

            elif ss == 2:
                qs,ERs = self.q_sample_core[0],self.ER_sample_core[0]
                rawdis = 1e-5/self.insig2rho_core(v)/self.irho_c(r)*np.random.exponential()

            v_prop2  = v**2  - 2.*ERs/self.mdm


            dx = np.array([rawdis*st*np.cos(phi) , rawdis*st*np.sin(phi) ,rawdis*ct])
            x2 ,xdotdx, dx2 = np.sum(x*x) , np.sum(x*dx), np.sum(dx*dx) 

            #ycheck = - xdotdx/dx2
            #xcheck = ycheck*dx + x
            #ssx , ssxp , ssxc = get_status(x),get_status(x+dx),get_status(xcheck)
            
            a = dx2 + 0j
            b = 2*xdotdx + 0j
            c1 = x2 - self.rEarth**2 + 0j
            c2 = x2 - self.rCore**2 + 0j
            
            solution_1 = (-b-np.sqrt(b**2-4*a*c1))/2/a
            solution_2 = (-b+np.sqrt(b**2-4*a*c1))/2/a
            solution_3 = (-b-np.sqrt(b**2-4*a*c2))/2/a
            solution_4 = (-b+np.sqrt(b**2-4*a*c2))/2/a
            
            solution_list = np.array([solution_1,solution_2,solution_3,solution_4])
            
            mask1 = np.isreal(solution_list)  
            mask2 = solution_list>=0
            mask3 = solution_list<1
            
            true_solutions = solution_list[np.where(mask1*mask2*mask3)] 
            
            if true_solutions.shape[0] != 0:
                
                # Take the first real 0 <= solution < 1
                solution = np.real(np.min(true_solutions))
                x = x + solution*dx*(1.+1e-5) # t
                ss = get_status(x)
                
            else:
                x = x + dx                    
                ss = get_status(x)

                beta = np.random.rand()*2*np.pi
                sb = np.sin(beta)
                cb = np.cos(beta)   

                v_prop = np.sqrt(v_prop2)
                ca = (v**2 + v_prop2 - qs**2/self.mdm**2 )/2./v/v_prop
                sa = np.sqrt(1.-ca**2)

                phi +=  np.arctan2(sa*sb,st*ca + ct*sa*cb)
                ct = ct*ca - st*sa*cb
                st = np.sqrt(1.-ct**2)

                v = v_prop          
                count += 1      

                  
            
            save_data(v,ca,phi,ct,st,x,ss)
            
                    
                     

             
                

                
                
        self.count = count
        self.v = np.array(v_vec)
        self.ca = np.array(ca_vec)
        self.phi = np.array(phi_vec)
        self.ct = np.array(ct_vec)
        self.st = np.array(st_vec)
        self.x = np.array(x_vec)
        self.ss = np.array(ss_vec) 

        word = '       | effective collisions: %.0f | final velocity: %.2e | final status %.0f \r'%(count,v,ss)
        print(word,end='') 




###################################################
#           Sig and Calculation                   #
###################################################


    def _SIG(self,v):
        
        # for Mantle
        q = self.q
        ER = self.ER
        insqrt = self.mdm**2*v**2 - 2*self.mdm*ER
        insqrt = np.where(insqrt<0.,0.,insqrt)
        qmin = self.mdm*v - np.sqrt(insqrt)
        qmax = self.mdm*v + np.sqrt(insqrt)
        msk = (q[:,None]>qmin[None,:])*(q[:,None]<qmax[None,:])
        # Integrating _SIG
        
        ndsig2rho_mantle =  msk[1:,1:] * self.sum_ndsig2rho_v2dlnEdlnq_mantle[1:,1:].T * np.diff(np.log(q))[:,None] * np.diff(np.log(ER))[None,:]/v**2
        nsig2rho_mantle = np.sum( ndsig2rho_mantle )        
        ndsig2rho_core =  msk[1:,1:] * self.sum_ndsig2rho_v2dlnEdlnq_core[1:,1:].T * np.diff(np.log(q))[:,None] * np.diff(np.log(ER))[None,:]/v**2
        nsig2rho_core = np.sum( ndsig2rho_core )             
        
   
        return nsig2rho_mantle , nsig2rho_core


    def inSIG2rhos(self):
        
        v_vec = np.logspace(np.log10(1e-20),np.log10(0.2),100)
        nsig2rhos_mantle=[]
        nsig2rhos_core=[]
        print('Calculating \int\sig_{ion} ...')
        for v in v_vec:
            nsig2rho_mantle, nsig2rho_core = self._SIG(v)
            nsig2rhos_mantle.append(nsig2rho_mantle)
            nsig2rhos_core.append(nsig2rho_core)
            
        nsig2rhos_mantle = np.array(nsig2rhos_mantle)
        nsig2rhos_core = np.array(nsig2rhos_core)
        self.insig2rho_mantle = lambda v:np.interp(v,v_vec,nsig2rhos_mantle)
        self.insig2rho_core = lambda v:np.interp(v,v_vec,nsig2rhos_core)

        str1 = '(v=1e-1) \sum <ni*sigi> (core)'
        str2 = '(v=1e-1) \sum <ni*sigi> (mantle)'
        print('{:<36}'.format(str1),'= %.2e'%(self.insig2rho_core(1e-1) * self.__mean_rho_core ),'(1/cm)|')
        print('{:<36}'.format(str2),'= %.2e'%(self.insig2rho_mantle(1e-1)  * self.__mean_rho_mantle),'(1/cm)|')

            
            
