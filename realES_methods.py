import numpy as np
import h5py
import sys
import EarthProfile as EP

###################################################
#           Contact:                              #                      
#                                                 #
#           Xiao Xue                              #
#           xuexiao@mail.itp.ac.cn                #
#           xxueitp@gmail.com                     #
#                                                 #
#           Xiao Xue                              #
#           xuexiao@mail.itp.ac.cn                #
#           xxueitp@gmail.com                     #
#                                                 #  
#           September 2020                        # 
###################################################


###################################################
#           - fun    Load_Events                  #                      
#           - class  Events                       #
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

        self.K_core,self.K_mantle,self.q,self.ER = EP.get_K()
        self.irho_c,self.irho_m = EP.get_irho()



###################################################
#           Get dsig                              #
###################################################


    def calc_ndsigv2dlogEdlogq2rho(self):

        mu = me*self.mdm/(me+self.mdm)    

        # v^2*dsig/dlogq/dlogE * n/d(rho)
        self.ndsigv2dlogEdlogq2rho_mantle = self.sige*me*a0**2/(2*mu**2) * self.q[None,:] \
                        * self.K_mantle * self.q[None,:] * self.ER[:,None] * np.log(10)**2

        self.ndsigv2dlogEdlogq2rho_core = self.sige*me*a0**2/(2*mu**2) * self.q[None,:] \
                        * self.K_core * self.q[None,:] * self.ER[:,None] * np.log(10)**2


###################################################
#           Cut it smaller                        #
###################################################

    def cut_ndsigv2(self):
        
        # first cut is based on the value of dsigv2
        ndsigv2_mantle = self.ndsigv2dlogEdlogq2rho_mantle
        ndsigv2_core = self.ndsigv2dlogEdlogq2rho_core

        check_m = np.where(ndsigv2_mantle > ndsigv2_mantle.max()*1e-3)
        check_c = np.where(ndsigv2_core > ndsigv2_core.max()*1e-3)

        pos_m = [check_m[1].min(),check_m[1].max(),check_m[0].min(),check_m[0].max()]
        pos_c = [check_c[1].min(),check_c[1].max(),check_c[0].min(),check_c[0].max()]
        pos = [min(pos_m[0],pos_c[0]),max(pos_m[1],pos_c[1]),min(pos_m[2],pos_c[2]),max(pos_m[3],pos_c[3])]
        
        ndsigv2_mantle_cut = ndsigv2_mantle[pos[2]:pos[3]+1,pos[0]:pos[1]+1]
        ndsigv2_core_cut = ndsigv2_core[pos[2]:pos[3]+1,pos[0]:pos[1]+1]

        q_cut = self.q[pos[0]:pos[1]+1]
        ER_cut = self.ER[pos[2]:pos[3]+1]     
 

        # second cut is based on 'mask', which is the physical constrain on ER and q which relies on v.

        self.ndsigv2_mantle_cut = ndsigv2_mantle_cut
        self.ndsigv2_core_cut = ndsigv2_core_cut 


        self.q_cut = q_cut
        self.ER_cut = ER_cut 



###################################################
#           Sampling methods                      #
###################################################
    #TODO
    def direct_sample(self,N=2**20):

        np.random.seed()

        # for Earth Mantle
        a,b =  self.ndsigv2_mantle_cut.shape
        x = np.random.randint(0,a,size=N)
        y = np.random.randint(0,b,size=N)
        dice = np.random.rand(N)*self.ndsigv2_mantle_cut.max()
        msk = dice < self.ndsigv2_mantle_cut[x,y]
        x_sel = x[msk]
        y_sel = y[msk]
        self.q_sample_mantle = self.q_cut[y_sel]
        self.ER_sample_mantle = self.ER_cut[x_sel]


        # for Earth Core
        a,b =  self.ndsigv2_core_cut.shape
        x = np.random.randint(0,a,size=N)
        y = np.random.randint(0,b,size=N)
        dice = np.random.rand(N)*self.ndsigv2_core_cut.max()
        msk = dice < self.ndsigv2_core_cut[x,y]
        x_sel = x[msk]
        y_sel = y[msk]
        self.q_sample_core = self.q_cut[y_sel]
        self.ER_sample_core = self.ER_cut[x_sel]

    


###################################################
#           Run                                   #
###################################################        

    def run_one(self,v0,x0):
        
        N1 = len(self.q_sample_mantle)
        N2 = len(self.q_sample_core)


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
        
        countM = 0
        countC = 0
        count = 1 # from 1 

        while ss!=0 and v>0.005 and countM<N1 and countC<N2:

            r = np.linalg.norm(x)
            # choose what to sample (or break the loop directly)
            if ss == 1:
                qs,ERs = self.q_sample_mantle[countM],self.ER_sample_mantle[countM]
                countM += 1
                rawdis = 1e-5/self.insig2rho_mantle(v)/self.irho_m(r)*np.random.exponential()
                
            elif ss == 2:
                qs,ERs = self.q_sample_core[countC],self.ER_sample_core[countC]
                countC += 1
                rawdis = 1e-5/self.insig2rho_core(v)/self.irho_c(r)*np.random.exponential()
                
                
            # to calculate
            v_prop2  = v**2  - 2.*ERs/self.mdm
            
            if v_prop2 < (v-qs/self.mdm)**2 :

                continue # break and to find the next sample
                
            else:
                
                dx = np.array([rawdis*st*np.cos(phi) , rawdis*st*np.sin(phi) ,rawdis*ct])
                x2 ,xdotdx, dx2 = np.sum(x*x) , np.sum(x*dx), np.sum(dx*dx) 

                ycheck = - xdotdx/dx2
                xcheck = ycheck*dx + x
                ssx , ssxp ,ssxc = get_status(x),get_status(x+dx),get_status(xcheck)
                
                # outward
                if ycheck <= 0 and ssx != ssxp and ssx == 2:
                    a = dx2 # > 0
                    b = 2*xdotdx
                    c = x2 - self.rCore**2 # < 0 
                    y = (-b+np.sqrt(b**2-4*a*c))/2/a*(1.+1e-3)
                    # update a new x
                    x = x + y*dx
                    ss = get_status(x)
                    
                # outward
                elif ycheck <= 0 and ssx != ssxp and ssx == 1:
                    a = dx2 # > 0
                    b = 2*xdotdx 
                    c = x2 - self.rEarth**2 # <0
                    y = (-b+np.sqrt(b**2-4*a*c))/2/a*(1.+1e-3)
                    # update a new x
                    x = x + y*dx
                    ss = get_status(x)
                    
                # inward
                elif (ycheck > 0 and ycheck <= 1) and ssx !=ssxc and ssx == 1:
                    dx = ycheck*dx
                    xdotdx, dx2 = np.sum(x*dx), np.sum(dx*dx)
                    a = dx2 # > 0
                    b = 2*xdotdx 
                    c = x2 - self.rCore**2 # > 0
                    y = (-b-np.sqrt(b**2-4*a*c))/2/a*(1.+1e-3)
                    # update a new x
                    x = x + y*dx
                    ss = get_status(x)

                # inward    
                elif ycheck > 1 and ssx != ssxp and ssx ==1:
                    a = dx2 # > 0
                    b = 2*xdotdx  # < 0
                    c = x2 - self.rCore**2 # >0
                    y = (-b-np.sqrt(b**2-4*a*c))/2/a*(1.+1e-3)
                    x = x + y*dx
                    ss = get_status(x)

                    
                    
                elif ssx == ssxp:

                    x = x + dx                    
                    ss = get_status(x)
                    
                    
                    beta = np.random.rand()*2*np.pi
                    sb = np.sin(beta)
                    cb = np.cos(beta)   
                                     
                    v_prop = np.sqrt(v_prop2)
                    ca = (v**2 + v_prop2 - qs**2/self.mdm**2 )/2./v/v_prop
                    sa = np.sqrt(1-ca**2)
                    
                    phi +=  np.arctan2(sa*sb,st*ca + ct*sa*cb)
                    ct = ct*ca - st*sa*cb
                    st = np.sqrt(1.-ct**2)
                    
                    v = v_prop
                    
                else:
                    print('error')
                    break    
                    
                    
                save_data(v,ca,phi,ct,st,x,ss)
                count += 1
                    
                     

             
                

                
                

        self.count = count
        self.v = np.array(v_vec)
        self.ca = np.array(ca_vec)
        self.phi = np.array(phi_vec)
        self.ct = np.array(ct_vec)
        self.st = np.array(st_vec)
        self.x = np.array(x_vec)
        self.ss = np.array(ss_vec) 

        word = '       | effective collisions: %.0f | final velocity: %.3f | final status %.0f \r'%(self.count,v,ss)
        print(word,end='') 




###################################################
#           Sig and Calculation                   #
###################################################


    def _SIG(self,v):
        
        # for Mantle
        logq = np.log10(self.q)
        logER = np.log10(self.ER)
        insqrt = self.mdm**2*v**2 - 2*self.mdm*10**logER
        insqrt = np.where(insqrt<0.,0.,insqrt)
        logqmin = np.log10(self.mdm*v - np.sqrt(insqrt))
        logqmax = np.log10(self.mdm*v + np.sqrt(insqrt))
        msk = (logq[:,None]>logqmin[None,:])*(logq[:,None]<logqmax[None,:])

        ndsig2rho_mantle =  msk[1:,1:] * self.ndsigv2dlogEdlogq2rho_mantle[1:,1:].T * np.diff(logq)[:,None] * np.diff(logER)[None,:]/v**2
        nsig2rho_mantle = np.sum( ndsig2rho_mantle )        
        ndsig2rho_core =  msk[1:,1:] * self.ndsigv2dlogEdlogq2rho_core[1:,1:].T * np.diff(logq)[:,None] * np.diff(logER)[None,:]/v**2
        nsig2rho_core = np.sum( ndsig2rho_core )             
        
   
        return nsig2rho_mantle , nsig2rho_core


    def inSIG2rhos(self):
        
        v_vec = np.linspace(1e-20,0.5,100)
        nsig2rhos_mantle=[]
        nsig2rhos_core=[]
        for v in v_vec:
            nsig2rho_mantle, nsig2rho_core = self._SIG(v)
            nsig2rhos_mantle.append(nsig2rho_mantle)
            nsig2rhos_core.append(nsig2rho_core)
        self.insig2rho_mantle = lambda v:np.interp(v,v_vec,nsig2rhos_mantle)
        self.insig2rho_core = lambda v:np.interp(v,v_vec,nsig2rhos_core)

        str1 = '| n * sigma / rho (Mantle)'
        str2 = '| n * sigma / rho (Core)'
        print('{:<30}'.format(str1),':%.2e cm^2'%max(nsig2rhos_mantle))
        print('{:<30}'.format(str2),':%.2e cm^2'%max(nsig2rhos_core))

            
            
