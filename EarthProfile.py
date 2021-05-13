import numpy as np
import h5py
GeV2kg = 1.78266192e-27
# mass density profile 
# km kg/cm^3
rCore = 3483.
rEarth = 6371.
def get_irho():

    # kg/cm^3
    rho_c = np.array([[0., 13.0658e-3], [156.18, 13.0494e-3], [356.95, 
        13.0411e-3], [525.48, 13e-3], [671.71, 12.9671e-3], [805.55, 
        12.9259e-3], [964.16, 12.8601e-3], [1095.51, 12.8025e-3], [1229.33, 
        12.7367e-3], [1231.58, 12.1278e-3], [1395.15, 12.062e-3], [1583.48, 
        11.9386e-3], [1769.34, 11.8234e-3], [1927.93, 11.6918e-3], [2101.38, 
        11.5519e-3], [2272.35, 11.3956e-3], [2460.66, 11.2063e-3], [2666.31, 
        10.9842e-3], [2849.65, 10.7703e-3], [3030.51, 10.5316e-3], [3233.66, 
        10.2601e-3], [3476.44, 9.9228e-3], [3483., 9.88e-3]])
    # kg/cm^3
    rho_m = np.array([[3483., 5.562e-3], [3650.73, 5.4633e-3], [3898.56, 
        5.3563e-3], [4173.64, 5.2082e-3], [4399.15, 5.093e-3], [4612.28, 
        4.9778e-3], [4822.93, 4.8791e-3], [5080.66, 4.731e-3], [5348.29, 
        4.5665e-3], [5568.85, 4.443e-3], [5697.72, 4.3854e-3], [5705., 
        3.9658e-3], [5784.32, 3.9658e-3], [5888.37, 3.8259e-3], [5967.65, 
        3.719e-3], [5972.53, 3.5215e-3], [6056.79, 3.4804e-3], [6148.48, 
        3.4392e-3], [6150.93, 3.3405e-3], [6232.73, 3.3652e-3], [6339.32, 
        3.3734e-3], [6344.1, 3e-3], [6356.49, 3e-3], [6371., 3e-3]])

    irho_c = lambda r: np.interp(r,rho_c[:,0],rho_c[:,1])
    irho_m = lambda r: np.interp(r,rho_m[:,0],rho_m[:,1])
    
    mean_rho_core = (np.diff(rho_c[:,0])*rho_c[:-1,1]*rho_c[:-1,0]**2).sum()/(np.diff(rho_c[:,0])*rho_c[:-1,0]**2).sum()
    mean_rho_mantle = (np.diff(rho_m[:,0])*rho_m[:-1,1]*rho_m[:-1,0]**2).sum()/(np.diff(rho_m[:,0])*rho_m[:-1,0]**2).sum()
    print('<rho_core>   = %.2e'%mean_rho_core,'(kg/cm^3)')
    print('<rho_mantle> = %.2e'%mean_rho_mantle,'(kg/cm^3)')

    return irho_c, irho_m, mean_rho_core, mean_rho_mantle






# for each element, import K data (Assume that they gridded the same way)
def get_K():

    # mass composition
    # mass per atom (GeV) / mass percentage in Core / mass percentage in Mantle
    mass_dict = {'O':(14.9,	8,	0.,		0.44),
                'Mg':(22.3,	12,	0.,		0.228),
                'Al':(25.1, 	13,	0.,		0.0235),
                'Si':(26.1,	14,	0.06,		0.210),
                'S': (29.8,	16,	0.019,		0.00025),
                'Ca':(37.2,	20,	0.,		0.0253),
                'Fe':(52.1,	26,	0.855,		0.0626),
                'Ni':(58.7,	28,	0.052,		0.00196)}


    # calcualte relative number density of each element. 
    # Need to be multiplied with irho to get the true number density
    _name = [m for m in mass_dict]
    _info = np.array([mass_dict[m] for m in _name])
    _info[:,2] /= _info[:,0]*GeV2kg
    _info[:,3] /= _info[:,0]*GeV2kg
    n2rho_core = {_name[i]:_info[i][2] for i in range(len(_name))}
    n2rho_mantle   = {_name[i]:_info[i][3] for i in range(len(_name))}
    
    ne2rho_core = np.sum(_info[:,2]*_info[:,1])
    ne2rho_mantle = np.sum(_info[:,3]*_info[:,1])
    print('ne / rho_core   = %.2e'%ne2rho_core,'(1/kg)')
    print('ne / rho_mantle = %.2e'%ne2rho_mantle,'(1/kg)')

    K_dict = {}
    for key in mass_dict:
        f = h5py.File('EarthAtomicResponse/'+key+'_Ktot.hdf5','r')
        K = np.asarray(f['Ktot'])
        q = np.asarray(f['q'])
        ER = np.asarray(f['ER'])
        f.close()
        K_dict.update({key:K})


    # get total K in 
    K_core = 0.
    K_mantle = 0.
    for key in mass_dict:
        K_core += K_dict[key]*n2rho_core[key]
        K_mantle += K_dict[key]*n2rho_mantle[key]
    

    return K_core,K_mantle,q,ER,n2rho_core,n2rho_mantle,ne2rho_core,ne2rho_mantle
    


