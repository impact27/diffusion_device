# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:32:10 2017

@author: quentinpeter
"""
import numpy as np

#%%
def poiseuille(Zgrid, Ygrid, Wz, Wy, Q, get_interface=False):
    """
    Compute the poiseuille flow profile
    
    Parameters
    ----------
    Zgrid:  integer
        Number of Z pixel
    Ygrid:  integer
        Number of Y pixel
    Wz: float
        Channel height [m]
    Wy: float 
        Channel width [m]
    Q:  float
        The flux in the channel in [ul/h]
    get_interface: Bool, defaults False
        Also returns poisuille flow between pixels
    Returns
    -------
    V: 2d array
        The poiseuille flow
    if get_interface is True:
    Viy: 2d array
        The poiseuille flow between y pixels
    Viz: 2d array
        The poiseuille flow between z pixels
    """
        
    #Poiseuille flow
    V = np.zeros((Zgrid, Ygrid), dtype='float64')    
    for j in range(Ygrid):
        for i in range(Zgrid):
            nz = np.arange(1, 100, 2)[:, None]
            ny = np.arange(1, 100, 2)[None, :]
            V[i, j] = np.sum(1/(nz*ny*(nz**2/Wz**2+ny**2/Wy**2))*
                      (np.sin(nz*np.pi*(i+.5)/Zgrid)*
                       np.sin(ny*np.pi*(j+.5)/Ygrid)))
    Q /= 3600*1e9 #transorm in m^3/s
    #Normalise
    normfactor = Q/(np.mean(V)* Wy *Wz)
    V *= normfactor
    
    if not get_interface:
        return V
    #Y interface
    Viy = np.zeros((Zgrid, Ygrid-1), dtype='float64')    
    for j in range(1, Ygrid):
        for i in range(Zgrid):
            nz = np.arange(1, 100, 2)[:, None]
            ny = np.arange(1, 100, 2)[None, :]
            Viy[i, j-1] = np.sum(1/(nz*ny*(nz**2/Wz**2+ny**2/Wy**2))*
                      (np.sin(nz*np.pi*(i+.5)/Zgrid)*
                       np.sin(ny*np.pi*(j)/Ygrid)))
    Viy *= normfactor
    #Z interface       
    Viz = np.zeros((Zgrid-1, Ygrid), dtype='float64')    
    for j in range(Ygrid):
        for i in range(1, Zgrid):
            nz = np.arange(1, 100, 2)[:, None]
            ny = np.arange(1, 100, 2)[None, :]
            Viz[i-1, j] = np.sum(1/(nz*ny*(nz**2/Wz**2+ny**2/Wy**2))*
                      (np.sin(nz*np.pi*(i)/Zgrid)*
                       np.sin(ny*np.pi*(j+.5)/Ygrid)))
            
    Viz *= normfactor
    return V, Viy, Viz
#%%    
def stepMatrix(Zgrid, Ygrid, Wz, Wy, Q, *, muEoD=0, outV=None, 
               method='Trapezoid', dxfactor=1, Zmirror=False):
    """
    Compute the step matrix and corresponding position step
    
    Parameters
    ----------
    Zgrid:  integer
        Number of Z pixel
    Ygrid:  integer
        Number of Y pixel
    Wz: float
        Channel height [m]
    Wy: float 
        Channel width [m]
    Q:  float
        The flux in the channel in [ul/h]
    muEoD: float, default 0
        In case of electrophoresis, q*E/k/T = muE/D[m^-1]
    outV: 2d float array
        array to use for the return
    method: string, default 'Trapezoid'
        Method for integration
        'Trapezoid': Mixed integration
         'Explicit': explicit integration 
         'Implicit': implicit integration
    dxfactor: float, default 1
        Factor to change the value of dx
    Zmirror: bool, default False
        should we use a mirror for Z?
        
    Returns
    -------
    F:  2d array
        The step matrix (independent on Q)
    dxtD: float 
        The position step multiplied by the diffusion coefficient
    """
     
    V, Viy, __ = poiseuille(Zgrid, Ygrid, Wz, Wy, Q, get_interface=True)
    
    if outV is not None:
        outV[:] = V
    
    #% Get The step matrix
    dy = Wy/Ygrid
    dz = Wz/Zgrid
    
    if Zmirror:
        Zodd = Zgrid%2==1
        halfZgrid = (Zgrid+1)//2 
        V = V[:halfZgrid,:]
        Viy = Viy[:halfZgrid, :]
        Zgrid = halfZgrid
    
    #flatten V
    V = np.ravel(V)
    qy = getQy(Zgrid, Ygrid)
    qz = getQz(Zgrid, Ygrid, Zmirror, Zodd)
    
    #get Cyy
    Cyy = np.diag(1/V)@((qy[-1]-2*qy[0]+qy[1])/dy**2)
    
    #get Czz
    Czz = 0
    if Zgrid>1:
#        udiag = 1/V[:-Ygrid]
#        ldiag = 1/V[Ygrid:]
#        Czz = np.diag(udiag, Ygrid)
#        if Zmirror and Zodd:
#            ldiag[-Ygrid:] = 2*ldiag[-Ygrid:]
#        Czz += np.diag(ldiag, -Ygrid)
#        Czz -= np.diag(np.sum(Czz, 1))
#        Czz /= dz**2
#        Czz2 = Czz
        
        Czz = np.diag(1/V)@((qz[-1]-2*qz[0]+qz[1])/dz**2)
        
     
    #get F
    #The formula gives F=1+dx*D/V*(1/dy^2*dyyC+1/dz^2*dzzC)
    #Choosing dx as dx=dy^2*Vmin/D, The step matrix is:
    dxtD = np.min((dy, dz))**2*V.min()/2
    
    Cy = 0
    if muEoD != 0:
#        Cy = getCy5(muEoD, dxtD, V, Zgrid, Ygrid, dy)
        Cy = getCy(muEoD, dxtD, Viy, Zgrid, Ygrid, dy)
    
    Lapl = Cyy+Czz - muEoD*Cy
    
    
    if muEoD != 0:
        dxtD2 = V.min()/np.abs(muEoD)**2
        dxtD = np.min([dxtD, dxtD2])
        
    dxtD *= dxfactor
    dF = dxtD*Lapl

    #Get step matrix
    I = np.eye(Ygrid*Zgrid, dtype=float)
    if method == 'Explicit':
        #Explicit
        F = I+dF
    elif method == 'Implicit':
        #implicit
        F = np.linalg.inv(I-dF)
    elif method == 'Trapezoid':
        #Trapezoid
        F = np.linalg.inv(I-.5*dF)@(I+.5*dF)
    else:
        raise RuntimeError("Unknown integration Method: {}".format(method))
        
        
    #The maximal eigenvalue should be <=1! otherwhise no stability
    #The above dx should put it to 1
#    from numpy.linalg import eigvals
#    assert(np.max(np.abs(eigvals(F)))<=1.)
    return F, dxtD

def getQy(Zgrid, Ygrid):
    #Create the q matrices
    q = np.zeros((5, Zgrid*Ygrid, Zgrid*Ygrid))
    for i in range(-2, 3):
        q[i]=np.diag(np.ones(Zgrid*Ygrid-np.abs(i)), i)
        
    for i in range(0, Zgrid*Ygrid, Ygrid):
        #border 
        q[-2,i:i+2,i]=1
        q[-1,i,i]=1
        q[1,i+Ygrid-1,i+Ygrid-1]=1
        q[2,i+Ygrid-2:i+Ygrid,i+Ygrid-1]=1
        if i>0:
            q[:,i:i+2, i-2:i] = 0
            q[:,i-2:i, i:i+2] = 0
    return q

def getQz(Zgrid, Ygrid, Zmirror, Zodd):    
    #Create the q matrices
    q = np.zeros((5, Zgrid*Ygrid, Zgrid*Ygrid))
    for i in range(-2, 3):
        q[i]=np.diag(np.ones(Ygrid*(Zgrid-np.abs(i))), i*Ygrid)
        #Border
        if i < 0:
            q[i][(np.arange(Ygrid), 
                  np.arange(Ygrid))] = 1
        if i < -1:
            q[i][(np.arange(Ygrid)+Ygrid, 
                  np.arange(Ygrid))] = 1
        
        if i == 1:
            if Zmirror and Zodd:
                q[i][(np.arange(-Ygrid,0), 
                      np.arange(-Ygrid,0)-Ygrid)] = 1
            else:
                q[i][(np.arange(-Ygrid,0), 
                      np.arange(-Ygrid,0))] = 1
        if i == 2:
            if Zmirror:
                if Zodd:
                    q[i][(np.arange(-Ygrid,0)-Ygrid, 
                          np.arange(-Ygrid,0)-Ygrid)] = 1
                    q[i][(np.arange(-Ygrid,0), 
                          np.arange(-Ygrid,0)-2*Ygrid)] = 1
                else:
                    q[i][(np.arange(-Ygrid,0)-Ygrid, 
                          np.arange(-Ygrid,0))] = 1
                    q[i][(np.arange(-Ygrid,0), 
                          np.arange(-Ygrid,0)-Ygrid)] = 1
            else:
                q[i][(np.arange(-Ygrid,0), 
                      np.arange(-Ygrid,0))] = 1
                q[i][(np.arange(-Ygrid,0)-Ygrid, 
                      np.arange(-Ygrid,0))] = 1
    
    return q

def getCy5(muEoD, dxtD, V, Zgrid, Ygrid, dy):
    
    q = getQy(Zgrid, Ygrid)
    
    Cy = q[-2] - 8*q[-1] + 8*q[1] - q[2]
    Cy /= (12*dy)
    Cy = np.diag(1/V)@Cy
    return Cy

def getCy(muEoD, dxtD, Viy, Zgrid, Ygrid, dy):
    #ravel the Viy with 1 zero between each z
    iVyp = np.ravel(np.concatenate((1/Viy, np.zeros((Zgrid, 1))), 1))
    iVym = np.concatenate(([0], iVyp[:-1]))
    
    iVyp = np.diag(iVyp)
    iVym = np.diag(iVym)
    
    q = getQy(Zgrid, Ygrid)
    
    sigdy = np.zeros((3, Zgrid*Ygrid, Zgrid*Ygrid))
    for i in range(-1, 2):
        sigdy[i] = (q[i+1] - q[i-1])/2
    
    
    #Get nu
    nu = muEoD*dxtD
    
    #We want to represent the following equation as a matrix
    """\frac{1}{\Delta y}\left[ 
        \left( \frac{q_{i-1}}{V_{i-1/2}} - \frac{q_{i}}{V_{i+1/2}} \right) 
        + \frac{1}{2}\left( 
            \left( \frac{\Delta y \sigma_{i-1}}{V_{i-1/2}} 
                - \frac{\Delta y \sigma_{i}}{V_{i+1/2}} \right) 
            - \frac{u\Delta x}{\Delta y} \left( 
                \frac{\Delta y \sigma_{i-1}}{V_{i-1/2}^2} 
                - \frac{\Delta y \sigma_{i}}{V_{i+1/2}^2} 
    \right)\right) \right]"""
    
    neg =   muEoD < 0
    
    Cy = (iVym@q[-1+neg] - iVyp@q[0+neg] 
          + (.5-neg)*((iVym@sigdy[-1+neg] - iVyp@sigdy[0+neg])
              - nu*(iVym**2@sigdy[-1+neg] - iVyp**2@sigdy[0+neg])))
    
    Cy /= dy
    
    # minus to he a differntial operator
    return -Cy 

#@profile
def getprofiles(Cinit, Q, Radii, readingpos,  Wy=300e-6, Wz=50e-6, Zgrid=1,
                muEoD=0, *, fullGrid=False, central_profile=False,
                eta=1e-3, kT=1.38e-23*295, normalize=True, Zmirror=True,
                stepMuE=False, dxfactor=1):
    """Returns the theorical profiles for the input variables
    
    Parameters
    ----------
    Cinit:  1d array or 2d array
            The initial profile. If 1D (shape (x, ) not (x, 1)) Zgrid is 
            used to pad the array
    Q:  float
        The flux in the channel in [ul/h]
    Radii: 1d array
        The simulated radius. Must be in increasing order [m]
        OR: The mobilities (if stepMuE is True)
    readingpos: 1d array float
        Position to read at
    Wy: float, defaults 300e-6 
        Channel width [m]
    Wz: float, defaults 50e-6
        Channel height [m]  
    Zgrid:  integer, defaults 1
        Number of Z pixel if Cinit is unidimentional
    muEoD: float, default 0
        mobility times transverse electric field divided by diffusion constant
    fullGrid: bool , false
        Should return full grid?
    central_profile: Bool, default False
        If true, returns only the central profile
    eta: float
        eta
    kT: float
        kT
    normalize: Bool, default True
        Should the profiles be normalized?
    Zmirror: Bool, default True
        Should the Z mirror be used to bet basis functions
    stepMuE: Bool, default False
        Radii is in fact muEs
    dxfactor: float, default 1
        Factor to change dx size if the step size seems too big (Useless?)

    Returns
    -------
    profilespos: 3d array
        The list of profiles for the 12 positions at the required radii
    
    """  
    Radii = np.array(Radii)
    if stepMuE:
        if muEoD==0:
            raise RuntimeError("Can't calculate for 0 qE")
    else:
        if np.any(Radii<=0):
            raise RuntimeError("Can't work with negative radii!")
    #Functions to access F
    def getF(Fdir, NSteps):
        if NSteps not in Fdir:
            Fdir[NSteps] = np.dot(Fdir[NSteps//2], Fdir[NSteps//2])
        return Fdir[NSteps]  
    
    def initF(Zgrid, Ygrid, Wz, Wy, Q, muEoD, Zmirror, dxfactor):
        key = (Zgrid, Ygrid, Wz, Wy, Q, muEoD, Zmirror, dxfactor)
        if not hasattr(getprofiles, 'dirFList') :
            getprofiles.dirFList = {}
        #Create dictionnary if doesn't exist
        if key in getprofiles.dirFList:
            return getprofiles.dirFList[key]
        else:
            Fdir = {}
            Fdir[1], dxtd = stepMatrix(Zgrid, Ygrid, Wz, Wy, Q, muEoD=muEoD, 
                                        Zmirror=Zmirror, dxfactor=dxfactor)
            getprofiles.dirFList[key] = (Fdir, dxtd)
            return Fdir, dxtd
        
    #Prepare input and Initialize arrays
    readingpos = np.asarray(readingpos)
    
    ZgridEffective = Zgrid
    if Zmirror:
        ZgridEffective = (Zgrid+1)//2
    
    Cinit = np.asarray(Cinit, dtype=float)
    if len(Cinit.shape)<2:
        Cinit = np.tile(Cinit[np.newaxis, :], (ZgridEffective, 1))
    else:
        if Cinit.shape[0]!=ZgridEffective:
            raise RuntimeError("Cinit Z dim and Zgrid not aligned.")
        
    Ygrid = Cinit.shape[1];
    NRs = len(Radii)
    Nrp = len(readingpos)
    profilespos = np.tile(np.ravel(Cinit), (NRs*Nrp, 1))
    
    #get step matrix
    Fdir, dxtD = initF(Zgrid, Ygrid, Wz, Wy, Q, muEoD, Zmirror, dxfactor)       

    #Get Nsteps for each radius and position
    Nsteps = np.empty((NRs*Nrp,), dtype=int)         
    for i, v in enumerate(Radii):
        if stepMuE:
            dx = np.abs(dxtD*muEoD/v)
        else:
            D = kT/(6*np.pi*eta*v)
            dx = dxtD/D
        Nsteps[Nrp*i:Nrp*(i+1)] = np.asarray(readingpos/dx, dtype=int)
     
    print('{} steps'.format(Nsteps.max()))
    #transform Nsteps to binary array
    pow2 = 1<<np.arange(int(np.floor(np.log2(Nsteps.max())+1)))
    pow2 = pow2[:, None]
    binSteps = np.bitwise_and(Nsteps[None, :], pow2)>0
    
    #Sort for less calculations
    sortedbs = np.argsort([str(num) 
                            for num in np.asarray(binSteps, dtype=int).T])
    
    #for each unit
    for i, bsUnit in enumerate(binSteps):
        F = getF(Fdir, 2**i)
        #save previous number
        prev = np.zeros(i+1, dtype=bool)
        for j, bs in enumerate(bsUnit[sortedbs]):#[sortedbs]
            prof = profilespos[sortedbs[j], :]
            act = binSteps[:i+1, sortedbs[j]]
            #If we have a one, multiply by the current step function
            if bs:
                #If this is the same as before, no need to recompute
                if (act==prev).all():
                    prof[:] = profilespos[sortedbs[j-1]]
                else:
                    prof[:] = np.dot(F, prof)
            prev = act
         
    #reshape correctly
    profilespos.shape = (NRs, Nrp, ZgridEffective, Ygrid)
    
    if Zmirror:
        profilespos = np.concatenate((profilespos, 
                                      profilespos[:, :, -1-Zgrid%2::-1, :]), 2)
        Cinit = np.concatenate((Cinit, Cinit[-1-Zgrid%2::-1, :]), 0)
    
    #If full grid, stop here
    if fullGrid:
        return profilespos
    
    if central_profile:
        #Take central profile
        central_idx = int((Zgrid-1)/2)
        profilespos = profilespos[:, :, central_idx, :]
    else:
        #Take mean
        profilespos = np.mean(profilespos, -2)
        
    if normalize:
        #Normalize as the z position is not known
        profilespos /= (np.sum(profilespos, -1)[:, :, np.newaxis]
                        / np.sum(Cinit/Zgrid))
        
    return profilespos

def getElectroProfiles(Cinit, Q, absmuEoDs, muEs, readingpos,  Wy=300e-6,
                       Wz=50e-6, 
                       Zgrid=1, *, fullGrid=False, central_profile=False,
                       eta=1e-3, kT=1.38e-23*295, Zmirror=True, dxfactor=1):
    """Returns the theorical profiles for the input variables
    
    Parameters
    ----------
    Cinit:  1d array or 2d array
            The initial profile. If 1D (shape (x, ) not (x, 1)) Zgrid is 
            used to pad the array
    Q:  float
        The flux in the channel in [ul/h]
    absmuEoDs: array floats
        absolute values of the muE/D to test
    muE: array float
        values od muE to test
    readingpos: 1d array float
        Position to read at
    Wy: float, defaults 300e-6 
        Channel width [m]
    Wz: float, defaults 50e-6
        Channel height [m]  
    Zgrid:  integer, defaults 1
        Number of Z pixel if Cinit is unidimentional
    fullGrid: bool , false
        Should return full grid?
    outV: 2d float array
        array to use for the poiseuiile flow
    central_profile: Bool, default False
        If true, returns only the central profile
    eta: float
        eta
    kT: float
        kT
    Zmirror: Bool, default True
        Should the Z mirror be used to bet basis functions
        
    dxfactor: float, default 1
        Factor to change dx size if the step size seems too big (Useless?)

    Returns
    -------
    profilespos: 3d array
        The list of profiles for the 12 positions at the required radii
    
    """ 

    muEs = np.asarray(muEs)
    absmuEoDs = np.abs(absmuEoDs)
    NqE = len(absmuEoDs)
    negmuE = muEs[muEs<0]
    posmuE = muEs[muEs>0]
    
    Nrp = len(readingpos)
    Ygrid = Cinit.shape[-1]
    
    def getret(muEs, muEoDs):
        NuEs = len(muEs)
        if fullGrid:
            rets = np.zeros((NqE, NuEs, Nrp, Zgrid, Ygrid))
        else:
            rets = np.zeros((NqE, NuEs, Nrp, Ygrid))
            
        for muEoD, ret in zip(muEoDs, rets):
            ret[:] = getprofiles(Cinit, Q, muEs, readingpos,  Wy, Wz, Zgrid, 
                                 muEoD, fullGrid=fullGrid, eta=eta, kT=kT, 
                                 Zmirror=Zmirror, 
                                 central_profile=central_profile,
                                 normalize=False, stepMuE=True, 
                                 dxfactor=dxfactor)
        return rets
    
    N_neg_muEs = len(negmuE)
    N_pos_muEs = len(posmuE)
    NmuEs = N_neg_muEs + N_pos_muEs
    if fullGrid:
        rets = np.zeros((NqE, NmuEs, Nrp, Zgrid, Ygrid))
    else:
        rets = np.zeros((NqE, NmuEs, Nrp, Ygrid))
        
    if N_neg_muEs>0:
        rets[:, :N_neg_muEs] = getret(negmuE, -absmuEoDs)
    if N_pos_muEs>0:   
        rets[:, N_neg_muEs:] = getret(posmuE, absmuEoDs)
    
    return rets