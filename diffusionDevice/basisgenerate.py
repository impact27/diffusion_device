# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:32:10 2017

@author: quentinpeter
"""
import numpy as np
from scipy.linalg import toeplitz

#%%

    
    
def poiseuille(Zgrid,Ygrid,Wz,Wy,Q,outV=None):
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
    outV: 2d float array
        array to use for thr return
    Returns
    -------
    V: 2d array
        The poiseuille flow
    """
        
    #Poiseuille flow
    if outV is not None:
        assert(outV.shape==(Zgrid,Ygrid))
        V=outV
    else:
        V=np.zeros((Zgrid,Ygrid),dtype='float64')
    for j in range(Ygrid):
        for i in range(Zgrid):
            nz=np.arange(1,100,2)[:,None]
            ny=np.arange(1,100,2)[None,:]
            V[i,j]=np.sum(1/(nz*ny*(nz**2/Wz**2+ny**2/Wy**2))*
                      (np.sin(nz*np.pi*(i+.5)/Zgrid)*
                       np.sin(ny*np.pi*(j+.5)/Ygrid)))
    Q/=3600*1e9 #transorm in m^3/s
    #Normalize
    V*=Q/(np.mean(V)* Wy *Wz)
    return V
#%%    
def stepMatrix(Zgrid,Ygrid,Wz,Wy,Q,outV=None):
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
    outV: 2d float array
        array to use for the return
    Returns
    -------
    F:  2d array
        The step matrix (independent on Q)
    dxtD: float 
        The position step multiplied by the diffusion coefficient
    """
    V=poiseuille(Zgrid,Ygrid,Wz,Wy,Q,outV)
    #% Get The step matrix
    dy=Wy/Ygrid
    dz=Wz/Zgrid
    #flatten V
    V=np.ravel(V)
    
    
    #get Cyy
    line=np.zeros(Ygrid*Zgrid)
    line[:2]=[-2,1]
    Cyy=toeplitz(line,line) #toeplitz creation of matrice which repeat in diagonal 
    for i in range(0,Ygrid*Zgrid,Ygrid):
        Cyy[i,i]=-1
        Cyy[i-1+Ygrid,i-1+Ygrid]=-1
        if i>0 :
            Cyy[i-1,i]=0
            Cyy[i,i-1]=0
    #get Czz
    Czz=0
    if Zgrid>1:
        line=np.zeros(Ygrid*Zgrid)
        line[0]=-2
        line[Ygrid]=1
        Czz=toeplitz(line,line)
        for i in range(Ygrid):
            Czz[i,i]=-1
            Czz[Ygrid*Zgrid-i-1,Ygrid*Zgrid-i-1]=-1
    
    #get F
    #The formula gives F=1+dx*D/V*(1/dy^2*dyyC+1/dz^2*dzzC)
    #Choosing dx as dx=dy^2*Vmin/D, The step matrix is:
    F=1/dy**2*Cyy+1/dz**2*Czz
    dxtD=np.min((dy,dz))**2*V.min()/2
    I=np.eye(Ygrid*Zgrid, dtype=float)
    F=I+dxtD*np.dot(np.diagflat(1/V),F)
    #The maximal eigenvalue should be <=1! otherwhise no stability
    #The above dx should put it to 1
#    from numpy.linalg import eigvals
#    assert(np.max(np.abs(eigvals(F)))<=1.)
    return F, dxtD

def dxtDd(Zgrid,Ygrid,Wz,Wy,Q,outV=None):
    
    V=poiseuille(Zgrid,Ygrid,Wz,Wy,Q,outV)
    #% Get The step matrix
    dy=Wy/Ygrid
    dz=Wz/Zgrid    

    dxtD=np.min((dy,dz))**2*V.min()/2
    return dxtD


#@profile
def getprofiles(Cinit,Q, Rs, readingpos,  Wy = 300e-6, Wz= 50e-6, Zgrid=1,
                *,fullGrid=False, outV=None):
    """Returns the theorical profiles for the input variables
    
    Parameters
    ----------
    Cinit:  1d array or 2d array
            The initial profile. If 1D (shape (x,) not (x,1)) Zgrid is 
            used to pad the array
    Q:  float
        The flux in the channel in [ul/h]
    Rs: 1d array
        The simulated radius. Must be in increasing order [m]
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
    readingpos: 1d array float
        Position to read at
    Returns
    -------
    profilespos: 3d array
        The list of profiles for the 12 positions at the required radii
        
    Notes
    -----
    Depending if Q or Rs are small, a small dx is required to maintain stability
    """
    
    def getF(Fdir,NSteps):
        if NSteps not in Fdir:
            print('Newstep')
            Fdir[NSteps]=np.dot(Fdir[NSteps//2],Fdir[NSteps//2])
        return Fdir[NSteps]  
    
    def initF(Zgrid,Ygrid,Wz,Wy,Q,outV):
        key=(Zgrid,Ygrid,Wz,Wy)
        if not hasattr(getprofiles,'dirFList') :
            getprofiles.dirFList = {}
        #Create dictionnary if doesn't exist
        if key in getprofiles.dirFList:
            return getprofiles.dirFList[key], dxtDd(*key,Q,outV)
        else:
            print('new F!')
            Fdir={}
            Fdir[1],dxtd=stepMatrix(Zgrid,Ygrid,Wz,Wy,Q,outV)
            getprofiles.dirFList[key]=Fdir
            return Fdir,dxtd
        
    
    # Settings that are unlikly to change    
    kT = 1.38e-23*295;
    eta = 1e-3;
    
    #Prepare input and Initialize arrays
    readingpos=np.asarray(readingpos)
    
    Cinit=np.asarray(Cinit,dtype=float)
    if len(Cinit.shape)<2:
        Cinit=np.tile(Cinit[:,np.newaxis],(1,Zgrid)).T
    Ygrid = Cinit.shape[1];
    
    NRs=len(Rs)
    Nrp=len(readingpos)
    profilespos=np.tile(np.ravel(Cinit),(NRs*Nrp,1))
    
    #get step matrix
    Fdir,dxtD=initF(Zgrid,Ygrid,Wz,Wy,Q,outV)
#    F,dxtD=stepMatrix(Zgrid,Ygrid,Wz,Wy,Q,outV)        

    #Get Nsteps for each radius and position
    Nsteps=np.empty((NRs*Nrp,),dtype=int)         
    for i,r in enumerate(Rs):
        D = kT/(6*np.pi*eta*r)
        dx=dxtD/D
        Nsteps[Nrp*i:Nrp*(i+1)]=np.asarray(readingpos//dx,dtype=int)
        
    #transform Nsteps to binary array
    pow2=1<<np.arange(int(np.floor(np.log2(Nsteps.max())+1)))
    pow2=pow2[:,None]
    binSteps=np.bitwise_and(Nsteps[None,:],pow2)>0
    
    #Sort for less calculations
    sortedbs=np.argsort([str(num) for num in np.asarray(binSteps,dtype=int).T])
    
    #for each unit
    for i,bsUnit in enumerate(binSteps):
        F=getF(Fdir,2**i)
#        if i>0:#The oth step is just F
#            #Compute next F
#            F=np.dot(F,F)
            
        print("NSteps=%d" % 2**i)
        #save previous number
        prev=np.zeros(i+1,dtype=bool)
        for j,bs in enumerate(bsUnit[sortedbs]):#[sortedbs]
            prof=profilespos[sortedbs[j],:]
            act=binSteps[:i+1,sortedbs[j]]
            #If we have a one, multiply by the current step function
            if bs:
                #If this is the same as before, no need to recompute
                if (act==prev).all():
                    prof[:]=profilespos[sortedbs[j-1]]
                else:
                    prof[:]=np.dot(F,prof)
            prev=act
         
    #reshape correctly
    profilespos.shape=(NRs,Nrp,Zgrid,Ygrid)
    
    #Take mean unless asked for
    if not fullGrid:
        return np.mean(profilespos,-2)
    return profilespos
#%%        
def stepMatrixElectro(Zgrid,Ygrid,Wz,Wy,Q,D,muE,outV=None):
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
    outV: 2d float array
        array to use for the return
    Returns
    -------
    F:  2d array
        The step matrix (independent on Q)
    dxtD: float 
        The position step multiplied by the diffusion coefficient
    """
    
    #% Get The step matrix
    dy=Wy/Ygrid
#    dz=Wz/Zgrid
    #flatten V
    Vx=Q/(3600*1e9)/Wy/Wz
#    V=poiseuille(Zgrid,Ygrid,Wz,Wy,Q,outV)
#    Vx=np.ravel(V)
    
    #get Dyy
    line=np.zeros(Ygrid*Zgrid)
    line[:2]=[-2,1]
    Dyy=toeplitz(line,line) #toeplitz creation of matrice which repeat in diagonal 
    for i in range(0,Ygrid*Zgrid,Ygrid):
        Dyy[i,i]=-1
        Dyy[i-1+Ygrid,i-1+Ygrid]=-1
        if i>0 :
            Dyy[i-1,i]=0
            Dyy[i,i-1]=0

            
    #get Dy
    if muE>0:
        Dy=np.diag(np.ones(Ygrid*Zgrid),0)+np.diag(-np.ones(Ygrid*Zgrid-1),-1)
    else:
        Dy=np.diag(np.ones(Ygrid*Zgrid-1),1)+np.diag(-np.ones(Ygrid*Zgrid),0)
#    Dy=np.diag(np.ones(Ygrid*Zgrid-1),1)+np.diag(-np.ones(Ygrid*Zgrid-1),-1)
#    Dy=Dy/2    
    for i in range(0,Ygrid*Zgrid,Ygrid):
        Dy[i,i]=0
        Dy[i-1+Ygrid,i-1+Ygrid]=0
        if i>0 :
            Dy[i-1,i]=0
            Dy[i,i-1]=0
    Dy/=(dy)
    #get F
    #The formula gives F=1+dx*D/V*(1/dy^2*dyyC+1/dz^2*dzzC)
    #Choosing dx as dx=dy^2*Vmin/D, The step matrix is:
    F=1/dy**2*Dyy#+1/dz**2*Dzz
#    dx=Vx.min()/(2*D/(np.min((dy,dz))**2)+muE/(dy))/2
    dx=np.nanmin([dy*np.min(Vx)/muE,np.min(Vx)*dy**2/D/2])/2

    I=np.eye(Ygrid*Zgrid, dtype=float)
#    F=I+dx*np.dot(np.diagflat(1/Vx), D*F-muE*Dy)
#    F=I+dx*np.dot(np.diagflat(1/Vx), -muE*Dy)
    #F=I+dx*np.dot(np.diagflat(1/Vx), D*F-muE*Dy)
    F=I+dx*1/Vx*(D*F-muE*Dy)#
    
#
    #
#    F=np.dot(I+dx*D/Vx*1/dy**2*Dyy,I-dx*muE/Vx*Dy)
    
    

    #The maximal eigenvalue should be <=1! otherwhise no stability
    #The above dx should put it to 1
#    from numpy.linalg import eigvals
#    assert(np.max(np.abs(eigvals(F)))<=1.)
    return F, dx


        
#%%
if __name__ == "__main__": #__name__==" __main__" means that part is read only if it run directly and not if it is imported
    #if the script
    from glob import glob
    from natsort import natsorted
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure, plot, imshow
    cmap = matplotlib.cm.get_cmap('inferno')
    
    Yinit=np.loadtxt('basis200ulph_1to10nmLin_height50um_12prof/init.dat')
    Rs=np.arange(1,10.1,.5)*1e-9
    Np=11
    profilespos=np.empty((Np,len(Rs),12,len(Yinit)),dtype=float)  
    for i, profpos in enumerate(profilespos):
        profilespos[i]= getprofiles(Yinit,200, Rs,Zgrid=i+1)     
    
    #%%

    figure()
    for i in [0,8,17]:
        fns=natsorted(glob('basis200ulph_1to10nmLin_height50um_12prof/basis*_%03d.dat'%i))
        B=np.array([np.loadtxt(fn) for fn in fns])
        #basis is upside down
        B=B[::-1]
        errors=[]
        for irp in range(0,12,11):
            a=B[irp,:]
            figure()
            plot(a/np.sum(a))
            for b in profilespos[:,i,irp,:]:
                plot(b/np.sum(b))
            plt.legend(['basis',*[str(i) for i in range(Np)] ])
    #%%
    f=figure()
    handle=imshow(Rs[:,np.newaxis],cmap=cmap)
    f.clear()
    args=np.empty(len(Rs))
    for i in range(19):
        
        fns=natsorted(glob('basis200ulph_1to10nmLin_height50um_12prof/basis*_%03d.dat'%i))
        B=np.array([np.loadtxt(fn) for fn in fns])
        #basis is upside down
        B=B[::-1]
        errors=np.zeros((Np,),dtype=float)
        for irp in range(12):
            a=B[irp,:]
            a=a/np.sum(a)
            for j,b in enumerate(profilespos[:,i,irp,:]):
                b=b/np.sum(b)
                errors[j]+=np.mean((a-b)**2)
        X=np.arange(Np)    
        RMS=np.sqrt(errors/Np)
        plt.plot(X,RMS,'x-',c=cmap(i/19))
        argFlip=np.argwhere(RMS<1.01*RMS[-1])[0][0]
        args[i]=np.poly1d(np.polyfit(RMS[argFlip-1:argFlip+1],
                         [argFlip-1,argFlip],1))(1.01*RMS[-1])
        
    
        
        
    plt.colorbar(handle).set_label('Radii')
    plt.xlabel('Zgrid')
    plt.ylabel('Least square error')
    plt.savefig('RMS.pdf')
    #%
    F=np.poly1d(np.polyfit(np.log(Rs),args,1))    
    figure()
    plt.semilogx(Rs,args,'x')
    plt.semilogx(Rs,F(np.log(Rs)))
    plt.xlabel('Radius')
    plt.ylabel('1% error Zgrid')
    plt.savefig('1prct.pdf')
