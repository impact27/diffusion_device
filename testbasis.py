# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:27:09 2017

@author: quentinpeter
"""
#if the script
from glob import glob
from natsort import natsorted
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, imshow
cmap = matplotlib.cm.get_cmap('inferno')
import numpy as np

Yinit=np.loadtxt('init.dat')
Rs=np.arange(1,10.1,.5)*1e-9
Np=11
profilespos=np.empty((Np,len(Rs),12,len(Yinit)),dtype=float)  
for i, profpos in enumerate(profilespos):
    profilespos[i]= getprofiles(Yinit,200, Rs,Zgrid=i+1)     

#%%

figure()
for i in [0,8,17]:
    fns=natsorted(glob('basis*_%03d.dat'%i))
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
    
    fns=natsorted(glob('basis*_%03d.dat'%i))
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
