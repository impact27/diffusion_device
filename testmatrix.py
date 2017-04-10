# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 22:53:06 2017

@author: quentinpeter
"""

import diffusionDevice.basisgenerate as ddbg
import numpy as np
from matplotlib.pyplot import figure, plot, imshow
import matplotlib.pyplot as plt
import importlib
importlib.reload(ddbg)
Zgrid=1

Q= 200 #ul/h
#Eta= 0.001 Pa*s
Wz= 50e-6 #um
Wy= 300e-6 #um
Rs=1e-9
l=90.5*1e-3
kT = 1.38e-23*295;
eta = 1e-3;
D = kT/(6*np.pi*eta*Rs)
basis=np.loadtxt('basis1_000.dat')
init=np.loadtxt('init.dat')
init2=np.zeros(192)
init2[:96]=1
init/=np.mean(init)
basis=basis/np.mean(basis)*np.mean(init)
Ygrid=len(basis)

V,Viy,Viz=ddbg.poiseuille(Zgrid,Ygrid,Wz,Wy,Q,get_interface=True)
#%%
dy=Wy/Ygrid

N=(Ygrid+1)//2-1
M=np.diag(np.ones(N-1),1)+np.diag(np.ones(N))
M[-1,-1]=2

C=1/V[0,1:N+1]
E=2*np.linalg.inv(M)@C
if Ygrid%2==0:
    E=np.concatenate((E,[E[-1]],E[::-1]))
else:
    E=np.concatenate((E,E[::-1]))
Cyy=np.diag(E,1)+np.diag(E,-1)

Cyy-=np.diag(np.sum(Cyy,1))
Cyy/=dy**2

dxtD1=dy**2*V.min()/2
I=np.eye(Ygrid*Zgrid, dtype=float)
M2=I+dxtD1*(Cyy)

#%%
halfy=np.ravel(np.concatenate((1/Viy,np.zeros((Zgrid,1))),1))[:-1]
#get Cyy
Cyy=np.diag(halfy,-1)+np.diag(halfy,1)
Cyy-=np.diag(np.sum(Cyy,0))
Cyy/=dy**2

M3=I+dxtD1*(Cyy)

#%%
udiag=np.ones(Ygrid*Zgrid-1)
udiag[Ygrid::Ygrid]=0
Cyy=np.diag(udiag,1)+np.diag(udiag,-1)
Cyy-=np.diag(np.sum(Cyy,0))
print(Cyy)
Dy=np.diag(-np.ones(Ygrid-1),-1)+np.diag(np.ones(Ygrid-1),1)
Dy[-1,-1]=1
Dy[0,0]=-1
Cyy=np.dot(np.diag(1/V[0]),Cyy)
Cyy=Cyy+np.diag(Dy@(1/V[0]))@Dy
Cyy/=dy**2


M4=I+dxtD1*(Cyy)
#%%
M1,dxtD=ddbg.stepMatrix(Zgrid,Ygrid,Wz,Wy,Q)

init=np.tile(init,Zgrid)

Nsteps=int(l/(dxtD/D))

p1=np.mean(np.dot(np.linalg.matrix_power(M1,Nsteps),np.ravel(init)).reshape((Zgrid,Ygrid)),0)
p2=np.mean(np.dot(np.linalg.matrix_power(M2,Nsteps),np.ravel(init)).reshape((Zgrid,Ygrid)),0)
p3=np.mean(np.dot(np.linalg.matrix_power(M3,Nsteps),np.ravel(init)).reshape((Zgrid,Ygrid)),0)
p4=np.mean(np.dot(np.linalg.matrix_power(M4,Nsteps),np.ravel(init)).reshape((Zgrid,Ygrid)),0)



#%%
figure()
plot(basis,label='simulation')
plot(p1,label='First Method')
plot(p1/np.mean(p1),label='First Method normalized')

#plot(p4,label='Edges')

plt.legend()
#%%
dx=dxtD/D
Nsteps=47834//2
p1=np.mean(np.dot(np.linalg.matrix_power(M1,Nsteps),np.ravel(init)).reshape((Zgrid,Ygrid)),0)
figure()
plot(p1)
a2=plt.axes().twinx()
a2.plot(np.diff(p1)*np.diff(V[0]),c='C1')

#%%
res=[]
listNsteps=np.exp(np.linspace(0,12))
for n in listNsteps:
    p1=np.mean(np.dot(np.linalg.matrix_power(M1,int(n)),np.ravel(init)).reshape((Zgrid,Ygrid)),0)
    res.append(sum(p1))
figure()
plot(listNsteps[:-1]+np.diff(listNsteps),np.diff(res),'x')
