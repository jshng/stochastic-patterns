# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:30:25 2019

@author: Jshin
"""

" Code to numerically solve parabolic SPDE for pattern formation analysis"
from datetime import datetime
startTime = datetime.now()

import sys
import numpy as np
import scipy as sp
import scipy.linalg
from scipy.sparse import diags
from scipy import fftpack # Used in icspde_dst1 func
import math
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
pi = math.pi


# Functions used to calculate random paths for stoachstic simulation
def icspde_dst1(u):
    return sp.fftpack.dct(u,type=1,axis=0)/2

def get_onedD_bj(dtref,J,a,r):
    """
    Adapted for Python by T. Shardlow from Alg 10.1 Page 440 of Lord
    https://github.com/tonyshardlow/PICSPDE
    """
    jj=np.arange(1,J)
    myeps=0.001
    root_qj=jj ** - ((2 * r + 1 + myeps) / 2)
    bj=root_qj * np.sqrt(2 * dtref / a)
    return bj
#
def get_onedD_dW(bj,kappa,iFspace,M):
    """
    Adapted for Python by T. Shardlow from Alg 10.2 Page 441 of Lord
    https://github.com/tonyshardlow/PICSPDE
    """
    if (kappa == 1):
        nn=np.random.randn(M,bj.size)
    else:
        nn=np.sum(np.random.randn(kappa,M,bj.size),axis=0)
    X=(bj*nn)
    if (iFspace == 1):
        dW=X
    else:
        dW=icspde_dst1(np.squeeze(X))#,type=1,axis=0)
        dW=dW.reshape(X.shape)
    return dW

'Time and Space Parameters'
reps = 1                    # Number of times the simulation is to be repeated
a = 0                        # Spatial domain x \in [a,b]
b = 1                        #        "         "
T = 1                        # Time period t \in [0,T]
dx = 1/128                   # Spatial discretisation
dt = 0.0001                  # Time discretisation
J = round((b-a)/dx)          # Total spatial steps
N = round(T/dt)              # Total time steps
n = J+1                      # Nodes per dependent variable

'Reaction Parameters'
rS = 800                    # gamma
aS = 0.2                          
bS = 1                       # Note: bS>aS
epsln = -0.1                    # Distance from bifurcation point

'Stochastic Process Parameters'
sigma_1 = 0.2                  # Amplitude of Wiener process in u equaiton
sigma_2 = 0.2                  # Amplitude of WP in v equation
sigmaS = 0                   # Amplitude of stochastic forcing of rS (gamma)
MS = 10                      # Number of stochastic realisations
rW = 0.5                     # Regulatity of Wiener Process        
aW = aS                      # Sample space interval 
kappaW = 5                   # Sample time interval = \kappaW*dt

'Reaction Kinetics'
## Schnakenberg
u_ast = aS + bS              # Steady State in u
v_ast =  bS/((aS+bS)**2)     # Steady State in v
fu=-1 + 2*u_ast*v_ast        # Derrivatives
fv = u_ast**2                #   "    "
gu = -2*u_ast*v_ast          #   "    "
gv = -u_ast**2;              #   "    "


## Geirer Meinhardt
#u_ast = (aS+1)/bS                 # Steady state
#v_ast =  u_ast**2                 # Steady state
#fu=(-bS + 2*u_ast/v_ast)          # Derrivatives
#fv = -(u_ast**2)/(v_ast**2)       #   "    "
#gu = 2*u_ast                      #   "    "
#gv = -1                           #   "    "


'calculating critical diffusion coefficient'
# Find characterisitc polynomial
aPp = fu**2
bPp = 2*(2*fv*gu - fu*gv)
cPp = gv**2
Pp = [aPp, bPp, cPp]
# Find roots of characterisitc polynomial
dd = np.roots(Pp)
# Set critical diffusion coeff as largest root
if dd[0]>1:
    dc = dd[0]
elif dd[0] <= 1:
    dc = dd[1]

# Set diffusion coefficients
D_1 = 1                            # Diffusivity in u
D_2 = dc + epsln                   # Diffusivity in v

'Calculate deterministic wavenumber'
#Critical and Maximal Wavenumber
kc2   = rS*((fu*gv - fv*gu)/dc)**0.5
km2   = rS*(D_2*fu+gv)/(2*D_2)
n_eig = (km2**(0.5))/(2*pi)

print("Critical wavenumber =",  kc2)
print("Maximal wavenumber =", km2)


# Eigenvalue corresponding to max wavenumber
Vv_1   = fu+gv
Vv_2   = D_2*fu+gv
Det    = fu*gv- fv*gu  
aP     = km2*(1+D_2) - rS*(fu+gv)
h_km   = D_2*km2**2 - rS*(D_2*fu + gv)*km2 + (rS**2)*Det 
P_eig  = [1, aP, h_km]
Eig    = np.roots(P_eig)
lamb_1 = Eig[0]
lamb_2 = Eig[1]

if (Vv_2**2-4*D_2*Det)>=0:
    k2_min= rS*(Vv_2 -(Vv_2**2 - 4*D_2*Det)**(1/2))/(2*D_2)
    k2_max = rS*(Vv_2 + (Vv_2**2 - 4*D_2*Det)**(1/2))/(2*D_2)
elif(Vv_2**2-4*D_2*Det)<0:
    z = (np.abs(Vv_2**2-4*D_2*Det)**(1/2))
    k2_min= np.complex(rS*Vv_2/(2*D_2), -(z**0.5)/(2*D_2))
    k2_max= np.complex(rS*Vv_2/(2*D_2), +(z**0.5)/(2*D_2))
    
kk = np.arange(k2_min, k2_max + 0.1, 0.1);
k_m= np.sqrt((k2_min+ k2_max)/2)/(2*pi)
m_eig=len(kk);
lambda_k=np.zeros((m_eig,1));
N_eig   = np.zeros((m_eig))
j=0;

for k2 in sp.arange(k2_min, k2_max + 0.1, 0.1):
    aPl = k2*(1+D_2) - rS*(fu+gv)
    h_k = D_2*k2**2 - rS*(D_2*fu + gv)*k2 + rS**2*Det
    lambda_k[j] = 0.5*(- aPl + (aPl**2 - 4*h_k)**(1/2))
    N_eig[j] = (k2**(0.5))/(2*pi)
    j=j+1

plt.figure(1)
plt.subplot(211)
plt.plot(kk, np.real(lambda_k),'k-', linewidth=1)

plt.subplot(212)
plt.plot(kk, N_eig, 'k--', linewidth=1)

# Calculate r for numerical analysis
r_1 = D_1*dt/(dx**2)
r_2 = D_2*dt/(dx**2)

#Vectors for storing cumulative totals
U        = np.zeros((n,N))
V        = np.zeros((n,N))
Ustore   = np.zeros((n,N,reps))
Vstore   = np.zeros((n,N,reps))
Uro      = np.zeros((N,))
Vro      = np.zeros((N,))
UroStore = np.zeros((N,reps))
VroStore = np.zeros((N,reps))
#modes    = np.ones((reps)) #one mode = haf a wavelength so modes >= 1
modes    = np.ones((N,))
rSaRep   = np.zeros((reps,))
B1       = sp.sparse.bsr_matrix((n,n))
B2       = sp.sparse.bsr_matrix((n,n))
L        = sp.sparse.bsr_matrix((n,n))
Lower1   = sp.sparse.bsr_matrix((n,n))
Lower2   = sp.sparse.bsr_matrix((n,n))
Upper1   = sp.sparse.bsr_matrix((n,n))
Upper2   = sp.sparse.bsr_matrix((n,n))
x        = np.zeros((n,1))
indexI   = np.zeros((n,1))
#Umod     = np.zeros((n, reps))  # For const xi
#Umod     = np.zeros((n,N,reps))     # variable xi
Umod     = np.zeros((n,N))
#UmodTot  = np.zeros((n,N))
# Construct matrix for solving linear part of equation
L = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n)).toarray()
# Boundary Conditions
L[0,1] = -2
L[n-1,n-2] = -2

# Central column is (1+2r), add in identity matrix
B1 = np.identity(n) + r_1*L
B2 = np.identity(n) + r_2*L

# UL decomposition of B matrices
perm1, Lower1, Upper1 = scipy.linalg.lu(B1, permute_l=False, overwrite_a=False,
                                        check_finite=True)
perm2, Lower2, Upper2 = scipy.linalg.lu(B2, permute_l=False, overwrite_a=False,
                                        check_finite=True)

#Initial data for stochastic process
indexI = np.transpose(range(n))
x = np.transpose(indexI*dx + a)
for ii in range(reps):
    #  Initialization of matrices and vectors recalculated fro each rep
    u      = np.zeros((n,1))
    v      = np.zeros((n,1))
    F      = np.zeros((n,1))
    G      = np.zeros((n,1))
    y1     = np.zeros((n,1))
    y2     = np.zeros((n,1))
    z1     = np.zeros((n,1))
    z2     = np.zeros((n,1))
    Y_1    = np.zeros((n,N))
    Y_2    = np.zeros((n,N))
    
    #Apply small pertubation from steady state
    eta = np.random.rand(len(x))
    u = u_ast + 0.1*(2*eta-1)
    v = v_ast + 0.1*(2*eta-1)
    PS = 10
    
    #Additive model parameter noise
    rSa = rS*np.ones((n,))
    rSu = rS*np.ones((n,))
    rSv = rS*np.ones((n,))
    rSa_Store = np.zeros((J+1,N))
    rSu_Store = np.zeros(N)
    
    # Time Stepping Procedure
    for nt in range(N):          
    #    Gradually increae rSa
#        rSa = rSa + 1500/N
        for js in range(MS):
            #Stochastic Processes
            # xi()/xi(n,) for time/space dependence
            xi = np.random.randn(n,) # xi()/xi(n,) for time/space dependence
            #Q-Weiner Process
            bj = get_onedD_bj(dt, n+1, aW, rW)
            dW = (get_onedD_dW(bj, kappaW, 0, 1))

            'Model parameter noise'
            rSa = rSa + sigmaS*xi           
           
            'Calculate value of functions at (xj,tn)'
            # Schnakenberg
            F = rSa*(aS - u + ((u**2)*v))
            G = rSa*(bS - (u**2)*v)
            # Geirer Meinhardt
#            F = rSa*(aS-bS*u+(u**2)/v);
#            G = rSa*((u**2)-v)
           
            'Update RHS of linear system'
            y1 = u + dt*F + u*sigma_1*dW
            y2 = v + dt*G + v*sigma_2*dW
            # Use transpose for Q-Weiner Process (squash and flip)
            y1 = np.squeeze(np.transpose(y1))
            y2 = np.squeeze(np.transpose(y2))
            
            'Solve linear system'
            #forward substitution to solve Lower1*z1 = y1 for z1
            z1 = np.linalg.solve(Lower1,y1)
            #back substitution to solve Upper1*u = z1 for u
            u  = np.linalg.solve(Upper1,z1)
            #forward substitution to solve Lower2*z2 = y2 for z2
            z2 = np.linalg.solve(Lower2,y2)
            #back substitution to solve Upper1*u = z2 for v
            v  = np.linalg.solve(Upper2,z2)
            
            'Intermediate values at each stochastic realisation'
            Y_1[:,nt] = Y_1[:,nt] + u/MS
            Y_2[:,nt] = Y_2[:,nt] + v/MS
        
    
        'Update solution matrix with (u,v) at each time step'
        U[:,nt] = U[:,nt] + u
        V[:,nt] = V[:,nt] + v

        
        'Calculate ro at each timestep to quantify patterning'
    #   Calculate means
        Umean = np.ones((J+1,))*np.mean(u)
        Vmean = np.ones((J+1,))*np.mean(v)
    #   Squared deviation from mean
        UdevSq = np.square(u - Umean)
        VdevSq = np.square(v - Vmean)
    #   Cumulate rho at each step 
        Uro[nt] = Uro[nt] + np.sum(UdevSq)
        Vro[nt] = Vro[nt] + np.sum(VdevSq)
        
        'Create storage vectors for debugging etc'
        # U and V
        Ustore[:,nt,ii] = u
        Vstore[:,nt,ii] = v
        # rSa
        rSa_Store[:,nt] = rSa
        #rho
        UroStore[nt,ii] = np.sum(UdevSq)
        VroStore[nt,ii] = np.sum(VdevSq)      

' Calculate averages over no. reps'
Uavg   = np.divide(U,reps)
Vavg   = np.divide(V,reps)        
UroAvg = np.divide(Uro, reps)
VroAvg = np.divide(Vro, reps)

' Algorithm to count number of modes for variable xi'
for nn in range(N):
    Ut = Uavg[:,nn]
    Umod[:,nn] = Ut - np.mean(Ut)
    for xx in range(len(Umod)):
        if Umod[xx,nn] < 0:
            Umod[xx,nn] = 0
        elif Umod[xx,nn] >= 0:
            Umod[xx,nn] = 1
    for xx in range(len(Umod)-1):
        step = abs(Umod[xx,nn] -Umod[xx+1, nn])
        modes[nn] = modes[nn] + step
UroT = np.zeros(N,)

'Calculate moving average for Rho' 
for nn in range(N-200):
    UroT[nn+100] = 0.005*(sum(UroAvg[nn:nn+200]))
for nn in range(100):
    UroT[nn] = 0.01*sum(UroAvg[nn:nn+100])
    UroT[N-nn-1] = 0.01*sum(UroAvg[N-100-nn-1:N-nn-1])
   
'plotting'   
fig, (ax1, ax2) = plt.subplots(figsize=(15, 5), ncols=2)
Uval = ax1.imshow(Uavg, interpolation='none', aspect = 'auto')
Vval = ax2.imshow(Vavg, interpolation='none', aspect = 'auto')
fig.colorbar(Vval, ax=ax2)
fig.colorbar(Uval, ax=ax1)
ax1.title.set_text('Concentration of u')
ax2.title.set_text('Concentration of v')
ax1.set_xlabel('t')
ax2.set_xlabel('t')
ax1.set_ylabel('x')
ax2.set_ylabel('x')

fig, (ax1, ax2) = plt.subplots(figsize = (15,5), ncols=2)
dUro = ax1.plot(UroAvg, alpha = 0.5)
Uroa = ax1.plot(UroT, color = 'k')
dVro = ax2.plot(VroAvg)
ax1.title.set_text('Relative patterning intensity in u')
ax2.title.set_text('Relative patterning intensity in v')
ax1.set_xlabel('t')
ax2.set_xlabel('t')
ax1.set_ylabel(r'$\rho$')
ax2.set_ylabel(r'$\rho$')

fig, (ax1, ax2) = plt.subplots(figsize = (15,5), ncols=2)
modeplot = ax1.hist(modes, bins = (np.amax(modes.astype('int32'))- 
                                   np.amin(modes.astype('int32'))+1), 
    range = [np.amin(modes), np.amax(modes)+1], alpha = 0.5, histtype = 
    'bar', ec='black')
sharp    = ax2.imshow(Umod, aspect = 'auto')
ax1.title.set_text('Modes present in averaged simulation')
ax2.title.set_text('Sharpened concetration of u')
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_xlabel('Mode')
ax2.set_xlabel('t')
ax1.set_ylabel('Frequency')
ax2.set_ylabel('x')
print(datetime.now() - startTime)

#plt.plot(rSa_Store)
#plt.xlabel('t')
#plt.ylabel('r')
#plt.title('Evolution of $r$')

    