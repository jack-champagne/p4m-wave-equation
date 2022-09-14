import numpy as np
from skimage.transform import resize # for Coarsening 
import wave2

def ParallelCompute(v,vt,vel,velX,dx,dX,dt,dT,cT):
    '''
    Compute coarse and fine solutions
    Input
    v,vt: array of wavefields, on fine grid,
            containing initial condition
    vel,velX: wavespeed on fine and coarse grid
    dx, dX: fine and coarse grid size
    dt, dT: fien and coarse step size
    cT: timeslice duration
    '''
    ncT = v.shape[2]
    ny,nx = velX.shape
    Ny,Nx = vel.shape
    
    # Allocate arrays for output
    uf = np.zeros([Ny,Nx,ncT])
    utf = np.zeros([Ny,Nx,ncT])
    uc = np.zeros([ny,nx,ncT])
    utc = np.zeros([ny,nx,ncT])
    
    # Parallel loop
    # Each rhs is independent of lhs
    for j in range(ncT-1):
            ucx,utcx = wave2.wave2(resize(v[:,:,j],[ny,nx],order=4),resize(vt[:,:,j],[ny,nx],order=4),\
                                        velX,dX,dT,cT)
            uc[:,:,j+1] = ucx#resize(ucx,[Ny,Nx],order=4)
            utc[:,:,j+1] = utcx#resize(utcx,[Ny,Nx],order=4)
            
            uf[:,:,j+1],utf[:,:,j+1] = wave2.wave2(v[:,:,j],vt[:,:,j],vel,dx,dt,cT)
            
    return uc,utc,uf,utf


def ParallelSyncCompute(v,vt,vel,velX,dx,dX,dt,dT,cT):
    ncT = v.shape[2]
    ny,nx = velX.shape
    Ny,Nx = vel.shape
    
    # Allocate arrays for output
    uf = np.zeros([Ny,Nx,ncT])
    utf = np.zeros([Ny,Nx,ncT])
    uc = np.zeros([Ny,Nx,ncT])
    utc = np.zeros([Ny,Nx,ncT])
    
    # Parallel loop
    # Each rhs is independent of lhs
    for j in range(ncT):
            ucx,utcx = wave2.wave2(resize(v[:,:,j],[ny,nx],order=4),resize(vt[:,:,j],[ny,nx],order=4),\
                                        velX,dX,dT,cT)
            uc[:,:,j] = resize(ucx,[Ny,Nx],order=4)
            utc[:,:,j] = resize(utcx,[Ny,Nx],order=4)
            
            uf[:,:,j],utf[:,:,j] = wave2.wave2(v[:,:,j],vt[:,:,j],vel,dx,dt,cT)
            
    return uc,utc,uf,utf
    
    