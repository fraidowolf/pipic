# ----------------------------------------------------------------------------------------
#                     SIMULATION PARAMETERS FOR THE PIC-CODE SMILEI
# ----------------------------------------------------------------------------------------

import cmath, math
import scipy
from numpy import exp, sqrt, arctan, vectorize, real
from math import log
import numpy as np
import os,sys
import happi
import matplotlib.pyplot as plt
import h5py
import pickle
#from vacuum_propagation import *

fp_initial = './lwfa.h5'
fp_rotated = './rotated_field.h5'


def field_interp(x,y,z,field):
    points = (x,y,z)
    values = field
    return scipy.interpolate.RegularGridInterpolator(points,values,fill_value=0,bounds_error=False)

def rotate_grid(x,y,z,angle,cp):
    x_rot = x - cp[0]
    y_rot = y - cp[1]
    z_rot = z - cp[2]

    # rotation in x-z plane
    x_rot = x_rot*np.cos(angle) + z_rot*np.sin(angle)
    z_rot = -x_rot*np.sin(angle) + z_rot*np.cos(angle)
    x_rot += cp[0]
    y_rot += cp[1]
    z_rot += cp[2]

    return x_rot,y_rot,z_rot

def rotate_vector_field(field,angle):
    print(len(field))
    c1_rot = lambda x: np.cos(angle)*field[0](x) - np.sin(angle)*field[2](x)
    c2_rot = lambda x: field[1](x)
    c3_rot = lambda x: np.sin(angle)*field[0](x) + np.cos(angle)*field[2](x)
    return c1_rot,c2_rot,c3_rot

field_names = ['Ex','Ey','Ez','Bx','By','Bz']#'Bx_m','By_m','Bz_m',]
fields = []

# load fields
with h5py.File(fp_initial,'r') as f:
    for i,fn in enumerate(field_names):
        fields.append(f[fn][-1])
    x = f['x_axis'][:]
    y = f['y_axis'][:]
    z = f['z_axis'][:]


x_grid,y_grid,z_grid = np.meshgrid(x,y,z)
angle = -np.pi/4 #np.pi/4 
cp = [0,0,0] # center of rotation

# rotated grid
rnx = 2**10 
rny = 2**5 
rnz = 2**10 
x_lim = 2*np.array([x.min(),x.max()])
y_lim = 2*np.array([y.min(),y.max()])
z_lim = 2*np.array([z.min(),z.max()])
new_grid = np.mgrid[z_lim[0]:z_lim[1]:(rnx)*1j,
                    y_lim[0]:y_lim[1]:(rny)*1j,
                    z_lim[0]:z_lim[1]:(rnz)*1j]
rot_grid_x,rot_grid_y,rot_grid_z = rotate_grid(new_grid[0],new_grid[1],new_grid[2],angle,cp)

print('interpolating fields')
# interpolate fields
fieldsinterp = [field_interp(x,y,z,field) for field in fields]

# rotate fields
Exp_rot,Eyp_rot,Ezp_rot = rotate_vector_field(fieldsinterp[:3],angle)
Bxp_rot,Byp_rot,Bzp_rot = rotate_vector_field(fieldsinterp[3:],angle)

plt.figure()
plt.imshow(fields[-1][:,:,2**9],aspect='auto',cmap='coolwarm')
print(np.abs(fields[-1]).mean())
plt.colorbar()
plt.savefig('field.png')


plt.figure()
grid = np.array([x_grid.flatten(),y_grid.flatten(),z_grid.flatten()]).T
by_rot = fieldsinterp[-1](grid).reshape(x_grid.shape)
plt.imshow(by_rot[:,16,:],aspect='auto',cmap='coolwarm')
print(np.abs(by_rot).mean())
plt.colorbar()
plt.savefig('field_rot_by.png')

# detta är fel
plt.figure()
plt.imshow(Bzp_rot(grid).reshape(x_grid.shape)[:,16,:],aspect='auto',cmap='coolwarm')
plt.colorbar()
plt.savefig('field_rot_by_p.png')

print('evaluating fields on rotated grid')
# evaluate rotated fields on rotated grid
rot_grid = np.array([rot_grid_x.flatten(),
                     rot_grid_y.flatten(),
                     rot_grid_z.flatten()]).T
fields_to_eval = [Exp_rot,Eyp_rot,Ezp_rot,Bxp_rot,Byp_rot,Bzp_rot]
fields_on_grid = [field(rot_grid) for field in fields_to_eval]
fields_on_grid = [field.reshape(rnx,rny,rnz) for field in fields_on_grid]

plt.pcolormesh(new_grid[0][:,16,:],new_grid[2][:,16,:],fields_on_grid[-1][:,16,:],cmap='coolwarm')
plt.colorbar()
plt.savefig('field_rot.png')

# save rotated fields
with h5py.File(fp_rotated,'w') as f:
    for i,fn in enumerate(field_names):
        f.create_dataset(fn,data=fields_on_grid[i])
    f.create_dataset('x_axis',data=new_grid[0][:,0,0])
    f.create_dataset('y_axis',data=new_grid[1][0,:,0])
    f.create_dataset('z_axis',data=new_grid[2][0,0,:])

