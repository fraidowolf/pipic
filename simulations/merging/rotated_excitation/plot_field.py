import matplotlib.pyplot as plt
import numpy as np
import h5py

with h5py.File('lwfa.h5','r+') as f:
    field = f['rho'][16][:,32,:]
    ny = field.shape[1]
    x = f['x_axis'][:]
    z = f['z_axis'][:]
    plt.pcolormesh(z,x,field,cmap='Reds',shading='auto')
    plt.colorbar()
    print(f['Ex'].shape)
#plt.hlines(0,-0.08,0.08,linestyles='dashed')
x = np.linspace(-0.005,0.005)
#plt.plot(x,-x)
#plt.xlim(-0.0025,0.0025)
#plt.ylim(-0.0025,0.0025)

plt.savefig('field.png',dpi=1000)