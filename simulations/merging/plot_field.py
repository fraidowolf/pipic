import matplotlib.pyplot as plt
import numpy as np
import h5py

with h5py.File('rotated_field.h5','r+') as f:
    field = f['Bx'][:,16,:]
    ny = field.shape[1]
    plt.imshow(field[:,:],aspect='equal',cmap='coolwarm',extent=[f['z_axis'][0],f['z_axis'][-1],f['x_axis'][0],f['x_axis'][-1]])
    plt.colorbar()
    print(f['Ex'].shape)
#plt.hlines(0,-0.08,0.08,linestyles='dashed')
x = np.linspace(-0.005,0.005)
#plt.plot(x,-x)
#plt.xlim(-0.0025,0.0025)
#plt.ylim(-0.0025,0.0025)

plt.savefig('field.png',dpi=1000)