#Basic setup for a laser pulse interation with a solid-density plasma layer 
#for results see fig. 6 in arXiv:2302.01893
import sys
import pipic
from pipic import consts,types
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numba import cfunc, carray, types as nbt
from pipic.extensions import moving_window
import h5py



def create_hdf5(fp, shape, dsets=['Ex','Ez','Ey','Bx','Bz','By','rho'],mode="w"):

    # Create a new HDF5 file
    file = h5py.File(fp, mode)

    # Create a dataset
    for d in dsets:
        file.create_dataset(d, shape=shape, dtype=np.double)
    
    file.close()



fp_rot = './rotated_field.h5'
nz,ny,nx = 2**10, 2**5, 2**10
Ex_init = np.zeros((nx,ny,nz), dtype=np.double)
Ey_init = np.zeros((nx,ny,nz), dtype=np.double)
Ez_init = np.zeros((nx,ny,nz), dtype=np.double)
Bx_init = np.zeros((nx,ny,nz), dtype=np.double)
By_init = np.zeros((nx,ny,nz), dtype=np.double)
Bz_init = np.zeros((nx,ny,nz), dtype=np.double)

# load fields
with h5py.File(fp_rot,'r') as f:
    Ex_init = f['Ex'][:]
    Ey_init = f['Ey'][:]
    Ez_init = f['Ez'][:]
    Bx_init = f['Bx'][:]
    By_init = f['By'][:]
    Bz_init = f['Bz'][:]

    x = f['x_axis'][:]
    y = f['y_axis'][:]
    z = f['z_axis'][:]


# constants
c = 299792458*1e2 #cm/s
m = 9.1093837*1e-31*1e3 #g
e = 4.80320425*1e-10  # statCoulomb

### Smilei parameters
# references
wl = 0.8e-4 # cm
omega_r = 2*np.pi*c/wl # rad/s

# conversions
Nr = m*omega_r**2/(np.pi*4*e**2) # 1/cm³
Kr = 0.51099895000 # m*c**2, MeV 
Lr = c/omega_r # cm/2*pi
Tr = 1/omega_r # s/2*pi
Er = m*c*omega_r/e #statV/cm = (sqrt(g/cm)/s)

n0 = 4e18
n1 = n0/3

n0_s = n0/Nr
a0 = 4.
omega_s = 1. 
fwhm_duration_laser_pulse_s = 16*2*np.pi
waist_s = 10*2*np.pi #4*fwhm_duration_laser_pulse/2.335
distance_laser_peak_window_border_s = 1.7*fwhm_duration_laser_pulse_s

#===========================SIMULATION INITIALIZATION===========================
xmin, xmax = x.min(),x.max()
ymin, ymax = y.min(),y.max()
zmin, zmax = z.min(),z.max()
dx, dy, dz = (xmax - xmin)/nx, (ymax - ymin)/ny, (zmax - zmin)/nz
timestep = dz/consts.light_velocity/2
thickness = 10 # thickness (in dx) of the area where the density and field is restored/removed 

s = int(10*fwhm_duration_laser_pulse_s*Tr/timestep) #3000*10 # number of iterations 
checkpoint = 100   
print(s)



#---------------------setting solver and simulation region----------------------
sim=pipic.init(solver='ec2',nx=nx,ny=ny,nz=nz,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,zmin=zmin,zmax=zmax)
sim.en_corr_type(2)

#---------------------------setting field of the pulse--------------------------

# conversions laser parameters
omega_l = omega_s * omega_r # [rad/s]
fieldAmplitude = a0*m*c*omega_l/e # [statV/cm]
wavelength = 2*np.pi*consts.light_velocity/omega_l # [cm]

fwhm_duration_laser_pulse = Tr*fwhm_duration_laser_pulse_s
pulseWidth_x = (fwhm_duration_laser_pulse/2.355)*consts.light_velocity # [cm]

waist = waist_s*Lr
print(waist)
init_laser_pos = 0
focusPosition = np.pi*waist**2/wavelength 

@cfunc(types.field_loop_callback)
def initiate_field_callback(ind, r, E, B, data_double, data_int):
    if data_int[0] == 0:
        E[0] = Ex_init[ind[0],ind[1],ind[2]]
        E[1] = Ey_init[ind[0],ind[1],ind[2]]
        E[2] = Ez_init[ind[0],ind[1],ind[2]]
        B[0] = Bx_init[ind[0],ind[1],ind[2]]
        B[1] = By_init[ind[0],ind[1],ind[2]]
        B[2] = Bz_init[ind[0],ind[1],ind[2]]


@cfunc(types.field_loop_callback)
def remove_field(ind, r, E, B, data_double, data_int):

    rollback = np.floor(data_int[0]*timestep*consts.light_velocity/dz)
    if rollback%(thickness//2)==0:
        r_rel = zmin + dz*(rollback%nz)  
        r_min = r_rel - thickness*dz
        r_max = r_rel 
        if (r[2] > r_min and r[2] < r_max) or (r[2] > zmax - (zmin - r_min)) or (r[2] < zmin + (r_max - zmax)): 
            E[1] = 0
            B[2] = 0
            E[2] = 0 
            B[1] = 0
            E[0] = 0
            B[0] = 0 


#=================================OUTPUT========================================
#-------------------------preparing output of fields (x)-----------------------------
Ex = np.zeros((nx,nz), dtype=np.double) 
Ey = np.zeros((nx,nz), dtype=np.double)
Ez = np.zeros((nx,nz), dtype=np.double)
Bx = np.zeros((nx,nz), dtype=np.double)
By = np.zeros((nx,nz), dtype=np.double)
Bz = np.zeros((nx,nz), dtype=np.double)

rho = np.zeros((ny,nz), dtype=np.double) 
# not done
pmin = 0
pmax = 1e-15
nps = 2**8
ps = np.zeros((nps,nz), dtype=np.double) 

#------------------get functions-----------------------------------------------
    
@cfunc(types.particle_loop_callback)
def get_density(r, p, w, id, data_double, data_int):   
    ix = int(nx*(r[0] - xmin)/(xmax - xmin))
    iy = int(ny*(r[1] - ymin)/(ymax - ymin))
    iz = int(nz*(r[2] - zmin)/(zmax - zmin))

    data = carray(data_double, rho.shape, dtype=np.double)
    
    if (iy < rho.shape[0] and 
        ix == nx//2 and
        iz < rho.shape[1]):
        data[iy, iz] += w[0]/(dx*dy*dz)

@cfunc(types.particle_loop_callback)
def get_phase_space(r, p, w, id, data_double, data_int):   
    iz = int(ps.shape[1]*(r[2] - zmin)/(zmax - zmin))
    ip = int(ps.shape[0]*(p[2] - pmin)/(pmax - pmin))
    data = carray(data_double, ps.shape, dtype=np.double)
    
    if ip>=0 and ip < ps.shape[0] and iz < ps.shape[1]:
        data[ip,iz] += w[0]/(dx*dy*dz) 


@cfunc(types.field_loop_callback)
def get_field_Ex(ind, r, E, B, data_double, data_int):
    if ind[1] == ny//2:
        _E = carray(data_double, Ex.shape, dtype=np.double)
        _E[ind[0], ind[2]] = E[0]

@cfunc(types.field_loop_callback)
def get_field_Ey(ind, r, E, B, data_double, data_int):
    if ind[1] == ny//2:
        _E = carray(data_double, Ey.shape, dtype=np.double)
        _E[ind[0], ind[2]] = E[1]

@cfunc(types.field_loop_callback)
def get_field_Ez(ind, r, E, B, data_double, data_int):
    if ind[1] == ny//2:
        _E = carray(data_double, Ez.shape, dtype=np.double)
        _E[ind[0], ind[2]] = E[2]

@cfunc(types.field_loop_callback)
def get_field_Bx(ind, r, E, B, data_double, data_int):
    if ind[1] == ny//2:
        _B = carray(data_double, Bx.shape, dtype=np.double)
        _B[ind[0], ind[2]] = B[0]

@cfunc(types.field_loop_callback)
def get_field_By(ind, r, E, B, data_double, data_int):
    if ind[1] == ny//2:
        _B = carray(data_double, By.shape, dtype=np.double)
        _B[ind[0], ind[2]] = B[1]

@cfunc(types.field_loop_callback)
def get_field_Bz(ind, r, E, B, data_double, data_int):
    if ind[1] == ny//2:
        _B = carray(data_double, Bz.shape, dtype=np.double)
        _B[ind[0], ind[2]] = B[2]

def load_fields():
    sim.field_loop(handler=get_field_Ey.address, data_double=pipic.addressof(Ey),use_omp=True)
    sim.field_loop(handler=get_field_Ex.address, data_double=pipic.addressof(Ex),use_omp=True)
    sim.field_loop(handler=get_field_Ez.address, data_double=pipic.addressof(Ez),use_omp=True)
    sim.field_loop(handler=get_field_Bx.address, data_double=pipic.addressof(Bx),use_omp=True)
    sim.field_loop(handler=get_field_By.address, data_double=pipic.addressof(By),use_omp=True)
    sim.field_loop(handler=get_field_Bz.address, data_double=pipic.addressof(Bz),use_omp=True)

#===============================SIMULATION======================================

data_int = np.zeros((1, ), dtype=np.intc) # data for passing the iteration number
window_speed = consts.light_velocity #speed of moving window

#-----------------------adding the handler of extension-------------------------

# variable for passing density to the handler  
data_double = np.zeros((1, ), dtype=np.double)

#-----------------------initiate field and plasma-------------------------
sim.fourier_solver_settings(divergence_cleaning=1, sin2_kfilter=-1)
sim.advance(time_step=0, number_of_iterations=1,use_omp=True)
sim.field_loop(handler=initiate_field_callback.address, data_int=pipic.addressof(data_int),
                use_omp=True)
sim.advance(time_step=0, number_of_iterations=1,use_omp=True)
sim.fourier_solver_settings(divergence_cleaning=0, sin2_kfilter=-1)
'''

sim.field_loop(handler=initiate_field_callback.address, data_int=pipic.addressof(data_int),
                use_omp=True)
'''
#-----------------------run simulation-------------------------


dsets = ['Ex','Ey','Ez','Bx','By','Bz']
fields = [Ex,Ey,Ez,Bx,By,Bz]
ncp = s//checkpoint

hdf5_fp = 'lwfa_restarted.h5'
create_hdf5(hdf5_fp, shape=(ncp,nx,nz),dsets=dsets[:])

create_hdf5(hdf5_fp,shape=(nx,),dsets=['x_axis',],mode="r+")
create_hdf5(hdf5_fp,shape=(ny,),dsets=['y_axis',],mode="r+")
create_hdf5(hdf5_fp,shape=(nz,),dsets=['z_axis',],mode="r+")

x_axis = np.linspace(xmin, xmax, nx)
y_axis = np.linspace(ymin, ymax, ny)
z_axis = np.linspace(zmin, zmax, nz)
p_axis = np.linspace(pmin, pmax, nps)

with h5py.File(hdf5_fp,"r+") as file:
    file['x_axis'][:] = x_axis
    file['y_axis'][:] = y_axis
    file['z_axis'][:] = z_axis


for i in range(s):
    
    data_int[0] = i 

    sim.advance(time_step=timestep, number_of_iterations=1,use_omp=True)
    '''
    sim.field_loop(handler=remove_field.address, data_int=pipic.addressof(data_int),
                use_omp=True)
    '''
    if i%checkpoint==0:
        #with open('./out.txt', 'a') as f:
        #    print(f'{i}\n', file=f)
        print(i)
        # load fields and densities            
        load_fields()
        print('Ex:',Ex.sum())
        roll_back =  0#int(np.floor(i*timestep*window_speed/dz))  

        try:
            with h5py.File(hdf5_fp,"r+") as file:
                for j,f in enumerate(dsets):
                    file[f][i//checkpoint] = np.roll(fields[j],-roll_back,axis=-1) 
        except IOError:
            input('Another process are accessing the hdf5 file. Close the file and press enter.')
            with h5py.File(hdf5_fp,"r+") as file:
                for j,f in enumerate(dsets):
                    file[f][i//checkpoint] = np.roll(fields[j],-roll_back,axis=-1)    

