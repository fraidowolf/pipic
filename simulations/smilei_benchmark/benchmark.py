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


# Smilei parameters
dz_s = 0.8            # longitudinal resolution
dy_s = 4.
dx_s = dy_s              # transverse resolution
dt = 0.8*dz_s          # timestep
nz,ny,nx = 32*10,32,32 #32,32
Lx = nx*dx_s 
Ly = ny*dy_s
Lz = nz*dz_s
a0 = 4.0
n0 = 10e18 # [1/cm^3]
n1 = n0/3

c = consts.light_velocity # [cm/s]
m = consts.electron_mass # [g]
e = consts.electron_charge # [statC]
omega_r = np.sqrt(np.pi*4*e**2*n0/m)

# conversion factors
Lr = c/omega_r
Tr = 1/omega_r


#===========================SIMULATION INITIALIZATION===========================
xmin, xmax = -Lx*Lr/2, Lx*Lr/2
ymin, ymax = -Ly*Lr/2, Ly*Lr/2
zmin, zmax = -Lz*Lr/2, Lz*Lr/2
nz = 32*16
ny = 32
nx = ny
dx, dy, dz = (xmax - xmin)/nx, (ymax - ymin)/ny, (zmax - zmin)/nz
timestep = dt*Tr/1e1
thickness = 10 # thickness (in dx) of the area where the density and field is restored/removed 

s = 1000 #3000*10 # number of iterations 
checkpoint = 10   


#---------------------setting solver and simulation region----------------------
sim=pipic.init(solver='ec',nx=nx,ny=ny,nz=nz,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,zmin=zmin,zmax=zmax)

#---------------------------setting field of the pulse--------------------------

# conversions laser parameters
omega_l = 2.5 * omega_r # [1/s]
fieldAmplitude = a0*m*c*omega_l/e # [statV/cm]
omega0 = 2*np.pi*consts.light_velocity/2.e-4 # [1/s] # note not the laser frequency!
fwhm_duration_laser_pulse = Tr*20e-15 * omega0 *2**0.5 # [s]
pulseWidth_x = (fwhm_duration_laser_pulse/2.355)*consts.light_velocity # [cm]
wavelength = 2*np.pi*consts.light_velocity/omega_l # [cm]
waist = fwhm_duration_laser_pulse*consts.light_velocity 
focusPosition = Lz*Lr
init_laser_pos = Lz*Lr/2-1.7*fwhm_duration_laser_pulse*Lr/Tr

omega_p = np.sqrt(4*np.pi*n0*consts.electron_charge**2/consts.electron_mass)
wp = 2*np.pi*consts.light_velocity/omega_p
print(focusPosition,init_laser_pos)

@cfunc(types.field_loop_callback)
def initiate_field_callback(ind, r, E, B, data_double, data_int):

    if data_int[0] == 0:       
        x = r[2] - init_laser_pos
        rho2 = r[1]**2 + r[0]**2
        
        k = 2*np.pi/wavelength
        # Rayleigh length
        Zr = np.pi*waist**2/wavelength 
        # curvature
        R = focusPosition*(1+(Zr/focusPosition)**2)
        
        spotsize_init = waist*np.sqrt(1+(focusPosition/Zr)**2)
        phase = np.arctan(focusPosition/Zr)    
        amp = fieldAmplitude*(waist/spotsize_init)*np.exp(-rho2/spotsize_init**2)
        curvature = np.exp(1j*k*rho2/(2*R))

        gp = np.real(amp*curvature*np.exp(-1j*(k*x + phase))*np.exp(-x**2/(2*pulseWidth_x**2)))

        # x-polarized 
        E[0] = gp       
        B[1] = gp        

#--------------------------- setting plasma profile --------------------------

density = n0
debye_length = .1*wavelength/64.0 #>>3/(4*pi*density), <dx ???
temperature = 0 #4 * np.pi * density * (consts.electron_charge ** 2) * debye_length ** 2
particles_per_cell = 1 # 8 in smilei

Lupramp  = 10*dz_s*Lr
Lplateau = Lz*1.5*Lr
Ldownramp = 10*dz_s*Lr

begin_upramp = (Lz/2)*Lr
xplateau = begin_upramp + Lupramp # Start of the plateau
begin_downramp = xplateau + Lplateau # Beginning of the output ramp.

xplateau2 = begin_downramp + Ldownramp/100 # Beginning of the output ramp.
begin_downramp2 = xplateau2+ Lplateau*3 # Beginning of the output ramp.
finish = begin_downramp2 + Ldownramp # End of plasma


@cfunc(types.add_particles_callback)
def density_profile(r, data_double, data_int):
    # r is the position in the 'lab frame'  
    R = r[2]
    
    if R < begin_upramp:
        return 0
    elif R < xplateau:
        return n0*((R-begin_upramp)/Lupramp)
    elif R < begin_downramp: 
        return n0
    elif R < xplateau2:
        return n0*(1-((R-begin_downramp)/(Ldownramp/100)))
    elif R < begin_downramp2:
        return n1
    elif R < finish:
        return n1*((R-begin_downramp2)/Ldownramp)
    else:
        return 0
 
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
Ex = np.zeros((nx,ny,nz), dtype=np.double) 
Ey = np.zeros((nx,ny,nz), dtype=np.double) 
Ez = np.zeros((nx,ny,nz), dtype=np.double) 

rho = np.zeros((nx,ny,nz), dtype=np.double) 
# not done
pmin = 0
pmax = 1e-15
nps = 2**8
ps = np.zeros((nps,nz), dtype=np.double) 

#------------------get functions-----------------------------------------------
    
@cfunc(types.particle_loop_callback)
def get_density(r, p, w, id, data_double, data_int):   
    ix = int(rho.shape[0]*(r[0] - xmin)/(xmax - xmin))
    iy = int(rho.shape[1]*(r[1] - ymin)/(ymax - ymin))
    iz = int(rho.shape[2]*(r[2] - zmin)/(zmax - zmin))

    data = carray(data_double, rho.shape, dtype=np.double)
    
    if (iy < rho.shape[1] and 
        ix < rho.shape[0] and
        iz < rho.shape[2]):
        data[ix, iy, iz] += w[0]/(dx*dy*dz)

@cfunc(types.particle_loop_callback)
def get_phase_space(r, p, w, id, data_double, data_int):   
    iz = int(ps.shape[0]*(r[2] - zmin)/(zmax - zmin))
    ip = int(ps.shape[1]*(p[2] - pmin)/(pmax - pmin))
    data = carray(data_double, ps.shape, dtype=np.double)
    
    if ip>=0 and ip < ps.shape[1] and iz < ps.shape[0]:
        data[ip,iz] += w[0]/(dx*dy*dz) 


@cfunc(types.field_loop_callback)
def get_field_Ey(ind, r, E, B, data_double, data_int):
    _E = carray(data_double, Ey.shape, dtype=np.double)
    _E[ind[0], ind[1], ind[2]] = E[1]

@cfunc(types.field_loop_callback)
def get_field_Ex(ind, r, E, B, data_double, data_int):       
    _E = carray(data_double, Ex.shape, dtype=np.double)
    _E[ind[0], ind[1], ind[2]] = E[0]

@cfunc(types.field_loop_callback)
def get_field_Ez(ind, r, E, B, data_double, data_int):      
    _E = carray(data_double, Ez.shape, dtype=np.double)
    _E[ind[0], ind[1], ind[2]] = E[2]


def load_fields():
    sim.field_loop(handler=get_field_Ey.address, data_double=pipic.addressof(Ey))
    sim.field_loop(handler=get_field_Ex.address, data_double=pipic.addressof(Ex))
    sim.field_loop(handler=get_field_Ez.address, data_double=pipic.addressof(Ez))

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

# This part is just for initiating the electron species, 
# so that the algorithm knows that there is a species called electron
# it is therefore not important where the electrons are or what density they have
sim.add_particles(name='electron', number=int(ny*nz*nx),
                charge=consts.electron_charge, mass=consts.electron_mass,
                temperature=temperature, density=density_profile.address,
                data_int=pipic.addressof(data_int))

density_handler_adress = moving_window.handler(thickness=thickness,
                                               particles_per_cell=particles_per_cell,
                                               temperature=temperature,
                                               density = density_profile.address,)
sim.add_handler(name=moving_window.name, 
                subject='electron,cells',
                handler=density_handler_adress,
                data_int=pipic.addressof(data_int),)

sim.field_loop(handler=initiate_field_callback.address, data_int=pipic.addressof(data_int),
                use_omp=True)

#-----------------------run simulation-------------------------


dsets = ['Ex','Ey','Ez','rho','ps']
fields = [Ex,Ey,Ez,rho,ps]
ncp = s//checkpoint

hdf5_fp = 'lwfa.h5'
create_hdf5(hdf5_fp, shape=(ncp,nx,ny,nz),dsets=dsets[:-1])
create_hdf5(hdf5_fp, shape=(ncp,nps,nz),dsets=['ps'],mode="r+")

create_hdf5(hdf5_fp,shape=(nx,),dsets=['x_axis',],mode="r+")
create_hdf5(hdf5_fp,shape=(ny,),dsets=['y_axis',],mode="r+")
create_hdf5(hdf5_fp,shape=(nz,),dsets=['z_axis',],mode="r+")
create_hdf5(hdf5_fp,shape=(nps,),dsets=['p_axis',],mode="r+")

x_axis = np.linspace(xmin, xmax, nx)
y_axis = np.linspace(ymin, ymax, ny)
z_axis = np.linspace(zmin, zmax, nz)
p_axis = np.linspace(pmin, pmax, nps)

with h5py.File(hdf5_fp,"r+") as file:
    file['x_axis'][:] = x_axis
    file['y_axis'][:] = y_axis
    file['z_axis'][:] = z_axis
    file['p_axis'][:] = p_axis





for i in range(s):
    
    data_int[0] = i 

    sim.advance(time_step=timestep, number_of_iterations=1,use_omp=True)
    
    sim.field_loop(handler=remove_field.address, data_int=pipic.addressof(data_int),
                use_omp=True)
    
  
    if i%checkpoint==0:
        print(i)
        rho.fill(0)
        ps.fill(0)
        # load fields and densities            
        sim.particle_loop(name='electron', handler=get_density.address,
                    data_double=pipic.addressof(rho))
        sim.particle_loop(name='electron', handler=get_phase_space.address,
                    data_double=pipic.addressof(ps))
        load_fields()
        roll_back =  int(np.floor(i*timestep*window_speed/dz))  

        try:
            with h5py.File(hdf5_fp,"r+") as file:
                for j,f in enumerate(dsets):
                    file[f][i//checkpoint] = np.roll(fields[j],-roll_back,axis=-1) 
        except IOError:
            input('Another process are accessing the hdf5 file. Close the file and press enter.')
            with h5py.File(hdf5_fp,"r+") as file:
                for j,f in enumerate(dsets):
                    file[f][i//checkpoint] = np.roll(fields[j],-roll_back,axis=-1)    

