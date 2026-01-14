#Basic setup for a laser pulse interation with a solid-density plasma layer 
#for results see fig. 6 in arXiv:2302.01893
import sys
import pipic
from pipic import consts,types
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numba import cfunc, carray, types as nbt
import h5py



def create_hdf5(fp, shape, dsets=['Ex','Ez','Ey','Bx','Bz','By','rho'],mode="w"):

    # Create a new HDF5 file
    file = h5py.File(fp, mode)

    # Create a dataset
    for d in dsets:
        file.create_dataset(d, shape=shape, dtype=np.double)
    
    file.close()


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

# smilei simulation parameters
dx_s = 2*np.pi/64           # longitudinal resolution
dt_s = dx_s/4         # timestep
nx = 16*32*32
Lx = nx*dx_s

n0 = 4e18
n1 = n0/3

n0_s = n0/Nr
a0 = 4.
omega_s = 1. 
fwhm_duration_laser_pulse_s = 15*2*np.pi
waist_s = 20*2*np.pi #4*fwhm_duration_laser_pulse/2.335
distance_laser_peak_window_border_s = 1.7*fwhm_duration_laser_pulse_s

#===========================SIMULATION INITIALIZATION===========================
xmin, xmax = -Lx*Lr/2, Lx*Lr/2
dx = (xmax - xmin)/nx
timestep = dt_s*Tr

s = 80000 #3000*10 # number of iterations 
checkpoint = 100  



#---------------------setting solver and simulation region----------------------
sim=pipic.init(solver='ec2',nx=nx,xmin=xmin,xmax=xmax)
sim.en_corr_type(2)
#--------------------------- setting plasma profile --------------------------




begin_upramp = xmin/2
end_upramp = Lx*Lr*4 + begin_upramp
begin_downramp = end_upramp
end_downramp = begin_downramp
endplateau = end_downramp + 4*Lx*Lr
finish = endplateau


debye_length = .1*wl/64.0 #>>3/(4*pi*density), <dx ???
temperature = 0 #4 * np.pi * n0 * (consts.electron_charge ** 2) * debye_length ** 2
totalNumberOfParticles = n0*(xmax-begin_upramp)**2/(end_upramp-begin_upramp)/2
particles_per_macro = 0.5e10
particles = totalNumberOfParticles/particles_per_macro


@cfunc(types.add_particles_callback)
def density_profile(r, data_double, data_int):
    # r is the position in the 'lab frame'  
    R = r[0] #+ thickness*dz

    if R >= begin_upramp and R < end_upramp:
        return n0*((R-begin_upramp)/(end_upramp-begin_upramp))
    elif R >= end_upramp and R < begin_downramp: 
        return n0
    elif R >= begin_downramp and R > end_downramp and (end_downramp != begin_downramp):
        return n0-(n0-n1)*(R-begin_downramp)/(end_downramp-begin_downramp)
    elif R >= end_downramp and R < endplateau:
        return n1
    else:
        return 0
    
#---------------------------setting field of the pulse--------------------------

# conversions laser parameters
omega_l = omega_s * omega_r # [rad/s]
fieldAmplitude = -a0*m*c*omega_l/e # [statV/cm]
wavelength = 2*np.pi*consts.light_velocity/omega_l # [cm]

fwhm_duration_laser_pulse = Tr*fwhm_duration_laser_pulse_s
pulseWidth_x = (fwhm_duration_laser_pulse/2.355)*consts.light_velocity # [cm]

waist = waist_s*Lr
init_laser_pos = xmin+distance_laser_peak_window_border_s*Lr - c*8*dt_s*Tr 
focusPosition = begin_downramp - init_laser_pos


@cfunc(types.field_loop_callback)
def initiate_field_callback(ind, r, E, B, data_double, data_int):

    if data_int[0] == 0:       
        x = r[0] - init_laser_pos
        
        k = 2*np.pi/wavelength
        # Rayleigh length
             
      
        amp = fieldAmplitude

        gp = np.real(amp*np.exp(-1j*(k*x))*np.exp(-x**2/(2*pulseWidth_x**2)))

        # y-polarized 
        E[1] = -gp       
        B[2] = -gp        


#=================================OUTPUT========================================
#-------------------------preparing output of fields (x)-----------------------------
Ex = np.zeros((nx,), dtype=np.double) 
Ey = np.zeros((nx,), dtype=np.double) 
Ez = np.zeros((nx,), dtype=np.double) 

rho = np.zeros((nx,), dtype=np.double) 
# not done
pmin = 0
pmax = 1e-15
nps = 2**8
ps = np.zeros((nps,nx), dtype=np.double) 
weights = np.zeros((nx,), dtype=np.double) 
#------------------get functions-----------------------------------------------
    
@cfunc(types.particle_loop_callback)
def get_density(r, p, w, id, data_double, data_int):   
    ix = int(nx*(r[0] - xmin)/(xmax - xmin))
    data = carray(data_double, rho.shape, dtype=np.double)
    
    if ix < rho.shape[0]:
        data[ix] += w[0]/(dx)

@cfunc(types.particle_loop_callback)
def get_phase_space(r, p, w, id, data_double, data_int):   
    ix = int(ps.shape[1]*(r[0] - xmin)/(xmax - xmin))
    ip = int(ps.shape[0]*(p[0] - pmin)/(pmax - pmin))
    data = carray(data_double, ps.shape, dtype=np.double)

    if ip>=0 and ip < ps.shape[0] and ix < ps.shape[1]:
        data[ip,ix] += w[0]/(dx) 


@cfunc(types.field_loop_callback)
def get_field_Ey(ind, r, E, B, data_double, data_int):
    _E = carray(data_double, Ey.shape, dtype=np.double)
    _E[ind[0]] = E[1]

@cfunc(types.field_loop_callback)
def get_field_Ex(ind, r, E, B, data_double, data_int):   
    _E = carray(data_double, Ex.shape, dtype=np.double)
    _E[ind[0]] = E[0]

@cfunc(types.field_loop_callback)
def get_field_Ez(ind, r, E, B, data_double, data_int):
    _E = carray(data_double, Ez.shape, dtype=np.double)
    _E[ind[0]] = E[2]


def load_fields():
    sim.field_loop(handler=get_field_Ey.address, data_double=pipic.addressof(Ey))
    sim.field_loop(handler=get_field_Ex.address, data_double=pipic.addressof(Ex))
    sim.field_loop(handler=get_field_Ez.address, data_double=pipic.addressof(Ez))

#===============================SIMULATION======================================

data_int = np.zeros((1, ), dtype=np.intc) # data for passing the iteration number
#-----------------------adding the handler of extension-------------------------

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
sim.add_particles(name='electron', number=int(particles),
                charge=consts.electron_charge, mass=consts.electron_mass,
                temperature=temperature, density=density_profile.address,
                data_int=pipic.addressof(data_int))


sim.field_loop(handler=initiate_field_callback.address, data_int=pipic.addressof(data_int),
                use_omp=True)

#-----------------------run simulation-------------------------


dsets = ['Ex','Ey','Ez','rho','ps']
fields = [Ex,Ey,Ez,rho,ps]
ncp = s//checkpoint

hdf5_fp = 'lwfa_neg.h5'
create_hdf5(hdf5_fp, shape=(ncp,nx),dsets=dsets[:-1])
create_hdf5(hdf5_fp, shape=(ncp,nps,nx),dsets=['ps'],mode="r+")

create_hdf5(hdf5_fp,shape=(nx,),dsets=['x_axis',],mode="r+")
create_hdf5(hdf5_fp,shape=(nps,),dsets=['p_axis',],mode="r+")

x_axis = np.linspace(xmin, xmax, nx)
p_axis = np.linspace(pmin, pmax, nps)

with h5py.File(hdf5_fp,"r+") as file:
    file['x_axis'][:] = x_axis
    file['p_axis'][:] = p_axis


for i in range(s):
    
    data_int[0] = i 

    sim.advance(time_step=timestep, number_of_iterations=1,use_omp=True)
    
    
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
        roll_back =  0 
        try:
            with h5py.File(hdf5_fp,"r+") as file:
                for j,f in enumerate(dsets):
                    file[f][i//checkpoint] = np.roll(fields[j],-roll_back,axis=-1) 
        except IOError:
            input('Another process are accessing the hdf5 file. Close the file and press enter.')
            with h5py.File(hdf5_fp,"r+") as file:
                for j,f in enumerate(dsets):
                    file[f][i//checkpoint] = np.roll(fields[j],-roll_back,axis=-1)    


