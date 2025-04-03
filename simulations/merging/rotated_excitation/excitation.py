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


# constants
c = 299792458*1e2 #cm/s
m = 9.1093837*1e-31*1e3 #g
e = 4.80320425*1e-10  # statCoulomb

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
dz_s = 2*np.pi/8         # longitudinal resolution
dy_s = 2*np.pi
dx_s = dy_s              # transverse resolution
dt_s = dz_s/2            # timestep
nz,ny,nx = 2**10,2**7,2**7
Lz = nz*dz_s 
Ly = ny*dy_s
Lx = nx*dx_s

n0 = 1e18
n1 = n0/3

n0_s = n0/Nr
a0 = 4.
omega_s = 1. 
fwhm_duration_laser_pulse_s = 15*2*np.pi
waist_s = 20*2*np.pi 
distance_laser_peak_window_border_s = 1.7*fwhm_duration_laser_pulse_s

#===========================SIMULATION INITIALIZATION===========================
xmin, xmax = -Lx*Lr/2, Lx*Lr/2
ymin, ymax = -Ly*Lr/2, Ly*Lr/2
zmin, zmax = -Lz*Lr/2, Lz*Lr/2
dx, dy, dz = (xmax - xmin)/nx, (ymax - ymin)/ny, (zmax - zmin)/nz
timestep = dt_s*Tr
angle_of_rotation = np.pi/3

thickness = 10 # thickness (in dx) of the area where the density and field is restored/removed 

begin_upramp = 0
end_upramp = Lz*Lr/10 + begin_upramp
begin_downramp = end_upramp + 4*Lz*Lr
end_downramp = begin_downramp
endplateau = end_downramp
finish = endplateau

s = 5*int(end_upramp/c/timestep)
checkpoint = int(20)  
print(s,checkpoint)

#---------------------setting solver and simulation region----------------------
sim=pipic.init(solver='ec',nx=nx,ny=ny,nz=nz,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,zmin=zmin,zmax=zmax)
sim.en_corr_type(2)
#--------------------------- setting plasma profile --------------------------


temperature = 0 #4 * np.pi * density * (consts.electron_charge ** 2) * debye_length ** 2
particles_per_cell = 1 # 8 in smilei

@cfunc(types.add_particles_callback)
def density_profile(r, data_double, data_int):
    # r is the position in the 'lab frame'  
    R = r[0]*np.sin(angle_of_rotation) + r[2]*np.cos(angle_of_rotation) #+ thickness*dz

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
fieldAmplitude = a0*m*c*omega_l/e # [statV/cm]
wavelength = 2*np.pi*consts.light_velocity/omega_l # [cm]

fwhm_duration_laser_pulse = Tr*fwhm_duration_laser_pulse_s
pulseWidth_x = (fwhm_duration_laser_pulse/2.355)*consts.light_velocity # [cm]

waist = waist_s*Lr
init_laser_pos = 0 #zmax-distance_laser_peak_window_border_s*Lr
focusPosition = begin_downramp - init_laser_pos


@cfunc(types.field_loop_callback)
def initiate_field_callback(ind, r, E, B, data_double, data_int):

    if data_int[0] == 0:  

        r0_rot = r[0]*np.cos(angle_of_rotation) + r[2]*np.sin(angle_of_rotation)
        r1_rot = r[1]
        r2_rot = - r[0]*np.sin(angle_of_rotation) + r[2]*np.cos(angle_of_rotation)

        x = r2_rot #- init_laser_pos
        rho2 = r1_rot**2 + r0_rot**2
        
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
        E[0] = gp*np.cos(angle_of_rotation) #- gp*np.sin(angle_of_rotation)
        E[2] = gp*np.sin(angle_of_rotation) #+ gp*np.cos(angle_of_rotation)       
        B[1] = gp        

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
Bx = np.zeros((nx,ny,nz), dtype=np.double)
By = np.zeros((nx,ny,nz), dtype=np.double)
Bz = np.zeros((nx,ny,nz), dtype=np.double)
rho = np.zeros((nx,ny,nz), dtype=np.double)

#------------------get functions-----------------------------------------------
@cfunc(types.particle_loop_callback)
def get_density(r, p, w, id, data_double, data_int):   
    ix = int(nx*(r[0] - xmin)/(xmax - xmin))
    iy = int(ny*(r[1] - ymin)/(ymax - ymin))
    iz = int(nz*(r[2] - zmin)/(zmax - zmin))
    data = carray(data_double, rho.shape, dtype=np.double)
    #if iy == ny//2:
    data[ix,iy,iz] += w[0]/(dx*dy*dz)


@cfunc(types.particle_loop_callback)
def count_particles(r, p, w, id, data_double, data_int):
    data = carray(data_double, 1, dtype=np.double)
    data[0] += 1

@cfunc(types.field_loop_callback)
def get_field_Ex(ind, r, E, B, data_double, data_int):
    _E = carray(data_double, Ex.shape, dtype=np.double)
    _E[ind[0], ind[1], ind[2]] = E[0]

@cfunc(types.field_loop_callback)
def get_field_Ey(ind, r, E, B, data_double, data_int):
    _E = carray(data_double, Ey.shape, dtype=np.double)
    _E[ind[0], ind[1], ind[2]] = E[1]

@cfunc(types.field_loop_callback)
def get_field_Ez(ind, r, E, B, data_double, data_int):
    _E = carray(data_double, Ez.shape, dtype=np.double)
    _E[ind[0], ind[1], ind[2]] = E[2]

@cfunc(types.field_loop_callback)
def get_field_Bx(ind, r, E, B, data_double, data_int):
    _B = carray(data_double, Bx.shape, dtype=np.double)
    _B[ind[0], ind[1], ind[2]] = B[0]

@cfunc(types.field_loop_callback)
def get_field_By(ind, r, E, B, data_double, data_int):
    _B = carray(data_double, By.shape, dtype=np.double)
    _B[ind[0], ind[1], ind[2]] = B[1]

@cfunc(types.field_loop_callback)
def get_field_Bz(ind, r, E, B, data_double, data_int):
    _B = carray(data_double, Bz.shape, dtype=np.double)
    _B[ind[0], ind[1], ind[2]] = B[2]


def load_fields():
    sim.field_loop(handler=get_field_Ey.address, data_double=pipic.addressof(Ey),use_omp=True)
    sim.field_loop(handler=get_field_Ex.address, data_double=pipic.addressof(Ex),use_omp=True)
    sim.field_loop(handler=get_field_Ez.address, data_double=pipic.addressof(Ez),use_omp=True)
    sim.field_loop(handler=get_field_Bx.address, data_double=pipic.addressof(Bx),use_omp=True)
    sim.field_loop(handler=get_field_By.address, data_double=pipic.addressof(By),use_omp=True)
    sim.field_loop(handler=get_field_Bz.address, data_double=pipic.addressof(Bz),use_omp=True)
    sim.particle_loop(name='electron', handler=get_density.address, data_double=pipic.addressof(rho))

#===============================SIMULATION======================================

data_int = np.zeros((1, ), dtype=np.intc) # data for passing the iteration number
window_speed = consts.light_velocity*np.cos(angle_of_rotation) #speed of moving window

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

density_handler_adress = moving_window.handler(sim.ensemble_data(),
                                               thickness=thickness,
                                               particles_per_cell=particles_per_cell,
                                               temperature=temperature,
                                               density = density_profile.address,
                                               velocity=window_speed,)
sim.add_handler(name=moving_window.name, 
                subject='electron,cells',
                handler=density_handler_adress,
                data_int=pipic.addressof(data_int),)

sim.field_loop(handler=initiate_field_callback.address, data_int=pipic.addressof(data_int),
                use_omp=True)

#-----------------------run simulation-------------------------


dsets = ['Ex','Ey','Ez','Bx','By','Bz','rho']
fields = [Ex,Ey,Ez,Bx,By,Bz,rho]
ncp = s//checkpoint +1 


hdf5_fp = 'lwfa.h5'
create_hdf5(hdf5_fp, shape=(ncp,nx,ny,nz),dsets=dsets[:])
create_hdf5(hdf5_fp,shape=(nx,),dsets=['x_axis',],mode="r+")
create_hdf5(hdf5_fp,shape=(ny,),dsets=['y_axis',],mode="r+")
create_hdf5(hdf5_fp,shape=(nz,),dsets=['z_axis',],mode="r+")

x_axis = np.linspace(xmin, xmax, nx)
y_axis = np.linspace(ymin, ymax, ny)
z_axis = np.linspace(zmin, zmax, nz)

with h5py.File(hdf5_fp,"r+") as file:
    file['x_axis'][:] = x_axis
    file['y_axis'][:] = y_axis
    file['z_axis'][:] = z_axis

print('start simulation')
for i in range(s):
    data_int[0] = i 
    sim.advance(time_step=timestep, number_of_iterations=1,use_omp=True)
    print('iteration',i)
    sim.field_loop(handler=remove_field.address, data_int=pipic.addressof(data_int),
                use_omp=True)
    
    if i%checkpoint==0:
        with open('./out.txt', 'a') as f:
            print(f'{i}\n', file=f)
        print(i)
        rho.fill(0)
        load_fields()
        roll_back =  0 #int(np.floor(i*timestep*window_speed/dz))  

        try:
            with h5py.File(hdf5_fp,"r+") as file:
                for j,f in enumerate(dsets):
                    file[f][i//checkpoint] = np.roll(fields[j],-roll_back,axis=-1) 
        except IOError:
            input('Another process are accessing the hdf5 file. Close the file and press enter.')
            with h5py.File(hdf5_fp,"r+") as file:
                for j,f in enumerate(dsets):
                    file[f][i//checkpoint] = np.roll(fields[j],-roll_back,axis=-1)    

travelled_distance = (roll_back*dz)%Lz
print(travelled_distance)
number_of_particles = np.zeros(1, dtype=np.double)
sim.particle_loop(name='electron',handler=count_particles.address, data_double=pipic.addressof(number_of_particles))
print(number_of_particles)
ps = np.zeros((int(number_of_particles[0]),7), dtype=np.double)

ind = np.zeros(1, dtype=np.intc)
@cfunc(types.particle_loop_callback)
def get_phase_space(r, p, w, id, data_double, data_int):   
    data = carray(data_double, ps.shape, dtype=np.double)
    ind = data_int[0]
    data[ind,0] = r[0]
    data[ind,1] = r[1]
    data[ind,2] = r[2] - travelled_distance
    data[ind,3] = p[0]
    data[ind,4] = p[1]
    data[ind,5] = p[2]
    data[ind,6] = w[0] 
    data_int[0] += 1


sim.particle_loop(name='electron',handler=get_phase_space.address, data_double=pipic.addressof(ps),data_int=pipic.addressof(ind))
print(ps,ind,number_of_particles)
with h5py.File(hdf5_fp,"r+") as file:
    file.create_dataset('phase_space', data=ps)




