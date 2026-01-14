# ----------------------------------------------------------------------------------------
#                     SIMULATION PARAMETERS FOR THE PIC-CODE SMILEI
# ----------------------------------------------------------------------------------------

import cmath, math
import scipy.constants
from numpy import exp, sqrt, arctan, vectorize, real
from math import log
import numpy as np


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

dx = 2*np.pi/32           # longitudinal resolution
dy = 2*np.pi*4
dz = dy              # transverse resolution
dt = dx/4         # timestep
Lx_ = 4*256*dx + 4*32*32*dx #256*dx
Lx = 4*32*32*dx #256*dx
Ly = 64*dy
Lz = 64*dz

n0 = 4e18
n0_sim = n0/Nr
a0 = 4.
omega_l = 1*omega_r # 2*pi/s
fwhm_duration_laser_pulse = 15*2*np.pi
waist = 20*2*np.pi #4*fwhm_duration_laser_pulse/2.335


Main(
    geometry = "3Dcartesian",
    
    interpolation_order = 2 ,
    
    cell_length = [dx,dy,dz],
    grid_length  =[Lx_,Ly,Lz],
        
    number_of_patches = [ 256, 4, 4 ],    
    timestep = dt,
    simulation_time = 4*40000*dt,
     
    EM_boundary_conditions = [
        ['silver-muller'],
    ],
    
    solve_relativistic_poisson = False,
    maxwell_solver = "M4",
    print_every = 100,
    #cluster_width=2,

)


begin_upramp = Lx_
#Lupramp  = 10*dx
end_upramp = Lx*4 + begin_upramp
begin_downramp = end_upramp # Beginning of the output ramp.
end_downramp = begin_downramp
#Ldownramp = 10*dx
#xplateau = begin_upramp + Lupramp # Start of the plateau

#end_downramp = xplateau + Lplateau # Beginning of the output ramp.
endplateau = end_downramp + Lx*4 # End of plasma
finish = endplateau # End of plasma


focus_x_laser = end_upramp


distance_laser_peak_window_border = 2.5*fwhm_duration_laser_pulse
longp = polygonal(xpoints=[begin_upramp, 
                           end_upramp, 
                           begin_downramp, 
                           end_downramp, 
                           endplateau, 
                           finish], 
                  xvalues=[0, n0_sim, n0_sim, n0_sim/3, n0_sim/3, 0.])

pulse_duration = fwhm_duration_laser_pulse*2*3/2.335 # +- 3sigma 
init_laser_pos = - pulse_duration/2
travel_time = (Lx_-distance_laser_peak_window_border) - init_laser_pos 
travel_time = np.ceil(travel_time/(dt*100))*dt*100
init_laser_pos = (Lx_-distance_laser_peak_window_border) - travel_time
pulse_duration = -2*init_laser_pos

LaserGaussian3D(
    box_side         = "xmin",
    omega            = 1,
    a0               = a0,
    focus            = [focus_x_laser, Ly/2., Lz/2.],
    waist            = waist, # from um to normalized units
    time_envelope    = tgaussian(#center=(Lx-distance_laser_peak_window_border), 
                                 #center = pulse_duration/2,
                                 fwhm = fwhm_duration_laser_pulse,
                                 duration = pulse_duration),
)


MovingWindow(
    time_start = travel_time, 
    velocity_x = 0.99, #165, #np.sqrt(1.-n0),
)


Species(
    name = 'eon',
    position_initialization = 'random',
    momentum_initialization = 'cold',
    particles_per_cell = 8,
    mass = 1.0,
    charge = -1.0,
    number_density = longp,
    boundary_conditions = [
        ["remove", "remove"],
        ["remove", "remove"],
        ["remove", "remove"],
    ],
    pusher = "boris",
)


LoadBalancing(
    every = 100,
    cell_load = 1.,
    frozen_particle_load = 0.1
)



DiagFields(
    every = 100,
    subgrid = np.s_[:, (Ly/dy)//2, :],
    fields = ['Ex','Ey','Ez','Bx','By','Bz','Rho_eon','Rho'],#'Env_E_abs'
)

DiagParticleBinning(
    deposited_quantity = "weight",
    every = 250,
    species = ["eon"],
    axes = [
        ["moving_x", 0, Lx_, 300],
        ["ekin", 0, 10, 100]
    ],
)

