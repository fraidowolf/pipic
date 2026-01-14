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

dx = 2*np.pi/64           # longitudinal resolution 
dt = dx/4         # timestep
Lx = 16*32*32*dx #256*dx


n0 = 4e18
n0_sim = n0/Nr
a0 = 4.
omega_l = 1*omega_r # 2*pi/s
fwhm_duration_laser_pulse = 15*2*np.pi
waist = 20*2*np.pi #4*fwhm_duration_laser_pulse/2.335


Main(
    geometry = "1Dcartesian",
    
    interpolation_order = 2 ,
    
    cell_length = [dx,],
    grid_length  =[Lx,],
    
    number_of_patches = [ 4,],
    
    timestep = dt,
    simulation_time = 80000*dt,
     
    EM_boundary_conditions = [
        ['silver-muller'],
    ],
    
    solve_poisson = False,
    #solve_relativistic_poisson = True,
    
    print_every = 100,
    #cluster_width=2,
    maxwell_solver = 'M4',

)


begin_upramp = Lx/4
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


distance_laser_peak_window_border = 1.7*fwhm_duration_laser_pulse
longp = polygonal(xpoints=[begin_upramp, 
                           end_upramp, 
                           begin_downramp, 
                           end_downramp, 
                           endplateau, 
                           finish], 
                  xvalues=[0, n0_sim, n0_sim, n0_sim/3, n0_sim/3, 0.])

pulse_duration = fwhm_duration_laser_pulse*2*3/2.335  #+ 0.03973240173513659*2 # +- 3sigma 


pulse_duration = fwhm_duration_laser_pulse*2*3/2.335 # +- 3sigma 
init_laser_pos = - pulse_duration/2
travel_time = distance_laser_peak_window_border - init_laser_pos 
travel_time = np.ceil(travel_time/(dt*20))*dt*20
init_laser_pos = distance_laser_peak_window_border - travel_time
pulse_duration = -2*init_laser_pos # for timing reasons


LaserPlanar1D(
    box_side         = "xmin",
    omega            = 1,
    a0               = a0,
    time_envelope    = tgaussian(#center=(Lx-distance_laser_peak_window_border), 
                                 #center = pulse_duration/2,
                                 fwhm = fwhm_duration_laser_pulse,
                                 duration = pulse_duration),
)



Species(
    name = 'eon',
    position_initialization = 'random',
    momentum_initialization = 'cold',
    temperature=[0],
    particles_per_cell = 8,
    mass = 1.0,
    charge = -1.0,
    number_density = longp,
    boundary_conditions = [
        ["remove"],
    ],
    pusher = "boris",
)


LoadBalancing(
    every = 100,
    cell_load = 1.,
    frozen_particle_load = 0.1
)


print(dt*8*2)

DiagFields(
    every = 20,
    fields = ['Ex','Ey','Ez','Bx','By','Bz','Rho_eon','Rho'],#'Env_E_abs'
)

DiagParticleBinning(
    deposited_quantity = "weight",
    every = 100,
    species = ["eon"],
    axes = [
        ["moving_x", 0, Lx, 300],
        ["ekin", 0, 10, 100]
    ],
)
