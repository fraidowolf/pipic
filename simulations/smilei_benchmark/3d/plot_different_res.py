import happi
import numpy as np
import matplotlib.pyplot as plt
import scipy
import h5py
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib

# add folder to path
import sys
sys.path.append('../../../../plot_style/')
import style


style.load_preset(scale=1)

def load_data(fp,fields,axes=[],ind=-1):
    with h5py.File(fp,"r") as file: 
        fields_out = []
        axes_out = []
        for field in fields:
            fields_out.append(file[field][ind])
        for a in axes:
            axes_out.append(file[a][:])
    return fields_out,axes_out




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

fwhm_duration_laser_pulse = 15*2*np.pi
pulse_duration = fwhm_duration_laser_pulse*2*3/2.335
distance_laser_peak_window_border = 2.5*fwhm_duration_laser_pulse
distance_laser_peak_window_border_pipic = 1.7*fwhm_duration_laser_pulse
init_laser_pos = - pulse_duration/2



fps_pipic = ['res32/lwfa.h5','./res16/lwfa_full_p.h5','./res8/lwfa.h5','./res4/lwfa.h5']
#fps_pipic_boris = ['res32_boris/lwfa.h5','./res16_boris/lwfa_full_p.h5','./res8_boris/lwfa.h5','./res4_boris/lwfa.h5']

smilei_path = './smilei/'
fps_smilei = [smilei_path+'res32_M4_solver/',smilei_path + 'res16_M4_solver/',smilei_path + 'res8_M4_solver/',smilei_path + 'res4_M4_solver/']


cmap = matplotlib.cm.get_cmap('coolwarm')
colors_pp = [cmap(i) for i in np.linspace(0.,1,4)] #['tab:blue','tab:orange','tab:green','tab:red']
linestyles = ['-','-.','--',':']

nx_ = [4*256 + 4*32*32,2*1000+2*32*32,1000+32*32,1000+32*32]

_ny = [2**6,2**6,2**6,2**6]
add = [0,0,0,0]#[0,1,15]
timesteps = [200]


figw = style.figsize['inch']['double_column_width']*0.5


fig,ax = plt.subplots(2,1,figsize=(figw,figw/1.), gridspec_kw = {'wspace':-0.8, 'hspace':-0.8})

for i,fp in enumerate(zip(fps_pipic,fps_smilei)):
    fpp,fps = fp

    for ts in timesteps:
        ip = int(ts/(2**i))
        print(ip)

        dx = 2*np.pi/(32/2**i)           # longitudinal resolution
        dt = dx/4         # timestep
        dy = 2*np.pi*4  
        nx = 2**(12-i)
        ny = _ny[i]
        Lx_ = nx_[i]*dx
        Lx = nx*dx
        Ly = _ny[i]*dy

        travel_time = (Lx_-distance_laser_peak_window_border) - init_laser_pos 
        travel_time = np.ceil(travel_time/(dt*100))*dt*100
        init_laser_pos_eff = (Lx_-distance_laser_peak_window_border) - travel_time
        pulse_duration_eff = -2*init_laser_pos_eff
        init_laser_pos_pipic_eff = Lx_ - distance_laser_peak_window_border_pipic
        propagation_time = pulse_duration_eff/2 + init_laser_pos_pipic_eff 

        speed_of_light_smilei = c#*(1-0.005*2**i*dt_diff[i]) #29415635978.96
        # the lag of smilei caused by light moving slower than c
        lag = (1-speed_of_light_smilei/c)*(propagation_time + ip*100*dt)  
        ism = round((propagation_time+lag)/dt/100 + ip) 
    
        diff = ism*100 - (propagation_time+lag)/dt - ip*100
        print(diff*dt)
        if not fps == None:
            S = happi.Open(fps)
            Ex_smilei = S.Field.Field0("Ex",timestep_indices=ism).getData()[0]
        
            f = S.Field(0,'Ex',moving=True)
            x_moved = f.getXmoved(f.getTimesteps()[ism])
            # there are diffrent positions of the upramp in the two simulations
            upramp_pos_smilei = Lx_ - Lx/2
            upramp_pos_pipic = Lx/2
            x = f.getAxis('x')*Lr  + Lr*x_moved - Lr*Lx/2 - Lr*(upramp_pos_smilei - upramp_pos_pipic) - diff*dt*c*Tr
            y = np.linspace(-Lr*Ly/2,Lr*Ly/2,32)
            ax[1].plot(x,Ex_smilei[:,ny//2]*Er,linestyles[i],color=colors_pp[i],lw=1,label=r'Smilei, '+r'$\Delta x=\lambda/$'+str(32//2**i))
        f,a = load_data(fpp,fields=['Ez'],axes=['z_axis','y_axis'],ind=ip)
        Ex_pipic = f[0]
        #f,a = load_data(fps_pipic_boris[i],fields=['Ez'],axes=['z_axis','y_axis'],ind=ip)
        Ex_pipic_boris = f[0]
        z_axis,y_axis = a
        z_axis += ip*100*dt*c*Tr 

        print(fpp)
        ax[0].plot(z_axis,Ex_pipic[ny//2,:],linestyles[i],color=colors_pp[i],lw=1,label=r'$\Delta x=\lambda/$'+str(32//2**i))
        #ax[0].plot(z_axis,Ex_pipic_boris[ny//2,:],linestyles[i],color='k',lw=1,label=r'$\Delta x=\lambda/$'+str(32//2**i))


for i in range(2):
    ax[i].set_xlim(0.0080,0.0170)
    ax[i].set_ylim(-7.5e6,7.5e6)
    ax[i].set_yticks([-5e6,0,5e6])
    ax[i].set_xticks([0.010,0.012,0.014,0.016])
    ax[i].xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: (r'%g') % (x*1e4)))
    ax[i].set_ylabel(r'$E_z$ [$10^{6}$statV/cm]')
    ax[i].yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: (r'%g') % (x*1e-6)))
    ax[i].tick_params(which='both',direction='in')


ax[0].legend(frameon=False,ncols=2,bbox_to_anchor=(0.5, 1.35), loc='upper center')
ax[1].set_xlabel(r'$z$ [$\mu$m]')
ax[1].tick_params(axis='x',top=True)
ax[0].tick_params(axis='x',labelbottom=False)

ax[0].text(0.03, 0.85, r'$\pi$-PIC',transform=ax[0].transAxes)
ax[1].text(0.03, 0.85, r'Smilei',transform=ax[1].transAxes)

plt.savefig('res_compare.png')