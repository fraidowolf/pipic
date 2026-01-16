import happi
import numpy as np
import matplotlib.pyplot as plt
import scipy
import h5py
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

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

dx = 2*np.pi/16           # longitudinal resolution
dy = 2*np.pi*4
dz = dy              # transverse resolution
dt = dx/4         # timestep
Lx_ = 2*1000*dx + 2*32*32*dx #256*dx
Lx = 2*32*32*dx #256*dx
Ly = 64*dy
Lz = 64*dz

# calculation of the extra time for smilei for propagation of 
# the pulse into the simulation box and to the initial position
# of the pipic pulse
fwhm_duration_laser_pulse = 15*2*np.pi
pulse_duration = fwhm_duration_laser_pulse*2*3/2.335
distance_laser_peak_window_border = 2.5*fwhm_duration_laser_pulse
init_laser_pos = - pulse_duration/2
travel_time = (Lx_-distance_laser_peak_window_border) - init_laser_pos 
travel_time = np.ceil(travel_time/(dt*100))*dt*100
init_laser_pos = (Lx_-distance_laser_peak_window_border) - travel_time
pulse_duration = -2*init_laser_pos
distance_laser_peak_window_border_pipic = 1.7*fwhm_duration_laser_pulse
init_laser_pos_pipic = Lx_ - distance_laser_peak_window_border_pipic
propagation_time = pulse_duration/2 + init_laser_pos_pipic 


begin_upramp = Lr*Lx/2
end_upramp = Lx*Lr*4 + begin_upramp
begin_downramp = end_upramp
end_downramp = begin_downramp
endplateau = end_downramp + 4*Lx*Lr
finish = endplateau

n0 = 4e18
n1 = n0/3

def plasma_profile(R):
    # R is the position in the 'lab frame'  
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

density = np.vectorize(plasma_profile)
x = np.linspace(0,finish,1000)

figw = style.figsize['inch']['double_column_width']
fig = plt.figure(figsize=(figw,figw*0.8))
outer = gridspec.GridSpec(2, 3, height_ratios = [1, 1], wspace = 0.3) 
#make nested gridspecs
gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec = outer[0,:], wspace = .05)
gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[1,:2])
gs3 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = outer[1, 2], hspace = 0)

#gs = plt.GridSpec(3,3)

ax = [plt.subplot(gs1[0]),
      plt.subplot(gs1[1]),
      plt.subplot(gs1[2])]

ax_profile = plt.subplot(gs2[0])
ax_ekin = [plt.subplot(gs3[0]),
           plt.subplot(gs3[1])]
ax_ekin[0].sharex(ax_ekin[1])
#fig,ax = plt.subplots(2,3)

ax_profile.plot(x,density(x))

S = happi.Open('smilei/res16_M4_solver/')
path = './res16/lwfa_full_p.h5'
path_fb = './res16_boris/lwfa_full_p.h5'
#path = path_fb

fields = ['Ez','Ex','rho','ps']
axes = ['z_axis','y_axis','p_axis'] 

ii = [100,300,500]
rhovmin = 0
rhovmax = 10

for ai,i in enumerate(ii):

    f,a = load_data(path,fields=fields,axes=axes,ind=i)
    ffb,afb = load_data(path_fb,fields=fields,axes=axes,ind=i)
    Ez,Ex,rho,ps = f
    z_axis,y_axis,p_axis = a
    z_axis += i*100*dt*c*Tr 

    im = ax[ai].pcolormesh(z_axis,y_axis[32:],rho[32:,:]/1e18,vmin=rhovmin,vmax=rhovmax,shading='nearest',cmap='RdYlBu_r')

    ax_twinx = ax[ai].twinx()
    ax_twinx.plot(z_axis,Ez[32,:],'-',lw=0.8,color='black',label=r'$\pi$-PIC')
    Ezfb,_,_,_ = ffb
    ax_twinx.plot(z_axis,Ezfb[32,:],'-',lw=0.8,linestyle=(0,(2,3)),color='tab:orange',label=r'$\pi$-PIC')


    speed_of_light_smilei = c #29415635978.96
    # the lag of smilei caused by light moving slower than c
    lag = (1-speed_of_light_smilei/c)*(propagation_time + i*100*dt)  
    i = int((propagation_time+lag)//dt//100) + i + 1 

    Rho = S.Field.Field0("Rho_eon",timestep_indices=i).getData()[0]
    Ey = S.Field.Field0("Ey",timestep_indices=i).getData()[0]
    Ex = S.Field.Field0("Ex",timestep_indices=i).getData()[0]
    Ez = S.Field.Field0("Ez",timestep_indices=i).getData()[0]
    ekin = S.ParticleBinning(diagNumber=0,timestep_indices=i).getData()[0] 

    # calculate the current moving x axis
    f = S.Field(0,'Ex',moving=True)
    x_moved = f.getXmoved(f.getTimesteps()[i])
    # there are diffrent positions of the upramp in the two simulations
    upramp_pos_smilei = Lx_ - Lx/2
    upramp_pos_pipic = Lx/2
    x = f.getAxis('x')*Lr  + Lr*x_moved - Lr*Lx/2 - Lr*(upramp_pos_smilei - upramp_pos_pipic)

    im = ax[ai].pcolormesh(x,y_axis[:32],-Nr*Rho[:,:32].T/1e18,vmin=rhovmin,vmax=rhovmax,shading='nearest',cmap='RdYlBu_r')
    ax_twinx.plot(x,Er*Ex[:,32],'--',lw=0.8,color='tab:green')
    ax_twinx.set_ylim(-3e7,3e7)

    ax[ai].set_ylim(-0.007,0.007)
    ax[ai].set_xlim(z_axis[0],z_axis[-1])
    #ax[ai].set_xlim(0.015,0.016)
    ax_profile.vlines(z_axis[-1],0,4.2e18,color='black',lw=0.8)
    ax[ai].hlines(0,x[0],x[-1],color='black',linestyle='-',lw=2.)
    ax[ai].hlines(0,x[0],x[-1],color='white',linestyle='-',lw=1.)
    ax[ai].text(0.85,0.9,r'('+chr(ord('`')+ai+1)+')',transform=ax[ai].transAxes,color='black')
    ax_profile.text(z_axis[-1]-0.007,4e18,r'('+chr(ord('`')+ai+1)+')')

    ax[ai].tick_params(axis='both',direction='in',
                        right=False,left=True,
                        labelright=False,labelleft=False)
    ax_twinx.tick_params(axis='y',direction='in',
                        right=True,left=False,
                        labelright=False,labelleft=False)
    

    
    if ai == 0: 
        ax[ai].set_ylabel(r'$y$ [$\mu$m]')
        ax[ai].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: ('%g') % (x * 1e4)))   
        ax[ai].tick_params(axis='y',labelleft=True)
        lines = ax_twinx.get_lines()
        ax[ai].text(0.05,0.05,r'Smilei - M4 solver',transform=ax[ai].transAxes,color='black')
        ax[ai].text(0.05,0.9,r'$\pi$-PIC - EC solver',transform=ax[ai].transAxes,color='black')

    if ai == 2:
        ax_twinx.set_ylabel(r'$E_z$ [StatV/cm]')
        ax_twinx.tick_params(axis='y',labelright=True)

        # make colorbar
        cbar = plt.colorbar(im,ax=ax[:],label=r'$n_e$ [$10^{18}$cm$^{-3}$]', 
                            orientation='horizontal',
                            location='top',
                            pad=0.00,
                            shrink=0.325,
                            anchor=(0.,0.5),)

# custom legend outside plot
plt.legend(lines,
           ['$\pi$-PIC - EC solver', '$\pi$PIC - FB solver','Smilei - M4 solver'],
           frameon=False,
           bbox_to_anchor=(-1., 1.4), 
           ncols=1,
           loc='upper left')


    #ax[0,ai].set_xlabel(r'$z$')


evmin = 14
evmax = 16

print(ii,i)

# note i is the timestep in the smilei simulation
ekin = S.ParticleBinning(diagNumber=0,timestep_indices=i).getData()[0]*Nr # cm^{-3}
ekin_axis_smilei = S.ParticleBinning(diagNumber=0,timestep_indices=i).getAxis('ekin')*Kr # MeV
x = S.ParticleBinning(diagNumber=0,timestep_indices=i).getAxis('moving_x')*Lr + Lr*x_moved - Lr*Lx/2 - Lr*(upramp_pos_smilei - upramp_pos_pipic)    
dx_ = x[1]-x[0] # cm
de_ = ekin_axis_smilei[1]-ekin_axis_smilei[0] # MeV
print(np.diff(ekin_axis_smilei))
#ekin /= (de_)
#ekin /= Lr**2*Ly**2 # cm^{-1}
im = ax_ekin[1].pcolormesh(x,ekin_axis_smilei,np.log10(ekin.T),shading='nearest',cmap='Reds',vmin=evmin,vmax=evmax)
cbar = plt.colorbar(im,ax=ax_ekin[1],label=r'$n_e$ [cm$^{-3}$]',pad=0)
cbar.ax.set_yticks([15,16])
cbar.ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: (r'$10^{%g}$') % (x)))

nz,ny,nx = 2*32*32,64,64
dz_s = 2*np.pi/16           # longitudinal resolution
dy_s = 2*np.pi*4
dx_s = dy_s 

#ekin_axis_pipic = np.sqrt((p_axis*c)**2 + m**2*c**4) #- m*c**2
ekin_axis_pipic_gamma = np.sqrt(1+(p_axis/(m*c))**2)
ekin_axis_pipic = (ekin_axis_pipic_gamma-1)*m*c**2
ekin_axis_pipic *= 624.15*1e3 # MeV

#ps *=  dz_s*dy_s*dx_s*Lr**3 
de = np.diff(ekin_axis_pipic)[:,np.newaxis]
dx = z_axis[1]-z_axis[0]
ps /= ny**2
#ps[:-1] = ps[:-1]/de

ps_log = np.log10(ps)
#ps_log[ps_log<-1e18] = 0
print(np.diff(ekin_axis_pipic))
im = ax_ekin[0].pcolormesh(z_axis,ekin_axis_pipic,ps_log,cmap='Reds',shading='nearest',vmin=evmin,vmax=evmax)
cbar = plt.colorbar(im,ax=ax_ekin[0],label=r'$n_e$ [cm$^{-3}$]',pad=0.0)
cbar.ax.set_yticks([15,16])
cbar.ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: (r'$10^{%g}$') % (x)))
ax_ekin[0].set_xlim(z_axis[0],z_axis[-1])
ax_ekin[1].set_ylim(ekin_axis_pipic[0],30)
ax_ekin[0].set_ylim(ekin_axis_pipic[0],30)
for i in range(2): 
    ax_ekin[i].set_ylabel(r'$\varepsilon$ [MeV]')
    #ax_ekin[i].yaxis.set_label_position("right")

for i in range(2): ax_ekin[i].tick_params(axis='both',direction='in',labelleft=True,left=True)

#ax[0].text(0.05,0.9,r'$\pi$PIC',transform=ax[0].transAxes,color='white')
#ax[1].text(0.05,0.1,r'Smilei',transform=ax[0].transAxes,color='white')
ax_ekin[0].text(0.05,0.8,r'$\pi$-PIC - EC',transform=ax_ekin[0].transAxes,)
ax_ekin[1].text(0.05,0.8,r'Smilei - M4',transform=ax_ekin[1].transAxes)

a = [ax_profile,ax_ekin[1]]
a.extend(ax[:])
for aa in a: 
    aa.set_xlabel(r'$z$ [$\mu$m]')
    aa.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: ('%g') % (x * 1e4)))   
ax_profile.set_ylabel(r'$n_e$ [$10^{18}$cm$^{-3}$]')
ax_profile.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: ('%g') % (x * 1e-18)))

ax_profile.text(0.92,0.91,r'(d)',transform=ax_profile.transAxes)
ax_ekin[0].text(0.85,0.85,r'(e)',transform=ax_ekin[0].transAxes)
ax_ekin[1].text(0.85,0.85,r'(f)',transform=ax_ekin[1].transAxes)
ax_ekin[0].tick_params(axis='x',labelbottom=False)

#plt.tight_layout()
fig.patch.set_facecolor('red')
fig.patch.set_alpha(0.)
plt.savefig('im_.png',dpi=900,transparent=True)


