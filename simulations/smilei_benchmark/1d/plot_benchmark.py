import h5py
import numpy as np
import matplotlib.pyplot as plt
import happi
import matplotlib as mpl

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
            fields_out.append(file[field][:ind])
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

dx = 2*np.pi/64 # longitudinal resolution
Lx = 16*32*32*dx #256*dx
dt = dx/4         # timestep


figw = style.figsize['inch']['double_column_width']*0.6
fig, ax = plt.subplots(2,2,figsize=(figw,figw*0.8))
cmap = mpl.cm.get_cmap('coolwarm')
colors_pp = [cmap(i) for i in np.linspace(0.,1,4)] #['tab:blue','tab:orange','tab:green','tab:red']

fields = ['Ex','rho']
axes = ['x_axis']#,'t_axis']
x_sr =[0.0191e4,0.0193e4] #[0.0051e4,0.0056e4]#   

i = 450 + 100 
print('pipic',100*dt*Tr*i)
f,a = load_data('./0,5e10_dx0,25_corr_delayed/lwfa_neg.h5',fields=fields,axes=axes,ind=i+1)
Ex,rho = f
x_axis = a[0]
x_axis *= 1e4
x_axis += 1e4*Lx*Lr/2

ax[0,0].plot(x_axis,Ex[i]*1e-6,color='k',label=r'$\pi$PIC - Energy conserving solver')
ax[1,0].plot(x_axis,rho[i]*1e-18,color='k')

ix1 = np.argwhere(x_sr[0]<x_axis)[0][0]
ix2 = np.argwhere(x_sr[1]<x_axis)[0][0]


ax[0,1].plot(x_axis[ix1:ix2],Ex[i][ix1:ix2]*1e-6,color='k')
ax[1,1].plot(x_axis[ix1:ix2],rho[i][ix1:ix2]*1e-18,color='k')


#f,a = load_data('./boris/lwfa_neg_delayed.h5',fields=fields,axes=axes,ind=i+1)
f,a = load_data('./boris_with_div_cleaning/lwfa_neg_delayed_div_clean.h5',fields=fields,axes=axes,ind=i+1)
Ex,rho = f
x_axis = a[0]
x_axis *= 1e4
x_axis += 1e4*Lx*Lr/2

ax[0,0].plot(x_axis,Ex[i]*1e-6,linestyle=(0,(2,3)),color='tab:orange',label=r'$\pi$PIC - Fourier-Boris solver')
ax[1,0].plot(x_axis,rho[i]*1e-18,linestyle=(0,(2,3)),color='tab:orange')

ax[0,1].plot(x_axis[ix1:ix2],Ex[i][ix1:ix2]*1e-6,linestyle=(0,(2,3)),color='tab:orange')
ax[1,1].plot(x_axis[ix1:ix2],rho[i][ix1:ix2]*1e-18,linestyle=(0,(2,3)),color='tab:orange')


S = happi.Open('../../../../smilei/benchmark/1d/4res_M4_solver/')

fwhm_duration_laser_pulse = 15*2*np.pi
pulse_duration = fwhm_duration_laser_pulse*2*3/2.335 #+- 3sigma 
Lx = 16*32*32*dx #256*dx
distance_laser_peak_window_border = 1.7*fwhm_duration_laser_pulse
propagation_time = pulse_duration/2 + distance_laser_peak_window_border 

#s = 2783 

s = int((propagation_time/dt + i*100)/20) 

print('Propagation time',propagation_time*Tr)
print('smilei',dt*Tr*s*20 - propagation_time*Tr)
print('pipic',100*dt*Tr*i)
print('diff', 100*dt*i - (dt*s*20 - propagation_time))
print(dt*8*2)


Rho = S.Field.Field0("Rho",timestep_indices=s).getData()
Ey = S.Field.Field0("Ey",timestep_indices=s).getData()
Ex = S.Field.Field0("Ex",timestep_indices=s).getData()
Ez = S.Field.Field0("Ez",timestep_indices=s).getData()

x = np.arange(0,Lx*Lr+dx*Lr,dx*Lr) 
x -= 0 # - c*8*dt*Tr # compensation for diffrent starting points?
x *= 1e4
ax[0,0].plot(x,Er*Ex[0]*1e-6,'-.',color='tab:green',label='Smilei - M4 Solver')
ax[1,0].plot(x,-Nr*Rho[0]*1e-18,'-.',color='tab:green')

ix1 = np.argwhere(x_sr[0]<x)[0][0]
ix2 = np.argwhere(x_sr[1]<x)[0][0]

ax[0,1].plot(x[ix1:ix2],Er*Ex[0][ix1:ix2]*1e-6,'-.',color='tab:green')
ax[1,1].plot(x[ix1:ix2],-Nr*Rho[0][ix1:ix2]*1e-18,'-.',color='tab:green')
#ax[1,1].vlines(Lx*Lr*1e4/4,0,0.01)

ylim = ax[0,1].get_ylim()
ax[0,0].fill_between([x_sr[0],x_sr[1]],[ylim[0],ylim[0]],[ylim[1],ylim[1]],
                      facecolor=(1,0,0,0), 
                      edgecolor=(0,0,0,1),
                      zorder=100,
                      lw=1)


ylim = ax[1,1].get_ylim()
ax[1,0].fill_between([x_sr[0],x_sr[1]],[ylim[0],ylim[0]],[ylim[1],ylim[1]],
                      facecolor=(1,0,0,0), 
                      edgecolor=(0,0,0,1),
                      zorder=100,
                      lw=1)

ax[0,0].set_ylim(-5,5)
ax[1,0].set_ylim(0,3)

ax[0,0].tick_params(which='both',direction='in',
                    labelbottom=False)
ax[0,0].set_ylabel(r'$E_x$ [$10^{6}$statV/cm]')


ax[1,0].tick_params(which='both',direction='in')
ax[1,0].set_ylabel(r'$\rho$ [$10^{18}$cm$\,^{-3}$]')
ax[1,0].set_xlabel(r'$z$ ($\mu$m)')

ax[0,1].set_ylabel(r'$E_x$ [$10^{6}$statV/cm]')
ax[0,1].yaxis.tick_right()
ax[0,1].yaxis.set_label_position("right")
ax[0,1].tick_params(which='both',direction='in',
                    labelbottom=False)

ax[1,1].set_ylabel(r'$\rho$ [$10^{18}$cm$\,^{-3}$]')
ax[1,1].yaxis.tick_right()
ax[1,1].yaxis.set_label_position("right")
ax[1,1].tick_params(which='both',direction='in')
ax[1,1].set_xlabel(r'$z$ ($\mu$m)')

#ax[0,0].set_yticks([-1,1,3])
#ax[1,0].set_yticks([0.4,1.5,2,2.5])
#ax[1,1].set_yticks([0.3,0.7,1.1,1.5])

ax[0,0].set_xlim(80,205)
ax[1,0].set_xlim(80,205)
ax[0,1].set_xlim(x_sr[0],x_sr[1])
ax[1,1].set_xlim(x_sr[0],x_sr[1])

for i in range(2):ax[i,0].set_xticks([100,140,180])
for i in range(2):ax[i,1].set_xticks([191.3,192.1,192.8])
ax[1,0].set_yticks([0,1,2])

t = ['(a)','(b)','(c)','(d)']
ax[0,0].text(0.05,0.87,t[0],transform=ax[0,0].transAxes)
ax[1,0].text(0.05,0.87,t[1],transform=ax[1,0].transAxes)
ax[0,1].text(0.83,0.87,t[2],transform=ax[0,1].transAxes)
ax[1,1].text(0.05,0.87,t[3],transform=ax[1,1].transAxes)



fig.legend(frameon=False,ncol=2, bbox_to_anchor=(1.0, 1.15))

mpl.rcParams["figure.constrained_layout.use"] = False

plt.subplots_adjust(wspace=0.05, hspace=0.05,top=10)

#plt.tight_layout()
plt.savefig('im.png',dpi=600,bbox_inches='tight')



