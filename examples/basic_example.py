from pipic.pipic_tools import *
import _pipic as pipic
import matplotlib.pyplot as plt
import numpy as np

# ===========================SIMULATION INITIALIZATION===========================
Temperature = 1e-6 * electronMass * lightVelocity ** 2
Density = 1e+18
DebyeLength = sqrt(Temperature / (4 * pi * Density * electronCharge ** 2))
PlasmaPeriod = sqrt(pi * electronMass / (Density * electronCharge ** 2))
L = 128 * DebyeLength
XMin, XMax = -L / 2, L / 2
FieldAmplitude = 0.01 * 4 * pi * (XMax - XMin) * electronCharge * Density
nx = 128
timeStep = PlasmaPeriod / 64
# ---------------------setting solver and simulation region----------------------
sim = pipic.init(solver='ec', nx=nx, XMin=XMin, XMax=XMax)


# ------------------------------adding electrons---------------------------------
@cfunc(type_addParticles)
def density_callback(r, dataDouble, dataInt):
    return Density * (abs(r[0]) < L / 4)


sim.addParticles(name='electron', number=nx * 500, \
                 charge=-electronCharge, mass=electronMass, \
                 temperature=Temperature, density=density_callback.address)


# ---------------------------setting initial field-------------------------------
@cfunc(type_fieldLoop)
def setField_callback(ind, r, E, B, dataDouble, dataInt):
    E[0] = FieldAmplitude * sin(4 * pi * r[0] / (XMax - XMin)) * (abs(r[0]) < L / 4)


sim.fieldLoop(handler=setField_callback.address)
# =================================OUTPUT========================================
fig, axs = plt.subplots(2, constrained_layout=True)
# -------------preparing output for electron distrribution f(x, px)--------------
xpx_dist = numpy.zeros((64, 128), dtype=numpy.double)
pxLim = 5 * sqrt(Temperature * electronMass)
inv_dx_dpx = (xpx_dist.shape[1] / (XMax - XMin)) * (xpx_dist.shape[0] / (2 * pxLim))


@cfunc(type_particleLoop)
def xpx_callback(r, p, w, id, dataDouble, dataInt):
    ix = int(xpx_dist.shape[1] * (r[0] - XMin) / (XMax - XMin))
    iy = int(xpx_dist.shape[0] * 0.5 * (1 + p[0] / pxLim))
    data = carray(dataDouble, xpx_dist.shape, dtype=numpy.double)
    if iy >= 0 and iy < xpx_dist.shape[0]:
        data[iy, ix] += w[0] * inv_dx_dpx / (3 * Density / pxLim)


axs[0].set_title('$\partial N / \partial x \partial p_x$ (s g$^{-1}$cm$^{-2}$)')
axs[0].set(ylabel='$p_x$ (cm g/s)')
axs[0].xaxis.set_ticklabels([])
plot0 = axs[0].imshow(xpx_dist, vmin=0, vmax=1, \
                      extent=[XMin, XMax, -pxLim, pxLim], interpolation='none', \
                      aspect='auto', cmap='YlOrBr')
fig.colorbar(plot0, ax=axs[0], location='right')


def plot_xpx():
    xpx_dist.fill(0)
    sim.particleLoop(name='electron', handler=xpx_callback.address, \
                     dataDouble=addressOf(xpx_dist))
    plot0.set_data(xpx_dist)


# -------------------------preparing output of Ex(x)-----------------------------
Ex = numpy.zeros((32,), dtype=numpy.double)


@cfunc(type_it2r)
def Ex_it2r(it, r, dataDouble, dataInt):
    r[0] = XMin + (it[0] + 0.5) * (XMax - XMin) / Ex.shape[0]


@cfunc(type_field2data)
def get_Ex(it, r, E, B, dataDouble, dataInt):
    dataDouble[it[0]] = E[0]


axs[1].set_xlim([XMin, XMax])
axs[1].set_ylim([-FieldAmplitude, FieldAmplitude])
axs[1].set(xlabel='$x$ (cm)', ylabel='$E_x$ (cgs units)')
x_axis = np.linspace(XMin, XMax, Ex.shape[0])
plot_Ex_, = axs[1].plot(x_axis, Ex)


def plot_Ex():
    sim.customFieldLoop(numberOfIterations=Ex.shape[0], it2r=Ex_it2r.address, \
                        field2data=get_Ex.address, dataDouble=addressOf(Ex))
    plot_Ex_.set_ydata(Ex)


# ===============================SIMULATION======================================
outputFolder = 'basic_example_output'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
time_start = time.time()
for i in range(32):
    sim.advance(timeStep=timeStep, numberOfIterations=8)
    plot_xpx()
    plot_Ex()
    fig.savefig(outputFolder + '/im' + str(i) + '.png')
    print(i, '/', 32)
