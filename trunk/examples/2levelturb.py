import numpy as np
from spharm import Spharmt
from twolevel import TwoLevel
import numpy.random as npran
import gobject, sys, time

# animates solution from twolevel.py on screen, using matplotlib.
# requires GTKAgg matplotlib backend.

# set model parameters.
nlons = 128  # number of longitudes
ntrunc = nlons/3 # spectral truncation (for alias-free computations)
nlats = (nlons/2)+1 # for regular grid.
gridtype = 'regular'
dt = 900 # time step in seconds
tdiab = 20.*86400 # thermal relaxation time scale
tdrag = 4.*86400. # lower layer drag
efold = 1.*3600. # hyperdiffusion time scale
rsphere = 6.37122e6 # earth radius
jetexp = 2
umax = 40

# create spherical harmonic instance.
sp = Spharmt(nlons,nlats,rsphere,gridtype=gridtype)

# create model instance.
model =\
TwoLevel(sp,dt,ntrunc,efold=efold,tdiab=tdiab,tdrag=tdrag,jetexp=jetexp,umax=umax)

# initial state is equilbrium jet + random noise.
vg = np.zeros((sp.nlat,sp.nlon,2),np.float32)
ug = model.uref
vrtspec, divspec = sp.getvrtdivspec(ug,vg,model.ntrunc)
psispec = np.zeros(vrtspec.shape, vrtspec.dtype)
psispec.real += npran.normal(scale=1.e4,size=(psispec.shape))
psispec.imag += npran.normal(scale=1.e4,size=(psispec.shape))
vrtspec = vrtspec + model.lap[:,np.newaxis]*psispec
thetaspec = model.nlbalance(vrtspec)
divspec = np.zeros(thetaspec.shape, thetaspec.dtype)
model.vrt = sp.spectogrd(vrtspec)
model.theta = sp.spectogrd(thetaspec)

# animate vorticity from solution as it is running
import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic
fig = plt.figure()
m = Basemap(projection='kav7',lon_0=0)
lons1d = model.lons[0,:]*180./np.pi
lats1d = model.lats[:,0]*180./np.pi
data,lons1dx = addcyclic(model.vrt[:,:,1],lons1d)
lons, lats = np.meshgrid(lons1dx,lats1d)
x,y = m(lons,lats)
m.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1])
m.drawparallels(np.arange(-60,91,30),labels=[1,0,0,0])
levs = np.arange(-1.6e-4,1.61e-4,2.e-5)
CS=m.contourf(x,y,data,levs,cmap=plt.cm.spectral,extend='both')
m.colorbar(location='right',format='%3.1e')
t = 0.
txt = plt.title('Upper Level Vorticity (T%s, hour %6.2f)' % (ntrunc,t/3600.))

manager = plt.get_current_fig_manager()
# callback function to update contour plot
def updatefig(*args):
    global cnt,vrtspec,divspec,thetaspec,CS
    t = cnt*model.dt
    vrtspec, divspec, thetaspec = model.rk4step(vrtspec, divspec, thetaspec)
    data,lons1dx = addcyclic(model.vrt[:,:,1],lons1d)
    # remove old contours, add new ones.
    for c in CS.collections: c.remove()
    CS=m.contourf(x,y,data,levs,cmap=plt.cm.spectral,extend='both')
    txt.set_text('Upper Level Vorticity (T%s, hour %6.2f)' % (ntrunc,t/3600.))
    manager.canvas.draw()
    cnt = cnt+1
    return True

cnt = 0 # image counter
gobject.idle_add(updatefig)
sys.stdout.write('close window to exit ...\n')
plt.show()
sys.stdout.write('done\n')
