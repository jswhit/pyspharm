import numpy as np
from spharm import Spharmt
from twolevel import TwoLevel
import numpy.random as npran
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic

# animates solution from twolevel.py on screen, using matplotlib.

# set model parameters.
nlons = 128  # number of longitudes
ntrunc = 42  # spectral truncation (for alias-free computations)
nlats = (nlons/2)+1 # for regular grid.
gridtype = 'regular'
dt = 900 # time step in seconds
tdiab = 12.*86400 # thermal relaxation time scale
tdrag = 4.*86400. # lower layer drag
efold = 4*dt # hyperdiffusion time scale
rsphere = 6.37122e6 # earth radius
jetexp = 2
umax = 40
moistfact = 0.1

# create spherical harmonic instance.
sp = Spharmt(nlons,nlats,ntrunc=ntrunc,rsphere=rsphere,gridtype=gridtype)

# create model instance.
model =\
TwoLevel(sp,dt,ntrunc,efold=efold,tdiab=tdiab,tdrag=tdrag,jetexp=jetexp,umax=umax,moistfact=moistfact)

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
model.heat = np.zeros(model.theta.shape, model.theta.dtype)

# animate vorticity from solution as it is running
# first, create initial frame
fig = plt.figure(figsize=(8,10))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
m = Basemap(projection='kav7',lon_0=0)
lons1d = model.lons[0,:]*180./np.pi
lats1d = model.lats[:,0]*180./np.pi
data1,lons1dx = addcyclic(model.vrt[:,:,1],lons1d)
data2,lons1dx = addcyclic(model.theta,lons1d)
lons, lats = np.meshgrid(lons1dx,lats1d)
x,y = m(lons,lats)
m.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],ax=ax1)
m.drawparallels(np.arange(-60,91,30),labels=[1,0,0,0],ax=ax1)
m.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],ax=ax2)
m.drawparallels(np.arange(-60,91,30),labels=[1,0,0,0],ax=ax2)
levs1 = np.arange(-1.6e-4,1.61e-4,2.e-5)
levs2 = np.arange(-60,21,2)
CS1=m.contourf(x,y,data1,levs1,cmap=plt.cm.spectral,extend='both',ax=ax1)
CS2=m.contourf(x,y,data2,levs2,cmap=plt.cm.spectral,extend='both',ax=ax2)
cb1 = m.colorbar(CS1,location='right',format='%3.1e',ax=ax1)
cb2 = m.colorbar(CS2,location='right',format='%g',ax=ax2)
t = 0.
txt1 = ax1.set_title('Upper Level Vorticity (T%s, hour %6.2f)' % (ntrunc,t/3600.))
txt2 = ax2.set_title('Temperature (T%s, hour %6.2f)' % (ntrunc,t/3600.))


# run model, update contours and title on the fly.
def updatefig(*args):
    global vrtspec,divspec,thetaspec,CS1,CS2,txt1,txt2,t
    t += model.dt
    vrtspec, divspec, thetaspec = model.rk4step(vrtspec, divspec, thetaspec)
    data1,lons1dx = addcyclic(model.vrt[:,:,1],lons1d)
    data2,lons1dx = addcyclic(model.theta,lons1d)
    # remove old contours, add new ones.
    for c in CS1.collections: c.remove()
    CS1=m.contourf(x,y,data1,levs1,cmap=plt.cm.spectral,extend='both',ax=ax1)
    for c in CS2.collections: c.remove()
    CS2=m.contourf(x,y,data2,levs2,cmap=plt.cm.spectral,extend='both',ax=ax2)
    # update titles.
    txt1.set_text('Upper Level Vorticity (T%s, hour %6.2f)' % (ntrunc,t/3600.))
    txt2.set_text('Temperature (T%s, hour %6.2f)' % (ntrunc,t/3600.))
ani = animation.FuncAnimation(fig, updatefig)
plt.show()
