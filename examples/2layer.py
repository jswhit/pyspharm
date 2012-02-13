import numpy as np
from spharm import Spharmt, getspecindx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic
import time

# modification of Galewsky et al test case for two-layer
# baroclinically unstable jet using model from
# Keppenne, Christian L., Steven L. Marcus, Masahide Kimoto, Michael Ghil,
# 2000: Intraseasonal Variability in a Two-Layer Model and Observations. J.
# Atmos. Sci., 57, 1010-1028.

# grid, time step info
nlons = 256  # number of longitudes
ntrunc = nlons/3 # spectral truncation (for alias-free computations)
nlats = (nlons/2)+1 # for regular grid.
gridtype = 'regular'
dt = 60 # time step in seconds
itmax = 5*(86400/dt) # integration length in days

# parameters for test
rsphere = 6.37122e6 # earth radius
omega = 7.292e-5 # rotation rate
grav = 9.80616 # gravity
hbar1 = 5.e3 # resting depth of each layer
hbar2 = 15.e3 # resting depth of each layer
umax = 30. # jet speed
phi0 = np.pi/7.; phi1 = 0.5*np.pi - phi0; phi2 = 0.25*np.pi
en = np.exp(-4.0/(phi1-phi0)**2)
alpha = 1./3.; beta = 1./15.
hamp = 150. # amplitude of height perturbation to zonal jet
efold = 3.*3600. # efolding timescale at ntrunc for hyperdiffusion
ndiss = 8 # order for hyperdiffusion
densityratio = 0.9 # density of upper layer/lower layer

# setup up spherical harmonic instance, set lats/lons of grid

x = Spharmt(nlons,nlats,rsphere,gridtype=gridtype)
delta = 2.*np.pi/nlons
lats1d = 0.5*np.pi-delta*np.arange(nlats)
lons1d = np.arange(-np.pi,np.pi,delta)
lons,lats = np.meshgrid(lons1d,lats1d)
f = 2.*omega*np.sin(lats) # coriolis

# zonal jet.
vg = np.zeros((nlats,nlons,2),np.float32)
u1 = (umax/en)*np.exp(1./((lats1d-phi0)*(lats1d-phi1)))
ug = np.zeros((nlats),np.float32)
ug = np.where(np.logical_and(lats1d < phi1, lats1d > phi0), u1, ug)
ug.shape = (nlats,1,1)
# broadcast to shape (nlats,nlons,2)
ug = ug*np.ones((nlats,nlons,2),dtype=np.float32)
ug[:,:,0] = 0. # state of rest in lower layer.
# simpler jet
ug[:,:,1] = umax*(np.sin(2.*lats))**2
# height perturbation.
hbump = hamp*np.cos(lats)*np.exp(-(lons/alpha)**2)*np.exp(-(phi2-lats)**2/beta)

# initial vorticity, divergence in spectral space
vrtspec, divspec =  x.getvrtdivspec(ug,vg,ntrunc)

# create spectral indexing arrays, laplacian operator and its inverse.
indxm, indxn = getspecindx(ntrunc)
lap = -(indxn*(indxn+1.0)/rsphere**2).astype(np.float32)
ilap = np.zeros(lap.shape, np.float32)
ilap[1:] = 1./lap[1:]
ilap = ilap[:,np.newaxis]; lap = lap[:,np.newaxis]
hyperdiff_fact = np.exp((-dt/efold)*(lap/lap[-1])**(ndiss/2))

# solve nonlinear balance eqn to get initial zonal geopotential,
# add localized bump (not balanced).
vrtg = x.spectogrd(vrtspec)
tmpg1 = ug*(vrtg+f[:,:,np.newaxis]); tmpg2 = vg*(vrtg+f[:,:,np.newaxis])
tmpspec1, tmpspec2 = x.getvrtdivspec(tmpg1,tmpg2,ntrunc)
tmpspec2 = x.grdtospec(0.5*(ug**2+vg**2),ntrunc)
phispec = ilap*tmpspec1 - tmpspec2
# lower layer at rest implies this
phispec[:,1] = phispec[:,1]/(1.-densityratio)
phispec[:,0] = -densityratio*phispec[:,1]
phig = x.spectogrd(phispec)
# add basic state of rest.
phig[:,:,0] += grav*hbar1
phig[:,:,1] += grav*hbar2
# add geopotential pertubation to upper layer.
phig[:,:,1] += grav*hbump
# grid to spectral
phispec = x.grdtospec(phig,ntrunc)

# initialize spectral tendency arrays
ddivdtspec = np.zeros(vrtspec.shape+(3,), np.complex64)
dvrtdtspec = np.zeros(vrtspec.shape+(3,), np.complex64)
dphidtspec = np.zeros(vrtspec.shape+(3,), np.complex64)
nnew = 0; nnow = 1; nold = 2

# time loop.

time1 = time.clock()
for ncycle in range(itmax+1):
    t = ncycle*dt
# get vort,u,v,phi on grid
    vrtg = x.spectogrd(vrtspec)
    ug,vg = x.getuv(vrtspec,divspec)
    phig = x.spectogrd(phispec)
    print 't=%6.2f hours: min/max %6.2f, %6.2f' % (t/3600.,vg.min(), vg.max())
# compute tendencies.
    tmpg1 = ug*(vrtg+f[:,:,np.newaxis]); tmpg2 = vg*(vrtg+f[:,:,np.newaxis])
    ddivdtspec[:,:,nnew], dvrtdtspec[:,:,nnew] = x.getvrtdivspec(tmpg1,tmpg2,ntrunc)
    dvrtdtspec[:,:,nnew] *= -1
    tmpg1 = ug*phig; tmpg2 = vg*phig
    tmpspec, dphidtspec[:,:,nnew] = x.getvrtdivspec(tmpg1,tmpg2,ntrunc)
    dphidtspec[:,:,nnew] *= -1
    pgf = np.empty(phig.shape, phig.dtype)
    pgf[:,:,0] = phig[:,:,0] + densityratio*phig[:,:,1]
    pgf[:,:,1] = phig[:,:,0] + phig[:,:,1]
    tmpspec = x.grdtospec(pgf+0.5*(ug**2+vg**2),ntrunc)
    ddivdtspec[:,:,nnew] += -lap*tmpspec
# update vort,div,phiv with third-order adams-bashforth.
# forward euler, then 2nd-order adams-bashforth time steps to start.
    if ncycle == 0:
        dvrtdtspec[:,:,nnow] = dvrtdtspec[:,:,nnew]
        dvrtdtspec[:,:,nold] = dvrtdtspec[:,:,nnew]
        ddivdtspec[:,:,nnow] = ddivdtspec[:,:,nnew]
        ddivdtspec[:,:,nold] = ddivdtspec[:,:,nnew]
        dphidtspec[:,:,nnow] = dphidtspec[:,:,nnew]
        dphidtspec[:,:,nold] = dphidtspec[:,:,nnew]
    elif ncycle == 1:
        dvrtdtspec[:,:,nold] = dvrtdtspec[:,:,nnew]
        ddivdtspec[:,:,nold] = ddivdtspec[:,:,nnew]
        dphidtspec[:,:,nold] = dphidtspec[:,:,nnew]
    vrtspec += dt*( \
            (23./12.)*dvrtdtspec[:,:,nnew] - (16./12.)*dvrtdtspec[:,:,nnow]+ \
            (5./12.)*dvrtdtspec[:,:,nold] )
    divspec += dt*( \
            (23./12.)*ddivdtspec[:,:,nnew] - (16./12.)*ddivdtspec[:,:,nnow]+ \
            (5./12.)*ddivdtspec[:,:,nold] )
    phispec += dt*( \
            (23./12.)*dphidtspec[:,:,nnew] - (16./12.)*dphidtspec[:,:,nnow]+ \
            (5./12.)*dphidtspec[:,:,nold] )
    # implicit hyperdiffusion for vort and div.
    vrtspec *= hyperdiff_fact
    divspec *= hyperdiff_fact
# switch indices, do next time step.
    nsav1 = nnew; nsav2 = nnow
    nnew = nold; nnow = nsav1; nold = nsav2

time2 = time.clock()
print 'CPU time = ',time2-time1

# make a orthographic plot of potential vorticity.
m = Basemap(projection='ortho',lat_0=45,lon_0=-100)
# dimensionless upper layer PV
pvg = (0.5*hbar2*grav/omega)*(vrtg+f[:,:,np.newaxis])/phig
pvg,lons1d = addcyclic(pvg[:,:,1],lons1d*180./np.pi)
print 'max/min PV',pvg.min(), pvg.max()
lons, lats = np.meshgrid(lons1d,lats1d*180./np.pi)
x,y = m(lons,lats)
levs = np.arange(-0.2,1.801,0.1)
m.drawmeridians(np.arange(-180,180,60))
m.drawparallels(np.arange(-80,81,20))
CS=m.contourf(x,y,pvg,levs,cmap=plt.cm.spectral,extend='both')
m.colorbar()
plt.title('PV (T%s with hyperdiffusion, hour %6.2f)' % (ntrunc,t/3600.))
plt.show()
