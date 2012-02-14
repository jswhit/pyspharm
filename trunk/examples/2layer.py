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
itmax = 3*(86400/dt) # integration length in days

# parameters for test
rsphere = 6.37122e6 # earth radius
omega = 7.292e-5 # rotation rate
grav = 9.80616 # gravity
cp = 1004.
theta1 = 280. ; theta2 = 310
ztop = 15.e3
pitop = cp - grav*ztop/theta1
phi0 = np.pi/7.; phi1 = 0.5*np.pi - phi0; phi2 = 0.25*np.pi
en = np.exp(-4.0/(phi1-phi0)**2)
alpha = 1./3.; beta = 1./15.
hamp = 1. # amplitude of height perturbation to zonal jet
efold = 3.*3600. # efolding timescale at ntrunc for hyperdiffusion
ndiss = 8 # order for hyperdiffusion

# setup up spherical harmonic instance, set lats/lons of grid

x = Spharmt(nlons,nlats,rsphere,gridtype=gridtype)
delta = 2.*np.pi/nlons
lats1d = 0.5*np.pi-delta*np.arange(nlats)
lons1d = np.arange(-np.pi,np.pi,delta)
lons,lats = np.meshgrid(lons1d,lats1d)
f = 2.*omega*np.sin(lats) # coriolis


# balanced zonal jets
deltapi = 0.9*(cp-pitop)
#f1 = np.cos(2.*lats)
#f3 = (1./2.)*f1*( (np.sin(2.*lats))**2 + 2.)
#df3 = -np.sin(2.*lats)* \
#      ((np.sin(2.*lats))**2+2.) + (1./2.)* \
#      np.cos(2.*lats)*4.*np.sin(2.*lats)*np.cos(2.*lats)
rnorm3 = -3./2.
rnorm5 = 1./(-(5./8.)+(5./48.)-(1./80.))
rnorm7 = (7.*rnorm5)/6.
#    rnorm9 = (9.*rnorm7)/8.
#    rnorm11 = (11.*rnorm9)/10.
#    rnorm13 = (13.*rnorm11)/12.
c = np.cos(2.*lats); s = np.sin(2.*lats)
f1 = c; df1 = -2.*s
f3 = -rnorm3*(1./3.)*f1*(s**2+2)
df3 = rnorm3*(2./3.)*s*(s**2+2)-rnorm3*(1./3.)*c*4.*s*c
f5 = rnorm5*(-(5./8.)*c+(5./48.)*np.cos(6.*lats)-(1./80.)*np.cos(10.*lats))
df5 = rnorm5*((5./4.)*s-(15./24.)*np.sin(6.*lats)+(1./8.)*np.sin(10.*lats)) 
f7 = rnorm7*(-(1./7.)*c*s**6+(6./7.)*(f5/rnorm5))
df7 = rnorm7*( (1./7.)*2.*s**7-12.*s**5*c**2+(6./7.)*(df5/rnorm5))
#        f1 = cos(2.*rlat)
#        df1 = -2.*sin(2.*rlat)
#        f3 = -rnorm3*(1./3.)*f1*( (sin(2.*rlat))**2 + 2.)
#        df3 = rnorm3*(2./3.)*sin(2.*rlat)*
#       *         ((sin(2.*rlat))**2+2.) -
#       *         rnorm3*(1./3.)*
#       *         cos(2.*rlat)*4.*sin(2.*rlat)*cos(2.*rlat)
#        f5 = rnorm5*(-(5./8.)*cos(2.*rlat) + (5./48.)*cos(6.*rlat) -
#    *        (1./80.)*cos(10.*rlat))
#        df5 = rnorm5*((5./4.)*sin(2.*rlat) -(15./24.)*sin(6.*rlat) +
#    *        (1./8.)*sin(10.*rlat))
#        f7 = rnorm7*( -(1./7.)*(sin(2.*rlat))**6*(cos(2.*rlat)) +
#    *                   (6./7.)*(f5/rnorm5) )
#        df7 = rnorm7*( (1./7)*( 2.*(sin(2.*rlat))**7 - 12.*
#    *   (sin(2.*rlat))**5*(cos(2.*rlat))**2 ) + (6./7.)*(df5/rnorm5))
#        f9 = rnorm9*(-(1./9.)*(sin(2.*rlat))**8*(cos(2.*rlat))+
#    *                   (8./9.)*(f7/rnorm7) )
#        df9 = 2.*rnorm9*(sin(2.*rlat))**9
#        f11 = rnorm11*(-(1./11.)*(sin(2.*rlat))**10*(cos(2.*rlat))+
#    *                  (10./11.)*(f9/rnorm9) )
#        df11 = 2.*rnorm11*(sin(2.*rlat))**11
#        f13 = rnorm13*((-1./13.)*(sin(2.*rlat))**12*(cos(2.*rlat))+
#    *                  (12./13)*(f11/rnorm11) )
#        df13 = 2.*rnorm13*(sin(2.*rlat))**13
pimid = 0.5*(cp+pitop) + 0.5*f5*deltapi
zint = (theta1/grav)*(cp - pimid)
s = np.sin(lats)
s[nlats/2,:] = 1. # avoid singularity at equator
c = np.cos(lats)
ct = c/s
dpidphi = 0.5*deltapi*df5
dmdphi = (theta2-theta1)*dpidphi
rad = (rsphere*omega*c)**2 - ct*dmdphi
ug = np.zeros((nlats,nlons,2),np.float32)
vg = np.zeros((nlats,nlons,2),np.float32)
deltapig = np.zeros((nlats,nlons,2),np.float32)
ubg = c*(-rsphere*omega*c + np.sqrt( rad ) )
print ubg.min(), ubg.max()
#plt.plot(lats[:,0],zint[:,0])
#plt.show()
#raise SystemExit
ug[:,:,1] = ubg/c
# add geopotential pertubation to lower layer.
hbump = hamp*np.cos(lats)*np.exp(-(lons/alpha)**2)*np.exp(-(phi2-lats)**2/beta)
deltapig[:,:,0] = cp+grav*hbump-pimid 
deltapig[:,:,1] = pimid-pitop
print deltapig[:,:,0].min(), deltapig[:,:,0].max()
print deltapig[:,:,1].min(), deltapig[:,:,1].max()
# initial vorticity, divergence and layer thicknesses in spectral space
vrtspec, divspec =  x.getvrtdivspec(ug,vg,ntrunc)
deltapispec = x.grdtospec(deltapig,ntrunc)


# create spectral indexing arrays, laplacian operator and its inverse.
indxm, indxn = getspecindx(ntrunc)
lap = -(indxn*(indxn+1.0)/rsphere**2).astype(np.float32)
ilap = np.zeros(lap.shape, np.float32)
ilap[1:] = 1./lap[1:]
ilap = ilap[:,np.newaxis]; lap = lap[:,np.newaxis]
hyperdiff_fact = np.exp((-dt/efold)*(lap/lap[-1])**(ndiss/2))

# initial perturbation
psibump = 4.e6*np.exp(-(lons/(20.*np.pi/180.))**2)*np.sin(2.*lats)**12
psibump = np.ones((nlats,nlons,2),np.float32)*psibump[:,:,np.newaxis]
vrtbumpspec = lap*x.grdtospec(psibump,ntrunc)
divbumpspec = np.zeros(vrtbumpspec.shape, np.complex64)
upert,vpert = x.getuv(vrtbumpspec,divbumpspec)
ug = ug + upert; vg = vg + vpert
# solve balance eqn.

# initialize spectral tendency arrays
ddivdtspec = np.zeros(vrtspec.shape+(3,), np.complex64)
dvrtdtspec = np.zeros(vrtspec.shape+(3,), np.complex64)
ddeltapidtspec = np.zeros(vrtspec.shape+(3,), np.complex64)
mstrm = np.zeros(deltapig.shape, deltapig.dtype)
nnew = 0; nnow = 1; nold = 2

# time loop.

time1 = time.clock()
for ncycle in range(itmax+1):
    t = ncycle*dt
# get vort,u,v,phi on grid
    vrtg = x.spectogrd(vrtspec)
    ug,vg = x.getuv(vrtspec,divspec)
    deltapig = x.spectogrd(deltapispec)
    print 't=%6.2f hours: min/max %6.2f, %6.2f' % (t/3600.,vg.min(), vg.max())
# compute tendencies.
    tmpg1 = ug*(vrtg+f[:,:,np.newaxis]); tmpg2 = vg*(vrtg+f[:,:,np.newaxis])
    ddivdtspec[:,:,nnew], dvrtdtspec[:,:,nnew] = x.getvrtdivspec(tmpg1,tmpg2,ntrunc)
    dvrtdtspec[:,:,nnew] *= -1
    tmpg1 = ug*deltapig; tmpg2 = vg*deltapig
    tmpspec, ddeltapidtspec[:,:,nnew] = x.getvrtdivspec(tmpg1,tmpg2,ntrunc)
    ddeltapidtspec[:,:,nnew] *= -1
    mstrm[:,:,0] = theta1*(deltapig[:,:,0] + deltapig[:,:,1] + pitop)
    mstrm[:,:,1] = mstrm[:,:,0] + (theta2-theta1)*(deltapig[:,:,1] + pitop)
    tmpspec = x.grdtospec(mstrm+0.5*(ug**2+vg**2),ntrunc)
    ddivdtspec[:,:,nnew] += -lap*tmpspec
# update vort,div,phiv with third-order adams-bashforth.
# forward euler, then 2nd-order adams-bashforth time steps to start.
    if ncycle == 0:
        dvrtdtspec[:,:,nnow] = dvrtdtspec[:,:,nnew]
        dvrtdtspec[:,:,nold] = dvrtdtspec[:,:,nnew]
        ddivdtspec[:,:,nnow] = ddivdtspec[:,:,nnew]
        ddivdtspec[:,:,nold] = ddivdtspec[:,:,nnew]
        ddeltapidtspec[:,:,nnow] = ddeltapidtspec[:,:,nnew]
        ddeltapidtspec[:,:,nold] = ddeltapidtspec[:,:,nnew]
    elif ncycle == 1:
        dvrtdtspec[:,:,nold] = dvrtdtspec[:,:,nnew]
        ddivdtspec[:,:,nold] = ddivdtspec[:,:,nnew]
        ddeltapidtspec[:,:,nold] = ddeltapidtspec[:,:,nnew]
    vrtspec += dt*( \
            (23./12.)*dvrtdtspec[:,:,nnew] - (16./12.)*dvrtdtspec[:,:,nnow]+ \
            (5./12.)*dvrtdtspec[:,:,nold] )
    divspec += dt*( \
            (23./12.)*ddivdtspec[:,:,nnew] - (16./12.)*ddivdtspec[:,:,nnow]+ \
            (5./12.)*ddivdtspec[:,:,nold] )
    deltapispec += dt*( \
            (23./12.)*ddeltapidtspec[:,:,nnew] - (16./12.)*ddeltapidtspec[:,:,nnow]+ \
            (5./12.)*ddeltapidtspec[:,:,nold] )
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
pvg = (vrtg+f[:,:,np.newaxis])/deltapig
pvg,lons1d = addcyclic(pvg[:,:,1],lons1d*180./np.pi)
print 'max/min PV',pvg.min(), pvg.max()
lons, lats = np.meshgrid(lons1d,lats1d*180./np.pi)
x,y = m(lons,lats)
levs = np.arange(-0.2,1.801,0.1)
m.drawmeridians(np.arange(-180,180,60))
m.drawparallels(np.arange(-80,81,20))
CS=m.contourf(x,y,pvg,20,cmap=plt.cm.spectral,extend='both')
m.colorbar()
plt.title('PV (T%s with hyperdiffusion, hour %6.2f)' % (ntrunc,t/3600.))
plt.show()
