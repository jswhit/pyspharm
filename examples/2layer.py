import numpy as np
from spharm import Spharmt, getspecindx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic
import time

# two-layer baroclinically unstable jet test case

# grid, time step info
nlons = 256  # number of longitudes
ntrunc = nlons/3 # spectral truncation (for alias-free computations)
nlats = (nlons/2)+1 # for regular grid.
gridtype = 'regular'
dt = 45 # time step in seconds
itmax = 10*(86400/dt) # integration length in days

# parameters for test
rsphere = 6.37122e6 # earth radius
omega = 7.292e-5 # rotation rate
grav = 9.80616 # gravity
cp = 1004.
theta1 = 300. ; theta2 = 330
delth = theta2-theta1
zmid = 5.e3
ztop = 15.e3
exnftop = cp - grav*ztop/theta1
exnfmid = cp  - grav*zmid/theta1
efold = 3.*3600. # efolding timescale at ntrunc for hyperdiffusion
ndiss = 8 # order for hyperdiffusion
umax = 40. # jet speed
jetexp = 6 # parameter controlling jet width

# setup up spherical harmonic instance, set lats/lons of grid

x = Spharmt(nlons,nlats,rsphere,gridtype=gridtype)
delta = 2.*np.pi/nlons
lats1d = 0.5*np.pi-delta*np.arange(nlats)
lons1d = np.arange(-np.pi,np.pi,delta)
lons,lats = np.meshgrid(lons1d,lats1d)
f = 2.*omega*np.sin(lats) # coriolis

# create spectral indexing arrays, laplacian operator and its inverse.
indxm, indxn = getspecindx(ntrunc)
lap = -(indxn*(indxn+1.0)/rsphere**2).astype(np.float32)
ilap = np.zeros(lap.shape, np.float32)
ilap[1:] = 1./lap[1:]
ilap = ilap[:,np.newaxis]; lap = lap[:,np.newaxis]
hyperdiff_fact = np.exp((-dt/efold)*(lap/lap[-1])**(ndiss/2))

# initial conditions
psibump = np.zeros((nlats,nlons,2),np.float32)
psibump[:,:,1] = 5.e6*np.sin((lons-np.pi))**12*np.sin(2.*lats)**12
psibump = np.where(lons[:,:,np.newaxis] > 0., 0, psibump)
psibump = np.where(lats[:,:,np.newaxis] < 0., psibump, -psibump)
ug = np.zeros((nlats,nlons,2),np.float32)
vg = np.zeros((nlats,nlons,2),np.float32)
ug[:,:,1] = umax*np.sin(2.*lats)**jetexp
vrtspec, divspec = x.getvrtdivspec(ug,vg,ntrunc)
vrtspec = vrtspec + lap*x.grdtospec(psibump,ntrunc)
# solve balance eqn.
vrtg = x.spectogrd(vrtspec)
ug,vg = x.getuv(vrtspec,divspec)
tmpg1 = ug*(vrtg+f[:,:,np.newaxis]); tmpg2 = vg*(vrtg+f[:,:,np.newaxis])
tmpspec1, tmpspec2 = x.getvrtdivspec(tmpg1,tmpg2,ntrunc)
tmpspec2 = x.grdtospec(0.5*(ug**2+vg**2),ntrunc)
mspec = ilap*tmpspec1 - tmpspec2
lyrthkspec = np.zeros(mspec.shape, mspec.dtype)
lyrthkspec[:,0] = mspec[:,0]/theta1
lyrthkspec[:,1] = (mspec[:,1]-mspec[:,0])/delth
lyrthkspec[:,0] = lyrthkspec[:,0] - lyrthkspec[:,1]
lyrthkspec[0,0] = (2./np.sqrt(2.))*(cp - exnfmid)
lyrthkspec[0,1] = (2./np.sqrt(2.))*(exnfmid - exnftop)
lyrthkspec = (theta1/grav)*lyrthkspec # put into height units (m)
lyrthkg = x.spectogrd(lyrthkspec)
print lyrthkg[:,:,0].min(), lyrthkg[:,:,0].max()
print lyrthkg[:,:,1].min(), lyrthkg[:,:,1].max()
if lyrthkg.min() < 0:
    raise ValueError('negative layer thickness! adjust jet parameters')


# initialize spectral tendency arrays
ddivdtspec = np.zeros(vrtspec.shape+(3,), np.complex64)
dvrtdtspec = np.zeros(vrtspec.shape+(3,), np.complex64)
dlyrthkdtspec = np.zeros(vrtspec.shape+(3,), np.complex64)
mstrm = np.zeros((nlats,nlons,2),np.float32)
nnew = 0; nnow = 1; nold = 2

# time loop.

time1 = time.clock()
for ncycle in range(itmax+1):
    t = ncycle*dt
# get vort,u,v,phi on grid
    vrtg = x.spectogrd(vrtspec)
    ug,vg = x.getuv(vrtspec,divspec)
    lyrthkg = x.spectogrd(lyrthkspec)
    print 't=%6.2f hours: min/max %6.2f, %6.2f' % (t/3600.,vg.min(), vg.max())
# compute tendencies.
    tmpg1 = ug*(vrtg+f[:,:,np.newaxis]); tmpg2 = vg*(vrtg+f[:,:,np.newaxis])
    ddivdtspec[:,:,nnew], dvrtdtspec[:,:,nnew] = x.getvrtdivspec(tmpg1,tmpg2,ntrunc)
    dvrtdtspec[:,:,nnew] *= -1
    tmpg1 = ug*lyrthkg; tmpg2 = vg*lyrthkg
    tmpspec, dlyrthkdtspec[:,:,nnew] = x.getvrtdivspec(tmpg1,tmpg2,ntrunc)
    dlyrthkdtspec[:,:,nnew] *= -1
    mstrm[:,:,0] = grav*(lyrthkg[:,:,0] + lyrthkg[:,:,1]) 
    mstrm[:,:,1] = mstrm[:,:,0] + (grav*delth/theta1)*lyrthkg[:,:,1] 
    tmpspec = x.grdtospec(mstrm+0.5*(ug**2+vg**2),ntrunc)
    ddivdtspec[:,:,nnew] += -lap*tmpspec
# update vort,div,phiv with third-order adams-bashforth.
# forward euler, then 2nd-order adams-bashforth time steps to start.
    if ncycle == 0:
        dvrtdtspec[:,:,nnow] = dvrtdtspec[:,:,nnew]
        dvrtdtspec[:,:,nold] = dvrtdtspec[:,:,nnew]
        ddivdtspec[:,:,nnow] = ddivdtspec[:,:,nnew]
        ddivdtspec[:,:,nold] = ddivdtspec[:,:,nnew]
        dlyrthkdtspec[:,:,nnow] = dlyrthkdtspec[:,:,nnew]
        dlyrthkdtspec[:,:,nold] = dlyrthkdtspec[:,:,nnew]
    elif ncycle == 1:
        dvrtdtspec[:,:,nold] = dvrtdtspec[:,:,nnew]
        ddivdtspec[:,:,nold] = ddivdtspec[:,:,nnew]
        dlyrthkdtspec[:,:,nold] = dlyrthkdtspec[:,:,nnew]
    vrtspec += dt*( \
            (23./12.)*dvrtdtspec[:,:,nnew] - (16./12.)*dvrtdtspec[:,:,nnow]+ \
            (5./12.)*dvrtdtspec[:,:,nold] )
    divspec += dt*( \
            (23./12.)*ddivdtspec[:,:,nnew] - (16./12.)*ddivdtspec[:,:,nnow]+ \
            (5./12.)*ddivdtspec[:,:,nold] )
    lyrthkspec += dt*( \
            (23./12.)*dlyrthkdtspec[:,:,nnew] - (16./12.)*dlyrthkdtspec[:,:,nnow]+ \
            (5./12.)*dlyrthkdtspec[:,:,nold] )
    # implicit hyperdiffusion for vort and div.
    vrtspec *= hyperdiff_fact
    divspec *= hyperdiff_fact
# switch indices, do next time step.
    nsav1 = nnew; nsav2 = nnow
    nnew = nold; nnow = nsav1; nold = nsav2

time2 = time.clock()
print 'CPU time = ',time2-time1

# make a orthographic plot of potential vorticity.
#m = Basemap(projection='moll',lat_0=0,lon_0=0)
#m = Basemap(projection='ortho',lon_0=-90,lat_0=40)
m = Basemap(projection='npaeqd',boundinglat=0,lon_0=0)
# dimensionless upper layer PV
pvg = (vrtg+f[:,:,np.newaxis])/lyrthkg
pvg,lons1d = addcyclic(pvg[:,:,1],lons1d*180./np.pi)
pvg = (0.5*(ztop-zmid)/omega)*pvg
print 'max/min PV',pvg.min(), pvg.max()
lons, lats = np.meshgrid(lons1d,lats1d*180./np.pi)
x,y = m(lons,lats)
levs = np.arange(-2,12,1)
m.drawmeridians(np.arange(-180,180,60))
m.drawparallels(np.arange(-80,81,20))
CS=m.contourf(x,y,pvg,20,cmap=plt.cm.spectral,extend='both')
m.colorbar()
plt.title('PV (T%s with hyperdiffusion, hour %6.2f)' % (ntrunc,t/3600.))
plt.show()
