#!/usr/bin/env python
#
#  Test program for spharm module - non-linear steady-state geostropic
#  flow in a shallow water model.
#
#
#  errors should be O(10E-6) or less.
#
#     the nonlinear shallow-water equations on the sphere are
#     solved using a spectral method based on the spherical harmonics.
#     the method is described in the paper:
#
# [1] p. n. swarztrauber, spectral transform methods for solving
#     the shallow-water equations on the sphere, p.n. swarztrauber,
#     monthly weather review, vol. 124, no. 4, april 1996, pp. 730-744.
#
#     this program implements test case 3 (steady nonlinear rotated flow)
#     in the paper:
#
# [2] d.l. williamson, j.b. drake, j.j. hack, r. jakob, and
#     p.n. swarztrauber, j. comp. phys., a standard test set
#     for numerical approximations to the shallow-water
#     equations in spherical geometry, j. comp. phys.,
#     vol. 102, no. 1, sept. 1992, pp. 211-224.
#
import numpy, math, sys, time
from spharm import Spharmt, getspecindx
from FortranFormat import *

nlon = 128
nlat = 65
ntrunc = 42
gridtype = 'regular'
legfunc = 'stored'
rsphere = 6.3712e6

pi = math.pi
hpi = pi/2.
dtr = pi/180.
aa = rsphere
omega = 7.292e-5
fzero = omega+omega
uzero = 40.
pzero = 2.94e4
alphad = 60.
alpha = dtr*alphad

dt = 300.
itmax = int(86400.*5.0/dt)
mprint = itmax/10

 
#     compute the derivative of the unrotated geopotential
#     p as a function of latitude

def getui(amp,thetad):
    """
      computes the initial unrotated longitudinal velocity
      see section 3.3.
    """
    thetab=-math.pi/6.
    thetae= math.pi/2.
    xe=3.e-1
    x =xe*(thetad-thetab)/(thetae-thetab)
    ui = 0.
    if x <= 0. or x >= xe:
      return 0.
    else:
      return amp*math.exp(-1./x-1./(xe-x)+4./xe)

def atanxy(x,y):
    if x == 0. and y == 0.0: 
        return 0.
    else:
        return math.atan2(y,x)

def sine(x):
    """ computes the sine transform"""
    n = x.shape[0]
    arg = math.pi/(n+1)
    w = numpy.zeros(n,numpy.float32)
    for j in range(1,n+1):
        w[j-1] = 0.
        for i in range(1,n+1):
            w[j-1] = w[j-1]+x[i-1]*math.sin(i*j*arg)
    return 2*w
 
def getcosine(cf,theta):
    """ computes the cosine transform"""
    n = cf.shape[0]
    cosine = 0.
    for i in range(1,n+1):
      cosine = cosine+cf[i-1]*math.cos(i*theta)
    return cosine
 
nl = 91
nlm1 = nl-1
nlm2 = nl-2
cfn = 1./nlm1
dlath = pi/nlm1
phlt = numpy.zeros((nlm2),numpy.float32)
for i in range(1,nlm2+1):
    theta = i*dlath
    sth = math.sin(theta)
    cth = math.cos(theta)
    uhat = getui(uzero,hpi-theta)
    phlt[i-1] = cfn*cth*uhat*(uhat/sth+aa*fzero)
 
#     compute sine transform of the derivative of the geopotential
#     for the purpose of computing the geopotential by integration
#     see equation (3.9) in reference [1] above
 
phlt = sine(phlt)
 
#     compute the cosine coefficients of the unrotated geopotential
#     by the formal integration of the sine series representation
#
for i in range(nlm2):
    phlt[i] = -phlt[i]/float(i+1)
 
#     phlt(i) contains the coefficients in the cosine series
#     representation of the unrotated geopotential that are used
#     below to compute the geopotential on the rotated grid.
 
#     compute the initial values of  east longitudinal
#     and latitudinal velocities u and v as well as the
#     geopotential p and coriolis f on the rotated grid.
 
ca = math.cos(alpha)
sa = math.sin(alpha)
dlam = (pi+pi)/nlon
dtheta = pi/(nlat-1)
dtor = pi/180.

if gridtype == 'gaussian':
    gaulats,weights = gaussian_lats_wts(nlat)

uxact = numpy.zeros((nlat,nlon),numpy.float32)
vxact = numpy.zeros((nlat,nlon),numpy.float32)
pxact = numpy.zeros((nlat,nlon),numpy.float32)
f     = numpy.zeros((nlat,nlon),numpy.float32)

for j in range(nlon):
    lamda = j*dlam
    cl = math.cos(lamda)
    sl = math.sin(lamda)
    for i in range(nlat):
 
#     lamda is longitude, theta is colatitude, and pi/2-theta is
#     latitude on the rotated grid. lhat and that are longitude
#     and colatitude on the unrotated grid. see text starting at
#     equation (3.10)
 
        if gridtype == 'gaussian':
            theta = hpi-dtor*gaulats[i]
        else:
            theta = i*dtheta
        st = math.cos(theta)
        ct = math.sin(theta)
        sth = ca*st+sa*ct*cl
        cthclh = ca*ct*cl-sa*st
        cthslh = ct*sl
        lhat = atanxy(cthclh,cthslh)
        clh = math.cos(lhat)
        slh = math.sin(lhat)
        cth = clh*cthclh+slh*cthslh
        that = atanxy(sth,cth)
        uhat = getui(uzero,hpi-that)
        pxact[i,j] = getcosine(phlt,that)
        uxact[i,j] = uhat*(ca*sl*slh+cl*clh)
        vxact[i,j] = uhat*(ca*cl*slh*st-clh*sl*st+sa*slh*ct)
        f[i,j] = fzero*sth

print repr(nlon)+'x'+repr(nlat)+' '+gridtype+' grid, T'+repr(ntrunc)+' truncation'+', leg. functions are '+legfunc

x = Spharmt(nlon,nlat,rsphere,gridtype=gridtype,legfunc=legfunc)

# find L1 and L2 norms of exact solution.

v2max = numpy.add.reduce(numpy.add.reduce(uxact*uxact)) + numpy.add.reduce(numpy.add.reduce(vxact*vxact))
p2max = numpy.add.reduce(numpy.add.reduce(pxact*pxact))
umax = numpy.fabs(uxact).max()
vmax = numpy.fabs(vxact).max()
vmax = max(umax,vmax)
pmax = numpy.fabs(pxact).max()
 
# initialize first time step
 
ug = uxact.astype(numpy.float32)
vg = vxact.astype(numpy.float32)
pg = pxact.astype(numpy.float32)

# compute spectral coeffs of initial vrt,div,p.

vrtspec, divspec =  x.getvrtdivspec(ug,vg,ntrunc)
pspec = x.grdtospec(pg,ntrunc)

ddivdtspec = numpy.array(numpy.zeros(((ntrunc+1)*(ntrunc+2)/2,3)),numpy.complex)
dvrtdtspec = numpy.array(numpy.zeros(((ntrunc+1)*(ntrunc+2)/2,3)),numpy.complex)
dpdtspec = numpy.array(numpy.zeros(((ntrunc+1)*(ntrunc+2)/2,3)),numpy.complex)

# create spectral indexing arrays, laplacian operator.

indxm, indxn = getspecindx(ntrunc)
lap = -(indxn*(indxn+1.0)/rsphere**2).astype(numpy.float32)

#==> time step loop.

time1 = time.clock()

nnew = 0
nnow = 1
nold = 2
for ncycle in range(itmax+1):
  t = ncycle*dt

#==> INVERSE TRANSFORM TO GET VORT AND PHIG ON GRID.
 
  vrtg = x.spectogrd(vrtspec)
  pg = x.spectogrd(pspec)
 
#==> compute u and v on grid from spectral coeffs. of vort and div.
 
  ug,vg = x.getuv(vrtspec,divspec)

#==> use the following to exercise getpsichi, getgrad.
#    (these lines can be commented out)

  psigrid, chigrid = x.getpsichi(ug, vg, ntrunc)
  chispec = x.grdtospec(chigrid, ntrunc)
  psispec = x.grdtospec(psigrid, ntrunc)
  uchi, vchi = x.getgrad(chispec)
  vpsi, upsi = x.getgrad(psispec)
  ug = uchi - upsi
  vg = vchi + vpsi

#==> compute error statistics.

  if ncycle%mprint == 0:
    divg =  x.spectogrd(divspec)
    ht = t/3600.

    dvgm = numpy.fabs(divg).max()
    dvmax = numpy.add.reduce(numpy.add.reduce((ug-uxact)*(ug-uxact))) + numpy.add.reduce(numpy.add.reduce((vg-vxact)*(vg-vxact)))
    dpmax = numpy.add.reduce(numpy.add.reduce((pg-pxact)*(pg-pxact)))
    eumax = numpy.fabs((ug-uxact)).max()
    evmax = numpy.fabs((vg-vxact)).max()
    evmax = max(eumax,evmax)
    epmax = numpy.fabs((pg-pxact)).max()
    dvmax = math.sqrt(dvmax/v2max)
    dpmax = math.sqrt(dpmax/p2max)
    evmax = evmax/vmax
    epmax = epmax/pmax

    format = FortranFormat('A8,1F7.2')
    line = FortranLine([' time = ',ht], format)
    print str(line)
    format = FortranFormat('A22,1E15.6')
    line = FortranLine([' max error in velocity',evmax], format)
    print str(line)
    line = FortranLine([' max error in geopot. ',epmax], format)
    print str(line)
    line = FortranLine([' l2 error in velocity ',dvmax], format)
    print str(line)
    line = FortranLine([' l2 error in geopot.  ',dpmax], format)
    print str(line)
    line = FortranLine([' maximum divergence   ',dvgm], format)
    print str(line)

#==> COMPUTE RIGHT-HAND SIDES OF PROGNOSTIC EQNS. 
 
  scrg1 = ug*(vrtg+f)
  scrg2 = vg*(vrtg+f)
  ddivdtspec[:,nnew],dvrtdtspec[:,nnew] = x.getvrtdivspec(scrg1,scrg2,ntrunc)
  dvrtdtspec[:,nnew]=-dvrtdtspec[:,nnew]
  scrg1 = ug*(pg+pzero)
  scrg2 = vg*(pg+pzero)
  tmpspec, dpdtspec[:,nnew] = x.getvrtdivspec(scrg1,scrg2,ntrunc)
  dpdtspec[:,nnew]=-dpdtspec[:,nnew]
  scrg1 = pg+0.5*(ug**2+vg**2)
  tmpspec = x.grdtospec(scrg1,ntrunc)
  ddivdtspec[:,nnew]=ddivdtspec[:,nnew]-lap*tmpspec

#==> update vrt and div with third-order adams-bashforth.

#==> forward euler, then 2nd-order adams-bashforth time steps to start.

  if ncycle == 0:
    dvrtdtspec[:,nnow] = dvrtdtspec[:,nnew]
    dvrtdtspec[:,nold] = dvrtdtspec[:,nnew]
    ddivdtspec[:,nnow] = ddivdtspec[:,nnew]
    ddivdtspec[:,nold] = ddivdtspec[:,nnew]
    dpdtspec[:,nnow] = dpdtspec[:,nnew]
    dpdtspec[:,nold] = dpdtspec[:,nnew]
  elif ncycle == 1:
    dvrtdtspec[:,nold] = dvrtdtspec[:,nnew]
    ddivdtspec[:,nold] = ddivdtspec[:,nnew]
    dpdtspec[:,nold] = dpdtspec[:,nnew]
  vrtspec = vrtspec + dt*( \
  (23./12.)*dvrtdtspec[:,nnew] - (16./12.)*dvrtdtspec[:,nnow]+ \
  (5./12.)*dvrtdtspec[:,nold] )
  divspec = divspec + dt*( \
  (23./12.)*ddivdtspec[:,nnew] - (16./12.)*ddivdtspec[:,nnow]+ \
  (5./12.)*ddivdtspec[:,nold] )
  pspec = pspec + dt*( \
  (23./12.)*dpdtspec[:,nnew] - (16./12.)*dpdtspec[:,nnow]+ \
  (5./12.)*dpdtspec[:,nold] )
 
#==> switch indices, do next time step.
 
  nsav1 = nnew
  nsav2 = nnow
  nnew = nold  
  nnow = nsav1
  nold = nsav2

time2 = time.clock()
print 'CPU time = ',time2-time1
