#!/usr/bin/env python

from spharm import gaussian_lats_wts, Spharmt, legendre, specintrp, getgeodesicpts, regrid
import numpy, math, sys

# Rossby-Haurwitz wave field

def rhwave(wavenum,omega,re,lats,lons):
    return -re**2*omega*numpy.sin(lats)+re**2*omega*((numpy.cos(lats))**wavenum)*numpy.sin(lats)*numpy.cos(wavenum*lons)

# create Rossby-Haurwitz wave data on 144x73 regular and 192x94 gaussian grids.

nlats_reg = 73
nlons_reg = 144
nlats_gau = 94
nlons_gau = 192
gaulats, wts = gaussian_lats_wts(nlats_gau)
lats_reg = numpy.zeros((nlats_reg,nlons_reg),numpy.float64)
lons_reg = numpy.zeros((nlats_reg,nlons_reg),numpy.float64)
lons_gau = numpy.zeros((nlats_gau,nlons_gau),numpy.float64)

wavenum = 4.0
omega = 7.848e-6
re = 6.371e6
delat = 2.*math.pi/nlons_reg
lats = (0.5*math.pi-delat*numpy.indices(lats_reg.shape)[0,:,:])
lons = (delat*numpy.indices(lons_reg.shape)[1,:,:])
psi_reg_exact = rhwave(wavenum,omega,re,lats,lons)
delat = 2.*math.pi/nlons_gau
lats = (math.pi/180.)*numpy.transpose(gaulats*numpy.ones((nlons_gau,nlats_gau),'d'))
lons = (delat*numpy.indices(lons_gau.shape)[1,:,:])
psi_gau = rhwave(wavenum,omega,re,lats,lons)

# create Spharmt instances for regular and gaussian grids.

reggrid = Spharmt(nlons_reg,nlats_reg,gridtype='regular')
gaugrid = Spharmt(nlons_gau,nlats_gau,gridtype='gaussian')

# regrid from gaussian to regular grid.

psi_reg = regrid(gaugrid,reggrid,psi_gau)

print('reggrid error (should be less than 1.e-6):')
print(numpy.fabs(psi_reg-psi_reg_exact).max()/numpy.fabs(psi_reg_exact).max())

# spectrally interpolate to geodesic grid.

ntrunc = nlats_reg-1
latpts,lonpts = getgeodesicpts(7) # compute geodesic points
dataspec = reggrid.grdtospec(psi_reg_exact,ntrunc) # compute spectral coeffs
nlat = 0
dg2rad = math.pi/180. # degrees to radians factor.
err = []
for lat, lon in zip(latpts,lonpts):
    legfuncs = legendre(lat,ntrunc) # compute legendre functions
    intrp = specintrp(lon,dataspec,legfuncs) # do spectral interpolation
    exact = rhwave(wavenum,omega,re,dg2rad*lat,dg2rad*lon) # exact soln
    err.append(exact-intrp) # error
    nlat = nlat+1
print('spectral interpolation error (should be less than 1.e-6):')
print(max(err)/numpy.fabs(psi_reg_exact).max())
