import numpy as np
from spharm import Spharmt, getspecindx, gaussian_lats_wts

# two-level baroclinic primitive equation model of
# Lee, S. and I. M. Held: 1993: Baroclinic Wave Packets in Models and Observations
# J. Atmos. Sci., 50, 1413-1428.

class TwoLevel(object):

    def __init__(self,sp,dt,ntrunc,ptop=0.,p0=1.e5,grav=9.80616,omega=7.292e-5,cp=1004,\
            rgas=287.,efold=3600.,ndiss=8,tdrag=1.e30,tdiab=1.e30,umax=40,delth=20):
        # setup model parameters
        self.p0 = p0 # mean surface pressure
        self.ptop = ptop # model top pressure
        self.rgas rgas # gas constant for dry air
        self.grav = grav # gravity
        self.omega = omega # rotation rate
        self.cp = cp # Specific Heat of Dry Air at Constant Pressure,
        self.delth = delth # vertical stability
        dp = 0.5*(ptop-p0)
        exnf1 = cp*((p0+0.5*dp)/p0)**(r/cp)
        exnf2 = cp*((p0+1.5*dp)/p0)**(r/cp)
        self.delta_exnf = exnf2-exnf1 # diff in exner function between 2 levs.
        # efolding time scale for hyperdiffusion at shortest wavenumber
        self.efold = efold 
        self.ndiss = ndiss # order of hyperdiffusion (2 for laplacian)
        self.sp = sp # Spharmt instance
        self.ntrunc = ntrunc # triangular truncation wavenumber
        self.dt = dt # time step (secs)
        self.tdiab = tdiab # lower layer drag timescale
        self.tdrag = tdrag # interface relaxation timescale
        # create lat/lon arrays
        delta = 2.*np.pi/sp.nlon
        if sp.gridtype == 'regular':
           lats1d = 0.5*np.pi-delta*np.arange(sp.nlat)
        else:
           lats1d,wts = gaussian_lats_wts(sp.nlat)
           lats1d = lats1d*np.pi/180.
        lons1d = np.arange(-np.pi,np.pi,delta)
        lons,lats = np.meshgrid(lons1d,lats1d)
        self.lons = lons
        self.lats = lats
        self.f = 2.*omega*np.sin(lats)[:,:,np.newaxis] # coriolis
        # create laplacian operator and its inverse.
        indxm, indxn = getspecindx(ntrunc)
        indxn = indxn.astype(np.float32)[:,np.newaxis]
        totwavenum = indxn*(indxn+1.0)
        self.lap = -totwavenum/sp.rsphere**2
        self.ilap = np.zeros(self.lap.shape, np.float32)
        self.ilap[1:,:] = 1./self.lap[1:,:]
        # hyperdiffusion operator
        self.hyperdiff = -(1./efold)*(totwavenum/totwavenum[-1])**(ndiss/2)
        # set equilibrium layer thicknes profile.
        self._interface_profile(umax)

    def _interface_profile(self,umax):
        ug = np.zeros((self.sp.nlat,self.sp.nlon,2),np.float32) 
        vg = np.zeros((self.sp.nlat,self.sp.nlon,2),np.float32) 
        ug[:,:,1] = umax*np.sin(2.*self.lats)**2
        vrtspec, divspec = self.sp.getvrtdivspec(ug,vg,self.ntrunc)
        lyrthkspec = self.nlbalance(vrtspec)
        self.lyrthkref = self.sp.spectogrd(lyrthkspec)
        self.uref = ug
        if self.lyrthkref.min() < 0:
            raise ValueError('negative layer thickness! adjust equilibrium jet parameter')

    def nlbalance(self,vrtspec):
        # solve nonlinear balance eqn to get layer thickness given vorticity.
        divspec = np.zeros(vrtspec.shape, vrtspec.dtype)
        lyrthkspec = np.zeros(vrtspec.shape, vrtspec.dtype)
        vrtg = self.sp.spectogrd(vrtspec)
        ug,vg = self.sp.getuv(vrtspec,divspec)
        tmpg1 = ug*(vrtg+self.f); tmpg2 = vg*(vrtg+self.f)
        tmpspec1, tmpspec2 = self.sp.getvrtdivspec(tmpg1,tmpg2,self.ntrunc)
        tmpspec2 = self.sp.grdtospec(0.5*(ug**2+vg**2),self.ntrunc)
        mspec = self.ilap*tmpspec1 - tmpspec2
        lyrthkspec[:,0] =\
        (mspec[:,0]-self.sp.grdtospec(self.grav*self.orog,self.ntrunc))/self.theta1
        lyrthkspec[:,1] = (mspec[:,1]-mspec[:,0])/self.delth
        lyrthkspec[:,0] = lyrthkspec[:,0] - lyrthkspec[:,1]
        exnftop = self.cp - (self.grav*self.ztop/self.theta1)
        exnfmid = self.cp - (self.grav*self.zmid/self.theta1)
        lyrthkspec[0,0] = (2./np.sqrt(2.))*(self.cp - exnfmid)
        lyrthkspec[0,1] = (2./np.sqrt(2.))*(exnfmid - exnftop)
        lyrthkspec = (self.theta1/self.grav)*lyrthkspec # convert from exner function to height units (m)
        return lyrthkspec

    def gettend(self,vrtspec,divspec,thetaspec):
        # compute tendencies.
        # first, transform fields from spectral space to grid space.
        vrtg = self.sp.spectogrd(vrtspec)
        # this is baroclinic div = divupper-divlower
        # omega = dp*divg/2
        # baroptropic div is zero.
        divg = self.sp.spectogrd(divspec)
        divspec2 = np.empty(vrtspec.shape, vrtspec.dtype)
        divspec2[:,0] = -0.5*divspec
        divspec2[:,1] = 0.5*divspec
        ug,vg = self.sp.getuv(vrtspec,divspec2)
        thetag = self.sp.spectogrd(thetaspec)
        self.u = ug; self.v = vg; self.divg = divg
        self.vrt = vrtg; self.thetag = lyrthkg
        vadvu = 0.25*(divg*(ug[:,:,1]-ug[:,:,0])
        vadvv = 0.25*(divg*(vg[:,:,1]-vg[:,:,0])
        # horizontal vorticity flux
        tmpg1 = ug*(vrtg+self.f); tmpg2 = vg*(vrtg+self.f)
        # add lower layer drag and vertical advection contributions
        tmpg1[:,:,0] += vadvv + vg[:,:,0]/self.tdrag 
        tmpg2[:,:,0] += -vadvu - ug[:,:,0]/self.tdrag
        tmpg1[:,:,1] += vadvv 
        tmpg2[:,:,1] += -vadvu 
        # compute vort flux contributions to vorticity and divergence tend.
        tmpspec, dvrtdtspec = self.sp.getvrtdivspec(tmpg1,tmpg2,self.ntrunc)
        ddivdtspec = tmpspec[:,1]-tmpspec[:,0]
        dvrtdtspec *= -1
        # vorticity hyperdiffusion
        dvrtdtspec += self.hyperdiff*vrtspec
        # add laplacian term and hyperdiffusion to div tend.
        ke = 0.5*(ug**2+vg**2)
        tmpspec = self.sp.grdtospec(ke[:,:,1]-ke[:,:,0],ntrunc)
        ddivdtspec += self.hyperdiff*divspec - \
                ilap*(tmpspec - self.delta_exnf*thetaspec)
        # horizontal gradient of pot. temp.
        thetagradx, thetagrady = self.sp.getgrad(thetaspec)
        # tendency of pot. temp.
        vadvtheta = 0.5*divg*self.delth
        umean = 0.5*(ug[:,:,1]+ug[:,:,0])
        vmean = 0.5*(vg[:,:,1]+vg[:,:,0])
        # advection
        tmpg = -umean*thetagradx - vmean*thetagrady - vadvtheta
        # thermal relaxation term.
        tmpg += (self.thetaref-thetag)/self.tdiab
        dthetadtspec = self.sp.grdtospec(tmpg, ntrunc)
        # hyperdiffusion
        dthetadtspec += self.hyperdiff*thetaspec
        return dvrtdtspec,ddivdtspec,dlyrthkdtspec

    def rk4step(self,vrtspec,divspec,lyrthkspec):
        # update state using 4th order runge-kutta
        dt = self.dt
        k1vrt,k1div,k1thk = \
        self.gettend(vrtspec,divspec,lyrthkspec)
        k2vrt,k2div,k2thk = \
        self.gettend(vrtspec+0.5*dt*k1vrt,divspec+0.5*dt*k1div,lyrthkspec+0.5*dt*k1thk)
        k3vrt,k3div,k3thk = \
        self.gettend(vrtspec+0.5*dt*k2vrt,divspec+0.5*dt*k2div,lyrthkspec+0.5*dt*k2thk)
        k4vrt,k4div,k4thk = \
        self.gettend(vrtspec+dt*k3vrt,divspec+dt*k3div,lyrthkspec+dt*k3thk)
        vrtspec += dt*(k1vrt+2.*k2vrt+2.*k3vrt+k4vrt)/6.
        divspec += dt*(k1div+2.*k2div+2.*k3div+k4div)/6.
        lyrthkspec += dt*(k1thk+2.*k2thk+2.*k3thk+k4thk)/6.
        return vrtspec,divspec,lyrthkspec

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap, addcyclic
    import time
    
    # grid, time step info
    nlons = 128  # number of longitudes
    ntrunc = nlons/3 # spectral truncation (for alias-free computations)
    nlats = (nlons/2)+1 # for regular grid.
    gridtype = 'regular'
    #nlats = nlons/2 # for gaussian grid.
    #gridtype = 'gaussian'
    dt = 900 # time step in seconds
    itmax = 8*(86400/dt) # integration length in days
    umax = 50. # jet speed
    jetexp = 10 # parameter controlling jet width

    # create spherical harmonic instance.
    rsphere = 6.37122e6 # earth radius
    sp = Spharmt(nlons,nlats,rsphere,gridtype=gridtype)

    # create model instance using default parameters.
    model = TwoLayer(sp,dt,ntrunc)
    
    # vort, div initial conditions
    psipert = np.zeros((sp.nlat,sp.nlon,2),np.float32)
    psipert[:,:,1] = 5.e6*np.sin((model.lons-np.pi))**12*np.sin(2.*model.lats)**12
    psipert = np.where(model.lons[:,:,np.newaxis] > 0., 0, psipert)
    ug = np.zeros((sp.nlat,sp.nlon,2),np.float32)
    vg = np.zeros((sp.nlat,sp.nlon,2),np.float32)
    ug[:,:,1] = umax*np.sin(2.*model.lats)**jetexp
    vrtspec, divspec = sp.getvrtdivspec(ug,vg,model.ntrunc)
    vrtspec = vrtspec + model.lap*sp.grdtospec(psipert,model.ntrunc)
    vrtg = sp.spectogrd(vrtspec)
    lyrthkspec = model.nlbalance(vrtspec)
    lyrthkg = sp.spectogrd(lyrthkspec)
    print lyrthkg[:,:,0].min(), lyrthkg[:,:,0].max()
    print lyrthkg[:,:,1].min(), lyrthkg[:,:,1].max()
    if lyrthkg.min() < 0:
        raise ValueError('negative layer thickness! adjust jet parameters')

    # time loop.
    time1 = time.clock()
    for ncycle in range(itmax+1):
        t = ncycle*model.dt
        vrtspec, divspec, lyrthkspec = model.rk4step(vrtspec, divspec, lyrthkspec)
        pvg = (0.5*model.zmid/model.omega)*(model.vrt + model.f)/model.lyrthk
        print 't=%6.2f hours: v min/max %6.2f, %6.2f pv min/max %6.2f, %6.2f'%\
        (t/3600.,model.v.min(), model.v.max(), pvg.min(), pvg.max())
    time2 = time.clock()
    print 'CPU time = ',time2-time1
    
    # make a plot of upper layer thickness
    m = Basemap(projection='ortho',lat_0=60,lon_0=180)
    lons1d = model.lons[0,:]*180./np.pi
    lats1d = model.lats[:,0]*180./np.pi
    lyrthk,lons1dx = addcyclic(model.lyrthk[:,:,1],lons1d)
    print 'max/min upper layer thk',lyrthk.min(), lyrthk.max()
    lons, lats = np.meshgrid(lons1dx,lats1d)
    x,y = m(lons,lats)
    m.drawmeridians(np.arange(-180,180,60))
    m.drawparallels(np.arange(-80,81,20))
    CS=m.contourf(x,y,lyrthk,30,cmap=plt.cm.spectral,extend='both')
    m.colorbar()
    plt.title('Upper-Layer Thickness (T%s, hour %6.2f)' % (ntrunc,t/3600.))
    plt.show()
