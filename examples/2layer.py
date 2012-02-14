import numpy as np
from spharm import Spharmt, getspecindx, gaussian_lats_wts

# two-layer baroclinic primitive equation model

class TwoLayer(object):

    def __init__(self,sp,theta1,theta2,grav,omega,cp,rsphere,zmid,ztop,efold,ndiss,ntrunc,dt):
        # setup model parameters
        self.theta1 = theta1
        self.theta2 = theta2
        self.delth = delth
        self.grav = grav
        self.omega = omega
        self.cp = cp
        self.rsphere = rsphere
        self.zmid = zmid
        self.ztop = ztop
        self.efold = efold
        self.ndiss = ndiss
        self.sp = sp
        self.ntrunc = ntrunc
        self.dt = dt
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
        self.lap = -totwavenum/rsphere**2
        self.ilap = np.zeros(self.lap.shape, np.float32)
        self.ilap[1:,:] = 1./self.lap[1:,:]
        # hyperdiffusion operator
        self.hyperdiff = -(1./efold)*(totwavenum/totwavenum[-1])**(ndiss/2)

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
        lyrthkspec[:,0] = mspec[:,0]/self.theta1
        lyrthkspec[:,1] = (mspec[:,1]-mspec[:,0])/self.delth
        lyrthkspec[:,0] = lyrthkspec[:,0] - lyrthkspec[:,1]
        exnftop = self.cp - (self.grav*self.ztop/self.theta1)
        exnfmid = self.cp - (self.grav*self.zmid/self.theta1)
        lyrthkspec[0,0] = (2./np.sqrt(2.))*(self.cp - exnfmid)
        lyrthkspec[0,1] = (2./np.sqrt(2.))*(exnfmid - exnftop)
        lyrthkspec = (self.theta1/self.grav)*lyrthkspec # convert from exner function to height units (m)
        return lyrthkspec

    def gettend(self,vrtspec,divspec,lyrthkspec):
        # compute tendencies.
        vrtg = self.sp.spectogrd(vrtspec)
        ug,vg = self.sp.getuv(vrtspec,divspec)
        lyrthkg = self.sp.spectogrd(lyrthkspec)
        tmpg1 = ug*(vrtg+self.f); tmpg2 = vg*(vrtg+self.f)
        ddivdtspec, dvrtdtspec = self.sp.getvrtdivspec(tmpg1,tmpg2,self.ntrunc)
        dvrtdtspec *= -1
        dvrtdtspec += self.hyperdiff*vrtspec
        tmpg1 = ug*lyrthkg; tmpg2 = vg*lyrthkg
        tmpspec, dlyrthkdtspec = self.sp.getvrtdivspec(tmpg1,tmpg2,self.ntrunc)
        dlyrthkdtspec *= -1
        mstrm = np.empty((self.sp.nlat,self.sp.nlon,2),np.float32)
        mstrm[:,:,0] = self.grav*(lyrthkg[:,:,0] + lyrthkg[:,:,1]) 
        mstrm[:,:,1] = mstrm[:,:,0] + (self.grav*self.delth/self.theta1)*lyrthkg[:,:,1] 
        ddivdtspec += -self.lap*self.sp.grdtospec(mstrm+0.5*(ug**2+vg**2),self.ntrunc) 
        ddivdtspec += self.hyperdiff*divspec
        return dvrtdtspec,ddivdtspec,dlyrthkdtspec

    def rk4step(self,vrtspec,divspec,lyrthkspec):
        # update state using 4th order runge-kutta time step
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
    dt = 240 # time step in seconds
    itmax = 7*(86400/dt) # integration length in days
    
    # parameters for test
    rsphere = 6.37122e6 # earth radius
    omega = 7.292e-5 # rotation rate
    grav = 9.80616 # gravity
    cp = 1004.
    theta1 = 300. ; theta2 = 330
    delth = theta2-theta1
    zmid = 5.e3
    ztop = 15.e3
    efold = 3.*3600. # efolding timescale at ntrunc for hyperdiffusion
    ndiss = 8 # order for hyperdiffusion
    umax = 30. # jet speed
    jetexp = 4 # parameter controlling jet width

    # create spherical harmonic instance.
    sp = Spharmt(nlons,nlats,rsphere,gridtype=gridtype)

    # create model instance.
    model =\
    TwoLayer(sp,theta1,theta2,grav,omega,cp,rsphere,zmid,ztop,efold,ndiss,ntrunc,dt)
    
    # vort, div initial conditions
    psipert = np.zeros((sp.nlat,sp.nlon,2),np.float32)
    psipert[:,:,1] = 5.e6*np.sin((model.lons-np.pi))**12*np.sin(2.*model.lats)**12
    psipert = np.where(model.lons[:,:,np.newaxis] > 0., 0, psipert)
    psipert = np.where(model.lats[:,:,np.newaxis] < 0., psipert, -psipert)
    ug = np.zeros((sp.nlat,sp.nlon,2),np.float32)
    vg = np.zeros((sp.nlat,sp.nlon,2),np.float32)
    ug[:,:,1] = umax*np.sin(2.*model.lats)**jetexp
    vrtspec, divspec = sp.getvrtdivspec(ug,vg,model.ntrunc)
    vrtspec = vrtspec + model.lap*sp.grdtospec(psipert,model.ntrunc)
    lyrthkspec = model.nlbalance(vrtspec)
    lyrthkg = sp.spectogrd(lyrthkspec)
    print lyrthkg[:,:,0].min(), lyrthkg[:,:,0].max()
    print lyrthkg[:,:,1].min(), lyrthkg[:,:,1].max()
    if lyrthkg.min() < 0:
        raise ValueError('negative layer thickness! adjust jet parameters')
    
    
    # to double check, recompute zonal wind from layer thickness.
    #s = np.sin(model.lats); c = np.cos(model.lats)
    #if sp.gridtype == 'regular': s[sp.nlat/2,:]=1
    #ct = c/s
    #mstrm = model.grav*(lyrthkg[:,:,0] + lyrthkg[:,:,1]) +\
    #(model.grav*model.delth/model.theta1)*lyrthkg[:,:,1] 
    #mspec = sp.grdtospec(mstrm)
    #mgradx, mgrady = sp.getgrad(mspec)
    #rad = (model.rsphere*model.omega*c)**2 - ct*model.rsphere*mgrady
    #u = -model.rsphere*model.omega*c + np.sqrt(rad) 
    #print u.min(),u.max()
    #print ug.min(), ug.max()
    #plt.plot(model.lats[:,0]*180./np.pi,u[:,0],'r-')
    #plt.plot(model.lats[:,0]*180./np.pi,ug[:,0,1],'b-')
    #plt.show()
    #raise SystemExit
    
    # time loop.
    time1 = time.clock()
    for ncycle in range(itmax+1):
        t = ncycle*model.dt
        ug,vg = sp.getuv(vrtspec,divspec)
        print 't=%6.2f hours: min/max %6.2f, %6.2f' % (t/3600.,vg.min(), vg.max())
        vrtspec, divspec, lyrthkspec = model.rk4step(vrtspec, divspec, lyrthkspec)
    time2 = time.clock()
    print 'CPU time = ',time2-time1
    vrtg = sp.spectogrd(vrtspec)
    ug,vg = sp.getuv(vrtspec,divspec)
    lyrthkg = sp.spectogrd(lyrthkspec)
    pvg = (0.5*model.zmid/model.omega)*(vrtg + model.f)/lyrthkg
    
    # make a orthographic plot of potential vorticity.
    #m = Basemap(projection='moll',lat_0=0,lon_0=0)
    #m = Basemap(projection='ortho',lon_0=-90,lat_0=40)
    m = Basemap(projection='npaeqd',boundinglat=0,lon_0=0,round=True)
    # dimensionless upper layer PV
    lons1d = model.lons[0,:]
    lats1d = model.lats[:,0]
    pvg,lons1d = addcyclic(pvg[:,:,1],lons1d*180./np.pi)
    print 'max/min PV',pvg.min(), pvg.max()
    lons, lats = np.meshgrid(lons1d,lats1d*180./np.pi)
    x,y = m(lons,lats)
    levs = np.arange(0.,4,0.02)
    m.drawmeridians(np.arange(-180,180,60))
    m.drawparallels(np.arange(-80,81,20))
    CS=m.contourf(x,y,pvg,levs,cmap=plt.cm.spectral,extend='both')
    m.colorbar()
    plt.title('PV (T%s with hyperdiffusion, hour %6.2f)' % (ntrunc,t/3600.))
    plt.show()
