#!/usr/bin/env python
# rm MANIFEST, use --no-defaults when making sdist
# use "config_fc --fcompiler=gnu95 install" to build and install with
# another fortran compiler (e.g. gfortran).

from numpy.distutils.core  import setup, Extension
import os, sys

srcs_spherepack =  ['src/gaqd.f','src/shses.f','src/shaes.f','src/vhaes.f','src/vhses.f','src/shsgs.f','src/shags.f','src/vhags.f','src/vhsgs.f','src/sphcom.f','src/hrfft.f','src/shaec.f','src/shagc.f','src/shsec.f','src/shsgc.f','src/vhaec.f','src/vhagc.f','src/vhsec.f','src/vhsgc.f','src/ihgeod.f','src/alf.f']

srcs_local = ['src/_spherepack.pyf','src/getlegfunc.f','src/specintrp.f','src/onedtotwod.f','src/onedtotwod_vrtdiv.f','src/twodtooned.f','src/twodtooned_vrtdiv.f','src/multsmoothfact.f','src/lap.f','src/invlap.f']

ext = Extension(name          = '_spherepack',
                sources       = srcs_local+srcs_spherepack)

#havefiles = [os.path.isfile(f) for f in srcs_spherepack]
#
#if havefiles.count(False) and sys.argv[1] not in ['sdist','clean']:
#    sys.stdout.write("""
# SPHEREPACK fortran source files not in src directory.
# The SPHEREPACK license forbids redistribution of the source.
# You can download the tarfile from http://www.scd.ucar.edu/softlib/SPHERE.html
# and copy the *.f files to the src directory, or it can be done
# automatically for you now.
# 
# WARNING: By downloading the SPHEREPACK source files, you are agreeing to
# the terms of the SPHEREPACK license at
# http://www2.cisl.ucar.edu/resources/legacy/spherepack/license\n
# """)
#    download = raw_input('Do you want to download SPHEREPACK now? (yes or no)')
#    if download not in ['Y','y','yes','Yes','YES']:
#        sys.exit(0)
#    import urllib, tarfile
#    tarfname = 'spherepack3.2.tar'
#    URL="https://www2.cisl.ucar.edu/sites/default/files/"+tarfname
#    urllib.urlretrieve(URL,tarfname)
#    if not os.path.isfile(tarfname):
#        raise IOError('Sorry, download failed')
#    tarf = tarfile.open(tarfname)
#    for f in tarf.getnames():
#        ff = os.path.join('src',os.path.basename(f))
#        if ff in srcs_spherepack:
#            sys.stdout.write(f+'\n')
#            mem = tarf.extractfile(f)
#            fout = open(ff,'w')
#            for line in mem.readlines():
#                fout.write(line)
#            fout.close()
#    tarf.close()

if __name__ == "__main__":
    setup(name = 'pyspharm',
          version           = "1.0.9",
          description       = "Python Spherical Harmonic Transform Module",
          author            = "Jeff Whitaker",
          author_email      = "jeffrey.s.whitaker@noaa.gov",
          url               = "http://code.google.com/p/pyspharm",
          ext_modules       = [ext],
	  packages          = ['spharm'],
	  package_dir       = {'spharm':'Lib'}
          )
