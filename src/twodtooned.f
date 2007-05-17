      subroutine twodtooned(dataspec,a,b,nlat,ntrunc,nt)
      integer nlat,ntrunc,nt,nmstrt,i
      complex dataspec((ntrunc+1)*(ntrunc+2)/2,nt)
      real a(nlat,nlat,nt),b(nlat,nlat,nt)

      scale = 0.5

      nmstrt = 0
      do i=1,nt
      do m=1,ntrunc+1
      do n=m,ntrunc+1
         nm = nmstrt + n - m + 1
         dataspec(nm,i) = scale*cmplx(a(m,n,i),b(m,n,i))
      enddo
      nmstrt = nmstrt + ntrunc - m + 2
      enddo
      enddo

      return
      end
