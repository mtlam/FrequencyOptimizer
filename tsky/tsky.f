      program testsky
      real l,b,freq,psr_tsky,tsky
      integer narg,iargc
      character*80 tmp

      narg=iargc()

      if (narg.eq.3) then
        call getarg(1,tmp)
	read(tmp,*) l
	call getarg(2,tmp)
	read(tmp,*) b
	call getarg(3,tmp)
	read(tmp,*) freq
        tsky=psr_tsky(l,b,freq)
	write(*,*) 'l',l,' deg, b',b,' deg, freq',freq,
     &  ' MHz, Tsky',tsky,' K'
      else
        write(*,'(''Enter l (deg), b (deg), freq (MHz) > ''$)')
        read(*,*) l,b,freq
        tsky=psr_tsky(l,b,freq)
        write(*,*) 'Sky background temperature is',tsky,' K'
      endif

      end



c---------------------------------------
      real function psr_tsky(l, b, freq)
c---------------------------------------
c
c     Returns tsky for l,b.
c     If the first entry, reads the data from file.
c
      implicit none 
      logical first
      real l, b, nsky(90, 180),freq
      integer j,i,nl,lun
c
c     Check for first entry..
c
      data first / .true. /
      save first,nsky
      if (first) then
c
c       Read in catalogue
c
        call glun(lun)
        open(unit=lun, status='old', file=
     &  './tsky.dat', err=999) 
        read(unit=lun, fmt=1000, end=998) ((nsky(i,j),j = 1, 180)
     &  ,i = 1, 90)
 1000   format(16f5.1)
        close(unit=lun) 
        first = .false.
      end if
c
c     Convert to standard l,b
c
      j = b + 91.5
      if (j .gt. 180) j = 180
      nl = l - 0.5
      if (l .lt. 0.5) nl = 359
      i = (nl / 4) + 1
c
c     Read off tsky from array converting from
c     408MHz n.b assume spectral index of -2.6
c
      psr_tsky = nsky(i,j) * (freq/408.0)**(-2.6)
      return 
c
c     Error messages, prog terminaes
c
  998 write(unit=*, fmt=1010) 
 1010 format(/40h ***** Unexpected end to TSKY file *****)
  999 write(unit=*, fmt=1020) 
 1020 format(/37h ***** Unable to open TSKY file *****)
      stop 
      end



c==============================================================================
C nicked from pgplot
C*GRGLUN -- get a Fortran logical unit number (Sun/Convex-UNIX)
C+
      SUBROUTINE GLUN(LUN)
      INTEGER LUN
C
C Get an unused Fortran logical unit number. 
C Returns a Logical Unit Number that is not currently opened.
C After GRGLUN is called, the unit should be opened to reserve
C the unit number for future calls.  Once a unit is closed, it
C becomes free and another call to GRGLUN could return the same
C number.  Also, GRGLUN will not return a number in the range 1-9
C as older software will often use these units without warning.
C
C Arguments:
C  LUN    : receives the logical unit number, or -1 on error.
C--
C 12-Feb-1989 [AFT/TJP].
C DRL adapted to subroutine GLUN for use with stand-alone software
C 16-Jul-1993 @ JB
c==============================================================================
      INTEGER I
      LOGICAL QOPEN
C---
      DO 10 I=99,10,-1
          INQUIRE (UNIT=I,  OPENED=QOPEN)
          IF (.NOT.QOPEN) THEN
              LUN = I
              RETURN
          END IF
   10 CONTINUE
C none left
      STOP 'RAN OUT OF LUNs!' 
      END
