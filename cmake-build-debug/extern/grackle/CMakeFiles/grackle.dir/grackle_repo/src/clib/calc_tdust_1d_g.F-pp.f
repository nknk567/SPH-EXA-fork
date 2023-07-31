# 1 "/Users/noah/Documents/SPH/4 Jan 22/SPH-EXA-fork/extern/grackle/grackle_repo/src/clib/calc_tdust_1d_g.F"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/Users/noah/Documents/SPH/4 Jan 22/SPH-EXA-fork/cmake-build-debug//"
# 1 "/Users/noah/Documents/SPH/4 Jan 22/SPH-EXA-fork/extern/grackle/grackle_repo/src/clib/calc_tdust_1d_g.F"

# 1 "/Users/noah/Documents/SPH/4 Jan 22/SPH-EXA-fork/extern/grackle/grackle_repo/src/clib/phys_const.def" 1

# 19 "/Users/noah/Documents/SPH/4 Jan 22/SPH-EXA-fork/extern/grackle/grackle_repo/src/clib/phys_const.def"

# 33 "/Users/noah/Documents/SPH/4 Jan 22/SPH-EXA-fork/extern/grackle/grackle_repo/src/clib/phys_const.def"



# 2 "/Users/noah/Documents/SPH/4 Jan 22/SPH-EXA-fork/extern/grackle/grackle_repo/src/clib/calc_tdust_1d_g.F" 2

!=======================================================================
!////////////////////  SUBROUTINE CALC_TDUST_1D_G  \\\\\\\\\\\\\\\\\\\\

      subroutine calc_tdust_1d_g(
     &     tdust, tgas, nh, gasgr, gamma_isrfa, isrf, itmask,
     &     trad, in, is, ie, j, k)

!  CALCULATE EQUILIBRIUM DUST TEMPERATURE
!
!  written by: Britton Smith
!  date:       February, 2011
!  modified1: 
!
!  PURPOSE:
!    Calculate dust temperature.
!
!  INPUTS:
!     in       - dimension of 1D slice
!
!     tdust    - dust temperature
!
!     tgas     - gas temperature
!     nh       - H number density
!     gasgr    - gas/grain heat transfer rate
!     gamma_isrfa - heating from interstellar radiation field
!     isrf     - interstellar radiation field in Habing units
!
!     trad     - CMB temperature
!
!     is,ie    - start and end indices of active region (zero based)
!     j,k      - indices of 1D slice
!
!     itmask   - iteration mask
!
!  PARAMETERS:
!
!-----------------------------------------------------------------------

      implicit NONE

# 1 "/Users/noah/Documents/SPH/4 Jan 22/SPH-EXA-fork/extern/grackle/grackle_repo/src/clib/grackle_fortran_types.def" 1
!=======================================================================
!
!
! Grackle fortran variable types
!
!
! Copyright (c) 2013, Enzo/Grackle Development Team.
!
! Distributed under the terms of the Enzo Public Licence.
!
! The full license is in the file LICENSE, distributed with this 
! software.
!=======================================================================












      integer, parameter :: RKIND=8





      integer, parameter :: DKIND=8
      integer, parameter :: DIKIND=8
# 43 "/Users/noah/Documents/SPH/4 Jan 22/SPH-EXA-fork/extern/grackle/grackle_repo/src/clib/calc_tdust_1d_g.F" 2

!  Arguments

      integer in, is, ie, j, k

      real*8 tdust(in), tgas(in), nh(in), gasgr(in), isrf(in)
      real*8 gamma_isrfa, trad

!  Iteration mask

      logical itmask(in)

!  Parameters

      real*8 t_subl
      parameter(t_subl = 1.5e3_DKIND) ! grain sublimation temperature
      real*8 radf
      parameter(radf = 4._DKIND * 5.670373d-5)
      real*8 kgr1
      parameter(kgr1 = 4.0e-4_DKIND)
      real*8 tol, bi_tol, minpert, gamma_isrf(in)
      parameter(tol = 1.e-5_DKIND, bi_tol = 1.e-3_DKIND, 
     &     minpert = 1.e-10_DKIND)
      integer itmax, bi_itmax
      parameter(itmax = 50, bi_itmax = 30)

!  Locals

      integer i, iter, c_done, c_total, nm_done

      real*8 pert_i, trad4

!  Slice Locals

      real*8 kgr(in), kgrplus(in), sol(in), solplus(in), 
     &       slope(in), tdplus(in), tdustnow(in), tdustold(in),
     &       pert(in),
     &       bi_t_mid(in), bi_t_high(in)
      logical nm_itmask(in), bi_itmask(in)

!\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\/////////////////////////////////
!=======================================================================

      pert_i = 1.e-3_DKIND

      trad  = max(1._DKIND, trad)
      trad4 = trad**4

!     Set total cells for calculation

      c_done = 0
      nm_done = 0
      c_total = ie - is + 1

!     Set local iteration mask and initial guess

      do i = is+1, ie+1
         if ( itmask(i) ) then
            gamma_isrf(i) = isrf(i) * gamma_isrfa
         endif
      enddo

      do i = is+1, ie+1
         nm_itmask(i) = itmask(i)
         bi_itmask(i) = itmask(i)
         if ( nm_itmask(i) ) then

            if (trad .ge. tgas(i)) then
               tdustnow(i) = trad
               nm_itmask(i) = .false.
               bi_itmask(i) = .false.
               c_done = c_done + 1
               nm_done = nm_done + 1
            else if (tgas(i) .gt. t_subl) then
!     Use bisection if T_gas > grain sublimation temperature.
               nm_itmask(i) = .false.
               nm_done = nm_done + 1
            else
               tdustnow(i) = max(trad,
     &              (gamma_isrf(i) / radf / kgr1)**0.17_DKIND)
               pert(i) = pert_i
            endif

         else
            c_done = c_done + 1
            nm_done = nm_done + 1
         endif
      enddo

!     Iterate to convergence with Newton's method

      do iter = 1, itmax

!     Loop over slice

         do i = is+1, ie+1
            if ( nm_itmask(i) ) then

               tdplus(i) = max(1.e-3_DKIND, ((1._DKIND + pert(i))
     &              * tdustnow(i)))

            endif
         enddo

!     Calculate grain opacities

         call calc_kappa_gr_g(tdustnow, kgr, nm_itmask,
     &        in, is, ie, t_subl)

         call calc_kappa_gr_g(tdplus, kgrplus, nm_itmask,
     &        in, is, ie, t_subl)

!     Calculate heating/cooling balance

         call calc_gr_balance_g(tdustnow, tgas, kgr, trad4, gasgr,
     &        gamma_isrf, nh, nm_itmask, sol, in, is, ie)

         call calc_gr_balance_g(tdplus, tgas, kgrplus, trad4, gasgr,
     &        gamma_isrf, nh, nm_itmask, solplus, in, is, ie)

         do i = is+1, ie+1
            if ( nm_itmask(i) ) then

!     Use Newton's method to solve for Tdust

               slope(i) = (solplus(i) - sol(i)) / 
     &              (pert(i) * tdustnow(i))

               tdustold(i) = tdustnow(i)
               tdustnow(i) = tdustnow(i) - (sol(i) / slope(i))

               pert(i) = max(min(pert(i), 
     &              (0.5_DKIND * abs(tdustnow(i) - tdustold(i)) / 
     &              tdustnow(i))), minpert)

!     If negative solution calculated, give up and wait for bisection step.
               if (tdustnow(i) .lt. trad) then
                  nm_itmask(i) = .false.
                  nm_done = nm_done + 1
!     Check for convergence of solution
               else if (abs(sol(i) / solplus(i)) .lt. tol) then
                  nm_itmask(i) = .false.
                  c_done = c_done + 1
                  bi_itmask(i) = .false.
                  nm_done = nm_done + 1
               endif

!     if ( nm_itmask(i) )
            endif

!     End loop over slice
         enddo

!     Check for all cells converged
         if (c_done .ge. c_total) go to 666

!     Check for all cells done with Newton method
!     This includes attempts where a negative solution was found
         if (nm_done .ge. c_total) go to 555

!     End iteration loop for Newton's method
      enddo

 555  continue

!     If iteration count exceeded, try once more with bisection
      if (c_done .lt. c_total) then
         do i = is+1, ie+1
            if ( bi_itmask(i) ) then
               tdustnow(i)  = trad
               bi_t_high(i) = tgas(i)
            endif
         enddo

         do iter = 1, bi_itmax

            do i = is+1, ie+1
               if ( bi_itmask(i) ) then

                  bi_t_mid(i) = 0.5_DKIND * (tdustnow(i) + bi_t_high(i))
                  if (iter .eq. 1) then
                     bi_t_mid(i) = min(bi_t_mid(i), t_subl)
                  endif

               endif
            enddo

            call calc_kappa_gr_g(bi_t_mid, kgr, bi_itmask,
     &           in, is, ie, t_subl)

            call calc_gr_balance_g(bi_t_mid, tgas, kgr, trad4, gasgr,
     &           gamma_isrf, nh, bi_itmask, sol, in, is, ie)

            do i = is+1, ie+1
               if ( bi_itmask(i) ) then

                  if (sol(i) .gt. 0._DKIND) then
                     tdustnow(i) = bi_t_mid(i)
                  else
                     bi_t_high(i) = bi_t_mid(i)
                  endif

                  if ((abs(bi_t_high(i) - tdustnow(i)) / tdustnow(i)) 
     &                 .le. bi_tol) then
                     bi_itmask(i) = .false.
                     c_done = c_done + 1
                  endif

!     Check for all cells converged
                  if (c_done .ge. c_total) go to 666

!     if ( bi_itmask(i) )
               endif

!     End loop over slice
            enddo

!     End iteration loop for bisection
         enddo

!     If iteration count exceeded with bisection, end of the line.
         if (iter .gt. itmax) then



            write(6,*) 'CALC_TDUST_1D_G failed using both methods for ',
     &           (c_total - c_done), 'cells.'



c            ERROR_MESSAGE
         endif

!     if (iter .gt. itmax) then
      endif

 666  continue

!     Copy values back to thrown slice
      do i = is+1, ie+1
         if ( itmask(i) ) then

!     Check for bad solutions
            if (tdustnow(i) .lt. 0._DKIND) then



               write(6, *) 'CALC_TDUST_1D_G Newton method - ',
     &              'T_dust < 0: i = ', i, 'j = ', j,
     &              'k = ', k, 'nh = ', nh(i), 
     &              't_gas = ', tgas(i), 't_rad = ', trad,
     &              't_dust = ', tdustnow(i)



c               ERROR_MESSAGE
            endif

            tdust(i) = tdustnow(i)
         endif
      enddo

      return
      end

!=======================================================================
!////////////////////  SUBROUTINE CALC_KAPPA_GR_G  \\\\\\\\\\\\\\\\\\\\

      subroutine calc_kappa_gr_g(
     &     tdust, kgr, itmask, in, is, ie, t_subl)

!  CALCULATE GRAIN PLANK MEAN OPACITY
!
!  written by: Britton Smith
!  date:       September, 2011
!  modified1: 
!
!  PURPOSE:
!    Calculate grain plank mean opacity
!
!  INPUTS:
!     in       - i dimension of 3D fields
!
!     tdust    - dust temperature
!
!     is,ie    - start and end indices of active region (zero based)
!
!     itmask   - iteration mask
!
!     t_subl   - grain sublimation temperature
!
!  OUTPUTS:
!     kgr      - opacities
!
!  PARAMETERS:
!
!-----------------------------------------------------------------------

      implicit NONE

# 1 "/Users/noah/Documents/SPH/4 Jan 22/SPH-EXA-fork/extern/grackle/grackle_repo/src/clib/grackle_fortran_types.def" 1
!=======================================================================
!
!
! Grackle fortran variable types
!
!
! Copyright (c) 2013, Enzo/Grackle Development Team.
!
! Distributed under the terms of the Enzo Public Licence.
!
! The full license is in the file LICENSE, distributed with this 
! software.
!=======================================================================












      integer, parameter :: RKIND=8





      integer, parameter :: DKIND=8
      integer, parameter :: DIKIND=8
# 343 "/Users/noah/Documents/SPH/4 Jan 22/SPH-EXA-fork/extern/grackle/grackle_repo/src/clib/calc_tdust_1d_g.F" 2

!  Arguments

      integer in, is, ie
      real*8 t_subl
      real*8 tdust(in)

!  Iteration mask

      logical itmask(in)

!  Parameters

      real*8 kgr1, kgr200
      parameter(kgr1 = 4.0e-4_DKIND, kgr200 = 16.0_DKIND)

!  Locals

      integer i

!  Slice Locals

      real*8 kgr(in)

!\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\/////////////////////////////////
!=======================================================================

      do i = is+1, ie+1
         if ( itmask(i) ) then

!     Temperature dependence from Dopcke et al. (2011).
!     Normalized to Omukai (2000).

            if (tdust(i) .lt. 200._DKIND) then
               kgr(i) = kgr1 * tdust(i)**2
            else if (tdust(i) .lt. t_subl) then
               kgr(i) = kgr200
            else
               kgr(i) = max(1.d-20, 
     &              (kgr200 * (tdust(i) / 1.5e3_DKIND)**(-12)))
            endif

         endif
      enddo

      return
      end

!=======================================================================
!////////////////////  SUBROUTINE CALC_GR_BALANCE  \\\\\\\\\\\\\\\\\\\\

      subroutine calc_gr_balance_g(
     &     tdust, tgas, kgr, trad4, gasgr, gamma_isrf, nh,
     &     itmask, sol, in, is, ie)

!  CALCULATE GRAIN HEAT BALANCE
!
!  written by: Britton Smith
!  date:       September, 2019
!  modified1:
!
!  PURPOSE:
!    Calculate grain heating/cooling balance
!
!  INPUTS:
!     in       - i dimension of 3D fields
!
!     tdust    - dust temperature
!     tgas     - gas temperature
!     kgr      - grain opacity
!     trad4    - CMB temperature to 4th power
!     gasgr    - gas/grain heat transfer rate
!     gamma_isrf - heating from interstellar radiation field
!     nh       - hydrogen number density
!
!     is,ie    - start and end indices of active region (zero based)
!
!     itmask   - iteration mask
!
!
!  OUTPUTS:
!     sol      - heating/cooling balance (heating - cooling)
!
!  PARAMETERS:
!
!-----------------------------------------------------------------------

      implicit NONE

# 1 "/Users/noah/Documents/SPH/4 Jan 22/SPH-EXA-fork/extern/grackle/grackle_repo/src/clib/grackle_fortran_types.def" 1
!=======================================================================
!
!
! Grackle fortran variable types
!
!
! Copyright (c) 2013, Enzo/Grackle Development Team.
!
! Distributed under the terms of the Enzo Public Licence.
!
! The full license is in the file LICENSE, distributed with this 
! software.
!=======================================================================












      integer, parameter :: RKIND=8





      integer, parameter :: DKIND=8
      integer, parameter :: DIKIND=8
# 432 "/Users/noah/Documents/SPH/4 Jan 22/SPH-EXA-fork/extern/grackle/grackle_repo/src/clib/calc_tdust_1d_g.F" 2

!  Arguments

      integer in, is, ie
      real*8 tdust(in), tgas(in), kgr(in), gasgr(in), nh(in), trad4,
     &     gamma_isrf(in)

!  Iteration mask

      logical itmask(in)

!  Parameters

      real*8 radf
      parameter(radf = 4._DKIND * 5.670373d-5)

!  Locals

      integer i

!  Slice Locals

      real*8 sol(in)

!\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\/////////////////////////////////
!=======================================================================

      do i = is+1, ie+1
         if ( itmask(i) ) then

            sol(i) = gamma_isrf(i) + radf * kgr(i) *
     &           (trad4 - tdust(i)**4) +
     &           (gasgr(i) * nh(i) *
     &           (tgas(i) - tdust(i)))

         endif
      enddo

      return
      end
