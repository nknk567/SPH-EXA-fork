#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/findneighbors.hpp"
#include "cstone/primitives/stl.hpp"
#include "cstone/util/array.hpp"

namespace sph
{

//! @brief per-pass statistics of one smoothing-length NR iteration (see computeVeNR)
struct NRPassStats
{
    //! @brief particles whose relative h change was >= the tolerance
    size_t numUnconverged;
    //! @brief particles clamped by the per-iteration up (1.1x) / down (0.5x) step limits
    size_t numCapUp;
    size_t numCapDown;
    /*! @brief particles reset by the convergence-point neighbor-count band check this pass
     *
     * Counted as ballmass == 0 after the pass: recompute signals entering a pass are consumed
     * by its re-seed branch before the band check can fire again (see VeNRPostamble), so
     * post-pass zeros are exactly the resets decided in this pass — each reset is counted once,
     * in the step that decided it.
     */
    size_t numReset;
};

//! @brief compute time-step based on the signal velocity
template<class T1, class T2, class T3>
HOST_DEVICE_FUN auto tsKCourant(T1 maxvsignal, T2 h, T3 c, float Kcour)
{
    using T = std::common_type_t<T1, T2, T3>;
    T v     = maxvsignal > T(0) ? maxvsignal : c;
    assert(h > 0);
    return T(Kcour * h / v);
}

/*! @brief estimate updated smoothing length to bring the neighbor count closer to ng0
 *
 * @tparam T    float or double
 * @param ng0   target neighbor count
 * @param nc    current neighbor count
 * @param h     current smoothing length
 * @return      updated smoothing length
 */
template<class T>
HOST_DEVICE_FUN T updateH(unsigned ng0, unsigned nc, T h)
{
    constexpr T c0  = 1023.0;
    constexpr T exp = 1.0 / 10.0;
    return h * T(0.5) * std::pow(T(1) + c0 * ng0 / T(nc), exp);
}

/*! @brief neighbor-count guard for Newton-Raphson controlled smoothing lengths
 *
 * With NR iterations, h is controlled by the constraint rho * h^3 = ballmassEta(ng0) * m; the
 * neighbor count only guards the neighbor-list capacity, so the bounds are wide and this pass
 * must interfere with the converged NR solution as rarely and as gently as possible: any h
 * change here is undone by the NR iterations pulling h back to its root, and if that pull-back
 * exceeds the list extension @p hExtFactor, the neighbor list built in between misses pairs
 * inside the final support (observed as a steady energy drift in shocks, where the count at
 * the NR root can reach ~2x ng0).
 * The upper bound is the user-set ngmax: it is enforced (not just approached), so
 * count(2h) <= ngmax. The neighbor lists are built with the larger capacity ngmaxExt
 * (see ve_hydro_nr.hpp), sized so that both the plain 2h list and the extended capture shell
 * (radius scaled by @p hExtFactor for completeness under the NR iterations) fit; if the
 * extended search overflows that capacity, the list builder falls back to the plain 2h
 * search for the affected particles. Shrinking h never invalidates the halos discovered for
 * the larger h; growing h is capped at @p hExtFactor per step to stay within the halo search
 * margin.
 */
template<class Tc, class T, class KeyType>
HOST_DEVICE_FUN void updateHIterativeNR(unsigned ng0, unsigned ngmax, float hExtFactor, const cstone::Box<Tc>& box,
                                        const cstone::OctreeNsView<Tc, KeyType>& treeView, cstone::LocalIndex i,
                                        const Tc* __restrict__ x, const Tc* __restrict__ y, const Tc* __restrict__ z,
                                        T* __restrict__ h, unsigned* __restrict__ nc)
{
    constexpr int  maxIteration = 10;
    const unsigned bandMin      = ng0 / 4;
    const unsigned bandMax      = ngmax;

    unsigned ncSph = 1 + findNeighbors(i, x, y, z, h, treeView, box, ngmax);

    if (ncSph < bandMin)
    {
        h[i]  = stl::min(T(hExtFactor) * h[i], updateH(ng0, ncSph, h[i]));
        ncSph = 1 + findNeighbors(i, x, y, z, h, treeView, box, ngmax);
    }
    else
    {
        int iteration = 0;
        while ((ncSph - 1) > bandMax && iteration++ < maxIteration)
        {
            //! gentle shrink targeting just below the threshold, minimizing the NR pull-back
            h[i] *= std::cbrt(T(0.85) * bandMax / T(ncSph - 1));
            ncSph = 1 + findNeighbors(i, x, y, z, h, treeView, box, ngmax);
        }
    }

    nc[i] = ncSph;
}

/*! @brief iterative neighbor-count guard for the smoothing length
 *
 * With per-particle NR constraint targets, corrections made here are transient: the NR
 * iterations pull h back towards the target's root. Whether a correction should stick (target
 * re-seeded at the corrected h) is decided at the NR converging point, not here — see the
 * neighbor-count band check in VeNRPostamble, which resets out-of-band converging points to
 * this guard's output.
 */
template<class Tc, class T, class KeyType>
HOST_DEVICE_FUN void updateHIterative(unsigned ng0, unsigned ngmax, const cstone::Box<Tc>& box,
                                      const cstone::OctreeNsView<Tc, KeyType>& treeView, cstone::LocalIndex i,
                                      const Tc* __restrict__ x, const Tc* __restrict__ y, const Tc* __restrict__ z,
                                      T* __restrict__ h, unsigned* __restrict__ nc)
{
    constexpr int  maxIteration = 10;
    const unsigned ngmin        = ng0 / 2;
    //    const unsigned ngmin = 0.8 * ng0;
    //    if (ngmax > 1.2 * ng0) { ngmax = 1.2 * ng0; }

    unsigned ncSph = 1 + findNeighbors(i, x, y, z, h, treeView, box, ngmax);

    int iteration = 0;
    while ((ngmin > ncSph || (ncSph - 1) > ngmax) && iteration++ < maxIteration)
    {
        h[i]  = updateH(ng0, ncSph, h[i]);
        ncSph = 1 + findNeighbors(i, x, y, z, h, treeView, box, ngmax);
    }

    if ((ncSph - 1) > ngmax)
    {
        T high = h[i];

        h[i]  = updateH(ng0, ncSph, h[i]);
        ncSph = 1 + findNeighbors(i, x, y, z, h, treeView, box, ngmax);
        assert(ncSph <= ng0);

        T        low   = h[i];
        unsigned ncLow = ncSph;
        for (int iteration = 0; iteration < maxIteration; ++iteration)
        {
            h[i]  = (low + high) / T(2);
            ncSph = 1 + findNeighbors(i, x, y, z, h, treeView, box, ngmax);
            if (ncSph == ng0) { break; }
            else if (ncSph < ng0)
            {
                low   = h[i];
                ncLow = ncSph;
            }
            else { high = h[i]; }
        }
        if ((ncSph - 1) > ngmax)
        {
            h[i]  = low;
            ncSph = ncLow;
        }
    }
    assert((ncSph - 1) <= ngmax);

    if (ngmin > ncSph) { ncSph = 1; }

    nc[i] = ncSph;
}

//! @brief sinc(PI/2 * v)
template<typename T>
HOST_DEVICE_FUN inline T wharmonic_std(T v)
{
    if (v == 0.0) { return 1.0; }

    const T Pv = M_PI_2 * v;
    return std::sin(Pv) / Pv;
}

/*! @brief Derivative of sinc(PI/2 * v) w.r to v
 *
 * Unoptimized for clarity as this is only used to construct look-up tables
 */
template<typename T>
HOST_DEVICE_FUN inline T wharmonic_derivative_std(T v)
{
    if (v == 0.0) return 0.0;

    constexpr T piHalf = M_PI_2;
    const T     Pv     = piHalf * v;
    const T     sincv  = std::sin(Pv) / (Pv);

    return sincv * piHalf * ((std::cos(Pv) / std::sin(Pv)) - T(1) / Pv);
}

/*! @brief calculate the artificial viscosity between a pair of two particles
 *
 * @tparam T      float or double
 * @param alpha_i viscosity switch of particle i
 * @param alpha_j viscosity switch of particle j
 * @param c_i     speed of sound particle i
 * @param c_j     speed of sound particle j
 * @param w_ij    relative velocity (v_i - v_j), projected onto the connecting axis (r_i - r_j)
 * @return        the viscosity
 */
template<typename T>
HOST_DEVICE_FUN inline T artificial_viscosity(T alpha_i, T alpha_j, T c_i, T c_j, T w_ij)
{
    // alpha is const for now, but will be different for each particle when using viscosity switching
    constexpr T beta = T(2.0);

    T viscosity_ij = T(0.0);
    if (w_ij < T(0.0))
    {
        T vij_signal = (alpha_i + alpha_j) * T(0.25) * (c_i + c_j) - beta * w_ij;
        viscosity_ij = -vij_signal * w_ij;
    }

    return viscosity_ij;
}

//! @brief symmetric 3x3 matrix-vector product
template<class Tv, class Tm>
HOST_DEVICE_FUN HOST_DEVICE_INLINE util::array<Tv, 3> symv(const util::array<Tm, 6>& mat, const util::array<Tv, 3>& vec)
{
    util::array<Tv, 3> ret;
    ret[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
    ret[1] = mat[3] * vec[1] + mat[4] * vec[2];
    ret[2] = mat[5] * vec[2];
    return ret;
}

} // namespace sph
