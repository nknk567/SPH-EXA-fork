#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/findneighbors.hpp"
#include "cstone/primitives/stl.hpp"
#include "cstone/util/array.hpp"

namespace sph
{

/*! @brief NR mode: extension factor for the halo search and the neighbor-list capture radius
 *
 * Halos and neighbor lists are built before the NR iterations move h; extending both search radii
 * by this factor keeps them complete as long as h grows by less than this factor within a step
 * (the interaction kernels always cut at the live 2h). It also caps the per-step growth of h by
 * the neighbor-count management, which runs after halo discovery, so that nudged particles stay
 * within the halo margin as well.
 */
constexpr float hNRExtFactor = 1.05;

/*! @brief NR mode: relative smoothing-length convergence tolerance of the NR iterations
 *
 * The iterations of a step stop as soon as the largest relative h change of a locally owned
 * particle falls below this value (or when the --nrIter cap is reached).
 */
constexpr float hNRTol = 1e-4;

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
 * exceeds the list extension hNRExtFactor, the neighbor list built in between misses pairs
 * inside the final support (observed as a steady energy drift in shocks, where the count at
 * the NR root can reach ~2x ng0).
 * The upper bound is the user-set ngmax: it is enforced (not just approached), so
 * count(2h) <= ngmax. The neighbor lists are built with the larger capacity ngmaxExt
 * (see ve_hydro.hpp), sized so that both the plain 2h list and the extended capture shell
 * (radius scaled by hNRExtFactor for completeness under the NR iterations) fit; if the
 * extended search overflows that capacity, the list builder falls back to the plain 2h
 * search for the affected particles. Shrinking h never invalidates the halos discovered for
 * the larger h; growing h is capped at hNRExtFactor per step to stay within the halo search
 * margin.
 */
template<class Tc, class T, class KeyType>
HOST_DEVICE_FUN void updateHIterativeNR(unsigned ng0, unsigned ngmax, const cstone::Box<Tc>& box,
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
        h[i]  = stl::min(T(hNRExtFactor) * h[i], updateH(ng0, ncSph, h[i]));
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

template<class Tc, class T, class KeyType>
HOST_DEVICE_FUN void updateHIterative(unsigned ng0, unsigned ngmax, const cstone::Box<Tc>& box,
                                      const cstone::OctreeNsView<Tc, KeyType>& treeView, cstone::LocalIndex i,
                                      const Tc* __restrict__ x, const Tc* __restrict__ y, const Tc* __restrict__ z,
                                      T* __restrict__ h, unsigned* __restrict__ nc)
{
    constexpr int  maxIteration = 10;
    const unsigned ngmin        = ng0 / 4;
    //    const unsigned ngmin = 0.8 * ng0;
    //    if (ngmax > 1.2 * ng0) { ngmax = 1.2 * ng0; }

    unsigned ncSph = 1 + findNeighbors(i, x, y, z, h, treeView, box, ngmax);

    int iteration = 0;
    while ((ngmin > ncSph || (ncSph - 1) > ngmax) && iteration++ < maxIteration)
    {
        h[i]  = updateH(ng0, ncSph, h[i]);
        ncSph = 1 + findNeighbors(i, x, y, z, h, treeView, box, ngmax);
    }

    if (ngmin > ncSph || (ncSph - 1) > ngmax)
    {
        ncSph = 1;
        h[i]  = cstone::invalidateH(h[i]);
    }

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
