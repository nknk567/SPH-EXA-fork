#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/box.hpp"

#include "sph/kernels.hpp"
#include "sph/table_lookup.hpp"

namespace sph
{

template<size_t stride = 1, class Tc, class Tm, class T>
HOST_DEVICE_FUN inline void IADJLoopSTD(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box,
                                        const cstone::LocalIndex* neighbors, unsigned neighborsCount, const Tc* x,
                                        const Tc* y, const Tc* z, const T* h, const Tm* m, const T* rho, const T* wh,
                                        const T* /*whd*/, T* c11, T* c12, T* c13, T* c22, T* c23, T* c33)
{
    T tau11 = 0.0, tau12 = 0.0, tau13 = 0.0, tau22 = 0.0, tau23 = 0.0, tau33 = 0.0;

    auto xi = x[i];
    auto yi = y[i];
    auto zi = z[i];

    auto hi    = h[i];
    auto hiInv = T(1) / hi;

    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        T rx = (xi - x[j]);
        T ry = (yi - y[j]);
        T rz = (zi - z[j]);

        applyPBC(box, T(2) * hi, rx, ry, rz);

        T dist = std::sqrt(rx * rx + ry * ry + rz * rz);

        T vloc = dist * hiInv;
        T w    = lt::lookup(wh, vloc);

        T mj_roj_w = m[j] / rho[j] * w;

        tau11 += rx * rx * mj_roj_w;
        tau12 += rx * ry * mj_roj_w;
        tau13 += rx * rz * mj_roj_w;
        tau22 += ry * ry * mj_roj_w;
        tau23 += ry * rz * mj_roj_w;
        tau33 += rz * rz * mj_roj_w;
    }

    auto getExp    = [](T val) { return (val == T(0) ? 0 : std::ilogb(val)); };
    int  tauExpSum = getExp(tau11) + getExp(tau12) + getExp(tau13) + getExp(tau22) + getExp(tau23) + getExp(tau33);
    // normalize with 2^-averageTauExponent, ldexp(a, b) == a * 2^b
    T normalization = std::ldexp(T(1), -tauExpSum / 6);

    tau11 *= normalization;
    tau12 *= normalization;
    tau13 *= normalization;
    tau22 *= normalization;
    tau23 *= normalization;
    tau33 *= normalization;

    T det = tau11 * tau22 * tau33 + T(2) * tau12 * tau23 * tau13 - tau11 * tau23 * tau23 - tau22 * tau13 * tau13 -
            tau33 * tau12 * tau12;

    // Note normalization factor: cij have units of 1/tau because det is proportional to tau^3 so we have to
    // divide by K/h^3.
    T factor = normalization * (hi * hi * hi) / (det * K);

    c11[i] = (tau22 * tau33 - tau23 * tau23) * factor;
    c12[i] = (tau13 * tau23 - tau33 * tau12) * factor;
    c13[i] = (tau12 * tau23 - tau22 * tau13) * factor;
    c22[i] = (tau11 * tau33 - tau13 * tau13) * factor;
    c23[i] = (tau13 * tau12 - tau11 * tau23) * factor;
    c33[i] = (tau11 * tau22 - tau12 * tau12) * factor;
    if (neighborsCount > 25 && neighborsCount + 1 < 150) { c11[i] = c12[i] = c13[i] = c22[i] = c23[i] = c33[i] = 0.; }
}

template<size_t stride = 1, typename Tc, class T>
HOST_DEVICE_FUN inline void
divV_curlVJLoopSTD(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box, const cstone::LocalIndex* neighbors,
                   unsigned neighborsCount, const Tc* x, const Tc* y, const Tc* z, const T* vx, const T* vy,
                   const T* vz, const T* h, const T* c11, const T* c12, const T* c13, const T* c22, const T* c23,
                   const T* c33, const T* wh, const T* /*whd*/, const T* m, const T* rho, T* divv)
{
    auto xi   = x[i];
    auto yi   = y[i];
    auto zi   = z[i];
    auto vxi  = vx[i];
    auto vyi  = vy[i];
    auto vzi  = vz[i];
    auto hi   = h[i];
    auto rhoi = rho[i];

    auto hiInv  = T(1) / hi;
    auto hiInv3 = hiInv * hiInv * hiInv;

    // the 3 components of these vectors will be the derivatives in x,y,z directions
    cstone::Vec3<T> dVxi{0., 0., 0.}, dVyi{0., 0., 0.}, dVzi{0., 0., 0.};

    auto c11i = c11[i];
    auto c12i = c12[i];
    auto c13i = c13[i];
    auto c22i = c22[i];
    auto c23i = c23[i];
    auto c33i = c33[i];

    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        T rx = xi - x[j];
        T ry = yi - y[j];
        T rz = zi - z[j];

        applyPBC(box, T(2) * hi, rx, ry, rz);

        T r2   = rx * rx + ry * ry + rz * rz;
        T dist = std::sqrt(r2);

        T vx_ji = vx[j] - vxi;
        T vy_ji = vy[j] - vyi;
        T vz_ji = vz[j] - vzi;

        T v1 = dist * hiInv;
        T Wi = lt::lookup(wh, v1);

        cstone::Vec3<T> termA;
        termA[0] = -(c11i * rx + c12i * ry + c13i * rz) * Wi;
        termA[1] = -(c12i * rx + c22i * ry + c23i * rz) * Wi;
        termA[2] = -(c13i * rx + c23i * ry + c33i * rz) * Wi;

        //        T xmassj = xm[j];
        T vol_j = m[j] / rho[j];
        dVxi += (vx_ji * vol_j) * termA;
        dVyi += (vy_ji * vol_j) * termA;
        dVzi += (vz_ji * vol_j) * termA;
    }

    T norm_kxi = K * hiInv3;
    divv[i]    = norm_kxi * (dVxi[0] + dVyi[1] + dVzi[2]);

    //    if (curlv != nullptr)
    //    {
    //        cstone::Vec3<T> curlV{dVzi[1] - dVyi[2], dVxi[2] - dVzi[0], dVyi[0] - dVxi[1]};
    //        curlv[i] = norm_kxi * std::sqrt(norm2(curlV));
    //    }
    //
    //    if (doGradV)
    //    {
    //        dV11[i] = norm_kxi * dVxi[0];
    //        dV12[i] = norm_kxi * (dVxi[1] + dVyi[0]);
    //        dV13[i] = norm_kxi * (dVxi[2] + dVzi[0]);
    //        dV22[i] = norm_kxi * dVyi[1];
    //        dV23[i] = norm_kxi * (dVyi[2] + dVzi[1]);
    //        dV33[i] = norm_kxi * dVzi[2];
    //    }
}
} // namespace sph
