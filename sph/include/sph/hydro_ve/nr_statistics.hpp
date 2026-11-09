//
// Created by Noah Kubli on 09.11.2026.
//

#pragma once

namespace sph
{

//! @brief per-pass statistics of one smoothing-length NR iteration (see computeVeNR)
struct NRStatistics
{
    //! @brief number of particles whose relative h change was >= the tolerance
    size_t n_unconverged;

    //! @brief number of particles with ballmass reset
    size_t n_ballmass_reset;
};
} // namespace sph
