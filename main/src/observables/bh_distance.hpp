//
// Created by Noah Kubli on 10.07.2026.
//

#ifndef SPHEXA_BH_DISTANCE_HPP
#define SPHEXA_BH_DISTANCE_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <mpi.h>
#include "bh_distance_gpu.hpp"

namespace bh_merger
{
template<class Dataset>
double computeBhDistance(size_t startIndex, size_t endIndex, Dataset& d, MPI_Comm comm, int rank, uint64_t id_0,
                         uint64_t id_1)
{
    std::array<double, 6> pos_packed;
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        pos_packed =
            computeBhPositionsGPU(startIndex, endIndex, rawPtr(d.x), rawPtr(d.y), rawPtr(d.z), rawPtr(d.id), id_0, id_1);
    }
    else
    {
        auto it0 = std::find(d.id.begin() + startIndex, d.id.begin() + endIndex, id_0);
        if (it0 != d.id.end())
        {
            size_t i      = it0 - (d.id.begin() + startIndex);
            pos_packed[0] = d.x[i];
            pos_packed[1] = d.y[i];
            pos_packed[2] = d.z[i];
        }
        auto it1 = std::find(d.id.begin() + startIndex, d.id.begin() + endIndex, id_1);
        if (it1 != d.id.end())
        {
            size_t i      = it1 - (d.id.begin() + startIndex);
            pos_packed[3] = d.x[i];
            pos_packed[4] = d.y[i];
            pos_packed[5] = d.z[i];
        }
    }
    std::array<double, 6> pos_global{};

    MPI_Reduce(pos_packed.data(), pos_global.data(), 6, MPI_DOUBLE, MPI_SUM, 0, comm);
    if (rank == 0)
    {
        const double dx = pos_global[0] - pos_global[3];
        const double dy = pos_global[1] - pos_global[4];
        const double dz = pos_global[2] - pos_global[5];

        const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        return distance;
    }
    else { return 0.; }
}

} // namespace bh_merger

#endif // SPHEXA_BH_DISTANCE_HPP
