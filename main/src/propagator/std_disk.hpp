
/*! @file
 * @brief A Propagator class for Protoplanetary disk
 *
 * @author Noah Kubli
 */

#pragma once

#include <cstdio>

#include "cstone/fields/field_get.hpp"
#include "io/arg_parser.hpp"
#include "ipropagator.hpp"
#include "std_hydro.hpp"
#include "sph/particles_data.hpp"
#include "sph/sph.hpp"

#include "accretion.hpp"
#include "beta_cooling.hpp"
#include "central_force.hpp"
#include "exchange_star_position.hpp"
#include "star_data.hpp"

namespace sphexa
{

using namespace sph;
using util::FieldList;

template<class DomainType, class DataType>
class DiskProp : public HydroProp<DomainType, DataType>
{
protected:
    using Base = HydroProp<DomainType, DataType>;
    using Base::timer;

    using T = typename DataType::RealType;

    //    disk::StarData star;

public:
    DiskProp(std::ostream& output, size_t rank, const InitSettings& settings)
        : Base(output, rank)
    {
    }

    void load(const std::string& initCond, IFileReader* reader) override
    {
        // Read star position from hdf5 File
        //        std::string path = removeModifiers(initCond);
        //        if (std::filesystem::exists(path))
        //        {
        //            int snapshotIndex = numberAfterSign(initCond, ":");
        //            reader->setStep(path, snapshotIndex, FileMode::independent);
        //            star.loadOrStoreAttributes(reader);
        //            reader->closeStep();
        //            std::printf("star position: %lf\t%lf\t%lf\n", star.position[0], star.position[1],
        //            star.position[2]); std::printf("star mass: %lf\n", star.m);
        //        }
    }
    //    void save(IFileWriter* writer) override { star.loadOrStoreAttributes(writer); }
    void activateFields(DataType& simData) override
    {
        simData.star.active = true;
        Base::activateFields(simData);
    }
    void computeForces(DomainType& domain, DataType& simData) override
    {
        Base::computeForces(domain, simData);

        auto&        d     = simData.hydro;
        const size_t first = domain.startIndex();
        const size_t last  = domain.endIndex();

        disk::betaCooling(first, last, d, simData.star);
        timer.step("betaCooling");

        disk::computeCentralForce(first, last, d, simData.star);
        timer.step("computeCentralForce");
        //
        //        transferToHost(d, first, last, {"nb_it_stat"});
        //        std::array<size_t, 9> histogram{};
        //        for (size_t i = first; i < last; i++)
        //        {
        //            size_t bin = (d.nb_it_stat[i] >= histogram.size() ? histogram.size() - 1 : d.nb_it_stat[i]);
        //            histogram[bin]++;
        //        }
        //
        //        MPI_Allreduce(MPI_IN_PLACE, histogram.data(), histogram.size(), MpiType<size_t>{}, MPI_SUM,
        //        MPI_COMM_WORLD);
        //
        //        if (Base::rank_ == 0)
        //        {
        //            printf("Neighbour iterations");
        //            for (size_t i = 0; i < histogram.size(); i++)
        //            {
        //                printf("ncIt: %zu, nPart: %zu\n", i, histogram[i]);
        //            }
        //        }
    }

    void integrate(DomainType& domain, DataType& simData) override
    {
        const size_t first = domain.startIndex();
        const size_t last  = domain.endIndex();
        auto&        d     = simData.hydro;
        auto&        star  = simData.star;

        //        d.minDtRho = rhoTimestep(first, last, d);

        disk::duTimestep(first, last, d, star);
        timer.step("duTimestep");

        computeTimestep(first, last, d, star.t_du);
        timer.step("Timestep");
        //        double minDtAcc           = accelerationTimestep(first, last, d);
        //        double minDtAccGlobal     = INFINITY;
        //        double minDtCourantGlobal = INFINITY;
        //        double minDtRhoGlobal     = INFINITY;
        //        double minDtUGlobal       = INFINITY;
        //        MPI_Allreduce(&minDtAcc, &minDtAccGlobal, 1, MpiType<T>{}, MPI_MIN, MPI_COMM_WORLD);
        //        MPI_Allreduce(&d.minDtCourant, &minDtCourantGlobal, 1, MpiType<T>{}, MPI_MIN, MPI_COMM_WORLD);
        //        MPI_Allreduce(&d.minDtRho, &minDtRhoGlobal, 1, MpiType<T>{}, MPI_MIN, MPI_COMM_WORLD);
        //        MPI_Allreduce(&star.t_du, &minDtUGlobal, 1, MpiType<T>{}, MPI_MIN, MPI_COMM_WORLD);

        //        if (Base::rank_ == 0)
        //        {
        //            std::printf("acc: %lf\n", minDtAccGlobal);
        //            std::printf("courant: %lf\n", minDtCourantGlobal);
        //            std::printf("rho: %lf\n", minDtRhoGlobal);
        //            std::printf("u: %lf\n\n", minDtRhoGlobal);
        //        }

        computePositions(Base::groups_.view(), d, domain.box(), d.minDt, {float(d.minDt_m1)});
        updateSmoothingLength(Base::groups_.view(), d);
        timer.step("UpdateQuantities");

        disk::computeAndExchangeStarPosition(star, d.minDt, d.minDt_m1);
        timer.step("computeAndExchangeStarPosition");

        disk::computeAccretionCondition(first, last, d, star);
        timer.step("computeAccretionCondition");

        disk::exchangeAndAccreteOnStar(star, d.minDt_m1, Base::rank_);
        timer.step("exchangeAndAccreteOnStar");

        if (Base::rank_ == 0)
        {
            std::printf("star position: %lf\t%lf\t%lf\n", star.position[0], star.position[1], star.position[2]);
            std::printf("star mass: %lf\n", star.m);
            std::printf("additional pot. erg.: %lf\n", star.potential);
        }
        //        auto stats = Base::mHolder_.readStats();
        //        // numP2P, maxP2P, numM2P, maxM2P, maxStack
        //        MPI_Allreduce(MPI_IN_PLACE, stats.data(), 1, MpiType<uint64_t>{}, MPI_SUM, MPI_COMM_WORLD);
        //        MPI_Allreduce(MPI_IN_PLACE, stats.data() + 2, 1, MpiType<uint64_t>{}, MPI_SUM, MPI_COMM_WORLD);
        //        MPI_Allreduce(MPI_IN_PLACE, stats.data() + 1, 1, MpiType<uint64_t>{}, MPI_MAX, MPI_COMM_WORLD);
        //        MPI_Allreduce(MPI_IN_PLACE, stats.data() + 3, 2, MpiType<uint64_t>{}, MPI_MAX, MPI_COMM_WORLD);
        //        if (Base::rank_ == 0)
        //        {
        //            std::cout << "numP2P: " << stats[0] << ", ";
        //            std::cout << "maxP2P: " << stats[1] << ", ";
        //            std::cout << "numM2P: " << stats[2] << ", ";
        //            std::cout << "maxM2P: " << stats[3] << ", ";
        //            std::cout << "maxStack: " << stats[4] << "\n";
        //        }
    }
    void saveFields(IFileWriter* writer, size_t first, size_t last, DataType& simData,
                    const cstone::Box<T>& box) override
    {
        Base::saveFields(writer, first, last, simData, box);
        simData.star.loadOrStoreAttributes(writer);
    }
};

} // namespace sphexa
