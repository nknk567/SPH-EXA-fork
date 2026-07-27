/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief A Propagator class for modern SPH with generalized volume elements
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 */

#pragma once

#include "cstone/fields/field_get.hpp"
#include "sph/particles_data.hpp"
#include "sph/sph.hpp"

#include "ipropagator.hpp"
#include "gravity_wrapper.hpp"

namespace sphexa
{

using namespace sph;
using util::FieldList;

template<bool avClean, class DomainType, class DataType, util::StructuralString TempField = "temp">
class HydroVeProp : public Propagator<DomainType, DataType>
{
protected:
    using Base = Propagator<DomainType, DataType>;
    using Base::pmReader;
    using Base::timer;

    using T             = typename DataType::RealType;
    using KeyType       = typename DataType::KeyType;
    using Tmass         = typename DataType::HydroData::Tmass;
    using MultipoleType = ryoanji::CartesianQuadrupole<Tmass>;

    using Acc       = typename DataType::Exec;
    using MHolder_t = std::conditional_t<cstone::execution::HaveGpu<Acc>{},
                                         MultipoleHolderGpu<MultipoleType, DomainType, typename DataType::HydroData>,
                                         MultipoleHolderCpu<MultipoleType, DomainType, typename DataType::HydroData>>;

    MHolder_t      mHolder_;
    GroupData<Acc> groups_;

    template<class VType>
    using AccVector =
        std::conditional_t<cstone::execution::HaveGpu<Acc>{}, cstone::DeviceVector<VType>, std::vector<VType>>;

    /*! @brief SPH-smoothed converged particle volume, the VE weights xm of the next step (NR mode)
     *
     * Computed at force evaluation time, but only assigned to xm in integrate(), i.e. after the
     * (checkpoint) output, so that restart files contain the weights belonging to the dumped positions.
     */
    AccVector<typename DataType::HydroData::HydroType> volstd_;

    /*! @brief the list of conserved particles fields with values preserved between iterations
     *
     * x, y, z, h and m are automatically considered conserved and must not be specified in this list
     */
    /* xm is conserved because with Newton-Raphson smoothing length iterations (hNRIterMax > 0) the
     * volume elements are carried over from the converged density of the previous step (as in SPHYNX);
     * without NR iterations it is recomputed from scratch every step and could be a dependent field.
     */
    using ConservedFields_ = FieldList<"vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1", "alpha", "id", "xm">;

    //! @brief the energy variable is selectable per instantiation: "temp" (default) or "u"
    using ConservedFields = decltype(FieldList<TempField>{} + ConservedFields_{});

    //! @brief list of dependent fields, these may be used as scratch space during domain sync
    using DependentFields_ = FieldList<"ax", "ay", "az", "prho", "c", "du", "c11", "c12", "c13", "c22", "c23", "c33",
                                       "kx", "nc", "dtCourant">;

    //! @brief velocity gradient fields will only be allocated when avClean is true
    using GradVFields = FieldList<"dV11", "dV12", "dV13", "dV22", "dV23", "dV33">;

    //! @brief what will be allocated based AV cleaning choice
    using DependentFields =
        std::conditional_t<avClean, decltype(DependentFields_{} + GradVFields{}), decltype(DependentFields_{})>;

public:
    HydroVeProp(std::ostream& output, size_t rank)
        : Base(output, rank)
    {
        if (avClean && rank == 0) { std::cout << "AV cleaning is activated" << std::endl; }
    }

    std::vector<std::string> conservedFields() const override
    {
        std::vector<std::string> ret{"x", "y", "z", "h", "m"};
        for_each_tuple([&ret](auto f) { ret.push_back(f.value); }, make_tuple(ConservedFields{}));
        return ret;
    }

    void activateFields(DataType& simData) override
    {
        auto& d = simData.hydro;
        //! @brief Fields accessed in domain sync (x,y,z,h,m,keys) are not part of extensible lists.
        d.setConserved("x", "y", "z", "h", "m");
        d.setDependent("keys");
        std::apply([&d](auto... f) { d.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.setDependent(f.value...); }, make_tuple(DependentFields{}));
    }

    void sync(DomainType& domain, DataType& simData) override
    {
        auto& d = simData.hydro;
        if (d.hNRIterMax > 0)
        {
            /* After halo discovery, h can still grow before the force kernels run: by up to
             * hNRExtFactor per step through the neighbor-count management and, in converged NR
             * tracking, by sub-percent amounts through the NR iterations. Enlarging the halo
             * search by the same factor keeps all remote particles within the final support
             * radius 2h present as halos. */
            domain.setHaloFactor(sph::hNRExtFactor);
        }
        /* Symmetric pair sets need the remote big-h side of cross-rank pairs present as a halo
         * even when it is beyond the reach of all local search spheres. */
        domain.setSymmetricHalos(d.useSymmetricInteractions());
        if (d.g != 0.0)
        {
            domain.syncGrav(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d), get<"m">(d),
                            get<ConservedFields>(d), get<DependentFields>(d));
        }
        else
        {
            domain.sync(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d),
                        std::tuple_cat(std::tie(get<"m">(d)), get<ConservedFields>(d)), get<DependentFields>(d));
        }
        d.treeView = domain.octreeProperties();
        /* The neighbor list is built right after this sync, but the NR iterations move h before
         * the force kernels run. The list builders extend the capture radius to
         * 2h * searchExtFactor (the interaction kernels always cut at the live 2h), so the lists
         * stay complete for the final h as long as the iterations grow h by less than the
         * extension factor. */
        if (d.hNRIterMax > 0) { d.treeView.searchExtFactor = sph::hNRExtFactor; }
        /* Symmetric pair processing (within 2 * max(h_i, h_j)): restores the reaction to the
         * neighbor-kernel term of the pair force, which the plain gather formulation drops for
         * pairs whose distance exceeds one side's support. Essential for energy conservation
         * with NR-iterated h (strong h contrasts at density discontinuities). */
        d.treeView.symmetric = d.useSymmetricInteractions();
    }

    /*! @brief diagnostic: print global extrema of the VE state over the locally owned particles
     *
     * Pinpoints which quantity degenerates when the time step collapses: garbage divv -> rho
     * constraint, kx/xm spikes at vacuum boundaries -> pressure/force spikes, gradh ~ 0 -> prho
     * blow-up. Prints one line per step: '# ve-state: maxAbsDivv=... gradh=[..] kx=[..] maxXm=...'.
     * Costs four minMax reductions and one small Allreduce; the call site is safe to comment out.
     */
    void printVeStateExtrema(typename DataType::HydroData& d, size_t first, size_t last)
    {
        auto extrema = [first, last](const auto& field)
        {
            if constexpr (cstone::execution::HaveGpu<Acc>{})
            {
                return cstone::minMax(cstone::execution::gpuDefaultStream, rawPtr(field) + first,
                                      rawPtr(field) + last);
            }
            else { return cstone::minMax(cstone::execution::cpu, field.data() + first, field.data() + last); }
        };
        auto [divvMin, divvMax]   = extrema(get<"divv">(d));
        auto [gradhMin, gradhMax] = extrema(get<"gradh">(d));
        auto [kxMin, kxMax]       = extrema(get<"kx">(d));
        auto [xmMin, xmMax]       = extrema(get<"xm">(d));

        util::array<double, 6> ex{double(std::max(std::abs(divvMin), std::abs(divvMax))),
                                  -double(gradhMin),
                                  double(gradhMax),
                                  -double(kxMin),
                                  double(kxMax),
                                  double(xmMax)},
            exOut;
        MPI_Allreduce(ex.data(), exOut.data(), ex.size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (Base::rank_ == 0)
        {
            std::cout << "# ve-state: maxAbsDivv=" << exOut[0] << " gradh=[" << -exOut[1] << "," << exOut[2]
                      << "] kx=[" << -exOut[3] << "," << exOut[4] << "] maxXm=" << exOut[5] << std::endl;
        }
    }

    /*! @brief diagnostic: globally count non-finite du/ax/ay/az entries of the owned particles
     *
     * Placed after each force stage (sph momentum, gravity, disk central force), the first tag
     * with a nonzero count names the stage that produces NaN/inf. Call sites are single lines,
     * safe to comment out.
     */
    template<class FieldVector>
    unsigned long long nonFiniteCount(const FieldVector& field, size_t first, size_t last)
    {
        if constexpr (cstone::execution::HaveGpu<Acc>{})
        {
            return gpu::countNonFiniteGpu(rawPtr(field), first, last);
        }
        else
        {
            unsigned long long n = 0;
            const auto*        p = field.data();
#pragma omp parallel for reduction(+ : n)
            for (size_t i = first; i < last; ++i)
            {
                n += !std::isfinite(p[i]);
            }
            return n;
        }
    }

    void printNonFinite(const char* tag, typename DataType::HydroData& d, size_t first, size_t last)
    {
        util::array<unsigned long long, 4> c{nonFiniteCount(get<"du">(d), first, last),
                                             nonFiniteCount(get<"ax">(d), first, last),
                                             nonFiniteCount(get<"ay">(d), first, last),
                                             nonFiniteCount(get<"az">(d), first, last)},
            cOut;
        MPI_Allreduce(c.data(), cOut.data(), c.size(), MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (Base::rank_ == 0)
        {
            std::cout << "# nonfinite[" << tag << "]: du=" << cOut[0] << " ax=" << cOut[1] << " ay=" << cOut[2]
                      << " az=" << cOut[3] << std::endl;
        }
    }

    //! @brief diagnostic twin of printNonFinite for the VE intermediate fields (divv/gradh/cij/prho, alpha)
    void printNonFiniteVe(const char* tag, typename DataType::HydroData& d, size_t first, size_t last)
    {
        util::array<unsigned long long, 5> c{nonFiniteCount(get<"divv">(d), first, last),
                                             nonFiniteCount(get<"gradh">(d), first, last),
                                             nonFiniteCount(get<"c11">(d), first, last),
                                             nonFiniteCount(get<"prho">(d), first, last),
                                             nonFiniteCount(get<"alpha">(d), first, last)},
            cOut;
        MPI_Allreduce(c.data(), cOut.data(), c.size(), MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (Base::rank_ == 0)
        {
            std::cout << "# nonfinite[" << tag << "]: divv=" << cOut[0] << " gradh=" << cOut[1] << " c11=" << cOut[2]
                      << " prho=" << cOut[3] << " alpha=" << cOut[4] << std::endl;
        }
    }

    void computeForces(DomainType& domain, DataType& simData) override
    {
        timer.start();
        pmReader.start();
        sync(domain, simData);
        timer.step("domain::sync");
        Base::logDomainStats(domain, simData);

        auto& d = simData.hydro;
        d.resize(domain.nParticlesWithHalos());
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        fillMassHalos(domain.exec(), get<"m">(d), first, last);

        if (d.hNRIterMax > 0)
        {
            /* The extended-radius neighbor list holds up to hNRExtFactor^3 more entries per
             * particle than found within 2h. The list build capacity ngmaxExt is sized such that
             * the extended list fits for the 2h-counts a strong shock can produce (with 0.9
             * headroom); beyond that, the list builder falls back to the plain 2h search.
             * ngmax itself is never modified: it remains the user-set neighbor-count bound that
             * updateHIterativeNR enforces on the 2h count. */
            const float    extVol   = std::pow(sph::hNRExtFactor, 3);
//          const unsigned ngmaxMin = std::ceil(1.5f * d.ng0 * extVol / 0.9f);
            const unsigned ngmaxMin = std::ceil(5.f * d.ng0 * extVol / 0.9f);

            if (d.ngmaxExt < ngmaxMin)
            {
                if (Base::rank_ == 0)
                {
                    std::cout << "Neighbor-list build capacity set to " << ngmaxMin
                              << " to fit the extended neighbor search of the smoothing-length NR iterations"
                              << " (h-update keeps 2h-counts within ngmax = " << d.ngmax << ")" << std::endl;
                }
                d.ngmaxExt = ngmaxMin;
            }
        }

        computeGroups(first, last, d, domain.box(), groups_);
        timer.step("computeGroups");
        if (d.hNRIterMax > 0) { updateSmoothingLengthIterativeNR(groups_.view(), d, domain.box()); }
        else { updateSmoothingLengthIterative(groups_.view(), d, domain.box()); }
        timer.step("updateSmoothingLengthIterative");
        findNeighborsSfc(groups_.view(), d, domain.box());
        timer.step("FindNeighbors");
        pmReader.step();

        /* With NR iterations, the volume elements are carried over from the smoothed converged
         * volume of the previous step (SPHYNX-style, see computeVolstd) and only initialized from
         * the standard SPH density on the first step. Recomputing xm from the CURRENT h every
         * step (tried as a diagnostic) makes the NR constraint ill-posed for undersampled
         * particles: with a self-dominated density, xm ~ h^3/(K*w0) and both sides of
         * kx * h^3 = eta * xm scale with h^3 — the root degenerates and h drifts against the
         * caps (observed as kx -> eta/(K*w0) spikes and permanent 9-iteration tug-of-war). */
        if (d.hNRIterMax == 0 || d.iteration == 1)
        {
            computeXMass(groups_.view(), d, domain.box());
            timer.step("XMass");
        }
        domain.exchangeHalos(std::tie(get<"xm">(d)), get<"ax">(d), get<"keys">(d));
        timer.step("mpi::synchronizeHalos");

        if (d.hNRIterMax > 0)
        {
            /* Newton-Raphson iterations converging h towards rho * h^3 = ballmassEta(ng0) * m,
             * such that the grad-h terms are formally consistent with dh/drho = -h / (3 * rho).
             * The target depends only on the desired neighbor count and the particle mass, i.e.
             * it is constant in time. The iterations reuse the fixed neighbor list with fixed
             * volume elements xm and are gather-only, so they require no communication and each
             * rank may stop as soon as its own particles are converged. Following SPHYNX
             * (Cabezon & Garcia-Senz).
             * volstd_ temporarily holds the pre-iteration h, capping the cumulative upward h
             * movement at the neighbor-list extension; it is reused for the smoothed volume
             * later in the step. */
            reallocate(volstd_, d.x.size(), d.getAllocGrowthRate());
            unsigned nrIterations = 0;
            while (nrIterations < d.hNRIterMax)
            {
                ++nrIterations;
                auto maxRelDh = computeVeNR(groups_.view(), d, domain.box(), cstone::rawPtr(volstd_),
                                            /*firstIteration*/ nrIterations == 1);
                if (maxRelDh < sph::hNRTol) { break; }
            }
            timer.logStatistics("hNRIterations", nrIterations);
            if (Base::rank_ == 0) { std::cout << "# hNRIterations: " << nrIterations << std::endl; }
            timer.step("hNewtonRaphson");
        }
        computeVe(groups_.view(), d, domain.box());
        timer.step("Generalized Volume Elements");
        if (d.hNRIterMax > 0)
        {
            //! h of locally owned particles changed: halos need updating, h_j enters the momentum equation
            domain.exchangeHalos(std::tuple_cat(std::tie(get<"h">(d)), get<"vx", "vy", "vz", "kx">(d)), get<"ax">(d),
                                 get<"keys">(d));
        }
        else { domain.exchangeHalos(get<"vx", "vy", "vz", "kx">(d), get<"ax">(d), get<"keys">(d)); }
        timer.step("mpi::synchronizeHalos");

        if (d.hNRIterMax > 0)
        {
            /* Smoothed converged volume, the VE weights of the next step; needs kx halos, computed
             * before ay/az are released. Assigned to xm in integrate(), after the output. */
            reallocate(volstd_, d.x.size(), d.getAllocGrowthRate());
            computeVolstd(groups_.view(), d, domain.box(), cstone::rawPtr(volstd_));
            timer.step("Volstd");
        }

        release(d, "ay", "az");
        acquire(d, "divv", "gradh");
        computeIadDivvCurlvGradh(groups_.view(), d, domain.box());
        d.minDtRho = rhoTimestep(first, last, d);
        timer.step("IadVelocityDivCurlGradh");

        computeEOS(first, last, d);
        timer.step("EquationOfState");

        printVeStateExtrema(d, first, last);   // VE state diagnostic, safe to comment out
        printNonFiniteVe("eos", d, first, last); // NaN-localizer diagnostic, safe to comment out

        domain.exchangeHalos(get<"c11", "c12", "c13", "c22", "c23", "c33", "divv", "c">(d), get<"ax">(d),
                             get<"keys">(d));
        timer.step("mpi::synchronizeHalos");

        computeAVswitches(groups_.view(), d, domain.box());
        timer.step("AVswitches");

        printNonFiniteVe("avswitch", d, first, last); // NaN-localizer diagnostic, safe to comment out

        if (avClean)
        {
            domain.exchangeHalos(get<"dV11", "dV12", "dV13", "dV22", "dV23", "dV33", "prho", "alpha">(d), get<"ax">(d),
                                 get<"keys">(d));
        }
        else { domain.exchangeHalos(get<"prho", "alpha">(d), get<"ax">(d), get<"keys">(d)); }
        timer.step("mpi::synchronizeHalos");

        release(d, "divv", "gradh");
        acquire(d, "ay", "az");
        computeMomentumEnergy<avClean>(groups_.view(), nullptr, d, domain.box());
        timer.step("MomentumAndEnergy");
        pmReader.step();

        printNonFinite("sph", d, first, last); // NaN-localizer diagnostic, safe to comment out

        if (d.g != 0.0)
        {
            auto groups = mHolder_.computeSpatialGroups(d, domain);
            mHolder_.upsweep(d, domain);
            timer.step("Upsweep");
            pmReader.step();
            mHolder_.traverse(groups, d, domain);
            timer.step("Gravity");
            pmReader.step();

            auto stats = mHolder_.readStats();
            timer.logStatistics("sumP2P", stats[0] / timer.getLastStepTime());
            timer.logStatistics("sumM2P", stats[2] / timer.getLastStepTime());
        }

        printNonFinite("gravity", d, first, last); // NaN-localizer diagnostic, safe to comment out
    }

    void integrate(DomainType& domain, DataType& simData) override
    {
        auto&  d     = simData.hydro;
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        computeTimestep(first, last, d);
        timer.step("Timestep");
        computePositions(groups_.view(), d, domain.box(), d.minDt, {float(d.minDt_m1)});
        /* With Newton-Raphson iterations active, h is converged towards rho * h^3 = eta * m during
         * the force computation; nudging h towards the neighbor count target here would displace it
         * from the converged solution every step. Unresolvable particles are still flagged. */
        bool haveUnconvergedParticles = updateSmoothingLength(groups_.view(), d, /*adjustH*/ d.hNRIterMax == 0);
        if (haveUnconvergedParticles && not d.removeUnconvergedParticles)
        {
            throw std::runtime_error("Neighbor search did not converge\n");
        }

        if (d.hNRIterMax > 0)
        {
            /* volume elements of the next step, the smoothed converged volume of this step
             * (SPHYNX-style); placed after the checkpoint output so that restarts see the weights
             * that belong to the dumped positions */
            setVolumeElements(groups_.view(), d, cstone::rawPtr(volstd_));
        }
        timer.step("UpdateQuantities");
    }

    void saveFields(IFileWriter* writer, size_t first, size_t last, DataType& simData,
                    const cstone::Box<T>& box) override
    {
        auto& d             = simData.hydro;
        auto  fieldPointers = d.data();
        auto  indicesDone   = d.outputFieldIndices;
        auto  namesDone     = d.outputFieldNames;

        auto output = [&]()
        {
            for (int i = int(indicesDone.size()) - 1; i >= 0; --i)
            {
                int fidx = indicesDone[i];
                if (d.isAllocated(fidx))
                {
                    int column = std::find(d.outputFieldIndices.begin(), d.outputFieldIndices.end(), fidx) -
                                 d.outputFieldIndices.begin();
                    std::visit(
                        [writer, c = column, key = namesDone[i]](auto field)
                        {
                            auto&& tmp = cstone::toHost(*field);
                            writeField(writer, key, tmp.data(), c);
                        },
                        fieldPointers[fidx]);
                    indicesDone.erase(indicesDone.begin() + i);
                    namesDone.erase(namesDone.begin() + i);
                }
            }
        };

        // first output pass: write everything allocated at the end of computeForces()
        output();

        // second output pass: write temporary quantities produced by the EOS
        release(d, "c11", "c12", "c13");
        acquire(d, "rho", "p", "gradh");
        computeEOS(first, last, d);
        output();
        release(d, "rho", "p", "gradh");
        acquire(d, "c11", "c12", "c13");

        // third output pass: recover temporary curlv and divv quantities
        release(d, "prho", "c");
        acquire(d, "divv", "curlv");
        // partial recovery of cij in range [first:last] without halos, which are not needed for divv and curlv
        if (!indicesDone.empty()) { computeIadDivvCurlvGradh(groups_.view(), d, box); }
        output();
        release(d, "divv", "curlv");
        acquire(d, "prho", "c");

        /* The following data is now lost and no longer available in the integration step
         *  c11, c12, c12: halos invalidated
         *  prho, c: destroyed
         */

        if (!indicesDone.empty() && Base::rank_ == 0)
        {
            std::cout << "WARNING: the following fields are not in use and therefore not output: ";
            for (std::size_t fidx = 0; fidx < indicesDone.size() - 1; ++fidx)
            {
                std::cout << d.fieldNames[indicesDone[fidx]] << ",";
            }
            std::cout << d.fieldNames[indicesDone.back()] << std::endl;
        }
        timer.step("FileOutput");
    }
};

} // namespace sphexa
