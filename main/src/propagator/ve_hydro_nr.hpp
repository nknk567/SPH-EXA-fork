/*! @file
 * @brief A VE Propagator with Newton-Raphson iterated smoothing lengths (SPHYNX-style)
 *
 * @author Noah Kubli
 */

#pragma once

#include <filesystem>

#include "io/arg_parser.hpp"
#include "ve_hydro.hpp"

namespace sphexa
{

using namespace sph;

/*! @brief VE propagator with Newton-Raphson smoothing-length iterations
 *
 * Extends the plain VE propagator: each step, h is converged to the volume-element consistency
 * rho * h^3 = ballmassEta(ng0) * m (grad-h terms formally consistent with dh/drho = -h/(3 rho)),
 * and the volume elements are carried over as the SPH-smoothed converged volume of the previous
 * step, following SPHYNX (Cabezon & Garcia-Senz). Pair interactions are processed symmetrically
 * within 2 * max(h_i, h_j), restoring the reaction to the neighbor-kernel term of the pair force
 * — essential for energy conservation with the strong h contrasts the NR scheme develops at
 * density discontinuities.
 *
 * Both features are active by construction. The NR parameters are checkpoint attributes:
 * initialized from the test-case settings (if named there), overridden by the values stored in
 * a restart file, and persisted with every checkpoint.
 */
template<bool avClean, class DomainType, class DataType, util::StructuralString TempField = "temp">
class HydroVeNRProp : public HydroVeProp<avClean, DomainType, DataType, TempField>
{
protected:
    using Base = HydroVeProp<avClean, DomainType, DataType, TempField>;
    using Base::groups_;
    using Base::pmReader;
    using Base::timer;

    using typename Base::Acc;
    using typename Base::ConservedFields;
    using typename Base::DependentFields;
    using T = typename DataType::RealType;

    //! @brief runtime parameters of the NR scheme, persisted as checkpoint attributes
    struct NRParams
    {
        //! @brief maximum number of NR iterations per step; the iterations stop early on convergence
        unsigned hNRIterMax{15};

        /*! @brief extension factor for the halo search and the neighbor-list capture radius
         *
         * Halos and neighbor lists are built before the NR iterations move h; extending both
         * search radii by this factor keeps them complete as long as h grows by less than this
         * factor within a step (the interaction kernels always cut at the live 2h). It also caps
         * the per-step h growth of the neighbor-count management and of the NR iterations.
         */
        float hNRExtFactor{1.05f};

        //! @brief relative smoothing-length change below which a particle counts as converged
        float hNRTol{1e-4f};

        /*! @brief lower grad-h (Omega) limit
         *
         * With NR-iterated h, Omega >= 0 by construction but approaches zero when all kernel
         * mass sits near the center or edge of the support (void or clustered-pair
         * configurations). prho ~ 1/Omega in the EOS: the floor keeps the pressure term bounded
         * in such transients; legitimate values stay well above it.
         */
        float gradhMin{0.1f};

        /*! @brief per-step change limits for a carried volume element (asymmetric)
         *
         * The carried weights are the SPH-smoothed converged volumes (volstd). UPWARD volume
         * drift at vacuum boundaries is a numerical ratchet of the V^2-weighted smoothing —
         * hence the tight growth limit; physical volume growth per step is O(|divv| * dt) << 1,
         * so it is inert in resolved flow. DOWNWARD movement is legitimate fast adaptation:
         * compressing debris correctly shrinks h at up to 0.5 * h0 per step and the carried
         * volume must follow at a comparable rate, or kx = eta * xm / h^3 spikes by the
         * staleness ratio — hence the looser shrink limit.
         */
        float volstdGrowFactor{2.0f};
        float volstdShrinkFactor{8.0f};

        /*! @brief volume-element source: 0 = carried smoothed volume, 1 = m/rho0 every step
         *
         * 0 (SPHYNX volstd): the weights are the smoothed converged volumes of the previous
         * step, frozen during the step. 1 (classical SPH-EXA VE definition): xm = m/rho0 is
         * recomputed from the step-start h every step. Known risk of mode 1 under NR: for
         * self-dominated (isolated) particles xm ~ h^3 at the step-start h, so the root has no
         * restoring force ACROSS steps and drifts against the caps; the neighbor-count guard
         * and the per-step walls must contain it.
         */
        unsigned xmSource{0};

        template<class Archive>
        void loadOrStoreAttributes(Archive* ar)
        {
            //! @brief load or store an attribute, skips non-existing attributes on load
            auto optionalIO = [ar](const std::string& attribute, auto* location, size_t attrSize)
            {
                try
                {
                    ar->stepAttribute(attribute, location, attrSize);
                }
                catch (std::out_of_range&)
                {
                    if (ar->rank() == 0)
                    {
                        std::cout << "Attribute " << attribute
                                  << " not set in file or initializer, setting to default value " << *location
                                  << std::endl;
                    }
                }
            };

            optionalIO("hNRIterMax", &hNRIterMax, 1);
            optionalIO("hNRExtFactor", &hNRExtFactor, 1);
            optionalIO("hNRTol", &hNRTol, 1);
            optionalIO("gradhMin", &gradhMin, 1);
            optionalIO("volstdGrowFactor", &volstdGrowFactor, 1);
            optionalIO("volstdShrinkFactor", &volstdShrinkFactor, 1);
            optionalIO("xmSource", &xmSource, 1);
        }
    };

    NRParams nrParams_;

    //! @brief neighbor-list build capacity for the extended NR search, sized once, see updateNgmaxExt
    unsigned ngmaxExt_{0};

    template<class VType>
    using AccVector =
        std::conditional_t<cstone::execution::HaveGpu<Acc>{}, cstone::DeviceVector<VType>, std::vector<VType>>;

    /*! @brief SPH-smoothed converged particle volume, the VE weights xm of the next step
     *
     * Computed at force evaluation time, but only assigned to xm in integrate(), i.e. after the
     * (checkpoint) output, so that restart files contain the weights belonging to the dumped positions.
     */
    AccVector<typename DataType::HydroData::HydroType> volstd_;

public:
    HydroVeNRProp(std::ostream& output, size_t rank, const InitSettings& settings)
        : Base(output, rank)
    {
        BuiltinWriter attributeWriter(settings);
        nrParams_.loadOrStoreAttributes(&attributeWriter);

        //! ng0 is owned by ParticlesData; read it here only to size the list build capacity
        unsigned ng0 = settings.count("ng0") ? unsigned(settings.at("ng0")) : 100u;
        updateNgmaxExt(ng0);
    }

    void load(const std::string& initCond, IFileReader* reader) override
    {
        const std::string path = removeModifiers(initCond);
        if (!std::filesystem::exists(path)) { return; }

        int snapshotIndex = numberAfterSign(initCond, ":");
        reader->setStep(path, snapshotIndex, FileMode::independent);
        nrParams_.loadOrStoreAttributes(reader);
        //! ng0 is owned/persisted by ParticlesData; read-only access to size the list build capacity
        unsigned ng0{typename DataType::HydroData{}.ng0};
        try
        {
            reader->stepAttribute("ng0", &ng0, 1);
        }
        catch (std::out_of_range&)
        {
        }
        reader->closeStep();

        /* Checkpoints written by propagators without NR iterations may store hNRIterMax = 0;
         * selecting this propagator is the NR opt-in, so restore the default instead of
         * silently running without iterations. */
        if (nrParams_.hNRIterMax == 0) { nrParams_.hNRIterMax = NRParams{}.hNRIterMax; }
        updateNgmaxExt(ng0);
    }

    void save(IFileWriter* writer) override { nrParams_.loadOrStoreAttributes(writer); }

    void sync(DomainType& domain, DataType& simData) override
    {
        auto& d = simData.hydro;
        /* After halo discovery, h can still grow before the force kernels run: by up to
         * hNRExtFactor per step through the neighbor-count management and, in converged NR
         * tracking, by sub-percent amounts through the NR iterations. Enlarging the halo
         * search by the same factor keeps all remote particles within the final support
         * radius 2h present as halos. */
        domain.setHaloFactor(nrParams_.hNRExtFactor);
        /* Symmetric pair sets need the remote big-h side of cross-rank pairs present as a halo
         * even when it is beyond the reach of all local search spheres. */
        domain.setSymmetricHalos(true);
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
        d.treeView.searchExtFactor = nrParams_.hNRExtFactor;
        /* Symmetric pair processing (within 2 * max(h_i, h_j)): restores the reaction to the
         * neighbor-kernel term of the pair force, which the plain gather formulation drops for
         * pairs whose distance exceeds one side's support. */
        d.treeView.symmetric = true;
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
        d.ngmaxExt = ngmaxExt_;

        computeGroups(first, last, d, domain.box(), groups_);
        timer.step("computeGroups");
        updateSmoothingLengthIterativeNR(groups_.view(), d, domain.box(), nrParams_.hNRExtFactor);
        timer.step("updateSmoothingLengthIterative");
        findNeighborsSfc(groups_.view(), d, domain.box());
        timer.step("FindNeighbors");
        pmReader.step();

        /* The volume elements are carried over from the smoothed converged volume of the
         * previous step (SPHYNX-style, see computeVolstd) and only initialized from the
         * standard SPH density on the first step. Recomputing xm from the CURRENT h every
         * step (tried as a diagnostic) makes the NR constraint ill-posed for undersampled
         * particles: with a self-dominated density, xm ~ h^3/(K*w0) and both sides of
         * kx * h^3 = eta * xm scale with h^3 — the root degenerates and h drifts against the
         * caps (observed as kx -> eta/(K*w0) spikes and permanent 9-iteration tug-of-war). */
        if (nrParams_.xmSource != 0 || d.iteration == 1)
        {
            computeXMass(groups_.view(), d, domain.box());
            timer.step("XMass");
        }
        domain.exchangeHalos(std::tie(get<"xm">(d)), get<"ax">(d), get<"keys">(d));
        timer.step("mpi::synchronizeHalos");

        convergeSmoothingLengthNR(domain, simData);

        computeVe(groups_.view(), d, domain.box());
        timer.step("Generalized Volume Elements");
        //! h of locally owned particles changed: halos need updating, h_j enters the momentum equation
        domain.exchangeHalos(std::tuple_cat(std::tie(get<"h">(d)), get<"vx", "vy", "vz", "kx">(d)), get<"ax">(d),
                             get<"keys">(d));
        timer.step("mpi::synchronizeHalos");

        if (nrParams_.xmSource == 0)
        {
            /* Smoothed converged volume, the VE weights of the next step; needs kx halos,
             * computed before ay/az are released. Assigned to xm in integrate(), after the
             * output. */
            reallocateDestructive(volstd_, d.x.size(), d.getAllocGrowthRate());
            computeVolstd(groups_.view(), d, domain.box(), cstone::rawPtr(volstd_));
            timer.step("Volstd");
        }

        Base::computeForcesCommon(domain, simData, /*nrMode*/ true, nrParams_.gradhMin);
    }

    void integrate(DomainType& domain, DataType& simData) override
    {
        auto&  d     = simData.hydro;
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        computeTimestep(first, last, d);
        timer.step("Timestep");
        computePositions(groups_.view(), d, domain.box(), d.minDt, {float(d.minDt_m1)});
        /* h is converged towards rho * h^3 = eta * m during the force computation; nudging h
         * towards the neighbor count target here would displace it from the converged solution
         * every step. Unresolvable particles are still flagged. */
        bool haveUnconvergedParticles = updateSmoothingLength(groups_.view(), d, /*adjustH*/ false);
        if (haveUnconvergedParticles && not d.removeUnconvergedParticles)
        {
            throw std::runtime_error("Neighbor search did not converge\n");
        }

        if (nrParams_.xmSource == 0)
        {
            /* volume elements of the next step, the smoothed converged volume of this step
             * (SPHYNX-style); placed after the checkpoint output so that restarts see the
             * weights that belong to the dumped positions */
            setVolumeElements(groups_.view(), d, cstone::rawPtr(volstd_), nrParams_.volstdGrowFactor,
                              nrParams_.volstdShrinkFactor);
        }
        timer.step("UpdateQuantities");
    }

protected:
    /*! @brief size the neighbor-list build capacity for the extended NR search, once at startup
     *
     * The extended-radius neighbor list holds up to hNRExtFactor^3 more entries per particle
     * than found within 2h. The capacity is sized such that the extended list fits for the
     * 2h-counts a strong shock can produce (with 0.9 headroom); beyond that, the list builder
     * falls back to the plain 2h search. ngmax itself is never modified: it remains the
     * user-set neighbor-count bound that updateHIterativeNR enforces on the 2h count.
     */
    void updateNgmaxExt(unsigned ng0)
    {
        const float extVol = std::pow(nrParams_.hNRExtFactor, 3);
        ngmaxExt_          = std::ceil(5.f * ng0 * extVol / 0.9f);
        if (Base::rank_ == 0)
        {
            std::cout << "Neighbor-list build capacity set to " << ngmaxExt_
                      << " to fit the extended neighbor search of the smoothing-length NR iterations" << std::endl;
        }
    }

    /*! @brief converge the smoothing lengths to the volume-element consistency rho * h^3 = eta * m
     *
     * Newton-Raphson iterations converging h towards rho * h^3 = ballmassEta(ng0) * m,
     * such that the grad-h terms are formally consistent with dh/drho = -h / (3 * rho).
     * The target depends only on the desired neighbor count and the particle mass, i.e.
     * it is constant in time. The iterations reuse the fixed neighbor list with fixed
     * volume elements xm and are gather-only, so they require no communication and each
     * rank may stop as soon as its own particles are converged. Following SPHYNX
     * (Cabezon & Garcia-Senz).
     * volstd_ temporarily holds the pre-iteration h, capping the cumulative upward h
     * movement at the neighbor-list extension; it is reused for the smoothed volume
     * later in the step.
     */
    void convergeSmoothingLengthNR(DomainType& domain, DataType& simData)
    {
        auto& d = simData.hydro;

        reallocateDestructive(volstd_, d.x.size(), d.getAllocGrowthRate());
        unsigned            nrIterations = 0;
        std::vector<size_t> nrUnconverged;
        size_t              capIterUp = 0, capIterDown = 0;
        while (nrIterations < nrParams_.hNRIterMax)
        {
            ++nrIterations;
            NRPassStats stats =
                computeVeNR(groups_.view(), d, domain.box(), cstone::rawPtr(volstd_),
                            /*firstIteration*/ nrIterations == 1, nrParams_.hNRTol, nrParams_.hNRExtFactor);
            nrUnconverged.push_back(stats.numUnconverged);
            capIterUp += stats.numCapUp;
            capIterDown += stats.numCapDown;
            if (stats.numUnconverged == 0) { break; }
            /* Measured (TDE debris): ~98% of the particles converge within two passes; the
             * rest is a vacuum-edge residual that holds the global chain at 5-9 iterations.
             * Once the unconverged set is small, full neighbor-list passes over all
             * particles are wasted on it: finish those particles with per-particle
             * octree-traversal updates instead (exact, see computeVeNRTail). */
            if (stats.numUnconverged * 100 < groups_.view().lastBody - groups_.view().firstBody)
            {
                nrIterations +=
                    computeVeNRTail(groups_.view(), d, domain.box(), cstone::rawPtr(volstd_),
                                    nrParams_.hNRIterMax - nrIterations, nrUnconverged, nrParams_.hNRTol,
                                    nrParams_.hNRExtFactor, capIterUp, capIterDown);
                break;
            }
        }
        timer.logStatistics("hNRIterations", nrIterations);
        if (Base::rank_ == 0) { std::cout << "# hNRIterations: " << nrIterations << std::endl; }
        printNRUnconverged(nrUnconverged, nrParams_.hNRIterMax); // NR convergence diagnostic, safe to comment out
        //! cap statistics, safe to comment out; volstd_ still holds the step-start h here
        printNRCapped(capIterUp, capIterDown,
                      countHWallPinned(groups_.view(), d, cstone::rawPtr(volstd_), nrParams_.hNRExtFactor));
        timer.step("hNewtonRaphson");
    }

    /*! @brief diagnostic: print how often the NR iterations ran into the h step-limit walls
     *
     * One line per step: iterUp/iterDown = per-iteration 1.1x / 0.5x clamp events summed over
     * all passes; endUp/endDown = particles whose FINAL h is pinned at the cumulative walls
     * hNRExtFactor * h0 / 0.5 * h0. Wall-pinned particles are frozen OFF their NR root but
     * count as converged (relDh = 0 at the wall) — this line and the h-residual check of
     * verify_nr.py are the complementary views. Costs one small Allreduce; the call site is
     * a single line, safe to comment out.
     */
    void printNRCapped(size_t iterUp, size_t iterDown, std::pair<size_t, size_t> endWalls)
    {
        unsigned long long c[4] = {iterUp, iterDown, endWalls.first, endWalls.second}, cOut[4];
        MPI_Allreduce(c, cOut, 4, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (Base::rank_ == 0)
        {
            std::cout << "# hNRCapped: iterUp=" << cOut[0] << " iterDown=" << cOut[1] << " endUp=" << cOut[2]
                      << " endDown=" << cOut[3] << std::endl;
        }
    }

    /*! @brief diagnostic: print the global number of unconverged particles after each NR pass
     *
     * One line per step, e.g. '# hNRUnconverged: 7241511 2833900 9841 312 0': entry k is the
     * global number of particles whose relative h change in NR pass k+1 was still >= hNRTol;
     * the switch to the tail iterations is visible as the last percent-scale entry. Costs one
     * small Allreduce after the iterations finished (the NR loop itself stays
     * communication-free per rank; ranks that stopped early contribute zeros). The call site
     * is a single line, safe to comment out.
     */
    void printNRUnconverged(const std::vector<size_t>& counts, unsigned hNRIterMax)
    {
        std::vector<unsigned long long> c(hNRIterMax, 0), cOut(hNRIterMax, 0);
        std::copy(counts.begin(), counts.end(), c.begin());
        MPI_Allreduce(c.data(), cOut.data(), int(hNRIterMax), MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (Base::rank_ != 0) { return; }
        //! print up to the last nonzero entry plus the trailing converged zero
        size_t numEntries = 1;
        for (size_t k = 0; k < cOut.size(); ++k)
        {
            if (cOut[k] > 0) { numEntries = std::min(k + 2, cOut.size()); }
        }
        std::cout << "# hNRUnconverged:";
        for (size_t k = 0; k < numEntries; ++k)
        {
            std::cout << " " << cOut[k];
        }
        std::cout << std::endl;
    }
};

} // namespace sphexa
