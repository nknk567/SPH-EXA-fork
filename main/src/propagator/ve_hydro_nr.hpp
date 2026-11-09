/*! @file
 * @brief A VE Propagator with Newton-Raphson iterated smoothing lengths (SPHYNX-style)
 *
 * @author Noah Kubli
 */

#pragma once

#include <filesystem>
#include <limits>

#include "io/arg_parser.hpp"
#include "ve_hydro.hpp"
#include "sph/hydro_ve/nr_statistics.hpp"

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
template<bool avClean, class DomainType, class DataType>
class HydroVeNRProp : public HydroVeProp<avClean, DomainType, DataType>
{
protected:
    using Base = HydroVeProp<avClean, DomainType, DataType>;
    using Base::groups_;
    using Base::pmReader;
    using Base::timer;

    using typename Base::Acc;
    using T = typename DataType::RealType;

    using ConservedFields = decltype(typename Base::ConservedFields{} + FieldList<"ballmass">{});
    using Base::DependentFields;

    //! @brief runtime parameters of the NR scheme, persisted as checkpoint attributes
    struct NRParams
    {
        //! @brief Maximally allowed neighbours in NR iterations (smaller than ngmax of the neighbour list)
        unsigned nr_ngmax{150};

        //! @brief maximum number of NR iterations per step; the iterations stop early on convergence
        unsigned hNRIterMax{15};

        /*! @brief extension factor for the halo search and the neighbor-list capture radius
         *
         * Halos and neighbor lists are built before the NR iterations move h; extending both
         * search radii by this factor keeps them complete as long as h grows by less than this
         * factor within a step (the interaction kernels always cut at the live 2h). It is also
         * the default per-step upward wall of the NR iterations, see hNRWallFactor.
         */
        float hNRExtFactor{1.05f};

        /*! @brief per-step upward NR wall as a factor on the step-start h0; 0 = use hNRExtFactor
         *
         * Decoupled from the capture extension so upward h migration can exceed the per-step
         * list margin without paying the list-memory cost of a larger hNRExtFactor (which lands
         * in the dense regions, exactly where h does not need to grow). Growth beyond the
         * capture radius iterates on truncated pair sums (SPHYNX operates this way permanently):
         * the kx/dkx error is kernel-tail suppressed (zero for growth into vacuum), while the
         * neighbor count becomes a lower bound — the convergence-point band check defers its
         * below-band decision there, see VeNRPostamble. Keep <~ 1.3: beyond that the missed
         * shell reaches non-negligible kernel weights when the missed region is dense.
         */
        float hNRWallFactor{1.2};

        //! @brief relative smoothing-length change below which a particle counts as converged
        float hNRTol{1e-4f};

        /*! @brief NR constraint target source: 0 = fixed nominal target, 1 = per-particle target
         *
         * 0: ballmass is refilled with ballmassEta(ng0) * m every step — the fixed global
         * constraint (legacy behavior). 1 (after SPHYNX findneighbors.f90): the per-particle
         * field is frozen between steps and only rebased when the NR CONVERGING POINT has a
         * neighbor count outside the guard band [ng0/4, ngmax]: h is then reset to the
         * neighbor-count guard's (in-band) output and the target re-seeded to rho * h^3 there,
         * so out-of-band roots are eliminated while transient guard corrections — whose root
         * was fine all along — leave the target untouched. See the band check in VeNRPostamble.
         */
        unsigned particleBallmass{0};

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

            optionalIO("nr_ngmax", &nr_ngmax, 1);
            optionalIO("hNRIterMax", &hNRIterMax, 1);
            optionalIO("hNRExtFactor", &hNRExtFactor, 1);
            optionalIO("hNRWallFactor", &hNRWallFactor, 1);
            optionalIO("hNRTol", &hNRTol, 1);
            //            optionalIO("gradhMin", &gradhMin, 1);
            //            optionalIO("volstdGrowFactor", &volstdGrowFactor, 1);
            //            optionalIO("volstdShrinkFactor", &volstdShrinkFactor, 1);
            //            optionalIO("xmSource", &xmSource, 1);
            optionalIO("particleBallmass", &particleBallmass, 1);
        }
    };

    NRParams nrParams_;

    //! @brief neighbor-list build capacity for the extended NR search, sized once, see updateNgmaxExt
    //    unsigned ngmaxExt_{0};

    template<class VType>
    using AccVector =
        std::conditional_t<cstone::execution::HaveGpu<Acc>{}, cstone::DeviceVector<VType>, std::vector<VType>>;

    /*! @brief SPH-smoothed converged particle volume, the VE weights xm of the next step
     *
     * Computed at force evaluation time, but only assigned to xm in integrate(), i.e. after the
     * (checkpoint) output, so that restart files contain the weights belonging to the dumped positions.
     */
    //    AccVector<typename DataType::HydroData::HydroType> h_old;

public:
    HydroVeNRProp(std::ostream& output, size_t rank, const InitSettings& settings)
        : Base(output, rank)
    {
        BuiltinWriter attributeWriter(settings);
        nrParams_.loadOrStoreAttributes(&attributeWriter);

        //        //! ng0 is owned by ParticlesData; read it here only to size the list build capacity
        //        unsigned ng0 = settings.count("ng0") ? unsigned(settings.at("ng0")) : 100u;
        //        updateNgmaxExt(ng0);
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
        if (nrParams_.hNRIterMax == 0) { std::printf("warning: 0 NR iterations allowed\n"); }
        //        updateNgmaxExt(ng0);
    }

    void save(IFileWriter* writer) override { nrParams_.loadOrStoreAttributes(writer); }

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
        std::apply([&d](auto... f) { d.setDependent(f.value...); }, make_tuple(typename Base::DependentFields{}));
    }

    void sync(DomainType& domain, DataType& simData) override
    {
        updateNgmax(nrParams_.nr_ngmax, simData);
        auto& d = simData.hydro;

        /* After halo discovery, h can still grow before the force kernels run: by up to
         * hNRExtFactor per step through the neighbor-count management and, in converged NR
         * tracking, by sub-percent amounts through the NR iterations. Enlarging the halo
         * search by the same factor keeps all remote particles within the final support
         * radius 2h present as halos. NR growth beyond this margin (hNRWallFactor > ext) is
         * capture-truncated by design: neither the lists nor the tail traversal enumerate
         * beyond 2 * hNRExtFactor * h0, so halos beyond the margin would be dead weight. */
        domain.setHaloFactor(nrParams_.hNRExtFactor);
        /* Symmetric pair sets need the remote big-h side of cross-rank pairs present as a halo
         * even when it is beyond the reach of all local search spheres. */
#warning symmetric halos
        //        domain.setSymmetricHalos(true);
        if (d.g != 0.0)
        {
            domain.syncGrav(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d), get<"m">(d),
                            get<ConservedFields>(d), get<typename Base::DependentFields>(d));
        }
        else
        {
            domain.sync(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d),
                        std::tuple_cat(std::tie(get<"m">(d)), get<ConservedFields>(d)),
                        get<typename Base::DependentFields>(d));
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
#warning treeView.symmetric
        //        d.treeView.symmetric = true;
    }

    void computeForces(DomainType& domain, DataType& simData) override
    {
        timer.start();
        pmReader.start();
        sync(domain, simData);
        timer.step("domain::sync");
        Base::logDomainStats(domain, simData);

        auto& d = simData.hydro;
        cstone::fill(domain.exec(), d.keys.begin(), d.keys.end(), 0);
        d.resize(domain.nParticlesWithHalos());
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        fillMassHalos(domain.exec(), get<"m">(d), first, last);

        computeGroups(first, last, d, domain.box(), groups_);
        timer.step("computeGroups");

        if (nrParams_.particleBallmass == 0 || d.iteration == 1) { initializeBallmass(groups_.view(), d); }

        updateSmoothingLengthIterative(groups_.view(), d, domain.box(), nrParams_.nr_ngmax);
        timer.step("updateSmoothingLengthIterative");
        // h exchange prevents neighbor list overflow (symmetric counterpart)
        domain.exchangeHalos(std::tie(get<"h">(d)), get<"ax">(d), get<"keys">(d));
        timer.step("mpi::synchronizeHalos");
        d.treeView.symmetric_neighbors = true;
        findNeighborsSfc(groups_.view(), d, domain.box(), false);
        timer.step("FindNeighbors");
        pmReader.step();

        // Uses non-converged h; alternative is to use m/rho of the last step (sphynx)
        computeXMass(d, domain.box());
        timer.step("XMass");
        domain.exchangeHalos(std::tie(get<"xm">(d)), get<"ax">(d), get<"keys">(d));
        timer.step("mpi::synchronizeHalos");

        convergeSmoothingLengthNR(domain, simData);

        computeVe(d, domain.box());
        timer.step("Generalized Volume Elements");
        // momentum equation has a dependency on h_j
        domain.exchangeHalos(get<"h", "vx", "vy", "vz", "kx">(d), get<"ax">(d), get<"keys">(d));
        timer.step("mpi::synchronizeHalos");

        Base::computeForcesCommon(domain, simData, true);
    }

    void integrate(DomainType& domain, DataType& simData) override
    {
        auto&  d     = simData.hydro;
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        computeTimestep(first, last, d);
        timer.step("Timestep");
        computePositions(groups_.view(), d, domain.box(), d.minDt, {float(d.minDt_m1)});

        bool haveUnconvergedParticles = setRemovalKeysIfNeighborless(groups_.view(), d);
        if (haveUnconvergedParticles && not d.removeUnconvergedParticles)
        {
            throw std::runtime_error("Neighbor search did not converge\n");
        }

        timer.step("UpdateQuantities");
    }

protected:
    //! @brief Set the neighbour list capacity higher than the maximally allowed number of neighbours
    // to account for the extended neighbour search
    void updateNgmax(unsigned nr_ngmax, DataType& d)
    {
        const float volume_factor = std::pow(nrParams_.hNRExtFactor, 3);
        d.hydro.ngmax             = std::ceil(5.f * nr_ngmax * volume_factor);
        if (Base::rank_ == 0)
        {
            std::cout << "Neighbor-list build capacity set to " << d.hydro.ngmax
                      << " to fit the extended neighbor search of the smoothing-length NR iterations" << std::endl;
        }
    }

    /*! @brief converge the smoothing lengths to the volume-element consistency rho * h^3 = ballmass
     *
     * Newton-Raphson iterations converging h towards the per-particle target
     * rho * h^3 = ballmass (nominally ballmassEta(ng0) * m, see the particleBallmass parameter),
     * such that the grad-h terms are formally consistent with dh/drho = -h / (3 * rho).
     * The iterations reuse the fixed neighbor list with fixed
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

        const float ngmin = nrParams_.particleBallmass ? float(d.ng0 / 2) : 0.0f;
        const float ngmax = nrParams_.particleBallmass ? float(d.ngmax) : std::numeric_limits<float>::max();

        auto& h0_buf    = d.ax;
        auto& h_new_buf = d.ay;
        auto& nc_buf    = d.az;

        cstone::copy_n(domain.exec(), d.h.data(), d.h.size(), h0_buf.data());

        std::vector<size_t> n_unconverged;
        size_t              n_ballmass_reset_tot = 0;

        for (size_t i = 0; i < nrParams_.hNRIterMax; i++)
        {
            using cstone::rawPtr;
            computeVeNR(d, nrParams_.hNRExtFactor, nrParams_.hNRWallFactor, nrParams_.hNRTol, ngmin, ngmax,
                        rawPtr(h0_buf), rawPtr(h_new_buf), rawPtr(nc_buf));
            const NRStatistics stats = computeNRStat(groups_.view(), d, rawPtr(h_new_buf), nrParams_.hNRTol);
            cstone::copy_n(domain.exec(), h_new_buf.data(), h_new_buf.size(), d.h.data());

            n_unconverged.push_back(stats.n_unconverged);
            n_ballmass_reset_tot += stats.n_ballmass_reset;
            if (stats.n_unconverged == 0) { break; }
        }

        printNRUnconverged(n_unconverged, n_ballmass_reset_tot);
        timer.step("hNewtonRaphson");
    }

    void printNRUnconverged(const std::vector<size_t>& n_unconverged, size_t n_ballmass_reset_tot)
    {
        std::vector<size_t> values_local(n_unconverged.size() + 1);
        values_local[0] = n_ballmass_reset_tot;
        std::copy(n_unconverged.begin(), n_unconverged.end(), values_local.begin() + 1);
        std::vector<size_t> values_global(values_local.size());
        MPI_Reduce(values_local.data(), values_global.data(), values_local.size(), MpiType<size_t>{}, MPI_SUM, 0,
                   MPI_COMM_WORLD);
        if (Base::rank_ == 0)
        {
            std::printf("Ballmass adjusted: %zu\n", values_global[0]);
            std::printf("unconverged particles:");
            for (size_t i = 1; i < values_global.size(); i++)
            {
                std::printf("\t%zu", values_global[i]);
                if (values_global[i] == 0) { break; }
            }
            std::printf("\n");
        }
    }
};

} // namespace sphexa
