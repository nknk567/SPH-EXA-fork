//
// Created by Noah Kubli on 04.03.2026.
//

#pragma once

#include "cstone/fields/field_get.hpp"
#include "cstone/traversal/groups.hpp"
#include "sph/sph.hpp"
#include "util/timer.hpp"

#include "grid.hpp"
#include "init/settings.hpp"
#include "io/ifile_io.hpp"
#include "size_categorization.hpp"
#include "render_small.hpp"
#include "render_medium.hpp"
#include "FieldListExclude.hpp"
#include "RenderFieldsSpan.hpp"

namespace visual
{

inline void dump_to_file_c(const std::vector<double>& data, const char* filename)
{
    FILE* f = fopen(filename, "w");

    for (double v : data)
    {
        fprintf(f, "%.17g\n", v);
    }

    fclose(f);
}

template<template<typename> typename FieldVector>
struct RenderData
{
    //
    FieldVector<uint8_t> size_category;

    // Buffers
    FieldVector<double>   buf1;
    FieldVector<float>    buf2;
    FieldVector<unsigned> buf3;
    FieldVector<size_t>   buf4;
    FieldVector<uint64_t> buf5;
    FieldVector<uint8_t>  buf6;
    FieldVector<size_t>   buf7;
    FieldVector<size_t>   buf8;

    auto buffers() { return std::tie(buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8); }
};

template<typename DomainType, typename Dataset>
struct Visualizer
{
    using Acc = typename Dataset::AcceleratorType;
    cstone::GroupData<Acc> groups_;

    using ConservedFields = util::FieldList<"u", "vx", "vy", "vz">;

    //! @brief the list of dependent particle fields, these may be used as scratch space during domain sync
    //    using DependentFields =
    //        util::FieldList<"rho", "p", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33", "nc">;
    using DependentFields =
        util::FieldList<"rho", "p", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33", "nc">;
    using RenderingFields = util::FieldList<"rho">;
    //    using BufferTypes     = Excludes<RenderingFields, DependentFields>;
    using BufferTypes =
        util::FieldList<"p", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33", "nc">;

    RenderData<Dataset::HydroData::template FieldVector> render_data;

    const size_t rank;
    Visualizer(const size_t rank, const sphexa::InitSettings& settings)
        : rank(rank)
    {
    }

    std::vector<std::string> conservedFields() const
    {
        std::vector<std::string> ret{"x", "y", "z", "h", "m"};
        for_each_tuple([&ret](auto f) { ret.push_back(f.value); }, make_tuple(ConservedFields{}));
        return ret;
    }

    void activateFields(Dataset& simData) const
    {
        auto& d = simData.hydro;

        //! @brief Fields accessed in domain sync are not part of extensible lists.
        d.setConserved("x", "y", "z", "h", "m");
        d.setDependent("keys");
        std::apply([&d](auto... f) { d.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.setDependent(f.value...); }, make_tuple(DependentFields{}));
    }

    void load(const std::string& initCond, sphexa::IFileReader* reader) {}

    void sync(DomainType& domain, Dataset& simData)
    {
        auto& d = simData.hydro;
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
    }

    std::vector<double> render(const Grid& grid, Dataset& simData, size_t first, size_t last)
    {
        auto& d = simData.hydro;

        sphexa::Timer timer(std::cout);
        timer.start();

        //        typename Dataset::HydroData::FieldVector<size_t> tile_gpu;
        //        typename Dataset::HydroData::FieldVector<uint8_t> tile_gpu(last - first);
        //        timer.step("Device vector");

        //        std::vector<size_t> tile;
        //        const auto fields = makeRenderFieldsSpan<"rho">(d, first, last);

        const auto [n_small, n_large] = sizeCategorization(first, last, d, grid, render_data.size_category);
        timer.step("sizeCategorization");

        printf("n_small: %zu\n", n_small);
        printf("n_large: %zu\n", n_large);

        sortByCategory<ConservedFields, RenderingFields, BufferTypes>(first, last, d, render_data.size_category,
                                                                      render_data);
        timer.step("sortByCategory");

        // sort order:
        // small particles; large particles with ascending tile size; out of bound

        //        std::vector<double> result(grid.pixel_width * grid.pixel_height, 0.);
        typename Dataset::HydroData::FieldVector<double> result(grid.pixel_width * grid.pixel_height, 0.);
        renderSmall<BufferTypes>(first, first + n_small, d, grid, result, render_data);
        timer.step("renderSmall");

        renderMedium(first + n_small, first + n_small + n_large, d, grid, result);
        timer.step("renderMedium");
        return toHost(result);
    }
    void visualize(DomainType& domain, Dataset& simData)
    {
        sphexa::Timer timer(std::cout);
        timer.start();
        sync(domain, simData);
        timer.step("sync");

        auto& d = simData.hydro;
        d.resize(domain.nParticlesWithHalos());
        resizeNeighbors(d, domain.nParticles() * d.ngmax);

        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();
        domain.exchangeHalos(std::tie(get<"m">(d)), get<"ax">(d), get<"ay">(d));
        sph::findNeighborsSfc(first, last, d, domain.box());
        sph::computeGroups(first, last, d, domain.box(), groups_);
        sph::computeDensity(groups_.view(), d, domain.box());
        timer.step("SPH");

        printf("number of particles: %zu\n", last - first);
        render_data.size_category.resize(last - first);
        timer.step("resize");

        Grid                grid;
        std::vector<double> result = render(grid, simData, first, last);
        timer.step("render 1");
        dump_to_file_c(result, "render.txt");
        timer.step("write 1");

        Grid                grid2(-1000, 1000, -1000);
        std::vector<double> result2 = render(grid2, simData, first, last);
        timer.step("render 2");
        dump_to_file_c(result2, "render2.txt");
        timer.step("write 2");
    }
};
} // namespace visual
