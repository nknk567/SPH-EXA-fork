//
// Created by Noah Kubli on 04.03.2026.
//

#pragma once

#include "cstone/fields/field_get.hpp"
#include "init/settings.hpp"
#include "io/ifile_io.hpp"

namespace visual
{
template<typename DomainType, typename Dataset>
struct Visualizer
{

    using ConservedFields = util::FieldList<"u", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1", "id">;

    //! @brief the list of dependent particle fields, these may be used as scratch space during domain sync
    using DependentFields =
        util::FieldList<"rho", "p", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33", "nc">;

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

    void visualize(DomainType& domain, Dataset& simData)
    {
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();
        printf("number of particles: %zu\n", last - first);
        //        sizeCategorization();
        //        renderSmall();
        //        renderMedium();
        //        renderLarge();
        //        reduceImages();
    }
};
} // namespace visual
