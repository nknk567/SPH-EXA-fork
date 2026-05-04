//
// Created by Noah Kubli on 04.05.2026.
//

#pragma once

#include "cstone/fields/field_get.hpp"
#include "cstone/util/value_list.hpp"
#include <type_traits>

namespace visual
{
template<typename T, typename Th, typename Trho, typename Tm, typename Ta>
struct RenderFieldsSpan
{
    const T*    x;
    const T*    y;
    const T*    z;
    const Th*   h;
    const Trho* rho;
    const Tm*   m;
    const Ta*   render_quantity;
    size_t      size;
};

template<util::StructuralString RenderQuantity, typename Dataset>
auto makeRenderFieldsSpan(Dataset& d, size_t first, size_t last)
{
    using T    = typename std::decay_t<decltype(d.x)>::value_type;
    using Th   = typename std::decay_t<decltype(d.h)>::value_type;
    using Trho = typename std::decay_t<decltype(d.rho)>::value_type;
    using Tm   = typename std::decay_t<decltype(d.m)>::value_type;

    const auto& render_quantity = get<RenderQuantity>(d);
    using Ta                    = typename std::decay_t<decltype(render_quantity)>::value_type;

    RenderFieldsSpan<T, Th, Trho, Tm, Ta> span;

    span.x               = rawPtr(d.x) + first;
    span.y               = rawPtr(d.y) + first;
    span.z               = rawPtr(d.z) + first;
    span.h               = rawPtr(d.h) + first;
    span.rho             = rawPtr(d.rho) + first;
    span.m               = rawPtr(d.m) + first;
    span.render_quantity = rawPtr(render_quantity) + first;
    span.size            = last - first;
    return span;
}
} // namespace visual
