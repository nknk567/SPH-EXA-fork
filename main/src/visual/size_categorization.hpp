//
// Created by Noah Kubli on 04.03.2026.
//

#pragma once

#include "cstone/cuda/device_vector.h"

#include "grid.hpp"
#include "size_categorization_helper.hpp"
#include "size_categorization_gpu.hpp"
#include <tuple>
#include <vector>

#include "util/timer.hpp"

namespace visual
{

template<typename Dataset>
std::tuple<size_t, size_t> sizeCategorizationImpl(size_t startIndex, size_t endIndex, Dataset& d, const Grid& g,
                                                  auto& tileVec)
{
    using T = typename std::decay_t<decltype(tileVec)>::value_type;

    for (size_t i = startIndex; i < endIndex; i++)
    {
        tileVec[i - startIndex] = categorize<T>(d.x[i], d.y[i], d.z[i], d.h[i], g);
    }
    const size_t n_small = std::count(tileVec.begin(), tileVec.end(), T(0));
    const size_t n_large = std::count(tileVec.begin(), tileVec.end(), T(1));
    return std::make_tuple(n_small, n_large);
}

// Particles outside: -2; small particles inside: -1; rest: 0
template<class Dataset, typename Tc>
std::tuple<size_t, size_t> sizeCategorization(size_t startIndex, size_t endIndex, Dataset& d, const Grid& g,
                                              Tc& size_category)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        return sizeCategorizationGPU(startIndex, endIndex, rawPtr(d.x), rawPtr(d.y), rawPtr(d.z), rawPtr(d.h), g,
                                     size_category);
    }
    else { return sizeCategorizationImpl(startIndex, endIndex, d, g, size_category); }
}

template<typename Tk, class... Arrays1, class... Arrays2>
void gather(std::span<const Tk> ordering, const size_t offset, std::tuple<Arrays1&...> arrays,
            std::tuple<Arrays2&...> scratchBuffers)
{
    auto reorderArray = [ordering, &scratchBuffers, offset](auto& array)
    {
        using VectorRef  = decltype(array);
        using VectorType = std::decay_t<VectorRef>;
        auto& swapSpace  = util::pickType<decltype(array)>(scratchBuffers);
        swapSpace.resize(array.size());
        cstone::gatherAcc<IsDeviceVector<VectorType>{}>(ordering, rawPtr(array) + offset, rawPtr(swapSpace) + offset);
        swap(swapSpace, array);
    };

    util::for_each_tuple(reorderArray, arrays);
}

template<typename ConservedFields, typename DependentFields, typename BufferFields, typename Dataset>
void sortByKeysImpl(size_t startIndex, size_t endIndex, Dataset& d, auto& keysVec)
{
    //    std::vector<size_t> order(keysVec.size());
    std::iota(d.keys.begin(), d.keys.end(), 0zu);

    cstone::sort_by_key(keysVec.begin(), keysVec.end(), d.keys.begin());

    std::vector<double>   buf1;
    std::vector<float>    buf2;
    std::vector<unsigned> buf3;
    std::vector<uint64_t> buf4;

    auto buffers = std::tie(buf1, buf2, buf3, buf4);

    using DefaultFields = util::FieldList<"x", "y", "z", "h", "m", "keys">;

    using Fields = decltype(DefaultFields{} + ConservedFields{} + DependentFields{});
    gather(std::span(std::as_const(d.keys)), startIndex, get<Fields>(d), buffers);
}

template<typename ConservedFields, typename RenderingFields, typename BufferFields, typename Dataset,
         typename RenderData>
void sortByKeysGPU(size_t startIndex, size_t endIndex, Dataset& d, auto& keysVecDevice, RenderData& render_data)
{
    //    printf("start sortByKeysGPU\n");
    //    sphexa::Timer timer(std::cout);
    //    timer.start();

    using RenderCategoryType = typename std::decay_t<decltype(keysVecDevice)>::value_type;

    //    cstone::DeviceVector<size_t> order(keysVecDevice.size());
    //    timer.step("DeviceVector");
    assert(keysVecDevice.size() <= d.keys.size());
    cstone::sequenceAcc<true>(d.keys.begin(), d.keys.end(), size_t(0));
    //    timer.step("sequenceAcc");

    // oder cstone::sequece(0, order.size(), order.data (rawptr), 1.0); // growth rate = 1

    //    cstone::sort_by_key(keysVec.begin(), keysVec.end(), order.begin());
    //    raw pointer cast
    // Allocate key and value buffers
    //    cstone::DeviceVector<size_t> keysVecDevice(keysVec);

    //    cstone::DeviceVector<RenderCategoryType> key_buffer(keysVecDevice.size());
    using KeyType = typename Dataset::KeyType;
    //    cstone::DeviceVector<KeyType> value_buffer(keysVecDevice.size() * 4);
    //    timer.step("DeviceVector");

    cstone::sortByKey<true>(std::span<RenderCategoryType>(rawPtr(keysVecDevice), keysVecDevice.size()),
                            std::span<KeyType>(rawPtr(d.keys), keysVecDevice.size()), render_data.buf6,
                            render_data.buf5, 1.0);
    //    cstone::sortByKeyGpu(rawPtr(keysVecDevice), rawPtr(keysVecDevice) + keysVecDevice.size(), rawPtr(d.keys));
    //    timer.step("sortByKey");

    //    cstone::DeviceVector<double>   buf1;
    //    cstone::DeviceVector<float>    buf2;
    //    cstone::DeviceVector<unsigned> buf3;
    //    cstone::DeviceVector<uint64_t> buf4;

    //    auto buffers = std::tie(buf1, buf2, buf3, buf4);

    using DefaultFields = util::FieldList<"x", "y", "z", "h", "m", "keys">;

    //    using Fields = decltype(DefaultFields{} + ConservedFields{} + DependentFields{});
    using Fields = decltype(DefaultFields{} + ConservedFields{} + RenderingFields{});

    gather(
        std::span<const KeyType>(rawPtr(d.keys), keysVecDevice.size()), startIndex, get<Fields>(d),
        std::tuple_cat(
            get<BufferFields>(d),
            // std::tuple_cat(get<"p", "c", "ax", "ay", "az", "du", "c11", "c12", "c13", "c22", "c23", "c33", "nc">(d),
            render_data.buffers()));
    //    timer.step("gather");
}

template<typename ConservedFields, typename DependentFields, typename BufferFields, class Dataset, typename Tc,
         typename RenderData>
void sortByKey(size_t startIndex, size_t endIndex, Dataset& d, Tc& size_category, RenderData& render_data)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        sortByKeysGPU<ConservedFields, DependentFields, BufferFields>(startIndex, endIndex, d, size_category,
                                                                      render_data);
    }
    else { sortByKeysImpl<ConservedFields, DependentFields, BufferFields>(startIndex, endIndex, d, size_category); }
}

} // namespace visual