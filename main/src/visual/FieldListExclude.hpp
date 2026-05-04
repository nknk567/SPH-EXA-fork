//
// Created by Noah Kubli on 04.05.2026.
//

#pragma once

namespace visual
{

//! @brief Generate FieldList by excluding a certain field (by name) if included.
template<util::StructuralString exclude, typename Fields>
struct Exclude
{
};

template<util::StructuralString exclude, util::StructuralString field, util::StructuralString... fields>
struct Exclude<exclude, util::FieldList<field, fields...>>
{
    using rest = typename Exclude<exclude, util::FieldList<fields...>>::type;
    using eq   = std::bool_constant<exclude == field>;
    using type = std::conditional_t<eq::value, rest, decltype(util::FieldList<field>{} + rest{})>;
};

template<util::StructuralString exclude>
struct Exclude<exclude, util::FieldList<>>
{
    using type = util::FieldList<>;
};

template<util::StructuralString exclude, typename Fields>
using Exclude_t = typename Exclude<exclude, Fields>::type;

//! @brief Generate FieldList by excluding certain fields if included.
template<typename FieldListType, typename Fields>
struct Excludes
{
};

template<typename Fields, util::StructuralString exclude, util::StructuralString... excludes>
struct Excludes<util::FieldList<exclude, excludes...>, Fields>
{
    using recType = typename Excludes<util::FieldList<excludes...>, Fields>::type;
    using type    = Exclude_t<exclude, recType>;
};

template<typename Fields, util::StructuralString exclude>
struct Excludes<util::FieldList<exclude>, Fields>
{
    using type = Exclude_t<exclude, Fields>;
};

template<typename FieldListType, typename Fields>
using Excludes_t = typename Excludes<FieldListType, Fields>::type;

} // namespace visual
