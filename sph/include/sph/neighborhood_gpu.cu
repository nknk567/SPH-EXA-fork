#include "neighborhood_gpu.hpp"

namespace sph
{

void DeviceNeighborhoodData::disableNeighborLists() { impl->useNeighborLists = false; }
bool DeviceNeighborhoodData::neighbourListsEnabled() { return impl->useNeighborLists; }

DeviceNeighborhoodData::DeviceNeighborhoodData()
    : impl(std::make_unique<Impl>())
{
}
DeviceNeighborhoodData::~DeviceNeighborhoodData() {}

} // namespace sph
