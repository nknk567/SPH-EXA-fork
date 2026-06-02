#include "neighborhood_gpu.hpp"

namespace sph
{

template <bool depth_first>
DeviceNeighborhoodData<depth_first>::DeviceNeighborhoodData()
    : impl(std::make_unique<Impl>())
{
}
template <bool depth_first>
DeviceNeighborhoodData<depth_first>::~DeviceNeighborhoodData() {}

template class DeviceNeighborhoodData<true>;
template class DeviceNeighborhoodData<false>;

} // namespace sph
