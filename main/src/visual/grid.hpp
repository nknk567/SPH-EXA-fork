//
// Created by Noah Kubli on 04.03.2026.
//

#pragma once

namespace visual
{

struct Grid
{
    double xmin         = -10.;
    double xmax         = 10.;
    double ymin         = -10.;
    double ymax         = 10.;
    size_t pixel_width  = 1024;
    size_t pixel_height = 1024;
    double delta_x      = (xmax - xmin) / pixel_width;
    double delta_y      = (ymax - ymin) / pixel_height;
    size_t tile_size    = 8;
    double h_medium_max = delta_x * tile_size;
    double h_small_max  = 0.5 * delta_x;
};

} // namespace visual