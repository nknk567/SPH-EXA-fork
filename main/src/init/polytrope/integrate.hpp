//
//  integrate.hpp.h
//  Polytrop
//
//  Created by Noah Kubli on 14.02.2025.
//

#pragma once

#include "cstone/util/array.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

template <typename F, size_t N>
auto euler_step(F &&func, const util::array<double, N> &y, double t, double delta_t)
{
    return y + func(y, t) * delta_t;
    //Integrate with two steps
    //Estimate error
}

template <typename F, size_t N>
auto midpoint_step(F &&func, const util::array<double, N> &y, double t, double delta_t)
{
    auto y_mid = y + func(y, t) * delta_t / 2.;
    auto t_mid = t + delta_t / 2.;
    return y + func(y_mid, t_mid) * delta_t;
    //Integrate with two steps
    //Estimate error
}

template <typename F, size_t N>
auto rk4(F &&func, const util::array<double, N> &y, double t, double delta_t)
{
    const auto k1 = func(y, t);
    const auto k2 = func(y + k1 * (delta_t / 2.), t + delta_t / 2.);
    const auto k3 = func(y + k2 * (delta_t / 2.), t + delta_t / 2.);
    const auto k4 = func(y + k3 * delta_t, t + delta_t);
    
    return y + (k1 + k2 * 2. + k3 * 2. + k4) * (1. / 6.) * delta_t;
    //Integrate with two steps
    //Estimate error
}

size_t estimated_steps(double integration_distance, double delta_t) { return integration_distance / delta_t; }

//t_end > t_0
template <typename F, size_t N>
auto integrate_step(F &&func, double t, double delta_t, const util::array<double, N> y)
{
    const double t_new            = t + delta_t;
    const auto   y_new_0          = rk4(func, y, t, delta_t);
    const auto   y_new_1_halfstep = rk4(func, y, t, delta_t / 2.);
    const auto   y_new_1          = rk4(func, y_new_1_halfstep, t + delta_t / 2., delta_t / 2.);

    const double error = max(abs(y_new_1) - abs(y_new_0));
    return std::make_tuple(t_new, y_new_1, error);
}

double new_step_size_factor(double error, double target_error)
{
    if (std::isnan(error)) { return 0.5; }
    return std::max(0.5, std::min(1.5, std::sqrt(target_error / (error))));
}

template <typename F, size_t N>
auto integrate(F &&func, const util::array<double, N> &y_0, double t_0, double t_end, double maximal_error = 0.01)
{
    const double target_error = 0.5 * maximal_error;

    double       delta_t     = 0.01;
    const double n_estimated = estimated_steps(t_end - t_0, delta_t);

    std::vector<double>       t{t_0};
    std::vector<util::array<double,N>> y{y_0};
    t.reserve(n_estimated);
    y.reserve(n_estimated);

    size_t i        = 0;
    size_t it_count = 0;
    while (t.back() < t_end)
    {
        delta_t = std::min(delta_t, t_end - t.back());
        const auto [t_new, y_new, error] = integrate_step(func, t[i], delta_t, y[i]);

        double new_delta_t = delta_t * new_step_size_factor(error, target_error);
        if (has_nan(y_new)) { new_delta_t = delta_t / 2.; }

        bool accept = error <= maximal_error && !has_nan(y_new) && !std::isnan(error);
        
        if (accept)
        {
            t.push_back(t_new);
            y.push_back(y_new);
            i++;
        }
        delta_t = new_delta_t;
        it_count++;
    }
    printf("iterations: %zu\t accepted: %zu\t relation: %g\n", it_count, i, double(i) / it_count);
    return std::pair{t, y};
}

template <size_t N>
bool has_nan(const util::array<double, N> &a) noexcept
{
    for (size_t i = 0; i < N; i++)
    {
        if (std::isnan(a[i]))
        {
            return true;
        }
    }
    return false;
}

//Integrate until the first zero. Make the timestep smaller if it becomes negative
template <typename F, size_t N>
auto integrate_to_zero(F &&func, const util::array<double, N> &y_0, double t_0, double t_end, double min_step_size = 1e-3,
                       double maximal_error = 0.01)
{
    const double target_error = 0.5 * maximal_error;

    double       delta_t     = 0.01;
    const double n_estimated = estimated_steps(t_end - t_0, delta_t);

    std::vector<double>       t{t_0};
    std::vector<util::array<double, N>> y{y_0};
    t.reserve(n_estimated);
    y.reserve(n_estimated);

    auto close_to_zero = [&](double x) { return std::abs(x) < maximal_error; };
    
    size_t i        = 0;
    size_t it_count = 0;
    while (t.back() < t_end && !close_to_zero(y.back()[0]))
    {
        delta_t = std::min(min_step_size, std::min(delta_t, t_end - t.back()));
        const auto [t_new, y_new, error] = integrate_step(func, t[i], delta_t, y[i]);

        double new_delta_t = delta_t * new_step_size_factor(error, target_error);
        if (has_nan(y_new) || y_new[0] < 0.) { new_delta_t = delta_t / 2.; }

        bool accept = error <= maximal_error && !has_nan(y_new) && !std::isnan(error) && y_new[0] >= 0.;
        
        if (accept)
        {
            t.push_back(t_new);
            y.push_back(y_new);
            i++;
        }
        delta_t = new_delta_t;

        it_count++;
    }
    std::printf("iterations: %zu\t accepted: %zu\t relation: %g\n", it_count, i, double(i) / it_count);
    return std::pair{t, y};
}
//template <typename F1, typename F2>
//struct CachedIntegrator
//{
//    const F1 &f_ode;
//    const F2 &f_asympt;
//
//    const double              maximal_error    = 0.01;
//    const double              asymptotic_limit = 1e-4;
//    const double              theta_0          = 1.0;
//    std::vector<double>       ts{asymptotic_limit};
//    std::vector<util::Vec<2>> ys{f_asympt(1.0, asymptotic_limit)};
//
//    double integrate_cached(const double t, const size_t i)
//    {
//        auto [t_new, y_new] = integrate(f_ode, ys[i], ts[i], t, maximal_error);
//        if (i < ts.size() + 1 && t_new.back() > t)
//            throw std::runtime_error(
//                "Inserted before end but the last t-value is greater than the next one in the cache");
//        ts.insert(ts.begin() + i + 1, t_new.begin() + 1, t_new.end());
//        ys.insert(ys.begin() + i + 1, y_new.begin() + 1, y_new.end());
//        return y_new.back()[0];
//    }
//
//    size_t get_n_before(const double t)
//    {
//        const auto first_larger = std::upper_bound(ts.begin(), ts.end(), t);
//        assert(first_larger - ts.begin() > 0);
//        const auto   closest_smaller = (first_larger - 1);
//        const size_t n_smaller       = closest_smaller - ts.begin();
//        return n_smaller;
//    }
//
//    double operator()(const double t)
//    {
//        if (t <= asymptotic_limit) return f_asympt(theta_0, t)[0];
//        if (t >= ts.back()) { return integrate_cached(t, ts.size() - 1); }
//        else
//        {
//            const size_t n = get_n_before(t);
//            return integrate_cached(t, n);
//        }
//    }
//};

//template <typename F1, typename F2>
//CachedIntegrator(F1, F2) -> CachedIntegrator<F1, F2>;
//
//template <typename F1, typename F2>
//CachedIntegrator(F1, F2, double, double, double) -> CachedIntegrator<F1, F2>;
