/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Cosmological parameters and associated functions.
 *
 * @author Jonathan Coles <jonathan.coles@cscs.ch>
 */

#pragma once

#include <memory>
#include <cmath>
#include <cfloat>

#define DBG_COSMO_UTILS if (0)

template<std::floating_point T, int max_iter = 40, typename F1, typename F2>
T find_zero_newton(F1&& func, F2&& func_deriv, T x0, T xmin, T xmax, const T rel_error = 1e-7, const T abs_error = 1e-7)
{
    if (xmin > xmax) throw std::invalid_argument("Lower bound exceeds upper bound in Newton's Method.");

    //    T x0_save = x0;
    T x1 = x0;

    for (int i = 0; i < max_iter; i++)
    {
        T f = func(x0);
        if (f == 0) return x0;

        T fprime = func_deriv(x0);

        //        DBG_COSMO_UTILS printf(
        //            "%2i] %23.15e  ->  %23.15e  [xmin: %23.15e xmax: %23.15e] (diff: %23.15e  f: %23.15e  f':
        //            %23.15e)\n", i, x0_save, x0, xmin, xmax, fabs((x1 - x0_save)), f, fprime);

        if (f * fprime == 0)
        {
            x0 = std::min(xmax, x0 + abs_error * (xmax - xmin));
            continue;
        }

        if (f * fprime < 0)
            xmin = x0;
        else
            xmax = x0;

        //        x0_save = x0;

        x1 = x0 - f / fprime;

        if (std::abs(x1 - x0) <= std::max(rel_error * std::max(std::abs(x0), std::abs(x1)), abs_error)) { return x1; }

        if (xmin < x1 && x1 < xmax) { x0 = x1; }
        else { x0 = 0.5 * (xmin + xmax); }
    }

    DBG_COSMO_UTILS return -1234;
    throw std::runtime_error("Maximum number of iterations reached in Newton's Method.");
}

/*
 ** Romberg integrator for an open interval.
 */

template<typename T, int max_level = 13, typename F>
T integrate_romberg(F&& func, T a, T b, T eps)
{
    T   tllnew = (b - a) * func(0.5 * (b + a));
    T   tll    = std::numeric_limits<T>::max();
    T   tlk[max_level + 1];
    int n        = 1;
    int nsamples = 1;

    tlk[0] = tllnew;
    //    = (b - a) * func(0.5 * (b + a));
    if (a == b) return tllnew;

    //    tll = std::numeric_limits<T>::max();

    while ((std::abs((tllnew - tll) / tllnew) > eps) && (n < max_level))
    {
        /*
         * midpoint rule.
         */

        nsamples *= 3;
        T dx = (b - a) / nsamples;

        T s = 0;
        for (int i = 0; i < nsamples / 3; i++)
        {
            s += dx * func(a + (3 * i + 0.5) * dx);
            s += dx * func(a + (3 * i + 2.5) * dx);
        }

        T tmp  = tlk[0];
        tlk[0] = tlk[0] / 3.0 + s;

        /*
         * Romberg extrapolation.
         */

        for (int i = 0; i < n; i++)
        {
            T k      = std::pow(9.0, i + 1.0);
            T tlknew = (k * tlk[i] - tmp) / (k - 1.0);

            tmp        = tlk[i + 1];
            tlk[i + 1] = tlknew;
        }

        tll    = tllnew;
        tllnew = tlk[n];
        n++;
    }

    // printf("%23.15e\n", tllnew);
    // printf("%23.15e\n", tll);
    // printf("%23.15e\n", fabs((tllnew-tll)/(tllnew)));
    // printf("%23.15e\n", eps);
    // assert(fabs((tllnew-tll)/(tllnew)) <= eps);

    return tllnew;
}
