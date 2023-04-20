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
#include <cfloat>
#include <cassert>
#include <array>
#include <vector>
#include <variant>
#include <iostream>
#include <cmath>
#include "utils.hpp"

#define DBG_COSMO if (0)

namespace cosmo
{


template<typename T>
class Cosmology // : public cstone::FieldStates<CosmologyData<T>>
{
    //virtual T H(const T a) = 0;
    //virtual T time(const T a) = 0;
    //virtual T a(T t) = 0;
    virtual T driftTimeCorrection(T t, T dt) = 0;
    virtual T kickTimeCorrection(T t, T dt) = 0;
};

template<typename T>
class StaticUniverse : Cosmology<T>// : public cstone::FieldStates<CosmologyData<T>>
{
public:
    StaticUniverse() = default;

    T driftTimeCorrection(T t, T dt)
    {
        return dt;
    };

    T kickTimeCorrection(T t, T dt)
    {
        return dt;
    }
};

template<typename T, int rel_prec = -7, int abs_prec = -7>
class LambdaCDM : Cosmology<T> // : public cstone::FieldStates<CosmologyData<T>>
{
public:

    T relativeError = pow(10, rel_prec);
    T absoluteError = pow(10, abs_prec);
    
    //! @brief Hubble Constant today [km/Mpc/s]
    T H0{0.0};
    //! @brief Matter Density
    T OmegaMatter{0.0};
    //! @brief Radiation Density
    T OmegaRadiation{0.0};
    //! @brief Cosmology Constant (vacuum density)
    T OmegaLambda{0.0};

    struct Parameters
    {
        T H0, OmegaMatter, OmegaRadiation, OmegaLambda;
    };

    //
    // Values from 2018 Planck final data release, TT,TE,EE+lowE+lensing+BAO
    //
    static constexpr Parameters Planck2018 =
    {
        .H0 = 67.66,
        .OmegaMatter = 0.3111,
        .OmegaRadiation = 0.0,
        .OmegaLambda = 0.6889,
    };


    LambdaCDM() = default;

    LambdaCDM(struct Parameters p)
        : LambdaCDM(p.H0, p.OmegaMatter, p.OmegaRadiation, p.OmegaLambda)
    {
    }

    LambdaCDM(T H0, T OmegaMatter, T OmegaRadiation, T OmegaLambda) 
        : H0(H0)
        , OmegaMatter(OmegaMatter)
        , OmegaRadiation(OmegaRadiation)
        , OmegaLambda(OmegaLambda)
    {
        if (H0 <= 0)
            throw std::domain_error("Hubble0 parameter must be strictly positive.");

        if (OmegaMatter < 0)
            throw std::domain_error("OmegaMatter parameter must be positive.");

        if (OmegaRadiation < 0)
            throw std::domain_error("OmegaRadiation parameter must be positive.");

        if (OmegaLambda < 0)
            throw std::domain_error("OmegaLambda parameter must be positive.");

        if (OmegaRadiation == 0 && OmegaLambda == 0)
            throw std::domain_error("In LambdaCDM at least one of OmegaRadiation or OmegaLambda must be positive. Otherwise use CDM.");

        if (OmegaMatter == 0)
        {
            if (OmegaRadiation == 0)
                throw std::domain_error("OmegaRadiation parameter must be strictly positive when OmegaMatter is zero.");
        }
    }

    template<class Archive>
    void loadOrStoreAttributes(Archive* ar)
    {
        //! @brief load or store an attribute, skips non-existing attributes on load.
        auto optionalIO = [ar](const std::string& attribute, auto* location, size_t attrSize)
        {
            try
            {
                ar->stepAttribute(attribute, location, attrSize);
            }
            catch (std::out_of_range&)
            {
                std::cout << "Attribute " << attribute << " not set in file, setting to default value " << *location
                          << std::endl;
            }
        };

        optionalIO("Hubble0", &H0, 1);
        optionalIO("OmegaMatter", &OmegaMatter, 1);
        optionalIO("OmegaRadiation", &OmegaRadiation, 1);
        optionalIO("OmegaLambda", &OmegaLambda, 1);
    }

    T H(const T a)
    {
        T Omega0 = OmegaMatter + OmegaRadiation + OmegaLambda;
        T OmegaCurvature = T(1) - Omega0;

        T a2 = a * a;
        T a3 = a * a2;
        T a4 = a * a3;
        return (H0/a2) * sqrt(OmegaRadiation + OmegaMatter * a + OmegaCurvature
                * a2 + OmegaLambda * a4);
    }

    T time(const T a)
    {
        auto f = [this](const T x) { return 2. / (3. * x * H(pow(x, 2./3.))); };
        return romberg<T>(f, 0.0, pow(a, 1.5), 1e-2 * relativeError);
    }

    T a(T t)
    {
        auto f = [this, t](const T a) { return time(a) - t; };
        auto df = [this](const T a) { return 1.0/(a * H(a)); };

        return newton(f, df, t*H0, 0.0, 1.0e38, relativeError, absoluteError);
    }

    T driftTimeCorrection(T t, T dt)
    {
        auto f = [this](const T x) { return -x / H(T(1.0) / x); };
        return romberg<T>(f, T(1.0) / a(t), T(1.0) / a(t + dt), relativeError);
    }

    T kickTimeCorrection(T t, T dt)
    {
        auto f = [this](const T x) { return -T(1.0) / H(T(1.0) / x); };
        return romberg<T>(f, T(1.0) / a(t), T(1.0) / a(t + dt), relativeError);
    }
};

template<typename T, int rel_prec = -7, int abs_prec = -7>
class CDM : Cosmology<T> // : public cstone::FieldStates<CosmologyData<T>>
{
public:

    T relativeError = pow(10, rel_prec);
    T absoluteError = pow(10, abs_prec);
    
    //! @brief Hubble Constant today [km/Mpc/s]
    T H0{0.0};
    //! @brief Matter Density
    T OmegaMatter{0.0};

    struct Parameters
    {
        T H0, OmegaMatter;
    };
    
    CDM() = default;

    CDM(struct Parameters cd) : CDM(cd.H0, cd.OmegaMatter)
    {
    }

    CDM(T H0, T OmegaMatter) : H0(H0), OmegaMatter(OmegaMatter)
    {
        if (H0 <= 0)
            throw std::domain_error("Hubble0 parameter must be strictly positive.");

        if (OmegaMatter < 0)
            throw std::domain_error("OmegaMatter parameter must be positive.");

        // apparantly we can have a universe with nothing in it?!?
    }

    template<class Archive>
    void loadOrStoreAttributes(Archive* ar)
    {
        //! @brief load or store an attribute, skips non-existing attributes on load.
        auto optionalIO = [ar](const std::string& attribute, auto* location, size_t attrSize)
        {
            try
            {
                ar->stepAttribute(attribute, location, attrSize);
            }
            catch (std::out_of_range&)
            {
                std::cout << "Attribute " << attribute << " not set in file, setting to default value " << *location
                          << std::endl;
            }
        };

        optionalIO("Hubble0", &H0, 1);
        optionalIO("OmegaMatter", &OmegaMatter, 1);
    }

    T H(const T a)
    {
        T OmegaCurvature = T(1) - OmegaMatter;

        T a2 = a * a;
        T a3 = a * a2;
        return (H0/a2) * sqrt(OmegaMatter * a + OmegaCurvature * a2);
    }

    T time(const T a)
    {
        if (OmegaMatter == 1.0)
        {
            return a == 0.0
                   ? 0.0
                   : 2.0 / (3.0 * H0) * pow(a,1.5);
        }

        if (OmegaMatter > 1.0)
        {
            if (H0 == 0.0)
            {
                T B = 1.0 / sqrt(OmegaMatter);
                T eta = acos(1.0 - a);
                return B * (eta - sin(eta));
            }

            if (a == 0.0) return 0.0;

            T a0 = 1.0 / H0 / sqrt(OmegaMatter - 1.0);
            T A = 0.5 * OmegaMatter / (OmegaMatter - 1.0);
            T B = a0 * A;
            T eta = acos(1.0 - a/A);
            return B * (eta - sin(eta));

        }

        if (OmegaMatter > 0.0)
        {
            if (a == 0.0) return 0.0;
            T a0 = 1.0 / H0 / sqrt(1.0 - OmegaMatter);
            T A = 0.5 * OmegaMatter / (1.0 - OmegaMatter);
            T B = a0 * A;
            T eta = acosh(1.0 + a/A);
            return B * (sinh(eta) - eta);

        }

        if (OmegaMatter == 0.0)
        {
            if (a == 0.0) return 0.0;
            return a / H0;
        }

        assert(0); // This situation should be caught in the constructor.
    }

    T a(T t)
    {
        auto f = [this, t](const T a) { return time(a) - t; };
        auto df = [this](const T a) { return 1.0/(a * H(a)); };
        return newton(f, df, t*H0, 0.0, 1.0e38, relativeError, absoluteError);
    }


    T driftTimeCorrection(T t, T dt)
    {
        if (OmegaMatter == 1.0)
        {
        }

        if (OmegaMatter > 1.0)
        {
        }

        if (OmegaMatter > 0.0)
        {
        }

        if (OmegaMatter == 0.0)
        {
        }

        assert(0); // This situation should be caught in the constructor.
    }

    T kickTimeCorrection(T t, T dt)
    {
        if (OmegaMatter == 1.0)
        {
        }

        if (OmegaMatter > 1.0)
        {
        }

        if (OmegaMatter > 0.0)
        {
        }

        if (OmegaMatter == 0.0)
        {
        }

        assert(0); // This situation should be caught in the constructor.
    }

};

template<typename T>
std::unique_ptr<Cosmology<T>> cosmologyFactory(const std::string& fname)
{
    throw std::runtime_error("filename argument to cosmology factory not yet supported.");
}

template<typename T>
std::unique_ptr<Cosmology<T>> cosmologyFactory(const struct LambdaCDM<T>::Parameters p)
{
    if (p.H0 == 0 && p.OmegaMatter == 0 && p.OmegaRadiation == 0 && p.OmegaLambda == 0)
        return std::make_unique<StaticUniverse<T>>(p);

    if (p.H0 <= 0)
        throw std::domain_error("Hubble0 parameter must be strictly positive when other cosmological parameters are non-zero.");

    if (p.OmegaLambda == 0 && p.OmegaRadiation == 0)
        return std::make_unique<CDM<T>>(p);

    return std::make_unique<LambdaCDM<T>>(p);
}


} // namespace cosmo
