//
// Created by Noah Kubli on 05.03.2024.
//
#include "gtest/gtest.h"
#include <cmath>
#include <mpi.h>
#include "central_force.hpp"
#include "exchange_star_position.hpp"
#include "star_data.hpp"
#include "sph/particles_data.hpp"
#include "cstone/fields/field_get.hpp"

struct DiskTest : public ::testing::Test
{
    int rank = 0, numRanks = 0;

    DiskTest()
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized == 0) { MPI_Init(0, NULL); }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    }
};

TEST_F(DiskTest, testStarPosition)
{

    if (numRanks != 2) throw std::runtime_error("Must be excuted with two ranks");

    disk::StarData star;
    star.position    = {0., 1., 0.};
    star.position_m1 = {0., 0., 0.};
    star.force_local = {0., 1., 0., 0.};
    star.m           = 1.0;
    star.fixed_star  = 0;

    disk::computeAndExchangeStarPosition(star, 1., 1.);
    if (rank == 0)
    {
        printf("rank: %d star pos: %lf\t%lf\t%lf\n", rank, star.position[0], star.position[1], star.position[2]);
    }
    if (rank == 1)
    {
        printf("rank: %d star pos: %lf\t%lf\t%lf\n", rank, star.position[0], star.position[1], star.position[2]);
    }
    EXPECT_TRUE(star.position[0] == 2.);
    EXPECT_TRUE(star.position[1] == 1.);
    EXPECT_TRUE(star.position[2] == 0.);
}

TEST(CentralPotentials, testNewtonian)
{
    double x[] = {1., 0.};
    double y[] = {0., 1.};
    double z[] = {0., 0.};
    double m[] = {1., 1.};
    double ax[2]{};
    double ay[2]{};
    double az[2]{};

    disk::CentralForceData pass{x, y, z, m, ax, ay, az, 1.0, {0., 0., 0.}, 1.0, 0., 1.};
    cstone::Vec4<double>       star_force{};

    disk::newtonianGravity(pass, 0, star_force);
    disk::newtonianGravity(pass, 1, star_force);


    EXPECT_NEAR(ax[0], -1.0, 1e-8);
    EXPECT_NEAR(ay[0], 0.0, 1e-8);
    EXPECT_NEAR(az[0], 0.0, 1e-8);

    EXPECT_NEAR(ax[1], 0.0, 1e-8);
    EXPECT_NEAR(ay[1], -1.0, 1e-8);
    EXPECT_NEAR(az[1], 0.0, 1e-8);

    EXPECT_NEAR(std::sqrt(norm2(cstone::Vec3<double>{star_force[1], star_force[2], star_force[3]})), M_SQRT2, 1e-8);
    EXPECT_NEAR(star_force[0], -2.0, 1e-8);
}