#include <cstdio>
#include "Parameter.h"
#include "Mesh.h"
#include "Driver.cuh"
#include "Simulate.cuh"
#include "SST.cuh"

int main(int argc, char *argv[]) {
  cfd::Parameter parameter(&argc, &argv);

  cfd::Mesh mesh(parameter);

  const int species = parameter.get_int("species");
  const bool turbulent_laminar = parameter.get_bool("turbulence");
  const int reaction = parameter.get_int("reaction");
  int turbulent_method = parameter.get_int("turbulence_method");
  if (!turbulent_laminar) {
    parameter.update_parameter("turbulence_method", 0);
    turbulent_method = 0;
  }

  if (species == 1) {
    // Multiple species
    if (turbulent_method == 1) {
      // RANS
      if (reaction == 1) {
        // Finite rate chemistry
        cfd::Driver<MixtureModel::FR, cfd::SST<cfd::TurbSimLevel::RANS>> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      } else if (reaction == 2) {
        // Flamelet model
        cfd::Driver<MixtureModel::FL, cfd::SST<cfd::TurbSimLevel::RANS>> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      } else {
        // Pure mixing among species
        cfd::Driver<MixtureModel::Mixture, cfd::SST<cfd::TurbSimLevel::RANS>> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      }
    } else if (turbulent_method == 2) {
      // DES and air
      if (reaction == 1) {
        // Finite rate chemistry
        cfd::Driver<MixtureModel::FR, cfd::SST<cfd::TurbSimLevel::DES>> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      } else if (reaction == 2) {
        // Pure mixing among species
        cfd::Driver<MixtureModel::FL, cfd::SST<cfd::TurbSimLevel::DES>> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      } else {
        // Pure mixing among species
        cfd::Driver<MixtureModel::Mixture, cfd::SST<cfd::TurbSimLevel::DES>> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      }
    } else {
      // Laminar
      if (reaction == 1) {
        // Finite rate chemistry
        cfd::Driver<MixtureModel::FR, cfd::Laminar> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      } else {
        // Pure mixing among species
        cfd::Driver<MixtureModel::Mixture, cfd::Laminar> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      }
    }
  } else if (species == 2) {
    // Mixture fraction and mixture fraction variance are solved together with species mixing.
    if (turbulent_method == 1) {
      // RANS
      if (reaction == 0) {
        // Compute the species mixing, with mixture fraction and mixture fraction variance also solved.
        cfd::Driver<MixtureModel::MixtureFraction, cfd::SST<cfd::TurbSimLevel::RANS>> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      } else if (reaction == 2) {
        // Flamelet model
        cfd::Driver<MixtureModel::FL, cfd::SST<cfd::TurbSimLevel::RANS>> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      } else {
        printf("The combination of species model 2 and reaction model %d is not implemented", reaction);
      }
    } else if (turbulent_method == 2) {
      // DDES
      if (reaction == 0) {
        // Compute the species mixing, with mixture fraction and mixture fraction variance also solved.
        cfd::Driver<MixtureModel::MixtureFraction, cfd::SST<cfd::TurbSimLevel::DES>> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      } else if (reaction == 2) {
        // Flamelet model
        cfd::Driver<MixtureModel::FL, cfd::SST<cfd::TurbSimLevel::DES>> driver(parameter, mesh);
        driver.initialize_computation();
        simulate(driver);
      } else {
        printf("The combination of species model 2 and reaction model %d is not implemented", reaction);
      }
    } else {
      // Laminar
      cfd::Driver<MixtureModel::MixtureFraction, cfd::Laminar> driver(parameter, mesh);
      driver.initialize_computation();
      simulate(driver);
    }
  } else {
    // Air simulation
    if (turbulent_method == 1) {
      // RANS and air
      cfd::Driver<MixtureModel::Air, cfd::SST<cfd::TurbSimLevel::RANS>> driver(parameter, mesh);
      driver.initialize_computation();
      simulate(driver);
    } else if (turbulent_method == 2) {
      // DES and air
      cfd::Driver<MixtureModel::Air, cfd::SST<cfd::TurbSimLevel::DES>> driver(parameter, mesh);
      driver.initialize_computation();
      simulate(driver);
    } else {
      // Laminar and air
      cfd::Driver<MixtureModel::Air, cfd::Laminar> driver(parameter, mesh);
      driver.initialize_computation();
      simulate(driver);
    }
  }

  if (parameter.get_int("myid") == 0) {
    printf("Yeah, baby, we are ok now\n");
    std::ofstream out("Man, we are Finished.txt");
    out<<"Yeah, baby, we are ok now\n";
    out.close();
  }
  MPI_Finalize();
  return 0;
}
