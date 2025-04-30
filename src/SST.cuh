#pragma once

#include "Define.h"
#include "DParameter.cuh"
#include "TurbMethod.hpp"

namespace cfd {
struct DZone;

__global__ void implicit_treat_for_SST(DZone *zone, const DParameter *param);

template<TurbSimLevel level = TurbSimLevel::RANS>
struct SST {
  // Careful: There are many modifications for the SST model.
  // The most commonly used is the change of gamma1 and gamma2.
  // In that modified version, gamma1 = 5/9, gamma2 = 0.44.
  // As a comparison, the currently used values are gamma1 = 0.5531667, and gamma2 = 0.4403547, respectively.
  // How much influence this change of value has on the result is unknown.

  // Secondly, the modified version of SST uses strain rate instead of vorticity to compute the turbulent viscosity.
  // Third, the modified version uses 1e-10 instead of 1e-20 in CDKOmega computation.

// Model constants
  static constexpr double beta_star = 0.09;
  static constexpr double sqrt_beta_star = 0.3;
  static constexpr double kappa = 0.41;
// SST inner parameters, the first group:
  static constexpr double sigma_k1 = 0.85;
  static constexpr double sigma_omega1 = 0.5;
  static constexpr double beta_1 = 0.0750;
  static constexpr double a_1 = 0.31;
  static constexpr double gamma1 = beta_1 / beta_star - sigma_omega1 * kappa * kappa / sqrt_beta_star;
  static constexpr double C_des1 = 0.78;

// k-epsilon parameters, the second group:
  static constexpr double sigma_k2 = 1;
  static constexpr double sigma_omega2 = 0.856;
  static constexpr double beta_2 = 0.0828;
  static constexpr double gamma2 = beta_2 / beta_star - sigma_omega2 * kappa * kappa / sqrt_beta_star;
  static constexpr double C_des2 = 0.61;

// Mixed parameters, their difference, used in computations
  static constexpr double delta_sigma_k = sigma_k1 - sigma_k2;
  static constexpr double delta_sigma_omega = sigma_omega1 - sigma_omega2;
  static constexpr double delta_beta = beta_1 - beta_2;
  static constexpr double delta_gamma = gamma1 - gamma2;
  static constexpr double delta_C_des = C_des1 - C_des2;

  __device__ static void
  compute_mut(DZone *zone, int i, int j, int k, real mul, const DParameter *param);

  __device__ static void compute_source_and_mut(DZone *zone, int i, int j, int k, const DParameter *param);

  __device__ static void
  implicit_treat_for_dq0(DZone *zone, real diag, int i, int j, int k, const DParameter *param);

  __device__ static void
  implicit_treat_for_dqk(DZone *zone, real diag, int i, int j, int k, const real *dq_total,
                         const DParameter *param);

};

using sst = SST<TurbSimLevel::RANS>;

template<>
struct TurbMethod<SST<TurbSimLevel::RANS>> {
  static constexpr bool isLaminar = false;
  static constexpr bool hasMut = true;
  static constexpr auto type = TurbSimLevel::RANS;
  static constexpr auto label = TurbMethodLabel::SST;
  static constexpr bool needWallDistance = true;
  static constexpr bool hasImplicitTreat = true;
};

template<>
struct TurbMethod<SST<TurbSimLevel::DES>> {
  static constexpr bool isLaminar = false;
  static constexpr bool hasMut = true;
  static constexpr auto type = TurbSimLevel::DES;
  static constexpr auto label = TurbMethodLabel::SST;
  static constexpr bool needWallDistance = true;
  static constexpr bool hasImplicitTreat = true;
};

}