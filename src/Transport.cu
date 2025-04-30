#include "Transport.cuh"
#include "DParameter.cuh"
#include "Field.h"
#include "Constants.h"

__host__ __device__
real cfd::Sutherland(real temperature) {
  return 1.7894e-5 * pow(temperature / 288.16, 1.5) * (288.16 + 110) / (temperature + 110);
}

real cfd::compute_viscosity(real temperature, real mw_total, real const *Y, const Species &spec) {
  // This method can only be used on CPU, while for GPU the allocation may be performed in every step
  std::vector<real> x(spec.n_spec);
  std::vector<real> vis_spec(spec.n_spec);
  gxl::MatrixDyn<real> partition_fun(spec.n_spec, spec.n_spec);
  for (int i = 0; i < spec.n_spec; ++i) {
    x[i] = Y[i] * mw_total / spec.mw[i];
    const real t_dl{temperature * spec.LJ_potent_inv[i]};  // dimensionless temperature
    const real collision_integral{1.147 * std::pow(t_dl, -0.145) + std::pow(t_dl + 0.5, -2)};
    vis_spec[i] = spec.vis_coeff[i] * std::sqrt(temperature) / collision_integral;
  }
  for (int i = 0; i < spec.n_spec; ++i) {
    for (int j = 0; j < spec.n_spec; ++j) {
      if (i == j) {
        partition_fun(i, j) = 1.0;
      } else {
        const real numerator{1 + std::sqrt(vis_spec[i] / vis_spec[j]) * spec.WjDivWi_to_One4th(i, j)};
        partition_fun(i, j) = numerator * numerator * spec.sqrt_WiDivWjPl1Mul8(i, j);
      }
    }
  }
  real viscosity{0};
  for (int i = 0; i < spec.n_spec; ++i) {
    real vis_temp{0};
    for (int j = 0; j < spec.n_spec; ++j) {
      vis_temp += partition_fun(i, j) * x[j];
    }
    viscosity += vis_spec[i] * x[i] / vis_temp;
  }
  return viscosity;
}

__device__ void
cfd::compute_transport_property(int i, int j, int k, real temperature, real mw_total, const real *cp, DParameter *param,
                                DZone *zone) {
  const auto n_spec{param->n_spec};
  const real *mw = param->mw;

  real X[MAX_SPEC_NUMBER], vis[MAX_SPEC_NUMBER], d_ij[MAX_SPEC_NUMBER * MAX_SPEC_NUMBER];//, ZRot[MAX_SPEC_NUMBER];
  for (int l = 0; l < n_spec; ++l) {
    X[l] = zone->sv(i, j, k, l) * mw_total / mw[l];
    const real t_dl{temperature * param->LJ_potent_inv[l]}; //dimensionless temperature
    const real collision_integral{1.147 * std::pow(t_dl, -0.145) + std::pow(t_dl + 0.5, -2)};
    vis[l] = param->vis_coeff[l] * std::sqrt(temperature) / collision_integral;
  }
  // The binary-diffusion coefficients including self-diffusion coefficients are computed.
  const real t_3over2_over_p{temperature * std::sqrt(temperature) / zone->bv(i, j, k, 4)};
  for (int m = 0; m < n_spec; ++m) {
    for (int n = m; n < n_spec; ++n) {
      const real t_red{temperature * param->kb_over_eps_jk(m, n)};
      const real omega_d{compute_Omega_D(t_red)};
      d_ij[m * n_spec + n] = param->binary_diffusivity_coeff(m, n) * t_3over2_over_p / omega_d;
      if (m != n) {
        d_ij[n * n_spec + m] = d_ij[m * n_spec + n];
      } /*else {
        // compute ZRot
        real tRedInv = 1 / t_red;
        real FT = 1 + 0.5 * pi * pi * pi * sqrt(tRedInv) + (2 + 0.25 * pi * pi) * tRedInv +
                  sqrt(pi * pi * pi * tRedInv * tRedInv * tRedInv);
        ZRot[m] = param->ZRotF298[m] / FT;
      }*/
    }
  }

  real viscosity = 0;
  real conductivity = 0;
  // const real density = zone->bv(i, j, k, 0);
  for (int m = 0; m < n_spec; ++m) {
    // compute the thermal conductivity
//    real R = R_u / mw[m];
//    real lambda = 15 / 4.0 * vis[m] * R;
//    if (param->geometry[m] == 1) {
//      // Linear geometry
//      const real rhoD = density * d_ij[m * n_spec + m];
//      const real ADivPiB = (2.5 * vis[m] - rhoD) / ((pi * ZRot[m] * vis[m] + 10.0 / 3 * vis[m] + 2.0 * rhoD));
//      lambda += -5 * ADivPiB * vis[m] * R + rhoD * cp[m] + rhoD * (2 * ADivPiB - 2.5) * R;
//    } else if (param->geometry[m] == 2) {
//      // Non-linear geometry
//      const real rhoD = density * d_ij[m * n_spec + m];
//      const real ADivPiB = (2.5 * vis[m] - rhoD) / ((pi * ZRot[m] * vis[m] + 5 * vis[m] + 2.0 * rhoD));
//      lambda += -7.5 * ADivPiB * vis[m] * R + rhoD * cp[m] + rhoD * (3 * ADivPiB - 2.5) * R;
//    }

    real vis_temp{0};
    for (int n = 0; n < n_spec; ++n) {
      real partition_func{1.0};
      if (m != n) {
        const real numerator{1 + std::sqrt(vis[m] / vis[n]) * param->WjDivWi_to_One4th(m, n)};
        partition_func = numerator * numerator * param->sqrt_WiDivWjPl1Mul8(m, n);
      }
      vis_temp += partition_func * X[n];
    }
    const real cond_temp = 1.065 * vis_temp - 0.065 * X[m];
    viscosity += vis[m] * X[m] / vis_temp;
    const real lambda = vis[m] * (cp[m] + 1.25 * R_u / mw[m]);
    conductivity += lambda * X[m] / cond_temp;
  }
  zone->mul(i, j, k) = viscosity;
  zone->thermal_conductivity(i, j, k) = conductivity;

  /*
  // The diffusivity is now computed via constant Schmidt number method
  const real sc{param->Sc};
  for (auto l = 0; l < n_spec; ++l) {
    if (std::abs(X[l] - 1) < 1e-3) {
      zone->rho_D(i, j, k, l) = viscosity / sc;
    } else {
      zone->rho_D(i, j, k, l) = (1 - yk(i, j, k, l)) * viscosity / ((1 - X[l]) * sc);
    }
  }
  */
  // The diffusivity is computed by mixture-averaged method.
  constexpr real eps{1e-12};
  for (int l = 0; l < n_spec; ++l) {
    real num{0};
    real den{0};

    for (int n = 0; n < n_spec; ++n) {
      if (l != n) {
        num += (X[n] + eps) * mw[n];
        den += (X[n] + eps) / d_ij[l * n_spec + n];
      }
    }
    zone->rho_D(i, j, k, l) = zone->bv(i, j, k, 0) * num / (den * mw_total);
  }
}

__device__ real
cfd::compute_viscosity(int i, int j, int k, real temperature, real mw_total, DParameter *param, const DZone *zone) {
  const auto n_spec{param->n_spec};
  const real *mw = param->mw;
  const auto &yk = zone->sv;

  real X[MAX_SPEC_NUMBER], vis[MAX_SPEC_NUMBER];
  for (int l = 0; l < n_spec; ++l) {
    X[l] = yk(i, j, k, l) * mw_total / mw[l];
    const real t_dl{temperature * param->LJ_potent_inv[l]}; //dimensionless temperature
    const real collision_integral{1.147 * std::pow(t_dl, -0.145) + std::pow(t_dl + 0.5, -2)};
    vis[l] = param->vis_coeff[l] * std::sqrt(temperature) / collision_integral;
  }

  real viscosity = 0;
  for (int m = 0; m < n_spec; ++m) {
    real vis_temp{0};
    for (int n = 0; n < n_spec; ++n) {
      real partition_func{1.0};
      if (m != n) {
        const real numerator{1 + std::sqrt(vis[m] / vis[n]) * param->WjDivWi_to_One4th(m, n)};
        partition_func = numerator * numerator * param->sqrt_WiDivWjPl1Mul8(m, n);
      }
      vis_temp += partition_func * X[n];
    }
    viscosity += vis[m] * X[m] / vis_temp;
  }
  return viscosity;
}

__device__ real cfd::compute_Omega_D(real t_red) {
  return 1.0 / std::pow(t_red, 0.145) + 1.0 / ((t_red + 0.5) * (t_red + 0.5));
}

__device__ real cfd::compute_viscosity(real temperature, real mw_total, const real *Y, DParameter *param) {
  const auto n_spec{param->n_spec};
  const real *mw = param->mw;

  real X[MAX_SPEC_NUMBER], vis[MAX_SPEC_NUMBER];
  for (int l = 0; l < n_spec; ++l) {
    X[l] = Y[l] * mw_total / mw[l];
    const real t_dl{temperature * param->LJ_potent_inv[l]}; //dimensionless temperature
    const real collision_integral{1.147 * std::pow(t_dl, -0.145) + std::pow(t_dl + 0.5, -2)};
    vis[l] = param->vis_coeff[l] * std::sqrt(temperature) / collision_integral;
  }

  real viscosity = 0;
  for (int m = 0; m < n_spec; ++m) {
    real vis_temp{0};
    for (int n = 0; n < n_spec; ++n) {
      real partition_func{1.0};
      if (m != n) {
        const real numerator{1 + std::sqrt(vis[m] / vis[n]) * param->WjDivWi_to_One4th(m, n)};
        partition_func = numerator * numerator * param->sqrt_WiDivWjPl1Mul8(m, n);
      }
      vis_temp += partition_func * X[n];
    }
    viscosity += vis[m] * X[m] / vis_temp;
  }
  return viscosity;
}
