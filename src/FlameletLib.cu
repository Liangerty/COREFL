#include "FlameletLib.cuh"
#include "gxl_lib/MyString.h"
#include "Field.h"
#include <fstream>

namespace cfd {
FlameletLib::FlameletLib(const Parameter &parameter) : n_spec{parameter.get_int("n_spec")} {
  if (parameter.get_int("n_spec") == 0 || parameter.get_int("reaction") != 2) {
    return;
  }
  switch (parameter.get_int("flamelet_format")) {
    case 0:
      // ACANS format
      read_ACANS_flamelet(parameter);
      break;
    case 1:
    default:
      // FlameMaster format, not implemented yet.
      break;
  }
}

void FlameletLib::read_ACANS_flamelet(const Parameter &parameter) {
  const auto flamelet_file_name{"input/" + parameter.get_string("flamelet_file_name")};
  printf("\tReading flamelet file: %s\n", flamelet_file_name.c_str());
  std::ifstream file{flamelet_file_name};
  std::string input;
  std::istringstream line;
  gxl::getline_to_stream(file, input, line);
  line >> n_z >> n_zPrime >> n_chi;
  printf("\tn_z=%d, n_zPrime=%d, n_chi=%d\n", n_z, n_zPrime, n_chi);

  z.resize(n_z + 1);
  zPrime.resize(n_zPrime + 1, n_z + 1);
  chi_ave.resize(n_chi, n_zPrime + 1, n_z + 1, 0);
  yk.resize(n_spec, n_chi, n_zPrime + 1, n_z + 1, 0);
  for (int i = 0; i <= n_z; ++i) {
    for (int j = 0; j <= n_zPrime; ++j) {
      for (int k = 0; k < n_chi; ++k) {
        gxl::getline_to_stream(file, input, line);
        line >> z[i] >> zPrime(j, i) >> chi_ave(k, j, i);
        gxl::getline_to_stream(file, input, line);
        for (int l = 0; l < n_spec; ++l)
          line >> yk(l, k, j, i);
      }
    }
  }
  file.close();

  file.open("input/chemistry/erf.txt");
  gxl::getline_to_stream(file, input, line);
  line >> fzst;
  fz.resize(n_z + 1);
  gxl::getline_to_stream(file, input, line);
  for (int i = 0; i <= n_z; ++i) {
    line >> fz[i];
  }
  file.close();

  dz.resize(n_z);
  for (int i = 0; i < n_z; ++i) {
    dz[i] = z[i + 1] - z[i];
  }

  chi_min.resize(n_zPrime + 1, n_z + 1);
  chi_max.resize(n_zPrime + 1, n_z + 1);
  chi_min_j.resize(n_zPrime + 1, n_z + 1);
  chi_max_j.resize(n_zPrime + 1, n_z + 1);
  for (int i = 0; i <= n_z; ++i) {
    for (int j = 0; j <= n_zPrime; ++j) {
      chi_min(j, i) = chi_ave(0, j, i);
      chi_max(j, i) = chi_ave(0, j, i);
      for (int k = 0; k < n_chi; ++k) {
        if (chi_ave(k, j, i) <= chi_min(j, i)) {
          chi_min(j, i) = chi_ave(k, j, i);
          chi_min_j(j, i) = k;
        }
        if (chi_ave(k, j, i) >= chi_max(j, i)) {
          chi_max(j, i) = chi_ave(k, j, i);
          chi_max_j(j, i) = k;
        }
      }
    }
  }

//  ffz.resize(n_z + 1);
//  for (int i = 0; i <= n_z; ++i) {
//    ffz[i] = fz[i];
//  }

//  zi.resize(n_z + 7);
//  zi[0] = z[0];
//  zi[1] = 1e-6;
//  zi[2] = 1e-5;
//  zi[3] = 1e-4;
//  for (int i = 1; i < n_z; ++i) {
//    zi[i + 3] = z[i];
//  }
//  zi[n_z + 3] = 1 - 1e-4;
//  zi[n_z + 4] = 1 - 1e-5;
//  zi[n_z + 5] = 1 - 1e-6;
//  zi[n_z + 6] = 1;
}

__device__ void flamelet_source(DZone *zone, int i, int j, int k, const DParameter *param) {
  const auto &m = zone->metric(i, j, k);
  const real xi_x{m(1, 1)}, xi_y{m(1, 2)}, xi_z{m(1, 3)};
  const real eta_x{m(2, 1)}, eta_y{m(2, 2)}, eta_z{m(2, 3)};
  const real zeta_x{m(3, 1)}, zeta_y{m(3, 2)}, zeta_z{m(3, 3)};

  // compute the gradient of mixture fraction
  const auto &sv{zone->sv};
  const int i_fl{param->i_fl};

  const real mixFrac_x = 0.5 * (xi_x * (sv(i + 1, j, k, i_fl) - sv(i - 1, j, k, i_fl)) +
                                eta_x * (sv(i, j + 1, k, i_fl) - sv(i, j - 1, k, i_fl)) +
                                zeta_x * (sv(i, j, k + 1, i_fl) - sv(i, j, k - 1, i_fl)));
  const real mixFrac_y = 0.5 * (xi_y * (sv(i + 1, j, k, i_fl) - sv(i - 1, j, k, i_fl)) +
                                eta_y * (sv(i, j + 1, k, i_fl) - sv(i, j - 1, k, i_fl)) +
                                zeta_y * (sv(i, j, k + 1, i_fl) - sv(i, j, k - 1, i_fl)));
  const real mixFrac_z = 0.5 * (xi_z * (sv(i + 1, j, k, i_fl) - sv(i - 1, j, k, i_fl)) +
                                eta_z * (sv(i, j + 1, k, i_fl) - sv(i, j - 1, k, i_fl)) +
                                zeta_z * (sv(i, j, k + 1, i_fl) - sv(i, j, k - 1, i_fl)));

  const real prod_mixFrac = 2.0 * zone->mut(i, j, k) / param->Sct
                            * (mixFrac_x * mixFrac_x + mixFrac_y * mixFrac_y + mixFrac_z * mixFrac_z);
  zone->scalar_diss_rate(i, j, k) = 2 * 0.09 * sv(i, j, k, param->n_spec + 1) * sv(i, j, k, i_fl + 1) * param->c_chi;
  const real diss_mixFrac{zone->bv(i, j, k, 0) * zone->scalar_diss_rate(i, j, k)};

  zone->dq(i, j, k, param->i_fl_cv + 1) += zone->jac(i, j, k) * (prod_mixFrac - diss_mixFrac);
}

/// @brief Compute the mass fraction from the mixture fraction and scalar dissipation rate
/// @param zone Pointer to the computational zone (DZone) containing field variables
/// @param i Index in the x-direction
/// @param j Index in the y-direction
/// @param k Index in the z-direction
/// @param param Pointer to simulation parameters
/// @param yk_ave Array to store the computed average mass fractions
///
/// This device function calculates the mass fractions of chemical species based on the local
/// mixture fraction and scalar dissipation rate. It uses a flamelet library approach, which
/// involves interpolation in a pre-computed table of species mass fractions.
///
/// The function handles three cases:
/// 1. Pure oxidizer stream (mixture fraction < 1e-6)
/// 2. Pure fuel stream (mixture fraction > 1 - 1e-6)
/// 3. Intermediate mixtures (requires interpolation)
///
/// For intermediate mixtures, it performs a triple linear interpolation based on:
/// - Mixture fraction
/// - Mixture fraction variance
/// - Scalar dissipation rate
///
/// The interpolated mass fractions are stored in the yk_ave array.
__device__ void
compute_massFraction_from_MixtureFraction(DZone *zone, int i, int j, int k, DParameter *param, real *yk_ave) {
  const auto mixFrac_ave{zone->sv(i, j, k, param->i_fl)};

  const auto n_spec{param->n_spec};
  const auto &yk_lib{param->yk_lib};
  const auto mz_lib{param->n_z};

  // First, if the mixture fraction is 1/0, it is in pure fuel stream or pure oxidizer stream
  if (mixFrac_ave < 1e-6) {
    // Pure oxidizer stream
    for (int l = 0; l < n_spec; ++l) {
      yk_ave[l] = yk_lib(l, 0, 0, 0);
    }
    return;
  }
  if (mixFrac_ave > 1 - 1e-6) {
    // Pure fuel stream
    for (int l = 0; l < n_spec; ++l) {
      yk_ave[l] = yk_lib(l, 0, 0, mz_lib);
    }
    return;
  }

  // For most situations, we need the triple linear interpolation
  auto mixFracVariance{zone->sv(i, j, k, param->i_fl + 1)};
  if (mixFracVariance < 1e-12)
    mixFracVariance = 1e-12;
  else if (mixFracVariance > mixFrac_ave * (1 - mixFrac_ave))
    mixFracVariance = mixFrac_ave * (1 - mixFrac_ave);

  // Find the range of mixture fraction
  int z1{0};
  const auto z_lib{param->mix_frac};
  for (int l = 0; l < mz_lib; ++l) {
    if (mixFrac_ave >= z_lib[l] && mixFrac_ave <= z_lib[l + 1]) {
      z1 = l;
      break;
    }
  }
  const int z2{z1 + 1};

  // Next, apply a binary interpolation at z1 and z2, respectively.
  const auto chi_ave{zone->scalar_diss_rate(i, j, k)};
  // First, at z1
  real yk_z1[MAX_SPEC_NUMBER] = {};
  const int n_zPrime{param->n_zPrime};
  if (z1 == 0) {
    for (int l = 0; l < n_spec; ++l) {
      yk_z1[l] = yk_lib(l, 0, 0, 0);
    }
  } else {
    // Interpolate into the mixture fraction variance

    // Find the range for z_prime
    int z_prime_1{-1};
    for (int l = 0; l < n_zPrime; ++l) {
      if (mixFracVariance >= param->zPrime(l, z1) && mixFracVariance <= param->zPrime(l + 1, z1)) {
        z_prime_1 = l;
        break;
      }
    }
    int z_prime_2{z_prime_1 + 1};
    if (z_prime_1 == -1) {
      z_prime_1 = n_zPrime - 1;
      z_prime_2 = n_zPrime;
    }
    // Next, interpolate into the scalar dissipation rate
    real yk_z11[MAX_SPEC_NUMBER] = {}, yk_z12[MAX_SPEC_NUMBER] = {};
    interpolate_scalar_dissipation_rate_with_given_z_zPrime(chi_ave, n_spec, z1, z_prime_1, param, yk_z11);
    interpolate_scalar_dissipation_rate_with_given_z_zPrime(chi_ave, n_spec, z1, z_prime_2, param, yk_z12);
    // Finally, interpolate into the mixture fraction variance
    for (int l = 0; l < n_spec; ++l) {
      yk_z1[l] = yk_z11[l] + (yk_z12[l] - yk_z11[l]) / (param->zPrime(z_prime_2, z1) - param->zPrime(z_prime_1, z1)) *
                             (mixFracVariance - param->zPrime(z_prime_1, z1));
    }
    //if (z_prime_1 == -1) {
    //  //z_prime_1 = n_zPrime - 1;
    //  //z_prime_2 = n_zPrime;
    //  interpolate_scalar_dissipation_rate_with_given_z_zPrime(chi_ave, n_spec, z1, n_zPrime, param, yk_z1);
    //}
    //else {
    //  // Next, interpolate into the scalar dissipation rate
    //  real yk_z11[MAX_SPEC_NUMBER], yk_z12[MAX_SPEC_NUMBER];
    //  memset(yk_z11, 0, sizeof(real) * MAX_SPEC_NUMBER);
    //  memset(yk_z12, 0, sizeof(real) * MAX_SPEC_NUMBER);
    //  interpolate_scalar_dissipation_rate_with_given_z_zPrime(chi_ave, n_spec, z1, z_prime_1, param, yk_z11);
    //  interpolate_scalar_dissipation_rate_with_given_z_zPrime(chi_ave, n_spec, z1, z_prime_2, param, yk_z12);
    //  // Finally, interpolate into the mixture fraction variance
    //  for (int l = 0; l < n_spec; ++l) {
    //    yk_z1[l] = yk_z11[l] + (yk_z12[l] - yk_z11[l]) / (param->zPrime(z_prime_2, z1) - param->zPrime(z_prime_1, z1)) *
    //      (mixFracVariance - param->zPrime(z_prime_1, z1));
    //  }
    //}
  }

  // Next, at z2
  real yk_z2[MAX_SPEC_NUMBER] = {};
  if (z2 == mz_lib) {
    for (int l = 0; l < n_spec; ++l) {
      yk_z2[l] = yk_lib(l, 0, 0, mz_lib);
    }
  } else {
    // Interpolate into the mixture fraction variance

    // Find the range for z_prime
    int z_prime_1{-1};
    for (int l = 0; l < n_zPrime; ++l) {
      if (mixFracVariance >= param->zPrime(l, z2) && mixFracVariance < param->zPrime(l + 1, z2)) {
        z_prime_1 = l;
        break;
      }
    }
    int z_prime_2{z_prime_1 + 1};
    if (z_prime_1 == -1) {
      z_prime_1 = n_zPrime - 1;
      z_prime_2 = n_zPrime;
    }
    // Next, interpolate into the scalar dissipation rate
    real yk_z21[MAX_SPEC_NUMBER] = {}, yk_z22[MAX_SPEC_NUMBER] = {};
    interpolate_scalar_dissipation_rate_with_given_z_zPrime(chi_ave, n_spec, z2, z_prime_1, param, yk_z21);
    interpolate_scalar_dissipation_rate_with_given_z_zPrime(chi_ave, n_spec, z2, z_prime_2, param, yk_z22);
    // Finally, interpolate into the mixture fraction variance
    for (int l = 0; l < n_spec; ++l) {
      yk_z2[l] = yk_z21[l] + (yk_z22[l] - yk_z21[l]) / (param->zPrime(z_prime_2, z2) - param->zPrime(z_prime_1, z2)) *
                             (mixFracVariance - param->zPrime(z_prime_1, z2));
    }
    //if (z_prime_1 == -1) {
    //  //z_prime_1 = n_zPrime - 1;
    //  //z_prime_2 = n_zPrime;
    //  interpolate_scalar_dissipation_rate_with_given_z_zPrime(chi_ave, n_spec, z2, n_zPrime, param, yk_z2);
    //}
    //else {
    //  // Next, interpolate into the scalar dissipation rate
    //  real yk_z21[MAX_SPEC_NUMBER], yk_z22[MAX_SPEC_NUMBER];
    //  memset(yk_z21, 0, sizeof(real) * MAX_SPEC_NUMBER);
    //  memset(yk_z22, 0, sizeof(real) * MAX_SPEC_NUMBER);
    //  interpolate_scalar_dissipation_rate_with_given_z_zPrime(chi_ave, n_spec, z2, z_prime_1, param, yk_z21);
    //  interpolate_scalar_dissipation_rate_with_given_z_zPrime(chi_ave, n_spec, z2, z_prime_2, param, yk_z22);
    //  // Finally, interpolate into the mixture fraction variance
    //  for (int l = 0; l < n_spec; ++l) {
    //    yk_z2[l] = yk_z21[l] + (yk_z22[l] - yk_z21[l]) / (param->zPrime(z_prime_2, z2) - param->zPrime(z_prime_1, z2)) *
    //      (mixFracVariance - param->zPrime(z_prime_1, z2));
    //  }
    //}
  }

  // Finally, acquire the value from a linear interpolation between z1 and z2
  for (int l = 0; l < n_spec; ++l) {
    yk_ave[l] = yk_z1[l] + (yk_z2[l] - yk_z1[l]) / (z_lib[z2] - z_lib[z1]) * (mixFrac_ave - z_lib[z1]);
  }
}

__device__ int2
find_chi_range(const ggxl::Array3D<real> &chi_ave, real chi, int i_z, int i_zPrime, int n_chi) {
  int2 range{};

  real d_chi_l = 1e+6;
  real d_chi_r = 1e+6;
  for (int i = 0; i < n_chi; ++i) {
    const auto d_chi{chi - chi_ave(i, i_zPrime, i_z)};
    if (d_chi >= 0 && abs(d_chi) < d_chi_l) {
      range.x = i;
      d_chi_l = abs(d_chi);
    }
    if (d_chi <= 0 && abs(d_chi) < d_chi_r) {
      range.y = i;
      d_chi_r = abs(d_chi);
    }
  }

  return range;
}

__device__ void
interpolate_scalar_dissipation_rate_with_given_z_zPrime(real chi_ave, int n_spec, int i_z, int i_zPrime,
                                                        DParameter *param, real *yk) {
  const auto &yk_lib{param->yk_lib};
  const auto &chi_ave_lib{param->chi_ave};
  if (chi_ave <= param->chi_min(i_zPrime, i_z)) {
    // Use the minimum scalar dissipation rate
    for (int l = 0; l < n_spec; ++l) {
      yk[l] = yk_lib(l, param->chi_min_j(i_zPrime, i_z), i_zPrime, i_z);
    }
  } else if (chi_ave >= param->chi_max(i_zPrime, i_z)) {
    // Use the maximum scalar dissipation rate
    for (int l = 0; l < n_spec; ++l) {
      yk[l] = yk_lib(l, param->chi_max_j(i_zPrime, i_z), i_zPrime, i_z);
    }
  } else {
    const auto [left, right] = find_chi_range(chi_ave_lib, chi_ave, i_z, i_zPrime, param->n_chi);
    for (int l = 0; l < n_spec; ++l) {
      yk[l] = yk_lib(l, left, i_zPrime, i_z) +
              (yk_lib(l, right, i_zPrime, i_z) - yk_lib(l, left, i_zPrime, i_z)) /
              (chi_ave_lib(right, i_zPrime, i_z) - chi_ave_lib(left, i_zPrime, i_z)) *
              (chi_ave - chi_ave_lib(left, i_zPrime, i_z));
    }
  }

}

__global__ void update_n_fl_step(DParameter *param) {
  ++param->n_fl_step;
}
}// cfd