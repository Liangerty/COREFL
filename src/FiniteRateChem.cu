#include "FiniteRateChem.cuh"
#include "Field.h"
#include "Thermo.cuh"
#include "Constants.h"

namespace cfd {
__device__ void finite_rate_chemistry(DZone *zone, int i, int j, int k, const DParameter *param) {
  const auto &bv = zone->bv;
  const auto &sv = zone->sv;

  const int ns = param->n_spec;

  // compute the concentration of species in mol/cm3
  real c[MAX_SPEC_NUMBER];
  const real density{bv(i, j, k, 0)};
  const auto mw = param->mw;
  for (int l = 0; l < ns; ++l) {
    c[l] = density * sv(i, j, k, l) / mw[l] * 1e-3;
  }

  // compute the forward reaction rate
  const real t{bv(i, j, k, 5)};
  real kf[MAX_REAC_NUMBER];
  forward_reaction_rate(t, kf, c, param);

  // compute the backward reaction rate
  real kb[MAX_REAC_NUMBER] = {};
  backward_reaction_rate(t, kf, c, param, kb);

  // compute the rate of progress
  real q[MAX_REAC_NUMBER * 3];
  real *q1 = &q[MAX_REAC_NUMBER];
  real *q2 = &q[MAX_REAC_NUMBER * 2];
  rate_of_progress(kf, kb, c, q, q1, q2, param);

  // compute the chemical source
  real omega[MAX_REAC_NUMBER * 2];
  real *omega_d = &omega[MAX_REAC_NUMBER];
  chemical_source(q1, q2, omega_d, omega, param);

  for (int l = 0; l < ns; ++l) {
    zone->dq(i, j, k, l + 5) += zone->jac(i, j, k) * omega[l];
  }

  // If implicit treat
  switch (param->chemSrcMethod) {
    case 0: // Explicit treatment
      break;
    case 1: // Exact point implicit
      compute_chem_src_jacobian(zone, i, j, k, param, q1, q2);
      break;
    case 2: // Diagonal approximation
      compute_chem_src_jacobian_diagonal(zone, i, j, k, param, omega_d);
      break;
    default: // Default, explicit treatment
      break;
  }
}

__device__ void forward_reaction_rate(real t, real *kf, const real *concentration, const DParameter *param) {
  const auto A = param->A, b = param->b, Ea = param->Ea;
  const auto type = param->reac_type;
  const auto A2 = param->A2, b2 = param->b2, Ea2 = param->Ea2;
  const auto &third_body_coeff = param->third_body_coeff;
  const auto alpha = param->troe_alpha, t3 = param->troe_t3, t1 = param->troe_t1, t2 = param->troe_t2;
  for (int i = 0; i < param->n_reac; ++i) {
    kf[i] = arrhenius(t, A[i], b[i], Ea[i]);
    if (type[i] == 3) {
      // Duplicate reaction
      kf[i] += arrhenius(t, A2[i], b2[i], Ea2[i]);
    } else if (type[i] > 3) {
      real cc{0};
      for (int l = 0; l < param->n_spec; ++l) {
        cc += concentration[l] * third_body_coeff(i, l);
      }
      if (type[i] == 4) {
        // Third body reaction
        kf[i] *= cc;
      } else {
        const real kf_low = arrhenius(t, A2[i], b2[i], Ea2[i]);
        const real kf_high = kf[i];
        const real reduced_pressure = kf_low * cc / kf_high;
        real F = 1.0;
        if (type[i] > 5) {
          // Troe form
          real f_cent = (1 - alpha[i]) * std::exp(-t / t3[i]) + alpha[i] * std::exp(-t / t1[i]);
          if (type[i] == 7) {
            f_cent += std::exp(-t2[i] / t);
          }
          const real logFc = std::log10(f_cent);
          const real c = -0.4 - 0.67 * logFc;
          const real n = 0.75 - 1.27 * logFc;
          const real logPr = std::log10(reduced_pressure);
          const real tempo = (logPr + c) / (n - 0.14 * (logPr + c));
          const real p = logFc / (1.0 + tempo * tempo);
          F = std::pow(10, p);
        }
        kf[i] = kf_high * reduced_pressure / (1.0 + reduced_pressure) * F;
      }
    }
  }
}

__device__ real arrhenius(real t, real A, real b, real Ea) {
  return A * std::pow(t, b) * std::exp(-Ea / t);
}

__device__ void
backward_reaction_rate(real t, const real *kf, const real *concentration, const DParameter *param, real *kb) {
  int n_gibbs{param->n_reac};
  for (int i = 0; i < param->n_reac; ++i) {
    if (param->reac_type[i] == 0) {
      // Irreversible reaction
      kb[i] = 0;
      --n_gibbs;
    } else if (param->rev_type[i] == 1) {
      // REV reaction
      kb[i] = arrhenius(t, param->A2[i], param->b2[i], param->Ea2[i]);
      if (param->reac_type[i] == 4) {
        // Third body required
        real cc{0};
        for (int l = 0; l < param->n_spec; ++l) {
          cc += concentration[l] * param->third_body_coeff(i, l);
        }
        kb[i] *= cc;
      }
      --n_gibbs;
    }
  }
  if (n_gibbs < 1)
    return;

  real gibbs_rt[MAX_SPEC_NUMBER];
  compute_gibbs_div_rt(t, param, gibbs_rt);
  constexpr real temp_p = p_atm / R_u * 1e-3;   // Convert the unit to mol*K/cm3
  const real temp_t = temp_p / t; // Unit is mol/cm3
  const auto &stoi_f = param->stoi_f, &stoi_b = param->stoi_b;
  const auto order = param->reac_order;
  for (int i = 0; i < param->n_reac; ++i) {
    if (param->reac_type[i] != 2 && param->reac_type[i] != 0) {
      real d_gibbs{0};
      for (int l = 0; l < param->n_spec; ++l) {
        d_gibbs += gibbs_rt[l] * (stoi_b(i, l) - stoi_f(i, l));
      }
      const real kc{std::pow(temp_t, order[i]) * std::exp(-d_gibbs)};
      kb[i] = kf[i] / kc;
    }
  }

}

__device__ void
rate_of_progress(const real *kf, const real *kb, const real *c, real *q, real *q1, real *q2, const DParameter *param) {
  const int n_spec{param->n_spec};
  const auto &stoi_f{param->stoi_f}, &stoi_b{param->stoi_b};
  for (int i = 0; i < param->n_reac; ++i) {
    if (param->reac_type[i] != 0) {
      q1[i] = 1.0;
      q2[i] = 1.0;
      for (int j = 0; j < n_spec; ++j) {
        q1[i] *= std::pow(c[j], stoi_f(i, j));
        q2[i] *= std::pow(c[j], stoi_b(i, j));
      }
      q1[i] *= kf[i];
      q2[i] *= kb[i];
      q[i] = q1[i] - q2[i];
    } else {
      q1[i] = 1.0;
      q2[i] = 0.0;
      for (int j = 0; j < n_spec; ++j) {
        q1[i] *= std::pow(c[j], stoi_f(i, j));
      }
      q1[i] *= kf[i];
      q[i] = q1[i];
    }
  }
}

__device__ void chemical_source(const real *q1, const real *q2, real *omega_d, real *omega, const DParameter *param) {
  const int n_spec{param->n_spec};
  const int n_reac{param->n_reac};
  const auto &stoi_f = param->stoi_f, &stoi_b{param->stoi_b};
  const auto mw = param->mw;
  for (int i = 0; i < n_spec; ++i) {
    real creation = 0;
    omega_d[i] = 0;
    for (int j = 0; j < n_reac; ++j) {
      creation += q2[j] * stoi_f(j, i) + q1[j] * stoi_b(j, i);
      omega_d[i] += q1[j] * stoi_f(j, i) + q2[j] * stoi_b(j, i);
    }
    creation *= mw[i] * 1e+3;              // Unit is kg/(m3*s)
    omega_d[i] *= mw[i] * 1e+3;        // Unit is kg/(m3*s)
    omega[i] = creation - omega_d[i]; // Unit is kg/(m3*s)
  }
}

__device__ void
compute_chem_src_jacobian(DZone *zone, int i, int j, int k, const DParameter *param, const real *q1, const real *q2) {
  const int n_spec{param->n_spec}, n_reac{param->n_reac};
  auto &sv = zone->sv;
  const auto &stoi_f = param->stoi_f, &stoi_b = param->stoi_b;
  const real density{zone->bv(i, j, k, 0)};
  auto &chem_jacobian = zone->chem_src_jac;
  for (int m = 0; m < n_spec; ++m) {
    for (int n = 0; n < n_spec; ++n) {
      real zz{0};
      if (sv(i, j, k, n) > 1e-25) {
        for (int r = 0; r < n_reac; ++r) {
          // The q1 and q2 here are in cgs unit, that is, mol/(cm3*s)
          zz += (stoi_b(r, m) - stoi_f(r, m)) * (stoi_f(r, n) * q1[r] - stoi_b(r, n) * q2[r]);
        }
        zz /= density * sv(i, j, k, n);
      }
      chem_jacobian(i, j, k, m * n_spec + n) = param->mw[m] * zz * 1e+3; // //1e+3=1e-3(MW)*1e+6(cm->m)
    }
  }
}

__device__ void
compute_chem_src_jacobian_diagonal(DZone *zone, int i, int j, int k, const DParameter *param, const real *omega_d) {
  // The method described in 2015-Savard-JCP
  auto &chem_jacobian = zone->chem_src_jac;
  auto &sv = zone->sv;
  const real density{zone->bv(i, j, k, 0)};
  for (int l = 0; l < param->n_spec; ++l) {
    chem_jacobian(i, j, k, l) = 0;
    if (sv(i, j, k, l) > 1e-25) {
      chem_jacobian(i, j, k, l) = -omega_d[l] / (sv(i, j, k, l) * density);
    }
  }
}

__global__ void EPI(DZone *zone, int n_spec) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  auto &chem_jac = zone->chem_src_jac;
  real lhs[MAX_SPEC_NUMBER * MAX_SPEC_NUMBER] = {};
  const real dt{zone->dt_local(i, j, k)};

  for (int m = 0; m < n_spec; ++m) {
    for (int n = 0; n < n_spec; ++n) {
      if (m == n) {
        lhs[m * n_spec + n] = 1.0 - dt * chem_jac(i, j, k, m * n_spec + n);
      } else {
        lhs[m * n_spec + n] = -dt * chem_jac(i, j, k, m * n_spec + n);
      }
    }
  }
  solve_chem_system(lhs, zone, i, j, k, n_spec);
}

__device__ void EPI_for_dq0(DZone *zone, real diag, int i, int j, int k, int n_spec) {
  real lhs[MAX_SPEC_NUMBER * MAX_SPEC_NUMBER] = {};
  const real dt{zone->dt_local(i, j, k)};
  auto &chem_jac = zone->chem_src_jac;

  for (int m = 0; m < n_spec; ++m) {
    for (int n = 0; n < n_spec; ++n) {
      if (m == n) {
        lhs[m * n_spec + n] = diag - dt * chem_jac(i, j, k, m * n_spec + n);
      } else {
        lhs[m * n_spec + n] = -dt * chem_jac(i, j, k, m * n_spec + n);
      }
    }
  }
  solve_chem_system(lhs, zone, i, j, k, n_spec);
}

__device__ void
EPI_for_dqk(DZone *zone, real diag, int i, int j, int k, const real *dq_total, int n_spec) {
  real lhs[MAX_SPEC_NUMBER * MAX_SPEC_NUMBER] = {};
  const real dt{zone->dt_local(i, j, k)};

  auto &chem_jac = zone->chem_src_jac;
  for (int m = 0; m < n_spec; ++m) {
    for (int n = 0; n < n_spec; ++n) {
      if (m == n) {
        lhs[m * n_spec + n] = diag - dt * chem_jac(i, j, k, m * n_spec + n);
      } else {
        lhs[m * n_spec + n] = -dt * chem_jac(i, j, k, m * n_spec + n);
      }
    }
  }

  real rhs[MAX_SPEC_NUMBER] = {};
  for (int l = 0; l < n_spec; ++l) {
    rhs[l] = dq_total[5 + l] * dt;
  }
  solve_chem_system(lhs, rhs, n_spec);

  auto &dqk = zone->dqk;
  const auto &dq0 = zone->dq0;
  for (int l = 0; l < n_spec; ++l) {
    dqk(i, j, k, l + 5) = dq0(i, j, k, l + 5) + rhs[l];
  }
}

__device__ void solve_chem_system(real *lhs, DZone *zone, int i, int j, int k, int n_spec) {
  const int dim{n_spec};
  int ipiv[MAX_SPEC_NUMBER] = {};

  // Column pivot LU decomposition
  for (int n = 0; n < dim; ++n) {
    int ik{n};
    for (int m = n; m < dim; ++m) {
      for (int t = 0; t < n; ++t) {
        lhs[m * dim + n] -= lhs[m * dim + t] * lhs[t * dim + n];
      }
      if (std::abs(lhs[m * dim + n]) > std::abs(lhs[ik * dim + n])) {
        ik = m;
      }
    }
    ipiv[n] = ik;
    if (ik != n) {
      for (int t = 0; t < dim; ++t) {
        const auto mid = lhs[ik * dim + t];
        lhs[ik * dim + t] = lhs[n * dim + t];
        lhs[n * dim + t] = mid;
      }
    }
    for (int p = n + 1; p < dim; ++p) {
      for (int t = 0; t < n; ++t) {
        lhs[n * dim + p] -= lhs[n * dim + t] * lhs[t * dim + p];
      }
    }
    for (int m = n + 1; m < dim; ++m) {
      lhs[m * dim + n] /= lhs[n * dim + n];
    }
  }

  auto &b = zone->dq;
  // Solve the linear system with LU matrix
  for (int m = 0; m < dim; ++m) {
    const int t = ipiv[m];
    if (t != m) {
      const auto mid = b(i, j, k, 5 + t);
      b(i, j, k, 5 + t) = b(i, j, k, 5 + m);
      b(i, j, k, 5 + m) = mid;
    }
  }
  for (int m = 1; m < dim; ++m) {
    for (int t = 0; t < m; ++t) {
      b(i, j, k, 5 + m) -= lhs[m * dim + t] * b(i, j, k, 5 + t);
    }
  }
  b(i, j, k, 5 + dim - 1) /= lhs[dim * dim - 1]; // dim*dim-1 = (dim - 1)*dim+(dim - 1)
  for (int m = dim - 2; m >= 0; --m) {
    for (int t = m + 1; t < dim; ++t) {
      b(i, j, k, 5 + m) -= lhs[m * dim + t] * b(i, j, k, 5 + t);
    }
    b(i, j, k, 5 + m) /= lhs[m * dim + m];
  }
}

__device__ void solve_chem_system(real *lhs, real *rhs, int dim) {
  int ipiv[MAX_SPEC_NUMBER] = {};

  // Column pivot LU decomposition
  for (int n = 0; n < dim; ++n) {
    int ik{n};
    for (int m = n; m < dim; ++m) {
      for (int t = 0; t < n; ++t) {
        lhs[m * dim + n] -= lhs[m * dim + t] * lhs[t * dim + n];
      }
      if (std::abs(lhs[m * dim + n]) > std::abs(lhs[ik * dim + n])) {
        ik = m;
      }
    }
    ipiv[n] = ik;
    if (ik != n) {
      for (int t = 0; t < dim; ++t) {
        const auto mid = lhs[ik * dim + t];
        lhs[ik * dim + t] = lhs[n * dim + t];
        lhs[n * dim + t] = mid;
      }
    }
    for (int p = n + 1; p < dim; ++p) {
      for (int t = 0; t < n; ++t) {
        lhs[n * dim + p] -= lhs[n * dim + t] * lhs[t * dim + p];
      }
    }
    for (int m = n + 1; m < dim; ++m) {
      lhs[m * dim + n] /= lhs[n * dim + n];
    }
  }

  // Solve the linear system with LU matrix
  for (int m = 0; m < dim; ++m) {
    const int t = ipiv[m];
    if (t != m) {
      const auto mid = rhs[t];
      rhs[t] = rhs[m];
      rhs[m] = mid;
    }
  }
  for (int m = 1; m < dim; ++m) {
    for (int t = 0; t < m; ++t) {
      rhs[m] -= lhs[m * dim + t] * rhs[t];
    }
  }
  rhs[dim - 1] /= lhs[dim * dim - 1]; // dim*dim-1 = (dim - 1)*dim+(dim - 1)
  for (int m = dim - 2; m >= 0; --m) {
    for (int t = m + 1; t < dim; ++t) {
      rhs[m] -= lhs[m * dim + t] * rhs[t];
    }
    rhs[m] /= lhs[m * dim + m];
  }
}

__global__ void DA(DZone *zone, int n_spec) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  const real dt{zone->dt_local(i, j, k)};
  auto &chem_jac = zone->chem_src_jac;
  for (int l = 0; l < n_spec; ++l) {
    zone->dq(i, j, k, 5 + l) /= 1 - dt * chem_jac(i, j, k, l);
  }
}

} // cfd