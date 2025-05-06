#include "InviscidScheme.cuh"
#include "Constants.h"
#include "DParameter.cuh"
#include "Thermo.cuh"
#include "Field.h"

namespace cfd {
template<MixtureModel mix_model>
__global__ void
__launch_bounds__(64, 8)
compute_convective_term_weno_x(DZone *zone, DParameter *param) {
  const int i = static_cast<int>((blockDim.x - 1) * blockIdx.x + threadIdx.x) - 1;
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  const int max_extent = zone->mx;
  if (i >= max_extent) return;

  const int tid = static_cast<int>(threadIdx.x);
  const int block_dim = static_cast<int>(blockDim.x);
  const auto ngg{zone->ngg};
  const auto n_var{param->n_var};
  const auto n_reconstruct{n_var + 2};
  const int n_point = block_dim + 2 * ngg - 1;

  extern __shared__ real s[];
  real *cv = s;
  real *metric = &cv[n_point * n_reconstruct];
  real *jac = &metric[n_point * 3];
  real *fp = &jac[n_point];
  real *fm = &fp[n_point * n_var];
  real *fc = &fm[n_point * n_var];
  real *f_1st = nullptr;
  if (param->positive_preserving)
    f_1st = &fc[block_dim * n_var];

  const int i_shared = tid - 1 + ngg;
  for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
    cv[i_shared * n_reconstruct + l] = zone->cv(i, j, k, l);
  }
  cv[i_shared * n_reconstruct + n_var] = zone->bv(i, j, k, 4);
  if constexpr (mix_model != MixtureModel::Air)
    cv[i_shared * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, j, k);
  else
    cv[i_shared * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, j, k, 5));
  metric[i_shared * 3] = zone->metric(i, j, k)(1, 1);
  metric[i_shared * 3 + 1] = zone->metric(i, j, k)(1, 2);
  metric[i_shared * 3 + 2] = zone->metric(i, j, k)(1, 3);
  jac[i_shared] = zone->jac(i, j, k);

  // ghost cells
  constexpr int max_additional_ghost_point_loaded = 9; // This is for 11th-order weno, with 7 ghost points on each side.
  int ig_shared[max_additional_ghost_point_loaded];
  int additional_loaded{0};
  if (tid < ngg - 1) {
    ig_shared[additional_loaded] = tid;
    const int gi = i - (ngg - 1);

    for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
      cv[tid * n_reconstruct + l] = zone->cv(gi, j, k, l);
    }
    cv[tid * n_reconstruct + n_var] = zone->bv(gi, j, k, 4);
    if constexpr (mix_model != MixtureModel::Air)
      cv[tid * n_reconstruct + n_var + 1] = zone->acoustic_speed(gi, j, k);
    else
      cv[tid * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(gi, j, k, 5));
    metric[tid * 3] = zone->metric(gi, j, k)(1, 1);
    metric[tid * 3 + 1] = zone->metric(gi, j, k)(1, 2);
    metric[tid * 3 + 2] = zone->metric(gi, j, k)(1, 3);
    jac[tid] = zone->jac(gi, j, k);
    ++additional_loaded;
  }
  if (tid > block_dim - ngg - 1 || i > max_extent - ngg - 1) {
    const int iSh = tid + 2 * ngg - 1;
    ig_shared[additional_loaded] = iSh;
    const int gi = i + ngg;
    for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
      cv[iSh * n_reconstruct + l] = zone->cv(gi, j, k, l);
    }
    cv[iSh * n_reconstruct + n_var] = zone->bv(gi, j, k, 4);
    if constexpr (mix_model != MixtureModel::Air)
      cv[iSh * n_reconstruct + n_var + 1] = zone->acoustic_speed(gi, j, k);
    else
      cv[iSh * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(gi, j, k, 5));
    metric[iSh * 3] = zone->metric(gi, j, k)(1, 1);
    metric[iSh * 3 + 1] = zone->metric(gi, j, k)(1, 2);
    metric[iSh * 3 + 2] = zone->metric(gi, j, k)(1, 3);
    jac[iSh] = zone->jac(gi, j, k);
    ++additional_loaded;
  }
  if (i == max_extent - 1 && tid < ngg - 1) {
    const int n_more_left = ngg - 1 - tid - 1;
    for (int m = 0; m < n_more_left; ++m) {
      const int iSh = tid + m + 1;
      ig_shared[additional_loaded] = iSh;
      const int gi = i - (ngg - 1 - m - 1);

      for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
        cv[iSh * n_reconstruct + l] = zone->cv(gi, j, k, l);
      }
      cv[iSh * n_reconstruct + n_var] = zone->bv(gi, j, k, 4);
      if constexpr (mix_model != MixtureModel::Air)
        cv[iSh * n_reconstruct + n_var + 1] = zone->acoustic_speed(gi, j, k);
      else
        cv[iSh * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(gi, j, k, 5));
      metric[iSh * 3] = zone->metric(gi, j, k)(1, 1);
      metric[iSh * 3 + 1] = zone->metric(gi, j, k)(1, 2);
      metric[iSh * 3 + 2] = zone->metric(gi, j, k)(1, 3);
      jac[iSh] = zone->jac(gi, j, k);
      ++additional_loaded;
    }
    const int n_more_right = ngg - 1 - tid;
    for (int m = 0; m < n_more_right; ++m) {
      const int iSh = i_shared + m + 1;
      ig_shared[additional_loaded] = iSh;
      const int gi = i + (m + 1);
      for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
        cv[iSh * n_reconstruct + l] = zone->cv(gi, j, k, l);
      }
      cv[iSh * n_reconstruct + n_var] = zone->bv(gi, j, k, 4);
      if constexpr (mix_model != MixtureModel::Air)
        cv[iSh * n_reconstruct + n_var + 1] = zone->acoustic_speed(gi, j, k);
      else
        cv[iSh * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(gi, j, k, 5));
      metric[iSh * 3] = zone->metric(gi, j, k)(1, 1);
      metric[iSh * 3 + 1] = zone->metric(gi, j, k)(1, 2);
      metric[iSh * 3 + 2] = zone->metric(gi, j, k)(1, 3);
      jac[iSh] = zone->jac(gi, j, k);
      ++additional_loaded;
    }
  }
  __syncthreads();

  // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
  if (const auto sch = param->inviscid_scheme; sch == 51 || sch == 71) {
    compute_weno_flux_cp(cv, param, tid, metric, jac, fc, i_shared, fp, fm, ig_shared, additional_loaded, f_1st);
  } else if (sch == 52 || sch == 72) {
    compute_weno_flux_ch<mix_model>(cv, param, tid, metric, jac, fc, i_shared, fp, fm, ig_shared, additional_loaded,
                                    f_1st);
  }
  __syncthreads();

  if (param->positive_preserving) {
    real dt{0};
    if (param->dt > 0)
      dt = param->dt;
    else
      dt = zone->dt_local(i, j, k);
    positive_preserving_limiter(f_1st, n_var, tid, fc, param, i_shared, dt, i, max_extent, cv, jac);
  }
  __syncthreads();

  if (tid > 0 && i >= zone->iMin && i <= zone->iMax) {
    for (int l = 0; l < n_var; ++l) {
      zone->dq(i, j, k, l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model>
__global__ void
__launch_bounds__(64, 8)
compute_convective_term_weno_y(DZone *zone, DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>((blockDim.y - 1) * blockIdx.y + threadIdx.y) - 1;
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  const int max_extent = zone->my;
  if (j >= max_extent) return;

  const auto tid = static_cast<int>(threadIdx.y);
  const auto block_dim = static_cast<int>(blockDim.y);
  const auto ngg{zone->ngg};
  const auto n_var{param->n_var};
  const auto n_reconstruct{n_var + 2};
  const int n_point = block_dim + 2 * ngg - 1;

  extern __shared__ real s[];
  real *cv = s;
  real *metric = &cv[n_point * n_reconstruct];
  real *jac = &metric[n_point * 3];
  real *fp = &jac[n_point];
  real *fm = &fp[n_point * n_var];
  real *fc = &fm[n_point * n_var];
  real *f_1st = nullptr;
  if (param->positive_preserving)
    f_1st = &fc[block_dim * n_var];

  const int i_shared = tid - 1 + ngg;
  for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
    cv[i_shared * n_reconstruct + l] = zone->cv(i, j, k, l);
  }
  cv[i_shared * n_reconstruct + n_var] = zone->bv(i, j, k, 4);
  if constexpr (mix_model != MixtureModel::Air)
    cv[i_shared * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, j, k);
  else
    cv[i_shared * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, j, k, 5));
  metric[i_shared * 3] = zone->metric(i, j, k)(2, 1);
  metric[i_shared * 3 + 1] = zone->metric(i, j, k)(2, 2);
  metric[i_shared * 3 + 2] = zone->metric(i, j, k)(2, 3);
  jac[i_shared] = zone->jac(i, j, k);

  // ghost cells
  constexpr int max_additional_ghost_point_loaded = 9; // This is for 11th-order weno, with 7 ghost points on each side.
  int ig_shared[max_additional_ghost_point_loaded];
  int additional_loaded{0};
  if (tid < ngg - 1) {
    ig_shared[additional_loaded] = tid;
    const int gj = j - (ngg - 1);
    for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
      cv[tid * n_reconstruct + l] = zone->cv(i, gj, k, l);
    }
    cv[tid * n_reconstruct + n_var] = zone->bv(i, gj, k, 4);
    if constexpr (mix_model != MixtureModel::Air)
      cv[tid * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, gj, k);
    else
      cv[tid * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, gj, k, 5));
    metric[tid * 3] = zone->metric(i, gj, k)(2, 1);
    metric[tid * 3 + 1] = zone->metric(i, gj, k)(2, 2);
    metric[tid * 3 + 2] = zone->metric(i, gj, k)(2, 3);
    jac[tid] = zone->jac(i, gj, k);
    ++additional_loaded;
  }
  if (tid > block_dim - ngg - 1 || j > max_extent - ngg - 1) {
    const int iSh = tid + 2 * ngg - 1;
    ig_shared[additional_loaded] = iSh;
    const int gj = j + ngg;
    for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
      cv[iSh * n_reconstruct + l] = zone->cv(i, gj, k, l);
    }
    cv[iSh * n_reconstruct + n_var] = zone->bv(i, gj, k, 4);
    if constexpr (mix_model != MixtureModel::Air)
      cv[iSh * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, gj, k);
    else
      cv[iSh * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, gj, k, 5));
    metric[iSh * 3] = zone->metric(i, gj, k)(2, 1);
    metric[iSh * 3 + 1] = zone->metric(i, gj, k)(2, 2);
    metric[iSh * 3 + 2] = zone->metric(i, gj, k)(2, 3);
    jac[iSh] = zone->jac(i, gj, k);
    ++additional_loaded;
  }
  if (j == max_extent - 1 && tid < ngg - 1) {
    const int n_more_left = ngg - 1 - tid - 1;
    for (int m = 0; m < n_more_left; ++m) {
      const int iSh = tid + m + 1;
      ig_shared[additional_loaded] = iSh;
      const int gj = j - (ngg - 1 - m - 1);
      for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
        cv[iSh * n_reconstruct + l] = zone->cv(i, gj, k, l);
      }
      cv[iSh * n_reconstruct + n_var] = zone->bv(i, gj, k, 4);
      if constexpr (mix_model != MixtureModel::Air)
        cv[iSh * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, gj, k);
      else
        cv[iSh * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, gj, k, 5));
      metric[iSh * 3] = zone->metric(i, gj, k)(2, 1);
      metric[iSh * 3 + 1] = zone->metric(i, gj, k)(2, 2);
      metric[iSh * 3 + 2] = zone->metric(i, gj, k)(2, 3);
      jac[iSh] = zone->jac(i, gj, k);
      ++additional_loaded;
    }
    const int n_more_right = ngg - 1 - tid;
    for (int m = 0; m < n_more_right; ++m) {
      const int iSh = i_shared + m + 1;
      ig_shared[additional_loaded] = iSh;
      const int gj = j + (m + 1);
      for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
        cv[iSh * n_reconstruct + l] = zone->cv(i, gj, k, l);
      }
      cv[iSh * n_reconstruct + n_var] = zone->bv(i, gj, k, 4);
      if constexpr (mix_model != MixtureModel::Air)
        cv[iSh * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, gj, k);
      else
        cv[iSh * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, gj, k, 5));
      metric[iSh * 3] = zone->metric(i, gj, k)(2, 1);
      metric[iSh * 3 + 1] = zone->metric(i, gj, k)(2, 2);
      metric[iSh * 3 + 2] = zone->metric(i, gj, k)(2, 3);
      jac[iSh] = zone->jac(i, gj, k);
      ++additional_loaded;
    }
  }
  __syncthreads();

  // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
  if (const auto sch = param->inviscid_scheme; sch == 51 || sch == 71) {
    compute_weno_flux_cp(cv, param, tid, metric, jac, fc, i_shared, fp, fm, ig_shared, additional_loaded, f_1st);
  } else if (sch == 52 || sch == 72) {
    compute_weno_flux_ch<mix_model>(cv, param, tid, metric, jac, fc, i_shared, fp, fm, ig_shared, additional_loaded,
                                    f_1st);
  }
  __syncthreads();

  if (param->positive_preserving) {
    real dt{0};
    if (param->dt > 0)
      dt = param->dt;
    else
      dt = zone->dt_local(i, j, k);
    positive_preserving_limiter(f_1st, n_var, tid, fc, param, i_shared, dt, j, max_extent, cv, jac);
  }
  __syncthreads();

  if (tid > 0 && j >= zone->jMin && j <= zone->jMax) {
    for (int l = 0; l < n_var; ++l) {
      zone->dq(i, j, k, l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model>
__global__ void
__launch_bounds__(64, 8)
compute_convective_term_weno_z(DZone *zone, DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>((blockDim.z - 1) * blockIdx.z + threadIdx.z) - 1;
  const int max_extent = zone->mz;
  if (k >= max_extent) return;

  const auto tid = static_cast<int>(threadIdx.z);
  const auto block_dim = static_cast<int>(blockDim.z);
  const auto ngg{zone->ngg};
  const auto n_var{param->n_var};
  const auto n_reconstruct{n_var + 2};
  const int n_point = block_dim + 2 * ngg - 1;

  extern __shared__ real s[];
  real *cv = s;
  real *metric = &cv[n_point * n_reconstruct];
  real *jac = &metric[n_point * 3];
  real *fp = &jac[n_point];
  real *fm = &fp[n_point * n_var];
  real *fc = &fm[n_point * n_var];
  real *f_1st = nullptr;
  if (param->positive_preserving)
    f_1st = &fc[block_dim * n_var];

  const int i_shared = tid - 1 + ngg;
  for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
    cv[i_shared * n_reconstruct + l] = zone->cv(i, j, k, l);
  }
  cv[i_shared * n_reconstruct + n_var] = zone->bv(i, j, k, 4);
  if constexpr (mix_model != MixtureModel::Air)
    cv[i_shared * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, j, k);
  else
    cv[i_shared * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, j, k, 5));
  metric[i_shared * 3] = zone->metric(i, j, k)(3, 1);
  metric[i_shared * 3 + 1] = zone->metric(i, j, k)(3, 2);
  metric[i_shared * 3 + 2] = zone->metric(i, j, k)(3, 3);
  jac[i_shared] = zone->jac(i, j, k);

  // ghost cells
  constexpr int max_additional_ghost_point_loaded = 9; // This is for 11th-order weno, with 7 ghost points on each side.
  int ig_shared[max_additional_ghost_point_loaded];
  int additional_loaded{0};
  if (tid < ngg - 1) {
    ig_shared[additional_loaded] = tid;
    const int gk = k - (ngg - 1);
    for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
      cv[tid * n_reconstruct + l] = zone->cv(i, j, gk, l);
    }
    cv[tid * n_reconstruct + n_var] = zone->bv(i, j, gk, 4);
    if constexpr (mix_model != MixtureModel::Air)
      cv[tid * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, j, gk);
    else
      cv[tid * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, j, gk, 5));
    metric[tid * 3] = zone->metric(i, j, gk)(3, 1);
    metric[tid * 3 + 1] = zone->metric(i, j, gk)(3, 2);
    metric[tid * 3 + 2] = zone->metric(i, j, gk)(3, 3);
    jac[tid] = zone->jac(i, j, gk);
    ++additional_loaded;
  }
  if (tid > block_dim - ngg - 1 || k > max_extent - ngg - 1) {
    const int iSh = tid + 2 * ngg - 1;
    ig_shared[additional_loaded] = iSh;
    const int gk = k + ngg;
    for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
      cv[iSh * n_reconstruct + l] = zone->cv(i, j, gk, l);
    }
    cv[iSh * n_reconstruct + n_var] = zone->bv(i, j, gk, 4);
    if constexpr (mix_model != MixtureModel::Air)
      cv[iSh * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, j, gk);
    else
      cv[iSh * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, j, gk, 5));
    metric[iSh * 3] = zone->metric(i, j, gk)(3, 1);
    metric[iSh * 3 + 1] = zone->metric(i, j, gk)(3, 2);
    metric[iSh * 3 + 2] = zone->metric(i, j, gk)(3, 3);
    jac[iSh] = zone->jac(i, j, gk);
    ++additional_loaded;
  }
  if (k == max_extent - 1 && tid < ngg - 1) {
    const int n_more_left = ngg - 1 - tid - 1;
    for (int m = 0; m < n_more_left; ++m) {
      const int iSh = tid + m + 1;
      ig_shared[additional_loaded] = iSh;
      const int gk = k - (ngg - 1 - m - 1);
      for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
        cv[iSh * n_reconstruct + l] = zone->cv(i, j, gk, l);
      }
      cv[iSh * n_reconstruct + n_var] = zone->bv(i, j, gk, 4);
      if constexpr (mix_model != MixtureModel::Air)
        cv[iSh * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, j, gk);
      else
        cv[iSh * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, j, gk, 5));
      metric[iSh * 3] = zone->metric(i, j, gk)(3, 1);
      metric[iSh * 3 + 1] = zone->metric(i, j, gk)(3, 2);
      metric[iSh * 3 + 2] = zone->metric(i, j, gk)(3, 3);
      jac[iSh] = zone->jac(i, j, gk);
      ++additional_loaded;
    }
    const int n_more_right = ngg - 1 - tid;
    for (int m = 0; m < n_more_right; ++m) {
      const int iSh = i_shared + m + 1;
      ig_shared[additional_loaded] = iSh;
      const int gk = k + (m + 1);
      for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E
        cv[iSh * n_reconstruct + l] = zone->cv(i, j, gk, l);
      }
      cv[iSh * n_reconstruct + n_var] = zone->bv(i, j, gk, 4);
      if constexpr (mix_model != MixtureModel::Air)
        cv[iSh * n_reconstruct + n_var + 1] = zone->acoustic_speed(i, j, gk);
      else
        cv[iSh * n_reconstruct + n_var + 1] = sqrt(gamma_air * R_air * zone->bv(i, j, gk, 5));
      metric[iSh * 3] = zone->metric(i, j, gk)(3, 1);
      metric[iSh * 3 + 1] = zone->metric(i, j, gk)(3, 2);
      metric[iSh * 3 + 2] = zone->metric(i, j, gk)(3, 3);
      jac[iSh] = zone->jac(i, j, gk);
      ++additional_loaded;
    }
  }
  __syncthreads();

  // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
  if (const auto sch = param->inviscid_scheme; sch == 51 || sch == 71) {
    compute_weno_flux_cp(cv, param, tid, metric, jac, fc, i_shared, fp, fm, ig_shared, additional_loaded, f_1st);
  } else if (sch == 52 || sch == 72) {
    compute_weno_flux_ch<mix_model>(cv, param, tid, metric, jac, fc, i_shared, fp, fm, ig_shared, additional_loaded,
                                    f_1st);
  }
  __syncthreads();

  if (param->positive_preserving) {
    real dt{0};
    if (param->dt > 0)
      dt = param->dt;
    else
      dt = zone->dt_local(i, j, k);
    positive_preserving_limiter(f_1st, n_var, tid, fc, param, i_shared, dt, k, max_extent, cv, jac);
  }
  __syncthreads();

  if (tid > 0 && k >= zone->kMin && k <= zone->kMax) {
    for (int l = 0; l < n_var; ++l) {
      zone->dq(i, j, k, l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model>
void compute_convective_term_weno(const Block &block, DZone *zone, DParameter *param, int n_var,
                                  const Parameter &parameter) {
  // The implementation of classic WENO.
  const int extent[3]{block.mx, block.my, block.mz};

  constexpr int block_dim = 64;
  const int n_computation_per_block = block_dim + 2 * block.ngg - 1;
  auto shared_mem = (block_dim * n_var                                       // fc
                     + n_computation_per_block * n_var * 2                   // F+/F-
                     + n_computation_per_block * (n_var + 3)) * sizeof(real) // cv[n_var]+p+T+jacobian
                    + n_computation_per_block * 3 * sizeof(real);            // metric[3]
  if (parameter.get_bool("positive_preserving")) {
    shared_mem += block_dim * (n_var - 5) * sizeof(real); // f_1th
  }

  dim3 TPB(block_dim, 1, 1);
  dim3 BPG((extent[0] - 1) / (block_dim - 1) + 1, extent[1], extent[2]);
  compute_convective_term_weno_x<mix_model><<<BPG, TPB, shared_mem>>>(zone, param);

  TPB = dim3(1, block_dim, 1);
  BPG = dim3(extent[0], (extent[1] - 1) / (block_dim - 1) + 1, extent[2]);
  compute_convective_term_weno_y<mix_model><<<BPG, TPB, shared_mem>>>(zone, param);

  if (extent[2] > 1) {
    TPB = dim3(1, 1, 64);
    BPG = dim3(extent[0], extent[1], (extent[2] - 1) / (64 - 1) + 1);
    compute_convective_term_weno_z<mix_model><<<BPG, TPB, shared_mem>>>(zone, param);
  }
}

__device__ void
compute_flux(const real *Q, const DParameter *param, const real *metric, real jac, real *Fk) {
  const int n_var = param->n_var;
  const real jacUk{jac * (metric[0] * Q[1] + metric[1] * Q[2] + metric[2] * Q[3]) / Q[0]};
  const real pk{Q[n_var]};

  Fk[0] = jacUk * Q[0];
  Fk[1] = jacUk * Q[1] + jac * pk * metric[0];
  Fk[2] = jacUk * Q[2] + jac * pk * metric[1];
  Fk[3] = jacUk * Q[3] + jac * pk * metric[2];
  Fk[4] = jacUk * (Q[4] + pk);

  for (int l = 5; l < n_var; ++l) {
    Fk[l] = jacUk * Q[l];
  }
}

__device__ void compute_flux(const real *Q, const DParameter *param, const real *metric, real jac, real *Fp, real *Fm) {
  const int n_var = param->n_var;
  const real Uk{(Q[1] * metric[0] + Q[2] * metric[1] + Q[3] * metric[2]) / Q[0]};
  const real pk{Q[n_var]};
  const real cGradK = Q[n_var + 1] * sqrt(metric[0] * metric[0] + metric[1] * metric[1] + metric[2] * metric[2]);
  const real lambda0 = abs(Uk) + cGradK;

  Fp[0] = 0.5 * jac * (Uk * Q[0] + lambda0 * Q[0]);
  Fp[1] = 0.5 * jac * (Uk * Q[1] + pk * metric[0] + lambda0 * Q[1]);
  Fp[2] = 0.5 * jac * (Uk * Q[2] + pk * metric[1] + lambda0 * Q[2]);
  Fp[3] = 0.5 * jac * (Uk * Q[3] + pk * metric[2] + lambda0 * Q[3]);
  Fp[4] = 0.5 * jac * (Uk * (Q[4] + pk) + lambda0 * Q[4]);

  Fm[0] = 0.5 * jac * (Uk * Q[0] - lambda0 * Q[0]);
  Fm[1] = 0.5 * jac * (Uk * Q[1] + pk * metric[0] - lambda0 * Q[1]);
  Fm[2] = 0.5 * jac * (Uk * Q[2] + pk * metric[1] - lambda0 * Q[2]);
  Fm[3] = 0.5 * jac * (Uk * Q[3] + pk * metric[2] - lambda0 * Q[3]);
  Fm[4] = 0.5 * jac * (Uk * (Q[4] + pk) - lambda0 * Q[4]);

  for (int l = 5; l < n_var; ++l) {
    Fp[l] = 0.5 * jac * (Uk * Q[l] + lambda0 * Q[l]);
    Fm[l] = 0.5 * jac * (Uk * Q[l] - lambda0 * Q[l]);
  }
}

template<MixtureModel mix_model>
__device__ void
compute_weno_flux_ch(const real *cv, DParameter *param, int tid, const real *metric, const real *jac, real *fc,
                     int i_shared, real *Fp, real *Fm, const int *ig_shared, int n_add, real *f_1st) {
  const int n_var = param->n_var;

  // 0: acans; 1: li xinliang(own flux splitting); 2: my(same spectral radius)
  constexpr int method = 1;

  const auto m_l = &metric[i_shared * 3], m_r = &metric[(i_shared + 1) * 3];
  if constexpr (method == 1) {
    compute_flux(&cv[i_shared * (n_var + 2)], param, m_l, jac[i_shared], &Fp[i_shared * n_var], &Fm[i_shared * n_var]);
    for (size_t i = 0; i < n_add; i++) {
      compute_flux(&cv[ig_shared[i] * (n_var + 2)], param, &metric[ig_shared[i] * 3], jac[ig_shared[i]],
                   &Fp[ig_shared[i] * n_var], &Fm[ig_shared[i] * n_var]);
    }
  } else if constexpr (method == 2) {
    compute_flux(&cv[i_shared * (n_var + 2)], param, &metric[i_shared * 3], jac[i_shared], &Fp[i_shared * n_var]);
    for (size_t i = 0; i < n_add; i++) {
      compute_flux(&cv[ig_shared[i] * (n_var + 2)], param, &metric[ig_shared[i] * 3], jac[ig_shared[i]],
                   &Fp[ig_shared[i] * n_var]);
    }
  }

  // The first n_var in the cv array is conservative vars, followed by p and cm.
  const real *cvl{&cv[i_shared * (n_var + 2)]};
  const real *cvr{&cv[(i_shared + 1) * (n_var + 2)]};
  const real rhoL_inv{1.0 / cvl[0]}, rhoR_inv{1.0 / cvr[0]};
  // First, compute the Roe average of the half-point variables.
  const real rlc{sqrt(cvl[0]) / (sqrt(cvl[0]) + sqrt(cvr[0]))};
  const real rrc{sqrt(cvr[0]) / (sqrt(cvl[0]) + sqrt(cvr[0]))};
  const real um{rlc * cvl[1] * rhoL_inv + rrc * cvr[1] * rhoR_inv};
  const real vm{rlc * cvl[2] * rhoL_inv + rrc * cvr[2] * rhoR_inv};
  const real wm{rlc * cvl[3] * rhoL_inv + rrc * cvr[3] * rhoR_inv};
  const real ekm{0.5 * (um * um + vm * vm + wm * wm)};
  const real hl{(cvl[4] + cvl[n_var]) * rhoL_inv};
  const real hr{(cvr[4] + cvr[n_var]) * rhoR_inv};
  const real hm{rlc * hl + rrc * hr};

  real svm[MAX_SPEC_NUMBER] = {};
  for (int l = 0; l < n_var - 5; ++l) {
    svm[l] = rlc * cvl[l + 5] * rhoL_inv + rrc * cvr[l + 5] * rhoR_inv;
  }

  const int n_spec{param->n_spec};
  real mw_inv = 0;
  for (int l = 0; l < n_spec; ++l) {
    mw_inv += svm[l] / param->mw[l];
  }

  const real tl{cvl[n_var] * rhoL_inv};
  const real tr{cvr[n_var] * rhoR_inv};
  const real tm = (rlc * tl + rrc * tr) / (R_u * mw_inv);

  real cp_i[MAX_SPEC_NUMBER], h_i[MAX_SPEC_NUMBER];
  compute_enthalpy_and_cp(tm, h_i, cp_i, param);
  real cp{0}, cv_tot{0};
  for (int l = 0; l < n_spec; ++l) {
    cp += svm[l] * cp_i[l];
    cv_tot += svm[l] * (cp_i[l] - R_u / param->mw[l]);
  }
  const real gamma = cp / cv_tot;
  const real cm = sqrt(gamma * R_u * mw_inv * tm);
  const real gm1{gamma - 1};

  // Next, we compute the left characteristic matrix at i+1/2.
  const real jac_l{jac[i_shared]}, jac_r{jac[i_shared + 1]};
  real kxJ{m_l[0] * jac_l + m_r[0] * jac_r};
  real kyJ{m_l[1] * jac_l + m_r[1] * jac_r};
  real kzJ{m_l[2] * jac_l + m_r[2] * jac_r};
  real kx{kxJ / (jac_l + jac_r)};
  real ky{kyJ / (jac_l + jac_r)};
  real kz{kzJ / (jac_l + jac_r)};
  const real gradK{sqrt(kx * kx + ky * ky + kz * kz)};
  kx /= gradK;
  ky /= gradK;
  kz /= gradK;
  const real Uk_bar{kx * um + ky * vm + kz * wm};
  const real alpha{gm1 * ekm};

  // The matrix we consider here does not contain the turbulent variables, such as tke and omega.
  const real cm2_inv{1.0 / (cm * cm)};
  // Compute the characteristic flux with L.
  real fChar[5 + MAX_SPEC_NUMBER];
  constexpr real eps{1e-40};
  const real eps_scaled = eps * param->weno_eps_scale * 0.25 * (kxJ * kxJ + kyJ * kyJ + kzJ * kzJ);

  real alpha_l[MAX_SPEC_NUMBER];
  // compute the partial derivative of pressure to species density
  for (int l = 0; l < n_spec; ++l) {
    alpha_l[l] = gamma * R_u / param->mw[l] * tm - (gamma - 1) * h_i[l];
    // The computations including this alpha_l are all combined with a division by cm2.
    alpha_l[l] *= cm2_inv;
  }

  if constexpr (method == 1) {
    // Li Xinliang's flux splitting
    if (param->positive_preserving) {
      real spectralRadThis = abs((m_l[0] * cvl[1] + m_l[1] * cvl[2] + m_l[2] * cvl[3]) * rhoL_inv +
                                 cvl[n_var + 1] * sqrt(m_l[0] * m_l[0] + m_l[1] * m_l[1] + m_l[2] * m_l[2]));
      real spectralRadNext = abs((m_r[0] * cvr[1] + m_r[1] * cvr[2] + m_r[2] * cvr[3]) * rhoR_inv +
                                 cvr[n_var + 1] * sqrt(m_r[0] * m_r[0] + m_r[1] * m_r[1] + m_r[2] * m_r[2]));
      for (int l = 0; l < n_var - 5; ++l) {
        f_1st[tid * (n_var - 5) + l] =
            0.5 * (Fp[i_shared * n_var + l + 5] + spectralRadThis * cvl[l + 5] * jac_l) +
            0.5 * (Fp[(i_shared + 1) * n_var + l + 5] - spectralRadNext * cvr[l + 5] * jac_r);
      }
    }

    if (param->inviscid_scheme == 52) {
      for (int l = 0; l < 5; ++l) {
        real coeff_alpha_s{0.5};
        real L[5];
        switch (l) {
          case 0:
            L[0] = (alpha + Uk_bar * cm) * cm2_inv * 0.5;
            L[1] = -(gm1 * um + kx * cm) * cm2_inv * 0.5;
            L[2] = -(gm1 * vm + ky * cm) * cm2_inv * 0.5;
            L[3] = -(gm1 * wm + kz * cm) * cm2_inv * 0.5;
            L[4] = gm1 * cm2_inv * 0.5;
            break;
          case 1:
            coeff_alpha_s = -kx;
            L[0] = kx * (1 - alpha * cm2_inv) - (kz * vm - ky * wm) / cm;
            L[1] = kx * gm1 * um * cm2_inv;
            L[2] = (kx * gm1 * vm + kz * cm) * cm2_inv;
            L[3] = (kx * gm1 * wm - ky * cm) * cm2_inv;
            L[4] = -kx * gm1 * cm2_inv;
            break;
          case 2:
            coeff_alpha_s = -ky;
            L[0] = ky * (1 - alpha * cm2_inv) - (kx * wm - kz * um) / cm;
            L[1] = (ky * gm1 * um - kz * cm) * cm2_inv;
            L[2] = ky * gm1 * vm * cm2_inv;
            L[3] = (ky * gm1 * wm + kx * cm) * cm2_inv;
            L[4] = -ky * gm1 * cm2_inv;
            break;
          case 3:
            coeff_alpha_s = -kz;
            L[0] = kz * (1 - alpha * cm2_inv) - (ky * um - kx * vm) / cm;
            L[1] = (kz * gm1 * um + ky * cm) * cm2_inv;
            L[2] = (kz * gm1 * vm - kx * cm) * cm2_inv;
            L[3] = kz * gm1 * wm * cm2_inv;
            L[4] = -kz * gm1 * cm2_inv;
            break;
          case 4:
            L[0] = (alpha - Uk_bar * cm) * cm2_inv * 0.5;
            L[1] = -(gm1 * um - kx * cm) * cm2_inv * 0.5;
            L[2] = -(gm1 * vm - ky * cm) * cm2_inv * 0.5;
            L[3] = -(gm1 * wm - kz * cm) * cm2_inv * 0.5;
            L[4] = gm1 * cm2_inv * 0.5;
            break;
          default:
            break;
        }
        real vPlus[5] = {}, vMinus[5] = {};
        for (int m = 0; m < 5; ++m) {
          for (int n = 0; n < 5; ++n) {
            vPlus[m] += L[n] * Fp[(i_shared - 3 + m) * n_var + n];
            vMinus[m] += L[n] * Fm[(i_shared - 2 + m) * n_var + n];
          }
          for (int n = 0; n < n_spec; ++n) {
            vPlus[m] += coeff_alpha_s * alpha_l[n] * Fp[(i_shared - 2 + m) * n_var + 5 + n];
            vMinus[m] += coeff_alpha_s * alpha_l[n] * Fm[(i_shared - 1 + m) * n_var + 5 + n];
          }
        }
        fChar[l] = WENO5(vPlus, vMinus, eps_scaled);
      }
      for (int l = 0; l < n_spec; ++l) {
        real vPlus[5], vMinus[5];
        for (int m = 0; m < 5; ++m) {
          vPlus[m] = -svm[l] * Fp[(i_shared - 2 + m) * n_var] + Fp[(i_shared - 2 + m) * n_var + 5 + l];
          vMinus[m] = -svm[l] * Fm[(i_shared - 1 + m) * n_var] + Fm[(i_shared - 1 + m) * n_var + 5 + l];
        }
        fChar[5 + l] = WENO5(vPlus, vMinus, eps_scaled);
      }
    } else if (param->inviscid_scheme == 72) {
      for (int l = 0; l < 5; ++l) {
        real coeff_alpha_s{0.5};
        real L[5];
        switch (l) {
          case 0:
            L[0] = (alpha + Uk_bar * cm) * cm2_inv * 0.5;
            L[1] = -(gm1 * um + kx * cm) * cm2_inv * 0.5;
            L[2] = -(gm1 * vm + ky * cm) * cm2_inv * 0.5;
            L[3] = -(gm1 * wm + kz * cm) * cm2_inv * 0.5;
            L[4] = gm1 * cm2_inv * 0.5;
            break;
          case 1:
            coeff_alpha_s = -kx;
            L[0] = kx * (1 - alpha * cm2_inv) - (kz * vm - ky * wm) / cm;
            L[1] = kx * gm1 * um * cm2_inv;
            L[2] = (kx * gm1 * vm + kz * cm) * cm2_inv;
            L[3] = (kx * gm1 * wm - ky * cm) * cm2_inv;
            L[4] = -kx * gm1 * cm2_inv;
            break;
          case 2:
            coeff_alpha_s = -ky;
            L[0] = ky * (1 - alpha * cm2_inv) - (kx * wm - kz * um) / cm;
            L[1] = (ky * gm1 * um - kz * cm) * cm2_inv;
            L[2] = ky * gm1 * vm * cm2_inv;
            L[3] = (ky * gm1 * wm + kx * cm) * cm2_inv;
            L[4] = -ky * gm1 * cm2_inv;
            break;
          case 3:
            coeff_alpha_s = -kz;
            L[0] = kz * (1 - alpha * cm2_inv) - (ky * um - kx * vm) / cm;
            L[1] = (kz * gm1 * um + ky * cm) * cm2_inv;
            L[2] = (kz * gm1 * vm - kx * cm) * cm2_inv;
            L[3] = kz * gm1 * wm * cm2_inv;
            L[4] = -kz * gm1 * cm2_inv;
            break;
          case 4:
            L[0] = (alpha - Uk_bar * cm) * cm2_inv * 0.5;
            L[1] = -(gm1 * um - kx * cm) * cm2_inv * 0.5;
            L[2] = -(gm1 * vm - ky * cm) * cm2_inv * 0.5;
            L[3] = -(gm1 * wm - kz * cm) * cm2_inv * 0.5;
            L[4] = gm1 * cm2_inv * 0.5;
            break;
          default:
            break;
        }

        real vPlus[7] = {}, vMinus[7] = {};
        for (int m = 0; m < 7; ++m) {
          for (int n = 0; n < 5; ++n) {
            vPlus[m] += L[n] * Fp[(i_shared - 3 + m) * n_var + n];
            vMinus[m] += L[n] * Fm[(i_shared - 2 + m) * n_var + n];
          }
          for (int n = 0; n < n_spec; ++n) {
            vPlus[m] += coeff_alpha_s * alpha_l[n] * Fp[(i_shared - 3 + m) * n_var + 5 + n];
            vMinus[m] += coeff_alpha_s * alpha_l[n] * Fm[(i_shared - 2 + m) * n_var + 5 + n];
          }
        }
        fChar[l] = WENO7(vPlus, vMinus, eps_scaled);
      }
      for (int l = 0; l < n_spec; ++l) {
        real vPlus[7], vMinus[7];
        for (int m = 0; m < 7; ++m) {
          vPlus[m] = -svm[l] * Fp[(i_shared - 3 + m) * n_var] + Fp[(i_shared - 3 + m) * n_var + 5 + l];
          vMinus[m] = -svm[l] * Fm[(i_shared - 2 + m) * n_var] + Fm[(i_shared - 2 + m) * n_var + 5 + l];
        }
        fChar[5 + l] = WENO7(vPlus, vMinus, eps_scaled);
      }
    }
  } else {
    // My method
    gxl::Matrix<real, 5, 5> LR;
    LR(0, 0) = (alpha + Uk_bar * cm) * cm2_inv * 0.5;
    LR(0, 1) = -(gm1 * um + kx * cm) * cm2_inv * 0.5;
    LR(0, 2) = -(gm1 * vm + ky * cm) * cm2_inv * 0.5;
    LR(0, 3) = -(gm1 * wm + kz * cm) * cm2_inv * 0.5;
    LR(0, 4) = gm1 * cm2_inv * 0.5;
    LR(1, 0) = kx * (1 - alpha * cm2_inv) - (kz * vm - ky * wm) / cm;
    LR(1, 1) = kx * gm1 * um * cm2_inv;
    LR(1, 2) = (kx * gm1 * vm + kz * cm) * cm2_inv;
    LR(1, 3) = (kx * gm1 * wm - ky * cm) * cm2_inv;
    LR(1, 4) = -kx * gm1 * cm2_inv;
    LR(2, 0) = ky * (1 - alpha * cm2_inv) - (kx * wm - kz * um) / cm;
    LR(2, 1) = (ky * gm1 * um - kz * cm) * cm2_inv;
    LR(2, 2) = ky * gm1 * vm * cm2_inv;
    LR(2, 3) = (ky * gm1 * wm + kx * cm) * cm2_inv;
    LR(2, 4) = -ky * gm1 * cm2_inv;
    LR(3, 0) = kz * (1 - alpha * cm2_inv) - (ky * um - kx * vm) / cm;
    LR(3, 1) = (kz * gm1 * um + ky * cm) * cm2_inv;
    LR(3, 2) = (kz * gm1 * vm - kx * cm) * cm2_inv;
    LR(3, 3) = kz * gm1 * wm * cm2_inv;
    LR(3, 4) = -kz * gm1 * cm2_inv;
    LR(4, 0) = (alpha - Uk_bar * cm) * cm2_inv * 0.5;
    LR(4, 1) = -(gm1 * um - kx * cm) * cm2_inv * 0.5;
    LR(4, 2) = -(gm1 * vm - ky * cm) * cm2_inv * 0.5;
    LR(4, 3) = -(gm1 * wm - kz * cm) * cm2_inv * 0.5;
    LR(4, 4) = gm1 * cm2_inv * 0.5;

    real spec_rad[3] = {}, spectralRadThis, spectralRadNext;
    bool pp_limiter{param->positive_preserving};
    for (int l = -2; l < 4; ++l) {
      const real *Q = &cv[(i_shared + l) * (n_var + 2)];
      real c = Q[n_var + 1];
      real grad_k = sqrt(metric[(i_shared + l) * 3] * metric[(i_shared + l) * 3] +
                         metric[(i_shared + l) * 3 + 1] * metric[(i_shared + l) * 3 + 1] +
                         metric[(i_shared + l) * 3 + 2] * metric[(i_shared + l) * 3 + 2]);
      real Uk = (metric[(i_shared + l) * 3] * Q[1] + metric[(i_shared + l) * 3 + 1] * Q[2] +
                 metric[(i_shared + l) * 3 + 2] * Q[3]) / Q[0];
      real ukPc = abs(Uk + c * grad_k);
      real ukMc = abs(Uk - c * grad_k);
      spec_rad[0] = max(spec_rad[0], ukMc);
      spec_rad[1] = max(spec_rad[1], abs(Uk));
      spec_rad[2] = max(spec_rad[2], ukPc);
      if (pp_limiter && l == 0)
        spectralRadThis = ukPc;
      if (pp_limiter && l == 1)
        spectralRadNext = ukPc;
    }
    spec_rad[0] = max(spec_rad[0], abs((Uk_bar - cm) * gradK));
    spec_rad[1] = max(spec_rad[1], abs(Uk_bar * gradK));
    spec_rad[2] = max(spec_rad[2], abs((Uk_bar + cm) * gradK));

    if (pp_limiter) {
      for (int l = 0; l < n_var - 5; ++l) {
        f_1st[tid * (n_var - 5) + l] =
            0.5 * (Fp[i_shared * n_var + l + 5] + spectralRadThis * cv[i_shared * (n_var + 2) + l + 5] * jac_l) +
            0.5 *
            (Fp[(i_shared + 1) * n_var + l + 5] - spectralRadNext * cv[(i_shared + 1) * (n_var + 2) + l + 5] * jac_r);
      }
    }

    for (int l = 0; l < 5; ++l) {
      real lambda_l{spec_rad[1]};
      if (l == 0) {
        lambda_l = spec_rad[0];
      } else if (l == 4) {
        lambda_l = spec_rad[2];
      }
      real coeff_alpha_s{0.5};
      if (l == 1) {
        coeff_alpha_s = -kx;
      } else if (l == 2) {
        coeff_alpha_s = -ky;
      } else if (l == 3) {
        coeff_alpha_s = -kz;
      }

      if (param->inviscid_scheme == 52) {
        real vPlus[5] = {}, vMinus[5] = {};
        for (int m = 0; m < 5; ++m) {
          for (int n = 0; n < 5; ++n) {
            vPlus[m] += LR(l, n) * (Fp[(i_shared - 2 + m) * n_var + n] +
                                    lambda_l * cv[(i_shared - 2 + m) * (n_var + 2) + n] * jac[i_shared - 2 + m]);
            vMinus[m] += LR(l, n) * (Fp[(i_shared - 1 + m) * n_var + n] -
                                     lambda_l * cv[(i_shared - 1 + m) * (n_var + 2) + n] * jac[i_shared - 1 + m]);
          }
          for (int n = 0; n < n_spec; ++n) {
            vPlus[m] += coeff_alpha_s * alpha_l[n] * (Fp[(i_shared - 2 + m) * n_var + 5 + n] +
                                                      lambda_l * cv[(i_shared - 2 + m) * (n_var + 2) + n + 5] *
                                                      jac[i_shared - 2 + m]);
            vMinus[m] += coeff_alpha_s * alpha_l[n] * (Fp[(i_shared - 1 + m) * n_var + 5 + n] -
                                                       lambda_l * cv[(i_shared - 1 + m) * (n_var + 2) + n + 5] *
                                                       jac[i_shared - 1 + m]);
          }
          vPlus[m] *= 0.5;
          vMinus[m] *= 0.5;
        }
        fChar[l] = WENO5(vPlus, vMinus, eps_scaled);
      } else if (param->inviscid_scheme == 72) {
        real vPlus[7] = {}, vMinus[7] = {};
        for (int m = 0; m < 7; ++m) {
          for (int n = 0; n < 5; ++n) {
            vPlus[m] += LR(l, n) * (Fp[(i_shared - 3 + m) * n_var + n] +
                                    lambda_l * cv[(i_shared - 3 + m) * (n_var + 2) + n] * jac[i_shared - 3 + m]);
            vMinus[m] += LR(l, n) * (Fp[(i_shared - 2 + m) * n_var + n] -
                                     lambda_l * cv[(i_shared - 2 + m) * (n_var + 2) + n] * jac[i_shared - 2 + m]);
          }
          for (int n = 0; n < n_spec; ++n) {
            vPlus[m] += coeff_alpha_s * alpha_l[n] * (Fp[(i_shared - 3 + m) * n_var + 5 + n] +
                                                      lambda_l * cv[(i_shared - 3 + m) * (n_var + 2) + 5 + n] *
                                                      jac[i_shared - 3 + m]);
            vMinus[m] += coeff_alpha_s * alpha_l[n] * (Fp[(i_shared - 2 + m) * n_var + 5 + n] -
                                                       lambda_l * cv[(i_shared - 2 + m) * (n_var + 2) + 5 + n] *
                                                       jac[i_shared - 2 + m]);
          }
          vPlus[m] *= 0.5;
          vMinus[m] *= 0.5;
        }
        fChar[l] = WENO7(vPlus, vMinus, eps_scaled);
      }
    }
    for (int l = 0; l < n_spec; ++l) {
      const real lambda_l{spec_rad[1]};
      if (param->inviscid_scheme == 52) {
        real vPlus[5], vMinus[5];
        for (int m = 0; m < 5; ++m) {
          vPlus[m] = 0.5 * (Fp[(i_shared - 2 + m) * n_var + 5 + l] +
                            lambda_l * cv[(i_shared - 2 + m) * (n_var + 2) + 5 + l] * jac[i_shared - 2 + m] -
                            svm[l] * (Fp[(i_shared - 2 + m) * n_var] +
                                      lambda_l * cv[(i_shared - 2 + m) * (n_var + 2)] * jac[i_shared - 2 + m]));
          vMinus[m] = 0.5 * (Fp[(i_shared - 1 + m) * n_var + 5 + l] -
                             lambda_l * cv[(i_shared - 1 + m) * (n_var + 2) + 5 + l] * jac[i_shared - 1 + m] -
                             svm[l] * (Fp[(i_shared - 1 + m) * n_var] -
                                       lambda_l * cv[(i_shared - 1 + m) * (n_var + 2)] * jac[i_shared - 1 + m]));
        }
        fChar[5 + l] = WENO5(vPlus, vMinus, eps_scaled);
      } else if (param->inviscid_scheme == 72) {
        real vPlus[7], vMinus[7];
        for (int m = 0; m < 7; ++m) {
          vPlus[m] = 0.5 * (Fp[(i_shared - 3 + m) * n_var + 5 + l] +
                            lambda_l * cv[(i_shared - 3 + m) * (n_var + 2) + 5 + l] * jac[i_shared - 3 + m] -
                            svm[l] * (Fp[(i_shared - 3 + m) * n_var] +
                                      lambda_l * cv[(i_shared - 3 + m) * (n_var + 2)] * jac[i_shared - 3 + m]));
          vMinus[m] = 0.5 * (Fp[(i_shared - 2 + m) * n_var + 5 + l] -
                             lambda_l * cv[(i_shared - 2 + m) * (n_var + 2) + 5 + l] * jac[i_shared - 2 + m] -
                             svm[l] * (Fp[(i_shared - 2 + m) * n_var] -
                                       lambda_l * cv[(i_shared - 2 + m) * (n_var + 2)] * jac[i_shared - 2 + m]));
        }
        fChar[5 + l] = WENO7(vPlus, vMinus, eps_scaled);
      }
    }
  }

  // Project the flux back to physical space
  // We do not compute the right characteristic matrix here, because we explicitly write the components below.
  auto fci = &fc[tid * n_var];
  fci[0] = fChar[0] + kx * fChar[1] + ky * fChar[2] + kz * fChar[3] + fChar[4];
  fci[1] = (um - kx * cm) * fChar[0] + kx * um * fChar[1] + (ky * um - kz * cm) * fChar[2] +
           (kz * um + ky * cm) * fChar[3] + (um + kx * cm) * fChar[4];
  fci[2] = (vm - ky * cm) * fChar[0] + (kx * vm + kz * cm) * fChar[1] + ky * vm * fChar[2] +
           (kz * vm - kx * cm) * fChar[3] + (vm + ky * cm) * fChar[4];
  fci[3] = (wm - kz * cm) * fChar[0] + (kx * wm - ky * cm) * fChar[1] + (ky * wm + kx * cm) * fChar[2] +
           kz * wm * fChar[3] + (wm + kz * cm) * fChar[4];

  fci[4] = (hm - Uk_bar * cm) * fChar[0] + (kx * (hm - cm * cm / gm1) + (kz * vm - ky * wm) * cm) * fChar[1] +
           (ky * (hm - cm * cm / gm1) + (kx * wm - kz * um) * cm) * fChar[2] +
           (kz * (hm - cm * cm / gm1) + (ky * um - kx * vm) * cm) * fChar[3] +
           (hm + Uk_bar * cm) * fChar[4];
  real add{0};
  const real coeff_add = fChar[0] + kx * fChar[1] + ky * fChar[2] + kz * fChar[3] + fChar[4];
  for (int l = 0; l < n_spec; ++l) {
    fci[5 + l] = svm[l] * coeff_add + fChar[l + 5];
    add += alpha_l[l] * fChar[l + 5];
  }
  fci[4] -= add * cm * cm / gm1;
}

// The above function can actually realize the following ability, but the speed is slower than the specific version.
// Thus, we keep the current version.
template<>
__device__ void
compute_weno_flux_ch<MixtureModel::Air>(const real *cv, DParameter *param, int tid, const real *metric,
                                        const real *jac, real *fc, int i_shared, real *Fp, real *Fm,
                                        const int *ig_shared, int n_add, [[maybe_unused]] real *f_1st) {
  const int n_var = param->n_var;

  // 0: acans; 1: li xinliang(own flux splitting); 2: my(same spectral radius)
  constexpr int method = 0;

  if constexpr (method == 1) {
    compute_flux(&cv[i_shared * (n_var + 2)], param, &metric[i_shared * 3], jac[i_shared], &Fp[i_shared * n_var],
                 &Fm[i_shared * n_var]);
    for (size_t i = 0; i < n_add; i++) {
      compute_flux(&cv[ig_shared[i] * (n_var + 2)], param, &metric[ig_shared[i] * 3], jac[ig_shared[i]],
                   &Fp[ig_shared[i] * n_var], &Fm[ig_shared[i] * n_var]);
    }
  } else if constexpr (method == 2) {
    compute_flux(&cv[i_shared * (n_var + 2)], param, &metric[i_shared * 3], jac[i_shared], &Fp[i_shared * n_var]);
    for (size_t i = 0; i < n_add; i++) {
      compute_flux(&cv[ig_shared[i] * (n_var + 2)], param, &metric[ig_shared[i] * 3], jac[ig_shared[i]],
                   &Fp[ig_shared[i] * n_var]);
    }
  }

  // The first n_var in the cv array is conservative vars, followed by p and cm.
  const real *cvl{&cv[i_shared * (n_var + 2)]};
  const real *cvr{&cv[(i_shared + 1) * (n_var + 2)]};
  // First, compute the Roe average of the half-point variables.
  const real rlc{sqrt(cvl[0]) / (sqrt(cvl[0]) + sqrt(cvr[0]))};
  const real rrc{sqrt(cvr[0]) / (sqrt(cvl[0]) + sqrt(cvr[0]))};
  const real um{rlc * cvl[1] / cvl[0] + rrc * cvr[1] / cvr[0]};
  const real vm{rlc * cvl[2] / cvl[0] + rrc * cvr[2] / cvr[0]};
  const real wm{rlc * cvl[3] / cvl[0] + rrc * cvr[3] / cvr[0]};
  const real ekm{0.5 * (um * um + vm * vm + wm * wm)};
  constexpr real gm1{gamma_air - 1};
  const real hl{(cvl[4] + cvl[n_var]) / cvl[0]};
  const real hr{(cvr[4] + cvr[n_var]) / cvr[0]};
  const real hm{rlc * hl + rrc * hr};
  const real cm2{gm1 * (hm - ekm)};
  const real cm{sqrt(cm2)};

  // Next, we compute the left characteristic matrix at i+1/2.
  real kx{
    (jac[i_shared] * metric[i_shared * 3] + jac[i_shared + 1] * metric[(i_shared + 1) * 3]) /
    (jac[i_shared] + jac[i_shared + 1])
  };
  real ky{
    (jac[i_shared] * metric[i_shared * 3 + 1] + jac[i_shared + 1] * metric[(i_shared + 1) * 3 + 1]) /
    (jac[i_shared] + jac[i_shared + 1])
  };
  real kz{
    (jac[i_shared] * metric[i_shared * 3 + 2] + jac[i_shared + 1] * metric[(i_shared + 1) * 3 + 2]) /
    (jac[i_shared] + jac[i_shared + 1])
  };
  const real gradK{sqrt(kx * kx + ky * ky + kz * kz)};
  kx /= gradK;
  ky /= gradK;
  kz /= gradK;
  const real Uk_bar{kx * um + ky * vm + kz * wm};
  const real alpha{gm1 * ekm};

  // The matrix we consider here does not contain the turbulent variables, such as tke and omega.
  // The method which contains turbulent variables is not implemented yet.
  gxl::Matrix<real, 5, 5> LR;
  LR(0, 0) = (alpha + Uk_bar * cm) / cm2 * 0.5;
  LR(0, 1) = -(gm1 * um + kx * cm) / cm2 * 0.5;
  LR(0, 2) = -(gm1 * vm + ky * cm) / cm2 * 0.5;
  LR(0, 3) = -(gm1 * wm + kz * cm) / cm2 * 0.5;
  LR(0, 4) = gm1 / cm2 * 0.5;
  LR(1, 0) = kx * (1 - alpha / cm2) - (kz * vm - ky * wm) / cm;
  LR(1, 1) = kx * gm1 * um / cm2;
  LR(1, 2) = (kx * gm1 * vm + kz * cm) / cm2;
  LR(1, 3) = (kx * gm1 * wm - ky * cm) / cm2;
  LR(1, 4) = -kx * gm1 / cm2;
  LR(2, 0) = ky * (1 - alpha / cm2) - (kx * wm - kz * um) / cm;
  LR(2, 1) = (ky * gm1 * um - kz * cm) / cm2;
  LR(2, 2) = ky * gm1 * vm / cm2;
  LR(2, 3) = (ky * gm1 * wm + kx * cm) / cm2;
  LR(2, 4) = -ky * gm1 / cm2;
  LR(3, 0) = kz * (1 - alpha / cm2) - (ky * um - kx * vm) / cm;
  LR(3, 1) = (kz * gm1 * um + ky * cm) / cm2;
  LR(3, 2) = (kz * gm1 * vm - kx * cm) / cm2;
  LR(3, 3) = kz * gm1 * wm / cm2;
  LR(3, 4) = -kz * gm1 / cm2;
  LR(4, 0) = (alpha - Uk_bar * cm) / cm2 * 0.5;
  LR(4, 1) = -(gm1 * um - kx * cm) / cm2 * 0.5;
  LR(4, 2) = -(gm1 * vm - ky * cm) / cm2 * 0.5;
  LR(4, 3) = -(gm1 * wm - kz * cm) / cm2 * 0.5;
  LR(4, 4) = gm1 / cm2 * 0.5;

  // Compute the characteristic flux with L.
  real fChar[5];
  constexpr real eps{1e-8};
  // const real eps_scaled = eps * 2813.079264884305;  // If we want to be completely the same as Tian's case, this should be U_1^2 * rho^2, which is dU^2*rho^2 now.
  const real eps_scaled = eps * param->weno_eps_scale;  // If we want to be completely the same as Tian's case, this should be U_1^2 * rho^2, which is dU^2*rho^2 now.
  // constexpr real eps{1e-40};
  // const real jac1{jac[i_shared]}, jac2{jac[i_shared + 1]};
  // const real eps_scaled = eps * param->weno_eps_scale * 0.25 *
  //                         ((metric[i_shared * 3] * jac1 + metric[(i_shared + 1) * 3] * jac2) *
  //                          (metric[i_shared * 3] * jac1 + metric[(i_shared + 1) * 3] * jac2) +
  //                          (metric[i_shared * 3 + 1] * jac1 + metric[(i_shared + 1) * 3 + 1] * jac2) *
  //                          (metric[i_shared * 3 + 1] * jac1 + metric[(i_shared + 1) * 3 + 1] * jac2) +
  //                          (metric[i_shared * 3 + 2] * jac1 + metric[(i_shared + 1) * 3 + 2] * jac2) *
  //                          (metric[i_shared * 3 + 2] * jac1 + metric[(i_shared + 1) * 3 + 2] * jac2));

  if constexpr (method == 0) {
    // ACANS version
    real ap[3], an[3];
    const auto max_spec_rad = abs(Uk_bar) + cm;
    ap[0] = 0.5 * (Uk_bar - cm + max_spec_rad) * gradK;
    ap[1] = 0.5 * (Uk_bar + max_spec_rad) * gradK;
    ap[2] = 0.5 * (Uk_bar + cm + max_spec_rad) * gradK;
    an[0] = 0.5 * (Uk_bar - cm - max_spec_rad) * gradK;
    an[1] = 0.5 * (Uk_bar - max_spec_rad) * gradK;
    an[2] = 0.5 * (Uk_bar + cm - max_spec_rad) * gradK;
    if (param->inviscid_scheme == 52) {
      for (int l = 0; l < 5; ++l) {
        real vPlus[5] = {}, vMinus[5] = {};
        // ACANS version
        real lambda_p{ap[1]}, lambda_n{an[1]};
        if (l == 0) {
          lambda_p = ap[0];
          lambda_n = an[0];
        } else if (l == 4) {
          lambda_p = ap[2];
          lambda_n = an[2];
        }
        for (int m = 0; m < 5; m++) {
          for (int n = 0; n < 5; ++n) {
            vPlus[m] += lambda_p * LR(l, n) * cv[(i_shared - 2 + m) * (n_var + 2) + n] * 0.5 *
                (jac[i_shared] + jac[i_shared + 1]);
            vMinus[m] += lambda_n * LR(l, n) * cv[(i_shared - 1 + m) * (n_var + 2) + n] * 0.5 *
                (jac[i_shared] + jac[i_shared + 1]);
          }
        }
        fChar[l] = WENO5(vPlus, vMinus, eps_scaled);
      }
    } else if (param->inviscid_scheme == 72) {
      for (int l = 0; l < 5; ++l) {
        real vPlus[7] = {}, vMinus[7] = {};
        // ACANS version
        real lambda_p{ap[1]}, lambda_n{an[1]};
        if (l == 0) {
          lambda_p = ap[0];
          lambda_n = an[0];
        } else if (l == 4) {
          lambda_p = ap[2];
          lambda_n = an[2];
        }
        for (int m = 0; m < 7; m++) {
          for (int n = 0; n < 5; ++n) {
            vPlus[m] += lambda_p * LR(l, n) * cv[(i_shared - 3 + m) * (n_var + 2) + n] * 0.5 *
                (jac[i_shared] + jac[i_shared + 1]);
            vMinus[m] += lambda_n * LR(l, n) * cv[(i_shared - 2 + m) * (n_var + 2) + n] * 0.5 *
                (jac[i_shared] + jac[i_shared + 1]);
          }
        }
        fChar[l] = WENO7(vPlus, vMinus, eps_scaled);
      }
    }
  } else if constexpr (method == 1) {
    // Li Xinliang version
    if (param->inviscid_scheme == 52) {
      for (int l = 0; l < 5; ++l) {
        real vPlus[5] = {}, vMinus[5] = {};
        for (int m = 0; m < 5; m++) {
          for (int n = 0; n < 5; n++) {
            vPlus[m] += LR(l, n) * Fp[(i_shared - 2 + m) * 5 + n];
            vMinus[m] += LR(l, n) * Fm[(i_shared - 1 + m) * 5 + n];
          }
        }
        fChar[l] = WENO5(vPlus, vMinus, eps_scaled);
      }
    } else if (param->inviscid_scheme == 72) {
      for (int l = 0; l < 5; ++l) {
        real vPlus[7] = {}, vMinus[7] = {};
        for (int m = 0; m < 7; ++m) {
          for (int n = 0; n < 5; ++n) {
            vPlus[m] += LR(l, n) * Fp[(i_shared - 3 + m) * 5 + n];
            vMinus[m] += LR(l, n) * Fm[(i_shared - 2 + m) * 5 + n];
          }
        }
        fChar[l] = WENO7(vPlus, vMinus, eps_scaled);
      }
    }
  } else {
    // My method
    real spec_rad[3] = {};

    if (param->inviscid_scheme == 52) {
      for (int l = -2; l < 4; ++l) {
        const real *Q = &cv[(i_shared + l) * (n_var + 2)];
        real c = sqrt(gamma_air * Q[n_var] / Q[0]);
        real grad_k = sqrt(metric[(i_shared + l) * 3] * metric[(i_shared + l) * 3] +
                           metric[(i_shared + l) * 3 + 1] * metric[(i_shared + l) * 3 + 1] +
                           metric[(i_shared + l) * 3 + 2] * metric[(i_shared + l) * 3 + 2]);
        real Uk = (metric[(i_shared + l) * 3] * Q[1] + metric[(i_shared + l) * 3 + 1] * Q[2] +
                   metric[(i_shared + l) * 3 + 2] * Q[3]) / Q[0];
        real ukPc = abs(Uk + c * grad_k);
        real ukMc = abs(Uk - c * grad_k);
        spec_rad[0] = max(spec_rad[0], ukMc);
        spec_rad[1] = max(spec_rad[1], abs(Uk));
        spec_rad[2] = max(spec_rad[2], ukPc);
      }
      spec_rad[0] = max(spec_rad[0], abs((Uk_bar - cm) * gradK));
      spec_rad[1] = max(spec_rad[1], abs(Uk_bar * gradK));
      spec_rad[2] = max(spec_rad[2], abs((Uk_bar + cm) * gradK));

      for (int l = 0; l < 5; ++l) {
        real lambda_l{spec_rad[1]};
        if (l == 0) {
          lambda_l = spec_rad[0];
        } else if (l == 4) {
          lambda_l = spec_rad[2];
        }

        real vPlus[5] = {}, vMinus[5] = {};
        for (int m = 0; m < 5; ++m) {
          for (int n = 0; n < 5; ++n) {
            vPlus[m] += LR(l, n) * (Fp[(i_shared - 2 + m) * 5 + n] + lambda_l * cv[(i_shared - 2 + m) * 7 + n] *
                                    jac[i_shared - 2 + m]);
            vMinus[m] += LR(l, n) * (Fm[(i_shared - 1 + m) * 5 + n] - lambda_l * cv[(i_shared - 1 + m) * 7 + n] *
                                     jac[i_shared - 1 + m]);
          }
          vPlus[m] *= 0.5;
          vMinus[m] *= 0.5;
        }
        fChar[l] = WENO5(vPlus, vMinus, eps_scaled);
      }
    } else if (param->inviscid_scheme == 72) {
      for (int l = -3; l < 5; ++l) {
        const real *Q = &cv[(i_shared + l) * (n_var + 2)];
        real c = sqrt(gamma_air * Q[n_var] / Q[0]);
        real grad_k = sqrt(metric[(i_shared + l) * 3] * metric[(i_shared + l) * 3] +
                           metric[(i_shared + l) * 3 + 1] * metric[(i_shared + l) * 3 + 1] +
                           metric[(i_shared + l) * 3 + 2] * metric[(i_shared + l) * 3 + 2]);
        real Uk = (metric[(i_shared + l) * 3] * Q[1] + metric[(i_shared + l) * 3 + 1] * Q[2] +
                   metric[(i_shared + l) * 3 + 2] * Q[3]) / Q[0];
        real ukPc = abs(Uk + c * grad_k);
        real ukMc = abs(Uk - c * grad_k);
        spec_rad[0] = max(spec_rad[0], ukMc);
        spec_rad[1] = max(spec_rad[1], abs(Uk));
        spec_rad[2] = max(spec_rad[2], ukPc);
      }
      spec_rad[0] = max(spec_rad[0], abs((Uk_bar - cm) * gradK));
      spec_rad[1] = max(spec_rad[1], abs(Uk_bar * gradK));
      spec_rad[2] = max(spec_rad[2], abs((Uk_bar + cm) * gradK));

      for (int l = 0; l < 5; ++l) {
        real lambda_l{spec_rad[1]};
        if (l == 0) {
          lambda_l = spec_rad[0];
        } else if (l == 4) {
          lambda_l = spec_rad[2];
        }

        real vPlus[7] = {}, vMinus[7] = {};
        for (int m = 0; m < 7; ++m) {
          for (int n = 0; n < 5; ++n) {
            vPlus[m] += LR(l, n) * (Fp[(i_shared - 2 + m) * 5 + n] + lambda_l * cv[(i_shared - 2 + m) * 7 + n] *
                                    jac[i_shared - 2 + m]);
            vMinus[m] += LR(l, n) * (Fm[(i_shared - 1 + m) * 5 + n] - lambda_l * cv[(i_shared - 1 + m) * 7 + n] *
                                     jac[i_shared - 1 + m]);
          }
          vPlus[m] *= 0.5;
          vMinus[m] *= 0.5;
        }
        fChar[l] = WENO7(vPlus, vMinus, eps_scaled);
      }
    }
  }

  // Compute the right characteristic matrix
  LR(0, 0) = 1.0;
  LR(0, 1) = kx;
  LR(0, 2) = ky;
  LR(0, 3) = kz;
  LR(0, 4) = 1.0;
  LR(1, 0) = um - kx * cm;
  LR(1, 1) = kx * um;
  LR(1, 2) = ky * um - kz * cm;
  LR(1, 3) = kz * um + ky * cm;
  LR(1, 4) = um + kx * cm;
  LR(2, 0) = vm - ky * cm;
  LR(2, 1) = kx * vm + kz * cm;
  LR(2, 2) = ky * vm;
  LR(2, 3) = kz * vm - kx * cm;
  LR(2, 4) = vm + ky * cm;
  LR(3, 0) = wm - kz * cm;
  LR(3, 1) = kx * wm - ky * cm;
  LR(3, 2) = ky * wm + kx * cm;
  LR(3, 3) = kz * wm;
  LR(3, 4) = wm + kz * cm;
  LR(4, 0) = hm - Uk_bar * cm;
  LR(4, 1) = kx * alpha / gm1 + (kz * vm - ky * wm) * cm;
  LR(4, 2) = ky * alpha / gm1 + (kx * wm - kz * um) * cm;
  LR(4, 3) = kz * alpha / gm1 + (ky * um - kx * vm) * cm;
  LR(4, 4) = hm + Uk_bar * cm;

  // Project the flux back to physical space
  auto fci = &fc[tid * n_var];
  for (int l = 0; l < 5; ++l) {
    real f{0};
    for (int m = 0; m < 5; ++m) {
      f += LR(l, m) * fChar[m];
    }
    fci[l] = f;
  }
}

__device__ void
compute_weno_flux_cp(const real *cv, DParameter *param, int tid, const real *metric, const real *jac, real *fc,
                     int i_shared, real *Fp, real *Fm, const int *ig_shared, int n_add, real *f_1st) {
  const int n_var = param->n_var;

  compute_flux(&cv[i_shared * (n_var + 2)], param, &metric[i_shared * 3], jac[i_shared], &Fp[i_shared * n_var],
               &Fm[i_shared * n_var]);
  for (size_t i = 0; i < n_add; i++) {
    compute_flux(&cv[ig_shared[i] * (n_var + 2)], param, &metric[ig_shared[i] * 3], jac[ig_shared[i]],
                 &Fp[ig_shared[i] * n_var], &Fm[ig_shared[i] * n_var]);
  }
  __syncthreads();

  //  const real eps_ref = 1e-6 * param->weno_eps_scale;
  constexpr real eps{1e-20};
  const real jac1{jac[i_shared]}, jac2{jac[i_shared + 1]};
  const real eps_ref = eps * param->weno_eps_scale * 0.25 *
                       ((metric[i_shared * 3] * jac1 + metric[(i_shared + 1) * 3] * jac2) *
                        (metric[i_shared * 3] * jac1 + metric[(i_shared + 1) * 3] * jac2) +
                        (metric[i_shared * 3 + 1] * jac1 + metric[(i_shared + 1) * 3 + 1] * jac2) *
                        (metric[i_shared * 3 + 1] * jac1 + metric[(i_shared + 1) * 3 + 1] * jac2) +
                        (metric[i_shared * 3 + 2] * jac1 + metric[(i_shared + 1) * 3 + 2] * jac2) *
                        (metric[i_shared * 3 + 2] * jac1 + metric[(i_shared + 1) * 3 + 2] * jac2));
  real eps_scaled[3];
  eps_scaled[0] = eps_ref;
  eps_scaled[1] = eps_ref * param->v_ref * param->v_ref;
  eps_scaled[2] = eps_scaled[1] * param->v_ref * param->v_ref;

  if (param->positive_preserving) {
    for (int l = 0; l < n_var - 5; ++l) {
      f_1st[tid * (n_var - 5) + l] = 0.5 * Fp[i_shared * n_var + l + 5] + 0.5 * Fm[(i_shared + 1) * n_var + l + 5];
    }
  }

  const auto fci = &fc[tid * n_var];

  for (int l = 0; l < n_var; ++l) {
    real eps_here{eps_scaled[0]};
    if (l == 1 || l == 2 || l == 3) {
      eps_here = eps_scaled[1];
    } else if (l == 4) {
      eps_here = eps_scaled[2];
    }

    if (param->inviscid_scheme == 51) {
      real vp[5], vm[5];
      vp[0] = Fp[(i_shared - 2) * n_var + l];
      vp[1] = Fp[(i_shared - 1) * n_var + l];
      vp[2] = Fp[i_shared * n_var + l];
      vp[3] = Fp[(i_shared + 1) * n_var + l];
      vp[4] = Fp[(i_shared + 2) * n_var + l];
      vm[0] = Fm[(i_shared - 1) * n_var + l];
      vm[1] = Fm[i_shared * n_var + l];
      vm[2] = Fm[(i_shared + 1) * n_var + l];
      vm[3] = Fm[(i_shared + 2) * n_var + l];
      vm[4] = Fm[(i_shared + 3) * n_var + l];

      fci[l] = WENO5(vp, vm, eps_here);
    } else if (param->inviscid_scheme == 71) {
      real vp[7], vm[7];
      vp[0] = Fp[(i_shared - 3) * n_var + l];
      vp[1] = Fp[(i_shared - 2) * n_var + l];
      vp[2] = Fp[(i_shared - 1) * n_var + l];
      vp[3] = Fp[i_shared * n_var + l];
      vp[4] = Fp[(i_shared + 1) * n_var + l];
      vp[5] = Fp[(i_shared + 2) * n_var + l];
      vp[6] = Fp[(i_shared + 3) * n_var + l];
      vm[0] = Fm[(i_shared - 2) * n_var + l];
      vm[1] = Fm[(i_shared - 1) * n_var + l];
      vm[2] = Fm[i_shared * n_var + l];
      vm[3] = Fm[(i_shared + 1) * n_var + l];
      vm[4] = Fm[(i_shared + 2) * n_var + l];
      vm[5] = Fm[(i_shared + 3) * n_var + l];
      vm[6] = Fm[(i_shared + 4) * n_var + l];

      fci[l] = WENO7(vp, vm, eps_here);
    }
  }
}

__device__ void
positive_preserving_limiter(const real *f_1st, int n_var, int tid, real *fc, const DParameter *param, int i_shared,
                            real dt, int idx_in_mesh, int max_extent, const real *cv, const real *jac) {
  const real alpha = param->dim == 3 ? 1.0 / 3.0 : 0.5;

  const int ns = n_var - 5;
  const int offset_yq_l = i_shared * (n_var + 2) + 5;
  const int offset_yq_r = (i_shared + 1) * (n_var + 2) + 5;
  real *fc_yq_i = &fc[tid * n_var + 5];

  for (int l = 0; l < ns; ++l) {
    real theta_p = 1.0, theta_m = 1.0;
    if (idx_in_mesh > -1) {
      const real up = 0.5 * alpha * cv[offset_yq_l + l] * jac[i_shared] - dt * fc_yq_i[l];
      if (up < 0) {
        const real up_lf = 0.5 * alpha * cv[offset_yq_l + l] * jac[i_shared] - dt * f_1st[tid * ns + l];
        if (abs(up - up_lf) > 1e-20) {
          theta_p = (0 - up_lf) / (up - up_lf);
          if (theta_p > 1)
            theta_p = 1.0;
          else if (theta_p < 0)
            theta_p = 0;
        }
      }
    }

    if (idx_in_mesh < max_extent - 1) {
      const real um = 0.5 * alpha * cv[offset_yq_r + l] * jac[i_shared + 1] + dt * fc_yq_i[l];
      if (um < 0) {
        const real um_lf = 0.5 * alpha * cv[offset_yq_r + l] * jac[i_shared + 1] + dt * f_1st[tid * ns + l];
        if (abs(um - um_lf) > 1e-20) {
          theta_m = (0 - um_lf) / (um - um_lf);
          if (theta_m > 1)
            theta_m = 1.0;
          else if (theta_m < 0)
            theta_m = 0;
        }
      }
    }

    fc_yq_i[l] = min(theta_p, theta_m) * (fc_yq_i[l] - f_1st[tid * ns + l]) + f_1st[tid * ns + l];
  }
}

__device__ real WENO5(const real *vp, const real *vm, real eps) {
  constexpr real one6th{1.0 / 6};
  real v0{one6th * (2 * vp[2] + 5 * vp[3] - vp[4])};
  real v1{one6th * (-vp[1] + 5 * vp[2] + 2 * vp[3])};
  real v2{one6th * (2 * vp[0] - 7 * vp[1] + 11 * vp[2])};
  constexpr real thirteen12th{13.0 / 12};
  real beta0 = thirteen12th * (vp[2] + vp[4] - 2 * vp[3]) * (vp[2] + vp[4] - 2 * vp[3]) +
               0.25 * (3 * vp[2] - 4 * vp[3] + vp[4]) * (3 * vp[2] - 4 * vp[3] + vp[4]);
  real beta1 = thirteen12th * (vp[1] + vp[3] - 2 * vp[2]) * (vp[1] + vp[3] - 2 * vp[2]) +
               0.25 * (vp[1] - vp[3]) * (vp[1] - vp[3]);
  real beta2 = thirteen12th * (vp[0] + vp[2] - 2 * vp[1]) * (vp[0] + vp[2] - 2 * vp[1]) +
               0.25 * (vp[0] - 4 * vp[1] + 3 * vp[2]) * (vp[0] - 4 * vp[1] + 3 * vp[2]);
  constexpr real three10th{0.3}, six10th{0.6}, one10th{0.1};
  real tau5sqr{(beta0 - beta2) * (beta0 - beta2)};
  real a0{three10th + three10th * tau5sqr / ((eps + beta0) * (eps + beta0))};
  real a1{six10th + six10th * tau5sqr / ((eps + beta1) * (eps + beta1))};
  real a2{one10th + one10th * tau5sqr / ((eps + beta2) * (eps + beta2))};
  const real fPlus{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

  v0 = one6th * (11 * vm[2] - 7 * vm[3] + 2 * vm[4]);
  v1 = one6th * (2 * vm[1] + 5 * vm[2] - vm[3]);
  v2 = one6th * (-vm[0] + 5 * vm[1] + 2 * vm[2]);
  beta0 = thirteen12th * (vm[2] + vm[4] - 2 * vm[3]) * (vm[2] + vm[4] - 2 * vm[3]) +
          0.25 * (3 * vm[2] - 4 * vm[3] + vm[4]) * (3 * vm[2] - 4 * vm[3] + vm[4]);
  beta1 = thirteen12th * (vm[1] + vm[3] - 2 * vm[2]) * (vm[1] + vm[3] - 2 * vm[2]) +
          0.25 * (vm[1] - vm[3]) * (vm[1] - vm[3]);
  beta2 = thirteen12th * (vm[0] + vm[2] - 2 * vm[1]) * (vm[0] + vm[2] - 2 * vm[1]) +
          0.25 * (vm[0] - 4 * vm[1] + 3 * vm[2]) * (vm[0] - 4 * vm[1] + 3 * vm[2]);
  tau5sqr = (beta0 - beta2) * (beta0 - beta2);
  a0 = one10th + one10th * tau5sqr / ((eps + beta0) * (eps + beta0));
  a1 = six10th + six10th * tau5sqr / ((eps + beta1) * (eps + beta1));
  a2 = three10th + three10th * tau5sqr / ((eps + beta2) * (eps + beta2));
  const real fMinus{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

  return fPlus + fMinus;
}

__device__ real WENO7(const real *vp, const real *vm, real eps) {
  constexpr real one6th{1.0 / 6};
  constexpr real d12{13.0 / 12.0}, d13{1043.0 / 960}, d14{1.0 / 12};

  // Re-organize the data to improve locality
  // 1st order derivative
  real s1{one6th * (-2 * vp[0] + 9 * vp[1] - 18 * vp[2] + 11 * vp[3])};
  // 2nd order derivative
  real s2{-vp[0] + 4 * vp[1] - 5 * vp[2] + 2 * vp[3]};
  // 3rd order derivative
  real s3{-vp[0] + 3 * vp[1] - 3 * vp[2] + vp[3]};
  real beta0{s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3};

  s1 = one6th * (vp[1] - 6 * vp[2] + 3 * vp[3] + 2 * vp[4]);
  s2 = vp[2] - 2 * vp[3] + vp[4];
  s3 = -vp[1] + 3 * vp[2] - 3 * vp[3] + vp[4];
  real beta1{s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3};

  s1 = one6th * (-2 * vp[2] - 3 * vp[3] + 6 * vp[4] - vp[5]);
  s3 = -vp[2] + 3 * vp[3] - 3 * vp[4] + vp[5];
  real beta2{s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3};

  s1 = one6th * (-11 * vp[3] + 18 * vp[4] - 9 * vp[5] + 2 * vp[6]);
  s2 = 2 * vp[3] - 5 * vp[4] + 4 * vp[5] - vp[6];
  s3 = -vp[3] + 3 * vp[4] - 3 * vp[5] + vp[6];
  real beta3{s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3};

  // real tau7sqr{(beta0 - beta3) * (beta0 - beta3)};
  constexpr real c0{1.0 / 35}, c1{12.0 / 35}, c2{18.0 / 35}, c3{4.0 / 35};
  // real a0{c0 + c0 * tau7sqr / ((eps + beta0) * (eps + beta0))};
  // real a1{c1 + c1 * tau7sqr / ((eps + beta1) * (eps + beta1))};
  // real a2{c2 + c2 * tau7sqr / ((eps + beta2) * (eps + beta2))};
  // real a3{c3 + c3 * tau7sqr / ((eps + beta3) * (eps + beta3))};
  real a0{c0 / ((eps + beta0) * (eps + beta0))};
  real a1{c1 / ((eps + beta1) * (eps + beta1))};
  real a2{c2 / ((eps + beta2) * (eps + beta2))};
  real a3{c3 / ((eps + beta3) * (eps + beta3))};

  constexpr real one12th{1.0 / 12};
  real v0{one12th * (-3 * vp[0] + 13 * vp[1] - 23 * vp[2] + 25 * vp[3])};
  real v1{one12th * (vp[1] - 5 * vp[2] + 13 * vp[3] + 3 * vp[4])};
  real v2{one12th * (-vp[2] + 7 * vp[3] + 7 * vp[4] - vp[5])};
  real v3{one12th * (3 * vp[3] + 13 * vp[4] - 5 * vp[5] + vp[6])};
  const real fPlus{(a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3) / (a0 + a1 + a2 + a3)};

  // Minus part
  s1 = one6th * (-2 * vm[6] + 9 * vm[5] - 18 * vm[4] + 11 * vm[3]);
  s2 = -vm[6] + 4 * vm[5] - 5 * vm[4] + 2 * vm[3];
  s3 = -vm[6] + 3 * vm[5] - 3 * vm[4] + vm[3];
  beta0 = s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3;

  s1 = one6th * (vm[5] - 6 * vm[4] + 3 * vm[3] + 2 * vm[2]);
  s2 = vm[4] - 2 * vm[3] + vm[2];
  s3 = -vm[5] + 3 * vm[4] - 3 * vm[3] + vm[2];
  beta1 = s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3;

  s1 = one6th * (-2 * vm[4] - 3 * vm[3] + 6 * vm[2] - vm[1]);
  s3 = -vm[4] + 3 * vm[3] - 3 * vm[2] + vm[1];
  beta2 = s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3;

  s1 = one6th * (-11 * vm[3] + 18 * vm[2] - 9 * vm[1] + 2 * vm[0]);
  s2 = 2 * vm[3] - 5 * vm[2] + 4 * vm[1] - vm[0];
  s3 = -vm[3] + 3 * vm[2] - 3 * vm[1] + vm[0];
  beta3 = s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3;

  // tau7sqr = (beta0 - beta3) * (beta0 - beta3);
  // a0 = c0 + c0 * tau7sqr / ((eps + beta0) * (eps + beta0));
  // a1 = c1 + c1 * tau7sqr / ((eps + beta1) * (eps + beta1));
  // a2 = c2 + c2 * tau7sqr / ((eps + beta2) * (eps + beta2));
  // a3 = c3 + c3 * tau7sqr / ((eps + beta3) * (eps + beta3));
  a0 = c0 / ((eps + beta0) * (eps + beta0));
  a1 = c1 / ((eps + beta1) * (eps + beta1));
  a2 = c2 / ((eps + beta2) * (eps + beta2));
  a3 = c3 / ((eps + beta3) * (eps + beta3));

  v0 = one12th * (-3 * vm[6] + 13 * vm[5] - 23 * vm[4] + 25 * vm[3]);
  v1 = one12th * (vm[5] - 5 * vm[4] + 13 * vm[3] + 3 * vm[2]);
  v2 = one12th * (-vm[4] + 7 * vm[3] + 7 * vm[2] - vm[1]);
  v3 = one12th * (3 * vm[3] + 13 * vm[2] - 5 * vm[1] + vm[0]);
  const real fMinus{(a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3) / (a0 + a1 + a2 + a3)};

  return fPlus + fMinus;
}

__device__ real WENO5_new(const real *vp, const real *vm, real eps) {
  constexpr real one6th{1.0 / 6};
  real v0{one6th * (2 * vp[2] + 5 * vp[3] - vp[4])};
  real v1{one6th * (-vp[1] + 5 * vp[2] + 2 * vp[3])};
  real v2{one6th * (2 * vp[0] - 7 * vp[1] + 11 * vp[2])};
  constexpr real thirteen12th{13.0 / 12};
  real beta0 = thirteen12th * (vp[2] + vp[4] - 2 * vp[3]) * (vp[2] + vp[4] - 2 * vp[3]) +
               0.25 * (3 * vp[2] - 4 * vp[3] + vp[4]) * (3 * vp[2] - 4 * vp[3] + vp[4]);
  real beta1 = thirteen12th * (vp[1] + vp[3] - 2 * vp[2]) * (vp[1] + vp[3] - 2 * vp[2]) +
               0.25 * (vp[1] - vp[3]) * (vp[1] - vp[3]);
  real beta2 = thirteen12th * (vp[0] + vp[2] - 2 * vp[1]) * (vp[0] + vp[2] - 2 * vp[1]) +
               0.25 * (vp[0] - 4 * vp[1] + 3 * vp[2]) * (vp[0] - 4 * vp[1] + 3 * vp[2]);
  constexpr real three10th{0.3}, six10th{0.6}, one10th{0.1};
  real tau5sqr{(beta0 - beta2) * (beta0 - beta2)};
  real a0{three10th + three10th * tau5sqr / ((eps + beta0) * (eps + beta0))};
  real a1{six10th + six10th * tau5sqr / ((eps + beta1) * (eps + beta1))};
  real a2{one10th + one10th * tau5sqr / ((eps + beta2) * (eps + beta2))};
  const real fPlus{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

  v0 = one6th * (11 * vm[3] - 7 * vm[4] + 2 * vm[5]);
  v1 = one6th * (2 * vm[3] + 5 * vm[3] - vm[4]);
  v2 = one6th * (-vm[1] + 5 * vm[2] + 2 * vm[3]);
  beta0 = thirteen12th * (vm[3] + vm[5] - 2 * vm[4]) * (vm[3] + vm[5] - 2 * vm[4]) +
          0.25 * (3 * vm[3] - 4 * vm[4] + vm[5]) * (3 * vm[3] - 4 * vm[4] + vm[5]);
  beta1 = thirteen12th * (vm[2] + vm[4] - 2 * vm[3]) * (vm[2] + vm[4] - 2 * vm[3]) +
          0.25 * (vm[2] - vm[4]) * (vm[2] - vm[4]);
  beta2 = thirteen12th * (vm[1] + vm[3] - 2 * vm[2]) * (vm[1] + vm[3] - 2 * vm[2]) +
          0.25 * (vm[1] - 4 * vm[2] + 3 * vm[3]) * (vm[1] - 4 * vm[2] + 3 * vm[3]);
  tau5sqr = (beta0 - beta2) * (beta0 - beta2);
  a0 = one10th + one10th * tau5sqr / ((eps + beta0) * (eps + beta0));
  a1 = six10th + six10th * tau5sqr / ((eps + beta1) * (eps + beta1));
  a2 = three10th + three10th * tau5sqr / ((eps + beta2) * (eps + beta2));
  const real fMinus{(a0 * v0 + a1 * v1 + a2 * v2) / (a0 + a1 + a2)};

  return fPlus + fMinus;
}

__device__ real WENO7_new(const real *vp, const real *vm, real eps) {
  constexpr real one6th{1.0 / 6};
  constexpr real d12{13.0 / 12.0}, d13{1043.0 / 960}, d14{1.0 / 12};

  // Re-organize the data to improve locality
  // 1st order derivative
  real s1{one6th * (-2 * vp[0] + 9 * vp[1] - 18 * vp[2] + 11 * vp[3])};
  // 2nd order derivative
  real s2{-vp[0] + 4 * vp[1] - 5 * vp[2] + 2 * vp[3]};
  // 3rd order derivative
  real s3{-vp[0] + 3 * vp[1] - 3 * vp[2] + vp[3]};
  real beta0{s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3};

  s1 = one6th * (vp[1] - 6 * vp[2] + 3 * vp[3] + 2 * vp[4]);
  s2 = vp[2] - 2 * vp[3] + vp[4];
  s3 = -vp[1] + 3 * vp[2] - 3 * vp[3] + vp[4];
  real beta1{s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3};

  s1 = one6th * (-2 * vp[2] - 3 * vp[3] + 6 * vp[4] - vp[5]);
  s3 = -vp[2] + 3 * vp[3] - 3 * vp[4] + vp[5];
  real beta2{s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3};

  s1 = one6th * (-11 * vp[3] + 18 * vp[4] - 9 * vp[5] + 2 * vp[6]);
  s2 = 2 * vp[3] - 5 * vp[4] + 4 * vp[5] - vp[6];
  s3 = -vp[3] + 3 * vp[4] - 3 * vp[5] + vp[6];
  real beta3{s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3};

  real tau7sqr{(beta0 - beta3) * (beta0 - beta3)};
  constexpr real c0{1.0 / 35}, c1{12.0 / 35}, c2{18.0 / 35}, c3{4.0 / 35};
  real a0{c0 + c0 * tau7sqr / ((eps + beta0) * (eps + beta0))};
  real a1{c1 + c1 * tau7sqr / ((eps + beta1) * (eps + beta1))};
  real a2{c2 + c2 * tau7sqr / ((eps + beta2) * (eps + beta2))};
  real a3{c3 + c3 * tau7sqr / ((eps + beta3) * (eps + beta3))};

  constexpr real one12th{1.0 / 12};
  real v0{one12th * (-3 * vp[0] + 13 * vp[1] - 23 * vp[2] + 25 * vp[3])};
  real v1{one12th * (vp[1] - 5 * vp[2] + 13 * vp[3] + 3 * vp[4])};
  real v2{one12th * (-vp[2] + 7 * vp[3] + 7 * vp[4] - vp[5])};
  real v3{one12th * (3 * vp[3] + 13 * vp[4] - 5 * vp[5] + vp[6])};
  const real fPlus{(a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3) / (a0 + a1 + a2 + a3)};

  // Minus part
  s1 = one6th * (-2 * vm[7] + 9 * vm[6] - 18 * vm[5] + 11 * vm[4]);
  s2 = -vm[7] + 4 * vm[6] - 5 * vm[5] + 2 * vm[4];
  s3 = -vm[7] + 3 * vm[6] - 3 * vm[5] + vm[4];
  beta0 = s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3;

  s1 = one6th * (vm[6] - 6 * vm[5] + 3 * vm[4] + 2 * vm[3]);
  s2 = vm[5] - 2 * vm[4] + vm[3];
  s3 = -vm[6] + 3 * vm[5] - 3 * vm[4] + vm[3];
  beta1 = s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3;

  s1 = one6th * (-2 * vm[5] - 3 * vm[4] + 6 * vm[3] - vm[2]);
  s3 = -vm[5] + 3 * vm[4] - 3 * vm[3] + vm[2];
  beta2 = s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3;

  s1 = one6th * (-11 * vm[4] + 18 * vm[3] - 9 * vm[2] + 2 * vm[1]);
  s2 = 2 * vm[4] - 5 * vm[3] + 4 * vm[2] - vm[1];
  s3 = -vm[4] + 3 * vm[3] - 3 * vm[2] + vm[1];
  beta3 = s1 * s1 + d12 * s2 * s2 + d13 * s3 * s3 + d14 * s1 * s3;

  tau7sqr = (beta0 - beta3) * (beta0 - beta3);
  a0 = c0 + c0 * tau7sqr / ((eps + beta0) * (eps + beta0));
  a1 = c1 + c1 * tau7sqr / ((eps + beta1) * (eps + beta1));
  a2 = c2 + c2 * tau7sqr / ((eps + beta2) * (eps + beta2));
  a3 = c3 + c3 * tau7sqr / ((eps + beta3) * (eps + beta3));

  v0 = one12th * (-3 * vm[7] + 13 * vm[6] - 23 * vm[5] + 25 * vm[4]);
  v1 = one12th * (vm[6] - 5 * vm[5] + 13 * vm[4] + 3 * vm[3]);
  v2 = one12th * (-vm[5] + 7 * vm[4] + 7 * vm[3] - vm[2]);
  v3 = one12th * (3 * vm[4] + 13 * vm[3] - 5 * vm[2] + vm[1]);
  const real fMinus{(a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3) / (a0 + a1 + a2 + a3)};

  return fPlus + fMinus;
}

template void
compute_convective_term_weno<MixtureModel::Air>(const Block &block, DZone *zone, DParameter *param, int n_var,
                                                const Parameter &parameter);

template void
compute_convective_term_weno<MixtureModel::Mixture>(const Block &block, DZone *zone, DParameter *param,
                                                    int n_var, const Parameter &parameter);
}
