#include "InviscidScheme.cuh"
#include "Mesh.h"
#include "Parameter.h"
#include "AWENO.cuh"
#include "Field.h"
#include "DParameter.cuh"
#include "Reconstruction.cuh"
#include "RiemannSolver.cuh"
#include "Parallel.h"

namespace cfd {
template<MixtureModel mix_model>
void compute_convective_term_pv(const Block &block, DZone *zone, DParameter *param, int n_var,
  const Parameter &parameter) {
  const int extent[3]{block.mx, block.my, block.mz};
  constexpr int block_dim = 64;
  const int n_computation_per_block = block_dim + 2 * block.ngg - 1;
  auto shared_mem = (block_dim * n_var                                            // fc
                     + n_computation_per_block * (n_var + 3 + 1)) * sizeof(real); // pv[n_var]+metric[3]+jacobian

  for (auto dir = 0; dir < 2; ++dir) {
    int tpb[3]{1, 1, 1};
    tpb[dir] = block_dim;
    int bpg[3]{extent[0], extent[1], extent[2]};
    bpg[dir] = (extent[dir] - 1) / (tpb[dir] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    compute_convective_term_pv_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, dir, extent[dir], param);
  }

  if (extent[2] > 1) {
    // 3D computation
    // Number of threads in the 3rd direction cannot exceed 64
    int tpb[3]{1, 1, 1};
    tpb[2] = 64;
    int bpg[3]{extent[0], extent[1], extent[2]};
    bpg[2] = (extent[2] - 1) / (tpb[2] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    compute_convective_term_pv_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, 2, extent[2], param);
  }
}

template<MixtureModel mix_model>
__global__ void
compute_convective_term_pv_1D(DZone *zone, int direction, int max_extent, DParameter *param) {
  int labels[3]{0, 0, 0};
  labels[direction] = 1;
  const int tid = static_cast<int>(threadIdx.x * labels[0] + threadIdx.y * labels[1] + threadIdx.z * labels[2]);
  const int block_dim = static_cast<int>(blockDim.x * blockDim.y * blockDim.z);
  const auto ngg{zone->ngg};
  const int n_point = block_dim + 2 * ngg - 1;

  int idx[3];
  idx[0] = static_cast<int>((blockDim.x - labels[0]) * blockIdx.x + threadIdx.x);
  idx[1] = static_cast<int>((blockDim.y - labels[1]) * blockIdx.y + threadIdx.y);
  idx[2] = static_cast<int>((blockDim.z - labels[2]) * blockIdx.z + threadIdx.z);
  idx[direction] -= 1;
  if (idx[direction] >= max_extent) return;

  // load variables to shared memory
  extern __shared__ real s[];
  const auto n_var{param->n_var};
  auto n_reconstruct{n_var};
  real *pv = s;
  real *metric = &pv[n_point * n_reconstruct];
  real *jac = &metric[n_point * 3];
  real *fc = &jac[n_point];
  memset(&fc[tid * n_var], 0, n_var * sizeof(real));

  const int i_shared = tid - 1 + ngg;
  for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
    pv[i_shared * n_reconstruct + l] = zone->bv(idx[0], idx[1], idx[2], l);
  }
  const auto n_scalar{param->n_scalar};
  for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
    pv[i_shared * n_reconstruct + 5 + l] = zone->sv(idx[0], idx[1], idx[2], l);
  }
  for (auto l = 1; l < 4; ++l) {
    metric[i_shared * 3 + l - 1] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, l);
  }
  jac[i_shared] = zone->jac(idx[0], idx[1], idx[2]);

  // ghost cells
  if (tid == 0) {
    // Responsible for the left (ngg-1) points
    for (auto i = 1; i < ngg; ++i) {
      const auto ig_shared = ngg - 1 - i;
      const int g_idx[3]{idx[0] - i * labels[0], idx[1] - i * labels[1], idx[2] - i * labels[2]};

      for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
        pv[ig_shared * n_reconstruct + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, Z, Z_prime...
        pv[ig_shared * n_reconstruct + 5 + l] = zone->sv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  if (tid == block_dim - 1 || idx[direction] == max_extent - 1) {
    // Responsible for the right ngg points
    for (auto i = 1; i <= ngg; ++i) {
      const auto ig_shared = tid + i + ngg - 1;
      const int g_idx[3]{idx[0] + i * labels[0], idx[1] + i * labels[1], idx[2] + i * labels[2]};

      for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
        pv[ig_shared * n_reconstruct + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, Z, Z_prime...
        pv[ig_shared * n_reconstruct + 5 + l] = zone->sv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  __syncthreads();

  // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
  constexpr int n_reconstruction_max =
      7 + MAX_SPEC_NUMBER + 4 + 2; // rho,u,v,w,p,Y_{1...Ns},(k,omega,z,z_prime),E,gamma
  real pv_l[n_reconstruction_max], pv_r[n_reconstruction_max];
  reconstruction<mix_model>(pv, pv_l, pv_r, i_shared, param);
  __syncthreads();

  // compute the half-point numerical flux with the chosen Riemann solver
  switch (param->inviscid_scheme) {
    case 1:
      riemannSolver_laxFriedrich<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    case 3: // AUSM+
      riemannSolver_ausmPlus<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    case 4: // HLLC
      riemannSolver_hllc<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    default:
      riemannSolver_ausmPlus<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
  }
  __syncthreads();

  if (tid > 0) {
    for (int l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model>
__device__ void
reconstruction(real *pv, real *pv_l, real *pv_r, const int idx_shared, DParameter *param) {
  auto n_var = param->n_var;
  switch (param->reconstruction) {
    case 2:
      MUSCL_reconstruct(pv, pv_l, pv_r, idx_shared, n_var, param->limiter);
      break;
    case 3:
      NND2_reconstruct(pv, pv_l, pv_r, idx_shared, n_var, param->limiter);
      break;
    default:
      first_order_reconstruct(pv, pv_l, pv_r, idx_shared, n_var);
  }
  if constexpr (mix_model != MixtureModel::Air) {
    real el = 0.5 * (pv_l[1] * pv_l[1] + pv_l[2] * pv_l[2] + pv_l[3] * pv_l[3]);
    real er = 0.5 * (pv_r[1] * pv_r[1] + pv_r[2] * pv_r[2] + pv_r[3] * pv_r[3]);
    const auto n_spec = param->n_spec;
    real mw_inv_l{0.0}, mw_inv_r{0.0};
    for (int l = 0; l < n_spec; ++l) {
      mw_inv_l += pv_l[5 + l] / param->mw[l];
      mw_inv_r += pv_r[5 + l] / param->mw[l];
    }
    const real t_l = pv_l[4] / (pv_l[0] * R_u * mw_inv_l);
    const real t_r = pv_r[4] / (pv_r[0] * R_u * mw_inv_r);

    real hl[MAX_SPEC_NUMBER], hr[MAX_SPEC_NUMBER], cpl_i[MAX_SPEC_NUMBER], cpr_i[MAX_SPEC_NUMBER];
    compute_enthalpy_and_cp(t_l, hl, cpl_i, param);
    compute_enthalpy_and_cp(t_r, hr, cpr_i, param);
    real cpl{0}, cpr{0}, cvl{0}, cvr{0};
    for (auto l = 0; l < n_spec; ++l) {
      cpl += cpl_i[l] * pv_l[l + 5];
      cpr += cpr_i[l] * pv_r[l + 5];
      cvl += pv_l[l + 5] * (cpl_i[l] - R_u / param->mw[l]);
      cvr += pv_r[l + 5] * (cpr_i[l] - R_u / param->mw[l]);
      el += hl[l] * pv_l[l + 5];
      er += hr[l] * pv_r[l + 5];
    }
    pv_l[n_var] = pv_l[0] * el - pv_l[4]; //total energy
    pv_r[n_var] = pv_r[0] * er - pv_r[4];

    pv_l[n_var + 1] = cpl / cvl; //specific heat ratio
    pv_r[n_var + 1] = cpr / cvr;
  } else {
    const real el = 0.5 * (pv_l[1] * pv_l[1] + pv_l[2] * pv_l[2] + pv_l[3] * pv_l[3]);
    const real er = 0.5 * (pv_r[1] * pv_r[1] + pv_r[2] * pv_r[2] + pv_r[3] * pv_r[3]);
    pv_l[n_var] = el * pv_l[0] + pv_l[4] / (gamma_air - 1);
    pv_r[n_var] = er * pv_r[0] + pv_r[4] / (gamma_air - 1);
  }
}

template<MixtureModel mix_model>
void compute_convective_term_aweno(const Block &block, DZone *zone, DParameter *param, int n_var) {
  // The implementation of AWENO is based on Fig.9 of (Ye, C-C, Zhang, P-J-Y, Wan, Z-H, and Sun, D-J (2022)
  // An alternative formulation of targeted ENO scheme for hyperbolic conservation laws. Computers & Fluids, 238, 105368.
  // doi:10.1016/j.compfluid.2022.105368.)

  const int extent[3]{block.mx, block.my, block.mz};

  constexpr int block_dim = 64;
  const int n_computation_per_block = block_dim + 2 * block.ngg - 1;
  auto shared_mem = (block_dim * n_var                                       // fc
                     + n_computation_per_block * (n_var + 3)) * sizeof(real) // cv[n_var]+p+T+jacobian
                    + n_computation_per_block * 3 * sizeof(real);            // metric[3]
  auto shared_cds = block_dim * n_var * sizeof(real);                        // f_i

  for (auto dir = 0; dir < 2; ++dir) {
    int tpb[3]{1, 1, 1};
    tpb[dir] = block_dim;
    int bpg[3]{extent[0], extent[1], extent[2]};
    bpg[dir] = (extent[dir] - 1) / (tpb[dir] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    compute_convective_term_aweno_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, dir, extent[dir], param);

    bpg[dir] = (extent[dir] - 1) / (tpb[dir] - 2 * block.ngg) + 1;

    dim3 BPG2(bpg[0], bpg[1], bpg[2]);
    CDSPart1D<mix_model><<<BPG2, TPB, shared_cds>>>(zone, dir, extent[dir], param);
  }

  if (extent[2] > 1) {
    // 3D computation
    // Number of threads in the 3rd direction cannot exceed 64
    constexpr int tpb[3]{1, 1, 64};
    const int bpg[3]{extent[0], extent[1], (extent[2] - 1) / (tpb[2] - 1) + 1};

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    compute_convective_term_aweno_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, 2, extent[2], param);

    dim3 BPG2(extent[0], extent[1], (extent[2] - 1) / (tpb[2] - 2 * block.ngg) + 1);
    CDSPart1D<mix_model><<<BPG2, TPB, shared_cds>>>(zone, 2, extent[2], param);
  }
}

template<MixtureModel mix_model>
__global__ void
compute_convective_term_aweno_1D(DZone *zone, int direction, int max_extent, DParameter *param) {
  int labels[3]{0, 0, 0};
  labels[direction] = 1;
  const auto tid = static_cast<int>(threadIdx.x * labels[0] + threadIdx.y * labels[1] + threadIdx.z * labels[2]);
  const auto block_dim = static_cast<int>(blockDim.x * blockDim.y * blockDim.z);
  const auto ngg{zone->ngg};
  const int n_point = block_dim + 2 * ngg - 1;

  int idx[3];
  idx[0] = static_cast<int>((blockDim.x - labels[0]) * blockIdx.x + threadIdx.x);
  idx[1] = static_cast<int>((blockDim.y - labels[1]) * blockIdx.y + threadIdx.y);
  idx[2] = static_cast<int>((blockDim.z - labels[2]) * blockIdx.z + threadIdx.z);
  idx[direction] -= 1;
  if (idx[direction] >= max_extent) return;

  // load variables to shared memory
  extern __shared__ real s[];
  const auto n_var{param->n_var};
  const auto n_reconstruct{n_var + 2};
  real *cv = s;
  real *metric = &cv[n_point * n_reconstruct];
  real *jac = &metric[n_point * 3];
  real *fc = &jac[n_point];


  const int i_shared = tid - 1 + ngg;
  for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
    cv[i_shared * n_reconstruct + l] = zone->cv(idx[0], idx[1], idx[2], l);
  }
  cv[i_shared * n_reconstruct + n_var] = zone->bv(idx[0], idx[1], idx[2], 4);
  cv[i_shared * n_reconstruct + n_var + 1] = zone->bv(idx[0], idx[1], idx[2], 5);
  for (auto l = 1; l < 4; ++l) {
    metric[i_shared * 3 + l - 1] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, l);
  }
  jac[i_shared] = zone->jac(idx[0], idx[1], idx[2]);

  // ghost cells
  if (tid == 0) {
    // Responsible for the left (ngg-1) points
    for (auto i = 1; i < ngg; ++i) {
      const auto ig_shared = ngg - 1 - i;
      const int g_idx[3]{idx[0] - i * labels[0], idx[1] - i * labels[1], idx[2] - i * labels[2]};

      for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
        cv[ig_shared * n_reconstruct + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      cv[ig_shared * n_reconstruct + n_var] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
      cv[ig_shared * n_reconstruct + n_var + 1] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 5);
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  if (tid == block_dim - 1 || idx[direction] == max_extent - 1) {
    // Responsible for the right ngg points
    for (auto i = 1; i <= ngg; ++i) {
      const auto ig_shared = tid + i + ngg - 1;
      const int g_idx[3]{idx[0] + i * labels[0], idx[1] + i * labels[1], idx[2] + i * labels[2]};

      for (auto l = 0; l < n_var; ++l) { // 0-rho,1-rho*u,2-rho*v,3-rho*w,4-rho*E, ..., Nv-rho*scalar
        cv[ig_shared * n_reconstruct + l] = zone->cv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      cv[ig_shared * n_reconstruct + n_var] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 4);
      cv[ig_shared * n_reconstruct + n_var + 1] = zone->bv(g_idx[0], g_idx[1], g_idx[2], 5);
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  __syncthreads();

  // reconstruct the half-point left/right primitive variables with the chosen reconstruction method.
  constexpr int n_reconstruction_max =
      7 + MAX_SPEC_NUMBER + 4 + 2; // rho,u,v,w,p,Y_{1...Ns},(k,omega,z,z_prime),E,gamma
  real pv_l[n_reconstruction_max], pv_r[n_reconstruction_max];
  AWENO_interpolation<mix_model>(cv, pv_l, pv_r, i_shared, n_var, metric, param);
  __syncthreads();

  // compute the half-point numerical flux with the chosen Riemann solver
  switch (param->inviscid_scheme) {
    case 1:
      riemannSolver_laxFriedrich<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    case 3: // AUSM+
      riemannSolver_ausmPlus<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    case 4: // HLLC
      riemannSolver_hllc<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
    default:
      riemannSolver_ausmPlus<mix_model>(pv_l, pv_r, param, tid, metric, jac, fc, i_shared);
      break;
  }
  __syncthreads();

  if (tid > 0) {
    for (int l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model>
void Roe_compute_inviscid_flux(const Block &block, DZone *zone, DParameter *param, int n_var,
  const Parameter &parameter) {
  const int extent[3]{block.mx, block.my, block.mz};

  // Compute the entropy fix delta
  dim3 thread_per_block{8, 8, 4};
  if (extent[2] == 1) {
    thread_per_block = {16, 16, 1};
  }
  dim3 block_per_grid{
    (extent[0] + 1) / thread_per_block.x + 1,
    (extent[1] + 1) / thread_per_block.y + 1,
    (extent[2] + 1) / thread_per_block.z + 1
  };
  compute_entropy_fix_delta<mix_model><<<block_per_grid, thread_per_block>>>(zone, param);

  constexpr int block_dim = 128;
  const int n_computation_per_block = block_dim + 2 * block.ngg - 1;
  auto shared_mem = (block_dim * n_var                                       // fc
                     + n_computation_per_block * (n_var + 1)) * sizeof(real) // pv[n_var]+jacobian
                    + n_computation_per_block * 3 * sizeof(real)             // metric[3]
                    + n_computation_per_block * sizeof(real);                // entropy fix delta

  for (auto dir = 0; dir < 2; ++dir) {
    int tpb[3]{1, 1, 1};
    tpb[dir] = block_dim;
    int bpg[3]{extent[0], extent[1], extent[2]};
    bpg[dir] = (extent[dir] - 1) / (tpb[dir] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    Roe_compute_inviscid_flux_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, dir, extent[dir], param);
  }

  if (extent[2] > 1) {
    // 3D computation
    // Number of threads in the 3rd direction cannot exceed 64
    int tpb[3]{1, 1, 1};
    tpb[2] = 64;
    int bpg[3]{extent[0], extent[1], extent[2]};
    bpg[2] = (extent[2] - 1) / (tpb[2] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    Roe_compute_inviscid_flux_1D<mix_model><<<BPG, TPB, shared_mem>>>(zone, 2, extent[2], param);
  }
}

template<MixtureModel mix_model>
__global__ void compute_entropy_fix_delta(DZone *zone, DParameter *param) {
  const int mx{zone->mx}, my{zone->my}, mz{zone->mz};
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) - 1;
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y) - 1;
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z) - 1;
  if (i >= mx + 1 || j >= my + 1 || k >= mz + 1) return;

  const auto &bv{zone->bv};
  const auto &metric{zone->metric(i, j, k)};

  const real U = abs(bv(i, j, k, 1) * metric(1, 1) + bv(i, j, k, 2) * metric(1, 2) + bv(i, j, k, 3) * metric(1, 3));
  const real V = abs(bv(i, j, k, 1) * metric(2, 1) + bv(i, j, k, 2) * metric(2, 2) + bv(i, j, k, 3) * metric(2, 3));
  const real W = abs(bv(i, j, k, 1) * metric(3, 1) + bv(i, j, k, 2) * metric(3, 2) + bv(i, j, k, 3) * metric(3, 3));

  const real kx = sqrt(metric(1, 1) * metric(1, 1) + metric(1, 2) * metric(1, 2) + metric(1, 3) * metric(1, 3));
  const real ky = sqrt(metric(2, 1) * metric(2, 1) + metric(2, 2) * metric(2, 2) + metric(2, 3) * metric(2, 3));
  const real kz = sqrt(metric(3, 1) * metric(3, 1) + metric(3, 2) * metric(3, 2) + metric(3, 3) * metric(3, 3));

  real acoustic_speed{0};
  if constexpr (mix_model != MixtureModel::Air) {
    acoustic_speed = zone->acoustic_speed(i, j, k);
  } else {
    acoustic_speed = sqrt(gamma_air * bv(i, j, k, 4) / bv(i, j, k, 0));
  }
  if (param->dim == 2) {
    zone->entropy_fix_delta(i, j, k) =
        param->entropy_fix_factor * (U + V + acoustic_speed * 0.5 * (kx + ky));
  } else {
    // 3D
    zone->entropy_fix_delta(i, j, k) =
        param->entropy_fix_factor * (U + V + W + acoustic_speed * (kx + ky + kz) / 3.0);
  }
}

template<MixtureModel mix_model>
__global__ void
Roe_compute_inviscid_flux_1D(DZone *zone, int direction, int max_extent, DParameter *param) {
  int labels[3]{0, 0, 0};
  labels[direction] = 1;
  const auto tid = static_cast<int>(threadIdx.x * labels[0] + threadIdx.y * labels[1] + threadIdx.z * labels[2]);
  const auto block_dim = static_cast<int>(blockDim.x * blockDim.y * blockDim.z);
  const auto ngg{zone->ngg};
  const int n_point = block_dim + 2 * ngg - 1;

  int idx[3];
  idx[0] = static_cast<int>((blockDim.x - labels[0]) * blockIdx.x + threadIdx.x);
  idx[1] = static_cast<int>((blockDim.y - labels[1]) * blockIdx.y + threadIdx.y);
  idx[2] = static_cast<int>((blockDim.z - labels[2]) * blockIdx.z + threadIdx.z);
  idx[direction] -= 1;
  if (idx[direction] >= max_extent) return;

  // load variables to shared memory
  extern __shared__ real s[];
  const auto n_var{param->n_var};
  auto n_reconstruct{n_var};
  real *pv = s;
  real *metric = &pv[n_point * n_reconstruct];
  real *jac = &metric[n_point * 3];
  real *entropy_fix_delta = &jac[n_point];
  real *fc = &entropy_fix_delta[n_point];

  const int i_shared = tid - 1 + ngg;
  for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
    pv[i_shared * n_reconstruct + l] = zone->bv(idx[0], idx[1], idx[2], l);
  }
  const auto n_scalar{param->n_scalar};
  for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, z, zPrime
    pv[i_shared * n_reconstruct + 5 + l] = zone->sv(idx[0], idx[1], idx[2], l);
  }
  for (auto l = 1; l < 4; ++l) {
    metric[i_shared * 3 + l - 1] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, l);
  }
  jac[i_shared] = zone->jac(idx[0], idx[1], idx[2]);
  entropy_fix_delta[i_shared] = zone->entropy_fix_delta(idx[0], idx[1], idx[2]);

  // ghost cells
  if (tid == 0) {
    // Responsible for the left (ngg-1) points
    for (auto i = 1; i < ngg; ++i) {
      const auto ig_shared = ngg - 1 - i;
      const int g_idx[3]{idx[0] - i * labels[0], idx[1] - i * labels[1], idx[2] - i * labels[2]};

      for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
        pv[ig_shared * n_reconstruct + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, Z, Z_prime...
        pv[ig_shared * n_reconstruct + 5 + l] = zone->sv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  if (tid == block_dim - 1 || idx[direction] == max_extent - 1) {
    entropy_fix_delta[tid + ngg] = zone->entropy_fix_delta(idx[0] + labels[0], idx[1] + labels[1], idx[2] + labels[2]);
    // Responsible for the right ngg points
    for (auto i = 1; i <= ngg; ++i) {
      const auto ig_shared = tid + i + ngg - 1;
      const int g_idx[3]{idx[0] + i * labels[0], idx[1] + i * labels[1], idx[2] + i * labels[2]};

      for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
        pv[ig_shared * n_reconstruct + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, Z, Z_prime...
        pv[ig_shared * n_reconstruct + 5 + l] = zone->sv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  __syncthreads();

  riemannSolver_Roe<mix_model>(zone, pv, tid, param, fc, metric, jac, entropy_fix_delta);
  __syncthreads();


  if (tid > 0) {
    for (int l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

void compute_convective_term_ep(const Block &block, DZone *zone, DParameter *param, int n_var) {
  // Implementation of the 6th-order EP scheme.
  // PIROZZOLI S. Stabilized non-dissipative approximations of Euler equations in generalized curvilinear coordinates[J/OL].
  // Journal of Computational Physics, 2011, 230(8): 2997-3014. DOI:10.1016/j.jcp.2011.01.001.

  const int extent[3]{block.mx, block.my, block.mz};

  constexpr int block_dim = 64;
  const int n_computation_per_block = block_dim + 2 * block.ngg;
  auto shared_mem = (block_dim * n_var                 // fc
                     + n_computation_per_block * n_var // pv[5]+sv[n_scalar]
                     + n_computation_per_block * 6     // metric[3]+jacobian+rhoE+Uk
                    ) * sizeof(real);

  for (int dir = 0; dir < 2; ++dir) {
    int tpb[3]{1, 1, 1};
    tpb[dir] = block_dim;
    int bpg[3]{extent[0], extent[1], extent[2]};
    bpg[dir] = (extent[dir] - 1) / tpb[dir] + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    compute_convective_term_ep_1D<<<BPG, TPB, shared_mem>>>(zone, dir, extent[dir], param);
  }

  if (extent[2] > 1) {
    // 3D computation
    // Number of threads in the 3rd direction cannot exceed 64
    constexpr int tpb[3]{1, 1, 64};
    const int bpg[3]{extent[0], extent[1], (extent[2] - 1) / tpb[2] + 1};

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    compute_convective_term_ep_1D<<<BPG, TPB, shared_mem>>>(zone, 2, extent[2], param);
  }
}

__global__ void compute_convective_term_ep_1D(DZone *zone, int direction, int max_extent, DParameter *param) {
  int labels[3]{0, 0, 0};
  labels[direction] = 1;
  const int tid = static_cast<int>(threadIdx.x * labels[0] + threadIdx.y * labels[1] + threadIdx.z * labels[2]);
  const int block_dim = static_cast<int>(blockDim.x * blockDim.y * blockDim.z);
  const auto ngg{zone->ngg};
  const int n_point = block_dim + 2 * ngg;

  int idx[3];
  idx[0] = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  idx[1] = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  idx[2] = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (idx[direction] >= max_extent) return;

  // auto &pv = zone->bv;
  const auto n_var{param->n_var};
  const auto n_scalar{param->n_scalar};

  extern __shared__ real s[];
  real *metric = s;
  real *jac = &metric[n_point * 3];
  // pv: 0-rho,1-u,2-v,3-w,4-p, 5-n_var-1: scalar
  real *pv = &jac[n_point];
  real *rhoE = &pv[n_point * n_var];
  real *uk = &rhoE[n_point];
  real *fc = &uk[n_point];
  const int is = tid + ngg;

  // compute the contra-variant velocity
  metric[is * 3] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, 1);
  metric[is * 3 + 1] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, 2);
  metric[is * 3 + 2] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, 3);
  jac[is] = zone->jac(idx[0], idx[1], idx[2]);
  for (int l = 0; l < 5; ++l) {
    pv[is * n_var + l] = zone->bv(idx[0], idx[1], idx[2], l);
  }
  for (int l = 0; l < n_scalar; ++l) {
    pv[is * n_var + 5 + l] = zone->sv(idx[0], idx[1], idx[2], l);
  }
  uk[is] = metric[is * 3] * pv[is * n_var + 1] +
           metric[is * 3 + 1] * pv[is * n_var + 2] +
           metric[is * 3 + 2] * pv[is * n_var + 3];
  rhoE[is] = zone->cv(idx[0], idx[1], idx[2], 4);
  // ghost cells
  if (tid < ngg) {
    int g_idx[3];
    g_idx[0] = idx[0] - ngg * labels[0];
    g_idx[1] = idx[1] - ngg * labels[1];
    g_idx[2] = idx[2] - ngg * labels[2];

    metric[tid * 3] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 1);
    metric[tid * 3 + 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 2);
    metric[tid * 3 + 2] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 3);
    jac[tid] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    for (int l = 0; l < 5; ++l) {
      pv[tid * n_var + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
    }
    for (int l = 0; l < n_scalar; ++l) {
      pv[tid * n_var + 5 + l] = zone->sv(g_idx[0], g_idx[1], g_idx[2], l);
    }
    uk[tid] = metric[tid * 3] * pv[tid * n_var + 1] +
              metric[tid * 3 + 1] * pv[tid * n_var + 2] +
              metric[tid * 3 + 2] * pv[tid * n_var + 3];
    rhoE[tid] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 4);
  }
  if (tid > block_dim - ngg - 1 || idx[direction] > max_extent - ngg - 1) {
    const int iSh = tid + 2 * ngg;
    int g_idx[3];
    g_idx[0] = idx[0] + ngg * labels[0];
    g_idx[1] = idx[1] + ngg * labels[1];
    g_idx[2] = idx[2] + ngg * labels[2];

    metric[iSh * 3] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 1);
    metric[iSh * 3 + 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 2);
    metric[iSh * 3 + 2] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 3);
    jac[iSh] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    for (int l = 0; l < 5; ++l) {
      pv[iSh * n_var + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
    }
    for (int l = 0; l < n_scalar; ++l) {
      pv[iSh * n_var + 5 + l] = zone->sv(g_idx[0], g_idx[1], g_idx[2], l);
    }
    uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
              metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
              metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
    rhoE[iSh] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 4);
  }
  if (idx[direction] == max_extent - 1 && tid < ngg) {
    const int n_more_left = ngg - tid - 1;
    for (int m = 0; m < n_more_left; ++m) {
      const int iSh = tid + m + 1;
      int g_idx[3];
      g_idx[0] = idx[0] - (ngg - m - 1) * labels[0];
      g_idx[1] = idx[1] - (ngg - m - 1) * labels[1];
      g_idx[2] = idx[2] - (ngg - m - 1) * labels[2];

      metric[iSh * 3] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 1);
      metric[iSh * 3 + 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 2);
      metric[iSh * 3 + 2] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 3);
      jac[iSh] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
      for (int l = 0; l < 5; ++l) {
        pv[iSh * n_var + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (int l = 0; l < n_scalar; ++l) {
        pv[iSh * n_var + 5 + l] = zone->sv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
                metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
                metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
      rhoE[iSh] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 4);
    }
    const int n_more_right = ngg - 1 - tid;
    for (int m = 0; m < n_more_right; ++m) {
      const int iSh = is + m + 1;
      int g_idx[3];
      g_idx[0] = idx[0] + (m + 1) * labels[0];
      g_idx[1] = idx[1] + (m + 1) * labels[1];
      g_idx[2] = idx[2] + (m + 1) * labels[2];

      metric[iSh * 3] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 1);
      metric[iSh * 3 + 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 2);
      metric[iSh * 3 + 2] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, 3);
      jac[iSh] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
      for (int l = 0; l < 5; ++l) {
        pv[iSh * n_var + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (int l = 0; l < n_scalar; ++l) {
        pv[iSh * n_var + 5 + l] = zone->sv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      uk[iSh] = metric[iSh * 3] * pv[iSh * n_var + 1] +
                metric[iSh * 3 + 1] * pv[iSh * n_var + 2] +
                metric[iSh * 3 + 2] * pv[iSh * n_var + 3];
      rhoE[iSh] = zone->cv(g_idx[0], g_idx[1], g_idx[2], 4);
    }
  }
  __syncthreads();

  constexpr int ep_width = 3;
  constexpr real central_1[3]{0.75, -3.0 / 20, 1.0 / 60};
  real df[5 + MAX_SPEC_NUMBER + MAX_PASSIVE_SCALAR_NUMBER] = {};
  memset(df, 0, n_var * sizeof(real));
  const auto *bv = &pv[is * n_var];
  const real H = (rhoE[is] + bv[4]) / bv[0];
  for (int l = 1; l <= ep_width; ++l) {
    const int isPl = is + l, isMl = is - l;
    const auto *bv1 = &pv[isPl * n_var], *bv2 = &pv[isMl * n_var];

    const real weight1 = (bv[0] + bv1[0]) * (uk[is] * jac[is] + uk[isPl] * jac[isPl]), pWeight1 = 2 * (bv[4] + bv1[4]);
    const real weight2 = (bv[0] + bv2[0]) * (uk[is] * jac[is] + uk[isMl] * jac[isMl]), pWeight2 = 2 * (bv[4] + bv2[4]);

    const auto coefficient = central_1[l - 1];
    df[0] += coefficient * (weight1 - weight2) * 2;
    df[1] += coefficient * (weight1 * (bv[1] + bv1[1]) - weight2 * (bv[1] + bv2[1])
                            + pWeight1 * (metric[is * 3] * jac[is] + metric[isPl * 3] * jac[isPl])
                            - pWeight2 * (metric[is * 3] * jac[is] + metric[isMl * 3] * jac[isMl]));
    df[2] += coefficient * (weight1 * (bv[2] + bv1[2]) - weight2 * (bv[2] + bv2[2])
                            + pWeight1 * (metric[is * 3 + 1] * jac[is] + metric[isPl * 3 + 1] * jac[isPl])
                            - pWeight2 * (metric[is * 3 + 1] * jac[is] + metric[isMl * 3 + 1] * jac[isMl]));
    df[3] += coefficient * (weight1 * (bv[3] + bv1[3]) - weight2 * (bv[3] + bv2[3])
                            + pWeight1 * (metric[is * 3 + 2] * jac[is] + metric[isPl * 3 + 2] * jac[isPl])
                            - pWeight2 * (metric[is * 3 + 2] * jac[is] + metric[isMl * 3 + 2] * jac[isMl]));
    df[4] += coefficient * (weight1 * (H + (rhoE[isPl] + bv1[4]) / bv1[0]) -
                            weight2 * (H + (rhoE[isMl] + bv2[4]) / bv2[0]));
    for (int n = 0; n < n_scalar; ++n) {
      df[5 + n] += coefficient * (weight1 * (bv[5 + n] + bv1[5 + n]) - weight2 * (bv[5 + n] + bv2[5 + n]));
    }
  }
  zone->dq(idx[0], idx[1], idx[2], 0) -= 0.25 * df[0];
  zone->dq(idx[0], idx[1], idx[2], 1) -= 0.25 * df[1];
  zone->dq(idx[0], idx[1], idx[2], 2) -= 0.25 * df[2];
  zone->dq(idx[0], idx[1], idx[2], 3) -= 0.25 * df[3];
  zone->dq(idx[0], idx[1], idx[2], 4) -= 0.25 * df[4];
  for (int l = 5; l < n_var; ++l) {
    zone->dq(idx[0], idx[1], idx[2], l) -= 0.25 * df[l];
  }
}

// template instantiation
template void compute_convective_term_pv<MixtureModel::Air>(const Block &block, DZone *zone, DParameter *param,
  int n_var, const Parameter &parameter);

template void compute_convective_term_pv<MixtureModel::Mixture>(const Block &block, DZone *zone, DParameter *param,
  int n_var, const Parameter &parameter);

template void compute_convective_term_aweno<MixtureModel::Air>(const Block &block, DZone *zone, DParameter *param,
  int n_var);

template void compute_convective_term_aweno<MixtureModel::Mixture>(const Block &block, DZone *zone, DParameter *param,
  int n_var);

template void Roe_compute_inviscid_flux<MixtureModel::Air>(const Block &block, DZone *zone, DParameter *param,
  int n_var, const Parameter &parameter);

template void Roe_compute_inviscid_flux<MixtureModel::Mixture>(const Block &block, DZone *zone, DParameter *param,
  int n_var, const Parameter &parameter);
}
