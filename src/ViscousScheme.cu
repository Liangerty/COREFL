#include "ViscousScheme.cuh"

namespace cfd {
__global__ void compute_gradient_alpha_damping_6th_order(DZone *zone, DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) - 1;
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y) - 1;
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z) - 1;
  if (i >= zone->mx || j >= zone->my || k >= zone->mz) return;

  // Chandravamsi et, al. Computers & Fluids, 2023, 258
  // Chamarthi et, al. JCP, 2022, 460
  constexpr real twoDiv3 = 2.0 / 3.0, oneDiv12 = 1.0 / 12.0;
  constexpr real alpha = 38.0 / 15, beta = -11.0 / 228;

  auto &grad = zone->grad_bv;
  const auto &pv = zone->bv;

  real u_k_i = twoDiv3 * (pv(i + 1, j, k, 1) - pv(i - 1, j, k, 1)) - oneDiv12 * (
                 pv(i + 2, j, k, 1) - pv(i - 2, j, k, 1));
  real u_k_iP1 = twoDiv3 * (pv(i + 2, j, k, 1) - pv(i, j, k, 1)) - oneDiv12 * (pv(i + 3, j, k, 1) - pv(i - 1, j, k, 1));
  real uR = pv(i + 1, j, k, 1) - 0.5 * u_k_iP1 + beta * (pv(i + 2, j, k, 1) - 2 * pv(i + 1, j, k, 1) + pv(i, j, k, 1));
  real uL = pv(i, j, k, 1) + 0.5 * u_k_i + beta * (pv(i + 1, j, k, 1) - 2 * pv(i, j, k, 1) + pv(i - 1, j, k, 1));
  real u_xi = 0.5 * (u_k_i + u_k_iP1) + 0.5 * alpha * (uR - uL);
  grad(i, j, k, 0) = u_xi;

  u_k_i = twoDiv3 * (pv(i, j + 1, k, 1) - pv(i, j - 1, k, 1)) - oneDiv12 * (pv(i, j + 2, k, 1) - pv(i, j - 2, k, 1));
  u_k_iP1 = twoDiv3 * (pv(i, j + 2, k, 1) - pv(i, j, k, 1)) - oneDiv12 * (pv(i, j + 3, k, 1) - pv(i, j - 1, k, 1));
  uR = pv(i, j + 1, k, 1) - 0.5 * u_k_iP1 + beta * (pv(i, j + 2, k, 1) - 2 * pv(i, j + 1, k, 1) + pv(i, j, k, 1));
  uL = pv(i, j, k, 1) + 0.5 * u_k_i + beta * (pv(i, j + 1, k, 1) - 2 * pv(i, j, k, 1) + pv(i, j - 1, k, 1));
  real u_eta = 0.5 * (u_k_i + u_k_iP1) + 0.5 * alpha * (uR - uL);
  grad(i, j, k, 1) = u_eta;

  u_k_i = twoDiv3 * (pv(i, j, k + 1, 1) - pv(i, j, k - 1, 1)) - oneDiv12 * (pv(i, j, k + 2, 1) - pv(i, j, k - 2, 1));
  u_k_iP1 = twoDiv3 * (pv(i, j, k + 2, 1) - pv(i, j, k, 1)) - oneDiv12 * (pv(i, j, k + 3, 1) - pv(i, j, k - 1, 1));
  uR = pv(i, j, k + 1, 1) - 0.5 * u_k_iP1 + beta * (pv(i, j, k + 2, 1) - 2 * pv(i, j, k + 1, 1) + pv(i, j, k, 1));
  uL = pv(i, j, k, 1) + 0.5 * u_k_i + beta * (pv(i, j, k + 1, 1) - 2 * pv(i, j, k, 1) + pv(i, j, k - 1, 1));
  real u_zeta = 0.5 * (u_k_i + u_k_iP1) + 0.5 * alpha * (uR - uL);
  grad(i, j, k, 2) = u_zeta;

  u_k_i = twoDiv3 * (pv(i + 1, j, k, 2) - pv(i - 1, j, k, 2)) - oneDiv12 * (pv(i + 2, j, k, 2) - pv(i - 2, j, k, 2));
  u_k_iP1 = twoDiv3 * (pv(i + 2, j, k, 2) - pv(i, j, k, 2)) - oneDiv12 * (pv(i + 3, j, k, 2) - pv(i - 1, j, k, 2));
  uR = pv(i + 1, j, k, 2) - 0.5 * u_k_iP1 + beta * (pv(i + 2, j, k, 2) - 2 * pv(i + 1, j, k, 2) + pv(i, j, k, 2));
  uL = pv(i, j, k, 2) + 0.5 * u_k_i + beta * (pv(i + 1, j, k, 2) - 2 * pv(i, j, k, 2) + pv(i - 1, j, k, 2));
  u_xi = 0.5 * (u_k_i + u_k_iP1) + 0.5 * alpha * (uR - uL);
  grad(i, j, k, 3) = u_xi;

  u_k_i = twoDiv3 * (pv(i, j + 1, k, 2) - pv(i, j - 1, k, 2)) - oneDiv12 * (pv(i, j + 2, k, 2) - pv(i, j - 2, k, 2));
  u_k_iP1 = twoDiv3 * (pv(i, j + 2, k, 2) - pv(i, j, k, 2)) - oneDiv12 * (pv(i, j + 3, k, 2) - pv(i, j - 1, k, 2));
  uR = pv(i, j + 1, k, 2) - 0.5 * u_k_iP1 + beta * (pv(i, j + 2, k, 2) - 2 * pv(i, j + 1, k, 2) + pv(i, j, k, 2));
  uL = pv(i, j, k, 2) + 0.5 * u_k_i + beta * (pv(i, j + 1, k, 2) - 2 * pv(i, j, k, 2) + pv(i, j - 1, k, 2));
  u_eta = 0.5 * (u_k_i + u_k_iP1) + 0.5 * alpha * (uR - uL);
  grad(i, j, k, 4) = u_eta;

  u_k_i = twoDiv3 * (pv(i, j, k + 1, 2) - pv(i, j, k - 1, 2)) - oneDiv12 * (pv(i, j, k + 2, 2) - pv(i, j, k - 2, 2));
  u_k_iP1 = twoDiv3 * (pv(i, j, k + 2, 2) - pv(i, j, k, 2)) - oneDiv12 * (pv(i, j, k + 3, 2) - pv(i, j, k - 1, 2));
  uR = pv(i, j, k + 1, 2) - 0.5 * u_k_iP1 + beta * (pv(i, j, k + 2, 2) - 2 * pv(i, j, k + 1, 2) + pv(i, j, k, 2));
  uL = pv(i, j, k, 2) + 0.5 * u_k_i + beta * (pv(i, j, k + 1, 2) - 2 * pv(i, j, k, 2) + pv(i, j, k - 1, 2));
  u_zeta = 0.5 * (u_k_i + u_k_iP1) + 0.5 * alpha * (uR - uL);
  grad(i, j, k, 5) = u_zeta;

  u_k_i = twoDiv3 * (pv(i + 1, j, k, 3) - pv(i - 1, j, k, 3)) - oneDiv12 * (pv(i + 2, j, k, 3) - pv(i - 2, j, k, 3));
  u_k_iP1 = twoDiv3 * (pv(i + 2, j, k, 3) - pv(i, j, k, 3)) - oneDiv12 * (pv(i + 3, j, k, 3) - pv(i - 1, j, k, 3));
  uR = pv(i + 1, j, k, 3) - 0.5 * u_k_iP1 + beta * (pv(i + 2, j, k, 3) - 2 * pv(i + 1, j, k, 3) + pv(i, j, k, 3));
  uL = pv(i, j, k, 3) + 0.5 * u_k_i + beta * (pv(i + 1, j, k, 3) - 2 * pv(i, j, k, 3) + pv(i - 1, j, k, 3));
  u_xi = 0.5 * (u_k_i + u_k_iP1) + 0.5 * alpha * (uR - uL);
  grad(i, j, k, 6) = u_xi;

  u_k_i = twoDiv3 * (pv(i, j + 1, k, 3) - pv(i, j - 1, k, 3)) - oneDiv12 * (pv(i, j + 2, k, 3) - pv(i, j - 2, k, 3));
  u_k_iP1 = twoDiv3 * (pv(i, j + 2, k, 3) - pv(i, j, k, 3)) - oneDiv12 * (pv(i, j + 3, k, 3) - pv(i, j - 1, k, 3));
  uR = pv(i, j + 1, k, 3) - 0.5 * u_k_iP1 + beta * (pv(i, j + 2, k, 3) - 2 * pv(i, j + 1, k, 3) + pv(i, j, k, 3));
  uL = pv(i, j, k, 3) + 0.5 * u_k_i + beta * (pv(i, j + 1, k, 3) - 2 * pv(i, j, k, 3) + pv(i, j - 1, k, 3));
  u_eta = 0.5 * (u_k_i + u_k_iP1) + 0.5 * alpha * (uR - uL);
  grad(i, j, k, 7) = u_eta;

  u_k_i = twoDiv3 * (pv(i, j, k + 1, 3) - pv(i, j, k - 1, 3)) - oneDiv12 * (pv(i, j, k + 2, 3) - pv(i, j, k - 2, 3));
  u_k_iP1 = twoDiv3 * (pv(i, j, k + 2, 3) - pv(i, j, k, 3)) - oneDiv12 * (pv(i, j, k + 3, 3) - pv(i, j, k - 1, 3));
  uR = pv(i, j, k + 1, 3) - 0.5 * u_k_iP1 + beta * (pv(i, j, k + 2, 3) - 2 * pv(i, j, k + 1, 3) + pv(i, j, k, 3));
  uL = pv(i, j, k, 3) + 0.5 * u_k_i + beta * (pv(i, j, k + 1, 3) - 2 * pv(i, j, k, 3) + pv(i, j, k - 1, 3));
  u_zeta = 0.5 * (u_k_i + u_k_iP1) + 0.5 * alpha * (uR - uL);
  grad(i, j, k, 8) = u_zeta;

  u_k_i = twoDiv3 * (pv(i + 1, j, k, 5) - pv(i - 1, j, k, 5)) - oneDiv12 * (pv(i + 2, j, k, 5) - pv(i - 2, j, k, 5));
  u_k_iP1 = twoDiv3 * (pv(i + 2, j, k, 5) - pv(i, j, k, 5)) - oneDiv12 * (pv(i + 3, j, k, 5) - pv(i - 1, j, k, 5));
  uR = pv(i + 1, j, k, 5) - 0.5 * u_k_iP1 + beta * (pv(i + 2, j, k, 5) - 2 * pv(i + 1, j, k, 5) + pv(i, j, k, 5));
  uL = pv(i, j, k, 5) + 0.5 * u_k_i + beta * (pv(i + 1, j, k, 5) - 2 * pv(i, j, k, 5) + pv(i - 1, j, k, 5));
  u_xi = 0.5 * (u_k_i + u_k_iP1) + 0.5 * alpha * (uR - uL);
  grad(i, j, k, 9) = u_xi;

  u_k_i = twoDiv3 * (pv(i, j + 1, k, 5) - pv(i, j - 1, k, 5)) - oneDiv12 * (pv(i, j + 2, k, 5) - pv(i, j - 2, k, 5));
  u_k_iP1 = twoDiv3 * (pv(i, j + 2, k, 5) - pv(i, j, k, 5)) - oneDiv12 * (pv(i, j + 3, k, 5) - pv(i, j - 1, k, 5));
  uR = pv(i, j + 1, k, 5) - 0.5 * u_k_iP1 + beta * (pv(i, j + 2, k, 5) - 2 * pv(i, j + 1, k, 5) + pv(i, j, k, 5));
  uL = pv(i, j, k, 5) + 0.5 * u_k_i + beta * (pv(i, j + 1, k, 5) - 2 * pv(i, j, k, 5) + pv(i, j - 1, k, 5));
  u_eta = 0.5 * (u_k_i + u_k_iP1) + 0.5 * alpha * (uR - uL);
  grad(i, j, k, 10) = u_eta;

  u_k_i = twoDiv3 * (pv(i, j, k + 1, 5) - pv(i, j, k - 1, 5)) - oneDiv12 * (pv(i, j, k + 2, 5) - pv(i, j, k - 2, 5));
  u_k_iP1 = twoDiv3 * (pv(i, j, k + 2, 5) - pv(i, j, k, 5)) - oneDiv12 * (pv(i, j, k + 3, 5) - pv(i, j, k - 1, 5));
  uR = pv(i, j, k + 1, 5) - 0.5 * u_k_iP1 + beta * (pv(i, j, k + 2, 5) - 2 * pv(i, j, k + 1, 5) + pv(i, j, k, 5));
  uL = pv(i, j, k, 5) + 0.5 * u_k_i + beta * (pv(i, j, k + 1, 5) - 2 * pv(i, j, k, 5) + pv(i, j, k - 1, 5));
  u_zeta = 0.5 * (u_k_i + u_k_iP1) + 0.5 * alpha * (uR - uL);
  grad(i, j, k, 11) = u_zeta;

  if (param->n_spec > 0 && param->gradPInDiffusionFlux) {
    u_k_i = twoDiv3 * (pv(i + 1, j, k, 4) - pv(i - 1, j, k, 4)) - oneDiv12 * (pv(i + 2, j, k, 4) - pv(i - 2, j, k, 4));
    u_k_iP1 = twoDiv3 * (pv(i + 2, j, k, 4) - pv(i, j, k, 4)) - oneDiv12 * (pv(i + 3, j, k, 4) - pv(i - 1, j, k, 4));
    uR = pv(i + 1, j, k, 4) - 0.5 * u_k_iP1 + beta * (pv(i + 2, j, k, 4) - 2 * pv(i + 1, j, k, 4) + pv(i, j, k, 4));
    uL = pv(i, j, k, 4) + 0.5 * u_k_i + beta * (pv(i + 1, j, k, 4) - 2 * pv(i, j, k, 4) + pv(i - 1, j, k, 4));
    u_xi = 0.5 * (u_k_i + u_k_iP1) + 0.5 * alpha * (uR - uL);
    grad(i, j, k, 12) = u_xi;

    u_k_i = twoDiv3 * (pv(i, j + 1, k, 4) - pv(i, j - 1, k, 4)) - oneDiv12 * (pv(i, j + 2, k, 4) - pv(i, j - 2, k, 4));
    u_k_iP1 = twoDiv3 * (pv(i, j + 2, k, 4) - pv(i, j, k, 4)) - oneDiv12 * (pv(i, j + 3, k, 4) - pv(i, j - 1, k, 4));
    uR = pv(i, j + 1, k, 4) - 0.5 * u_k_iP1 + beta * (pv(i, j + 2, k, 4) - 2 * pv(i, j + 1, k, 4) + pv(i, j, k, 4));
    uL = pv(i, j, k, 4) + 0.5 * u_k_i + beta * (pv(i, j + 1, k, 4) - 2 * pv(i, j, k, 4) + pv(i, j - 1, k, 4));
    u_eta = 0.5 * (u_k_i + u_k_iP1) + 0.5 * alpha * (uR - uL);
    grad(i, j, k, 13) = u_eta;

    u_k_i = twoDiv3 * (pv(i, j, k + 1, 4) - pv(i, j, k - 1, 4)) - oneDiv12 * (pv(i, j, k + 2, 4) - pv(i, j, k - 2, 4));
    u_k_iP1 = twoDiv3 * (pv(i, j, k + 2, 4) - pv(i, j, k, 4)) - oneDiv12 * (pv(i, j, k + 3, 4) - pv(i, j, k - 1, 4));
    uR = pv(i, j, k + 1, 4) - 0.5 * u_k_iP1 + beta * (pv(i, j, k + 2, 4) - 2 * pv(i, j, k + 1, 4) + pv(i, j, k, 4));
    uL = pv(i, j, k, 4) + 0.5 * u_k_i + beta * (pv(i, j, k + 1, 4) - 2 * pv(i, j, k, 4) + pv(i, j, k - 1, 4));
    u_zeta = 0.5 * (u_k_i + u_k_iP1) + 0.5 * alpha * (uR - uL);
    grad(i, j, k, 14) = u_zeta;
  }

  const auto &sv = zone->sv;
  for (int l = 0; l < param->n_scalar; ++l) {
    const int idx = l * 3 + 15;
    u_k_i = twoDiv3 * (sv(i + 1, j, k, l) - sv(i - 1, j, k, l)) - oneDiv12 * (sv(i + 2, j, k, l) - sv(i - 2, j, k, l));
    u_k_iP1 = twoDiv3 * (sv(i + 2, j, k, l) - sv(i, j, k, l)) - oneDiv12 * (sv(i + 3, j, k, l) - sv(i - 1, j, k, l));
    uR = sv(i + 1, j, k, l) - 0.5 * u_k_iP1 + beta * (sv(i + 2, j, k, l) - 2 * sv(i + 1, j, k, l) + sv(i, j, k, l));
    uL = sv(i, j, k, l) + 0.5 * u_k_i + beta * (sv(i + 1, j, k, l) - 2 * sv(i, j, k, l) + sv(i - 1, j, k, l));
    u_xi = 0.5 * (u_k_i + u_k_iP1) + 0.5 * alpha * (uR - uL);
    grad(i, j, k, idx) = u_xi;

    u_k_i = twoDiv3 * (sv(i, j + 1, k, l) - sv(i, j - 1, k, l)) - oneDiv12 * (sv(i, j + 2, k, l) - sv(i, j - 2, k, l));
    u_k_iP1 = twoDiv3 * (sv(i, j + 2, k, l) - sv(i, j, k, l)) - oneDiv12 * (sv(i, j + 3, k, l) - sv(i, j - 1, k, l));
    uR = sv(i, j + 1, k, l) - 0.5 * u_k_iP1 + beta * (sv(i, j + 2, k, l) - 2 * sv(i, j + 1, k, l) + sv(i, j, k, l));
    uL = sv(i, j, k, l) + 0.5 * u_k_i + beta * (sv(i, j + 1, k, l) - 2 * sv(i, j, k, l) + sv(i, j - 1, k, l));
    u_eta = 0.5 * (u_k_i + u_k_iP1) + 0.5 * alpha * (uR - uL);
    grad(i, j, k, idx + 1) = u_eta;

    u_k_i = twoDiv3 * (sv(i, j, k + 1, l) - sv(i, j, k - 1, l)) - oneDiv12 * (sv(i, j, k + 2, l) - sv(i, j, k - 2, l));
    u_k_iP1 = twoDiv3 * (sv(i, j, k + 2, l) - sv(i, j, k, l)) - oneDiv12 * (sv(i, j, k + 3, l) - sv(i, j, k - 1, l));
    uR = sv(i, j, k + 1, l) - 0.5 * u_k_iP1 + beta * (sv(i, j, k + 2, l) - 2 * sv(i, j, k + 1, l) + sv(i, j, k, l));
    uL = sv(i, j, k, l) + 0.5 * u_k_i + beta * (sv(i, j, k + 1, l) - 2 * sv(i, j, k, l) + sv(i, j, k - 1, l));
    u_zeta = 0.5 * (u_k_i + u_k_iP1) + 0.5 * alpha * (uR - uL);
    grad(i, j, k, idx + 2) = u_zeta;
  }
}

__global__ void compute_dFv_dx(DZone *zone, const DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= zone->mx || j >= zone->my || k >= zone->mz) return;

  const int nv = param->n_var;
  auto &dq = zone->dq;
  auto &fv = zone->vis_flux;
  dq(i, j, k, 1) += fv(i, j, k, 0) - fv(i - 1, j, k, 0);
  dq(i, j, k, 2) += fv(i, j, k, 1) - fv(i - 1, j, k, 1);
  dq(i, j, k, 3) += fv(i, j, k, 2) - fv(i - 1, j, k, 2);
  dq(i, j, k, 4) += fv(i, j, k, 3) - fv(i - 1, j, k, 3);
  for (int l = 5; l < nv; ++l) {
    dq(i, j, k, l) += fv(i, j, k, l - 1) - fv(i - 1, j, k, l - 1);
  }
}

__global__ void compute_dGv_dy(DZone *zone, const DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= zone->mx || j >= zone->my || k >= zone->mz) return;

  const int nv = param->n_var;
  auto &dq = zone->dq;
  auto &gv = zone->vis_flux;
  dq(i, j, k, 1) += gv(i, j, k, 0) - gv(i, j - 1, k, 0);
  dq(i, j, k, 2) += gv(i, j, k, 1) - gv(i, j - 1, k, 1);
  dq(i, j, k, 3) += gv(i, j, k, 2) - gv(i, j - 1, k, 2);
  dq(i, j, k, 4) += gv(i, j, k, 3) - gv(i, j - 1, k, 3);
  for (int l = 5; l < nv; ++l) {
    dq(i, j, k, l) += gv(i, j, k, l - 1) - gv(i, j - 1, k, l - 1);
  }
}

__global__ void compute_dHv_dz(DZone *zone, const DParameter *param) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= zone->mx || j >= zone->my || k >= zone->mz) return;

  const int nv = param->n_var;
  auto &dq = zone->dq;
  auto &hv = zone->vis_flux;
  dq(i, j, k, 1) += hv(i, j, k, 0) - hv(i, j, k - 1, 0);
  dq(i, j, k, 2) += hv(i, j, k, 1) - hv(i, j, k - 1, 1);
  dq(i, j, k, 3) += hv(i, j, k, 2) - hv(i, j, k - 1, 2);
  dq(i, j, k, 4) += hv(i, j, k, 3) - hv(i, j, k - 1, 3);
  for (int l = 5; l < nv; ++l) {
    dq(i, j, k, l) += hv(i, j, k, l - 1) - hv(i, j, k - 1, l - 1);
  }
}
} // namespace cfd
