#include "ViscousScheme.cuh"
#include "DParameter.cuh"

namespace cfd {
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
