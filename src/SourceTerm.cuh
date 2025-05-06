#pragma once

#include "Define.h"
#include "Field.h"
#include "DParameter.cuh"
#include "SST.cuh"
#include "FiniteRateChem.cuh"
#include "SpongeLayer.cuh"

namespace cfd {
template<MixtureModel mix_model, class turb_method>
__global__ void compute_source(DZone *zone, DParameter *param) {
  const int extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  // Finite rate chemistry will be computed
  finite_rate_chemistry(zone, i, j, k, param);
}
}