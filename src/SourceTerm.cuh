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

  if constexpr (TurbMethod<turb_method>::hasMut) {
    turb_method::compute_source_and_mut(zone, i, j, k, param);
  }

  if constexpr (mix_model == MixtureModel::FR) {
    // Finite rate chemistry will be computed
    finite_rate_chemistry(zone, i, j, k, param);
  } else if constexpr (mix_model == MixtureModel::FL || mix_model == MixtureModel::MixtureFraction) {
    // Flamelet model, the source term of the mixture fraction and its variance will be computed
    flamelet_source(zone, i, j, k, param);
  }

  if (param->sponge_layer) {
    // Sponge layer will be computed
    sponge_layer_source(zone, i, j, k, param);
  }
}
}