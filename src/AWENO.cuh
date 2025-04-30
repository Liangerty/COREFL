#pragma once

#include "Define.h"
#include "Mesh.h"
#include "gxl_lib/Array.hpp"

namespace cfd {
struct DZone;
struct DParameter;

template<MixtureModel mix_model>
__device__ void
AWENO_interpolation(const real *cv, real *pv_l, real *pv_r, int idx_shared, int n_var, const real *metric,
                    DParameter *param);

__device__ double2 WENO5(const real *L, const real *cv, int n_var, int i_shared, int l_row);

__device__ double2 WENO7(const real *L, const real *cv, int n_var, int i_shared, int l_row);

template<MixtureModel mix_model>
__global__ void CDSPart1D(DZone *zone, int direction, int max_extent, DParameter *param);
// Implementations
}