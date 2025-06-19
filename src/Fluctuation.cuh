#pragma once
#include "DParameter.cuh"
#include "Field.h"

namespace cfd {
void compute_fluctuation(const Block &block,DZone* zone, DParameter *param, int form, Parameter &parameter);

__global__ void ferrer_fluctuation(DZone* zone, DParameter *param);

__global__ void update_values_with_fluctuations(DZone* zone, DParameter *param);
}
