#pragma once

#include "Define.h"
#include "DPLUR.cuh"

namespace cfd {
template<MixtureModel mixture_model, class turb_method>
void
implicit_treatment(const Block &block, const DParameter *param, DZone *d_ptr, const Parameter &parameter, DZone *h_ptr,
                   DBoundCond &bound_cond, real diag_factor = 0) {
  switch (parameter.get_int("implicit_method")) {
    case 0: // Explicit
      if constexpr (mixture_model == MixtureModel::FR) {
        if (const int chem_src_method = parameter.get_int("chemSrcMethod");chem_src_method != 0) {
          const int extent[3]{block.mx, block.my, block.mz};
          const int dim{extent[2] == 1 ? 2 : 3};
          dim3 tpb{8, 8, 4};
          if (dim == 2) {
            tpb = {16, 16, 1};
          }
          const dim3 bpg{(extent[0] - 1) / tpb.x + 1, (extent[1] - 1) / tpb.y + 1, (extent[2] - 1) / tpb.z + 1};
          switch (chem_src_method) {
            case 1: // EPI
              EPI<<<bpg, tpb>>>(d_ptr, parameter.get_int("n_spec"));
              break;
            case 2: // DA
              DA<<<bpg, tpb>>>(d_ptr, parameter.get_int("n_spec"));
              break;
            default: // explicit
              break;
          }
        }
      }
      if constexpr (TurbMethod<turb_method>::hasImplicitTreat) {
        if (parameter.get_int("turb_implicit") == 1) {
          const int extent[3]{block.mx, block.my, block.mz};
          const int dim{extent[2] == 1 ? 2 : 3};
          dim3 tpb{8, 8, 4};
          if (dim == 2) {
            tpb = {16, 16, 1};
          }
          const dim3 bpg{(extent[0] - 1) / tpb.x + 1, (extent[1] - 1) / tpb.y + 1, (extent[2] - 1) / tpb.z + 1};
          if constexpr (TurbMethod<turb_method>::label == TurbMethodLabel::SST) {
            implicit_treat_for_SST<<<bpg, tpb>>>(d_ptr, param);
          }
        }
      }
      return;
    case 1: // DPLUR
      DPLUR<mixture_model, turb_method>(block, param, d_ptr, h_ptr, parameter, bound_cond, diag_factor);
    default:
      return;
  }
}
}