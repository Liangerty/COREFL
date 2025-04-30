#pragma once

#include "InviscidScheme.cuh"
#include "ViscousScheme.cuh"

namespace cfd {
template<MixtureModel mix_model>
void compute_inviscid_flux(const Block &block, DZone *zone, DParameter *param, int n_var, const Parameter &parameter) {
  switch (parameter.get_int("inviscid_type")) {
    case 0: // Compute the term with primitive reconstruction methods. (MUSCL/NND/1stOrder + LF/AUSM+/HLLC)
      compute_convective_term_pv<mix_model>(block, zone, param, n_var, parameter);
      break;
    case 1: // Compute the term with AWENO methods. (WENO-Z-5 + LF/AUSM+/HLLC)
      compute_convective_term_aweno<mix_model>(block, zone, param, n_var);
      break;
    case 3: // Compute the term with WENO-Z-5
      compute_convective_term_weno<mix_model>(block, zone, param, n_var, parameter);
      break;
    case 4:
      compute_convective_term_ep(block, zone, param, n_var);
      break;
    case 5:
      compute_convective_term_hybrid_weno_ep<mix_model>(block, zone, param, n_var, parameter);
      break;
    case 6:
      compute_convective_term_hybrid_ud_weno<mix_model>(block, zone, param, n_var, parameter);
      break;
    case 2:  // Roe scheme
    default: // Roe scheme
      Roe_compute_inviscid_flux<mix_model>(block, zone, param, n_var, parameter);
      break;
  }
}

template<MixtureModel mix_model, class turb_method>
void compute_viscous_flux(const Block &block, DZone *zone, DParameter *param, const Parameter &parameter) {
  const int viscous_order = parameter.get_int("viscous_order");

  if (viscous_order == 0)
    return;

  const auto mx = block.mx, my = block.my, mz = block.mz;
  const int dim{mz == 1 ? 2 : 3};

  dim3 tpb = {32, 8, 2};
  if (dim == 2)
    tpb = {32, 16, 1};
  dim3 BPG{(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};

  if (viscous_order == 2) {
    auto bpg = dim3(mx/*+1-1*/ / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1);
    compute_fv_2nd_order<mix_model, turb_method><<<bpg, tpb>>>(zone, param);
    compute_dFv_dx<<<BPG, tpb>>>(zone, param);

    bpg = dim3((mx - 1) / tpb.x + 1, my/*+1-1*/ / tpb.y + 1, (mz - 1) / tpb.z + 1);
    compute_gv_2nd_order<mix_model, turb_method><<<bpg, tpb>>>(zone, param);
    compute_dGv_dy<<<BPG, tpb>>>(zone, param);

    if (dim == 3) {
      dim3 TPB = {32, 8, 2};
      // constexpr dim3 TPB = {32, 8, 2};
      // bpg = dim3((mx - 1) / TPB.x + 1, (my - 1) / TPB.y + 1, mz/*+1-1*/ / TPB.z + 1);
      bpg = dim3((mx - 1) / TPB.x + 1, (my - 1) / TPB.y + 1, mz/*+1-1*/ / TPB.z + 1);
      compute_hv_2nd_order<mix_model, turb_method><<<bpg, TPB>>>(zone, param);

      compute_dHv_dz<<<BPG, tpb>>>(zone, param);
    }
  } else if (viscous_order == 6) {
    dim3 bpg{mx/*+1-1*/ / tpb.x + 1, my / tpb.y + 1, mz / tpb.z + 1};
    compute_gradient_alpha_damping_6th_order<<<bpg, tpb>>>(zone, param);

    bpg = dim3(mx/*+1-1*/ / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1);
    compute_fv_6th_order_alpha_damping<mix_model, turb_method><<<bpg, tpb>>>(zone, param);
    compute_dFv_dx<<<BPG, tpb>>>(zone, param);

    bpg = dim3((mx - 1) / tpb.x + 1, my/*+1-1*/ / tpb.y + 1, (mz - 1) / tpb.z + 1);
    compute_gv_6th_order_alpha_damping<mix_model, turb_method><<<bpg, tpb>>>(zone, param);
    compute_dGv_dy<<<BPG, tpb>>>(zone, param);

    if (dim == 3) {
      dim3 TPB = {32, 8, 2};
      // constexpr dim3 TPB = {32, 8, 2};
      // bpg = dim3((mx - 1) / TPB.x + 1, (my - 1) / TPB.y + 1, mz/*+1-1*/ / TPB.z + 1);
      bpg = dim3((mx - 1) / TPB.x + 1, (my - 1) / TPB.y + 1, mz/*+1-1*/ / TPB.z + 1);
      compute_hv_6th_order_alpha_damping<mix_model, turb_method><<<bpg, TPB>>>(zone, param);

      compute_dHv_dz<<<BPG, tpb>>>(zone, param);
    }
  }
}
}
