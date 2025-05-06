#pragma once

#include <curand_kernel.h>
#include "BoundCond.h"
#include "Mesh.h"
#include "Field.h"
#include "DParameter.cuh"
#include "FieldOperation.cuh"
#include "SST.cuh"
#include "gxl_lib/Array.cuh"
#include "gxl_lib/MyAlgorithm.h"

namespace cfd {
struct BCInfo {
  int label = 0;
  int n_boundary = 0;
  int2 *boundary = nullptr;
};

void read_profile(const Boundary &boundary, const std::string &file, const Block &block, Parameter &parameter,
  const Species &species, ggxl::VectorField3D<real> &profile, const std::string &profile_related_bc_name);

void read_lst_profile(const Boundary &boundary, const std::string &file, const Block &block, const Parameter &parameter,
  const Species &species, ggxl::VectorField3D<real> &profile, const std::string &profile_related_bc_name);

void read_dat_profile(const Boundary &boundary, const std::string &file, const Block &block, Parameter &parameter,
  const Species &species, ggxl::VectorField3D<real> &profile, const std::string &profile_related_bc_name);

struct DBoundCond {
  DBoundCond() = default;

  void initialize_bc_on_GPU(Mesh &mesh, std::vector<Field> &field, Species &species, Parameter &parameter);

  void link_bc_to_boundaries(Mesh &mesh, std::vector<Field> &field) const;

  template<MixtureModel mix_model, class turb>
  void
  apply_boundary_conditions(const Block &block, Field &field, DParameter *param, int step = -1) const;

  template<MixtureModel mix_model, class turb>
  void nonReflectingBoundary(const Block &block, Field &field, DParameter *param) const;

  void write_df(Parameter &parameter, const Mesh &mesh) const;

  // There may be time-dependent BCs, which need to be updated at each time step.
  // E.g., the turbulent library method, we need to update the profile and fluctuation.
  // E.g., the NSCBC
  // Therefore, this function may be extended in the future to be called "time-dependent bc update".
  // void time_dependent_bc_update(const Mesh &mesh, std::vector<Field> &field, DParameter *param, Parameter &parameter) const;

  int n_wall = 0, n_symmetry = 0, n_inflow = 0, n_outflow = 0, n_farfield = 0, n_subsonic_inflow = 0, n_back_pressure =
          0, n_periodic = 0;
  BCInfo *wall_info = nullptr;
  BCInfo *symmetry_info = nullptr;
  BCInfo *inflow_info = nullptr;
  BCInfo *outflow_info = nullptr;
  BCInfo *farfield_info = nullptr;
  BCInfo *subsonic_inflow_info = nullptr;
  BCInfo *back_pressure_info = nullptr;
  BCInfo *periodic_info = nullptr;
  Wall *wall = nullptr;
  Symmetry *symmetry = nullptr;
  Inflow *inflow = nullptr;
  Outflow *outflow = nullptr;
  FarField *farfield = nullptr;
  SubsonicInflow *subsonic_inflow = nullptr;
  BackPressure *back_pressure = nullptr;
  Periodic *periodic = nullptr;

  // Non-reflecting boundary
  std::vector<int> nonReflectingBCs{};

  // Profiles
  // For now, please make sure that all profiles are in the same plane, that the plane is not split into several parts.
  // There may be inflow with values of ghost grids also given.
  std::vector<ggxl::VectorField3D<real>> profile_hPtr_withGhost = {};
  ggxl::VectorField3D<real> *profile_dPtr_withGhost = nullptr;
  // Fluctuation profiles, with real part and imaginary part given for basic variables
  ggxl::VectorField3D<real> *fluctuation_dPtr = nullptr;

  // Digital filter related
  int n_df_face = 0;
  std::vector<int> df_label = {};
  std::vector<int> df_related_block = {};
  constexpr static int DF_N = 50;
  // Random values for digital filter.
  // E.g., the dimensions are often like (ny,nz,3), where 3 is for 3 components of velocity.
  ggxl::VectorField2D<real> *random_values_hPtr = nullptr;
  ggxl::VectorField2D<real> *random_values_dPtr = nullptr;
  ggxl::VectorField1D<real> *df_lundMatrix_dPtr = nullptr; // Lund matrix for digital filter
  // (0:my-1, 0:2*DF_N, 0:2): my*(2N+1)*3, the second index jj corresponds to jj-N
  ggxl::VectorField2D<real> *df_by_dPtr = nullptr;
  ggxl::VectorField2D<real> *df_bz_dPtr = nullptr;
  ggxl::VectorField2D<curandState> *rng_states_hPtr = nullptr; // Random number generator states for digital filter
  ggxl::VectorField2D<curandState> *rng_states_dPtr = nullptr; // Random number generator states for digital filter
  ggxl::VectorField2D<real> *df_fy_dPtr = nullptr;
  ggxl::VectorField2D<real> *df_velFluc_old_hPtr = nullptr;
  ggxl::VectorField2D<real> *df_velFluc_old_dPtr = nullptr;
  ggxl::VectorField2D<real> *df_velFluc_new_hPtr = nullptr;
  ggxl::VectorField2D<real> *df_velFluc_new_dPtr = nullptr;

  ggxl::VectorField2DHost<curandState> *df_rng_state_cpu = nullptr;
  ggxl::VectorField2DHost<real> *df_velFluc_cpu = nullptr;

  curandState *rng_d_ptr = nullptr;

private:
  void initialize_digital_filter(Parameter &parameter, Mesh &mesh);

  void initialize_df_memory(const Mesh &mesh, const std::vector<int> &N1, const std::vector<int> &N2);

  void get_digital_filter_lund_matrix(Parameter &parameter, const std::vector<int> &N1,
    const std::vector<std::vector<real>> &scaled_y) const;

  void get_digital_filter_convolution_kernel(Parameter &parameter, const std::vector<int> &N1,
    const std::vector<std::vector<real>> &y_scaled, real dz) const;

  void generate_random_numbers(int iFace, int my, int mz, int ngg) const;

  void apply_convolution(int iFace, int my, int mz, int ngg) const;

  void initialize_profile_and_rng(Parameter &parameter, Mesh &mesh, const Species &species);

  void compute_fluctuations(const DParameter *param, DZone *zone, const Inflow *inflowHere, int iFace, int my, int mz,
    int ngg) const;
};

void count_boundary_of_type_bc(const std::vector<Boundary> &boundary, int n_bc, int **sep, int blk_idx, int n_block,
  BCInfo *bc_info);

void link_boundary_and_condition(const std::vector<Boundary> &boundary, const BCInfo *bc, int n_bc, int **sep,
  int i_zone);

__global__ void initialize_rng(curandState *rng_states, int size, int64_t time_stamp);

__global__ void initialize_rest_rng(ggxl::VectorField2D<curandState> *rng_states, int iFace, int64_t time_stamp, int dy,
  int dz, int ngg, int my, int mz);

template<MixtureModel mix_model, class turb>
__global__ void apply_symmetry(DZone *zone, int i_face, DParameter *param) {
  const auto &b = zone->boundary[i_face];
  const auto range_start = b.range_start, range_end = b.range_end;
  const int i = range_start[0] + static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = range_start[1] + static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = range_start[2] + static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  const auto face = b.face;
  int dir[]{0, 0, 0};
  dir[face] = b.direction;

  const int inner_idx[3]{i - dir[0], j - dir[1], k - dir[2]};

  auto metric = zone->metric(i, j, k);
  real k_x{metric(face + 1, 1)}, k_y{metric(face + 1, 2)}, k_z{metric(face + 1, 3)};
  const real k_magnitude = sqrt(k_x * k_x + k_y * k_y + k_z * k_z);
  k_x /= k_magnitude;
  k_y /= k_magnitude;
  k_z /= k_magnitude;

  auto &bv = zone->bv;
  const real u1{bv(inner_idx[0], inner_idx[1], inner_idx[2], 1)},
      v1{bv(inner_idx[0], inner_idx[1], inner_idx[2], 2)},
      w1{bv(inner_idx[0], inner_idx[1], inner_idx[2], 3)};
  real u_k{k_x * u1 + k_y * v1 + k_z * w1};
  const real u_t{u1 - k_x * u_k}, v_t{v1 - k_y * u_k}, w_t{w1 - k_z * u_k};

  // The gradient of tangential velocity should be zero.
  bv(i, j, k, 1) = u_t;
  bv(i, j, k, 2) = v_t;
  bv(i, j, k, 3) = w_t;
  // The gradient of pressure, density, and scalars should also be zero.
  bv(i, j, k, 0) = bv(inner_idx[0], inner_idx[1], inner_idx[2], 0);
  bv(i, j, k, 4) = bv(inner_idx[0], inner_idx[1], inner_idx[2], 4);
  bv(i, j, k, 5) = bv(inner_idx[0], inner_idx[1], inner_idx[2], 5);
  auto &sv = zone->sv;
  for (int l = 0; l < param->n_scalar; ++l) {
    sv(i, j, k, l) = sv(inner_idx[0], inner_idx[1], inner_idx[2], l);
  }

  // For ghost grids
  for (int g = 1; g <= zone->ngg; ++g) {
    const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
    const int ii{i - g * dir[0]}, ij{j - g * dir[1]}, ik{k - g * dir[2]};

    bv(gi, gj, gk, 0) = bv(ii, ij, ik, 0);

    const auto u{bv(ii, ij, ik, 1)}, v{bv(ii, ij, ik, 2)}, w{bv(ii, ij, ik, 3)};
    u_k = k_x * u + k_y * v + k_z * w;
    bv(gi, gj, gk, 1) = u - 2 * u_k * k_x;
    bv(gi, gj, gk, 2) = v - 2 * u_k * k_y;
    bv(gi, gj, gk, 3) = w - 2 * u_k * k_z;
    bv(gi, gj, gk, 4) = bv(ii, ij, ik, 4);
    bv(gi, gj, gk, 5) = bv(ii, ij, ik, 5);
    for (int l = 0; l < param->n_scalar; ++l) {
      sv(gi, gj, gk, l) = sv(ii, ij, ik, l);
    }

    if constexpr (TurbMethod<turb>::hasMut) {
      zone->mut(gi, gj, gk) = zone->mut(ii, ij, ik);
    }

    compute_cv_from_bv_1_point<mix_model, turb>(zone, param, gi, gj, gk);
  }
}

template<MixtureModel mix_model, class turb>
__global__ void apply_outflow(DZone *zone, int i_face, const DParameter *param) {
  const int ngg = zone->ngg;
  int dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  const auto range_start = b.range_start, range_end = b.range_end;
  const int i = range_start[0] + static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = range_start[1] + static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = range_start[2] + static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  for (int g = 1; g <= ngg; ++g) {
    const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
    for (int l = 0; l < 6; ++l) {
      bv(gi, gj, gk, l) = bv(i, j, k, l);
    }
    for (int l = 0; l < param->n_scalar; ++l) {
      sv(gi, gj, gk, l) = sv(i, j, k, l);
    }
    if constexpr (TurbMethod<turb>::hasMut) {
      zone->mut(gi, gj, gk) = zone->mut(i, j, k);
    }
    compute_cv_from_bv_1_point<mix_model, turb>(zone, param, gi, gj, gk);
  }
}

template<MixtureModel mix_model, class turb>
__global__ void apply_inflow(DZone *zone, Inflow *inflow, int i_face, DParameter *param,
  ggxl::VectorField3D<real> *profile_d_ptr, curandState *rng_states_d_ptr,
  ggxl::VectorField3D<real> *fluctuation_dPtr) {
  const int ngg = zone->ngg;
  int dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  auto range_start = b.range_start, range_end = b.range_end;
  int i = range_start[0] + static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  int j = range_start[1] + static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  int k = range_start[2] + static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  const int n_scalar = param->n_scalar;

  real density, u, v, w, p, T, mut, vel;
  real sv_b[MAX_SPEC_NUMBER + 4 + MAX_PASSIVE_SCALAR_NUMBER];

  if (inflow->inflow_type == 1) {
    // Profile inflow
    // For this type of inflow, the profile may specify the values for not only the inflow face, but also the ghost layers.
    // Therefore, all parts including fluctuations and assigning values to ghost layers are done in this function.
    // After all operations, we return directly.

    const auto &prof = profile_d_ptr[inflow->profile_idx];
    int idx[3] = {i, j, k};
    idx[b.face] = 0;
    // idx[b.face] = b.direction == 1 ? 0 : ngg;

    density = prof(idx[0], idx[1], idx[2], 0);
    u = prof(idx[0], idx[1], idx[2], 1);
    v = prof(idx[0], idx[1], idx[2], 2);
    w = prof(idx[0], idx[1], idx[2], 3);
    p = prof(idx[0], idx[1], idx[2], 4);
    T = prof(idx[0], idx[1], idx[2], 5);
    for (int l = 0; l < n_scalar; ++l) {
      sv_b[l] = prof(idx[0], idx[1], idx[2], 6 + l);
    }
    if constexpr (TurbMethod<turb>::hasMut) {
      mut = density * sv_b[param->n_spec] / sv_b[param->n_spec + 1];
    }
    vel = sqrt(u * u + v * v + w * w);

    real bv_fluc_real[6], bv_fluc_imag[6];
    real uf{0}, vf{0};
    if (inflow->fluctuation_type == 1) {
      // White noise fluctuation
      // We assume it obeying a N(0,rms^2) distribution
      // The fluctuation is added to the velocity
      auto y = zone->y(i, j, k);
      if (y < 4 * inflow->delta_omega && y > -4 * inflow->delta_omega) {
        auto index{0};
        switch (b.face) {
          case 1:
            index = (k + ngg) * (zone->mx + 2 * ngg) + (i + ngg);
            break;
          case 2:
            index = (j + ngg) * (zone->mx + 2 * ngg) + (i + ngg);
            break;
          case 0:
          default:
            index = (k + ngg) * (zone->my + 2 * ngg) + (j + ngg);
            break;
        }
        auto &rng_state = rng_states_d_ptr[index];

        real rms = inflow->fluctuation_intensity;

        uf = curand_normal_double(&rng_state) * rms * vel;
        vf = curand_normal_double(&rng_state) * rms * vel;
        u += uf;
        v += vf;
        vel = sqrt(u * u + v * v + w * w);
      }
    } else if (inflow->fluctuation_type == 2) {
      // LST fluctuation
      int idx_fluc[3]{i, j, k};
      idx_fluc[b.face] = 0;
      const auto &fluc_info = fluctuation_dPtr[inflow->fluc_prof_idx];

      // rho u v w p t
      bv_fluc_real[0] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 0);
      bv_fluc_real[1] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 1);
      bv_fluc_real[2] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 2);
      bv_fluc_real[3] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 3);
      bv_fluc_real[4] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 4);
      bv_fluc_real[5] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 5);
      bv_fluc_imag[0] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 6);
      bv_fluc_imag[1] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 7);
      bv_fluc_imag[2] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 8);
      bv_fluc_imag[3] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 9);
      bv_fluc_imag[4] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 10);
      bv_fluc_imag[5] = fluc_info(idx_fluc[0], idx_fluc[1], idx_fluc[2], 11);

      real x = zone->x(i, j, k), z = zone->z(i, j, k);

      real A0 = inflow->fluctuation_intensity;
      real omega = 2.0 * pi * inflow->fluctuation_frequency;
      real alpha = 2.0 * pi / inflow->streamwise_wavelength;
      real beta = 2.0 * pi / inflow->spanwise_wavelength;
      real t = param->physical_time;
      real phi = alpha * x - omega * t;
      density += A0 * (bv_fluc_real[0] * cos(phi) - bv_fluc_imag[0] * sin(phi)) * cos(beta * z) * param->rho_ref;
      u += A0 * (bv_fluc_real[1] * cos(phi) - bv_fluc_imag[1] * sin(phi)) * cos(beta * z) * param->v_ref;
      v += A0 * (bv_fluc_real[2] * cos(phi) - bv_fluc_imag[2] * sin(phi)) * cos(beta * z) * param->v_ref;
      w += A0 * (bv_fluc_real[3] * cos(phi) - bv_fluc_imag[3] * sin(phi)) * cos(beta * z) * param->v_ref;
      T += A0 * (bv_fluc_real[5] * cos(phi) - bv_fluc_imag[5] * sin(phi)) * cos(beta * z) * param->T_ref;
      p = density * R_u / mw_air * T;
    }

    // Specify the boundary value as given.
    bv(i, j, k, 0) = density;
    bv(i, j, k, 1) = u;
    bv(i, j, k, 2) = v;
    bv(i, j, k, 3) = w;
    bv(i, j, k, 4) = p;
    bv(i, j, k, 5) = T;
    for (int l = 0; l < n_scalar; ++l) {
      sv(i, j, k, l) = sv_b[l];
    }
    if constexpr (TurbMethod<turb>::hasMut) {
      zone->mut(i, j, k) = mut;
    }
    compute_cv_from_bv_1_point<mix_model, turb>(zone, param, i, j, k);

    // For ghost grids
    for (int g = 1; g <= ngg; g++) {
      const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
      idx[0] = gi, idx[1] = gj, idx[2] = gk;
      idx[b.face] = -g;
      // idx[b.face] = b.direction == 1 ? g : ngg - g;

      density = prof(idx[0], idx[1], idx[2], 0);
      u = prof(idx[0], idx[1], idx[2], 1) + uf;
      v = prof(idx[0], idx[1], idx[2], 2) + vf;
      w = prof(idx[0], idx[1], idx[2], 3);
      p = prof(idx[0], idx[1], idx[2], 4);
      T = prof(idx[0], idx[1], idx[2], 5);
      for (int l = 0; l < n_scalar; ++l) {
        sv_b[l] = prof(idx[0], idx[1], idx[2], 6 + l);
      }
      if constexpr (TurbMethod<turb>::hasMut) {
        mut = density * sv_b[param->n_spec] / sv_b[param->n_spec + 1];
      }
      vel = sqrt(u * u + v * v + w * w);

      if (inflow->fluctuation_type == 2) {
        // LST fluctuation
        // int idx_fluc[3]{i, j, k};
        // idx_fluc[b.face] = 0;

        real x = zone->x(gi, gj, gk), z = zone->z(gi, gj, gk);

        real A0 = inflow->fluctuation_intensity;
        real omega = 2.0 * pi * inflow->fluctuation_frequency;
        real alpha = 2.0 * pi / inflow->streamwise_wavelength;
        real beta = 2.0 * pi / inflow->spanwise_wavelength;
        real t = param->physical_time;
        real phi = alpha * x - omega * t;
        density += A0 * (bv_fluc_real[0] * cos(phi) - bv_fluc_imag[0] * sin(phi)) * cos(beta * z) * param->rho_ref;
        u += A0 * (bv_fluc_real[1] * cos(phi) - bv_fluc_imag[1] * sin(phi)) * cos(beta * z) * param->v_ref;
        v += A0 * (bv_fluc_real[2] * cos(phi) - bv_fluc_imag[2] * sin(phi)) * cos(beta * z) * param->v_ref;
        w += A0 * (bv_fluc_real[3] * cos(phi) - bv_fluc_imag[3] * sin(phi)) * cos(beta * z) * param->v_ref;
        T += A0 * (bv_fluc_real[5] * cos(phi) - bv_fluc_imag[5] * sin(phi)) * cos(beta * z) * param->T_ref;
        p = density * R_u / mw_air * T;
      }

      bv(gi, gj, gk, 0) = density;
      bv(gi, gj, gk, 1) = u;
      bv(gi, gj, gk, 2) = v;
      bv(gi, gj, gk, 3) = w;
      bv(gi, gj, gk, 4) = p;
      bv(gi, gj, gk, 5) = T;
      for (int l = 0; l < n_scalar; ++l) {
        sv(gi, gj, gk, l) = sv_b[l];
      }
      if constexpr (TurbMethod<turb>::hasMut) {
        zone->mut(gi, gj, gk) = mut;
      }
      compute_cv_from_bv_1_point<mix_model, turb>(zone, param, gi, gj, gk);
    }
    return;
  }

  if (inflow->inflow_type == 2) {
    // Mixing layer inflow
    const real u_upper = inflow->u, u_lower = inflow->u_lower;
    auto y = zone->y(i, j, k);
    u = 0.5 * (u_upper + u_lower) + 0.5 * (u_upper - u_lower) * tanh(2 * y / inflow->delta_omega);
    if (y >= 0) {
      density = inflow->density;
      v = inflow->v;
      w = inflow->w;
      p = inflow->pressure;
      T = inflow->temperature;
      for (int l = 0; l < n_scalar; ++l) {
        sv_b[l] = inflow->sv[l];
      }
      vel = sqrt(u * u + v * v + w * w);
      if constexpr (TurbMethod<turb>::hasMut) {
        mut = inflow->mut;
      }
    } else {
      // The lower stream
      density = inflow->density_lower;
      v = inflow->v_lower;
      w = inflow->w_lower;
      p = inflow->p_lower;
      T = inflow->t_lower;
      for (int l = 0; l < n_scalar; ++l) {
        sv_b[l] = inflow->sv_lower[l];
      }
      vel = sqrt(u * u + v * v + w * w);
      if constexpr (TurbMethod<turb>::hasMut) {
        mut = inflow->mut_lower;
      }
    }

    if (inflow->fluctuation_type == 1) {
      // White noise fluctuation
      // We assume it obeying a N(0,rms^2) distribution
      // The fluctuation is added to the velocity
      // Besides, we assume the range of fluctuation is restricted to 4*delta_omega ranges.
      if (y < 4 * inflow->delta_omega && y > -4 * inflow->delta_omega) {
        auto index{0};
        switch (b.face) {
          case 1:
            index = (k + ngg) * (zone->mx + 2 * ngg) + (i + ngg);
            break;
          case 2:
            index = (j + ngg) * (zone->mx + 2 * ngg) + (i + ngg);
            break;
          case 0:
          default:
            index = (k + ngg) * (zone->my + 2 * ngg) + (j + ngg);
            break;
        }
        auto &rng_state = rng_states_d_ptr[index];

        real rms = inflow->fluctuation_intensity;

        u += curand_normal_double(&rng_state) * rms * inflow->u;
        v += curand_normal_double(&rng_state) * rms * inflow->u;
        w += curand_normal_double(&rng_state) * rms * inflow->u;
        vel = sqrt(u * u + v * v + w * w);
      }
    }
  } else {
    // Constant inflow
    density = inflow->density;
    u = inflow->u;
    v = inflow->v;
    w = inflow->w;
    p = inflow->pressure;
    T = inflow->temperature;
    for (int l = 0; l < n_scalar; ++l) {
      sv_b[l] = inflow->sv[l];
    }

    if constexpr (TurbMethod<turb>::hasMut) {
      mut = inflow->mut;
    }
    vel = inflow->velocity;

    if (inflow->fluctuation_type == 1) {
      // White noise fluctuation
      // We assume it obeying a N(0,rms^2) distribution
      // The fluctuation is added to the velocity
      auto index{0};
      switch (b.face) {
        case 1:
          index = (k + ngg) * (zone->mx + 2 * ngg) + (i + ngg);
          break;
        case 2:
          index = (j + ngg) * (zone->mx + 2 * ngg) + (i + ngg);
          break;
        case 0:
        default:
          index = (k + ngg) * (zone->my + 2 * ngg) + (j + ngg);
          break;
      }
      auto &rng_state = rng_states_d_ptr[index];

      real rms = inflow->fluctuation_intensity;

      u += curand_normal_double(&rng_state) * rms * vel;
      vel = sqrt(u * u + v * v + w * w);
    } else if (inflow->fluctuation_type == 3) {
      //S mode waves with wide band frequencies and spanwise wavenumbers are induced
      real t = param->physical_time;

      real x = zone->x(i, j, k);
      for (int m = 0; m <= 198; ++m) {
        constexpr real delta_f = 5000;
        real f = delta_f * m + 10000;
        real alpha = 2.0 * pi * f / (1.0 - 1.0 / param->mach_ref) / param->v_ref;
        real Am = 0;
        if (f <= 40000) {
          constexpr real CL = 3.953e-4;
          Am = sqrt(CL / f * delta_f * 0.5) * param->p_ref;
        } else if (f > 40000) {
          constexpr real CU = 126.5e6;
          Am = sqrt(CU / pow(f, 3.5) * delta_f * 0.5) * param->p_ref;
        }
        const real disturbance = Am * cos(alpha * x - 2.0 * pi * f * t + inflow->random_phase[m]);
        p += disturbance;
        u -= disturbance * param->mach_ref / param->rho_ref / param->v_ref;
        v += 0;
        w += 0;
        T += disturbance * (gamma_air - 1.0) * param->mach_ref * param->mach_ref / param->rho_ref / param->v_ref /
            param->v_ref * param->T_ref;
        density = p * mw_air / (R_u * T);
      }
    }
  }

  // Specify the boundary value as given.
  bv(i, j, k, 0) = density;
  bv(i, j, k, 1) = u;
  bv(i, j, k, 2) = v;
  bv(i, j, k, 3) = w;
  bv(i, j, k, 4) = p;
  bv(i, j, k, 5) = T;
  for (int l = 0; l < n_scalar; ++l) {
    sv(i, j, k, l) = sv_b[l];
  }
  if constexpr (TurbMethod<turb>::hasMut) {
    zone->mut(i, j, k) = mut;
  }
  compute_cv_from_bv_1_point<mix_model, turb>(zone, param, i, j, k);
  for (int g = 1; g <= ngg; g++) {
    const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};

    if (inflow->fluctuation_type == 3) {
      //S mode waves with wide band frequencies and spanwise wavenumbers are induced
      real t = param->physical_time;

      real x = zone->x(gi, gj, gk);
      for (int m = 0; m <= 198; ++m) {
        constexpr real delta_f = 5000;
        real f = delta_f * m + 10000;
        real alpha = 2.0 * pi * f / (1.0 - 1.0 / param->mach_ref) / param->v_ref;
        real Am = 0;
        if (f <= 40000) {
          constexpr real CL = 3.953e-4;
          Am = sqrt(CL / f * delta_f * 0.5) * param->p_ref;
        } else if (f > 40000) {
          constexpr real CU = 126.5e6;
          Am = sqrt(CU / pow(f, 3.5) * delta_f * 0.5) * param->p_ref;
        }
        const real disturbance = Am * cos(alpha * x - 2.0 * pi * f * t + inflow->random_phase[m]);
        p = inflow->pressure + disturbance;
        u = inflow->u - disturbance * param->mach_ref / param->rho_ref / param->v_ref;
        v = inflow->v;
        w = inflow->w;
        T = inflow->temperature +
            disturbance * (gamma_air - 1.0) * param->mach_ref * param->mach_ref / param->rho_ref / param->v_ref /
            param->v_ref * param->T_ref;
        density = p * mw_air / (R_u * T);
      }
    }

    bv(gi, gj, gk, 0) = density;
    bv(gi, gj, gk, 1) = u;
    bv(gi, gj, gk, 2) = v;
    bv(gi, gj, gk, 3) = w;
    bv(gi, gj, gk, 4) = p;
    bv(gi, gj, gk, 5) = T;
    for (int l = 0; l < n_scalar; ++l) {
      sv(gi, gj, gk, l) = sv_b[l];
    }
    if constexpr (TurbMethod<turb>::hasMut) {
      zone->mut(gi, gj, gk) = mut;
    }
    compute_cv_from_bv_1_point<mix_model, turb>(zone, param, gi, gj, gk);
  }
}

template<MixtureModel mix_model, class turb>
__global__ void apply_inflow_df(DZone *zone, Inflow *inflow, DParameter *param,
  ggxl::VectorField3D<real> *fluctuation_dPtr, int df_iFace) {
  const int ngg = zone->ngg;
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - ngg;
  const int k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - ngg;
  if (j >= zone->my + ngg || k >= zone->mz + ngg) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  constexpr int i = 0;
  const real u_upper = inflow->u, u_lower = inflow->u_lower;
  const auto y = zone->y(i, j, k);
  real u = 0.5 * (u_upper + u_lower) + 0.5 * (u_upper - u_lower) * tanh(2 * y / inflow->delta_omega);
  const real u_fluc = fluctuation_dPtr[df_iFace](0, j, k, 0) * param->delta_u;
  const real v_fluc = fluctuation_dPtr[df_iFace](0, j, k, 1) * param->delta_u;
  const real w_fluc = fluctuation_dPtr[df_iFace](0, j, k, 2) * param->delta_u;
  const real mw = y > 0 ? inflow->mw : inflow->mw_lower;
  real T = y > 0 ? inflow->temperature : inflow->t_lower;
  real rho = y > 0 ? inflow->density : inflow->density_lower;
  real p = y > 0 ? inflow->pressure : inflow->p_lower;
  real c_p = 0;
  if constexpr (mix_model != MixtureModel::Air) {
    const real *sv_b = y > 0 ? inflow->sv : inflow->sv_lower;
    real cpl[MAX_SPEC_NUMBER];
    compute_cp(T, cpl, param);
    for (int l = 0; l < param->n_spec; ++l) {
      c_p += cpl[l] * sv_b[l];
    }
  } else {
    c_p = gamma_air * R_u / mw_air / (gamma_air - 1);
  }
  real t_fluc = -u_fluc * u / c_p; // SRA
  t_fluc *= 0.25;                  // StreamS multiplies a parameter "dftscaling=0.25" here
  const real rho_fluc = -t_fluc * rho / T;
  u += u_fluc;
  rho += rho_fluc;
  T += t_fluc;
  p = rho * R_u / mw * T;
  for (int ig = 0; ig <= ngg; ++ig) {
    const int gi{-ig};
    bv(gi, j, k, 0) = rho;
    bv(gi, j, k, 1) = u;
    bv(gi, j, k, 2) = v_fluc;
    bv(gi, j, k, 3) = w_fluc;
    bv(gi, j, k, 4) = p;
    bv(gi, j, k, 5) = T;
    const real *sv_b = y > 0 ? inflow->sv : inflow->sv_lower;
    for (int l = 0; l < param->n_scalar; ++l) {
      sv(gi, j, k, l) = sv_b[l];
    }
    if constexpr (TurbMethod<turb>::hasMut) {
      zone->mut(gi, j, k) = y > 0 ? inflow->mut : inflow->mut_lower;
    }
    compute_cv_from_bv_1_point<mix_model, turb>(zone, param, gi, j, k);
  }
  //  real dt_old = param->dt / 3.0;
  //  real x_convection = param->v_char * dt_old;
  //
  //  for (int ig = 0; ig < ngg; ++ig) {
  //    const int gi{-i}, gj{j}, gk{k};
  //
  //    // Here with the assumption of dx is directly computed without a necessity of considering the metrics
  //    real dQDx = x_convection / (zone->x(gi, gj, gk) - zone->x(gi - 1, gj, gk));
  //
  //    // Taylor's hypothesis
  //    for (int l = 0; l < param->n_var; ++l) {
  //      cv(gi, gj, gk, l) -= dQDx * (cv(gi, gj, gk, l) - cv(gi - 1, gj, gk, l));
  //    }
  //    if (param->dim == 2) {
  //      cv(gi, gj, gk, 3) = 0;
  //    }
  //
  //    bv(gi, gj, gk, 0) = cv(gi, gj, gk, 0);
  //    const real density_inv = 1 / bv(gi, gj, gk, 0);
  //    bv(gi, gj, gk, 1) = cv(gi, gj, gk, 1) * density_inv;
  //    bv(gi, gj, gk, 2) = cv(gi, gj, gk, 2) * density_inv;
  //    bv(gi, gj, gk, 3) = cv(gi, gj, gk, 3) * density_inv;
  //    auto v2 = bv(gi, gj, gk, 1) * bv(gi, gj, gk, 1) + bv(gi, gj, gk, 2) * bv(gi, gj, gk, 2) +
  //              bv(gi, gj, gk, 3) * bv(gi, gj, gk, 3);
  //    if constexpr (mix_model != MixtureModel::Air) {
  //      compute_temperature_and_pressure(gi, gj, gk, param, zone, cv(gi, gj, gk, 4));
  //    } else {
  //      // Air
  //      bv(gi, gj, gk, 4) = (gamma_air - 1) * (cv(gi, gj, gk, 4) - 0.5 * bv(gi, gj, gk, 0) * v2);
  //      bv(gi, gj, gk, 5) = bv(gi, gj, gk, 4) * mw_air * density_inv / R_u;
  //    }
  //  }
  //
  //  // The outermost ghost grid
  //  i = -ngg;
  //  real u_fluc = fluctuation_dPtr[df_iFace](0, j, k, 0) * param->delta_u;
  //  real v_fluc = fluctuation_dPtr[df_iFace](0, j, k, 1) * param->delta_u;
  //  real w_fluc = fluctuation_dPtr[df_iFace](0, j, k, 2) * param->delta_u;
  //  real gamma{gamma_air}, mw{mw_air};
  //  real T = y > 0 ? inflow->temperature : inflow->t_lower;
  //  real rho = y > 0 ? inflow->density : inflow->density_lower;
  //  real p = y > 0 ? inflow->pressure : inflow->p_lower;
  //  if constexpr (mix_model != MixtureModel::Air) {
  //    real cpl[MAX_SPEC_NUMBER];
  //    compute_cp(T, cpl, param);
  //    real c_p{0}, c_v{0};
  //    mw = 0.0;
  //    for (int l = 0; l < param->n_spec; ++l) {
  //      c_p += cpl[l] * sv(i, j, k, l);
  //      c_v += (cpl[l] - R_u / param->mw[l]) * sv(i, j, k, l);
  //      mw += sv(i, j, k, l) / param->mw[l];
  //    }
  //    gamma = c_p / c_v;
  //    mw = 1 / mw;
  //  }
  //  real t_fluc = -(gamma - 1) / gamma * u_fluc * u * mw / R_u; // SRA
  //  t_fluc *= 0.25; // StreamS multiplies a parameter "dftscaling=0.25" here
  //  real rho_fluc = -t_fluc * rho / T;
  //  u += u_fluc;
  //  rho += rho_fluc;
  //  T += t_fluc;
  //  p = rho * R_u / mw * T;
  //  bv(i, j, k, 0) = rho;
  //  bv(i, j, k, 1) = u;
  //  bv(i, j, k, 2) = v_fluc;
  //  bv(i, j, k, 3) = w_fluc;
  //  bv(i, j, k, 4) = p;
  //  bv(i, j, k, 5) = T;
  //  const real *sv_b = y > 0 ? inflow->sv : inflow->sv_lower;
  //  for (int l = 0; l < param->n_scalar; ++l) {
  //    sv(i, j, k, l) = sv_b[l];
  //  }
  //  if constexpr (TurbMethod<turb>::hasMut) {
  //    zone->mut(i, j, k) = y > 0 ? inflow->mut : inflow->mut_lower;
  //  }
  //  compute_cv_from_bv_1_point<mix_model, turb>(zone, param, i, j, k);
}

template<MixtureModel mix_model, class turb>
__global__ void apply_farfield(DZone *zone, FarField *farfield, int i_face, DParameter *param) {
  const int ngg = zone->ngg;
  int dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  auto range_start = b.range_start, range_end = b.range_end;
  int i = range_start[0] + static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  int j = range_start[1] + static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  int k = range_start[2] + static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;

  real nx{zone->metric(i, j, k)(b.face + 1, 1)},
      ny{zone->metric(i, j, k)(b.face + 1, 2)},
      nz{zone->metric(i, j, k)(b.face + 1, 3)};
  real grad_n_inv = b.direction / sqrt(nx * nx + ny * ny + nz * nz);
  nx *= grad_n_inv;
  ny *= grad_n_inv;
  nz *= grad_n_inv;
  const real u_b{bv(i, j, k, 1)}, v_b{bv(i, j, k, 2)}, w_b{bv(i, j, k, 3)};
  const real u_face{nx * u_b + ny * v_b + nz * w_b};

  // Interpolate the scalar values from internal nodes, which are used to compute gamma, after which, acoustic speed.
  const int n_scalar = param->n_scalar;
  auto &sv = zone->sv;
  real gamma_b{gamma_air}, mw{mw_air};
  real sv_b[MAX_SPEC_NUMBER + 2];
  if constexpr (mix_model != MixtureModel::Air) {
    for (int l = 0; l < n_scalar; ++l) {
      sv_b[l] = sv(i, j, k, l);
    }
    gamma_b = zone->gamma(i, j, k);
    real mw_inv{0};
    for (int l = 0; l < param->n_spec; ++l) {
      mw_inv += sv_b[l] / param->mw[l];
    }
    mw = 1.0 / mw_inv;
  }
  const real p_b{bv(i, j, k, 4)}, rho_b{bv(i, j, k, 0)};
  const real a_b{sqrt(gamma_b * p_b / rho_b)};
  const real mach_b{u_face / a_b};

  if (mach_b <= -1) {
    // supersonic inflow
    const real density = farfield->density;
    const real u = farfield->u;
    const real v = farfield->v;
    const real w = farfield->w;
    const auto *i_sv = farfield->sv;

    // Specify the boundary value as given.
    bv(i, j, k, 0) = density;
    bv(i, j, k, 1) = u;
    bv(i, j, k, 2) = v;
    bv(i, j, k, 3) = w;
    bv(i, j, k, 4) = farfield->pressure;
    bv(i, j, k, 5) = farfield->temperature;
    for (int l = 0; l < n_scalar; ++l) {
      sv(i, j, k, l) = i_sv[l];
    }
    if constexpr (TurbMethod<turb>::hasMut) {
      zone->mut(i, j, k) = farfield->mut;
    }
    compute_cv_from_bv_1_point<mix_model, turb>(zone, param, i, j, k);

    for (int g = 1; g <= ngg; g++) {
      const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
      bv(gi, gj, gk, 0) = density;
      bv(gi, gj, gk, 1) = u;
      bv(gi, gj, gk, 2) = v;
      bv(gi, gj, gk, 3) = w;
      bv(gi, gj, gk, 4) = farfield->pressure;
      bv(gi, gj, gk, 5) = farfield->temperature;
      for (int l = 0; l < n_scalar; ++l) {
        sv(gi, gj, gk, l) = i_sv[l];
      }
      if constexpr (TurbMethod<turb>::hasMut) {
        zone->mut(gi, gj, gk) = farfield->mut;
      }
      compute_cv_from_bv_1_point<mix_model, turb>(zone, param, gi, gj, gk);
    }
  } else if (mach_b >= 1) {
    // supersonic outflow
    for (int g = 1; g <= ngg; ++g) {
      const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
      for (int l = 0; l < 6; ++l) {
        bv(gi, gj, gk, l) = bv(i, j, k, l);
      }
      for (int l = 0; l < n_scalar; ++l) {
        sv(gi, gj, gk, l) = sv(i, j, k, l);
      }
      if constexpr (TurbMethod<turb>::hasMut) {
        zone->mut(gi, gj, gk) = zone->mut(i, j, k);
      }
      compute_cv_from_bv_1_point<mix_model, turb>(zone, param, gi, gj, gk);
    }
  } else {
    // Temporarily using the ACANS method, only for air
    //    if (mach_b <= 0) {
    //      // Only for subsonic inflow!!
    //      int i1 = i - dir[0], j1 = j - dir[1], k1 = k - dir[2];
    //      real rho1 = bv(i1, j1, k1, 0), p1 = bv(i1, j1, k1, 4);
    //      real c1 = sqrt(gamma_air * p1 / rho1);
    //      real pb = 0.5 * (p1 + farfield->pressure - rho1 * c1 * ((farfield->u - bv(i1, j1, k1, 1)) * nx +
    //                                                              (farfield->v - bv(i1, j1, k1, 2)) * ny +
    //                                                              (farfield->w - bv(i1, j1, k1, 3)) * nz));
    //      real db = farfield->density + gamma_air * (pb - farfield->pressure) / (c1 * c1);
    //      real ub = farfield->u + (pb - farfield->pressure) / (rho1 * c1) * nx;
    //      real vb = farfield->v + (pb - farfield->pressure) / (rho1 * c1) * ny;
    //      real wb = farfield->w + (pb - farfield->pressure) / (rho1 * c1) * nz;
    //
    //      // Specify the boundary value as given.
    //      bv(i, j, k, 0) = db;
    //      bv(i, j, k, 1) = ub;
    //      bv(i, j, k, 2) = vb;
    //      bv(i, j, k, 3) = wb;
    //      bv(i, j, k, 4) = pb;
    //      bv(i, j, k, 5) = pb / (db * R_u / mw_air);
    //      for (int l = 0; l < n_scalar; ++l) {
    //        sv(i, j, k, l) = farfield->sv[l];
    //      }
    //      if constexpr (TurbMethod<turb>::hasMut) {
    //        zone->mut(i, j, k) = farfield->mut;
    //      }
    //        compute_cv_from_bv_1_point<mix_model, turb>(zone, param, i, j, k);
    //
    //      for (int g = 1; g <= ngg; g++) {
    //        const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
    //        const int ii{i - g * dir[0]}, ij{j - g * dir[1]}, ik{k - g * dir[2]};
    //
    //        real d2 = 2 * db - bv(ii, ij, ik, 0);
    //        real p2 = 2 * pb - bv(ii, ij, ik, 4);
    //        if (p2 < 0.1 * pb) p2 = pb;
    //
    //        bv(gi, gj, gk, 0) = (d2 <= 0.1 * db ? db : d2);
    //        bv(gi, gj, gk, 1) = 2 * ub - bv(ii, ij, ik, 1);
    //        bv(gi, gj, gk, 2) = 2 * vb - bv(ii, ij, ik, 2);
    //        bv(gi, gj, gk, 3) = 2 * wb - bv(ii, ij, ik, 3);
    //        bv(gi, gj, gk, 4) = p2;
    //        bv(gi, gj, gk, 5) = p2 / (d2 * R_u / mw_air);
    //        for (int l = 0; l < n_scalar; ++l) {
    //          sv(gi, gj, gk, l) = farfield->sv[l];
    //        }
    //        if constexpr (TurbMethod<turb>::hasMut) {
    //          zone->mut(gi, gj, gk) = farfield->mut;
    //        }
    //          compute_cv_from_bv_1_point<mix_model, turb>(zone, param, gi, gj, gk);
    //      }
    //    }

    // subsonic inflow and outflow

    // The positive riemann invariant is in the same direction of the boundary normal, which points to the outside of the computational domain.
    // Thus, it is computed from the internal nodes.
    const real riemann_pos{u_face + 2 * a_b / (gamma_b - 1)};
    const real u_inf{farfield->u * nx + farfield->v * ny + farfield->w * nz};
    const real riemann_neg{u_inf - 2 * farfield->acoustic_speed / (farfield->specific_heat_ratio - 1)};

    real s_b, density, pressure, temperature, u, v, w, mut;
    const real Un{0.5 * (riemann_pos + riemann_neg)};
    if constexpr (mix_model == MixtureModel::Air) {
      const real c_b{0.25 * (gamma_air - 1) * (riemann_pos - riemann_neg)};
      if (mach_b <= 0) {
        // inflow
        s_b = farfield->entropy;
        u = farfield->u + (Un - u_inf) * nx;
        v = farfield->v + (Un - u_inf) * ny;
        w = farfield->w + (Un - u_inf) * nz;
        for (int l = 0; l < n_scalar; ++l) {
          sv_b[l] = farfield->sv[l];
        }
        if constexpr (TurbMethod<turb>::hasMut)
          mut = farfield->mut;
      } else {
        // outflow
        s_b = p_b / pow(rho_b, gamma_air);
        u = u_b + (Un - u_face) * nx;
        v = v_b + (Un - u_face) * ny;
        w = w_b + (Un - u_face) * nz;
        if constexpr (TurbMethod<turb>::hasMut)
          mut = zone->mut(i, j, k);
      }
      density = pow(c_b * c_b / (gamma_air * s_b), 1 / (gamma_air - 1));
      pressure = density * c_b * c_b / gamma_air;
      temperature = pressure * mw_air / (density * R_u);
    } else {
      // Mixture
      if (mach_b < 0) {
        // inflow
        u = farfield->u + (Un - u_inf) * nx;
        v = farfield->v + (Un - u_inf) * ny;
        w = farfield->w + (Un - u_inf) * nz;
        for (int l = 0; l < n_scalar; ++l) {
          sv_b[l] = farfield->sv[l];
        }
        mw = farfield->mw;
        if constexpr (TurbMethod<turb>::hasMut)
          mut = farfield->mut;
      } else {
        // outflow
        u = u_b + (Un - u_face) * nx;
        v = v_b + (Un - u_face) * ny;
        w = w_b + (Un - u_face) * nz;
        // When this is the outflow condition, the sv_b should be interpolated from internal points,
        // which has been computed above
        if constexpr (TurbMethod<turb>::hasMut)
          mut = zone->mut(i, j, k);
      }
      real gamma{gamma_air}, err{1}, gamma_last;
      while (err > 1e-4) {
        gamma_last = gamma;
        const real c_b{0.25 * (gamma - 1) * (riemann_pos - riemann_neg)};
        if (mach_b <= 0) {
          // inflow
          s_b = farfield->entropy;
        } else {
          // outflow
          s_b = p_b / pow(rho_b, gamma);
        }
        density = pow(c_b * c_b / (gamma * s_b), 1 / (gamma - 1));
        pressure = density * c_b * c_b / gamma;
        temperature = pressure * mw / (density * R_u);
        real cp[MAX_SPEC_NUMBER];
        compute_cp(temperature, cp, param);
        real cp_tot{0};
        for (int l = 0; l < param->n_spec; ++l) {
          cp_tot += cp[l] * sv_b[l];
        }
        gamma = cp_tot / (cp_tot - R_u / mw);
        err = abs(1 - gamma / gamma_last);
      }
    }

    // Specify the boundary value as given.
    bv(i, j, k, 0) = density;
    bv(i, j, k, 1) = u;
    bv(i, j, k, 2) = v;
    bv(i, j, k, 3) = w;
    bv(i, j, k, 4) = pressure;
    bv(i, j, k, 5) = temperature;
    for (int l = 0; l < n_scalar; ++l) {
      sv(i, j, k, l) = sv_b[l];
    }
    if constexpr (TurbMethod<turb>::hasMut) {
      zone->mut(i, j, k) = mut;
    }
    compute_cv_from_bv_1_point<mix_model, turb>(zone, param, i, j, k);

    for (int g = 1; g <= ngg; g++) {
      const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
      bv(gi, gj, gk, 0) = density;
      bv(gi, gj, gk, 1) = u;
      bv(gi, gj, gk, 2) = v;
      bv(gi, gj, gk, 3) = w;
      bv(gi, gj, gk, 4) = pressure;
      bv(gi, gj, gk, 5) = temperature;
      for (int l = 0; l < n_scalar; ++l) {
        sv(gi, gj, gk, l) = sv_b[l];
      }
      if constexpr (TurbMethod<turb>::hasMut) {
        zone->mut(gi, gj, gk) = mut;
      }
      compute_cv_from_bv_1_point<mix_model, turb>(zone, param, gi, gj, gk);
    }
  }
}

template<MixtureModel mix_model, class turb>
__global__ void
apply_wall(DZone *zone, Wall *wall, DParameter *param, int i_face, int step = -1) {
  const int ngg = zone->ngg;
  int dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  auto range_start = b.range_start, range_end = b.range_end;
  int i = range_start[0] + static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  int j = range_start[1] + static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  int k = range_start[2] + static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  real t_wall{bv(i, j, k, 5)};

  const int idx[]{i - dir[0], j - dir[1], k - dir[2]};
  if (wall->thermal_type == Wall::ThermalType::isothermal) {
    t_wall = wall->temperature;
  } else if (wall->thermal_type == Wall::ThermalType::adiabatic) {
    t_wall = bv(idx[0], idx[1], idx[2], 5);
  } else if (wall->thermal_type == Wall::ThermalType::equilibrium_radiation) {
    const real t_in{bv(idx[0], idx[1], idx[2], 5)};

    constexpr real cp{gamma_air * R_u / mw_air / (gamma_air - 1)};
    const real lambda = Sutherland(t_wall) * cp / param->Pr;

    const real heat_flux{lambda * (t_in - t_wall) / zone->wall_distance(idx[0], idx[1], idx[2])};
    if (heat_flux > 0) {
      const real coeff =
          stefan_boltzmann_constants * wall->emissivity * zone->wall_distance(idx[0], idx[1], idx[2]) * t_wall *
          t_wall * t_wall;
      t_wall -= (coeff * t_wall + lambda * t_wall - lambda * t_in) / (4 * coeff + lambda);
    } else {
      t_wall = wall->temperature;
    }
    if (t_wall < wall->temperature) {
      t_wall = wall->temperature;
    }
  }
  const real p{bv(idx[0], idx[1], idx[2], 4)};

  real mw{mw_air};
  if constexpr (mix_model != MixtureModel::Air) {
    // Mixture
    const auto mwk = param->mw;
    mw = 0;
    for (int l = 0; l < param->n_spec; ++l) {
      sv(i, j, k, l) = sv(idx[0], idx[1], idx[2], l);
      mw += sv(i, j, k, l) / mwk[l];
    }
    mw = 1 / mw;
  }
  int if_fluctuation = wall->fluctuation_type;

  const real rho_wall = p * mw / (t_wall * R_u);
  bv(i, j, k, 0) = rho_wall;
  bv(i, j, k, 1) = 0;
  bv(i, j, k, 2) = 0;
  bv(i, j, k, 3) = 0;
  bv(i, j, k, 4) = p;
  bv(i, j, k, 5) = t_wall;

  if (wall->if_blow_shock_wave && step >= 0 && step <= 50) {
    gxl::Matrix<real, 3, 3, 1> bdJin;
    real d1 = zone->metric(i, j, k)(2, 1);
    real d2 = zone->metric(i, j, k)(2, 2);
    real d3 = zone->metric(i, j, k)(2, 3);
    real kk = sqrt(d1 * d1 + d2 * d2 + d3 * d3);
    bdJin(1, 1) = d1 / kk;
    bdJin(1, 2) = d2 / kk;
    bdJin(1, 3) = d3 / kk;

    d1 = bdJin(1, 2) - bdJin(1, 3);
    d2 = bdJin(1, 3) - bdJin(1, 1);
    d3 = bdJin(1, 1) - bdJin(1, 2);
    kk = sqrt(d1 * d1 + d2 * d2 + d3 * d3);
    bdJin(2, 1) = d1 / kk;
    bdJin(2, 2) = d2 / kk;
    bdJin(2, 3) = d3 / kk;

    d1 = bdJin(1, 2) * bdJin(2, 3) - bdJin(1, 3) * bdJin(2, 2);
    d2 = bdJin(1, 3) * bdJin(2, 1) - bdJin(1, 1) * bdJin(2, 3);
    d3 = bdJin(1, 1) * bdJin(2, 2) - bdJin(1, 2) * bdJin(2, 1);
    kk = sqrt(d1 * d1 + d2 * d2 + d3 * d3);
    bdJin(3, 1) = d1 / kk;
    bdJin(3, 2) = d2 / kk;
    bdJin(3, 3) = d3 / kk;

    real u = bv(i - dir[0], j - dir[1], k - dir[2], 1),
        v = bv(i - dir[0], j - dir[1], k - dir[2], 2),
        w = bv(i - dir[0], j - dir[1], k - dir[2], 3);
    real vn = bdJin(1, 1) * u + bdJin(1, 2) * v + bdJin(1, 3) * w;
    real vt = bdJin(2, 1) * u + bdJin(2, 2) * v + bdJin(2, 3) * w;
    real vs = bdJin(3, 1) * u + bdJin(3, 2) * v + bdJin(3, 3) * w;

    bv(i, j, k, 1) = bdJin(2, 1) * vt + bdJin(3, 1) * vs - bdJin(1, 1) * vn;
    bv(i, j, k, 2) = bdJin(2, 2) * vt + bdJin(3, 2) * vs - bdJin(1, 2) * vn;
    bv(i, j, k, 3) = bdJin(2, 3) * vt + bdJin(3, 3) * vs - bdJin(1, 3) * vn;
  }

  real v_blow{0};
  if (if_fluctuation == 1) {
    // Pirozzoli & Li fluctuations
    real phil[10] = {0.03, 0.47, 0.43, 0.73, 0.86, 0.36, 0.96, 0.47, 0.36, 0.61};
    real phim[5] = {0.31, 0.05, 0.03, 0.72, 0.93};
    real A0 = wall->fluctuation_intensity;
    real x0 = wall->fluctuation_x0;
    real x1 = wall->fluctuation_x1;
    real t = param->physical_time;

    real x = zone->x(i, j, k), z = zone->z(i, j, k);
    real zmax = abs(zone->z(0, 0, zone->mz - 1) - zone->z(0, 0, 0));
    real theta = 2 * pi * (x - x0) / (x1 - x0);
    real fx = 4.0 * sin(theta) * (1.0 - cos(theta)) / sqrt(27.0);

    real gz = 0;
    for (int l = 0; l < 10; ++l) {
      gz += wall->Zl[l] * sin(2.0 * pi * (l + 1) * (z / zmax + phil[l]));
    }
    real ht = 0;
    for (int m = 0; m < 5; ++m) {
      ht += wall->Tm[m] * sin(2.0 * pi * (m + 1) * (wall->fluctuation_frequency * t + phim[m]));
      //      ht += wall->Tm[m] * sin((m + 1) * omega * t + 2.0 * pi * (m + 1) * phim[m]);
    }
    if (x > x0 && x < x1) {
      v_blow = A0 * param->v_ref * fx * gz * ht;
      bv(i, j, k, 2) = A0 * param->v_ref * fx * gz * ht;
    }
  } else if (if_fluctuation == 3) {
    real A0 = wall->fluctuation_intensity;
    real omega = 2.0 * pi * wall->fluctuation_frequency;
    real beta = 2.0 * pi / wall->spanwise_wavelength;
    real x0 = wall->fluctuation_x0;
    real x1 = wall->fluctuation_x1;
    real t = param->physical_time;

    real x_middle = (x0 + x1) * 0.5;
    real xi = 0;
    real x = zone->x(i, j, k), z = zone->z(i, j, k);
    if (x >= x0 && x <= x_middle) {
      xi = (x - x0) / (x_middle - x0);
      const real xi3 = xi * xi * xi;
      bv(i, j, k, 2) = A0 * (15.1875 * xi3 * xi * xi - 35.4375 * xi3 * xi + 20.25 * xi3) * cos(beta * z) *
                       sin(omega * t) / param->v_ref;
    } else if (x >= x_middle && x <= x1) {
      xi = (x1 - x) / (x1 - x_middle);
      const real xi3 = xi * xi * xi;
      bv(i, j, k, 2) = -A0 * (15.1875 * xi3 * xi * xi - 35.4375 * xi3 * xi + 20.25 * xi3) * cos(beta * z) *
                       sin(omega * t) / param->v_ref;
    }
  }

  // turbulent boundary condition
  if constexpr (TurbMethod<turb>::label == TurbMethodLabel::SST) {
    // SST
    real mu_wall;
    if constexpr (mix_model != MixtureModel::Air) {
      mu_wall = compute_viscosity(i, j, k, t_wall, mw, param, zone);
    } else {
      mu_wall = Sutherland(t_wall);
    }
    const real dy = zone->wall_distance(idx[0], idx[1], idx[2]);
    const int n_spec = param->n_spec;
    sv(i, j, k, n_spec) = 0;
    if (dy > 1e-25) {
      sv(i, j, k, n_spec + 1) = 60 * mu_wall / (rho_wall * sst::beta_1 * dy * dy);
    } else {
      sv(i, j, k, n_spec + 1) = sv(idx[0], idx[1], idx[2], n_spec + 1);
    }
  }

  if (param->n_ps > 0) {
    const int i_ps{param->i_ps};
    for (int l = 0; l < param->n_ps; l++) {
      sv(i, j, k, i_ps + l) = sv(idx[0], idx[1], idx[2], i_ps + l);
    }
  }

  compute_cv_from_bv_1_point<mix_model, turb>(zone, param, i, j, k);

  for (int g = 1; g <= ngg; ++g) {
    const int i_in[]{i - g * dir[0], j - g * dir[1], k - g * dir[2]};
    const int i_gh[]{i + g * dir[0], j + g * dir[1], k + g * dir[2]};

    const real p_i{bv(i_in[0], i_in[1], i_in[2], 4)};
    const real t_i{bv(i_in[0], i_in[1], i_in[2], 5)};

    double t_g{t_i};
    if (wall->thermal_type == Wall::ThermalType::isothermal ||
        wall->thermal_type == Wall::ThermalType::equilibrium_radiation) {
      t_g = 2 * t_wall - t_i;    // 0.5*(t_i+t_g)=t_w
      if (t_g <= 0.1 * t_wall) { // improve stability
        t_g = t_wall;
      }
    }

    if constexpr (mix_model != MixtureModel::Air) {
      const auto mwk = param->mw;
      mw = 0;
      for (int l = 0; l < param->n_spec; ++l) {
        // The mass fraction is given by a symmetry condition, is this reasonable?
        sv(i_gh[0], i_gh[1], i_gh[2], l) = sv(i_in[0], i_in[1], i_in[2], l);
        mw += sv(i_gh[0], i_gh[1], i_gh[2], l) / mwk[l];
      }
      mw = 1 / mw;
    }

    const real rho_g{p_i * mw / (t_g * R_u)};
    bv(i_gh[0], i_gh[1], i_gh[2], 0) = rho_g;
    bv(i_gh[0], i_gh[1], i_gh[2], 1) = -bv(i_in[0], i_in[1], i_in[2], 1);
    bv(i_gh[0], i_gh[1], i_gh[2], 2) = v_blow * 2 - bv(i_in[0], i_in[1], i_in[2], 2);
    bv(i_gh[0], i_gh[1], i_gh[2], 3) = -bv(i_in[0], i_in[1], i_in[2], 3);
    bv(i_gh[0], i_gh[1], i_gh[2], 4) = p_i;
    bv(i_gh[0], i_gh[1], i_gh[2], 5) = t_g;

    // turbulent boundary condition
    if constexpr (TurbMethod<turb>::label == TurbMethodLabel::SST) {
      // SST
      const int n_spec = param->n_spec;
      sv(i_gh[0], i_gh[1], i_gh[2], n_spec) = 0;
      sv(i_gh[0], i_gh[1], i_gh[2], n_spec + 1) = sv(i, j, k, n_spec + 1);
      zone->mut(i_gh[0], i_gh[1], i_gh[2]) = 0;
    }

    if (param->n_ps > 0) {
      const int i_ps{param->i_ps};
      for (int l = 0; l < param->n_ps; l++) {
        sv(i_gh[0], i_gh[1], i_gh[2], i_ps + l) = sv(i_in[0], i_in[1], i_in[2], i_ps + l);
      }
    }

    compute_cv_from_bv_1_point<mix_model, turb>(zone, param, i_gh[0], i_gh[1], i_gh[2]);
  }
}

template<MixtureModel mix_model, class turb>
__global__ void apply_subsonic_inflow(DZone *zone, SubsonicInflow *inflow, DParameter *param, int i_face) {
  const int ngg = zone->ngg;
  int dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  const auto range_start = b.range_start, range_end = b.range_end;
  const int i = range_start[0] + static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = range_start[1] + static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = range_start[2] + static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  // Compute the normal direction of the face. The direction is from the inside to the outside of the computational domain.
  real nx{zone->metric(i, j, k)(b.face + 1, 1)},
      ny{zone->metric(i, j, k)(b.face + 1, 2)},
      nz{zone->metric(i, j, k)(b.face + 1, 3)};
  const real grad_n_inv = b.direction / sqrt(nx * nx + ny * ny + nz * nz);
  nx *= grad_n_inv;
  ny *= grad_n_inv;
  nz *= grad_n_inv;
  const real u_face{nx * bv(i, j, k, 1) + ny * bv(i, j, k, 2) + nz * bv(i, j, k, 3)};
  // compute the negative Riemann invariant with computed boundary value.
  const real acoustic_speed{sqrt(gamma_air * bv(i, j, k, 4) / bv(i, j, k, 0))};
  const real riemann_neg{abs(u_face) - 2 * acoustic_speed / (gamma_air - 1)};
  // compute the total enthalpy of the inflow.
  const real hti{
    bv(i, j, k, 4) / bv(i, j, k, 0) * gamma_air / (gamma_air - 1) +
    0.5 * (bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) +
           bv(i, j, k, 3) * bv(i, j, k, 3))
  };
  constexpr real qa{1 + 2.0 / (gamma_air - 1)};
  const real qb{2 * riemann_neg};
  const real qc{(gamma_air - 1) * (0.5 * riemann_neg * riemann_neg - hti)};
  const real delta{qb * qb - 4 * qa * qc};
  real a_new{acoustic_speed};
  if (delta >= 0) {
    const real a_plus{(-qb + sqrt(delta)) / (2 * qa)};
    const real a_minus{(-qb - sqrt(delta)) / (2 * qa)};
    a_new = a_plus > a_minus ? a_plus : a_minus;
  }

  const real u_new{riemann_neg + 2 * a_new / (gamma_air - 1)};
  const real mach{u_new / a_new};
  const real pressure{
    inflow->total_pressure * pow(1 + 0.5 * (gamma_air - 1) * mach * mach, -gamma_air / (gamma_air - 1))
  };
  const real temperature{
    inflow->total_temperature * pow(pressure / inflow->total_pressure, (gamma_air - 1) / gamma_air)
  };
  const real density{pressure * mw_air / (temperature * R_u)};

  // assign values for ghost cells
  for (int g = 1; g <= ngg; g++) {
    const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
    bv(gi, gj, gk, 0) = density;
    bv(gi, gj, gk, 1) = u_new * inflow->u;
    bv(gi, gj, gk, 2) = u_new * inflow->v;
    bv(gi, gj, gk, 3) = u_new * inflow->w;
    bv(gi, gj, gk, 4) = pressure;
    bv(gi, gj, gk, 5) = temperature;

    const real u_bar{bv(gi, gj, gk, 1) * nx + bv(gi, gj, gk, 2) * ny + bv(gi, gj, gk, 3) * nz};
    const int n_scalar = param->n_scalar;
    if (u_bar > 0) {
      // The normal velocity points out of the domain, which means the value should be acquired from internal nodes.
      for (int l = 0; l < n_scalar; ++l) {
        sv(gi, gj, gk, l) = sv(i, j, k, l);
      }
    } else {
      // The normal velocity points into the domain, which means the value should be acquired from the boundary.
      for (int l = 0; l < n_scalar; ++l) {
        sv(gi, gj, gk, l) = inflow->sv[l];
      }
    }

    // In CFL3D, only the first ghost layer is assigned with the value on the boundary, and the rest are assigned with 0.
    if constexpr (TurbMethod<turb>::hasMut)
      zone->mut(gi, gj, gk) = zone->mut(i, j, k);

    compute_cv_from_bv_1_point<mix_model, turb>(zone, param, gi, gj, gk);
  }
}

template<MixtureModel mix_model, class turb>
__global__ void apply_back_pressure(DZone *zone, BackPressure *backPressure, DParameter *param, int i_face) {
  const int ngg = zone->ngg;
  int dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  const auto range_start = b.range_start, range_end = b.range_end;
  const int i = range_start[0] + static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = range_start[1] + static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = range_start[2] + static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  // The multi-species type is not implemented.
  real nx{zone->metric(i, j, k)(b.face + 1, 1)},
      ny{zone->metric(i, j, k)(b.face + 1, 2)},
      nz{zone->metric(i, j, k)(b.face + 1, 3)};
  const real grad_n_inv = b.direction / sqrt(nx * nx + ny * ny + nz * nz);
  nx *= grad_n_inv;
  ny *= grad_n_inv;
  nz *= grad_n_inv;
  real u_b{bv(i, j, k, 1)}, v_b{bv(i, j, k, 2)}, w_b{bv(i, j, k, 3)};
  const real u_face{nx * u_b + ny * v_b + nz * w_b};
  real p_b{bv(i, j, k, 4)}, rho_b{bv(i, j, k, 0)};
  const real a_b{sqrt(gamma_air * p_b / rho_b)};
  const real mach_b{abs(u_face / a_b)};

  if (mach_b < 1) {
    p_b = backPressure->pressure;
    const int i1 = i - dir[0], j1 = j - dir[1], k1 = k - dir[2];
    const real d1{bv(i1, j1, k1, 0)}, u1{bv(i1, j1, k1, 1)}, v1{bv(i1, j1, k1, 2)}, w1{bv(i1, j1, k1, 3)},
        p1{bv(i1, j1, k1, 4)};
    const real c1{sqrt(gamma_air * p1 / d1)};
    rho_b = d1 + (p_b - p1) / (c1 * c1);
    u_b = u1 + nx * (p1 - p_b) / (d1 * c1);
    v_b = v1 + ny * (p1 - p_b) / (d1 * c1);
    w_b = w1 + nz * (p1 - p_b) / (d1 * c1);

    for (int g = 1; g <= ngg; ++g) {
      const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
      const int ii{i - g * dir[0]}, ij{j - g * dir[1]}, ik{k - g * dir[2]};

      const real p_g{2 * p_b - bv(ii, ij, ik, 4)}, rho_g{2 * rho_b - bv(ii, ij, ik, 0)};
      const real u_g{2 * u_b - bv(ii, ij, ik, 1)}, v_g{2 * v_b - bv(ii, ij, ik, 2)},
          w_g{2 * w_b - bv(ii, ij, ik, 3)};

      bv(gi, gj, gk, 0) = rho_g;
      bv(gi, gj, gk, 1) = u_g;
      bv(gi, gj, gk, 2) = v_g;
      bv(gi, gj, gk, 3) = w_g;
      bv(gi, gj, gk, 4) = p_g;
      bv(gi, gj, gk, 5) = p_g / (rho_g * R_u / mw_air);
      for (int l = 0; l < param->n_scalar_transported; ++l) {
        sv(gi, gj, gk, l) = sv(i, j, k, l);
      }

      if constexpr (TurbMethod<turb>::hasMut) {
        zone->mut(gi, gj, gk) = zone->mut(i, j, k);
      }
      compute_cv_from_bv_1_point<mix_model, turb>(zone, param, gi, gj, gk);
    }
  } else {
    for (int g = 1; g <= ngg; ++g) {
      const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};

      bv(gi, gj, gk, 0) = rho_b;
      bv(gi, gj, gk, 1) = u_b;
      bv(gi, gj, gk, 2) = v_b;
      bv(gi, gj, gk, 3) = w_b;
      bv(gi, gj, gk, 4) = p_b;
      bv(gi, gj, gk, 5) = p_b / (rho_b * R_u / mw_air);
      for (int l = 0; l < param->n_scalar_transported; ++l) {
        sv(gi, gj, gk, l) = sv(i, j, k, l);
      }

      if constexpr (TurbMethod<turb>::hasMut) {
        zone->mut(gi, gj, gk) = zone->mut(i, j, k);
      }
      compute_cv_from_bv_1_point<mix_model, turb>(zone, param, gi, gj, gk);
    }
  }
}

template<MixtureModel mix_model, class turb>
__global__ void apply_periodic(DZone *zone, DParameter *param, int i_face) {
  const int ngg = zone->ngg;
  int dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  const auto range_start = b.range_start, range_end = b.range_end;
  const int i = range_start[0] + static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = range_start[1] + static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = range_start[2] + static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  int idx_other[3]{i, j, k};
  switch (b.face) {
    case 0: // i face
      idx_other[0] = b.direction < 0 ? zone->mx - 1 : 0;
      break;
    case 1: // j face
      idx_other[1] = b.direction < 0 ? zone->my - 1 : 0;
      break;
    case 2: // k face
    default:
      idx_other[2] = b.direction < 0 ? zone->mz - 1 : 0;
      break;
  }

  for (int g = 1; g <= ngg; ++g) {
    const int gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
    const int ii{idx_other[0] + g * dir[0]}, ij{idx_other[1] + g * dir[1]}, ik{idx_other[2] + g * dir[2]};
    for (int l = 0; l < 6; ++l) {
      bv(gi, gj, gk, l) = bv(ii, ij, ik, l);
    }
    for (int l = 0; l < param->n_scalar; ++l) {
      sv(gi, gj, gk, l) = sv(ii, ij, ik, l);
    }
    if constexpr (TurbMethod<turb>::hasMut) {
      zone->mut(gi, gj, gk) = zone->mut(ii, ij, ik);
    }
    compute_cv_from_bv_1_point<mix_model, turb>(zone, param, gi, gj, gk);
  }
}

template<MixtureModel mix_model, class turb>
void DBoundCond::apply_boundary_conditions(const Block &block, Field &field, DParameter *param, int step) const {
  // Boundary conditions are applied in the order of priority, which with higher priority is applied later.
  // Finally, the communication between faces will be carried out after these bc applied
  // Priority: (-1 - inner faces >) 2-wall > 3-symmetry > 5-inflow = 7-subsonic inflow > 6-outflow = 9-back pressure > 4-farfield

  // 4-farfield
  for (size_t l = 0; l < n_farfield; ++l) {
    const auto nb = farfield_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = farfield_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &h_f = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = h_f.range_end[j] - h_f.range_start[j] + 1;
        tpb[j] = n_point <= 2 * ngg + 1 ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_farfield<mix_model, turb> <<<BPG, TPB>>>(field.d_ptr, &farfield[l], i_face, param);
    }
  }

  // 6-outflow
  for (size_t l = 0; l < n_outflow; l++) {
    const auto nb = outflow_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = outflow_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &h_f = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = h_f.range_end[j] - h_f.range_start[j] + 1;
        tpb[j] = n_point <= 2 * ngg + 1 ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_outflow<mix_model, turb> <<<BPG, TPB>>>(field.d_ptr, i_face, param);
    }
  }
  for (size_t l = 0; l < n_back_pressure; l++) {
    const auto nb = back_pressure_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = back_pressure_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &h_f = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = h_f.range_end[j] - h_f.range_start[j] + 1;
        tpb[j] = n_point <= 2 * ngg + 1 ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_back_pressure<mix_model, turb> <<<BPG, TPB>>>(field.d_ptr, &back_pressure[l], param, i_face);
    }
  }

  // 5-inflow
  for (size_t l = 0; l < n_inflow; l++) {
    const auto nb = inflow_info[l].n_boundary;
    if (df_label[l] > -1) {
      // Here we assume only one face corresponds to the inflow boundary
      // nb should be 1, and the number of points should be my, mz of the corresponding block
      for (size_t i = 0; i < nb; i++) {
        auto [i_zone, i_face] = inflow_info[l].boundary[i];
        if (i_zone != block.block_id) {
          continue;
        }
        int my = block.my, mz = block.mz, ngg = block.ngg;
        generate_random_numbers(df_label[l], my, mz, ngg);
        apply_convolution(df_label[l], my, mz, ngg);
        compute_fluctuations(param, field.d_ptr, &inflow[l], df_label[l], my, mz, ngg);
        dim3 TPB{32, 8};
        dim3 BPG{(my + 2 * ngg - 1) / TPB.x + 1, (mz + 2 * ngg - 1) / TPB.y + 1};
        apply_inflow_df<mix_model, turb> <<<BPG, TPB>>>(field.d_ptr, &inflow[l], param, fluctuation_dPtr,
                                                        df_label[l]);
      }
    } else {
      for (size_t i = 0; i < nb; i++) {
        auto [i_zone, i_face] = inflow_info[l].boundary[i];
        if (i_zone != block.block_id) {
          continue;
        }
        const auto &hf = block.boundary[i_face];
        const auto ngg = block.ngg;
        uint tpb[3], bpg[3], n_point[3];
        for (size_t j = 0; j < 3; j++) {
          n_point[j] = hf.range_end[j] - hf.range_start[j] + 1;
          tpb[j] = n_point[j] <= 2 * ngg + 1 ? 1 : 16;
          bpg[j] = (n_point[j] - 1) / tpb[j] + 1;
        }
        dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
        apply_inflow<mix_model, turb> <<<BPG, TPB>>>(field.d_ptr, &inflow[l], i_face, param,
                                                     profile_dPtr_withGhost, rng_d_ptr, fluctuation_dPtr);
      }
    }
  }
  // 7 - subsonic inflow
  for (size_t l = 0; l < n_subsonic_inflow; l++) {
    const auto nb = subsonic_inflow_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = subsonic_inflow_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &hf = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = hf.range_end[j] - hf.range_start[j] + 1;
        tpb[j] = n_point <= 2 * ngg + 1 ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_subsonic_inflow<mix_model, turb> <<<BPG, TPB>>>(field.d_ptr, &subsonic_inflow[l], param, i_face);
    }
  }

  // 3-symmetry
  for (size_t l = 0; l < n_symmetry; l++) {
    const auto nb = symmetry_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = symmetry_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &h_f = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = h_f.range_end[j] - h_f.range_start[j] + 1;
        tpb[j] = n_point <= 2 * ngg + 1 ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_symmetry<mix_model, turb> <<<BPG, TPB>>>(field.d_ptr, i_face, param);
    }
  }

  // 2 - wall
  for (size_t l = 0; l < n_wall; l++) {
    const auto nb = wall_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = wall_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &hf = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = hf.range_end[j] - hf.range_start[j] + 1;
        tpb[j] = n_point <= 2 * ngg + 1 ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_wall<mix_model, turb><<<BPG, TPB>>>(field.d_ptr, &wall[l], param, i_face, step);
    }
  }

  // 8 - periodic
  for (size_t l = 0; l < n_periodic; l++) {
    const auto nb = periodic_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = periodic_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &hf = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = hf.range_end[j] - hf.range_start[j] + 1;
        tpb[j] = n_point <= 2 * ngg + 1 ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_periodic<mix_model, turb><<<BPG, TPB>>>(field.d_ptr, param, i_face);
    }
  }
}

template<MixtureModel mix_model, class turb>
__global__ void apply_outflow_nr(DZone *zone, int i_face, const DParameter *param) {
  const auto &b = zone->boundary[i_face];
  const auto range_start = b.range_start, range_end = b.range_end;
  const int i = range_start[0] + static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = range_start[1] + static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = range_start[2] + static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;

  const int face = b.face;
  real kx{zone->metric(i, j, k)(face + 1, 1)}, ky{zone->metric(i, j, k)(face + 1, 2)},
      kz{zone->metric(i, j, k)(face + 1, 3)};
  const real gradKInv = 1 / norm3d(kx, ky, kz);

  const real u = bv(i, j, k, 1), v = bv(i, j, k, 2), w = bv(i, j, k, 3);
  const real U = u * kx + v * ky + w * kz;
  const real a = sqrt(gamma_air * bv(i, j, k, 4) / bv(i, j, k, 0));
  const real U_face = U * gradKInv;
  if (abs(U_face / a) > 1) {
    // the supersonic outflow does not specify any information
    return;
  }

  kx *= gradKInv;
  ky *= gradKInv;
  kz *= gradKInv;

  // Let us temporarily assume the discretization is 4th-order.
  constexpr real cc[5] = {-25 / 12.0, 48 / 12.0, -36 / 12.0, 16 / 12.0, -3 / 12.0};
  int sgn = 1;
  const bool start_face = b.direction == -1;
  if (!start_face) {
    // The big number face
    sgn = -1;
  }

  real rho_k{0}, u_k{0}, v_k{0}, w_k{0}, p_k{0};
  if (face == 0) {
    // x direction
    for (int l = 0; l <= 4; ++l) {
      const real c = sgn * cc[l];
      rho_k += c * bv(i + sgn * l, j, k, 0);
      u_k += c * bv(i + sgn * l, j, k, 1);
      v_k += c * bv(i + sgn * l, j, k, 2);
      w_k += c * bv(i + sgn * l, j, k, 3);
      p_k += c * bv(i + sgn * l, j, k, 4);
    }
  } else if (face == 1) {
    // y direction
    for (int l = 0; l <= 4; ++l) {
      const real c = sgn * cc[l];
      rho_k += c * bv(i, j + sgn * l, k, 0);
      u_k += c * bv(i, j + sgn * l, k, 1);
      v_k += c * bv(i, j + sgn * l, k, 2);
      w_k += c * bv(i, j + sgn * l, k, 3);
      p_k += c * bv(i, j + sgn * l, k, 4);
    }
  } else if (face == 2) {
    // z direction
    for (int l = 0; l <= 4; ++l) {
      const real c = sgn * cc[l];
      rho_k += c * bv(i, j, k + sgn * l, 0);
      u_k += c * bv(i, j, k + sgn * l, 1);
      v_k += c * bv(i, j, k + sgn * l, 2);
      w_k += c * bv(i, j, k + sgn * l, 3);
      p_k += c * bv(i, j, k + sgn * l, 4);
    }
  }

  const real aGradK = a / gradKInv;
  real Lambda[5]{U - aGradK, U, U, U, U + aGradK};

  // if (param->L1_wave == 0) // zero-reflection
  if (start_face) {
    // Lambda[0] = min(Lambda[0], 0.0);
    // Lambda[1] = min(Lambda[1], 0.0);
    // Lambda[2] = min(Lambda[2], 0.0);
    // Lambda[3] = min(Lambda[3], 0.0);
    Lambda[4] = min(Lambda[4], 0.0);
  } else {
    Lambda[0] = max(Lambda[0], 0.0);
    // Lambda[1] = max(Lambda[1], 0.0);
    // Lambda[2] = max(Lambda[2], 0.0);
    // Lambda[3] = max(Lambda[3], 0.0);
    // Lambda[4] = max(Lambda[4], 0.0);
  }

  real L[5]{};
  const real a2 = a * a, rho = bv(i, j, k, 0);

  L[0] = Lambda[0] * (-rho * a * (kx * u_k + ky * v_k + kz * w_k) + p_k);
  L[1] = Lambda[1] * (kx * a2 * rho_k - kz * v_k + ky * w_k - kx * p_k);
  L[2] = Lambda[2] * (kz * a2 * rho_k - ky * u_k + kx * v_k - kz * p_k);
  L[3] = Lambda[3] * (-ky * a2 * rho_k - kz * u_k + kx * w_k + ky * p_k);
  L[4] = Lambda[4] * (rho * a * (kx * u_k + ky * v_k + kz * w_k) + p_k);

  real d[5]{};
  d[0] = 1 / a2 * (kx * L[1] + kz * L[2] - ky * L[3] + 0.5 * (L[0] + L[4]));
  d[1] = 0.5 * kx * (L[4] - L[0]) / (rho * a) - ky * L[2] - kz * L[3];
  d[2] = 0.5 * ky * (L[4] - L[0]) / (rho * a) - kz * L[1] + kx * L[2];
  d[3] = 0.5 * kz * (L[4] - L[0]) / (rho * a) + ky * L[1] + kx * L[3];
  d[4] = 0.5 * (L[0] + L[4]);

  const real minusJacInv = -zone->jac(i, j, k);
  auto &rhs = zone->dq;
  rhs(i, j, k, 0) = minusJacInv * d[0];
  rhs(i, j, k, 1) = minusJacInv * (u * d[0] + rho * d[1]);
  rhs(i, j, k, 2) = minusJacInv * (v * d[0] + rho * d[2]);
  rhs(i, j, k, 3) = minusJacInv * (w * d[0] + rho * d[3]);
  rhs(i, j, k, 4) = minusJacInv * (0.5 * (u * u + v * v + w * w) * d[0] + rho * u * d[1] + rho * v * d[2] + rho * w *
                                   d[3] + d[4] / (gamma_air - 1));
}

template<MixtureModel mix_model, class turb> void DBoundCond::nonReflectingBoundary(const Block &block, Field &field,
  DParameter *param) const {
  // Non-reflecting boundary condition
  // First, for outflow boundaries.
  for (size_t l = 0; l < n_outflow; l++) {
    const int label = outflow_info[l].label;
    if (!gxl::exists(nonReflectingBCs, label))
      continue;
    const auto nb = outflow_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = outflow_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &h_f = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        const auto n_point = h_f.range_end[j] - h_f.range_start[j] + 1;
        tpb[j] = n_point <= 2 * ngg + 1 ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_outflow_nr<mix_model, turb> <<<BPG, TPB>>>(field.d_ptr, i_face, param);
    }
  }
}
} // cfd
