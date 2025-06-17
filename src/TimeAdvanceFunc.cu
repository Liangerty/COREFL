#include "TimeAdvanceFunc.cuh"
#include "Field.h"
#include <mpi.h>
#include "gxl_lib/MyAtomic.cuh"

__global__ void cfd::store_last_step(DZone *zone) {
  const int mx{zone->mx}, my{zone->my}, mz{zone->mz};
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= mx || j >= my || k >= mz) return;

  auto &bv = zone->bv;
  zone->bv_last(i, j, k, 0) = bv(i, j, k, 0);
  zone->bv_last(i, j, k, 1) = sqrt(
    bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3));
  zone->bv_last(i, j, k, 2) = bv(i, j, k, 4);
  zone->bv_last(i, j, k, 3) = bv(i, j, k, 5);
}

__global__ void cfd::compute_square_of_dbv(DZone *zone) {
  const int mx{zone->mx}, my{zone->my}, mz{zone->mz};
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int j = static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y);
  const int k = static_cast<int>(blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= mx || j >= my || k >= mz) return;

  auto &bv = zone->bv;
  auto &bv_last = zone->bv_last;

  bv_last(i, j, k, 0) = (bv(i, j, k, 0) - bv_last(i, j, k, 0)) * (bv(i, j, k, 0) - bv_last(i, j, k, 0));
  const real vel = sqrt(
    bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3));
  bv_last(i, j, k, 1) = (vel - bv_last(i, j, k, 1)) * (vel - bv_last(i, j, k, 1));
  bv_last(i, j, k, 2) = (bv(i, j, k, 4) - bv_last(i, j, k, 2)) * (bv(i, j, k, 4) - bv_last(i, j, k, 2));
  bv_last(i, j, k, 3) = (bv(i, j, k, 5) - bv_last(i, j, k, 3)) * (bv(i, j, k, 5) - bv_last(i, j, k, 3));
}

real cfd::global_time_step(const Mesh &mesh, const Parameter &parameter, const std::vector<Field> &field) {
  real dt{1e+6};

  constexpr int TPB{128};
  real dt_block;
  int num_sms, num_blocks_per_sm;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, min_of_arr, TPB, 0);
  for (int b = 0; b < mesh.n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    const int size = mx * my * mz;
    int n_blocks = std::min(num_blocks_per_sm * num_sms, (size + TPB - 1) / TPB);
    min_of_arr<<<n_blocks, TPB>>>(field[b].h_ptr->dt_local.data(), size); //, TPB * sizeof(real)
    min_of_arr<<<1, TPB>>>(field[b].h_ptr->dt_local.data(), n_blocks);    //, TPB * sizeof(real)
    cudaMemcpy(&dt_block, field[b].h_ptr->dt_local.data(), sizeof(real), cudaMemcpyDeviceToHost);
    dt = std::min(dt, dt_block);
  }

  if (parameter.get_bool("parallel")) {
    // Parallel reduction
    const real dt_temp{dt};
    MPI_Allreduce(&dt_temp, &dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  }

  return dt;
}

__global__ void cfd::update_physical_time(DParameter *param, real t) {
  param->physical_time = t;
}

__global__ void cfd::min_of_arr(real *arr, int size) {
  const int i = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);
  const int t = static_cast<int>(threadIdx.x);

  if (i >= size) {
    return;
  }
  real inp{1e+6};
  for (int idx = i; idx < size; idx += static_cast<int>(blockDim.x * gridDim.x)) {
    inp = min(inp, arr[idx]);
  }
  __syncthreads();

  inp = block_reduce_min(inp, i, size);
  __syncthreads();

  if (t == 0) {
    arr[blockIdx.x] = inp;
  }
}
