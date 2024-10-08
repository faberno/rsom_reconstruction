#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

namespace nb = nanobind;
using namespace nb::literals;

#define PI_F 3.141592654f
#define M 32


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template<typename scalar>
__global__ void sensitivity_kernel(const scalar* __restrict__ grid,
                                   const scalar* __restrict__ sensor_points,
                                   scalar* __restrict__ histogram,
                                   scalar* __restrict__ sensitivity_field,
                                   scalar* __restrict__ signal,
                                   const scalar sos, const scalar period,
                                   const int n_x, const int n_z, const int n_sensor, const int n_bins,
                                   const int histDim_x, const int histDim_z, const int n_channels, const int n_signal,
                                   const int histBlockIdx_x, const int histBlockIdx_z) {

    int hist_x =  threadIdx.x + blockDim.x * blockIdx.x;
    int x = hist_x + histBlockIdx_x * histDim_x;

    int hist_z = threadIdx.z + blockDim.z * blockIdx.z;
    int z = hist_z + histBlockIdx_z * histDim_z;

    if (!(x < n_x && z < n_z)) // thread outside of voxel grid
        return;

    scalar pos_x;
    scalar pos_z;

    scalar index;
    scalar pos_sensor_x, pos_sensor_y, pos_sensor_z, diff_x, diff_z, dist, weight;
    scalar sq_dist;

    scalar min_index = 1e10f;
    scalar max_index = 0.0f;
    scalar min_sqdist = 1e10f;
    scalar max_sqdist = 0.0f;
    int index_int;

    pos_x = grid[x * n_z * 3 + 3 * z + 0];
    pos_z = grid[x * n_z * 3 + 3 * z + 2];
    for (int p = 0; p < n_sensor; p++){
        pos_sensor_x = sensor_points[p * 3];
        pos_sensor_y = sensor_points[p * 3 + 1];
        pos_sensor_z = sensor_points[p * 3 + 2];
        diff_x = pos_x - pos_sensor_x;
        diff_z = pos_z - pos_sensor_z;

        sq_dist = diff_x * diff_x + pos_sensor_y * pos_sensor_y + diff_z * diff_z;
        min_sqdist = min(sq_dist, min_sqdist);
        max_sqdist = max(sq_dist, max_sqdist);
    }
    min_index = floorf(sqrt(min_sqdist) / sos / period);
    max_index = ceilf(sqrt(max_sqdist) / sos / period);

    for (int p = 0; p < n_sensor; p++){
        pos_sensor_x = sensor_points[p * 3];
        pos_sensor_y = sensor_points[p * 3 + 1];
        pos_sensor_z = sensor_points[p * 3 + 2];
        diff_x = pos_x - pos_sensor_x;
        diff_z = pos_z - pos_sensor_z;

        dist = sqrt(diff_x * diff_x + pos_sensor_y * pos_sensor_y + diff_z * diff_z);
        index = dist / sos / period - min_index;
        index_int = (int) index;

        weight  = 1.0f / (2.0f * PI_F * dist);
        histogram[hist_x * n_bins * histDim_z + n_bins * hist_z + index_int] += (ceilf(index) - index) * weight;
        histogram[hist_x * n_bins * histDim_z + n_bins * hist_z + index_int + 1] += (index - floorf(index)) * weight;
    }

    int n_indices = (int) (max_index - min_index) + 1;

    scalar minimum;
    scalar maximum;
    scalar conv;

    for (int c = 0; c < n_channels; c++){
        minimum = 1e16f;
        maximum = -1e16f;

        for (int i = (1 - n_signal); i < n_indices; i++){
            conv = 0.0f;
            for (int j = 0; j < n_signal; j++){
                if (((i + j) >= 0) && (i + j) < n_indices)
                    conv += (histogram[hist_x * n_bins * histDim_z + n_bins * hist_z + j + i] * signal[c * n_signal + j]);
            }
            minimum = min(minimum, conv);
            maximum = max(maximum, conv);
        }
        sensitivity_field[c * n_z * n_x + z + x * n_z] = maximum - minimum;
    }
}

template<typename scalar>
void calc_sensitivity(
        const nb::ndarray<scalar, nb::shape<-1, -1, 3>, nb::device::cuda, nb::c_contig> grid,
        const nb::ndarray<scalar, nb::shape<-1, 3>, nb::device::cuda, nb::c_contig> sensor_points,
        nb::ndarray<scalar, nb::ndim<3>, nb::device::cuda, nb::c_contig> histogram,
        nb::ndarray<scalar, nb::ndim<3>, nb::device::cuda, nb::c_contig> sensitivity_field,
        nb::ndarray<scalar, nb::ndim<2>, nb::device::cuda, nb::c_contig> signal,
        scalar sos,
        scalar period){

    const int n_x = grid.shape(0);
    const int n_z = grid.shape(1);
    const int n_sensor = sensor_points.shape(0);
    const int hist_x = histogram.shape(0);
    const int hist_z = histogram.shape(1);
    const int n_bins = histogram.shape(2);
    const int n_channels = signal.shape(0);
    const int n_signal = signal.shape(1);

    assert(n_channels == (sensitivity_field.shape(0)));

    const dim3 threads(M, 1, M);
    const dim3 blocks_cuda((hist_x + threads.x - 1) / threads.x,
                           1,
                           (hist_z + threads.z - 1) / threads.z);
    const dim3 blocks_iter((n_x + hist_x - 1) / hist_x,
                            1,
                            (n_z + hist_z - 1) / hist_z);

    std::chrono::steady_clock::time_point time_start = std::chrono::steady_clock::now();
    for (int i_x = 0; i_x < blocks_iter.x; i_x++){
        for (int i_z = 0; i_z < blocks_iter.z; i_z++){
            sensitivity_kernel<<<blocks_cuda, threads>>>(
                grid.data(),
                sensor_points.data(),
                histogram.data(),
                sensitivity_field.data(),
                signal.data(),
                sos, period,
                n_x, n_z, n_sensor, n_bins,
                hist_x, hist_z, n_channels, n_signal,
                i_x, i_z
            );
            cudaMemset(histogram.data(), 0.0f, (n_bins * hist_x * hist_z) *sizeof(scalar));  // reset histogram
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
        }
    }

    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
    std::cout << "Finished! Kernel Time: " << (float) std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() / 1000 << "s" << std::endl;
}


NB_MODULE(sensitivity_cuda, m) {
    m.def("calc_sensitivity", &calc_sensitivity<float>, "Create Sensitivity Field");
}
