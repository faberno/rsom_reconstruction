#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

namespace nb = nanobind;
using namespace nb::literals;

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
__global__ void saft_kernel(
    const scalar* __restrict__ signal,
    const scalar* __restrict__ sensor_pos,
    const scalar* __restrict__ voxel_pos_x,
    const scalar* __restrict__ voxel_pos_y,
    const scalar* __restrict__ voxel_pos_z,
    const scalar* __restrict__ sensitivity,
    const scalar* __restrict__ sensitivity_x,
    const scalar* __restrict__ sensitivity_z,
    const scalar* __restrict__ sensitivity_width,
    scalar* __restrict__ recon,
    const int n_x, const int n_y, const int n_z, const int n_x_sens,
    const int n_sensor, const int n_samples, const int n_channels,
    const scalar t_0, const scalar dt) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (!(x < n_x && y < n_y && z < n_z)) // thread outside of voxel grid
        return;

    int channelsize = n_x * n_y * n_z;
    int vox_index = z + n_z * y + n_z * n_y * x;

    scalar dx, dy, dz, hyper_a, hyper_b, delay, delay_floor, r_sq, r, r_floor;
    int delay_int, r_int;
    scalar lower_value, upper_value, value, lower_sensitivity_value, upper_sensitivity_value, sensitivity_value;

    scalar position_x = voxel_pos_x[x];
    scalar position_y = voxel_pos_y[y];
    scalar position_z = voxel_pos_z[z];

    scalar sfield_width;
    bool delay_calculated;
    bool r_calculated;

    scalar all_sensor_sum[3] = {0.0};
    scalar weight_sum[3] = {0.0};

    #pragma unroll
    for (int s = 0; s < n_sensor; s ++){
        dx = position_x - sensor_pos[2 * s];  // x-distance between voxel and sensor
        dy = position_y - sensor_pos[2 * s + 1];  // y-distance between voxel and sensor
        r_sq = dx * dx + dy * dy;  // squared x-y distance between voxel and sensor

        delay_calculated = false;

        for (int c = 0; c < n_channels; c++){
            sfield_width = sensitivity_width[c * n_z + z];
            if (r_sq > (sfield_width * sfield_width)){
                continue;
            }

            if (!delay_calculated){
                delay = sqrt(r_sq + position_z * position_z);
                delay = (copysign(delay, position_z) - t_0) / dt;
                if (delay < 0 || delay >= n_samples){ // if delay of voxel not available, go to next sensor
                    break;
                }
                delay_calculated = true;

                r = sqrt(r_sq) * 1000;  // get actual distance and convert to µm for sensitivity field index
                delay_floor = floor(delay);
                r_floor = floor(r);
                delay_int = (int) delay;
                r_int = (int) r;
            }

            lower_sensitivity_value = sensitivity[(c * n_x_sens * n_z) + (z * n_x_sens) + (r_int)];
            upper_sensitivity_value = sensitivity[(c * n_x_sens * n_z) + (z * n_x_sens) + (r_int + 1)];
            sensitivity_value = lower_sensitivity_value * (r_floor + 1 - r) + upper_sensitivity_value * (r - r_floor);

            lower_value = signal[(c * n_samples * n_sensor) + (s * n_samples) + (delay_int)];
            upper_value = signal[(c * n_samples * n_sensor) + (s * n_samples) + (delay_int + 1)];
            value = lower_value * (delay_floor + 1 - delay) + upper_value * (delay - delay_floor);  // linear interpolation

            all_sensor_sum[c] += value * sensitivity_value;
            weight_sum[c] += sensitivity_value;
        }
    }

    for (int c = 0; c < n_channels; c++){
        if (weight_sum[c] > 0.0){
            recon[vox_index + channelsize * c] = all_sensor_sum[c] / weight_sum[c]; // normalize
        }
    }

}


template<typename scalar>
void run_saft(
        const nb::ndarray<scalar, nb::ndim<3>, nb::device::cuda, nb::c_contig> signal,
        const nb::ndarray<scalar, nb::shape<-1, 2>, nb::device::cuda, nb::c_contig> sensor_pos,
        const nb::ndarray<scalar, nb::ndim<1>, nb::device::cuda, nb::c_contig> voxel_pos_x,
        const nb::ndarray<scalar, nb::ndim<1>, nb::device::cuda, nb::c_contig> voxel_pos_y,
        const nb::ndarray<scalar, nb::ndim<1>, nb::device::cuda, nb::c_contig> voxel_pos_z,
        const nb::ndarray<scalar, nb::ndim<3>, nb::device::cuda, nb::c_contig> sensitivity,
        const nb::ndarray<scalar, nb::ndim<1>, nb::device::cuda, nb::c_contig> sensitivity_x,
        const nb::ndarray<scalar, nb::ndim<1>, nb::device::cuda, nb::c_contig> sensitivity_z,
        const nb::ndarray<scalar, nb::ndim<2>, nb::device::cuda, nb::c_contig> sensitivity_width,
        const float t_0,
        const float dt,
        nb::ndarray<scalar, nb::ndim<4>, nb::device::cuda, nb::c_contig> recon,
        const bool verbose){

    const int n_x = voxel_pos_x.shape(0);
    const int n_y = voxel_pos_y.shape(0);
    const int n_z = voxel_pos_z.shape(0);
    const int n_x_sens = sensitivity.shape(2);
    const int n_sensor = sensor_pos.shape(0);
    const int n_samples = signal.shape(2);
    const int n_channels = signal.shape(0);

    assert(n_sensor == (signal.shape(0)));
    assert(n_z == (sensitivity.shape(0)));
    assert(n_channels == sensitivity.shape(0));
    assert(((int) sensitivity.shape(1)) == (sensitivity_z.shape(0)));
    assert(((int) sensitivity.shape(2)) == (sensitivity_x.shape(0)));
    assert(((int) sensitivity.shape(1)) == (sensitivity_width.shape(1)));

    const dim3 threads(8, 8, 8);
    const dim3 blocks((n_x + threads.x - 1) / threads.x,
                      (n_y + threads.y - 1) / threads.y,
                      (n_z + threads.z - 1) / threads.z);

    if (verbose){
        std::cout << "--------- RSOM Recontruction ----------" << std::endl;
        std::cout << "Signal: " << n_sensor << "x" << n_samples << std::endl;
        std::cout << "Volume: " << n_channels << "x" << n_x << "x" << n_y << "x"<< n_z << std::endl;
        std::cout << "Sensitivity Field: " << n_channels << sensitivity.shape(1) << "x" << n_x_sens << std::endl;
        std::cout << "Blocks: " << blocks.x << ", " << blocks.y << ", "<< blocks.z << std::endl;
        std::cout << "---------------------------------------" << std::endl;
    }

    std::chrono::steady_clock::time_point time_start = std::chrono::steady_clock::now();
    saft_kernel<scalar><<<blocks, threads>>>(
        signal.data(),
        sensor_pos.data(),
        voxel_pos_x.data(),
        voxel_pos_y.data(),
        voxel_pos_z.data(),
        sensitivity.data(),
        sensitivity_x.data(),
        sensitivity_z.data(),
        sensitivity_width.data(),
        recon.data(),
        n_x, n_y, n_z, n_x_sens, n_sensor, n_samples, n_channels,
        t_0, dt);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();

    if (verbose)
        std::cout << "Finished! Kernel Time: " << (float) std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() / 1000 << "s" << std::endl;
}


NB_MODULE(saft_cuda, m) {
    m.def("run_saft", &run_saft<float>, "Start the SAFT reconstruction");
}
