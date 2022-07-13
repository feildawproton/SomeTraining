#include <cstdint>
#include <iostream>
#include "helpers.cuh"
#include "encryption.cuh"

void encrypt_cpu(uint64_t * data, uint64_t num_entries, 
                 uint64_t num_iters, bool parallel=true) {

    #pragma omp parallel for if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        data[entry] = permute64(entry, num_iters);
}

__global__ 
void decrypt_gpu(uint64_t * data, uint64_t num_entries, uint64_t num_iters) {

    const uint64_t thrdID = blockIdx.x*blockDim.x+threadIdx.x;
    const uint64_t stride = blockDim.x*gridDim.x;

    for (uint64_t entry = thrdID; entry < num_entries; entry += stride)
        data[entry] = unpermute64(data[entry], num_iters);
}

bool check_result_cpu(uint64_t * data, uint64_t num_entries,
                      bool parallel=true) {

    uint64_t counter = 0;

    #pragma omp parallel for reduction(+: counter) if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        counter += data[entry] == entry;

    return counter == num_entries;
}

int main (int argc, char * argv[]) {

	const char * encrypted_file = "/dli/task/encrypted";

	Timer timer;

	const uint64_t num_entries = 1UL << 26;
	const uint64_t num_iters = 1UL << 10;
	const bool openmp = true;
	
	uint64_t * data_cpu;
	cudaMallocHost(&data_cpu, sizeof(uint64_t)*num_entries);
	check_last_error();

	if (!encrypted_file_exists(encrypted_file)) 
	{
		encrypt_cpu(data_cpu, num_entries, num_iters, openmp);
		write_encrypted_to_file(encrypted_file, data_cpu, sizeof(uint64_t)*num_entries);
	} 
	else 
	{
		read_encrypted_from_file(encrypted_file, data_cpu, sizeof(uint64_t)*num_entries);
	}

    
	//assignment
	int n_gpus;
	cudaGetDeviceCount(&n_gpus);
	
	const unsigned int chunk_size = sdiv(num_entries, n_gpus);	//round up division
	uint64_t *data_gpu[n_gpus];					//array of pointers to 
	
	// -- MULTI-GPU DATA MALLOC --
	for(unsigned gpu = 0; gpu < n_gpus; gpu++)
	{
		cudaSetDevice(gpu);
		
		const unsigned int lower = chunk_size * gpu;
		const unsigned int upper = min((unsigned int)(lower + chunk_size), (unsigned int)num_entries);
		const unsigned int width = upper - lower;
		
		cudaMalloc(&data_gpu[gpu], sizeof(uint64_t) * width);
	}
	//cudaMalloc    (&data_gpu, sizeof(uint64_t)*num_entries);
    	check_last_error();
	
	// -- MULTI-GPU DATA COPY TO GPU --
	for(unsigned gpu = 0; gpu < n_gpus; gpu++)
	{
		cudaSetDevice(gpu);
		
		const unsigned int lower = chunk_size * gpu;
		const unsigned int upper = min((unsigned int)(lower + chunk_size), (unsigned int)num_entries);
		const unsigned int width = upper - lower;

		cudaMemcpy(data_gpu[gpu], data_cpu + lower, sizeof(uint64_t) * width, cudaMemcpyHostToDevice);
	}
	//cudaMemcpy(data_gpu, data_cpu, sizeof(uint64_t)*num_entries, cudaMemcpyHostToDevice);
	check_last_error();

	// -- MULTI-GPU KERNEL LAUNCH --
	timer.start();
	for(unsigned gpu = 0; gpu < n_gpus; gpu++)
	{
		cudaSetDevice(gpu);
		
		const unsigned int lower = chunk_size * gpu;
		const unsigned int upper = min((unsigned int)(lower + chunk_size), (unsigned int)num_entries);
		const unsigned int width = upper - lower;

		decrypt_gpu<<<80*32, 64>>>(data_gpu[gpu], width, num_iters);
	}
	//decrypt_gpu<<<80*32, 64>>>(data_gpu, num_entries, num_iters);
	timer.stop("total kernel execution on GPUs");		
	check_last_error();
	
	// -- MULTI-GPU DATA COPY TO GPU --
	for(unsigned gpu = 0; gpu < n_gpus; gpu++)
	{
		cudaSetDevice(gpu);
		
		const unsigned int lower = chunk_size * gpu;
		const unsigned int upper = min((unsigned int)(lower + chunk_size), (unsigned int)num_entries);
		const unsigned int width = upper - lower;

		cudaMemcpy(data_cpu + lower, data_gpu[gpu], sizeof(uint64_t) * width, cudaMemcpyDeviceToHost);
	}
	//cudaMemcpy(data_cpu, data_gpu, sizeof(uint64_t)*num_entries, cudaMemcpyDeviceToHost);
	check_last_error();

	const bool success = check_result_cpu(data_cpu, num_entries, openmp);
	std::cout << "STATUS: test " << ( success ? "passed" : "failed") << std::endl;

	cudaFreeHost(data_cpu);
	// -- MULTI-GPU DATA FREE --
	for(unsigned gpu = 0; gpu < n_gpus; gpu++)
	{
		cudaSetDevice(gpu);
		cudaFree(data_gpu[gpu]);
	}
	//cudaFree    (data_gpu);
    	check_last_error();
	check_last_error();
}
