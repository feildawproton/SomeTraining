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

    
	// -- ASSIGNMENT --
	int n_gpus;
	cudaGetDeviceCount(&n_gpus);
	int n_streams = 32;
	
	// -- CREATE MULTIPLE STREAMS --
	cudaStream_t streams[n_gpus][n_streams];			//2D array containing the streams for each gpu
	for (int gpu = 0; gpu < n_gpus; gpu++)
	{
		cudaSetDevice(gpu);					//set gpu
		for(int stream = 0; stream < n_streams; stream++)
		{
			cudaStreamCreate(&streams[gpu][stream]);	//create and store streams
		}
	}
	check_last_error();
	
	const unsigned int gpu_chunk_size	= sdiv(num_entries, n_gpus);		//round up division
	const unsigned int stream_chunk_size	= sdiv(gpu_chunk_size, n_streams);
	
	const unsigned int chunk_size		= sdiv(num_entries, n_gpus);	
	uint64_t *data_gpu[n_gpus];							//array of pointers to 
	
	// -- MULTI-GPU DATA MALLOC --
	// memory is not allocated with streams
	for(unsigned gpu = 0; gpu < n_gpus; gpu++)
	{
		cudaSetDevice(gpu);
		
		const unsigned int lower = gpu_chunk_size * gpu;
		const unsigned int upper = min((unsigned int)(lower + chunk_size), (unsigned int)num_entries);
		const unsigned int width = upper - lower;
		
		cudaMalloc(&data_gpu[gpu], sizeof(uint64_t) * width);
	}
    	check_last_error();
	
	timer.start();
	// -- MULTI-GPU AND MULTI STREAM --
	for(unsigned gpu = 0; gpu < n_gpus; gpu++)						//loop over gpus
	{
		cudaSetDevice(gpu);								//set active gpu
		
		for (int stream = 0; stream < n_streams; stream++)				//loop through streams
		{
			const uint64_t stream_offset = stream_chunk_size * stream;		//calculate offset within the gpu data
			
			const uint64_t lower = gpu_chunk_size * gpu + stream_offset;		//lower bounds within the cpu data
			const uint64_t upper = min(lower + stream_chunk_size, num_entries);	//
			const uint64_t width = upper - lower;
			
			// -- COPY HOST TO DEVICE -- 
			cudaMemcpyAsync(data_gpu[gpu] + stream_offset, data_cpu + lower, sizeof(uint64_t) * width, cudaMemcpyHostToDevice, streams[gpu][stream]);
			
			// -- LAUNCH KERNELS -- 
			decrypt_gpu <<< 80*32, 64, 0, streams[gpu][stream] >>> (data_gpu[gpu] + stream_offset, width, num_iters);
			
			// -- COPY BACK --
			cudaMemcpyAsync(data_cpu + lower, data_gpu[gpu] + stream_offset, sizeof(uint64_t) * width, cudaMemcpyDeviceToHost, streams[gpu][stream]);
		}
	}
	check_last_error();
	timer.stop("total time on GPUs");		
	

	const bool success = check_result_cpu(data_cpu, num_entries, openmp);
	std::cout << "STATUS: test " << ( success ? "passed" : "failed") << std::endl;

	cudaFreeHost(data_cpu);
	// -- MULTI-GPU DATA FREE --
	for(unsigned gpu = 0; gpu < n_gpus; gpu++)
	{
		cudaSetDevice(gpu);
		cudaFree(data_gpu[gpu]);
	}
    	check_last_error();
	check_last_error();
}
