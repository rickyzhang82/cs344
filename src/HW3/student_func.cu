/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <cfloat>

//template specialization for shared memory
// non-specialized class template
template <class T>
class SharedMem
{
public:
    // Ensure that we won't compile any un-specialized types
    __device__ T* getPointer() { return NULL; };
};

// specialization for int
template <>
class SharedMem <int>
{
public:
    __device__ int* getPointer() { extern __shared__ int s_int[]; return s_int; }
};

// specialization for float
template <>
class SharedMem <float>
{
public:
    __device__ float* getPointer() { extern __shared__ float s_float[]; return s_float; }
};


template <typename T>
struct IBinary_Operator
{
public:
    __device__ __host__ virtual T operator() (const T& a, const T& b)=0;
};

template <typename T>
struct Max_Operator: public IBinary_Operator<T>
{
public:
    __device__ __host__ virtual T operator() (const T&a, const T& b){
        T result;
        if(isnan(a))
            return b;
        if(isnan(b))
            return a;
        result =  (a<b)? b : a;
        return result;
    }
};

template <typename T>
struct Min_Operator: public IBinary_Operator<T>
{
public:
    __device__ __host__ virtual T operator() (const T&a, const T& b){
        T result;
        if(isnan(a))
            return b;
        if(isnan(b))
            return a;
        result =  (a>b)? b : a;
        return result;
    }
};

template <typename T>
struct Add_Operator: public IBinary_Operator<T>
{
public:
    __device__ __host__ virtual T operator() (const T&a, const T& b){

        return a + b;
    }
};

template <typename T, typename T_Bin_Op>
__global__ void _reduction_global_mem_sub_( T * d_out,
                                            T *  d_in,
                                            const size_t totalCount,
                                            T_Bin_Op operation)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    if(myId >= totalCount)
        return;
    // do reduction in global memory

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if(myId +s < totalCount)
                d_in[myId] = operation(d_in[myId], d_in[myId + s]);
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global memory
    if (tid == 0)
        d_out[blockIdx.x] = d_in[myId];


}

/*wrapper function for global memory*/
template <typename T, typename T_Bin_Op>
void reduction_global_mem(  const T* const d_in,
                            const size_t numElement,
                            T* d_out,
                            T_Bin_Op operation)
{
    /*Two passes reduction*/

    /*problem scale*/
    const int maxThreadsPerBlock = 1024;
    int threads = maxThreadsPerBlock;
    int blocks = (numElement - 1)/ maxThreadsPerBlock + 1;

    /*clone d_in --> d_input_array*/
    T* d_input_array;

    checkCudaErrors(cudaMalloc(&d_input_array,    sizeof(T) * numElement));
    checkCudaErrors(cudaMemcpy(d_input_array,   d_in,   sizeof(T) * numElement, cudaMemcpyDeviceToDevice));

    /*allocate intermediate result for first pass*/
    T* d_intermediate_result;
    checkCudaErrors(cudaMalloc(&d_intermediate_result,    sizeof(T) * blocks));

    /*On first pass, compute local reduction per each thread block*/
    _reduction_global_mem_sub_<T,T_Bin_Op><<<blocks, threads>>>(d_intermediate_result, d_input_array, numElement, operation);

    /*On second pass, compute global reduction*/
    _reduction_global_mem_sub_<T,T_Bin_Op><<<1, threads>>>(d_out, d_intermediate_result, blocks, operation);

    checkCudaErrors(cudaFree(d_input_array));

}

template <typename T, typename T_Bin_Op>
__global__ void _reduction_shared_mem_sub_( T * d_out,
                                            const T * const d_in,
                                            const size_t totalCount,
                                            T_Bin_Op operation)
{
    /*halve the number of threads in one block*/
    int myId = threadIdx.x + 2 * blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    if(myId >= totalCount)
        return;

    SharedMem<T> sharedmem;

    T* s_array = sharedmem.getPointer();

    /*reduce directly in first step*/
    int gap = blockDim.x;
    if(myId + gap < totalCount)
        s_array[tid] = operation(d_in[myId], d_in[myId + gap]);
    else
        s_array[tid] = d_in[myId];

    /*synch all threads within block*/
    __syncthreads();


    // do reduction in shared memory

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if(myId +s < totalCount)
                s_array[tid] = operation(s_array[tid], s_array[tid + s]);
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global memory
    if (tid == 0)
        d_out[blockIdx.x] = s_array[0];


}

/*wrapper function for shared memory*/
template <typename T, typename T_Bin_Op>
void reduction_shared_mem(  const T* const d_in,
                            const size_t numElement,
                            T* d_out,
                            T_Bin_Op operation)
{
    /*Two passes reduction*/

    /*problem scale*/
    const int maxThreadsPerBlock = 1024;
    int threads = maxThreadsPerBlock;
    int blocks = (numElement - 1)/ maxThreadsPerBlock + 1;
    T* d_intermediate_result;
    T* d_input_array = (T*) d_in;
    size_t input_array_element = numElement;

    do{
        /*allocate intermediate result for first pass*/
        checkCudaErrors(cudaMalloc(&d_intermediate_result,    sizeof(T) * blocks));

        /*On first pass, compute local reduction per each thread block*/
        _reduction_shared_mem_sub_<T, T_Bin_Op><<<blocks, threads / 2, sizeof(T) * threads / 2>>>(d_intermediate_result, d_input_array, input_array_element, operation);

        if(d_input_array != d_in)
            checkCudaErrors(cudaFree(d_input_array));

        /*On second pass, compute global reduction.
         *If the number of element of intermediate result can fit into one thread block, run final reduction.*/
        if(blocks <= maxThreadsPerBlock)

            _reduction_shared_mem_sub_<T, T_Bin_Op><<<1, threads / 2, sizeof(T) * threads / 2>>>(d_out, d_intermediate_result, blocks, operation);

        else{
            /*Otherwise, repeat first pass until it fits in one thread block.*/
            input_array_element = blocks;

            blocks = (blocks - 1)/ maxThreadsPerBlock + 1;

            d_input_array = d_intermediate_result;

        }

    }while(blocks > maxThreadsPerBlock);
}

template <typename T, typename T_Bin_Op>
void reduction( const T* const d_in,
                const size_t numElement,
                T * h_out,
                T_Bin_Op operation)

{
    /*allocate d_out*/
    T * d_out;

    checkCudaErrors(cudaMalloc(&d_out,    sizeof(T) * 1));

    //reduction_global_mem<T, T_Bin_Op>(d_in, numElement, d_out, operation);

    reduction_shared_mem<T, T_Bin_Op>(d_in, numElement, d_out, operation);

    /*copy device output to host*/
    checkCudaErrors(cudaMemcpy(h_out,   d_out,   sizeof(T) * 1, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_out));
}

template <typename T>
__global__ void _histogram_atomic_version_(	const T* const d_in,
											int* d_out,
											T min_logLum,
											T lumRange,
											const size_t numElement,
											const size_t numBins)
{
	int tid = threadIdx.x;

	int myId = blockDim.x * blockIdx.x + tid;

	if(myId > numElement)
		return;

	int binIndex = min( (int)floor((d_in[myId] - min_logLum) / lumRange * numBins), (int)numBins - 1 );

	atomicAdd(&(d_out[binIndex]), 1);
}

template <typename T>
__global__ void _histogram_atomic_free_version_(	const T* const d_in,
													int* d_out,
													T min_logLum,
													T lumRange,
													const size_t numElement,
													const size_t numBins,
													const int totalThreads,
													const int elementPerThread)
{
	int tid = threadIdx.x;

	int myId = blockDim.x * blockIdx.x + tid;

	int input_index;

	int binIndex;

	for(int i = 0; i < elementPerThread ; i++){

		input_index = (totalThreads * i) + myId;

		if(input_index < numElement){

			binIndex = min( (int)floor((d_in[input_index] - min_logLum) / lumRange * numBins), (int)numBins - 1 );

			*(d_out + totalThreads * binIndex + myId)= *(d_out + totalThreads * binIndex + myId) + 1;
		}
	}
}

template <typename T>
void histogram( const T* const d_in,
				size_t numElements,
                size_t numBins,
                T min_logLum,
                T max_logLum,
                int* d_out)
{
	if(1){
	int threads = 512;

    int blocks = (numElements - 1) / threads + 1;

    T lumRange = max_logLum - min_logLum;

    _histogram_atomic_version_<T> <<<blocks, threads>>> (d_in, d_out, min_logLum, lumRange, numElements, numBins);
	}
    /*testing atomic free version histogram*/
    /*too much overhead. It slows things down.*/
    if(0){
	int threads = 32;

	int elementsPerThread = 512;

	int blocks = numElements / (threads * elementsPerThread);

	int totla_threads = threads * blocks;

	int*  d_intermediate_hist;

	float lumRange = max_logLum - min_logLum;

	checkCudaErrors(cudaMalloc(&d_intermediate_hist, sizeof(int) * numBins * totla_threads));
	
    checkCudaErrors(cudaMemset(d_intermediate_hist, 0, sizeof(int) * numBins * totla_threads));

	_histogram_atomic_free_version_<T> <<<blocks, threads>>>(d_in, d_intermediate_hist, min_logLum, lumRange, numElements, numBins, totla_threads, elementsPerThread);

	int* h_bin = (int*) malloc(sizeof(int)*numBins);

	/*reduce bin*/
	for(int bin_index = 0 ; bin_index < numBins; bin_index++)

		   reduction< int, Add_Operator<int> >(d_intermediate_hist + bin_index * totla_threads, totla_threads, h_bin + bin_index, Add_Operator<int>());

	//copy to device memory
    checkCudaErrors(cudaMemcpy(d_out,   h_bin,   sizeof(int) * numBins, cudaMemcpyHostToDevice));

	free(h_bin);
    }
}
void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

	int * d_hist;
	/*allocate histogram output in device memory*/
	checkCudaErrors(cudaMalloc(&d_hist, sizeof(int) * numBins));
	checkCudaErrors(cudaMemset(d_hist, 0, sizeof(int) * numBins));

    reduction< float,Min_Operator<float> >(d_logLuminance, numRows * numCols, &min_logLum, Min_Operator<float>());

    reduction< float,Max_Operator<float> >(d_logLuminance, numRows * numCols, &max_logLum, Max_Operator<float>());

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    histogram< float > (d_logLuminance, numRows * numCols, numBins, min_logLum, max_logLum, d_hist);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //debug
    if(0)
    {
    int* h_hist;
    h_hist = (int*) malloc(sizeof(int) * numBins);
    checkCudaErrors(cudaMemcpy(h_hist,   d_hist,   sizeof(int) * numBins, cudaMemcpyDeviceToHost));
    for(int i=0;i<numBins;i++)
    	std::cout<<h_hist[i]<<" ";
    }

    //exclusive_scan...
    checkCudaErrors(cudaFree(d_hist));
}
