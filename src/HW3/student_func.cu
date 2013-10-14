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

template <typename T>
struct IBinary_Operator
{
public:
	__device__ virtual T operator() (const T& a, const T& b)=0;
};

template <typename T>
struct Max_Operator: public IBinary_Operator<T>
{
public:
	__device__ virtual T operator() (const T&a, const T& b){
		T result;
		result =  (a>=b)? a : b;
		return result;
	}
};

template <typename T>
struct Min_Operator: public IBinary_Operator<T>
{
public:
	__device__ virtual T operator() (const T&a, const T& b){
		T result;
		result =  (a<=b)? a : b;
		return result;
	}
};

template <typename T, typename T_Bin_Op>
__global__ void _reduction_global_mem_sub_(	T * d_out,
											T * d_in,
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

template <typename T, typename T_Bin_Op>
void reduction_global_mem(	const T* const d_in,
							const size_t numElement,
							T* d_out,
							T_Bin_Op operation)
{
	/*Two step reduction*/

	/*problem scale*/
    const int maxThreadsPerBlock = 1024;
    int threads = maxThreadsPerBlock;
    int blocks = (numElement - 1)/ maxThreadsPerBlock + 1;

	/*allocate  result*/
	T* d_input_array;

	checkCudaErrors(cudaMalloc(&d_input_array,    sizeof(T) * numElement));
	checkCudaErrors(cudaMemcpy(d_input_array,   d_in,   sizeof(T) * numElement, cudaMemcpyDeviceToDevice));

	T* d_intermediate_result;
	checkCudaErrors(cudaMalloc(&d_intermediate_result,    sizeof(T) * blocks));


	/*First, compute local reduction per each thread block*/

    _reduction_global_mem_sub_<T><<<blocks, threads>>>(d_intermediate_result, d_input_array, numElement, operation);

	/*Secondly, compute global reduction*/
    threads = blocks;
    blocks = 1;

    _reduction_global_mem_sub_<T><<<blocks, threads>>>(d_out, d_intermediate_result, threads, operation);

    checkCudaErrors(cudaFree(d_input_array));

}


template <typename T, typename T_Bin_Op>
void reduction(	const T* const d_in,
				const size_t numElement,
				T * h_out,
				T_Bin_Op operation)

{
	/*allocate d_out*/
	T * d_out;

	checkCudaErrors(cudaMalloc(&d_out,    sizeof(T) * 1));

	reduction_global_mem<T, T_Bin_Op>(d_in, numElement, d_out, operation);

	/*copy device output to host*/
	checkCudaErrors(cudaMemcpy(h_out,   d_out,   sizeof(T) * 1, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_out));
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


	reduction<float>(d_logLuminance, numRows * numCols, &min_logLum, Min_Operator<float>());
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	reduction<float>(d_logLuminance, numRows * numCols, &max_logLum, Max_Operator<float>());
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}
