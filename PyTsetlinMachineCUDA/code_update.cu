

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CLASSES 1
#define CLAUSES 1
#define FEATURES 1
#define STATE_BITS 1
#define BOOST_TRUE_POSITIVE_FEEDBACK 1
#define S 1
#define THRESHOLD 1
#define Q 1

#define NEGATIVE_CLAUSES 1

#define PATCHES 1

#define NUMBER_OF_EXAMPLES 1

#include <curand_kernel.h>
#define INT_SIZE 32

#define LA_CHUNKS (((FEATURES-1)/INT_SIZE + 1))
#define CLAUSE_CHUNKS ((CLAUSES-1)/INT_SIZE + 1)

#if (FEATURES % 32 != 0)
#define FILTER (~(0xffffffff << (FEATURES % INT_SIZE)))
#else
#define FILTER 0xffffffff
#endif



extern "C"
{
	// Increment the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
	__device__ inline void inc(unsigned int *ta_state, int clause, int chunk, unsigned int active)
	{
		unsigned int carry, carry_next;
		int id = clause*LA_CHUNKS*STATE_BITS + chunk*STATE_BITS;
		carry = active;
		for (int b = 0; b < STATE_BITS; ++b) {
			if (carry == 0)
				break;

			carry_next = ta_state[id + b] & carry; // Sets carry bits (overflow) passing on to next bit
			ta_state[id + b] = ta_state[id + b] ^ carry; // Performs increments with XOR
			carry = carry_next;
		}

		if (carry > 0) {
			for (int b = 0; b < STATE_BITS; ++b) {
				ta_state[id + b] |= carry;
			}
		}   
	}

	// Decrement the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
	__device__ inline void dec(unsigned int *ta_state, int clause, int chunk, unsigned int active)
	{
		unsigned int carry, carry_next;
		int id = clause*LA_CHUNKS*STATE_BITS + chunk*STATE_BITS;
		carry = active;
		for (int b = 0; b < STATE_BITS; ++b) {
			if (carry == 0)
				break;
			carry_next = (~ta_state[id + b]) & carry; // Sets carry bits (overflow) passing on to next bit
			ta_state[id + b] = ta_state[id + b] ^ carry; // Performs increments with XOR
			carry = carry_next;
		}

		if (carry > 0) {
			for (int b = 0; b < STATE_BITS; ++b) {
				ta_state[id + b] &= ~carry;
			}
		} 
	}

	__device__ inline void calculate_positive_clause_output(curandState *localState, unsigned int *ta_state, unsigned int *clause_output, int *clause_patch, unsigned int *drop_feature, int *X)
	{
		int output_one_patches[PATCHES];
		int output_one_patches_count;

		// Evaluate each patch (convolution)
		output_one_patches_count = 0;
		for (int patch = 0; patch < PATCHES; ++patch) {
			int patch_clause_output = 1;
			for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
				if ((ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] & (drop_feature[la_chunk] | X[patch*LA_CHUNKS + la_chunk])) != ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]) {
					patch_clause_output = 0;
					break;
				}
			}

			if (((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & (drop_feature[LA_CHUNKS-1] | X[patch*LA_CHUNKS + LA_CHUNKS - 1]) & FILTER) != (ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER))) {
				patch_clause_output = 0;
			}

			if (patch_clause_output) {
				output_one_patches[output_one_patches_count] = patch;
				output_one_patches_count++;
			}
		}
	
		if (output_one_patches_count > 0) {
			*clause_output = 1;
			int patch_id = curand(localState) % output_one_patches_count;
			*clause_patch = output_one_patches[patch_id];
		} else {
			*clause_output = 0;
			*clause_patch = -1;
		}
	}

	__device__ inline void calculate_negative_clause_output(curandState *localState, unsigned int *ta_state, unsigned int *clause_output, int *clause_patch, unsigned int *drop_feature, int *X)
	{
		int output_one_patches[PATCHES];
		int output_one_patches_count;
		int patch_clause_output;

		// Evaluate each patch (convolution)
		output_one_patches_count = 0;
		for (int patch = 0; patch < PATCHES; ++patch) {
			patch_clause_output = 1;
			for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
				//printf(" %d checking \\n", ((ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] & (drop_feature[la_chunk] | X[patch*LA_CHUNKS + la_chunk])) != ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]));
				if ((ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] & (drop_feature[la_chunk] | X[patch*LA_CHUNKS + la_chunk])) != ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]) {
					patch_clause_output = 0;
					break;
				}
			}

			if (((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & (drop_feature[LA_CHUNKS-1] | X[patch*LA_CHUNKS + LA_CHUNKS - 1]) & FILTER) != (ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER))) {
				patch_clause_output = 0;
			}

			if (patch_clause_output) {
				output_one_patches[output_one_patches_count] = patch;
				output_one_patches_count++;
			}
		}
		//printf(" %d neg \\n", output_one_patches_count);
		if (output_one_patches_count > 0) {
			*clause_output = 1;
			int patch_id = curand(localState) % output_one_patches_count;
			*clause_patch = output_one_patches[patch_id];
		} else {
			*clause_output = 0;
			*clause_patch = -1;
		}
	}
	__device__ inline void calculate_clause_output(curandState *localState, unsigned int *ta_state, unsigned int *clause_output, int *output_patches_ones, int *output_patches_zeros, unsigned int *drop_feature, int *X, int *posneg)
	{
		int output_one_patches[PATCHES];
		int output_zero_patches[PATCHES];
		int output_one_patches_count;
		int output_zero_patches_count;
		int patch_clause_output;

		// Evaluate each patch (convolution)
		output_one_patches_count = 0;
		output_zero_patches_count = 1;
		for (int patch = 0; patch < PATCHES; ++patch) {
			patch_clause_output = 1;
			for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
				//printf(" %d checking \\n", ((ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] & (drop_feature[la_chunk] | X[patch*LA_CHUNKS + la_chunk])) != ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]));
				if ((ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] & (drop_feature[la_chunk] | X[patch*LA_CHUNKS + la_chunk])) != ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]) {
					patch_clause_output = 0;
					break;
				}
			}

			if (((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & (drop_feature[LA_CHUNKS-1] | X[patch*LA_CHUNKS + LA_CHUNKS - 1]) & FILTER) != (ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER))) {
				patch_clause_output = 0;
			}

			if (patch_clause_output) {
				output_one_patches[output_one_patches_count] = patch;
				output_one_patches_count++;
			} else {
				output_zero_patches[output_zero_patches_count] = patch;
				output_zero_patches_count++;
			}
			//printf(" %d patch_clause_output \\n", patch_clause_output);
		}
		//printf(" %d neg \\n", output_one_patches_count);
		if (output_one_patches_count == 1) {
			*clause_output = 1;
			//int patch_id = curand(localState) % output_one_patches_count;
			*output_patches_ones = output_one_patches[0];
			*output_patches_zeros = output_zero_patches[0];
			*posneg = -1;
		} 
		if (output_one_patches_count == 2) {
			*clause_output = 1;
			int patch_id = curand(localState) % output_one_patches_count;
			*output_patches_ones = output_one_patches[patch_id];
			
			//*clause_patch = output_one_patches[patch_id];
			*posneg = 1;
		} 
		else {
			*clause_output = 0;
			int patch_id = curand(localState) % output_zero_patches_count;
			*output_patches_zeros = output_zero_patches[patch_id];
			//*clause_patch = -1;
			*posneg = 0;
		}
	}


	__device__ inline void update_clause(curandState *localState, int *clause_weight, unsigned int *ta_state, int clause_output, int *output_patches_ones, int *output_patches_zeros, unsigned int *drop_feature, int *X, int y, int class_sum, int posneg)
	{
		int target = 1 - 2*(class_sum > y);
		//printf(" %d %d \\n", clause_output, y);
		
		if (target == -1 && curand_uniform(localState) > 1.0*Q/max(1, CLASSES-1)) {
			return;
		}

		//printf(" %d before \\n", clause_output);
		
		int sign;
		int clause_patch1[2];
		int clause_patch2[2];
		int clause_patch1_nb;
		int clause_patch2_nb;
		int feedback1;
		int feedback2;
		int clause_output1;
		int clause_output2;
		//int sign = (*clause_weight >= 0) - (*clause_weight < 0);
		if (posneg == 1 && target == 1) {
			clause_patch1[0] = 0;
			clause_patch1[1] = 1;
			clause_patch1_nb = 2;
			feedback1 = 1;
			feedback2 = 0;
			clause_output1 = 1;
			//sign = 1;
		} else if (posneg == -1 && target == 1){
			//combat false negative
			clause_patch1[0] = *output_patches_zeros;
			clause_patch1_nb = 1;
			//damp positive patch
			//clause_patch2[0] = *output_patches_ones;
			//clause_patch2_nb = 1;
			feedback1 = 1;
			feedback2 = 0;
			clause_output1 = 0;
			//clause_output2 = 1;
			//sign = 1;
		} else if (posneg == -1 && target == -1){
			//boost positive
			clause_patch1[0] = *output_patches_ones;
			clause_patch1_nb = 1;
			//boost negative
			clause_patch2[0] = *output_patches_zeros;
			clause_patch2_nb = 1;
			feedback1 = 1;
			feedback2 = 1;
			clause_output1 = 1;
			clause_output2 = 1;
		} else if (posneg == 1 && target == -1){
			//combat one of positives
			clause_patch2[0] = *output_patches_ones;
			clause_patch2_nb = 1;
			feedback1 = 0;
			feedback2 = 1;
			clause_output2 = 1;
		} else if (posneg == 0 && target == 1){
			//combat both negatives
			clause_patch1[0] = 0;
			clause_patch1[1] = 1;
			clause_patch1_nb = 2;
			feedback1 = 1;
			feedback2 = 0;
			clause_output1 = 0;
		} else if (posneg == 0 && target == -1){
			//combat both negatives
			clause_patch1[0] = 0;
			clause_patch1[1] = 1;
			clause_patch1_nb = 2;
			feedback1 = 1;
			feedback2 = 0;
			clause_output1 = 0;
		}
		
		//output_patches_ones
		//int sign = (class_sum >= 0) - (class_sum < 0);
		//printf(" %d %d %d %d \\n", clause_output, clause_patch, posneg, target);

		//printf(" blabla \\n");
		int absolute_prediction_error = abs(y - class_sum);
		if (curand_uniform(localState) <= 1.0*absolute_prediction_error/(2*THRESHOLD)) {
			if (feedback1 == 1) {
				
				if (clause_output1 && abs(*clause_weight) < INT_MAX) {
					(*clause_weight) += sign;
				}

				// Type I Feedback
				for (int la_chunk = 0; la_chunk < LA_CHUNKS; ++la_chunk) {
					// Generate random bit values
					unsigned int la_feedback = 0;
					for (int b = 0; b < INT_SIZE; ++b) {
						if (curand_uniform(localState) <= 1.0/S) {
							la_feedback |= (1 << b);
						}
					}

					
					for (int clause_patch = 0; clause_patch < clause_patch1_nb; ++clause_patch) {
						if (clause_output1) {
							#if BOOST_TRUE_POSITIVE_FEEDBACK == 1
								inc(ta_state, 0, la_chunk, (~drop_feature[la_chunk]) & X[clause_patch1[clause_patch]*LA_CHUNKS + la_chunk]);
							#else
								inc(ta_state, 0, la_chunk, (~drop_feature[la_chunk]) & X[clause_patch1[clause_patch]*LA_CHUNKS + la_chunk] & (~la_feedback));
							#endif

							dec(ta_state, 0, la_chunk, (~drop_feature[la_chunk]) & (~X[clause_patch1[clause_patch]*LA_CHUNKS + la_chunk]) & la_feedback);
						} else {
							dec(ta_state, 0, la_chunk, (~drop_feature[la_chunk]) & la_feedback);
						}
					}
				}
			} else if (feedback2 && clause_output2) {
				// Type II Feedback

				if (abs(*clause_weight) > 1) {
					(*clause_weight) -= sign;
				}
				
				for (int clause_patch = 0; clause_patch < 2; ++clause_patch) {
					for (int la_chunk = 0; la_chunk < LA_CHUNKS; ++la_chunk) {
						inc(ta_state, 0, la_chunk, (~drop_feature[la_chunk]) & (~X[clause_patch2[clause_patch]*LA_CHUNKS + la_chunk]) & (~ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]));
					}
				}
			}
		}
	}

	// Evaluate example
	__global__ void evaluate(unsigned int *global_ta_state, int *clause_weights, int *class_sum, unsigned int *drop_clause, unsigned int *drop_feature, int *X, int example)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		for (int clause = index; clause < CLAUSES; clause += stride) {
			if (drop_clause[clause] == 1) {
				continue;
			}

			//unsigned int type_clause;
			//#if -1 == 1 - 2 * (clause % 2)
			//	type_clause = 0;
			//#else
			//	type_clause = 1;
			//#endif

			unsigned int *ta_state = &global_ta_state[clause*LA_CHUNKS*STATE_BITS];

			int clause_output;
			int sum_patches = 0;
			for (int patch = 0; patch < PATCHES; ++patch) {
				clause_output = 1;
				for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
					if ((ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] & (drop_feature[la_chunk] | X[example*(LA_CHUNKS*PATCHES) + patch*LA_CHUNKS + la_chunk])) != ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]) {
						clause_output = 0;
						break;
					}
				}

				if ((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & (drop_feature[LA_CHUNKS-1] | X[example*(LA_CHUNKS*PATCHES) + patch*LA_CHUNKS + LA_CHUNKS-1]) & FILTER) != (ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER)) {
					clause_output = 0;
				}

				if (clause_output) {
					sum_patches += clause_output;
					//break;
				}
			}

			if (sum_patches == 2) {
				int class_id = clause / (CLAUSES / CLASSES);
				int clause_weight = clause_weights[clause];
				atomicAdd(&class_sum[class_id], clause_weight);					
			}
			if (sum_patches == 1) {
				int class_id = clause / (CLAUSES / CLASSES);
				int clause_weight = clause_weights[clause];
				atomicAdd(&class_sum[class_id], -clause_weight);					
			}


		}
	}

	// Update state of Tsetlin Automata team
	__global__ void update(curandState *state, unsigned int *global_ta_state, int *clause_weights, int *class_sum, unsigned int *drop_clause, unsigned int *drop_feature, int *X, int *y, int example)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		/* Copy state to local memory for efficiency */  
		curandState localState = state[index];

		// Calculate clause output first
		for (unsigned long long clause = index; clause < CLAUSES; clause += stride) {
			if (drop_clause[clause] == 1) {
				continue;
			}
			
			//#unsigned int type_clause;
			//#/#if -1 == 1 - 2 * (clause % 2)
			//#	type_clause = 0;
			//##else
			//	type_clause = 1;
			//#endif
			//printf(" %d neg \\n", *X);
			
			
			int class_id = clause / (CLAUSES / CLASSES);

			unsigned int *ta_state = &global_ta_state[clause*LA_CHUNKS*STATE_BITS];

			int output_patches_ones;
			int output_patches_zeros;

			unsigned int clause_output;
			//int clause_patch;
			int posneg = 0;
			//#if type_clause == 1
			//	calculate_positive_clause_output(&localState, ta_state, &clause_output, &clause_patch, drop_feature, &X[example*(LA_CHUNKS*PATCHES)]);
			//#else
			//	calculate_negative_clause_output(&localState, ta_state, &clause_output, &clause_patch, drop_feature, &X[example*(LA_CHUNKS*PATCHES)]);
			//#endif
			calculate_clause_output(&localState, ta_state, &clause_output, &output_patches_ones, &output_patches_zeros, drop_feature, &X[example*(LA_CHUNKS*PATCHES)], &posneg);

			int local_class_sum = class_sum[class_id];

			//if (posneg == 0){
			//	continue;
			//}
			//printf(" %d \\n", clause_output);

			
			
			if (local_class_sum > THRESHOLD) {
				local_class_sum = THRESHOLD;
			} else if (local_class_sum < -THRESHOLD) {
				local_class_sum = -THRESHOLD;
			}
			update_clause(&localState, &clause_weights[clause], ta_state, clause_output, &output_patches_ones, &output_patches_zeros, drop_feature, &X[example*(LA_CHUNKS*PATCHES)], y[example*CLASSES + class_id], local_class_sum, posneg);
		}
	
		state[index] = localState;
	}
}