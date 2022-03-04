#include <curand_kernel.h>
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
    // Transform examples
    __global__ void transform(unsigned int *global_ta_state, int *X, int *transformed_X_P, int *transformed_X_N)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int i = index; i < CLASSES*CLAUSES; i += stride) {
            unsigned long long class_id = i / CLAUSES;
            unsigned long long clause = i % CLAUSES;

            //unsigned int type_clause;
			//#if -1 == 1 - 2 * (clause % 2)
			//	type_clause = 0;
			//#else
			//	type_clause = 1;
			//#endif

            unsigned int *ta_state = &global_ta_state[class_id*CLAUSES*LA_CHUNKS*STATE_BITS + clause*LA_CHUNKS*STATE_BITS];

            int all_exclude = 1;
            for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
                if (ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] > 0) {
                    all_exclude = 0;
                    break;
                }
            }

            if ((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER) > 0) {
                all_exclude = 0;
            }

            if (all_exclude) {
                for (unsigned long long e = 0; e < NUMBER_OF_EXAMPLES; ++e) {
                    transformed_X_P[e*CLASSES*CLAUSES + i] = 0;
                    transformed_X_N[e*CLASSES*CLAUSES + i] = 0;
                }
                
                continue;
            }

            for (int e = 0; e < NUMBER_OF_EXAMPLES; ++e) {
                int clause_output;
                int patches_sum = 0;
                for (int patch = 0; patch < PATCHES; ++patch) {
                    clause_output = 1;
                    for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
                        if ((ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] & X[e*(LA_CHUNKS*PATCHES) + patch*LA_CHUNKS + la_chunk]) != ta_state[la_chunk*STATE_BITS + STATE_BITS - 1]) {
                            clause_output = 0;
                            break;
                        }
                    }

                    if ((ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & X[e*(LA_CHUNKS*PATCHES) + patch*LA_CHUNKS + LA_CHUNKS-1] & FILTER) != (ta_state[(LA_CHUNKS-1)*STATE_BITS + STATE_BITS - 1] & FILTER)) {
                        clause_output = 0;
                    }

                    if (clause_output) {
                        patches_sum += clause_output;
                        break;
                    }
                }

                if (patches_sum == 2){
                    transformed_X_P[e*CLASSES*CLAUSES + i] = 1;
                    }
                else if (patches_sum == 1){
                    transformed_X_N[e*CLASSES*CLAUSES + i] = 1;
                    }

            }
        }
    }
}