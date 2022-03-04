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
    __global__ void prepare(curandState *state, unsigned int *global_ta_state, int *clause_weights, int *class_sum)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        curandState localState = state[index];

        for (unsigned long long clause = index; clause < CLAUSES; clause += stride) {
            //#if NEGATIVE_CLAUSES == 1
            //    clause_weights[clause] = 1 - 2 * (clause % 2);
            //#else
            //    clause_weights[clause] = 1;
            //#endif
            clause_weights[clause] = 1;
            
            unsigned int *ta_state = &global_ta_state[clause*LA_CHUNKS*STATE_BITS];
            for (int la_chunk = 0; la_chunk < LA_CHUNKS-1; ++la_chunk) {
                for (int b = 0; b < STATE_BITS-1; ++b) {
                    ta_state[la_chunk*STATE_BITS + b] = ~0;
                }
                ta_state[la_chunk*STATE_BITS + STATE_BITS - 1] = 0;
            }
        }

        state[index] = localState;
    }
}