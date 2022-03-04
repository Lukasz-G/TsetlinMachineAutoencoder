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
    __global__ void prepare_encode(unsigned int *X, unsigned int *encoded_X, int number_of_examples, int dim_x, int dim_y, int dim_z, int patch_dim_x, int patch_dim_y, int append_negated, int class_features)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        int number_of_features = dim_x;//class_features + patch_dim_x * patch_dim_y * dim_z + (dim_x - patch_dim_x) + (dim_y - patch_dim_y);
        int number_of_patches = 2;//(dim_x - patch_dim_x + 1) * (dim_y - patch_dim_y + 1);

        int number_of_ta_chunks;
        if (append_negated) {
            number_of_ta_chunks= (((2*number_of_features-1)/32 + 1));
        } else {
            number_of_ta_chunks= (((number_of_features-1)/32 + 1));
        }

        for (int i = index; i < number_of_examples * number_of_patches * number_of_ta_chunks; i += stride) {
            encoded_X[i] = 0;
        }
    }

    __global__ void encode(unsigned int *X, unsigned int *encoded_X, int number_of_examples, int dim_x, int dim_y, int dim_z, int patch_dim_x, int patch_dim_y, int append_negated, int class_features)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        int global_number_of_features = dim_x * dim_y * dim_z;
        int number_of_features = dim_x;//class_features + patch_dim_x * patch_dim_y * dim_z + (dim_x - patch_dim_x) + (dim_y - patch_dim_y);
        int number_of_patches = 2;//(dim_x - patch_dim_x + 1) * (dim_y - patch_dim_y + 1);

        int number_of_ta_chunks;
        if (append_negated) {
            number_of_ta_chunks= (((2*number_of_features-1)/32 + 1));
        } else {
            number_of_ta_chunks= (((number_of_features-1)/32 + 1));
        }

        unsigned int *Xi;
        unsigned int *encoded_Xi;

        unsigned int input_step_size = global_number_of_features;

        //printf(" %d %d %d neg \\n", number_of_features, number_of_ta_chunks, patch_dim_x);

        for (int i = index; i < number_of_examples; i += stride) {
            unsigned int encoded_pos = i * number_of_patches * number_of_ta_chunks;
            unsigned int input_pos = i * input_step_size;

            int patch_nr = 0;
            // Produce the patches of the current image
            for (int y = 0; y < dim_y - patch_dim_y + 1; ++y) {
                for (int x = 0; x < dim_x - patch_dim_x + 1; x += patch_dim_x) {
                    Xi = &X[input_pos];
                    encoded_Xi = &encoded_X[encoded_pos];

                    // Encode class into feature vector 
                    for (int class_feature = 0; class_feature < class_features; ++class_feature) {

                        int chunk_nr = (class_feature + number_of_features) / 32;
                        int chunk_pos = (class_feature + number_of_features) % 32;
                        encoded_Xi[chunk_nr] |= (1 << chunk_pos);
                    }

                    // Encode y coordinate of patch into feature vector 
                    /*for (int y_threshold = 0; y_threshold < dim_y - patch_dim_y; ++y_threshold) {
                       int patch_pos = class_features + y_threshold;

                        if (y > y_threshold) {
                            int chunk_nr = patch_pos / 32;
                            int chunk_pos = patch_pos % 32;
                            encoded_Xi[chunk_nr] |= (1 << chunk_pos);
                        } else if (append_negated) {
                            int chunk_nr = (patch_pos + number_of_features) / 32;
                            int chunk_pos = (patch_pos + number_of_features) % 32;
                            encoded_Xi[chunk_nr] |= (1 << chunk_pos);
                        }
                    } */

                    // Encode x coordinate of patch into feature vector
                    /*for (int x_threshold = 0; x_threshold < dim_x - patch_dim_x; ++x_threshold) {
                        int patch_pos = class_features + (dim_y - patch_dim_y) + x_threshold;

                        if (x > x_threshold) {
                            int chunk_nr = patch_pos / 32;
                            int chunk_pos = patch_pos % 32;

                            encoded_Xi[chunk_nr] |= (1 << chunk_pos);
                        } else if (append_negated) {
                            int chunk_nr = (patch_pos + number_of_features) / 32;
                            int chunk_pos = (patch_pos + number_of_features) % 32;
                            encoded_Xi[chunk_nr] |= (1 << chunk_pos);
                        }
                    } 
                    */

                    // Encode patch content into feature vector
                    for (int p_y = 0; p_y < patch_dim_y; ++p_y) {
                        for (int p_x = 0; p_x < patch_dim_x; ++p_x) {
                            for (int z = 0; z < dim_z; ++z) {
                                int image_pos = (y + p_y)*dim_x*dim_z + (x + p_x)*dim_z + z;
                                int patch_pos = class_features + (dim_y - patch_dim_y) + (dim_x - patch_dim_x) + p_y * patch_dim_x * dim_z + p_x * dim_z + z;
                                
                                if (Xi[image_pos] == 1) {
                                    int chunk_nr = patch_pos / 32;
                                    int chunk_pos = patch_pos % 32;
                                    encoded_Xi[chunk_nr] |= (1 << chunk_pos);
                                    //printf(" %d neg \\n", encoded_Xi[chunk_nr]);
                                } else if (append_negated) {
                                    int chunk_nr = (patch_pos + number_of_features) / 32;
                                    int chunk_pos = (patch_pos + number_of_features) % 32;
                                    encoded_Xi[chunk_nr] |= (1 << chunk_pos);
                                    //printf(" %d neg \\n", encoded_Xi[chunk_nr]);
                                }
                            }
                        }
                    }
                    encoded_pos += number_of_ta_chunks;
                    patch_nr++;
                }
            }
        }
    }
}