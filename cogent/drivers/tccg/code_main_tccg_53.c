//
//	Sample Code:
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// #define DEBUG_CORRECTNESS

void pre_Initializing_Input_Tensors();
void post_Correctness();

// 20 8 8 20 20
// # abe-abcd-ecd
// # t3 [a,b,e] += sum(c,d) * t2 [a,b,c,d] * v2 [e,c,d];
//
int main(int argc, char **argv)
{
    // for sd2
    double *host_C, *host_C_chk;
    double *host_A;
    double *host_B;
    int size_idx_a, size_idx_b, size_idx_c, size_idx_d, size_idx_e;

    // timing
    struct timeval t1, t2;
    double elapsedTime;

    // Problem Size
    size_idx_a = 20;
    size_idx_b = 8;
    size_idx_c = 8;
    size_idx_d = 20;
    size_idx_e = 20;

    //
    if (argc == 8)
    {
        size_idx_a = atoi(argv[1]);
        size_idx_b = atoi(argv[2]);
        size_idx_c = atoi(argv[3]);
        size_idx_d = atoi(argv[4]);
        size_idx_e = atoi(argv[5]);
    }

    int size_C;
    int size_A;
    int size_B;
    int size_internal;

    // # abe-abcd-ecd
    // # t3 [a,b,e] += sum(c,d) * t2 [a,b,c,d] * v2 [e,c,d];
    size_internal = size_idx_c * size_idx_d;
    size_C = size_idx_a * size_idx_b * size_idx_e;
    size_A = size_idx_a * size_idx_b * size_idx_c * size_idx_d;
    size_B = size_idx_e * size_idx_c * size_idx_d;

    //
    host_C = (double *)malloc(sizeof(double) * size_C);
    host_C_chk = (double *)malloc(sizeof(double) * size_C);
    host_A = (double *)malloc(sizeof(double) * size_A);
    host_B = (double *)malloc(sizeof(double) * size_B);

    printf("==========================================================================================================\n");
    printf(">>> abe-abcd-ecd\n");
    printf(">>> t3 [a,b,e] += sum(c,d) * t2 [a,b,c,d] * v2 [e,c,d];\n");
    printf(">>> Problem Size (a,b,e) and (c,d): (%2d,%2d,%2d) and (%2d,%2d)\n", size_idx_a, size_idx_b, size_idx_e, size_idx_c, size_idx_d);
    printf("==========================================================================================================\n");

    //
    // Initialze "1" Output and "2 x 9" Inputs
    pre_Initializing_Input_Tensors(host_C, host_C_chk, size_C, host_A, size_A, host_B, size_B);

    // Run the Kernels
    gettimeofday(&t1, NULL);
    sd_t_d2_fusion_(size_idx_a, size_idx_b, size_idx_c, size_idx_d, size_idx_e, host_C, host_A, host_B, 1, -1);
    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("\nTime taken just to run kernels: %f ms\n", elapsedTime);

    // Free
    free(host_C);
    free(host_C_chk);
    free(host_A);
    free(host_B);

    return 0;
}

// Initialize t3 (t3_temp), 9 t2 and 9 v2.
void pre_Initializing_Input_Tensors(double *h_C, double *h_C_chk, int size_C, double *h_A, int size_A, double *h_B, int size_B)
{
    // t3
    int i, j;
    for (i = 0; i < size_C; i++)
    {
        h_C[i] = 0.0;
        h_C_chk[i] = 0.0;
    }

    for (j = 0; j < size_A; j++)
    {
        h_A[j] = ((double)rand() / RAND_MAX);
    }

    for (j = 0; j < size_B; j++)
    {
        h_B[j] = ((double)rand() / RAND_MAX);
    }
}