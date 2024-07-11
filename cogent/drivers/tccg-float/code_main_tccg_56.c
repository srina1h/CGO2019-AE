// created by tc_code_include() in tc_code_include.py
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <locale.h>


void pre_Initializing_Input_Tensors(float* h_C, float* h_C_chk, int size_C, float* h_A, int size_A, float* h_B, int size_B);
void post_Correctness(float* h_C, float* h_C_chk, float* h_A, float* h_B, int size_idx_a, int size_idx_b, int size_idx_c);

//
//# ab-ac-bc 
//# t3 [a,4096,b,20] += sum(c,768) * t2 [a,c] * v2 [b,c];
//
int main(int argc, char** argv)
{
	// for sd2
	float *host_C, *host_C_chk;
	float *host_A;
	float *host_B;
	int size_idx_a, size_idx_b, size_idx_c; 

	// Problem Size
	size_idx_a = 4096;
	size_idx_b = 20;
	size_idx_c = 768;
	
	//
	if (argc == 4)
	{
		size_idx_a = atoi(argv[1]);
		size_idx_b = atoi(argv[2]);
		size_idx_c = atoi(argv[3]);
	}
	
	int size_C;
	int size_A;
	int size_B;
	int size_internal;

	//# ab-ac-bc 
	//# t3 [a,4096,b,20] += sum(c,768) * t2 [a,c] * v2 [b,c];
	size_internal = size_idx_c;
	size_C = size_idx_a * size_idx_b;
	size_A = size_idx_a * size_idx_c;
	size_B = size_idx_b * size_idx_c;

		//
	host_C 			= (float*)malloc(sizeof(float) * size_C);
	host_C_chk 	= (float*)malloc(sizeof(float) * size_C);
	host_A 			= (float*)malloc(sizeof(float) * size_A);
	host_B 			=	(float*)malloc(sizeof(float) * size_B);
	
	printf ("==========================================================================================================\n");
	printf (">>> ab-ac-bc\n");
	printf (">>> t3 [a,4096,b,20] += t2 [a,c] * v2 [b,c];\n");
	printf (">>> Problem Size (a,b) and (c): (%4d,%4d) and (%4d)\n", size_idx_a, size_idx_b, size_idx_c); 
	printf ("==========================================================================================================\n");
	
	//
	// Initialze "1" Output and "2 x 9" Inputs
	pre_Initializing_Input_Tensors(host_C, host_C_chk, size_C, host_A, size_A, host_B, size_B);
								
	
		// Run the Kernels
		sd_t_d2_fusion_(size_idx_a, size_idx_b, size_idx_c, host_C, host_A, host_B, 1, -1);
		

	// Correctness-Check
	post_Correctness(host_C, host_C_chk, host_A, host_B, size_idx_a, size_idx_b, size_idx_c);

	// Free
	free(host_C);   free(host_C_chk);
	free(host_A);
	free(host_B);

	return 0;
}

// Initialize t3 (t3_temp), 9 t2 and 9 v2.
void pre_Initializing_Input_Tensors(float* h_C, float* h_C_chk, int size_C, float* h_A, int size_A, float* h_B, int size_B)
{
	// t3
	int i, j;
	for (i = 0; i < size_C; i++)
	{
		h_C[i] 	= 0.0;
		h_C_chk[i] = 0.0;
	}

	for (j = 0; j < size_A; j++)
	{
		h_A[j] = ((float)rand() / RAND_MAX) / 100;
	}

	for (j = 0; j < size_B; j++)
	{
		h_B[j] = ((float)rand() / RAND_MAX) / 100;
	}
}

//
void post_Correctness(float* h_C, float* h_C_chk, float* h_A, float* h_B, int size_idx_a, int size_idx_b, int size_idx_c)
{
	//# ab-ac-bc 
	//# t3 [a,4096,b,20] += sum(c,768) * t2 [a,c] * v2 [b,c];
	int size_C = size_idx_a * size_idx_b;
	
	long long int    tmp_ops = 0;
	int              ops     = 0;
	int idx_a, idx_b, idx_c;
	for (idx_a = 0; idx_a < size_idx_a; idx_a++)
	for (idx_b = 0; idx_b < size_idx_b; idx_b++)
	{   
			ops = 0;
			for (idx_c = 0; idx_c < size_idx_c; idx_c++)
			{
					int tmp_r_idx = idx_a + (idx_b) * size_idx_a;
					h_C_chk[tmp_r_idx] += 	h_A[idx_a + (idx_c) * size_idx_a] * 
																	h_B[idx_b + (idx_c) * size_idx_b];
					
					ops++;
			}
			tmp_ops = tmp_ops + ops;
	}

	printf ("======================================= Correctness Check ==========================================\n");
	float   epsilon = 0.00001;
	int     diff    = 0;
	int     same    = 0;
	int 	i;
	for (i = 0; i < size_C; i++)
	{
		float check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			if (diff < 8)
			printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", tmp_ops);
	printf ("====================================================================================================\n");
}


