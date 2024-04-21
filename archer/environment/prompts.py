initial_benchmark_stats = """
Running kernel 1 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000132) s, performance: (   31.9) GFLOPS (%cuBLAS:  243.19%). size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.000256) s, performance: (  131.0) GFLOPS (%cuBLAS:  189.34%). size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.001508) s, performance: (  178.0) GFLOPS (%cuBLAS:    2.55%). size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.009664) s, performance: (  222.2) GFLOPS (%cuBLAS:    1.78%). size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.059962) s, performance: (  286.5) GFLOPS (%cuBLAS:    2.08%). size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
Average elapsed time: (0.470475) s, performance: (  292.1) GFLOPS (%cuBLAS:    1.98%). size: (4096).
""".strip()

initial_my_sgemm = """
/*

Matrix sizes:
MxK * KxN = MxN

*/

__global__ void sgemm(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // if statement is necessary to make things work under tile quantization
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}
""".strip()

initial_my_sgemm_runner = """
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void run_sgemm(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  sgemm<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
""".strip()

initial_gemm_prompt = lambda: f"""
Below is a simple starting implementation of SGEMM in CUDA. The kernel itself is implemented in `my_sgemm.cuh`, and the wrapper function for launching the kernel is implemented in `my_sgemm_runner.cuh`.

`my_sgemm.cuh`:
```
{initial_my_sgemm}
```

`my_sgemm_runner.cuh`:
```
{initial_my_sgemm_runner}
```

Over the course of this session, we will be optimizing this SGEMM implementation to cuBLAS-level performance.

To do so, we'll just be editing the above two files. The goal is a self-contained, external dependency-free, CUDA implementation of a fast SGEMM. We'll follow the following pseudocode together:

```
F = ("<initial contents of my_sgemm.cuh>", "<initial contents of my_sgemm_runner.cuh>")
F = improve_performance(F)
while True:
    compile_error = compile_benchmark(F) # I will run the compiler for you
    if compile_error:
        F = fix_compile_error(F)
        continue
    runtime_error, is_correct = run_benchmark(F) # I will exec the benchmark for you
    if runtime_error:
        F = fix_runtime_error(F)
        continue
    if not is_correct:
        F = fix_correctness(F)
    else:
        F = improve_performance(F)
```

To help put your optimizations in context, here is the performance of the current starting implementation, as reported by our benchmarking suite:
```
{initial_benchmark_stats}
```

TIPS:
- Start each of your responses with a rationale for how you think you should approach the task.
- Leverage your knowledge of CUDA and the performance of typical kinds of kernels to infer what the bottlenecks of the current implementation might be.
- You can choose to edit either `my_sgemm.cuh`, `my_sgemm_runner.cuh`, or both.
- You cannot edit other files or choose to create new ones.
- For each file you choose to edit, you will rewrite the entire contents of the file, including your changes.
- Even if you feel your knowledge is insufficient, propose and implement a change anyway. You will gain more information from seeing how the benchmarking results change in response to your change.
- Do not change the function prototypes of either the kernel of wrapper function. Their interface should remanain the same after your edits.
- You can add helper functions to either of the files if you wish.
- `my_sgemm.cuh` is included before `my_sgemm_runner.cuh` in the benchmarking source file. So, no need to add `#include`s to include either of these files.
- You may only add includes to the CUDA or C++ standard libraries.
- All code you write should be ready-to-go -- that is, ready to compile and run directly, without any placeholders.
- After presenting your rationale, present the edits to each file you choose to edit like this. For each file, follow this exact syntax -- I'll be using it to parse out the new contents of each file from your response:

`filename`:
```
<complete contents of rewritten `filename` ...>
```

Good luck!
---

To begin, please identify one or more ways the SGEMM implementation can be made more performant. Make the appropriate edits to one or more of the given files, structuring your response using the format given above.
""".strip()

compile_error_prompt = lambda stdout, stderr: f"""
I've run the compiler on the state of the codebase right now and it seems that there's an error. Below is the `stdout` and `stderr` from the compiler.

`stdout`:
```
{stdout}
```

`stderr`:
```
{stderr}
```

Please identify the source(s) of the compile error. Make the appropriate edits to one or more of the given files, structuring your response using the format given above.
""".strip()

runtime_error_prompt = lambda stdout, stderr: f"""
Thank you. I've compiled and run the benchmarking executable given the state of the codebase right now and it seems that there's a runtime error. Below is the `stdout` and `stderr` from the benchmarking executable.

`stdout`:
```
{stdout}
```

`stderr`:
```
{stderr}
```

Please identify the source(s) of the runtime error. Make the appropriate edits to one or more of the given files, structuring your response using the format given above.
""".strip()

correctness_error_prompt = lambda stdout, stderr: f"""
Thank you. I've compiled and run the benchmarking executable given the state of the codebase right now and it seems that there's a correctness error. Below is the `stdout` and `stderr` from the benchmarking executable.

`stdout`:
```
{stdout}
```

`stderr`:
```
{stderr}
```

Please identify the source(s) of the correctness error. Make the appropriate edits to one or more of the given files, structuring your response using the format given above.
""".strip()

improve_performance_prompt = lambda stdout, stderr: f"""
Thank you. I've compiled and run the benchmarking executable given the state of the codebase right now and the kernel produced correct results! Below is the `stdout` and `stderr` from the benchmarking executable.

`stdout`:
```
{stdout}
```

`stderr`:
```
{stderr}
```

Please interpret the benchmarking results, and identify one or more ways the SGEMM implementation can be made more performant. Make the appropriate edits to one or more of the given files, structuring your response using the format given above.
""".strip()
