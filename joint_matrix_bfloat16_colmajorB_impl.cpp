// Based on:
// https://github.com/intel/llvm/blob/sycl/sycl/test-e2e/Matrix/joint_matrix_bfloat16_colmajorA_colmajorB_impl.hpp

#include <random>
#include <sycl/sycl.hpp>

#define TM 8
#define TN 16
#define TK 16
#define SG_SZ 16

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
namespace syclintelex = sycl::ext::intel::experimental;
using bfloat16 = sycl::ext::oneapi::bfloat16;
using fp16 = sycl::half;

#define BF16_EPSILON 0.00781250

float make_fp32(bfloat16 x) {
  unsigned int y = *((int *)&x);
  y = y << 16;
  float *res = reinterpret_cast<float *>(&y);
  return *res;
}
float make_fp32(fp16 x) { return x; }

template <typename KernelName> size_t get_sg_size(queue q) {
  auto KernelID = get_kernel_id<KernelName>();
  auto KB =
      get_kernel_bundle<bundle_state::executable>(q.get_context(), {KernelID});
  auto kernel = KB.get_kernel(KernelID);

  return kernel
      .template get_info<info::kernel_device_specific::max_sub_group_size>(
          q.get_device());
}

void fill_matrix(bfloat16 *M, size_t Rows, size_t Cols) {
  std::random_device dev;
  std::uniform_real_distribution<float> fdistr(-1.0, 1.0);
  for (unsigned int i = 0; i < Rows; i++) {
    for (unsigned int j = 0; j < Cols; j++) {
      M[i * Cols + j] = bfloat16(fdistr(dev));
    }
  }
}
template <typename T, typename F>
void fill_matrix(T *src, unsigned int rows, unsigned int cols, F op) {
  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      src[i * cols + j] = T(op(i, j));
    }
  }
}
template <typename T>
void native_matmul_colmajorB(T *A, T *B, float *C, size_t M, size_t N,
                             size_t K) {
  memset(C, 0, sizeof(float) * M * N);
  for (unsigned int i = 0; i < M; i++) {
    for (unsigned int k = 0; k < K; k++) {
      for (unsigned int j = 0; j < N; j++) {
        C[i * N + j] += make_fp32(A[i * K + k]) * make_fp32(B[j * K + k]);
      }
    }
  }
}

void verify_result(float *result, float *ref, size_t M, size_t N, size_t K,
                   float floatTol = BF16_EPSILON) {
  for (unsigned int i = 0; i < M; i++) {
    for (unsigned int j = 0; j < N; j++) {
      float a = result[i * N + j];
      float b = ref[i * N + j];
      if ((fabs(a - b)) > floatTol) {
        std::cout << "failed at index " << i << ", " << j << ", res " << a
                  << " != ref " << b << " difference is " << a - b << "\n";
        return;
      }
      // assert((fabs(a) - fabs(b)) <= floatTol);
    }
  }
}

template <typename T>
void matrix_vnni(unsigned int rows, unsigned int cols, T *src, T *dest,
                 unsigned int vnniFactor = 2) {
  for (unsigned int i = 0; i < rows / vnniFactor; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      for (unsigned int k = 0; k < vnniFactor; k++) {
        dest[i * cols * vnniFactor + j * vnniFactor + k] =
            src[(i * vnniFactor + k) * cols + j];
      }
    }
  }
}

template <typename T1, typename T2, size_t M, size_t N, size_t K>
void matrix_multiply(T1 *C, T2 *A, T2 *B) {
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  buffer<fp16, 2> bufA(A, range<2>(M, K));
  buffer<fp16, 2> bufB(B, range<2>(K, N));
  buffer<float, 2> bufC((float *)C, range<2>(M, N));

  queue q;
  size_t sg_size = get_sg_size<class imatrix>(q);
  q.submit([&](handler &cgh) {
     sycl::stream cout(16384, 16384, cgh);
     auto accC = bufC.get_access<access::mode::read_write>(cgh);
     auto accA = bufA.get_access<access::mode::read_write>(cgh);
     auto accB = bufB.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<class imatrix>(
         nd_range<2>({NDRangeM, NDRangeN * sg_size}, {1, 1 * sg_size}), [=
     ](nd_item<2> it) [[intel::reqd_sub_group_size(16)]] {
           // The submatrix API has to be accessed by all the workitems in a
           // subgroup these functions will be called once by the subgroup no
           // code divergence between the workitems
           const auto global_idx = it.get_global_id(0);
           const auto global_idy = it.get_global_id(1);
           const auto sg_startx = global_idx - it.get_local_id(0);
           const auto sg_starty = global_idy - it.get_local_id(1);

           sub_group sg = it.get_sub_group();
           joint_matrix<sub_group, fp16, use::a, TM, TK, layout::row_major>
               sub_a;
           // For B, we assume B has been already VNNIed.
           joint_matrix<sub_group, fp16, use::b, TK, TN, layout::row_major>
               sub_b;
           joint_matrix<sub_group, float, use::accumulator, TM, TN> sub_c;

           joint_matrix_load(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / sg_size * TN,
               N, layout::row_major);

           auto ptrB =
               accB.template get_multi_ptr<access::decorated::no>().get_raw() +
               it.get_local_id(1) * K;
           for (int k = 0; k < K; k += TK) {
             joint_matrix_load(
                 sg, sub_a,
                 accA.template get_multi_ptr<access::decorated::no>() + k * M +
                     sg_startx * TM,
                 M);

             // joint_matrix_load workaround for B
             sycl::vec<fp16, 2> B_wa[TK / 2];
             auto curB = ptrB + k;
             for (int kk = 0; kk < TK; kk += 2) {
               B_wa[kk / 2] = *(sycl::vec<fp16, 2> *)(curB + kk);
             }
             cout << it.get_local_id(1) << "  " << k << '/' << K << ':'
                  << (int)(float)make_fp32(B_wa[0][0]) << '\n';
             syclintelex::matrix::joint_matrix_apply(
                 sg, sub_b, [=](fp16 &x, size_t row, size_t col) {
                   x = B_wa[row / 2][row % 2];
                   if (k == 0 && col == 0)
                     cout << row << " " << col << ": " << (float)make_fp32(x)
                          << '\n';
                 });
             joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
           }
           joint_matrix_store(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / sg_size * TN,
               N, layout::row_major);
         }); // parallel for
   }).wait();
}

int main() {
  static constexpr size_t MATRIX_M = TM; // * 2
  static constexpr size_t MATRIX_N = TN; // * 2
  static constexpr size_t MATRIX_K = TK; // * 2
  fp16 A[MATRIX_K][MATRIX_M];
  fp16 B[MATRIX_K][MATRIX_N];
  float C[MATRIX_M][MATRIX_N];
  float refC[MATRIX_M][MATRIX_N];

  fill_matrix((fp16 *)A, MATRIX_M, MATRIX_K,
              [](size_t i, size_t j) { return i + 1 + j * 0.01; });
  fill_matrix((fp16 *)B, MATRIX_N, MATRIX_K,
              [](size_t i, size_t j) { return i + 1 + j * 0.01; });
  native_matmul_colmajorB((fp16 *)A, (fp16 *)B, (float *)refC, MATRIX_M,
                          MATRIX_N, MATRIX_K);

  matrix_multiply<float, fp16, MATRIX_M, MATRIX_N, MATRIX_K>(
      (float *)C, (fp16 *)A, (fp16 *)B);

  verify_result((float *)C, (float *)refC, MATRIX_M, MATRIX_N, MATRIX_K);
  std::cout << "passed" << std::endl;
}
