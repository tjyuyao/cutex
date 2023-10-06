#include <inttypes.h>

typedef int64_t index_t;

#define _CUTEX_MEMCHECK_IMPL_(N) if (i < 0 || i >= size_[0]) printf("Out Of Bound Tensor access detected at dim[%" PRId64 "]: assert(%" PRId64 " < %" PRId64 ")\n", dims_ - N, i, size_[0]);

template <typename T, size_t N> struct Tensor {
  T *__restrict__ base;
  const index_t *size_;
  const index_t *stride_;
  const size_t dims_;

  __device__ Tensor(T *__restrict__ _data, const index_t *_size,
                    const index_t *_stride, size_t _dims=N)
      : base(_data), size_(_size), stride_(_stride), dims_(_dims) {
      }

  __device__ Tensor<T, N - 1> operator[](index_t i) {
#ifdef _CUTEX_MEMCHECK_
    _CUTEX_MEMCHECK_IMPL_(N);
#endif
    return Tensor<T, N - 1>(base + stride_[0] * i, size_ + 1, stride_ + 1, dims_);
  }

  __device__ const Tensor<T, N - 1> operator[](index_t i) const {
#ifdef _CUTEX_MEMCHECK_
    _CUTEX_MEMCHECK_IMPL_(N);
#endif
    return Tensor<T, N - 1>(base + stride_[0] * i, size_ + 1, stride_ + 1);
  }

  __device__ size_t size(index_t i) const {
    if (i < 0) i = dims_ + i;
#ifdef _CUTEX_MEMCHECK_
    if (i >= N || i < 0) printf("Out-Of-Bound detected when calling Tensor.size(%" PRId64 ")", i);
#endif
    return size_[i];
  }

};

template <typename T> struct Tensor<T, 1> {
  T *__restrict__ base;
  const index_t *size_;
  const index_t *stride_;
  const size_t dims_;

  __device__ Tensor(T *__restrict__ _data, const index_t *_size,
                    const index_t *_stride, size_t _dims=1)
      : base(_data), size_(_size), stride_(_stride), dims_(_dims) { }

  __device__ T& operator[](index_t i) {
#ifdef _CUTEX_MEMCHECK_
    _CUTEX_MEMCHECK_IMPL_((int64_t)1);
#endif
    return base[stride_[0]*i];
  }

  __device__ const T& operator[](index_t i) const {
#ifdef _CUTEX_MEMCHECK_
    _CUTEX_MEMCHECK_IMPL_((int64_t)1);
#endif
    return base[stride_[0]*i];
  }

  __device__ size_t size(index_t i = 0) const {
    if (i < 0) i = dims_ + i;
#ifdef _CUTEX_MEMCHECK_
    if (i >= 1 || i < 0) printf("Out-Of-Bound detected when calling Tensor.size(%" PRId64 ")", i);
#endif
    return size_[i];
  }

};
