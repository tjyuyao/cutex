#include <inttypes.h>

typedef int64_t index_t;

#define _ICE_MEMCHECK_IMPL_(N) if (i < 0 || i >= size[0]) printf("Out Of Bound Tensor access detected at dim[-%" PRId64 "]: %" PRId64 " not in range [0, %" PRId64 ")\n", N, i, size[0]);

template <typename T, size_t N> struct Tensor {
  T *__restrict__ base;
  const index_t *size;
  const index_t *stride;

  __device__ Tensor(T *__restrict__ _data, const index_t *_size,
                    const index_t *_strides)
      : base(_data), size(_size), stride(_strides) {
      }

  __device__ Tensor<T, N - 1> operator[](index_t i) {
#ifdef _ICE_MEMCHECK_
    _ICE_MEMCHECK_IMPL_(N);
#endif
    return Tensor<T, N - 1>(base + stride[0] * i, size + 1, stride + 1);
  }

  __device__ const Tensor<T, N - 1> operator[](index_t i) const {
#ifdef _ICE_MEMCHECK_
    _ICE_MEMCHECK_IMPL_(N);
#endif
    return Tensor<T, N - 1>(base + stride[0] * i, size + 1, stride + 1);
  }

};

template <typename T> struct Tensor<T, 1> {
  T *__restrict__ base;
  const index_t *size;
  const index_t *stride;

  __device__ Tensor(T *__restrict__ _data, const index_t *_size,
                    const index_t *_strides)
      : base(_data), size(_size), stride(_strides) { }

  __device__ T& operator[](index_t i) {
#ifdef _ICE_MEMCHECK_
    _ICE_MEMCHECK_IMPL_((int64_t)1);
#endif
    return base[stride[0]*i];
  }

  __device__ const T& operator[](index_t i) const {
#ifdef _ICE_MEMCHECK_
    _ICE_MEMCHECK_IMPL_((int64_t)1);
#endif
    return base[stride[0]*i];
  }

};
