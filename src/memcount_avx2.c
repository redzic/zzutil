#include <assert.h>
#include <immintrin.h>
#include <stddef.h>

#include "memcount.h"

// usually called with 131072 bytes

size_t memcount(const uint8_t *s, size_t n) {
  const size_t VEC_SIZE = sizeof(__m256i);
  const size_t UNROLL_FAC = 4;
  // number of elements (bytes) that can be processed by one inner loop
  const size_t FULL_ITER_SIZE =
      UNROLL_FAC * VEC_SIZE * (UNROLL_FAC * (255 / UNROLL_FAC));

  __m256i needle = _mm256_set1_epi8('\n');

  // accumulator registers
  __m256i a0 = _mm256_setzero_si256();
  __m256i a1 = _mm256_setzero_si256();
  __m256i a2 = _mm256_setzero_si256();
  __m256i a3 = _mm256_setzero_si256();

  const __m256i zv = _mm256_setzero_si256();

  const uint8_t *end_ptr = s + FULL_ITER_SIZE * (n / FULL_ITER_SIZE);
#pragma unroll 1
  while (s != end_ptr) {
    // local accumulator registers
    __m256i x0 = _mm256_setzero_si256();
    __m256i x1 = _mm256_setzero_si256();
    __m256i x2 = _mm256_setzero_si256();
    __m256i x3 = _mm256_setzero_si256();

    for (const uint8_t *end_ptr_local = s + FULL_ITER_SIZE; s != end_ptr_local;
         s += (VEC_SIZE * UNROLL_FAC)) {
      __m256i m0 = _mm256_load_si256((const __m256i *)(s + (0 * VEC_SIZE)));
      __m256i m1 = _mm256_load_si256((const __m256i *)(s + (1 * VEC_SIZE)));
      __m256i m2 = _mm256_load_si256((const __m256i *)(s + (2 * VEC_SIZE)));
      __m256i m3 = _mm256_load_si256((const __m256i *)(s + (3 * VEC_SIZE)));

      x0 = _mm256_sub_epi8(x0, _mm256_cmpeq_epi8(m0, needle));
      x1 = _mm256_sub_epi8(x1, _mm256_cmpeq_epi8(m1, needle));
      x2 = _mm256_sub_epi8(x2, _mm256_cmpeq_epi8(m2, needle));
      x3 = _mm256_sub_epi8(x3, _mm256_cmpeq_epi8(m3, needle));
    }

    a0 = _mm256_add_epi64(a0, _mm256_sad_epu8(x0, zv));
    a1 = _mm256_add_epi64(a1, _mm256_sad_epu8(x1, zv));
    a2 = _mm256_add_epi64(a2, _mm256_sad_epu8(x2, zv));
    a3 = _mm256_add_epi64(a3, _mm256_sad_epu8(x3, zv));
  }

  /* Number of bytes remaining that can be processed by SIMD. */
  size_t vec_remain = (size_t)((ptrdiff_t)n & -(ptrdiff_t)VEC_SIZE) -
                      (FULL_ITER_SIZE * (n / FULL_ITER_SIZE));

  if (vec_remain != 0) {
    end_ptr += vec_remain;

    __m256i lastsum = _mm256_setzero_si256();

#pragma unroll 2
    while (s != end_ptr) {
      __m256i cmp =
          _mm256_cmpeq_epi8(needle, _mm256_load_si256((const __m256i *)s));
      lastsum = _mm256_sub_epi8(lastsum, cmp);
      s += VEC_SIZE;
    }

    a3 = _mm256_add_epi64(a3, _mm256_sad_epu8(lastsum, zv));
  }

  a0 = _mm256_add_epi64(a0, a1);
  a2 = _mm256_add_epi64(a2, a3);
  a0 = _mm256_add_epi64(a0, a2);

  size_t count = _mm256_extract_epi64(a0, 0) + _mm256_extract_epi64(a0, 1) +
                 _mm256_extract_epi64(a0, 2) + _mm256_extract_epi64(a0, 3);

  size_t scalar_remain = n - (size_t)((ptrdiff_t)n & -(ptrdiff_t)VEC_SIZE);

  end_ptr += scalar_remain;
  while (s != end_ptr) {
    count += *s++ == '\n' ? 1 : 0;
  }

  return count;
}
