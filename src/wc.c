#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/mman.h>

#include <immintrin.h>

#include "hash.c"

#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)

/*
 * Returns the page size (in bytes).
 */
size_t get_page_size() { return sysconf(_SC_PAGESIZE); }

uint64_t count_lines(uint8_t *buffer, size_t length) {
  uint64_t num_lines = 0;

  for (size_t i = 0; i < length; i++)
    num_lines += buffer[i] == '\n' ? 1 : 0;

  return num_lines;
}

size_t memcount_n_avx2(const void *s) {
  const size_t n = 32 * 4096;
  __m256i cv = _mm256_set1_epi8('\n'), zv = _mm256_setzero_si256(), sum = zv,
          acr0, acr1, acr2, acr3;
  const char *p, *pe;

  for (p = s; p != (char *)s + (n - (n % (252 * 32)));) {
    for (acr0 = acr1 = acr2 = acr3 = zv, pe = p + 252 * 32; p != pe; p += 128) {
      acr0 = _mm256_sub_epi8(
          acr0, _mm256_cmpeq_epi8(cv, _mm256_load_si256((const __m256i *)p)));
      acr1 = _mm256_sub_epi8(
          acr1,
          _mm256_cmpeq_epi8(cv, _mm256_load_si256((const __m256i *)(p + 32))));
      acr2 = _mm256_sub_epi8(
          acr2,
          _mm256_cmpeq_epi8(cv, _mm256_load_si256((const __m256i *)(p + 64))));
      acr3 = _mm256_sub_epi8(
          acr3,
          _mm256_cmpeq_epi8(cv, _mm256_load_si256((const __m256i *)(p + 96))));
    }
    sum = _mm256_add_epi64(sum, _mm256_sad_epu8(acr0, zv));
    sum = _mm256_add_epi64(sum, _mm256_sad_epu8(acr1, zv));
    sum = _mm256_add_epi64(sum, _mm256_sad_epu8(acr2, zv));
    sum = _mm256_add_epi64(sum, _mm256_sad_epu8(acr3, zv));
  }

  for (acr0 = zv; p + 32 < (char *)s + n; p += 32)
    acr0 = _mm256_sub_epi8(
        acr0, _mm256_cmpeq_epi8(cv, _mm256_load_si256((const __m256i *)p)));
  sum = _mm256_add_epi64(sum, _mm256_sad_epu8(acr0, zv));

  size_t count = _mm256_extract_epi64(sum, 0) + _mm256_extract_epi64(sum, 1) +
                 _mm256_extract_epi64(sum, 2) + _mm256_extract_epi64(sum, 3);

  while (p != (char *)s + n)
    count += *p++ == '\n';
  return count;
}

size_t memcount_avx2(const void *s, size_t n) {
  __m256i cv = _mm256_set1_epi8('\n'), zv = _mm256_setzero_si256(), sum = zv,
          acr0, acr1, acr2, acr3;
  const char *p, *pe;

  for (p = s; p != (char *)s + (n - (n % (252 * 32)));) {
    for (acr0 = acr1 = acr2 = acr3 = zv, pe = p + 252 * 32; p != pe; p += 128) {
      acr0 = _mm256_sub_epi8(
          acr0, _mm256_cmpeq_epi8(cv, _mm256_load_si256((const __m256i *)p)));
      acr1 = _mm256_sub_epi8(
          acr1,
          _mm256_cmpeq_epi8(cv, _mm256_load_si256((const __m256i *)(p + 32))));
      acr2 = _mm256_sub_epi8(
          acr2,
          _mm256_cmpeq_epi8(cv, _mm256_load_si256((const __m256i *)(p + 64))));
      acr3 = _mm256_sub_epi8(
          acr3,
          _mm256_cmpeq_epi8(cv, _mm256_load_si256((const __m256i *)(p + 96))));
    }
    sum = _mm256_add_epi64(sum, _mm256_sad_epu8(acr0, zv));
    sum = _mm256_add_epi64(sum, _mm256_sad_epu8(acr1, zv));
    sum = _mm256_add_epi64(sum, _mm256_sad_epu8(acr2, zv));
    sum = _mm256_add_epi64(sum, _mm256_sad_epu8(acr3, zv));
  }

  for (acr0 = zv; p + 32 < (char *)s + n; p += 32)
    acr0 = _mm256_sub_epi8(
        acr0, _mm256_cmpeq_epi8(cv, _mm256_load_si256((const __m256i *)p)));
  sum = _mm256_add_epi64(sum, _mm256_sad_epu8(acr0, zv));

  size_t count = _mm256_extract_epi64(sum, 0) + _mm256_extract_epi64(sum, 1) +
                 _mm256_extract_epi64(sum, 2) + _mm256_extract_epi64(sum, 3);

  while (p != (char *)s + n)
    count += *p++ == '\n';
  return count;
}

static inline uint64_t hsum_epu64_scalar(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i sum2x64 = _mm_add_epi64(lo, hi); // narrow to 128

  hi = _mm_unpackhi_epi64(sum2x64, sum2x64);
  __m128i sum = _mm_add_epi64(hi, sum2x64); // narrow to 64
  return _mm_cvtsi128_si64(sum);
}

uint64_t count_lines_avx2(uint8_t *buffer, size_t length) {
  const size_t REG_SIZE = sizeof(__m256i);
  const size_t ITER_SIZE_BYTES = 4 * REG_SIZE;

  __m256i newline = _mm256_set1_epi8('\n');

  __m256i m0 = _mm256_setzero_si256();
  __m256i m1 = _mm256_setzero_si256();
  __m256i m2 = _mm256_setzero_si256();
  __m256i m3 = _mm256_setzero_si256();

  __m256i x0, x1, x2, x3;

  for (size_t i = 0; i < length; i += ITER_SIZE_BYTES) {
    x0 = _mm256_load_si256((const __m256i *)(buffer + (0 * REG_SIZE)));
    x1 = _mm256_load_si256((const __m256i *)(buffer + (1 * REG_SIZE)));
    x2 = _mm256_load_si256((const __m256i *)(buffer + (2 * REG_SIZE)));
    x3 = _mm256_load_si256((const __m256i *)(buffer + (3 * REG_SIZE)));

    buffer += ITER_SIZE_BYTES;

    x0 = _mm256_cmpeq_epi8(x0, newline);
    x1 = _mm256_cmpeq_epi8(x1, newline);
    x2 = _mm256_cmpeq_epi8(x2, newline);
    x3 = _mm256_cmpeq_epi8(x3, newline);

    m0 = _mm256_sub_epi8(m0, x0);
    m1 = _mm256_sub_epi8(m1, x1);
    m2 = _mm256_sub_epi8(m2, x2);
    m3 = _mm256_sub_epi8(m3, x3);
  }

  // accumulate
  m0 = _mm256_add_epi8(m0, m1);
  m2 = _mm256_add_epi8(m2, m3);
  m0 = _mm256_add_epi8(m0, m2);

  // horizontal sum

  return hsum_epu64_scalar(m0);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    puts("Usage: wc [FILE]");
    return 1;
  }

  // open file
  FILE *file = fopen(argv[1], "rb");
  assert(file);

  uint64_t num_lines = 0;

  size_t bufsize = (32 * get_page_size()) & -32;
  uint8_t *buffer = aligned_alloc(32, bufsize);
  while (1) {
    size_t nread = fread(buffer, sizeof *buffer, bufsize, file);

    if (unlikely(nread != bufsize && ferror(file))) {
      printf("error reading file %s\n", argv[1]);
      exit(1);
    }

    if (likely(nread == bufsize)) {
      num_lines += memcount_n_avx2(buffer);
    } else {
      num_lines += memcount_avx2(buffer, nread);
    }

    if (unlikely(nread < bufsize && feof(file))) {
      break;
    }
  }

  printf("num_lines: %lu\n", num_lines);

  free(buffer);
  fclose(file);
}