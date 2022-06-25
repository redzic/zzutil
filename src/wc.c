#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <tchar.h>
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

#include <immintrin.h>

#include "memcount.h"

#if _MSC_VER

#define zz_likely(x) (x)
#define zz_unlikely(x) (x)

#else

#define zz_likely(x) __builtin_expect((x), 1)
#define zz_unlikely(x) __builtin_expect((x), 0)

#endif

/*
 * Returns the page size (in bytes).
 */
size_t get_page_size() {
#if defined(_WIN32)
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return si.dwPageSize;
#else
  return sysconf(_SC_PAGESIZE);
#endif
}

uint64_t count_lines(const uint8_t *const buffer, size_t length) {
  uint64_t num_lines = 0;

  for (size_t i = 0; i < length; i++) {
    num_lines += buffer[i] == '\n' ? 1 : 0;
  }

  return num_lines;
}

enum { AVX_ALIGN = 32 };
enum { PAGE_SIZE_MULT = 32 };

int main(int argc, char **argv) {
  if (argc < 2) {
    puts("Usage: wc [FILE]");
    return 1;
  }

  // open file
  FILE *file = fopen(argv[1], "rbe");
  if (!file) {
    printf("wc: failed to open file '%s': %s\n", argv[1], strerror(errno));
    return 1;
  }

  uint64_t num_lines = 0;

  size_t bufsize = AVX_ALIGN * ((PAGE_SIZE_MULT * get_page_size()) / AVX_ALIGN);
  uint8_t *buffer = aligned_alloc(AVX_ALIGN, bufsize);

  while (1) {
    size_t nread = fread(buffer, 1, bufsize, file);

    if (zz_unlikely(nread != bufsize && ferror(file))) {
      printf("wc: error reading file '%s': %s\n", argv[1], strerror(errno));
      return 1;
    }

    num_lines += memcount(buffer, nread);

    if (zz_unlikely(nread < bufsize && feof(file))) {
      break;
    }
  }

  printf("%llu %s\n", num_lines, argv[1]);

  free(buffer);
  fclose(file);
}