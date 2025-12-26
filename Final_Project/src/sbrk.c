// Minimal sbrk implementation with a fixed-size heap.
#include <stddef.h>
#include <stdint.h>

extern char _end[];
extern char _fstack[];

static char* brk = _end;

void* sbrk(ptrdiff_t incr) {
  if (incr < 0) {
    return (void*)-1;
  }

  char* next_brk = brk + incr;
  // Check against stack pointer (approximate)
  // _fstack is the top of the stack. We leave 64KB for stack.
  if ((uintptr_t)next_brk > (uintptr_t)_fstack - 65536) {
    return (void*)-1;
  }

  void* ret = brk;
  brk = next_brk;
  return ret;
}