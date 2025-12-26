// TPU Helper functions for systolic array acceleration
#ifndef TPU_HELPER_H
#define TPU_HELPER_H

#include <stdint.h>

#include "cfu.h"

namespace tpu_helper {

#if defined(__riscv) && !defined(CFU_SOFTWARE_DEFINED)

#else
constexpr bool kHasHardwareTpu = false;
#endif

struct TpuStats {
  uint32_t start_timeouts;
  uint32_t finish_timeouts;
};

extern volatile uint32_t g_tpu_start_timeouts;
extern volatile uint32_t g_tpu_finish_timeouts;

void ResetTpuStats();
TpuStats GetTpuStats();

constexpr int kTileSize = 32;
constexpr int kPackedWords = kTileSize * ((kTileSize + 3) / 4);

// Pack a tile of matrix A (column-major in 4-pack layout) into TPU buffer
inline int PackTypeA(const int8_t* tile, int tile_m, int tile_k,
                     uint32_t* packed) {
  const int row_blocks = (tile_m + 3) / 4;  // ceil(tile_m / 4)
  for (int block = 0; block < row_blocks; ++block) {
    for (int col = 0; col < tile_k; ++col) {
      const int word_idx = block * tile_k + col;
      uint32_t packed_word = 0;
      for (int lane = 0; lane < 4; ++lane) {
        const int row = block * 4 + lane;
        const int shift = (3 - lane) * 8;
        if (row < tile_m) {
          const uint32_t byte_val =
              static_cast<uint8_t>(tile[row * tile_k + col]);
          packed_word |= byte_val << shift;
        }
      }
      packed[word_idx] = packed_word;
    }
  }
  return row_blocks * tile_k;
}

// Pack a tile of matrix B (row-major in 4-pack layout) into TPU buffer
inline int PackTypeB(const int8_t* tile, int tile_k, int tile_n,
                     uint32_t* packed) {
  const int col_blocks = (tile_n + 3) / 4;  // ceil(tile_n / 4)
  for (int block = 0; block < col_blocks; ++block) {
    for (int row = 0; row < tile_k; ++row) {
      const int word_idx = block * tile_k + row;
      uint32_t packed_word = 0;
      for (int lane = 0; lane < 4; ++lane) {
        const int col = block * 4 + lane;
        const int shift = (3 - lane) * 8;
        if (col < tile_n) {
          const uint32_t byte_val =
              static_cast<uint8_t>(tile[row * tile_n + col]);
          packed_word |= byte_val << shift;
        }
      }
      packed[word_idx] = packed_word;
    }
  }
  return col_blocks * tile_k;
}

// Launch one TPU matmul on a padded tile (up to 32x32). Results stored
// row-major in C_tile. Returns true on success, false on failure.
inline bool RunMatmulTilePacked(const uint32_t* packed_a, int words_a,
                                const uint32_t* packed_b, int words_b,
                                int tile_m, int tile_k, int tile_n,
                                int32_t* C_tile) {
#if !defined(__riscv) || defined(CFU_SOFTWARE_DEFINED)
  (void)packed_a;
  (void)words_a;
  (void)packed_b;
  (void)words_b;
  (void)tile_m;
  (void)tile_k;
  (void)tile_n;
  (void)C_tile;
  return false;
#else

#endif
}

inline bool RunMatmulTile(const int8_t* A_tile, const int8_t* B_tile,
                          int tile_m, int tile_k, int tile_n, int32_t* C_tile) {
#if !defined(__riscv) || defined(CFU_SOFTWARE_DEFINED)
  (void)A_tile;
  (void)B_tile;
  (void)tile_m;
  (void)tile_k;
  (void)tile_n;
  (void)C_tile;
  return false;
#else

#endif
}

}  // namespace tpu_helper

#endif  // TPU_HELPER_H
