// TPU Helper functions for systolic array acceleration
#ifndef TPU_HELPER_H
#define TPU_HELPER_H

#include <stdint.h>

#include "cfu.h"

namespace tpu_helper {

#if defined(__riscv) && !defined(CFU_SOFTWARE_DEFINED)
constexpr bool kHasHardwareTpu = true;
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

struct ConvDebugInfo {
  uint8_t recorded;
  int32_t batch;
  int32_t out_y;
  int32_t out_x;
  int32_t out_channel;
  int32_t filter_sum;
  int32_t ref_filter_sum;
  int32_t hw_acc_no_bias;
  int32_t hw_acc_with_bias;
  int32_t ref_acc;
};

extern ConvDebugInfo g_conv_debug_info;

void ResetConvDebugInfo();
ConvDebugInfo GetConvDebugInfo();
void RecordConvDebugInfo(const ConvDebugInfo& info);

constexpr int kTileSize = 32;
constexpr int kPackedWords = kTileSize * ((kTileSize + 3) / 4);

// Pack a tile of matrix A (column-major in 4-pack layout) into TPU buffer.
// row_stride specifies the stride between rows (in bytes) inside tile.
inline int PackTypeA(const int8_t* tile, int tile_m, int tile_k, int row_stride,
                     uint32_t* packed) {
  const int row_blocks = (tile_m + 3) / 4;  // ceil(tile_m / 4)
  const uintptr_t base_addr = reinterpret_cast<uintptr_t>(tile);
  const bool can_vectorize = ((row_stride & 3) == 0) && ((base_addr & 3) == 0);

  for (int block = 0; block < row_blocks; ++block) {
    const int row_base = block * 4;
    const int8_t* row_ptrs[4] = {nullptr, nullptr, nullptr, nullptr};
    for (int lane = 0; lane < 4; ++lane) {
      const int row = row_base + lane;
      if (row < tile_m) {
        row_ptrs[lane] = tile + row * row_stride;
      }
    }

    int col = 0;
    if (can_vectorize && tile_k >= 4) {
      const int vector_cols = tile_k & ~3;
      for (; col < vector_cols; col += 4) {
        uint32_t row_words[4] = {0, 0, 0, 0};
        for (int lane = 0; lane < 4; ++lane) {
          if (row_ptrs[lane] != nullptr) {
            row_words[lane] =
                *reinterpret_cast<const uint32_t*>(row_ptrs[lane] + col);
          }
        }
        for (int inner = 0; inner < 4; ++inner) {
          const int out_col = col + inner;
          uint32_t packed_word = 0;
          for (int lane = 0; lane < 4; ++lane) {
            const uint32_t byte_val =
                (row_ptrs[lane] != nullptr)
                    ? ((row_words[lane] >> (inner * 8)) & 0xffu)
                    : 0u;
            packed_word |= byte_val << ((3 - lane) * 8);
          }
          packed[block * tile_k + out_col] = packed_word;
        }
      }
    }

    for (; col < tile_k; ++col) {
      uint32_t packed_word = 0;
      for (int lane = 0; lane < 4; ++lane) {
        if (row_ptrs[lane] != nullptr) {
          const uint32_t byte_val = static_cast<uint8_t>(row_ptrs[lane][col]);
          packed_word |= byte_val << ((3 - lane) * 8);
        }
      }
      packed[block * tile_k + col] = packed_word;
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

#if defined(__riscv) && !defined(CFU_SOFTWARE_DEFINED)
inline void TpuWriteAWord(uint32_t addr, uint32_t data) {
  cfu_op3(0, addr, data);
}

inline void TpuWriteBWord(uint32_t addr, uint32_t data) {
  cfu_op3(1, addr, data);
}

inline void TpuStart(uint32_t params) { cfu_op3(2, params, 0); }

inline uint32_t TpuReadC(uint32_t cmd) { return cfu_op3(3, cmd, 0); }

inline uint32_t TpuQueryBusy() { return cfu_op3(4, 0, 0); }
#else
inline void TpuWriteAWord(uint32_t addr, uint32_t data) {
  (void)addr;
  (void)data;
}

inline void TpuWriteBWord(uint32_t addr, uint32_t data) {
  (void)addr;
  (void)data;
}

inline void TpuStart(uint32_t params) { (void)params; }

inline uint32_t TpuReadC(uint32_t cmd) {
  (void)cmd;
  return 0;
}

inline uint32_t TpuQueryBusy() { return 0; }
#endif

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
  if (words_a > kPackedWords || words_b > kPackedWords) {
    return false;
  }

  for (int i = 0; i < words_a; ++i) {
    const uint32_t addr = static_cast<uint32_t>(i) << 16;
    TpuWriteAWord(addr, packed_a[i]);
  }

  for (int i = 0; i < words_b; ++i) {
    const uint32_t addr = static_cast<uint32_t>(i) << 16;
    TpuWriteBWord(addr, packed_b[i]);
  }

  const uint32_t params = (static_cast<uint32_t>(tile_k) << 24) |
                          (static_cast<uint32_t>(tile_m) << 16) |
                          (static_cast<uint32_t>(tile_n) << 8);
  TpuStart(params);

  constexpr int kMaxBusyPolls = 1 << 21;
  uint32_t busy = 0;
  int poll_count = 0;

  while (poll_count < kMaxBusyPolls) {
    busy = TpuQueryBusy();
    if (busy != 0) {
      break;  // TPU acknowledged work.
    }
    ++poll_count;
  }
  if (busy == 0) {
    ++g_tpu_start_timeouts;
    return false;  // Hardware never started; fall back to reference path.
  }

  poll_count = 0;
  while (busy && poll_count < kMaxBusyPolls) {
    busy = TpuQueryBusy();
    ++poll_count;
  }
  if (busy != 0) {
    ++g_tpu_finish_timeouts;
    return false;  // Hardware failed to finish in time.
  }

  const int col_words = (tile_n + 3) / 4;
  const int c_stride_words = (col_words > 0) ? col_words : 1;
  for (int row = 0; row < tile_m; ++row) {
    for (int col_word = 0; col_word < col_words; ++col_word) {
      const uint32_t c_index =
          static_cast<uint32_t>(row * c_stride_words + col_word);
      for (int word_sel = 0; word_sel < 4; ++word_sel) {
        const int col = col_word * 4 + (3 - word_sel);
        const uint32_t cmd = (c_index << 16) | static_cast<uint32_t>(word_sel);
        const int32_t value = static_cast<int32_t>(TpuReadC(cmd));
        if (col < tile_n) {
          C_tile[row * kTileSize + col] = value;
        }
      }
    }
  }

  return true;  // Success
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
  uint32_t A_buffer[kPackedWords];
  uint32_t B_buffer[kPackedWords];

  const int words_a = PackTypeA(A_tile, tile_m, tile_k, tile_k, A_buffer);
  const int words_b = PackTypeB(B_tile, tile_k, tile_n, B_buffer);

  return RunMatmulTilePacked(A_buffer, words_a, B_buffer, words_b, tile_m,
                             tile_k, tile_n, C_tile);
#endif
}

}  // namespace tpu_helper

#endif  // TPU_HELPER_H
