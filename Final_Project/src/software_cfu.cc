// Copyright 2021 The CFU-Playground Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "software_cfu.h"

#include <array>
#include <cstdint>

namespace {
constexpr uint32_t MakeFunctionId(int funct3, int funct7) {
  return (static_cast<uint32_t>(funct7 & 0x7f) << 3) |
         static_cast<uint32_t>(funct3 & 0x7);
}

constexpr int kTpuTile = 32;
constexpr int kABufferWords = 16384;
constexpr int kCBufferWords = 2048;
constexpr int kABIndexMask = kABufferWords - 1;
constexpr int kCIndexMask = kCBufferWords - 1;

inline int32_t Dot4WithOffset(uint32_t packed_inputs, uint32_t packed_filter,
                              int8_t input_offset) {
  int32_t acc = 0;
  for (int i = 0; i < 4; ++i) {
    const int8_t input_val = static_cast<int8_t>(packed_inputs >> (8 * i));
    const int8_t filter_val = static_cast<int8_t>(packed_filter >> (8 * i));
    const int16_t adjusted_input =
        static_cast<int16_t>(input_val) + input_offset;
    acc +=
        static_cast<int32_t>(adjusted_input) * static_cast<int32_t>(filter_val);
  }
  return acc;
}

inline int32_t ApplyQuantizedMultiplier(int32_t value, uint32_t multiplier,
                                        int8_t shift) {
  const int total_shift = 31 - static_cast<int>(shift);
  const int64_t mul = static_cast<int64_t>(value) *
                      static_cast<int64_t>(static_cast<int32_t>(multiplier));
  int64_t result = 0;
  if (total_shift > 0) {
    const int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
    result = (mul + round) >> total_shift;
  } else {
    result = mul << (-total_shift);
  }
  return static_cast<int32_t>(result);
}

struct TpuState {
  uint32_t a_words[2][kABufferWords] = {};
  uint32_t b_words[2][kABufferWords] = {};
  int32_t c_words[2][kCBufferWords][4] = {};
  uint8_t a_bank = 0;
  uint8_t b_bank = 0;
  uint8_t c_bank = 0;
  bool busy = false;
} g_tpu_state;

inline int ClampTileDim(int value) {
  if (value < 0) {
    return 0;
  }
  if (value > kTpuTile) {
    return kTpuTile;
  }
  return value;
}

inline int8_t ExtractPackedByte(uint32_t word, int lane) {
  const int shift = (3 - lane) * 8;
  return static_cast<int8_t>((word >> shift) & 0xff);
}

void ClearPackedC(uint8_t bank) {
  for (int i = 0; i < kCBufferWords; ++i) {
    for (int lane = 0; lane < 4; ++lane) {
      g_tpu_state.c_words[bank][i][lane] = 0;
    }
  }
}

void RunSoftwareTpu(int raw_m, int raw_k, int raw_n) {
  const int tile_m = ClampTileDim(raw_m);
  const int tile_k = ClampTileDim(raw_k);
  const int tile_n = ClampTileDim(raw_n);
  ClearPackedC(g_tpu_state.c_bank);

  if (tile_m == 0 || tile_k == 0 || tile_n == 0) {
    return;
  }

  std::array<int8_t, kTpuTile * kTpuTile> a_tile{};
  std::array<int8_t, kTpuTile * kTpuTile> b_tile{};
  std::array<int32_t, kTpuTile * kTpuTile> c_tile{};

  const int row_blocks = (tile_m + 3) / 4;
  for (int block = 0; block < row_blocks; ++block) {
    for (int col = 0; col < tile_k; ++col) {
      const int word_idx = block * tile_k + col;
      const uint32_t word =
          g_tpu_state.a_words[g_tpu_state.a_bank][word_idx & kABIndexMask];
      for (int lane = 0; lane < 4; ++lane) {
        const int row = block * 4 + lane;
        if (row < tile_m) {
          a_tile[row * tile_k + col] = ExtractPackedByte(word, lane);
        }
      }
    }
  }

  const int col_blocks = (tile_n + 3) / 4;
  for (int block = 0; block < col_blocks; ++block) {
    for (int row = 0; row < tile_k; ++row) {
      const int word_idx = block * tile_k + row;
      const uint32_t word =
          g_tpu_state.b_words[g_tpu_state.b_bank][word_idx & kABIndexMask];
      for (int lane = 0; lane < 4; ++lane) {
        const int col = block * 4 + lane;
        if (col < tile_n) {
          b_tile[row * tile_n + col] = ExtractPackedByte(word, lane);
        }
      }
    }
  }

  for (int row = 0; row < tile_m; ++row) {
    for (int col = 0; col < tile_n; ++col) {
      int32_t acc = 0;
      for (int kk = 0; kk < tile_k; ++kk) {
        const int8_t a_val = a_tile[row * tile_k + kk];
        const int8_t b_val = b_tile[kk * tile_n + col];
        acc += static_cast<int32_t>(a_val) * static_cast<int32_t>(b_val);
      }
      c_tile[row * kTpuTile + col] = acc;
    }
  }

  const int col_words = (tile_n + 3) / 4;
  const int stride_words = (col_words > 0) ? col_words : 1;
  for (int row = 0; row < tile_m; ++row) {
    for (int col_word = 0; col_word < col_words; ++col_word) {
      const int word_index = row * stride_words + col_word;
      for (int lane = 0; lane < 4; ++lane) {
        const int col = col_word * 4 + (3 - lane);
        int32_t value = 0;
        if (col < tile_n) {
          value = c_tile[row * kTpuTile + col];
        }
        g_tpu_state
            .c_words[g_tpu_state.c_bank][word_index & kCIndexMask][lane] =
            value;
      }
    }
  }
}
}  // namespace

// Software emulation of the custom CFU used in this project.
uint32_t software_cfu(int funct3, int funct7, uint32_t rs1, uint32_t rs2) {
  static int8_t input_offset = 0;
  static uint32_t quant_multiplier = 0;
  static int8_t quant_shift = 0;
  static int32_t accumulator = 0;
  const uint32_t function_id = MakeFunctionId(funct3, funct7);

  switch (function_id) {
    case MakeFunctionId(0, 0):  // Set offset and echo rs1
      input_offset = static_cast<int8_t>(rs1 & 0xff);
      return rs1;
    case MakeFunctionId(0, 1): {  // Dot4 with offset
      return static_cast<uint32_t>(Dot4WithOffset(rs1, rs2, input_offset));
    }
    case MakeFunctionId(0, 2): {  // Partial tail dot (1-3 lanes)
      const int lane_count = static_cast<int>((rs2 >> 24) & 0xff);
      int32_t acc = 0;
      for (int i = 0; i < lane_count && i < 3; ++i) {
        const int8_t input_val = static_cast<int8_t>(rs1 >> (8 * i));
        const int8_t filter_val = static_cast<int8_t>(rs2 >> (8 * i));
        const int16_t adjusted_input =
            static_cast<int16_t>(input_val) + input_offset;
        acc += static_cast<int32_t>(adjusted_input) *
               static_cast<int32_t>(filter_val);
      }
      return static_cast<uint32_t>(acc);
    }
    case MakeFunctionId(1, 0):  // Set quantized multiplier
      quant_multiplier = rs1;
      return rs1;
    case MakeFunctionId(1, 1):  // Set quantized shift (signed)
      quant_shift = static_cast<int8_t>(rs1 & 0xff);
      return rs1;
    case MakeFunctionId(1,
                        2): {  // Apply quantized multiplier with stored params
      const int32_t value = static_cast<int32_t>(rs1);
      return static_cast<uint32_t>(
          ApplyQuantizedMultiplier(value, quant_multiplier, quant_shift));
    }
    case MakeFunctionId(2, 0):  // Clear accumulator
      accumulator = 0;
      return 0;
    case MakeFunctionId(2, 1): {  // Accumulate dot4 result
      accumulator += Dot4WithOffset(rs1, rs2, input_offset);
      return static_cast<uint32_t>(accumulator);
    }
    case MakeFunctionId(2, 2):  // Read accumulator
      return static_cast<uint32_t>(accumulator);
    case MakeFunctionId(2, 3):  // Quantize accumulator in-place
      accumulator =
          ApplyQuantizedMultiplier(accumulator, quant_multiplier, quant_shift);
      return static_cast<uint32_t>(accumulator);
    case MakeFunctionId(2, 4):  // Accumulator += bias
      accumulator += static_cast<int32_t>(rs1);
      return static_cast<uint32_t>(accumulator);
    case MakeFunctionId(3, 0): {  // Write A buffer word
      const uint32_t index = (rs1 >> 16) & 0xffff;
      const uint32_t bank = (index >> 15) & 0x1;
      g_tpu_state.a_words[bank][index & kABIndexMask] = rs2;
      return 0;
    }
    case MakeFunctionId(3, 1): {  // Write B buffer word
      const uint32_t index = (rs1 >> 16) & 0xffff;
      const uint32_t bank = (index >> 15) & 0x1;
      g_tpu_state.b_words[bank][index & kABIndexMask] = rs2;
      return 0;
    }
    case MakeFunctionId(3, 2): {  // Start TPU computation
      const int tile_k = static_cast<int>((rs1 >> 24) & 0xff);
      const int tile_m = static_cast<int>((rs1 >> 16) & 0xff);
      const int tile_n = static_cast<int>((rs1 >> 8) & 0xff);
      const uint32_t was_busy = g_tpu_state.busy ? 1u : 0u;
      g_tpu_state.busy = true;
      RunSoftwareTpu(tile_m, tile_k, tile_n);
      g_tpu_state.busy = false;
      return was_busy;
    }
    case MakeFunctionId(3, 3): {  // Read packed C buffer
      const uint32_t index = (rs1 >> 16) & 0xffff;
      const uint32_t bank = (index >> 15) & 0x1;
      const uint32_t word_index = index & kCIndexMask;
      const uint8_t word_sel = static_cast<uint8_t>(rs1 & 0x3);
      return static_cast<uint32_t>(
          g_tpu_state.c_words[bank][word_index][word_sel & 0x3]);
    }
    case MakeFunctionId(3, 4):  // Query busy flag
      return g_tpu_state.busy ? 1u : 0u;
    case MakeFunctionId(3, 5):  // Select A buffer bank
      g_tpu_state.a_bank = static_cast<uint8_t>(rs1 & 0x1);
      return rs1;
    case MakeFunctionId(3, 6):  // Select B buffer bank
      g_tpu_state.b_bank = static_cast<uint8_t>(rs1 & 0x1);
      return rs1;
    case MakeFunctionId(3, 7):  // Select C buffer bank
      g_tpu_state.c_bank = static_cast<uint8_t>(rs1 & 0x1);
      return rs1;
    default:
      // Preserve original passthrough behavior for any other function IDs.
      return rs1;
  }
}
