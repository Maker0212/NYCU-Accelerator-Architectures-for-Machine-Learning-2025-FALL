/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions andss
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#define DEBUG_CONV_LOG 0
#ifndef CONV_PROFILE
#define CONV_PROFILE 0
#endif
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#define DEBUG_TPU_FIRST_OUTPUT 0
#if DEBUG_CONV_LOG || DEBUG_TPU_FIRST_OUTPUT || CONV_PROFILE
#include <cstdio>
#endif

#include "cfu.h"
#if CONV_PROFILE
#include "perf.h"
#endif
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tpu_helper.h"

namespace tflite {
namespace reference_integer_ops {

namespace detail {
// Packs four consecutive int8 values into a 32-bit word, little-endian.
// Uses byte assembly to avoid unaligned 32-bit loads on targets that fault.
inline uint32_t Pack4Bytes(const int8_t* values) {
  if (((uintptr_t)values & 3) == 0) {
    return *reinterpret_cast<const uint32_t*>(values);
  }
  return static_cast<uint8_t>(values[0]) |
         (static_cast<uint8_t>(values[1]) << 8) |
         (static_cast<uint8_t>(values[2]) << 16) |
         (static_cast<uint8_t>(values[3]) << 24);
}

inline uint32_t PackPartialBytes(const int8_t* values, int count) {
  uint32_t word = 0;
  for (int i = 0; i < count; ++i) {
    const uint32_t byte_val = static_cast<uint8_t>(values[i]);
    word |= byte_val << (8 * i);
  }
  return word;
}

struct FilterPrefixEntry {
  const int8_t* filter_ptr = nullptr;
  int output_depth = 0;
  int filter_height = 0;
  int filter_width = 0;
  int filter_input_depth = 0;
  int prefix_width = 0;
  int prefix_height = 0;
  std::vector<int32_t> prefix_data;

  size_t PrefixCellCount() const {
    return static_cast<size_t>(prefix_width) * prefix_height;
  }
};

inline FilterPrefixEntry& GetOrCreateFilterPrefixEntry(const int8_t* filter_ptr,
                                                       int output_depth,
                                                       int filter_height,
                                                       int filter_width,
                                                       int filter_input_depth) {
  static std::vector<FilterPrefixEntry> cache;
  for (auto& entry : cache) {
    if (entry.filter_ptr == filter_ptr) {
      return entry;
    }
  }

  cache.emplace_back();
  FilterPrefixEntry& entry = cache.back();
  entry.filter_ptr = filter_ptr;
  entry.output_depth = output_depth;
  entry.filter_height = filter_height;
  entry.filter_width = filter_width;
  entry.filter_input_depth = filter_input_depth;
  entry.prefix_width = filter_width + 1;
  entry.prefix_height = filter_height + 1;

  const size_t prefix_cells = entry.PrefixCellCount();
  entry.prefix_data.assign(static_cast<size_t>(output_depth) * prefix_cells, 0);

  const int filter_out_stride =
      filter_height * filter_width * filter_input_depth;
  const int filter_row_stride = filter_width * filter_input_depth;
  const int filter_col_stride = filter_input_depth;

  for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
    int32_t* prefix = entry.prefix_data.data() +
                      static_cast<size_t>(out_channel) * prefix_cells;
    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
      for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
        const int spatial_offset =
            filter_y * filter_row_stride + filter_x * filter_col_stride;
        int32_t cell_sum = 0;
        for (int in_channel = 0; in_channel < filter_input_depth;
             ++in_channel) {
          const int filter_idx =
              out_channel * filter_out_stride + spatial_offset + in_channel;
          cell_sum += static_cast<int32_t>(filter_ptr[filter_idx]);
        }
        const int y1 = filter_y + 1;
        const int x1 = filter_x + 1;
        const int idx11 = y1 * entry.prefix_width + x1;
        const int idx10 = y1 * entry.prefix_width + (x1 - 1);
        const int idx01 = (y1 - 1) * entry.prefix_width + x1;
        const int idx00 = (y1 - 1) * entry.prefix_width + (x1 - 1);
        prefix[idx11] =
            cell_sum + prefix[idx10] + prefix[idx01] - prefix[idx00];
      }
    }
  }

  return entry;
}

struct FilterPackedBEntry {
  const int8_t* filter_ptr = nullptr;
  int output_depth = 0;
  int filter_height = 0;
  int filter_width = 0;
  int filter_input_depth = 0;
  int num_k_blocks = 0;
  int num_n_tiles = 0;
  std::vector<uint32_t> packed_b;
  std::vector<uint16_t> packed_words;
  std::vector<uint8_t> packed_valid;
};

// Cache for weight sums (used for input_offset correction)
struct WeightSumCache {
  const int8_t* filter_ptr = nullptr;
  int output_depth = 0;
  int K = 0;
  std::vector<int32_t> w_sums;  // per output channel
};

inline WeightSumCache& GetOrCreateWeightSumCache(const int8_t* filter_ptr,
                                                 int output_depth, int K) {
  static std::vector<WeightSumCache> cache;
  for (auto& entry : cache) {
    if (entry.filter_ptr == filter_ptr && entry.output_depth == output_depth &&
        entry.K == K) {
      return entry;
    }
  }

  cache.emplace_back();
  WeightSumCache& entry = cache.back();
  entry.filter_ptr = filter_ptr;
  entry.output_depth = output_depth;
  entry.K = K;
  entry.w_sums.resize(output_depth, 0);

  // Pre-compute weight sums for all output channels
  for (int oc = 0; oc < output_depth; ++oc) {
    const int8_t* w_ptr = filter_ptr + oc * K;
    int32_t sum = 0;
    int k = 0;
    // Unroll by 8 for speed
    for (; k + 7 < K; k += 8) {
      sum += w_ptr[k] + w_ptr[k + 1] + w_ptr[k + 2] + w_ptr[k + 3];
      sum += w_ptr[k + 4] + w_ptr[k + 5] + w_ptr[k + 6] + w_ptr[k + 7];
    }
    for (; k < K; ++k) {
      sum += w_ptr[k];
    }
    entry.w_sums[oc] = sum;
  }

  return entry;
}

#if CONV_PROFILE
inline uint64_t ConvProfileStart() { return perf_get_mcycle64(); }

inline void LogConvProfile(uint64_t start_cycles, const int8_t* filter_ptr,
                           int input_h, int input_w, int input_d, int filter_h,
                           int filter_w, int output_d, int output_h,
                           int output_w, int stride_h, int stride_w,
                           int dilation_h, int dilation_w) {
  static uint32_t call_index = 0;
  const uint64_t end_cycles = perf_get_mcycle64();
  const uint64_t delta = end_cycles - start_cycles;
  printf(
      "[conv_profile] #%lu filter=%p in=%dx%dx%d f=%dx%dx%dx%d out=%dx%dx%d "
      "s=%dx%d d=%dx%d cycles=%llu\n",
      static_cast<unsigned long>(call_index),
      static_cast<const void*>(filter_ptr), input_h, input_w, input_d, output_d,
      filter_h, filter_w, input_d, output_h, output_w, output_d, stride_h,
      stride_w, dilation_h, dilation_w, static_cast<unsigned long long>(delta));
  ++call_index;
}
#else
inline uint64_t ConvProfileStart() { return 0; }
inline void LogConvProfile(uint64_t, const int8_t*, int, int, int, int, int,
                           int, int, int, int, int, int, int) {}
#endif

inline FilterPackedBEntry& GetOrCreateFilterPackedBEntry(
    const int8_t* filter_ptr, int output_depth, int filter_height,
    int filter_width, int filter_input_depth) {
  static std::vector<FilterPackedBEntry> cache;
  for (auto& entry : cache) {
    if (entry.filter_ptr == filter_ptr) {
      return entry;
    }
  }

  cache.emplace_back();
  FilterPackedBEntry& entry = cache.back();
  entry.filter_ptr = filter_ptr;
  entry.output_depth = output_depth;
  entry.filter_height = filter_height;
  entry.filter_width = filter_width;
  entry.filter_input_depth = filter_input_depth;

  const int tile_size = tpu_helper::kTileSize;
  const int patch_size = filter_height * filter_width * filter_input_depth;
  entry.num_k_blocks = (patch_size + tile_size - 1) / tile_size;
  entry.num_n_tiles = (output_depth + tile_size - 1) / tile_size;

  const size_t cache_slots =
      static_cast<size_t>(entry.num_k_blocks) * entry.num_n_tiles;
  entry.packed_b.assign(cache_slots * tpu_helper::kPackedWords, 0);
  entry.packed_words.assign(cache_slots, 0);
  entry.packed_valid.assign(cache_slots, 0);
  return entry;
}

}  // namespace detail

namespace {

constexpr size_t kScratchAlignmentBytes = 16;

inline int8_t* EnsureAlignedStorage(std::vector<int8_t>& storage,
                                    size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  const size_t needed = bytes + (kScratchAlignmentBytes - 1);
  if (storage.size() < needed) {
    storage.resize(needed);
  }
  uintptr_t base_addr = reinterpret_cast<uintptr_t>(storage.data());
  const uintptr_t mask = static_cast<uintptr_t>(kScratchAlignmentBytes - 1);
  uintptr_t aligned_addr = (base_addr + mask) & ~mask;
  return reinterpret_cast<int8_t*>(aligned_addr);
}

inline std::vector<int8_t>& GetIm2colScratchStorage() {
  static std::vector<int8_t> buffer;
  return buffer;
}

inline std::vector<int8_t>& GetACacheStorage() {
  static std::vector<int8_t> buffer;
  return buffer;
}

inline std::vector<int32_t>& GetKInChannelStorage() {
  static std::vector<int32_t> buffer;
  return buffer;
}

inline std::vector<int32_t>& GetKFilterXStorage() {
  static std::vector<int32_t> buffer;
  return buffer;
}

inline std::vector<int32_t>& GetKFilterYStorage() {
  static std::vector<int32_t> buffer;
  return buffer;
}

inline std::vector<int32_t>& GetBiasCorrectionStorage() {
  static std::vector<int32_t> buffer;
  return buffer;
}

inline std::vector<int>& GetKBlockSizesStorage() {
  static std::vector<int> buffer;
  return buffer;
}

inline std::vector<uint8_t>& GetCacheFilledStorage() {
  static std::vector<uint8_t> buffer;
  return buffer;
}

inline std::vector<uint8_t>& GetPackedReadyStorage() {
  static std::vector<uint8_t> buffer;
  return buffer;
}

inline std::vector<uint32_t>& GetPackedCacheStorage() {
  static std::vector<uint32_t> buffer;
  return buffer;
}

inline std::vector<uint16_t>& GetPackedWordsStorage() {
  static std::vector<uint16_t> buffer;
  return buffer;
}

inline std::vector<uint32_t>& GetPackedOffsetsStorage() {
  static std::vector<uint32_t> buffer;
  return buffer;
}

inline std::vector<int32_t>& GetAccumulationStorage() {
  static std::vector<int32_t> buffer;
  return buffer;
}

inline std::vector<int8_t>& GetBCacheStorage() {
  static std::vector<int8_t> buffer;
  return buffer;
}

inline int CeilDivInt(int a, int b) {
  TFLITE_DCHECK_GT(b, 0);
  if (a >= 0) {
    return (a + b - 1) / b;
  }
  return a / b;
}

inline int FloorDivInt(int a, int b) {
  TFLITE_DCHECK_GT(b, 0);
  if (a >= 0) {
    return a / b;
  }
  return -(((-a) + b - 1) / b);
}

inline void ComputeValidFilterRange(int origin, int dilation, int limit,
                                    int filter_dim, int* start, int* end) {
  TFLITE_DCHECK_GT(dilation, 0);
  TFLITE_DCHECK_GT(limit, 0);
  int start_idx = 0;
  if (origin < 0) {
    start_idx = CeilDivInt(-origin, dilation);
    if (start_idx < 0) {
      start_idx = 0;
    } else if (start_idx > filter_dim) {
      start_idx = filter_dim;
    }
  }
  int end_idx = filter_dim;
  const int max_input = limit - 1;
  if (origin + dilation * (filter_dim - 1) > max_input) {
    end_idx = FloorDivInt(max_input - origin, dilation) + 1;
    if (end_idx < 0) {
      end_idx = 0;
    } else if (end_idx > filter_dim) {
      end_idx = filter_dim;
    }
  }
  if (start_idx > end_idx) {
    start_idx = end_idx;
  }
  *start = start_idx;
  *end = end_idx;
}

inline void ConvPerChannelReferenceKernel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  const int32_t input_offset = params.input_offset;
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const bool can_use_cfu = (filter_input_depth > 0);
  if (can_use_cfu && groups == 1) {
    // TPU Implementation for standard convolution with cached filter packing.
    static int32_t acc_buffer[32 * 32];
    static int32_t tmp_out[32 * 32];

    const int K_total = filter_height * filter_width * filter_input_depth;
    const int output_size = output_height * output_width;
    const int k_blocks = (K_total + 31) / 32;

    for (int batch = 0; batch < batches; ++batch) {
      for (int n_base = 0; n_base < output_depth; n_base += 32) {
        const int n_count = std::min(32, output_depth - n_base);

        // Pre-calculate filter sums for offset correction
        std::vector<int32_t> filter_sums(n_count, 0);
        for (int n = 0; n < n_count; ++n) {
          const int out_c = n_base + n;
          const int8_t* f_ptr = filter_data + out_c * K_total;
          int32_t sum = 0;
          for (int k = 0; k < K_total; ++k) {
            sum += f_ptr[k];
          }
          filter_sums[n] = sum;
        }

        // Pack B (Filters) directly into uint32_t buffer
        std::vector<uint32_t> packed_B_buffer(k_blocks *
                                              tpu_helper::kPackedWords);
        std::vector<int> k_base_offsets(k_blocks);
        std::vector<int> k_counts(k_blocks);

        for (int blk = 0; blk < k_blocks; ++blk) {
          const int k_base = blk * 32;
          const int k_count = std::min(32, K_total - k_base);
          k_base_offsets[blk] = k_base;
          k_counts[blk] = k_count;

          uint32_t* p_b =
              packed_B_buffer.data() + blk * tpu_helper::kPackedWords;

          // Pack 32 columns (n) in groups of 4.
          for (int col_block = 0; col_block < 8; ++col_block) {
            for (int row = 0; row < 32; ++row) {  // row is k inside tile
              uint32_t packed_word = 0;
              for (int lane = 0; lane < 4; ++lane) {
                const int n_idx = col_block * 4 + lane;
                const int k_idx_tile = row;

                int8_t val = 0;
                if (n_idx < n_count && k_idx_tile < k_count) {
                  const int out_c = n_base + n_idx;
                  const int k_global = k_base + k_idx_tile;
                  const int in_channel = k_global % filter_input_depth;
                  const int rem = k_global / filter_input_depth;
                  const int filter_x = rem % filter_width;
                  const int filter_y = rem / filter_width;
                  val = filter_data[Offset(filter_shape, out_c, filter_y,
                                           filter_x, in_channel)];
                }
                packed_word |= static_cast<uint8_t>(val) << ((3 - lane) * 8);
              }
              p_b[col_block * 32 + row] = packed_word;
            }
          }
        }

        for (int m_base = 0; m_base < output_size; m_base += 32) {
          const int m_count = std::min(32, output_size - m_base);
          std::fill(acc_buffer, acc_buffer + 32 * 32, 0);

          // [Optimized] Pre-calculate m-dependent values
          int in_y_origins[32];
          int in_x_origins[32];
          int32_t m_addr_offsets[32];
          const int input_row_stride_local = input_width * input_depth;
          const int32_t batch_offset =
              batch * input_height * input_row_stride_local;

          for (int m = 0; m < m_count; ++m) {
            const int idx = m_base + m;
            const int out_y = idx / output_width;
            const int out_x = idx % output_width;
            in_y_origins[m] = (out_y * stride_height) - pad_height;
            in_x_origins[m] = (out_x * stride_width) - pad_width;
            m_addr_offsets[m] = batch_offset +
                                in_y_origins[m] * input_row_stride_local +
                                in_x_origins[m] * input_depth;
          }

          const int8_t dummy_pad_val = static_cast<int8_t>(-input_offset);

          for (int blk = 0; blk < k_blocks; ++blk) {
            const int k_base = k_base_offsets[blk];
            const int k_count = k_counts[blk];

            uint32_t packed_A[tpu_helper::kPackedWords];

            // Pack A (Inputs) directly
            for (int row_block = 0; row_block < 8; ++row_block) {
              for (int col = 0; col < 32; ++col) {  // col is k inside tile
                uint32_t packed_word = 0;
                for (int lane = 0; lane < 4; ++lane) {
                  const int m_idx = row_block * 4 + lane;
                  const int k_idx_tile = col;

                  int8_t val = 0;
                  if (m_idx < m_count && k_idx_tile < k_count) {
                    const int k_global = k_base + k_idx_tile;
                    const int in_channel = k_global % filter_input_depth;
                    const int rem = k_global / filter_input_depth;
                    const int filter_x = rem % filter_width;
                    const int filter_y = rem / filter_width;

                    const int fy_offset = dilation_height_factor * filter_y;
                    const int fx_offset = dilation_width_factor * filter_x;

                    const int in_y = in_y_origins[m_idx] + fy_offset;
                    const int in_x = in_x_origins[m_idx] + fx_offset;

                    if (in_x >= 0 && in_x < input_width && in_y >= 0 &&
                        in_y < input_height) {
                      const int k_addr_offset =
                          fy_offset * input_row_stride_local +
                          fx_offset * input_depth + in_channel;
                      val = input_data[m_addr_offsets[m_idx] + k_addr_offset];
                    } else {
                      val = dummy_pad_val;
                    }
                  } else if (m_idx < m_count) {
                    // k padding -> 0
                    val = 0;
                  } else {
                    // m padding -> 0
                    val = 0;
                  }
                  packed_word |= static_cast<uint8_t>(val) << ((3 - lane) * 8);
                }
                packed_A[row_block * 32 + col] = packed_word;
              }
            }

            tpu_helper::RunMatmulTilePacked(
                packed_A, 256,
                packed_B_buffer.data() + blk * tpu_helper::kPackedWords, 256,
                32, 32, 32, tmp_out);
            for (int i = 0; i < 32 * 32; ++i) {
              acc_buffer[i] += tmp_out[i];
            }
          }

          for (int m = 0; m < m_count; ++m) {
            const int idx = m_base + m;
            const int out_y = idx / output_width;
            const int out_x = idx % output_width;
            for (int n = 0; n < n_count; ++n) {
              const int out_c = n_base + n;
              int32_t acc = acc_buffer[m * 32 + n];
#if DEBUG_CONV_LOG
              if (batch == 0 && out_y == 0 && out_x == 0 && out_c == 0) {
                printf("DEBUG: input_offset=%d\n", (int)input_offset);
                printf("DEBUG: dummy_pad_val=%d\n",
                       (int)static_cast<int8_t>(-input_offset));
                printf("DEBUG: acc_buffer[0]=%d\n", (int)acc);
                printf("DEBUG: filter_sums[0]=%d\n", (int)filter_sums[n]);
                printf("DEBUG: bias_data[0]=%d\n",
                       (int)(bias_data ? bias_data[out_c] : 0));
                printf("DEBUG: acc_before_mul=%d\n",
                       (int)(acc + input_offset * filter_sums[n] +
                             (bias_data ? bias_data[out_c] : 0)));
                fflush(stdout);
              }
#endif
              acc += input_offset * filter_sums[n];
              if (bias_data) {
                acc += bias_data[out_c];
              }
              acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[out_c],
                                                  output_shift[out_c]);
              acc += output_offset;
              acc = std::max(acc, output_activation_min);
              acc = std::min(acc, output_activation_max);
              output_data[Offset(output_shape, batch, out_y, out_x, out_c)] =
                  static_cast<int8_t>(acc);
            }
          }
        }
      }
    }
    return;
  } else if (can_use_cfu) {
    const int8_t packed_input_offset = static_cast<int8_t>(input_offset);
    cfu_op0(0, static_cast<uint8_t>(packed_input_offset), 0);
  }

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          int32_t acc = 0;
          int32_t cfu_tail_acc = 0;
          if (can_use_cfu) {
            cfu_op2(0, 0, 0);
          }
          if (can_use_cfu) {
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              const int in_y = in_y_origin + dilation_height_factor * filter_y;
              const int input_row_base =
                  ((batch * input_height + in_y) * input_width) * input_depth +
                  group * filter_input_depth;
              const int filter_row_base =
                  ((out_channel * filter_height + filter_y) * filter_width) *
                  filter_input_depth;
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height);

                if (!is_point_inside_image) {
                  continue;
                }

                const int input_base = input_row_base + in_x * input_depth;
                const int filter_base =
                    filter_row_base + filter_x * filter_input_depth;
                const int8_t* input_ptr = input_data + input_base;
                const int8_t* filter_ptr = filter_data + filter_base;

                int in_channel = 0;
                for (; in_channel + 15 < filter_input_depth; in_channel += 16) {
                  const uint32_t packed_input0 =
                      detail::Pack4Bytes(input_ptr + in_channel);
                  const uint32_t packed_filter0 =
                      detail::Pack4Bytes(filter_ptr + in_channel);
                  const uint32_t packed_input1 =
                      detail::Pack4Bytes(input_ptr + in_channel + 4);
                  const uint32_t packed_filter1 =
                      detail::Pack4Bytes(filter_ptr + in_channel + 4);
                  const uint32_t packed_input2 =
                      detail::Pack4Bytes(input_ptr + in_channel + 8);
                  const uint32_t packed_filter2 =
                      detail::Pack4Bytes(filter_ptr + in_channel + 8);
                  const uint32_t packed_input3 =
                      detail::Pack4Bytes(input_ptr + in_channel + 12);
                  const uint32_t packed_filter3 =
                      detail::Pack4Bytes(filter_ptr + in_channel + 12);
                  cfu_op2(1, packed_input0, packed_filter0);
                  cfu_op2(1, packed_input1, packed_filter1);
                  cfu_op2(1, packed_input2, packed_filter2);
                  cfu_op2(1, packed_input3, packed_filter3);
                }
                for (; in_channel + 7 < filter_input_depth; in_channel += 8) {
                  const uint32_t packed_input0 =
                      detail::Pack4Bytes(input_ptr + in_channel);
                  const uint32_t packed_filter0 =
                      detail::Pack4Bytes(filter_ptr + in_channel);
                  const uint32_t packed_input1 =
                      detail::Pack4Bytes(input_ptr + in_channel + 4);
                  const uint32_t packed_filter1 =
                      detail::Pack4Bytes(filter_ptr + in_channel + 4);
                  cfu_op2(1, packed_input0, packed_filter0);
                  cfu_op2(1, packed_input1, packed_filter1);
                }
                for (; in_channel + 3 < filter_input_depth; in_channel += 4) {
                  const uint32_t packed_input =
                      detail::Pack4Bytes(input_ptr + in_channel);
                  const uint32_t packed_filter =
                      detail::Pack4Bytes(filter_ptr + in_channel);
                  cfu_op2(1, packed_input, packed_filter);
                }
                if (in_channel < filter_input_depth) {
                  const int tail_lanes =
                      std::min(filter_input_depth - in_channel, 3);
                  const uint32_t packed_input = detail::PackPartialBytes(
                      input_ptr + in_channel, tail_lanes);
                  const uint32_t packed_filter = detail::PackPartialBytes(
                      filter_ptr + in_channel, tail_lanes);
                  const uint32_t lane_header =
                      static_cast<uint32_t>(tail_lanes & 0xff) << 24;
                  const uint32_t encoded_filter =
                      lane_header | (packed_filter & 0x00ffffffu);
                  cfu_tail_acc += static_cast<int32_t>(
                      cfu_op0(2, packed_input, encoded_filter));
                  in_channel = filter_input_depth;
                }
              }
            }
            acc = static_cast<int32_t>(cfu_op2(2, 0, 0));
            acc += cfu_tail_acc;
          } else {
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              const int in_y = in_y_origin + dilation_height_factor * filter_y;
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height);

                if (!is_point_inside_image) {
                  continue;
                }

                for (int in_channel = 0; in_channel < filter_input_depth;
                     ++in_channel) {
                  const int32_t input_val = input_data[Offset(
                      input_shape, batch, in_y, in_x,
                      in_channel + group * filter_input_depth)];
                  const int32_t filter_val =
                      filter_data[Offset(filter_shape, out_channel, filter_y,
                                         filter_x, in_channel)];
                  acc += filter_val * (input_val + input_offset);
                }
              }
            }
          }

          if (bias_data) {
            acc += bias_data[out_channel];
          }
          acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int8_t>(acc);
        }
      }
    }
  }
}

}  // namespace

// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  // --------------------------------------------
  // 0. Hardware Config
  // --------------------------------------------
  const int MAX_DEPTH_BUFFER = 2048;
  int8_t dummy_pad_buffer[MAX_DEPTH_BUFFER];

  // --------------------------------------------
  // 1. Extract parameters
  // --------------------------------------------
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;

  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_w = params.dilation_width_factor;
  const int dilation_h = params.dilation_height_factor;
  const int pad_w = params.padding_values.width;
  const int pad_h = params.padding_values.height;

  const int32_t act_min = params.quantized_activation_min;
  const int32_t act_max = params.quantized_activation_max;

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);

  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);

  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);

#if CONV_PROFILE
  const uint64_t conv_profile_start = detail::ConvProfileStart();
#endif

  const int K = filter_height * filter_width * input_depth;
  const int M = output_height * output_width;

  const int input_row_stride = input_width * input_depth;
  const int filter_ch_stride = filter_height * filter_width * input_depth;
  const int output_row_stride = output_width * output_depth;

  // Check if K fits in hardware buffer (16384 entries)
  if (K > 16384) {
    ConvPerChannelReferenceKernel(params, output_multiplier, output_shift,
                                  input_shape, input_data, filter_shape,
                                  filter_data, bias_shape, bias_data,
                                  output_shape, output_data);
#if CONV_PROFILE
    detail::LogConvProfile(
        conv_profile_start, filter_data, input_height, input_width, input_depth,
        filter_height, filter_width, output_depth, output_height, output_width,
        stride_height, stride_width, dilation_h, dilation_w);
#endif
    return;
  }

  // --------------------------------------------
  // [Optimization] Init Dummy Buffer
  // --------------------------------------------
  const int8_t pad_value = (int8_t)(-input_offset);
  int fill_depth =
      (input_depth < MAX_DEPTH_BUFFER) ? input_depth : MAX_DEPTH_BUFFER;
  memset(dummy_pad_buffer, pad_value, fill_depth);

  // --------------------------------------------
  // [Optimization] Pre-compute weight sums (once per layer)
  // --------------------------------------------
  detail::WeightSumCache& w_sum_cache =
      detail::GetOrCreateWeightSumCache(filter_data, output_depth, K);
  const int32_t* cached_w_sums = w_sum_cache.w_sums.data();

  std::vector<int32_t>& bias_correction_storage = GetBiasCorrectionStorage();
  if (bias_correction_storage.size() < static_cast<size_t>(output_depth)) {
    bias_correction_storage.resize(output_depth);
  }
  int32_t* bias_correction = bias_correction_storage.data();
  if (bias_data) {
    for (int ch = 0; ch < output_depth; ++ch) {
      bias_correction[ch] = bias_data[ch] + cached_w_sums[ch] * input_offset;
    }
  } else {
    for (int ch = 0; ch < output_depth; ++ch) {
      bias_correction[ch] = cached_w_sums[ch] * input_offset;
    }
  }

  constexpr uint32_t kAddrStep = 1u << 16;
  constexpr uint32_t kAddrBankBit = 1u << 31;
  constexpr int kCBufferEntries = 2048;
  constexpr int kMaxAccumEntries = 1 << 15;

  struct TileParams {
    int col_blocks;
    int tile_m;
    int tile_n;
  };

  auto ComputeTileParams = [&](int k_value) -> TileParams {
    const int max_blocks_by_k = std::min(16384 / k_value, 63);
    int max_col_blocks = std::min(max_blocks_by_k, (output_depth + 3) / 4);
    if (max_col_blocks < 1) max_col_blocks = 1;

    const int max_m_by_a = max_blocks_by_k * 4;
    int best_col_blocks = 1;
    int best_tile_m = 1;
    int best_cost = std::numeric_limits<int>::max();
    for (int cb = 1; cb <= max_col_blocks; ++cb) {
      int max_m_by_c = kCBufferEntries / cb;
      if (max_m_by_c < 1) max_m_by_c = 1;
      int tile_m = std::min(max_m_by_a, max_m_by_c);
      if (tile_m > 255) tile_m = 255;
      if (tile_m < 1) tile_m = 1;

      const int tile_n = cb * 4;
      const int num_out_ch_tiles = CeilDivInt(output_depth, tile_n);
      const int num_slide_tiles = CeilDivInt(M, tile_m);
      int cost = num_out_ch_tiles * num_slide_tiles;
      if ((cb & 1) && cb > 1) {
        cost += num_slide_tiles;
      }
      if (cost < best_cost ||
          (cost == best_cost && (cb * 4) > (best_col_blocks * 4))) {
        best_cost = cost;
        best_col_blocks = cb;
        best_tile_m = tile_m;
      }
    }

    return TileParams{best_col_blocks, best_tile_m, best_col_blocks * 4};
  };

  const TileParams base_params = ComputeTileParams(K);
  const int base_out_tiles = CeilDivInt(output_depth, base_params.tile_n);
  const int base_slide_tiles = CeilDivInt(M, base_params.tile_m);
  int best_cost = base_out_tiles * base_slide_tiles;
  int best_k_block = K;
  TileParams best_params = base_params;
  int best_spatial_block = filter_height * filter_width;
  int best_k_blocks = 1;

  const int spatial_size = filter_height * filter_width;
  const bool consider_blocking =
      (base_params.tile_m < 16 || base_params.tile_n < 16) &&
      (spatial_size > 1);
  if (consider_blocking) {
    for (int spatial_block = 1; spatial_block <= spatial_size;
         ++spatial_block) {
      if (spatial_size % spatial_block != 0) {
        continue;
      }
      const int k_block = spatial_block * input_depth;
      if (k_block <= 0 || k_block >= K || k_block > 16384) {
        continue;
      }
      const TileParams candidate_params = ComputeTileParams(k_block);
      const int num_out_ch_tiles =
          CeilDivInt(output_depth, candidate_params.tile_n);
      const int num_slide_tiles = CeilDivInt(M, candidate_params.tile_m);
      const int num_k_blocks = spatial_size / spatial_block;
      const int cost = num_out_ch_tiles * num_slide_tiles * num_k_blocks;
      const int accum_entries =
          num_slide_tiles * candidate_params.tile_m * candidate_params.tile_n;
      if (accum_entries > kMaxAccumEntries) {
        continue;
      }
      if (cost < best_cost || (cost == best_cost && k_block > best_k_block)) {
        best_cost = cost;
        best_k_block = k_block;
        best_params = candidate_params;
        best_spatial_block = spatial_block;
        best_k_blocks = num_k_blocks;
      }
    }
  }

  constexpr bool kForceW2LOp17KBlock = true;
  if (kForceW2LOp17KBlock && filter_height == 1 && filter_width == 32 &&
      input_depth == 250 && output_depth == 2000) {
    constexpr int kForcedSpatialBlocks[] = {4, 8, 16};
    int forced_best_cost = std::numeric_limits<int>::max();
    int forced_best_spatial_block = 0;
    int forced_best_k_block = 0;
    TileParams forced_best_params = best_params;

    for (int forced_spatial_block : kForcedSpatialBlocks) {
      if (spatial_size % forced_spatial_block != 0) {
        continue;
      }
      const int forced_k_block = forced_spatial_block * input_depth;
      if (forced_k_block <= 0 || forced_k_block >= K ||
          forced_k_block > 16384) {
        continue;
      }
      const TileParams forced_params = ComputeTileParams(forced_k_block);
      const int forced_num_out_tiles =
          CeilDivInt(output_depth, forced_params.tile_n);
      const int forced_num_slide_tiles = CeilDivInt(M, forced_params.tile_m);
      const int forced_num_k_blocks = spatial_size / forced_spatial_block;
      const int forced_cost =
          forced_num_out_tiles * forced_num_slide_tiles * forced_num_k_blocks;
      const int forced_accum_entries =
          forced_num_slide_tiles * forced_params.tile_m * forced_params.tile_n;
      if (forced_accum_entries > kMaxAccumEntries) {
        continue;
      }
      if (forced_cost < forced_best_cost ||
          (forced_cost == forced_best_cost &&
           forced_k_block > forced_best_k_block)) {
        forced_best_cost = forced_cost;
        forced_best_spatial_block = forced_spatial_block;
        forced_best_k_block = forced_k_block;
        forced_best_params = forced_params;
      }
    }

    if (forced_best_cost != std::numeric_limits<int>::max()) {
      best_k_block = forced_best_k_block;
      best_params = forced_best_params;
      best_spatial_block = forced_best_spatial_block;
      best_k_blocks = spatial_size / forced_best_spatial_block;
    }
  }

  // Depth-blocked 1x1 conv heuristic (targets the heavy 2000x2000 layer).
  bool use_depth_block = false;
  int depth_k_block = 0;
  int depth_k_blocks = 0;
  TileParams depth_params = base_params;
  if (filter_height == 1 && filter_width == 1 && stride_height == 1 &&
      stride_width == 1 && dilation_h == 1 && dilation_w == 1 &&
      input_depth >= 1024 && input_depth <= MAX_DEPTH_BUFFER &&
      output_depth >= 1024) {
    constexpr int kDepthBlockCandidates[] = {500, 1000};
    int best_depth_cost = std::numeric_limits<int>::max();
    for (int k_block : kDepthBlockCandidates) {
      if (k_block <= 0 || k_block > 16384) {
        continue;
      }
      if (input_depth % k_block != 0) {
        continue;
      }
      const TileParams candidate_params = ComputeTileParams(k_block);
      const int num_out_ch_tiles =
          CeilDivInt(output_depth, candidate_params.tile_n);
      const int num_slide_tiles = CeilDivInt(M, candidate_params.tile_m);
      const int num_k_blocks = input_depth / k_block;
      const int cost = num_out_ch_tiles * num_slide_tiles * num_k_blocks;
      const int accum_entries =
          num_slide_tiles * candidate_params.tile_m * candidate_params.tile_n;
      if (accum_entries > kMaxAccumEntries) {
        continue;
      }
      if (cost < best_depth_cost ||
          (cost == best_depth_cost && k_block > depth_k_block)) {
        best_depth_cost = cost;
        depth_k_block = k_block;
        depth_k_blocks = num_k_blocks;
        depth_params = candidate_params;
      }
    }
    if (depth_k_block > 0 && depth_k_block < input_depth) {
      use_depth_block = true;
    }
  }

  const bool use_k_block = (best_k_block < K);

  // --------------------------------------------
  // CFU Initialization
  // --------------------------------------------
  cfu_op0(
      0, input_offset,
      0);  // Set input offset (Note: TPU ignores this, we handle it manually)

  for (int batch = 0; batch < batches; ++batch) {
    const int8_t* input_batch_base =
        input_data + batch * input_height * input_row_stride;
    int8_t* output_batch_base =
        output_data + batch * output_height * output_row_stride;

    if (use_depth_block) {
      const int tile_m = depth_params.tile_m;
      const int tile_n = depth_params.tile_n;
      const int num_slide_tiles = CeilDivInt(M, tile_m);
      const uint32_t bank_addr_flags[2] = {0u, kAddrBankBit};

      const size_t accum_entries =
          static_cast<size_t>(num_slide_tiles) * tile_m * tile_n;
      auto& accum_storage = GetAccumulationStorage();
      if (accum_storage.size() < accum_entries) {
        accum_storage.resize(accum_entries);
      }
      int32_t* accum_data = accum_storage.data();

      auto PackWeightsBlock = [&](int out_ch, int valid_channels,
                                  int total_blocks, uint32_t addr_flag,
                                  int k_start, int k_count) {
        const int full_blocks = valid_channels / 4;
        for (int block = 0; block < full_blocks; ++block) {
          const int ch_base = out_ch + block * 4;
          const int8_t* w_ptr0 =
              filter_data + ch_base * filter_ch_stride + k_start;
          const int8_t* w_ptr1 = w_ptr0 + filter_ch_stride;
          const int8_t* w_ptr2 = w_ptr1 + filter_ch_stride;
          const int8_t* w_ptr3 = w_ptr2 + filter_ch_stride;

          uint32_t addr =
              (static_cast<uint32_t>(block * k_count) << 16) | addr_flag;
          int k = 0;
          for (; k + 3 < k_count; k += 4) {
            uint32_t w_pack0 =
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr0++)) << 24) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr1++)) << 16) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr2++)) << 8) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr3++)));
            cfu_op3(1, addr, w_pack0);
            addr += kAddrStep;

            uint32_t w_pack1 =
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr0++)) << 24) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr1++)) << 16) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr2++)) << 8) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr3++)));
            cfu_op3(1, addr, w_pack1);
            addr += kAddrStep;

            uint32_t w_pack2 =
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr0++)) << 24) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr1++)) << 16) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr2++)) << 8) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr3++)));
            cfu_op3(1, addr, w_pack2);
            addr += kAddrStep;

            uint32_t w_pack3 =
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr0++)) << 24) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr1++)) << 16) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr2++)) << 8) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr3++)));
            cfu_op3(1, addr, w_pack3);
            addr += kAddrStep;
          }
          for (; k < k_count; ++k) {
            uint32_t w_pack =
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr0++)) << 24) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr1++)) << 16) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr2++)) << 8) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr3++)));
            cfu_op3(1, addr, w_pack);
            addr += kAddrStep;
          }
        }

        if (full_blocks < total_blocks) {
          const int ch_base = out_ch + full_blocks * 4;
          const int8_t* w_ptr0 =
              (ch_base < output_depth)
                  ? filter_data + ch_base * filter_ch_stride + k_start
                  : nullptr;
          const int8_t* w_ptr1 =
              (ch_base + 1 < output_depth)
                  ? filter_data + (ch_base + 1) * filter_ch_stride + k_start
                  : nullptr;
          const int8_t* w_ptr2 =
              (ch_base + 2 < output_depth)
                  ? filter_data + (ch_base + 2) * filter_ch_stride + k_start
                  : nullptr;
          const int8_t* w_ptr3 =
              (ch_base + 3 < output_depth)
                  ? filter_data + (ch_base + 3) * filter_ch_stride + k_start
                  : nullptr;

          uint32_t addr =
              (static_cast<uint32_t>(full_blocks * k_count) << 16) | addr_flag;
          for (int k = 0; k < k_count; ++k) {
            uint32_t w_pack =
                (static_cast<uint32_t>(w_ptr0 ? static_cast<uint8_t>(*w_ptr0++)
                                              : 0u)
                 << 24) |
                (static_cast<uint32_t>(w_ptr1 ? static_cast<uint8_t>(*w_ptr1++)
                                              : 0u)
                 << 16) |
                (static_cast<uint32_t>(w_ptr2 ? static_cast<uint8_t>(*w_ptr2++)
                                              : 0u)
                 << 8) |
                (static_cast<uint32_t>(w_ptr3 ? static_cast<uint8_t>(*w_ptr3++)
                                              : 0u));
            cfu_op3(1, addr, w_pack);
            addr += kAddrStep;
          }
        }
      };

      auto PackInputTileDepthBlock = [&](int slide, int k_start, int k_count,
                                         uint32_t addr_base) {
        int oy[4], ox[4], iy0[4], ix0[4];
        bool valid_slide[4];
        bool fast_path = true;

        for (int s = 0; s < 4; ++s) {
          int idx = slide + s;
          if (idx < M) {
            oy[s] = idx / output_width;
            ox[s] = idx % output_width;
            iy0[s] = oy[s] * stride_height - pad_h;
            ix0[s] = ox[s] * stride_width - pad_w;
            valid_slide[s] = true;
            if (iy0[s] < 0 || iy0[s] >= input_height || ix0[s] < 0 ||
                ix0[s] >= input_width) {
              fast_path = false;
            }
          } else {
            valid_slide[s] = false;
            iy0[s] = -9999;
            ix0[s] = -9999;
            fast_path = false;
          }
        }

        uint32_t addr = addr_base;
        if (fast_path) {
          const int8_t* ptr0 = input_batch_base + iy0[0] * input_row_stride +
                               ix0[0] * input_depth + k_start;
          const int8_t* ptr1 = input_batch_base + iy0[1] * input_row_stride +
                               ix0[1] * input_depth + k_start;
          const int8_t* ptr2 = input_batch_base + iy0[2] * input_row_stride +
                               ix0[2] * input_depth + k_start;
          const int8_t* ptr3 = input_batch_base + iy0[3] * input_row_stride +
                               ix0[3] * input_depth + k_start;

          int ic = 0;
          for (; ic <= k_count - 4; ic += 4) {
            uint32_t val0 = detail::Pack4Bytes(ptr0);
            ptr0 += 4;
            uint32_t val1 = detail::Pack4Bytes(ptr1);
            ptr1 += 4;
            uint32_t val2 = detail::Pack4Bytes(ptr2);
            ptr2 += 4;
            uint32_t val3 = detail::Pack4Bytes(ptr3);
            ptr3 += 4;
            cfu_op5(10, val0, 0);
            cfu_op5(11, val1, 0);
            cfu_op5(12, val2, 0);
            cfu_op5(13, addr, val3);
            addr += kAddrStep * 4;
          }
          for (; ic < k_count; ++ic) {
            uint32_t in_pack = ((uint32_t)(uint8_t)(*ptr0++)) << 24;
            in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
            in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
            in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
            cfu_op3(0, addr, in_pack);
            addr += kAddrStep;
          }
        } else {
          bool in_bounds[4];
          for (int s = 0; s < 4; ++s) {
            in_bounds[s] = valid_slide[s] && (iy0[s] >= 0) &&
                           (iy0[s] < input_height) && (ix0[s] >= 0) &&
                           (ix0[s] < input_width);
          }
          const int8_t* ptr0 = in_bounds[0] ? input_batch_base +
                                                  iy0[0] * input_row_stride +
                                                  ix0[0] * input_depth + k_start
                                            : dummy_pad_buffer + k_start;
          const int8_t* ptr1 = in_bounds[1] ? input_batch_base +
                                                  iy0[1] * input_row_stride +
                                                  ix0[1] * input_depth + k_start
                                            : dummy_pad_buffer + k_start;
          const int8_t* ptr2 = in_bounds[2] ? input_batch_base +
                                                  iy0[2] * input_row_stride +
                                                  ix0[2] * input_depth + k_start
                                            : dummy_pad_buffer + k_start;
          const int8_t* ptr3 = in_bounds[3] ? input_batch_base +
                                                  iy0[3] * input_row_stride +
                                                  ix0[3] * input_depth + k_start
                                            : dummy_pad_buffer + k_start;

          int ic = 0;
          for (; ic <= k_count - 4; ic += 4) {
            uint32_t val0 = detail::Pack4Bytes(ptr0);
            ptr0 += 4;
            uint32_t val1 = detail::Pack4Bytes(ptr1);
            ptr1 += 4;
            uint32_t val2 = detail::Pack4Bytes(ptr2);
            ptr2 += 4;
            uint32_t val3 = detail::Pack4Bytes(ptr3);
            ptr3 += 4;
            cfu_op5(10, val0, 0);
            cfu_op5(11, val1, 0);
            cfu_op5(12, val2, 0);
            cfu_op5(13, addr, val3);
            addr += kAddrStep * 4;
          }
          for (; ic < k_count; ++ic) {
            uint32_t in_pack = ((uint32_t)(uint8_t)(*ptr0++)) << 24;
            in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
            in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
            in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
            cfu_op3(0, addr, in_pack);
            addr += kAddrStep;
          }
        }
      };

      auto PackInputTileDepthBlockToBuffer = [&](int slide, int k_start,
                                                 int k_count,
                                                 uint32_t* out_buffer) {
        int oy[4], ox[4], iy0[4], ix0[4];
        bool valid_slide[4];
        bool fast_path = true;

        for (int s = 0; s < 4; ++s) {
          int idx = slide + s;
          if (idx < M) {
            oy[s] = idx / output_width;
            ox[s] = idx % output_width;
            iy0[s] = oy[s] * stride_height - pad_h;
            ix0[s] = ox[s] * stride_width - pad_w;
            valid_slide[s] = true;
            if (iy0[s] < 0 || iy0[s] >= input_height || ix0[s] < 0 ||
                ix0[s] >= input_width) {
              fast_path = false;
            }
          } else {
            valid_slide[s] = false;
            iy0[s] = -9999;
            ix0[s] = -9999;
            fast_path = false;
          }
        }

        uint32_t* out_ptr = out_buffer;
        if (fast_path) {
          const int8_t* ptr0 = input_batch_base + iy0[0] * input_row_stride +
                               ix0[0] * input_depth + k_start;
          const int8_t* ptr1 = input_batch_base + iy0[1] * input_row_stride +
                               ix0[1] * input_depth + k_start;
          const int8_t* ptr2 = input_batch_base + iy0[2] * input_row_stride +
                               ix0[2] * input_depth + k_start;
          const int8_t* ptr3 = input_batch_base + iy0[3] * input_row_stride +
                               ix0[3] * input_depth + k_start;

          int ic = 0;
          for (; ic <= k_count - 4; ic += 4) {
            for (int i = 0; i < 4; ++i) {
              uint32_t in_pack = ((uint32_t)(uint8_t)(*ptr0++)) << 24;
              in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
              in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
              in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
              *out_ptr++ = in_pack;
            }
          }
          for (; ic < k_count; ++ic) {
            uint32_t in_pack = ((uint32_t)(uint8_t)(*ptr0++)) << 24;
            in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
            in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
            in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
            *out_ptr++ = in_pack;
          }
        } else {
          bool in_bounds[4];
          for (int s = 0; s < 4; ++s) {
            in_bounds[s] = valid_slide[s] && (iy0[s] >= 0) &&
                           (iy0[s] < input_height) && (ix0[s] >= 0) &&
                           (ix0[s] < input_width);
          }
          const int8_t* ptr0 = in_bounds[0] ? input_batch_base +
                                                  iy0[0] * input_row_stride +
                                                  ix0[0] * input_depth + k_start
                                            : dummy_pad_buffer + k_start;
          const int8_t* ptr1 = in_bounds[1] ? input_batch_base +
                                                  iy0[1] * input_row_stride +
                                                  ix0[1] * input_depth + k_start
                                            : dummy_pad_buffer + k_start;
          const int8_t* ptr2 = in_bounds[2] ? input_batch_base +
                                                  iy0[2] * input_row_stride +
                                                  ix0[2] * input_depth + k_start
                                            : dummy_pad_buffer + k_start;
          const int8_t* ptr3 = in_bounds[3] ? input_batch_base +
                                                  iy0[3] * input_row_stride +
                                                  ix0[3] * input_depth + k_start
                                            : dummy_pad_buffer + k_start;

          int ic = 0;
          for (; ic <= k_count - 4; ic += 4) {
            for (int i = 0; i < 4; ++i) {
              uint32_t in_pack = ((uint32_t)(uint8_t)(*ptr0++)) << 24;
              in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
              in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
              in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
              *out_ptr++ = in_pack;
            }
          }
          for (; ic < k_count; ++ic) {
            uint32_t in_pack = ((uint32_t)(uint8_t)(*ptr0++)) << 24;
            in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
            in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
            in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
            *out_ptr++ = in_pack;
          }
        }
      };

      const int k_count_per_block = depth_k_block;
      size_t cache_words_per_k = 0;
      for (int t = 0; t < num_slide_tiles; ++t) {
        const int slide = t * tile_m;
        const int tile_m_count = std::min(tile_m, M - slide);
        const int row_blocks_active = (tile_m_count + 3) / 4;
        cache_words_per_k +=
            static_cast<size_t>(row_blocks_active) * k_count_per_block;
      }
      const size_t total_cache_words =
          cache_words_per_k * static_cast<size_t>(depth_k_blocks);
      constexpr size_t kMaxPackedACacheWords = 1u << 20;
      const bool use_a_cache = (total_cache_words > 0) &&
                               (total_cache_words <= kMaxPackedACacheWords);
      auto& packed_cache = GetPackedCacheStorage();
      auto& packed_offsets = GetPackedOffsetsStorage();
      if (use_a_cache) {
        packed_cache.resize(total_cache_words);
        packed_offsets.resize(static_cast<size_t>(depth_k_blocks) *
                              num_slide_tiles);
        size_t cursor = 0;
        for (int kb = 0; kb < depth_k_blocks; ++kb) {
          const int k_start = kb * k_count_per_block;
          for (int t = 0; t < num_slide_tiles; ++t) {
            packed_offsets[static_cast<size_t>(kb) * num_slide_tiles + t] =
                static_cast<uint32_t>(cursor);
            const int slide = t * tile_m;
            const int tile_m_count = std::min(tile_m, M - slide);
            const int row_blocks_active = (tile_m_count + 3) / 4;
            uint32_t* tile_base = packed_cache.data() + cursor;
            for (int block = 0; block < row_blocks_active; ++block) {
              const int slide_base = slide + block * 4;
              PackInputTileDepthBlockToBuffer(
                  slide_base, k_start, k_count_per_block,
                  tile_base + block * k_count_per_block);
            }
            cursor +=
                static_cast<size_t>(row_blocks_active) * k_count_per_block;
          }
        }
      }

      auto LoadInputTileBlock = [&](int slide, int tile_m_count, int bank,
                                    int k_start, int k_count, int k_index) {
        const int row_blocks_active = (tile_m_count + 3) / 4;
        if (row_blocks_active <= 0) {
          return;
        }
        if (use_a_cache) {
          const int tile_index = slide / tile_m;
          const uint32_t* tile_ptr =
              packed_cache.data() +
              packed_offsets[static_cast<size_t>(k_index) * num_slide_tiles +
                             tile_index];
          const int tile_words = row_blocks_active * k_count;
          uint32_t addr = bank_addr_flags[bank];
          for (int i = 0; i < tile_words; ++i) {
            cfu_op3(0, addr, tile_ptr[i]);
            addr += kAddrStep;
          }
        } else {
          for (int block = 0; block < row_blocks_active; ++block) {
            const int slide_base = slide + block * 4;
            const uint32_t addr_base =
                (static_cast<uint32_t>(block * k_count) << 16) |
                bank_addr_flags[bank];
            PackInputTileDepthBlock(slide_base, k_start, k_count, addr_base);
          }
        }
      };

      auto AccumulateOrWriteOutputTile =
          [&](int tile_index, int read_tile_m_count, int read_c_bank,
              int valid_channels, int total_blocks, int out_ch,
              bool write_output) {
            int32_t* tile_accum =
                accum_data + static_cast<size_t>(tile_index) * tile_m * tile_n;
            uint32_t read_addr_base = bank_addr_flags[read_c_bank];
            const int slide = tile_index * tile_m;
            for (int row = 0; row < read_tile_m_count; ++row) {
              uint32_t read_addr =
                  (static_cast<uint32_t>(row * total_blocks) << 16) |
                  read_addr_base;
              if (write_output) {
                const int idx = slide + row;
                const int out_y = idx / output_width;
                const int out_x = idx % output_width;
                int8_t* dst = output_batch_base + (out_y * output_row_stride) +
                              (out_x * output_depth) + out_ch;
                for (int c = 0; c < valid_channels; ++c) {
                  const int word_sel = 3 - (c & 3);
                  uint32_t cmd = read_addr | static_cast<uint32_t>(word_sel);
                  const int32_t acc = cfu_op3(3, cmd, 0);
                  if ((c & 3) == 3) {
                    read_addr += kAddrStep;
                  }
                  const int ch = out_ch + c;
                  int32_t sum =
                      tile_accum[row * tile_n + c] + acc + bias_correction[ch];
                  sum = MultiplyByQuantizedMultiplier(
                      sum, output_multiplier[ch], output_shift[ch]);
                  sum += output_offset;
                  sum = std::max(sum, act_min);
                  sum = std::min(sum, act_max);
                  dst[c] = static_cast<int8_t>(sum);
                }
              } else {
                for (int c = 0; c < valid_channels; ++c) {
                  const int word_sel = 3 - (c & 3);
                  uint32_t cmd = read_addr | static_cast<uint32_t>(word_sel);
                  const int32_t acc = cfu_op3(3, cmd, 0);
                  if ((c & 3) == 3) {
                    read_addr += kAddrStep;
                  }
                  tile_accum[row * tile_n + c] += acc;
                }
              }
            }
          };

      for (int out_ch = 0; out_ch < output_depth; out_ch += tile_n) {
        const int valid_channels = std::min(tile_n, output_depth - out_ch);
        const int total_blocks = (valid_channels + 3) / 4;
        const int current_block_N = total_blocks * 4;

        std::fill(accum_data, accum_data + accum_entries, 0);

        for (int kb = 0; kb < depth_k_blocks; ++kb) {
          const bool final_k_block = (kb + 1 == depth_k_blocks);
          const int k_start = kb * depth_k_block;
          const int k_count = depth_k_block;

          PackWeightsBlock(out_ch, valid_channels, total_blocks,
                           bank_addr_flags[0], k_start, k_count);

          int slide = 0;
          int cur_bank = 0;
          int c_bank = 0;
          int cur_tile_m_count = std::min(tile_m, M - slide);
          LoadInputTileBlock(slide, cur_tile_m_count, cur_bank, k_start,
                             k_count, kb);

          bool have_pending = false;
          int pending_tile_m_count = 0;
          int pending_c_bank = 0;
          int pending_tile_index = 0;

          while (slide < M) {
            cfu_op3(5, static_cast<uint32_t>(cur_bank), 0);
            cfu_op3(6, 0, 0);
            cfu_op3(7, static_cast<uint32_t>(c_bank), 0);
            cfu_op3(2,
                    (k_count << 16) | (cur_tile_m_count << 8) | current_block_N,
                    0);

            const int next_slide = slide + tile_m;
            int next_tile_m_count = 0;
            const int next_bank = cur_bank ^ 1;
            if (next_slide < M) {
              next_tile_m_count = std::min(tile_m, M - next_slide);
              LoadInputTileBlock(next_slide, next_tile_m_count, next_bank,
                                 k_start, k_count, kb);
            }

            if (have_pending) {
              AccumulateOrWriteOutputTile(
                  pending_tile_index, pending_tile_m_count, pending_c_bank,
                  valid_channels, total_blocks, out_ch, final_k_block);
            }

            while (cfu_op3(4, 0, 0) != 0) {
            }

            pending_tile_index = slide / tile_m;
            pending_tile_m_count = cur_tile_m_count;
            pending_c_bank = c_bank;
            have_pending = true;

            slide = next_slide;
            cur_bank = next_bank;
            cur_tile_m_count = next_tile_m_count;
            c_bank ^= 1;
          }

          if (have_pending) {
            AccumulateOrWriteOutputTile(
                pending_tile_index, pending_tile_m_count, pending_c_bank,
                valid_channels, total_blocks, out_ch, final_k_block);
          }
        }
      }
      continue;
    }

    if (use_k_block) {
      const int tile_m = best_params.tile_m;
      const int tile_n = best_params.tile_n;
      const int num_slide_tiles = CeilDivInt(M, tile_m);
      const uint32_t bank_addr_flags[2] = {0u, kAddrBankBit};

      const size_t accum_entries =
          static_cast<size_t>(num_slide_tiles) * tile_m * tile_n;
      auto& accum_storage = GetAccumulationStorage();
      if (accum_storage.size() < accum_entries) {
        accum_storage.resize(accum_entries);
      }
      int32_t* accum_data = accum_storage.data();

      auto PackWeightsBlock = [&](int out_ch, int valid_channels,
                                  int total_blocks, uint32_t addr_flag,
                                  int k_start, int k_count) {
        const int full_blocks = valid_channels / 4;
        for (int block = 0; block < full_blocks; ++block) {
          const int ch_base = out_ch + block * 4;
          const int8_t* w_ptr0 =
              filter_data + ch_base * filter_ch_stride + k_start;
          const int8_t* w_ptr1 = w_ptr0 + filter_ch_stride;
          const int8_t* w_ptr2 = w_ptr1 + filter_ch_stride;
          const int8_t* w_ptr3 = w_ptr2 + filter_ch_stride;

          uint32_t addr =
              (static_cast<uint32_t>(block * k_count) << 16) | addr_flag;
          int k = 0;
          for (; k + 3 < k_count; k += 4) {
            uint32_t w_pack0 =
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr0++)) << 24) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr1++)) << 16) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr2++)) << 8) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr3++)));
            cfu_op3(1, addr, w_pack0);
            addr += kAddrStep;

            uint32_t w_pack1 =
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr0++)) << 24) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr1++)) << 16) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr2++)) << 8) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr3++)));
            cfu_op3(1, addr, w_pack1);
            addr += kAddrStep;

            uint32_t w_pack2 =
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr0++)) << 24) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr1++)) << 16) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr2++)) << 8) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr3++)));
            cfu_op3(1, addr, w_pack2);
            addr += kAddrStep;

            uint32_t w_pack3 =
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr0++)) << 24) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr1++)) << 16) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr2++)) << 8) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr3++)));
            cfu_op3(1, addr, w_pack3);
            addr += kAddrStep;
          }
          for (; k < k_count; ++k) {
            uint32_t w_pack =
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr0++)) << 24) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr1++)) << 16) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr2++)) << 8) |
                (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr3++)));
            cfu_op3(1, addr, w_pack);
            addr += kAddrStep;
          }
        }

        if (full_blocks < total_blocks) {
          const int ch_base = out_ch + full_blocks * 4;
          const int8_t* w_ptr0 =
              (ch_base < output_depth)
                  ? filter_data + ch_base * filter_ch_stride + k_start
                  : nullptr;
          const int8_t* w_ptr1 =
              (ch_base + 1 < output_depth)
                  ? filter_data + (ch_base + 1) * filter_ch_stride + k_start
                  : nullptr;
          const int8_t* w_ptr2 =
              (ch_base + 2 < output_depth)
                  ? filter_data + (ch_base + 2) * filter_ch_stride + k_start
                  : nullptr;
          const int8_t* w_ptr3 =
              (ch_base + 3 < output_depth)
                  ? filter_data + (ch_base + 3) * filter_ch_stride + k_start
                  : nullptr;

          uint32_t addr =
              (static_cast<uint32_t>(full_blocks * k_count) << 16) | addr_flag;
          for (int k = 0; k < k_count; ++k) {
            uint32_t w_pack =
                (static_cast<uint32_t>(w_ptr0 ? static_cast<uint8_t>(*w_ptr0++)
                                              : 0u)
                 << 24) |
                (static_cast<uint32_t>(w_ptr1 ? static_cast<uint8_t>(*w_ptr1++)
                                              : 0u)
                 << 16) |
                (static_cast<uint32_t>(w_ptr2 ? static_cast<uint8_t>(*w_ptr2++)
                                              : 0u)
                 << 8) |
                (static_cast<uint32_t>(w_ptr3 ? static_cast<uint8_t>(*w_ptr3++)
                                              : 0u));
            cfu_op3(1, addr, w_pack);
            addr += kAddrStep;
          }
        }
      };

      auto PackInputTileBlock = [&](int slide, int spatial_start,
                                    int spatial_count, uint32_t addr_base) {
        int oy[4], ox[4], iy0[4], ix0[4];
        bool valid_slide[4];
        bool fast_path = true;

        for (int s = 0; s < 4; ++s) {
          int idx = slide + s;
          if (idx < M) {
            oy[s] = idx / output_width;
            ox[s] = idx % output_width;
            iy0[s] = oy[s] * stride_height - pad_h;
            ix0[s] = ox[s] * stride_width - pad_w;
            valid_slide[s] = true;

            int y_start = iy0[s];
            int y_end = iy0[s] + (filter_height - 1) * dilation_h;
            int x_start = ix0[s];
            int x_end = ix0[s] + (filter_width - 1) * dilation_w;
            if (y_start < 0 || y_end >= input_height || x_start < 0 ||
                x_end >= input_width) {
              fast_path = false;
            }
          } else {
            valid_slide[s] = false;
            iy0[s] = -9999;
            ix0[s] = -9999;
            fast_path = false;
          }
        }

        uint32_t addr = addr_base;
        const int spatial_end = spatial_start + spatial_count;

        if (fast_path) {
          for (int s = spatial_start; s < spatial_end; ++s) {
            const int fy = s / filter_width;
            const int fx = s % filter_width;
            const int y_offset = fy * dilation_h * input_row_stride;
            const int8_t* base0 = input_batch_base + iy0[0] * input_row_stride +
                                  ix0[0] * input_depth + y_offset;
            const int8_t* base1 = input_batch_base + iy0[1] * input_row_stride +
                                  ix0[1] * input_depth + y_offset;
            const int8_t* base2 = input_batch_base + iy0[2] * input_row_stride +
                                  ix0[2] * input_depth + y_offset;
            const int8_t* base3 = input_batch_base + iy0[3] * input_row_stride +
                                  ix0[3] * input_depth + y_offset;

            const int x_offset = fx * dilation_w * input_depth;
            const int8_t* ptr0 = base0 + x_offset;
            const int8_t* ptr1 = base1 + x_offset;
            const int8_t* ptr2 = base2 + x_offset;
            const int8_t* ptr3 = base3 + x_offset;

            int ic = 0;
            for (; ic <= input_depth - 4; ic += 4) {
              for (int i = 0; i < 4; ++i) {
                uint32_t in_pack = ((uint32_t)(uint8_t)(*ptr0++)) << 24;
                in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
                in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
                in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
                cfu_op3(0, addr, in_pack);
                addr += kAddrStep;
              }
            }
            for (; ic < input_depth; ++ic) {
              uint32_t in_pack = ((uint32_t)(uint8_t)(*ptr0++)) << 24;
              in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
              in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
              in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
              cfu_op3(0, addr, in_pack);
              addr += kAddrStep;
            }
          }
        } else {
          for (int s = spatial_start; s < spatial_end; ++s) {
            const int fy = s / filter_width;
            const int fx = s % filter_width;
            bool y_valid[4];
            int in_y[4];
            for (int i = 0; i < 4; ++i) {
              in_y[i] = iy0[i] + fy * dilation_h;
              y_valid[i] =
                  valid_slide[i] && (in_y[i] >= 0) && (in_y[i] < input_height);
            }

            const int in_x0 = ix0[0] + fx * dilation_w;
            const int in_x1 = ix0[1] + fx * dilation_w;
            const int in_x2 = ix0[2] + fx * dilation_w;
            const int in_x3 = ix0[3] + fx * dilation_w;

            const int8_t* ptr0 =
                (y_valid[0] && in_x0 >= 0 && in_x0 < input_width)
                    ? input_batch_base + (in_y[0] * input_row_stride) +
                          (in_x0 * input_depth)
                    : dummy_pad_buffer;
            const int8_t* ptr1 =
                (y_valid[1] && in_x1 >= 0 && in_x1 < input_width)
                    ? input_batch_base + (in_y[1] * input_row_stride) +
                          (in_x1 * input_depth)
                    : dummy_pad_buffer;
            const int8_t* ptr2 =
                (y_valid[2] && in_x2 >= 0 && in_x2 < input_width)
                    ? input_batch_base + (in_y[2] * input_row_stride) +
                          (in_x2 * input_depth)
                    : dummy_pad_buffer;
            const int8_t* ptr3 =
                (y_valid[3] && in_x3 >= 0 && in_x3 < input_width)
                    ? input_batch_base + (in_y[3] * input_row_stride) +
                          (in_x3 * input_depth)
                    : dummy_pad_buffer;

            int ic = 0;
            for (; ic <= input_depth - 4; ic += 4) {
              for (int i = 0; i < 4; ++i) {
                uint32_t in_pack = ((uint32_t)(uint8_t)(*ptr0++)) << 24;
                in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
                in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
                in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
                cfu_op3(0, addr, in_pack);
                addr += kAddrStep;
              }
            }

            for (; ic < input_depth; ++ic) {
              uint32_t in_pack = 0;
              in_pack |= ((uint32_t)(uint8_t)(*ptr0++)) << 24;
              in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
              in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
              in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
              cfu_op3(0, addr, in_pack);
              addr += kAddrStep;
            }
          }
        }
      };

      auto PackInputTileBlockToBuffer = [&](int slide, int spatial_start,
                                            int spatial_count,
                                            uint32_t* out_buffer) {
        int oy[4], ox[4], iy0[4], ix0[4];
        bool valid_slide[4];
        bool fast_path = true;

        for (int s = 0; s < 4; ++s) {
          int idx = slide + s;
          if (idx < M) {
            oy[s] = idx / output_width;
            ox[s] = idx % output_width;
            iy0[s] = oy[s] * stride_height - pad_h;
            ix0[s] = ox[s] * stride_width - pad_w;
            valid_slide[s] = true;

            int y_start = iy0[s];
            int y_end = iy0[s] + (filter_height - 1) * dilation_h;
            int x_start = ix0[s];
            int x_end = ix0[s] + (filter_width - 1) * dilation_w;
            if (y_start < 0 || y_end >= input_height || x_start < 0 ||
                x_end >= input_width) {
              fast_path = false;
            }
          } else {
            valid_slide[s] = false;
            iy0[s] = -9999;
            ix0[s] = -9999;
            fast_path = false;
          }
        }

        uint32_t* out_ptr = out_buffer;
        const int spatial_end = spatial_start + spatial_count;

        if (fast_path) {
          for (int s = spatial_start; s < spatial_end; ++s) {
            const int fy = s / filter_width;
            const int fx = s % filter_width;
            const int y_offset = fy * dilation_h * input_row_stride;
            const int8_t* base0 = input_batch_base + iy0[0] * input_row_stride +
                                  ix0[0] * input_depth + y_offset;
            const int8_t* base1 = input_batch_base + iy0[1] * input_row_stride +
                                  ix0[1] * input_depth + y_offset;
            const int8_t* base2 = input_batch_base + iy0[2] * input_row_stride +
                                  ix0[2] * input_depth + y_offset;
            const int8_t* base3 = input_batch_base + iy0[3] * input_row_stride +
                                  ix0[3] * input_depth + y_offset;

            const int x_offset = fx * dilation_w * input_depth;
            const int8_t* ptr0 = base0 + x_offset;
            const int8_t* ptr1 = base1 + x_offset;
            const int8_t* ptr2 = base2 + x_offset;
            const int8_t* ptr3 = base3 + x_offset;

            int ic = 0;
            for (; ic <= input_depth - 4; ic += 4) {
              for (int i = 0; i < 4; ++i) {
                uint32_t in_pack = ((uint32_t)(uint8_t)(*ptr0++)) << 24;
                in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
                in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
                in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
                *out_ptr++ = in_pack;
              }
            }
            for (; ic < input_depth; ++ic) {
              uint32_t in_pack = ((uint32_t)(uint8_t)(*ptr0++)) << 24;
              in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
              in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
              in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
              *out_ptr++ = in_pack;
            }
          }
        } else {
          for (int s = spatial_start; s < spatial_end; ++s) {
            const int fy = s / filter_width;
            const int fx = s % filter_width;
            bool y_valid[4];
            int in_y[4];
            for (int i = 0; i < 4; ++i) {
              in_y[i] = iy0[i] + fy * dilation_h;
              y_valid[i] =
                  valid_slide[i] && (in_y[i] >= 0) && (in_y[i] < input_height);
            }

            const int in_x0 = ix0[0] + fx * dilation_w;
            const int in_x1 = ix0[1] + fx * dilation_w;
            const int in_x2 = ix0[2] + fx * dilation_w;
            const int in_x3 = ix0[3] + fx * dilation_w;

            const int8_t* ptr0 =
                (y_valid[0] && in_x0 >= 0 && in_x0 < input_width)
                    ? input_batch_base + (in_y[0] * input_row_stride) +
                          (in_x0 * input_depth)
                    : dummy_pad_buffer;
            const int8_t* ptr1 =
                (y_valid[1] && in_x1 >= 0 && in_x1 < input_width)
                    ? input_batch_base + (in_y[1] * input_row_stride) +
                          (in_x1 * input_depth)
                    : dummy_pad_buffer;
            const int8_t* ptr2 =
                (y_valid[2] && in_x2 >= 0 && in_x2 < input_width)
                    ? input_batch_base + (in_y[2] * input_row_stride) +
                          (in_x2 * input_depth)
                    : dummy_pad_buffer;
            const int8_t* ptr3 =
                (y_valid[3] && in_x3 >= 0 && in_x3 < input_width)
                    ? input_batch_base + (in_y[3] * input_row_stride) +
                          (in_x3 * input_depth)
                    : dummy_pad_buffer;

            int ic = 0;
            for (; ic <= input_depth - 4; ic += 4) {
              for (int i = 0; i < 4; ++i) {
                uint32_t in_pack = ((uint32_t)(uint8_t)(*ptr0++)) << 24;
                in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
                in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
                in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
                *out_ptr++ = in_pack;
              }
            }

            for (; ic < input_depth; ++ic) {
              uint32_t in_pack = 0;
              in_pack |= ((uint32_t)(uint8_t)(*ptr0++)) << 24;
              in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
              in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
              in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
              *out_ptr++ = in_pack;
            }
          }
        }
      };

      const int k_count_per_block = best_k_block;
      size_t cache_words_per_k = 0;
      for (int t = 0; t < num_slide_tiles; ++t) {
        const int slide = t * tile_m;
        const int tile_m_count = std::min(tile_m, M - slide);
        const int row_blocks_active = (tile_m_count + 3) / 4;
        cache_words_per_k +=
            static_cast<size_t>(row_blocks_active) * k_count_per_block;
      }
      const size_t total_cache_words =
          cache_words_per_k * static_cast<size_t>(best_k_blocks);
      constexpr size_t kMaxPackedACacheWords = 1u << 20;
      const bool use_a_cache = (total_cache_words > 0) &&
                               (total_cache_words <= kMaxPackedACacheWords);
      auto& packed_cache = GetPackedCacheStorage();
      auto& packed_offsets = GetPackedOffsetsStorage();
      if (use_a_cache) {
        packed_cache.resize(total_cache_words);
        packed_offsets.resize(static_cast<size_t>(best_k_blocks) *
                              num_slide_tiles);
        size_t cursor = 0;
        for (int kb = 0; kb < best_k_blocks; ++kb) {
          const int spatial_start = kb * best_spatial_block;
          const int spatial_count = best_spatial_block;
          for (int t = 0; t < num_slide_tiles; ++t) {
            packed_offsets[static_cast<size_t>(kb) * num_slide_tiles + t] =
                static_cast<uint32_t>(cursor);
            const int slide = t * tile_m;
            const int tile_m_count = std::min(tile_m, M - slide);
            const int row_blocks_active = (tile_m_count + 3) / 4;
            uint32_t* tile_base = packed_cache.data() + cursor;
            for (int block = 0; block < row_blocks_active; ++block) {
              const int slide_base = slide + block * 4;
              PackInputTileBlockToBuffer(slide_base, spatial_start,
                                         spatial_count,
                                         tile_base + block * k_count_per_block);
            }
            cursor +=
                static_cast<size_t>(row_blocks_active) * k_count_per_block;
          }
        }
      }

      auto LoadInputTileBlock = [&](int slide, int tile_m_count, int bank,
                                    int spatial_start, int spatial_count,
                                    int k_count, int k_index) {
        const int row_blocks_active = (tile_m_count + 3) / 4;
        if (row_blocks_active <= 0) {
          return;
        }
        if (use_a_cache) {
          const int tile_index = slide / tile_m;
          const uint32_t* tile_ptr =
              packed_cache.data() +
              packed_offsets[static_cast<size_t>(k_index) * num_slide_tiles +
                             tile_index];
          const int tile_words = row_blocks_active * k_count;
          uint32_t addr = bank_addr_flags[bank];
          for (int i = 0; i < tile_words; ++i) {
            cfu_op3(0, addr, tile_ptr[i]);
            addr += kAddrStep;
          }
        } else {
          for (int block = 0; block < row_blocks_active; ++block) {
            const int slide_base = slide + block * 4;
            const uint32_t addr_base =
                (static_cast<uint32_t>(block * k_count) << 16) |
                bank_addr_flags[bank];
            PackInputTileBlock(slide_base, spatial_start, spatial_count,
                               addr_base);
          }
        }
      };

      auto AccumulateOrWriteOutputTile =
          [&](int tile_index, int read_tile_m_count, int read_c_bank,
              int valid_channels, int total_blocks, int out_ch,
              bool write_output) {
            int32_t* tile_accum =
                accum_data + static_cast<size_t>(tile_index) * tile_m * tile_n;
            uint32_t read_addr_base = bank_addr_flags[read_c_bank];
            const int slide = tile_index * tile_m;
            for (int row = 0; row < read_tile_m_count; ++row) {
              uint32_t read_addr =
                  (static_cast<uint32_t>(row * total_blocks) << 16) |
                  read_addr_base;
              if (write_output) {
                const int idx = slide + row;
                const int out_y = idx / output_width;
                const int out_x = idx % output_width;
                int8_t* dst = output_batch_base + (out_y * output_row_stride) +
                              (out_x * output_depth) + out_ch;
                for (int c = 0; c < valid_channels; ++c) {
                  const int word_sel = 3 - (c & 3);
                  uint32_t cmd = read_addr | static_cast<uint32_t>(word_sel);
                  const int32_t acc = cfu_op3(3, cmd, 0);
                  if ((c & 3) == 3) {
                    read_addr += kAddrStep;
                  }
                  const int ch = out_ch + c;
                  int32_t sum =
                      tile_accum[row * tile_n + c] + acc + bias_correction[ch];
                  sum = MultiplyByQuantizedMultiplier(
                      sum, output_multiplier[ch], output_shift[ch]);
                  sum += output_offset;
                  sum = std::max(sum, act_min);
                  sum = std::min(sum, act_max);
                  dst[c] = static_cast<int8_t>(sum);
                }
              } else {
                for (int c = 0; c < valid_channels; ++c) {
                  const int word_sel = 3 - (c & 3);
                  uint32_t cmd = read_addr | static_cast<uint32_t>(word_sel);
                  const int32_t acc = cfu_op3(3, cmd, 0);
                  if ((c & 3) == 3) {
                    read_addr += kAddrStep;
                  }
                  tile_accum[row * tile_n + c] += acc;
                }
              }
            }
          };

      for (int out_ch = 0; out_ch < output_depth; out_ch += tile_n) {
        const int valid_channels = std::min(tile_n, output_depth - out_ch);
        const int total_blocks = (valid_channels + 3) / 4;
        const int current_block_N = total_blocks * 4;

        std::fill(accum_data, accum_data + accum_entries, 0);

        for (int kb = 0; kb < best_k_blocks; ++kb) {
          const bool final_k_block = (kb + 1 == best_k_blocks);
          const int spatial_start = kb * best_spatial_block;
          const int spatial_count = best_spatial_block;
          const int k_start = spatial_start * input_depth;
          const int k_count = spatial_count * input_depth;

          PackWeightsBlock(out_ch, valid_channels, total_blocks,
                           bank_addr_flags[0], k_start, k_count);

          int slide = 0;
          int cur_bank = 0;
          int c_bank = 0;
          int cur_tile_m_count = std::min(tile_m, M - slide);
          LoadInputTileBlock(slide, cur_tile_m_count, cur_bank, spatial_start,
                             spatial_count, k_count, kb);

          bool have_pending = false;
          int pending_tile_m_count = 0;
          int pending_c_bank = 0;
          int pending_tile_index = 0;

          while (slide < M) {
            cfu_op3(5, static_cast<uint32_t>(cur_bank), 0);
            cfu_op3(6, 0, 0);
            cfu_op3(7, static_cast<uint32_t>(c_bank), 0);
            cfu_op3(2,
                    (k_count << 16) | (cur_tile_m_count << 8) | current_block_N,
                    0);

            const int next_slide = slide + tile_m;
            int next_tile_m_count = 0;
            const int next_bank = cur_bank ^ 1;
            if (next_slide < M) {
              next_tile_m_count = std::min(tile_m, M - next_slide);
              LoadInputTileBlock(next_slide, next_tile_m_count, next_bank,
                                 spatial_start, spatial_count, k_count, kb);
            }

            if (have_pending) {
              AccumulateOrWriteOutputTile(
                  pending_tile_index, pending_tile_m_count, pending_c_bank,
                  valid_channels, total_blocks, out_ch, final_k_block);
            }

            while (cfu_op3(4, 0, 0) != 0) {
            }

            pending_tile_index = slide / tile_m;
            pending_tile_m_count = cur_tile_m_count;
            pending_c_bank = c_bank;
            have_pending = true;

            slide = next_slide;
            cur_bank = next_bank;
            cur_tile_m_count = next_tile_m_count;
            c_bank ^= 1;
          }

          if (have_pending) {
            AccumulateOrWriteOutputTile(
                pending_tile_index, pending_tile_m_count, pending_c_bank,
                valid_channels, total_blocks, out_ch, final_k_block);
          }
        }
      }
      continue;
    }

    const int tile_m = base_params.tile_m;
    const int tile_n = base_params.tile_n;

    auto PackWeights = [&](int out_ch, int valid_channels, int total_blocks,
                           uint32_t addr_flag) {
      const int full_blocks = valid_channels / 4;
      for (int block = 0; block < full_blocks; ++block) {
        const int ch_base = out_ch + block * 4;
        const int8_t* w_ptr0 = filter_data + ch_base * filter_ch_stride;
        const int8_t* w_ptr1 = w_ptr0 + filter_ch_stride;
        const int8_t* w_ptr2 = w_ptr1 + filter_ch_stride;
        const int8_t* w_ptr3 = w_ptr2 + filter_ch_stride;

        uint32_t addr = (static_cast<uint32_t>(block * K) << 16) | addr_flag;
        int k = 0;
        for (; k + 3 < K; k += 4) {
          uint32_t w_pack0 =
              (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr0++)) << 24) |
              (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr1++)) << 16) |
              (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr2++)) << 8) |
              (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr3++)));
          cfu_op3(1, addr, w_pack0);
          addr += kAddrStep;

          uint32_t w_pack1 =
              (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr0++)) << 24) |
              (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr1++)) << 16) |
              (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr2++)) << 8) |
              (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr3++)));
          cfu_op3(1, addr, w_pack1);
          addr += kAddrStep;

          uint32_t w_pack2 =
              (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr0++)) << 24) |
              (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr1++)) << 16) |
              (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr2++)) << 8) |
              (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr3++)));
          cfu_op3(1, addr, w_pack2);
          addr += kAddrStep;

          uint32_t w_pack3 =
              (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr0++)) << 24) |
              (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr1++)) << 16) |
              (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr2++)) << 8) |
              (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr3++)));
          cfu_op3(1, addr, w_pack3);
          addr += kAddrStep;
        }
        for (; k < K; ++k) {
          uint32_t w_pack =
              (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr0++)) << 24) |
              (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr1++)) << 16) |
              (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr2++)) << 8) |
              (static_cast<uint32_t>(static_cast<uint8_t>(*w_ptr3++)));
          cfu_op3(1, addr, w_pack);
          addr += kAddrStep;
        }
      }

      if (full_blocks < total_blocks) {
        const int ch_base = out_ch + full_blocks * 4;
        const int8_t* w_ptr0 = (ch_base < output_depth)
                                   ? filter_data + ch_base * filter_ch_stride
                                   : nullptr;
        const int8_t* w_ptr1 =
            (ch_base + 1 < output_depth)
                ? filter_data + (ch_base + 1) * filter_ch_stride
                : nullptr;
        const int8_t* w_ptr2 =
            (ch_base + 2 < output_depth)
                ? filter_data + (ch_base + 2) * filter_ch_stride
                : nullptr;
        const int8_t* w_ptr3 =
            (ch_base + 3 < output_depth)
                ? filter_data + (ch_base + 3) * filter_ch_stride
                : nullptr;

        uint32_t addr =
            (static_cast<uint32_t>(full_blocks * K) << 16) | addr_flag;
        for (int k = 0; k < K; ++k) {
          uint32_t w_pack =
              (static_cast<uint32_t>(w_ptr0 ? static_cast<uint8_t>(*w_ptr0++)
                                            : 0u)
               << 24) |
              (static_cast<uint32_t>(w_ptr1 ? static_cast<uint8_t>(*w_ptr1++)
                                            : 0u)
               << 16) |
              (static_cast<uint32_t>(w_ptr2 ? static_cast<uint8_t>(*w_ptr2++)
                                            : 0u)
               << 8) |
              (static_cast<uint32_t>(w_ptr3 ? static_cast<uint8_t>(*w_ptr3++)
                                            : 0u));
          cfu_op3(1, addr, w_pack);
          addr += kAddrStep;
        }
      }
    };

    auto PackInputTileToBuffer = [&](int slide, uint32_t* out_buffer) {
      int oy[4], ox[4], iy0[4], ix0[4];
      bool valid_slide[4];
      bool fast_path = true;

      for (int s = 0; s < 4; ++s) {
        int idx = slide + s;
        if (idx < M) {
          oy[s] = idx / output_width;
          ox[s] = idx % output_width;
          iy0[s] = oy[s] * stride_height - pad_h;
          ix0[s] = ox[s] * stride_width - pad_w;
          valid_slide[s] = true;

          int y_start = iy0[s];
          int y_end = iy0[s] + (filter_height - 1) * dilation_h;
          int x_start = ix0[s];
          int x_end = ix0[s] + (filter_width - 1) * dilation_w;
          if (y_start < 0 || y_end >= input_height || x_start < 0 ||
              x_end >= input_width) {
            fast_path = false;
          }
        } else {
          valid_slide[s] = false;
          iy0[s] = -9999;
          ix0[s] = -9999;
          fast_path = false;
        }
      }

      uint32_t* out_ptr = out_buffer;

      if (fast_path) {
        for (int fy = 0; fy < filter_height; ++fy) {
          int y_offset = fy * dilation_h * input_row_stride;
          const int8_t* base0 = input_batch_base + iy0[0] * input_row_stride +
                                ix0[0] * input_depth + y_offset;
          const int8_t* base1 = input_batch_base + iy0[1] * input_row_stride +
                                ix0[1] * input_depth + y_offset;
          const int8_t* base2 = input_batch_base + iy0[2] * input_row_stride +
                                ix0[2] * input_depth + y_offset;
          const int8_t* base3 = input_batch_base + iy0[3] * input_row_stride +
                                ix0[3] * input_depth + y_offset;

          for (int fx = 0; fx < filter_width; ++fx) {
            int x_offset = fx * dilation_w * input_depth;
            const int8_t* ptr0 = base0 + x_offset;
            const int8_t* ptr1 = base1 + x_offset;
            const int8_t* ptr2 = base2 + x_offset;
            const int8_t* ptr3 = base3 + x_offset;

            int ic = 0;
            for (; ic <= input_depth - 4; ic += 4) {
              for (int i = 0; i < 4; ++i) {
                uint32_t in_pack = ((uint32_t)(uint8_t)(*ptr0++)) << 24;
                in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
                in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
                in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
                *out_ptr++ = in_pack;
              }
            }
            for (; ic < input_depth; ++ic) {
              uint32_t in_pack = ((uint32_t)(uint8_t)(*ptr0++)) << 24;
              in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
              in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
              in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
              *out_ptr++ = in_pack;
            }
          }
        }
      } else {
        for (int fy = 0; fy < filter_height; ++fy) {
          bool y_valid[4];
          int in_y[4];
          for (int s = 0; s < 4; ++s) {
            in_y[s] = iy0[s] + fy * dilation_h;
            y_valid[s] =
                valid_slide[s] && (in_y[s] >= 0) && (in_y[s] < input_height);
          }

          for (int fx = 0; fx < filter_width; ++fx) {
            const int8_t* ptr0;
            const int8_t* ptr1;
            const int8_t* ptr2;
            const int8_t* ptr3;

            int in_x0 = ix0[0] + fx * dilation_w;
            if (y_valid[0] && in_x0 >= 0 && in_x0 < input_width)
              ptr0 = input_batch_base + (in_y[0] * input_row_stride) +
                     (in_x0 * input_depth);
            else
              ptr0 = dummy_pad_buffer;

            int in_x1 = ix0[1] + fx * dilation_w;
            if (y_valid[1] && in_x1 >= 0 && in_x1 < input_width)
              ptr1 = input_batch_base + (in_y[1] * input_row_stride) +
                     (in_x1 * input_depth);
            else
              ptr1 = dummy_pad_buffer;

            int in_x2 = ix0[2] + fx * dilation_w;
            if (y_valid[2] && in_x2 >= 0 && in_x2 < input_width)
              ptr2 = input_batch_base + (in_y[2] * input_row_stride) +
                     (in_x2 * input_depth);
            else
              ptr2 = dummy_pad_buffer;

            int in_x3 = ix0[3] + fx * dilation_w;
            if (y_valid[3] && in_x3 >= 0 && in_x3 < input_width)
              ptr3 = input_batch_base + (in_y[3] * input_row_stride) +
                     (in_x3 * input_depth);
            else
              ptr3 = dummy_pad_buffer;

            int ic = 0;
            for (; ic <= input_depth - 4; ic += 4) {
              for (int i = 0; i < 4; ++i) {
                uint32_t in_pack = ((uint32_t)(uint8_t)(*ptr0++)) << 24;
                in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
                in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
                in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
                *out_ptr++ = in_pack;
              }
            }

            for (; ic < input_depth; ++ic) {
              uint32_t in_pack = 0;
              in_pack |= ((uint32_t)(uint8_t)(*ptr0++)) << 24;
              in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
              in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
              in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
              *out_ptr++ = in_pack;
            }
          }
        }
      }
    };

    auto PackInputTile = [&](int slide, uint32_t addr_base) {
      int oy[4], ox[4], iy0[4], ix0[4];
      bool valid_slide[4];
      bool fast_path = true;

      for (int s = 0; s < 4; ++s) {
        int idx = slide + s;
        if (idx < M) {
          oy[s] = idx / output_width;
          ox[s] = idx % output_width;
          iy0[s] = oy[s] * stride_height - pad_h;
          ix0[s] = ox[s] * stride_width - pad_w;
          valid_slide[s] = true;

          // Check bounds for fast path
          int y_start = iy0[s];
          int y_end = iy0[s] + (filter_height - 1) * dilation_h;
          int x_start = ix0[s];
          int x_end = ix0[s] + (filter_width - 1) * dilation_w;
          if (y_start < 0 || y_end >= input_height || x_start < 0 ||
              x_end >= input_width) {
            fast_path = false;
          }
        } else {
          valid_slide[s] = false;
          iy0[s] = -9999;
          ix0[s] = -9999;
          fast_path = false;
        }
      }

      uint32_t addr = addr_base;

      if (fast_path) {
        for (int fy = 0; fy < filter_height; ++fy) {
          int y_offset = fy * dilation_h * input_row_stride;
          const int8_t* base0 = input_batch_base + iy0[0] * input_row_stride +
                                ix0[0] * input_depth + y_offset;
          const int8_t* base1 = input_batch_base + iy0[1] * input_row_stride +
                                ix0[1] * input_depth + y_offset;
          const int8_t* base2 = input_batch_base + iy0[2] * input_row_stride +
                                ix0[2] * input_depth + y_offset;
          const int8_t* base3 = input_batch_base + iy0[3] * input_row_stride +
                                ix0[3] * input_depth + y_offset;

          for (int fx = 0; fx < filter_width; ++fx) {
            int x_offset = fx * dilation_w * input_depth;
            const int8_t* ptr0 = base0 + x_offset;
            const int8_t* ptr1 = base1 + x_offset;
            const int8_t* ptr2 = base2 + x_offset;
            const int8_t* ptr3 = base3 + x_offset;

            int ic = 0;
            for (; ic <= input_depth - 4; ic += 4) {
              for (int i = 0; i < 4; ++i) {
                uint32_t in_pack = ((uint32_t)(uint8_t)(*ptr0++)) << 24;
                in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
                in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
                in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
                cfu_op3(0, addr, in_pack);
                addr += kAddrStep;
              }
            }
            for (; ic < input_depth; ++ic) {
              uint32_t in_pack = ((uint32_t)(uint8_t)(*ptr0++)) << 24;
              in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
              in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
              in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
              cfu_op3(0, addr, in_pack);
              addr += kAddrStep;
            }
          }
        }
      } else {
        for (int fy = 0; fy < filter_height; ++fy) {
          bool y_valid[4];
          int in_y[4];
          for (int s = 0; s < 4; ++s) {
            in_y[s] = iy0[s] + fy * dilation_h;
            y_valid[s] =
                valid_slide[s] && (in_y[s] >= 0) && (in_y[s] < input_height);
          }

          for (int fx = 0; fx < filter_width; ++fx) {
            const int8_t* ptr0;
            const int8_t* ptr1;
            const int8_t* ptr2;
            const int8_t* ptr3;

            int in_x0 = ix0[0] + fx * dilation_w;
            if (y_valid[0] && in_x0 >= 0 && in_x0 < input_width)
              ptr0 = input_batch_base + (in_y[0] * input_row_stride) +
                     (in_x0 * input_depth);
            else
              ptr0 = dummy_pad_buffer;

            int in_x1 = ix0[1] + fx * dilation_w;
            if (y_valid[1] && in_x1 >= 0 && in_x1 < input_width)
              ptr1 = input_batch_base + (in_y[1] * input_row_stride) +
                     (in_x1 * input_depth);
            else
              ptr1 = dummy_pad_buffer;

            int in_x2 = ix0[2] + fx * dilation_w;
            if (y_valid[2] && in_x2 >= 0 && in_x2 < input_width)
              ptr2 = input_batch_base + (in_y[2] * input_row_stride) +
                     (in_x2 * input_depth);
            else
              ptr2 = dummy_pad_buffer;

            int in_x3 = ix0[3] + fx * dilation_w;
            if (y_valid[3] && in_x3 >= 0 && in_x3 < input_width)
              ptr3 = input_batch_base + (in_y[3] * input_row_stride) +
                     (in_x3 * input_depth);
            else
              ptr3 = dummy_pad_buffer;

            int ic = 0;
            for (; ic <= input_depth - 4; ic += 4) {
              // Unroll 4 input channels
              for (int i = 0; i < 4; ++i) {
                uint32_t in_pack = ((uint32_t)(uint8_t)(*ptr0++)) << 24;
                in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
                in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
                in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
                cfu_op3(0, addr,
                        in_pack);  // Write Buffer A (funct3=3, funct7=0)
                addr += kAddrStep;
              }
            }

            for (; ic < input_depth; ++ic) {
              uint32_t in_pack = 0;
              in_pack |= ((uint32_t)(uint8_t)(*ptr0++)) << 24;
              in_pack |= ((uint32_t)(uint8_t)(*ptr1++)) << 16;
              in_pack |= ((uint32_t)(uint8_t)(*ptr2++)) << 8;
              in_pack |= ((uint32_t)(uint8_t)(*ptr3++));
              cfu_op3(0, addr, in_pack);  // Write Buffer A
              addr += kAddrStep;
            }
          }
        }
      }
    };

    const uint32_t bank_addr_flags[2] = {0u, kAddrBankBit};

    // Cache packed A tiles when multiple output-channel tiles reuse them.
    const int num_out_ch_tiles = (output_depth + tile_n - 1) / tile_n;
    const int num_slide_tiles = (M + tile_m - 1) / tile_m;
    constexpr size_t kMaxPackedACacheWords = 1u << 19;
    size_t total_cache_words = 0;
    for (int t = 0; t < num_slide_tiles; ++t) {
      const int slide = t * tile_m;
      const int tile_m_count = std::min(tile_m, M - slide);
      const int row_blocks_active = (tile_m_count + 3) / 4;
      total_cache_words += static_cast<size_t>(row_blocks_active) * K;
    }
    const bool use_a_cache = (num_out_ch_tiles > 1) &&
                             (total_cache_words > 0) &&
                             (total_cache_words <= kMaxPackedACacheWords);
    auto& packed_cache = GetPackedCacheStorage();
    auto& packed_offsets = GetPackedOffsetsStorage();
    if (use_a_cache) {
      packed_cache.resize(total_cache_words);
      packed_offsets.resize(num_slide_tiles);
      size_t cursor = 0;
      for (int t = 0; t < num_slide_tiles; ++t) {
        packed_offsets[t] = static_cast<uint32_t>(cursor);
        const int slide = t * tile_m;
        const int tile_m_count = std::min(tile_m, M - slide);
        const int row_blocks_active = (tile_m_count + 3) / 4;
        uint32_t* tile_base = packed_cache.data() + cursor;
        for (int block = 0; block < row_blocks_active; ++block) {
          const int slide_base = slide + block * 4;
          PackInputTileToBuffer(slide_base, tile_base + block * K);
        }
        cursor += static_cast<size_t>(row_blocks_active) * K;
      }
    }

    auto ValidChannels = [&](int out_ch) {
      return std::min(tile_n, output_depth - out_ch);
    };
    auto TotalBlocks = [&](int valid_channels) {
      return (valid_channels + 3) / 4;
    };
    auto LoadInputTile = [&](int slide, int tile_m_count, int bank) {
      const int row_blocks_active = (tile_m_count + 3) / 4;
      if (row_blocks_active <= 0) {
        return;
      }
      if (use_a_cache) {
        const int tile_index = slide / tile_m;
        const uint32_t* tile_ptr =
            packed_cache.data() + packed_offsets[tile_index];
        const int tile_words = row_blocks_active * K;
        uint32_t addr = bank_addr_flags[bank];
        for (int i = 0; i < tile_words; ++i) {
          cfu_op3(0, addr, tile_ptr[i]);
          addr += kAddrStep;
        }
      } else {
        for (int block = 0; block < row_blocks_active; ++block) {
          const int slide_base = slide + block * 4;
          const uint32_t addr_base =
              (static_cast<uint32_t>(block * K) << 16) | bank_addr_flags[bank];
          PackInputTile(slide_base, addr_base);
        }
      }
    };

    int out_ch = 0;
    int b_bank = 0;
    bool b_ready = false;
    while (out_ch < output_depth) {
      const int valid_channels = ValidChannels(out_ch);
      const int total_blocks = TotalBlocks(valid_channels);
      const int current_block_N = total_blocks * 4;

      if (!b_ready) {
        PackWeights(out_ch, valid_channels, total_blocks,
                    bank_addr_flags[b_bank]);
      }

      int slide = 0;
      int cur_bank = 0;
      int cur_tile_m_count = std::min(tile_m, M - slide);
      LoadInputTile(slide, cur_tile_m_count, cur_bank);

      const int next_out_ch = out_ch + tile_n;
      const int next_b_bank = b_bank ^ 1;
      bool next_b_ready = false;
      bool next_b_packed = false;

      auto ReadOutputTile = [&](int read_slide, int read_tile_m_count,
                                int read_c_bank) {
        const int n_blocks = total_blocks;
        for (int row = 0; row < read_tile_m_count; ++row) {
          const int idx = read_slide + row;
          const int out_y = idx / output_width;
          const int out_x = idx % output_width;
          int8_t* dst = output_batch_base + (out_y * output_row_stride) +
                        (out_x * output_depth) + out_ch;

          uint32_t read_addr = (static_cast<uint32_t>(row * n_blocks) << 16) |
                               bank_addr_flags[read_c_bank];
          for (int c = 0; c < valid_channels; ++c) {
            const int ch = out_ch + c;
            int word_sel = 3 - (c & 3);
            uint32_t cmd = read_addr | static_cast<uint32_t>(word_sel);

            int32_t acc = cfu_op3(3, cmd, 0);
            if ((c & 3) == 3) {
              read_addr += kAddrStep;
            }

            acc += bias_correction[ch];
            acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[ch],
                                                output_shift[ch]);
            acc += output_offset;
            acc = std::max(acc, act_min);
            acc = std::min(acc, act_max);
            dst[c] = (int8_t)acc;
          }
        }
      };

      int c_bank = 0;
      bool have_pending = false;
      int pending_slide = 0;
      int pending_tile_m_count = 0;
      int pending_c_bank = 0;

      while (slide < M) {
        cfu_op3(5, static_cast<uint32_t>(cur_bank), 0);
        cfu_op3(6, static_cast<uint32_t>(b_bank), 0);
        cfu_op3(7, static_cast<uint32_t>(c_bank), 0);
        cfu_op3(2, (K << 16) | (cur_tile_m_count << 8) | current_block_N, 0);

        const int next_slide = slide + tile_m;
        int next_tile_m_count = 0;
        const int next_bank = cur_bank ^ 1;
        if (next_slide < M) {
          next_tile_m_count = std::min(tile_m, M - next_slide);
          LoadInputTile(next_slide, next_tile_m_count, next_bank);
        }

        if (!next_b_packed && next_out_ch < output_depth) {
          const int next_valid_channels = ValidChannels(next_out_ch);
          const int next_total_blocks = TotalBlocks(next_valid_channels);
          PackWeights(next_out_ch, next_valid_channels, next_total_blocks,
                      bank_addr_flags[next_b_bank]);
          next_b_ready = true;
          next_b_packed = true;
        }

        if (have_pending) {
          ReadOutputTile(pending_slide, pending_tile_m_count, pending_c_bank);
        }

        while (cfu_op3(4, 0, 0) != 0);

        pending_slide = slide;
        pending_tile_m_count = cur_tile_m_count;
        pending_c_bank = c_bank;
        have_pending = true;

        slide = next_slide;
        cur_bank = next_bank;
        cur_tile_m_count = next_tile_m_count;
        c_bank ^= 1;
      }

      if (have_pending) {
        ReadOutputTile(pending_slide, pending_tile_m_count, pending_c_bank);
      }

      out_ch = next_out_ch;
      b_bank = next_b_bank;
      b_ready = next_b_ready;
    }
  }  // batch

#if CONV_PROFILE
  detail::LogConvProfile(conv_profile_start, filter_data, input_height,
                         input_width, input_depth, filter_height, filter_width,
                         output_depth, output_height, output_width,
                         stride_height, stride_width, dilation_h, dilation_w);
#endif
}

inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
