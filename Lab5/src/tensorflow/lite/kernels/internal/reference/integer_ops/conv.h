/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>
#include <cstdint>
#include <vector>

#include "perf.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tpu_helper.h"

namespace tflite {
namespace reference_integer_ops {

namespace {

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

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          int32_t acc = 0;
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
                const int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                const int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                acc += filter_val * (input_val + input_offset);
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
  perf_enable_counter(6);  // Track full ConvPerChannel execution.

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
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  TFLITE_DCHECK_NE(filter_height, 0);
  TFLITE_DCHECK_NE(filter_width, 0);
  TFLITE_DCHECK_NE(filter_input_depth, 0);
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int groups = input_depth / filter_input_depth;

  const bool needs_filter_sum = (input_offset != 0);
  const int num_patches = output_height * output_width;
  const int patch_size = filter_height * filter_width * filter_input_depth;
  const int input_batch_stride = input_height * input_width * input_depth;
  const int input_row_stride = input_width * input_depth;
  const int input_col_stride = input_depth;
  const int filter_out_stride =
      filter_height * filter_width * filter_input_depth;
  const int filter_row_stride = filter_width * filter_input_depth;
  const int filter_col_stride = filter_input_depth;

  if (!tpu_helper::kHasHardwareTpu) {
    ConvPerChannelReferenceKernel(params, output_multiplier, output_shift,
                                  input_shape, input_data, filter_shape,
                                  filter_data, bias_shape, bias_data,
                                  output_shape, output_data);
    perf_disable_counter(6);
    return;
  }

  if (groups != 1 || patch_size == 0 || output_depth == 0 || num_patches == 0) {
    ConvPerChannelReferenceKernel(params, output_multiplier, output_shift,
                                  input_shape, input_data, filter_shape,
                                  filter_data, bias_shape, bias_data,
                                  output_shape, output_data);
    perf_disable_counter(6);  // Stop ConvPerChannel counter.
    return;
  }

  const int64_t total_workload =
      static_cast<int64_t>(num_patches) * output_depth * patch_size;
  constexpr int64_t kSmallWorkloadThreshold =
      static_cast<int64_t>(16) * 16 * 16;  // Threshold tuned for medium layers.
  if (total_workload <= kSmallWorkloadThreshold) {
    ConvPerChannelReferenceKernel(params, output_multiplier, output_shift,
                                  input_shape, input_data, filter_shape,
                                  filter_data, bias_shape, bias_data,
                                  output_shape, output_data);
    perf_disable_counter(6);
    return;
  }

  const int tile_size = tpu_helper::kTileSize;
  constexpr int kTileArea = tpu_helper::kTileSize * tpu_helper::kTileSize;
  static int32_t c_tile[kTileArea];
  static int32_t acc_tile[kTileArea];
  static int32_t filter_sum_tile[kTileArea];

  int row_out_y[tpu_helper::kTileSize];
  int row_out_x[tpu_helper::kTileSize];
  int row_in_y_origin[tpu_helper::kTileSize];
  int row_in_x_origin[tpu_helper::kTileSize];
  int row_y_start[tpu_helper::kTileSize];
  int row_y_end[tpu_helper::kTileSize];
  int row_x_start[tpu_helper::kTileSize];
  int row_x_end[tpu_helper::kTileSize];
  bool row_active[tpu_helper::kTileSize];
  bool row_full[tpu_helper::kTileSize];
  const int num_k_blocks = (patch_size + tile_size - 1) / tile_size;
  std::vector<int32_t> k_in_channel(num_k_blocks * tile_size, 0);
  std::vector<int32_t> k_filter_x(num_k_blocks * tile_size, 0);
  std::vector<int32_t> k_filter_y(num_k_blocks * tile_size, 0);
  std::vector<int> k_block_sizes(num_k_blocks, 0);

  for (int block = 0; block < num_k_blocks; ++block) {
    const int cur_tile_k = std::min(tile_size, patch_size - block * tile_size);
    k_block_sizes[block] = cur_tile_k;
    int32_t* block_in_channel = k_in_channel.data() + block * tile_size;
    int32_t* block_filter_x = k_filter_x.data() + block * tile_size;
    int32_t* block_filter_y = k_filter_y.data() + block * tile_size;
    for (int col = 0; col < cur_tile_k; ++col) {
      const int global_k = block * tile_size + col;
      int tmp = global_k;
      const int in_channel = tmp % filter_input_depth;
      tmp /= filter_input_depth;
      const int filter_x = tmp % filter_width;
      tmp /= filter_width;
      const int filter_y = tmp;
      block_in_channel[col] = in_channel;
      block_filter_x[col] = filter_x;
      block_filter_y[col] = filter_y;
    }
  }

  const int num_m_tiles = (num_patches + tile_size - 1) / tile_size;
  const size_t cache_entry_count =
      static_cast<size_t>(num_m_tiles) * static_cast<size_t>(num_k_blocks);
  if (cache_entry_count == 0) {
    ConvPerChannelReferenceKernel(params, output_multiplier, output_shift,
                                  input_shape, input_data, filter_shape,
                                  filter_data, bias_shape, bias_data,
                                  output_shape, output_data);
    perf_disable_counter(6);
    return;
  }

  std::vector<int8_t> a_cache(cache_entry_count * kTileArea, 0);
  std::vector<uint8_t> cache_filled(cache_entry_count, 0);
  std::vector<uint32_t> a_packed_cache(
      cache_entry_count * tpu_helper::kPackedWords, 0);
  std::vector<uint16_t> a_packed_words(cache_entry_count, 0);
  std::vector<uint8_t> a_packed_ready(cache_entry_count, 0);
  std::vector<int8_t> b_cache(num_k_blocks * kTileArea, 0);
  std::vector<uint32_t> b_packed_cache(num_k_blocks * tpu_helper::kPackedWords,
                                       0);
  std::vector<uint16_t> b_packed_words(num_k_blocks, 0);
  const int hw_cells = filter_height * filter_width;
  std::vector<int32_t> block_cell_sums(
      needs_filter_sum ? num_k_blocks * hw_cells * tpu_helper::kTileSize : 0,
      0);
  const int prefix_width = filter_width + 1;
  const int prefix_height = filter_height + 1;
  const int prefix_cells = prefix_width * prefix_height;
  std::vector<int32_t> block_prefix_sums(
      needs_filter_sum ? num_k_blocks * prefix_cells * tpu_helper::kTileSize
                       : 0,
      0);
  std::vector<uint8_t> block_prefix_ready(
      needs_filter_sum ? num_k_blocks : 0, 0);
  std::vector<int32_t> filter_full_sums;
  if (needs_filter_sum) {
    filter_full_sums.resize(output_depth);
    for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
      int32_t sum = 0;
      for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
          const int spatial_offset =
              filter_y * filter_row_stride + filter_x * filter_col_stride;
          for (int in_channel = 0; in_channel < filter_input_depth;
               ++in_channel) {
            const int filter_idx =
                out_channel * filter_out_stride + spatial_offset + in_channel;
            sum += static_cast<int32_t>(filter_data[filter_idx]);
          }
        }
      }
      filter_full_sums[out_channel] = sum;
    }
  }

  for (int batch = 0; batch < batches; ++batch) {
    const int batch_base = batch * input_batch_stride;
    std::fill(cache_filled.begin(), cache_filled.end(), 0);
    std::fill(a_packed_ready.begin(), a_packed_ready.end(), 0);
    if (needs_filter_sum) {
      std::fill(block_prefix_ready.begin(), block_prefix_ready.end(), 0);
    }
    for (int n = 0; n < output_depth; n += tile_size) {
      const int cur_tile_n = std::min(tile_size, output_depth - n);

      for (int block = 0; block < num_k_blocks; ++block) {
        const int cur_tile_k = k_block_sizes[block];
        int8_t* block_b = b_cache.data() + block * kTileArea;
        if (cur_tile_k == 0) {
          std::fill(block_b, block_b + kTileArea, 0);
          b_packed_words[block] = 0;
          continue;
        }

        const int32_t* block_in_channel =
            k_in_channel.data() + block * tile_size;
        const int32_t* block_filter_x = k_filter_x.data() + block * tile_size;
        const int32_t* block_filter_y = k_filter_y.data() + block * tile_size;
        const int fill_size = cur_tile_k * cur_tile_n;
        std::fill(block_b, block_b + fill_size, 0);

        for (int row = 0; row < cur_tile_k; ++row) {
          const int filter_y = block_filter_y[row];
          const int filter_x = block_filter_x[row];
          const int in_channel = block_in_channel[row];
          const int spatial_offset = filter_y * filter_row_stride +
                                     filter_x * filter_col_stride + in_channel;
          for (int col = 0; col < cur_tile_n; ++col) {
            const int out_channel = n + col;
            const int filter_idx =
                out_channel * filter_out_stride + spatial_offset;
            const int8_t filter_val = filter_data[filter_idx];
            block_b[row * cur_tile_n + col] = filter_val;
          }
        }
        uint32_t* block_b_packed =
            b_packed_cache.data() + block * tpu_helper::kPackedWords;
        b_packed_words[block] = tpu_helper::PackTypeB(
            block_b, cur_tile_k, cur_tile_n, block_b_packed);
        if (needs_filter_sum) {
          block_prefix_ready[block] = 0;
        }
      }

      for (int m = 0; m < num_patches; m += tile_size) {
        const int cur_tile_m = std::min(tile_size, num_patches - m);
        const int m_tile_index = m / tile_size;
        bool tile_needs_partial = false;
        for (int row = 0; row < cur_tile_m; ++row) {
          const int global_m = m + row;
          const bool active = (global_m < num_patches);
          row_active[row] = active;
          if (active) {
            const int out_y = global_m / output_width;
            const int out_x = global_m % output_width;
            row_out_y[row] = out_y;
            row_out_x[row] = out_x;
            row_in_y_origin[row] = (out_y * stride_height) - pad_height;
            row_in_x_origin[row] = (out_x * stride_width) - pad_width;
            if (needs_filter_sum) {
              ComputeValidFilterRange(
                  row_in_y_origin[row], dilation_height_factor, input_height,
                  filter_height, &row_y_start[row], &row_y_end[row]);
              ComputeValidFilterRange(row_in_x_origin[row],
                                      dilation_width_factor, input_width,
                                      filter_width, &row_x_start[row],
                                      &row_x_end[row]);
              row_full[row] =
                  (row_y_start[row] == 0 && row_y_end[row] == filter_height &&
                   row_x_start[row] == 0 && row_x_end[row] == filter_width);
              if (!row_full[row]) {
                tile_needs_partial = true;
              }
            }
          } else if (needs_filter_sum) {
            row_full[row] = false;
            row_y_start[row] = 0;
            row_y_end[row] = 0;
            row_x_start[row] = 0;
            row_x_end[row] = 0;
          }
        }

        std::fill(acc_tile, acc_tile + kTileArea, 0);
        if (needs_filter_sum && tile_needs_partial) {
          std::fill(filter_sum_tile, filter_sum_tile + kTileArea, 0);
        }

        for (int block = 0; block < num_k_blocks; ++block) {
          const int cur_tile_k = k_block_sizes[block];
          if (cur_tile_k == 0) {
            continue;
          }

          const int8_t* block_b = b_cache.data() + block * kTileArea;
          const int32_t* block_in_channel =
              k_in_channel.data() + block * tile_size;
          const int32_t* block_filter_x = k_filter_x.data() + block * tile_size;
          const int32_t* block_filter_y = k_filter_y.data() + block * tile_size;
          const int32_t* block_prefix =
              needs_filter_sum
                  ? block_prefix_sums.data() +
                        block * prefix_cells * tpu_helper::kTileSize
                  : nullptr;
          const size_t cache_slot =
              static_cast<size_t>(m_tile_index) * num_k_blocks + block;
          const size_t cache_offset = cache_slot * kTileArea;
          int8_t* cached_a = a_cache.data() + cache_offset;

          if (!cache_filled[cache_slot]) {
            for (int row = 0; row < cur_tile_m; ++row) {
              int8_t* row_data = cached_a + row * cur_tile_k;
              std::fill(row_data, row_data + cur_tile_k, 0);
              if (!row_active[row]) {
                continue;
              }
              const int in_y_origin = row_in_y_origin[row];
              const int in_x_origin = row_in_x_origin[row];

              for (int col = 0; col < cur_tile_k; ++col) {
                const int filter_y = block_filter_y[col];
                const int filter_x = block_filter_x[col];
                const int in_channel = block_in_channel[col];
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const bool is_valid = (in_x >= 0) && (in_x < input_width) &&
                                      (in_y >= 0) && (in_y < input_height);
                if (is_valid) {
                  const int index = batch_base + in_y * input_row_stride +
                                    in_x * input_col_stride + in_channel;
                  row_data[col] = input_data[index];
                }
              }
            }
            cache_filled[cache_slot] = 1;
            a_packed_ready[cache_slot] = 0;
          }

          if (!a_packed_ready[cache_slot]) {
            uint32_t* packed_a =
                a_packed_cache.data() + cache_slot * tpu_helper::kPackedWords;
            a_packed_words[cache_slot] = tpu_helper::PackTypeA(
                cached_a, cur_tile_m, cur_tile_k, packed_a);
            a_packed_ready[cache_slot] = 1;
          }

          if (needs_filter_sum && tile_needs_partial) {
            if (!block_prefix_ready[block]) {
              int32_t* block_cells = block_cell_sums.data() +
                                     block * hw_cells * tpu_helper::kTileSize;
              std::fill(block_cells,
                        block_cells + hw_cells * tpu_helper::kTileSize, 0);
              for (int row = 0; row < cur_tile_k; ++row) {
                const int filter_y = block_filter_y[row];
                const int filter_x = block_filter_x[row];
                int32_t* cell_sum =
                    block_cells +
                    (filter_y * filter_width + filter_x) *
                        tpu_helper::kTileSize;
                for (int col = 0; col < cur_tile_n; ++col) {
                  cell_sum[col] +=
                      static_cast<int32_t>(block_b[row * cur_tile_n + col]);
                }
              }
              int32_t* block_prefix_write =
                  block_prefix_sums.data() +
                  block * prefix_cells * tpu_helper::kTileSize;
              std::fill(
                  block_prefix_write,
                  block_prefix_write + prefix_cells * tpu_helper::kTileSize, 0);
              for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                  const int cell_idx = (filter_y * filter_width + filter_x) *
                                       tpu_helper::kTileSize;
                  const int prefix_idx =
                      ((filter_y + 1) * prefix_width + (filter_x + 1)) *
                      tpu_helper::kTileSize;
                  const int prefix_left =
                      ((filter_y + 1) * prefix_width + filter_x) *
                      tpu_helper::kTileSize;
                  const int prefix_up =
                      (filter_y * prefix_width + (filter_x + 1)) *
                      tpu_helper::kTileSize;
                  const int prefix_diag = (filter_y * prefix_width + filter_x) *
                                          tpu_helper::kTileSize;
                  for (int col = 0; col < cur_tile_n; ++col) {
                    block_prefix_write[prefix_idx + col] =
                        block_cells[cell_idx + col] +
                        block_prefix_write[prefix_left + col] +
                        block_prefix_write[prefix_up + col] -
                        block_prefix_write[prefix_diag + col];
                  }
                }
              }
              block_prefix_ready[block] = 1;
              block_prefix = block_prefix_write;
            }

            for (int row = 0; row < cur_tile_m; ++row) {
              if (!row_active[row] || row_full[row]) {
                continue;
              }
              int32_t* sum_row = &filter_sum_tile[row * tpu_helper::kTileSize];
              const int y_start = row_y_start[row];
              const int y_end = row_y_end[row];
              const int x_start = row_x_start[row];
              const int x_end = row_x_end[row];
              if (y_start >= y_end || x_start >= x_end) {
                continue;
              }
              const int idx_y1 = y_end * prefix_width;
              const int idx_y0 = y_start * prefix_width;
              for (int out_c = 0; out_c < cur_tile_n; ++out_c) {
                const int idx11 =
                    (idx_y1 + x_end) * tpu_helper::kTileSize + out_c;
                const int idx10 =
                    (idx_y1 + x_start) * tpu_helper::kTileSize + out_c;
                const int idx01 =
                    (idx_y0 + x_end) * tpu_helper::kTileSize + out_c;
                const int idx00 =
                    (idx_y0 + x_start) * tpu_helper::kTileSize + out_c;
                sum_row[out_c] += block_prefix[idx11] - block_prefix[idx10] -
                                  block_prefix[idx01] + block_prefix[idx00];
              }
            }
          }

          const uint32_t* packed_a =
              a_packed_cache.data() + cache_slot * tpu_helper::kPackedWords;
          const uint32_t* packed_b =
              b_packed_cache.data() + block * tpu_helper::kPackedWords;
          const int words_a = a_packed_words[cache_slot];
          const int words_b = b_packed_words[block];
          bool ok = false;
          if (words_a > 0 && words_b > 0) {
            ok = tpu_helper::RunMatmulTilePacked(
                packed_a, words_a, packed_b, words_b, cur_tile_m, cur_tile_k,
                cur_tile_n, c_tile);
          }
          if (!ok) {
            for (int row = 0; row < cur_tile_m; ++row) {
              for (int col = 0; col < cur_tile_n; ++col) {
                int32_t acc = 0;
                for (int kk = 0; kk < cur_tile_k; ++kk) {
                  acc += static_cast<int32_t>(cached_a[row * cur_tile_k + kk]) *
                         static_cast<int32_t>(block_b[kk * cur_tile_n + col]);
                }
                c_tile[row * tpu_helper::kTileSize + col] = acc;
              }
            }
          }

          for (int row = 0; row < cur_tile_m; ++row) {
            for (int col = 0; col < cur_tile_n; ++col) {
              acc_tile[row * tpu_helper::kTileSize + col] +=
                  c_tile[row * tpu_helper::kTileSize + col];
            }
          }
        }

        for (int row = 0; row < cur_tile_m; ++row) {
          if (!row_active[row]) {
            continue;
          }
          const int out_y = row_out_y[row];
          const int out_x = row_out_x[row];
          for (int col = 0; col < cur_tile_n; ++col) {
            const int out_channel = n + col;
            int32_t acc = acc_tile[row * tpu_helper::kTileSize + col];
            if (needs_filter_sum) {
              int32_t filter_sum = filter_full_sums[out_channel];
              if (!row_full[row]) {
                filter_sum =
                    filter_sum_tile[row * tpu_helper::kTileSize + col];
              }
              acc += input_offset * filter_sum;
            }
            if (bias_data) {
              acc += bias_data[out_channel];
            }
            acc = MultiplyByQuantizedMultiplier(
                acc, output_multiplier[out_channel], output_shift[out_channel]);
            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            output_data[Offset(output_shape, batch, out_y, out_x,
                               out_channel)] = static_cast<int8_t>(acc);
          }
        }
      }
    }
  }

  perf_disable_counter(6);  // Stop ConvPerChannel counter.
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
