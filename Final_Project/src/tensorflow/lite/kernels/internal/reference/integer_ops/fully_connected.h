/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "cfu.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

namespace tflite {
namespace reference_integer_ops {
namespace detail {
inline uint32_t Pack4Bytes(const int8_t* values) {
  if ((reinterpret_cast<uintptr_t>(values) & 3u) == 0u) {
    return *reinterpret_cast<const uint32_t*>(values);
  }
  return static_cast<uint8_t>(values[0]) |
         (static_cast<uint8_t>(values[1]) << 8) |
         (static_cast<uint8_t>(values[2]) << 16) |
         (static_cast<uint8_t>(values[3]) << 24);
}
}  // namespace detail

// For per-channel functions, since it is defined in quantization spec that
// weights are symmetric
// (https://www.tensorflow.org/lite/performance/quantization_spec#symmetric_vs_asymmetric),
// zero_point (params.weights_offset) is always 0.
// However, for per-tensor functions, params.weights_offset is still applied for
// backward compatibility.

inline void FullyConnectedPerChannel(
    const FullyConnectedParams& params, const int32_t* output_multiplier,
    const int* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = output_shape.Dims(0);
  const int output_depth = output_shape.Dims(1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

  // Program CFU with input offset once per call.
  cfu_op0(0, static_cast<uint8_t>(static_cast<int8_t>(input_offset)), 0);

  for (int b = 0; b < batches; ++b) {
    const int8_t* input_base = input_data + b * accum_depth;
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      const int8_t* filter_base = filter_data + out_c * accum_depth;
      int32_t acc = 0;

      int d = 0;
      for (; d + 3 < accum_depth; d += 4) {
        const uint32_t packed_input = detail::Pack4Bytes(input_base + d);
        const uint32_t packed_filter = detail::Pack4Bytes(filter_base + d);
        acc += static_cast<int32_t>(cfu_op0(1, packed_input, packed_filter));
      }
      for (; d < accum_depth; ++d) {
        const int32_t input_val = input_base[d];
        const int32_t filter_val = filter_base[d];
        acc += filter_val * (input_val + input_offset);
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[out_c],
                                          output_shift[out_c]);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int8_t>(acc);
    }
  }
}

template <typename AccumScalar>
inline void FullyConnectedPerChannel(
    const FullyConnectedParams& params, const int32_t* output_multiplier,
    const int* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      AccumScalar acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += filter_val * input_val;
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      int32_t acc_scaled = MultiplyByQuantizedMultiplier(
          acc, output_multiplier[out_c], output_shift[out_c]);
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int16_t>(acc_scaled);
    }
  }
}

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

  // Fast path when weights_offset is zero; otherwise fall back to original
  // scalar path to preserve correctness.
  const bool can_use_cfu = (filter_offset == 0);
  if (can_use_cfu) {
    cfu_op0(0, static_cast<uint8_t>(static_cast<int8_t>(input_offset)), 0);
  }

  for (int b = 0; b < batches; ++b) {
    const int8_t* input_base = input_data + b * accum_depth;
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      const int8_t* filter_base = filter_data + out_c * accum_depth;
      int32_t acc = 0;
      int d = 0;
      if (can_use_cfu) {
        for (; d + 15 < accum_depth; d += 16) {
          const uint32_t packed_input0 = detail::Pack4Bytes(input_base + d);
          const uint32_t packed_filter0 = detail::Pack4Bytes(filter_base + d);
          const uint32_t packed_input1 = detail::Pack4Bytes(input_base + d + 4);
          const uint32_t packed_filter1 =
              detail::Pack4Bytes(filter_base + d + 4);
          const uint32_t packed_input2 = detail::Pack4Bytes(input_base + d + 8);
          const uint32_t packed_filter2 =
              detail::Pack4Bytes(filter_base + d + 8);
          const uint32_t packed_input3 =
              detail::Pack4Bytes(input_base + d + 12);
          const uint32_t packed_filter3 =
              detail::Pack4Bytes(filter_base + d + 12);
          acc +=
              static_cast<int32_t>(cfu_op0(1, packed_input0, packed_filter0));
          acc +=
              static_cast<int32_t>(cfu_op0(1, packed_input1, packed_filter1));
          acc +=
              static_cast<int32_t>(cfu_op0(1, packed_input2, packed_filter2));
          acc +=
              static_cast<int32_t>(cfu_op0(1, packed_input3, packed_filter3));
        }
        for (; d + 7 < accum_depth; d += 8) {
          const uint32_t packed_input0 = detail::Pack4Bytes(input_base + d);
          const uint32_t packed_filter0 = detail::Pack4Bytes(filter_base + d);
          const uint32_t packed_input1 = detail::Pack4Bytes(input_base + d + 4);
          const uint32_t packed_filter1 =
              detail::Pack4Bytes(filter_base + d + 4);
          acc +=
              static_cast<int32_t>(cfu_op0(1, packed_input0, packed_filter0));
          acc +=
              static_cast<int32_t>(cfu_op0(1, packed_input1, packed_filter1));
        }
        for (; d + 3 < accum_depth; d += 4) {
          const uint32_t packed_input = detail::Pack4Bytes(input_base + d);
          const uint32_t packed_filter = detail::Pack4Bytes(filter_base + d);
          acc += static_cast<int32_t>(cfu_op0(1, packed_input, packed_filter));
        }
      }
      for (; d < accum_depth; ++d) {
        int32_t input_val = input_base[d];
        int32_t filter_val = filter_base[d] + filter_offset;
        acc += filter_val * (input_val + input_offset);
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int8_t>(acc);
    }
  }
}

inline void FullyConnectedWithPackedInt4Weights(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK_NE(unpacked_filter_data, nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_data, filter_shape.FlatSize(), unpacked_filter_data);
  FullyConnected(params, input_shape, input_data, filter_shape,
                 unpacked_filter_data, bias_shape, bias_data, output_shape,
                 output_data);
}

template <typename AccumScalar>
inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      AccumScalar acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += (filter_val + filter_offset) * input_val;
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      int32_t acc_scaled =
          MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] = static_cast<int16_t>(acc_scaled);
    }
  }
}

inline void FullyConnectedWithPackedInt4Weights(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int16_t* output_data) {
  TFLITE_DCHECK_NE(unpacked_filter_data, nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_data, filter_shape.FlatSize(), unpacked_filter_data);
  FullyConnected<int32_t>(params, input_shape, input_data, filter_shape,
                          unpacked_filter_data, bias_shape, bias_data,
                          output_shape, output_data);
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_
