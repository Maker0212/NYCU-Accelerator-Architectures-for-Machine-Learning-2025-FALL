/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_

#include <algorithm>
#include <limits>
#include <type_traits>

#include "cfu.h"
#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_ops {

inline void LeakyRelu(const tflite::LeakyReluParams& params,
                      const RuntimeShape& input_shape, const float* input_data,
                      const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  const float* input_ptr = input_data;
  float* output_ptr = output_data;
  const int limit4 = flat_size & ~3;
  int i = 0;
  for (; i < limit4; i += 4) {
    const float v0 = input_ptr[0];
    const float v1 = input_ptr[1];
    const float v2 = input_ptr[2];
    const float v3 = input_ptr[3];
    output_ptr[0] = (v0 > 0) ? v0 : v0 * params.alpha;
    output_ptr[1] = (v1 > 0) ? v1 : v1 * params.alpha;
    output_ptr[2] = (v2 > 0) ? v2 : v2 * params.alpha;
    output_ptr[3] = (v3 > 0) ? v3 : v3 * params.alpha;
    input_ptr += 4;
    output_ptr += 4;
  }
  for (; i < flat_size; ++i, ++input_ptr, ++output_ptr) {
    const float val = *input_ptr;
    output_ptr[0] = (val > 0) ? val : val * params.alpha;
  }
}

template <typename T>
inline void QuantizeLeakyRelu(const LeakyReluParams& params,
                              const RuntimeShape& input_shape,
                              const T* input_data,
                              const RuntimeShape& output_shape,
                              T* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  static const int32_t quantized_min = std::numeric_limits<T>::min();
  static const int32_t quantized_max = std::numeric_limits<T>::max();
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier_identity = params.output_multiplier_identity;
  const int32_t output_multiplier_alpha = params.output_multiplier_alpha;
  const int output_shift_identity = params.output_shift_identity;
  const int output_shift_alpha = params.output_shift_alpha;

  if (std::is_same<T, int8_t>::value) {
    struct LutCache {
      bool valid;
      int32_t input_offset;
      int32_t output_offset;
      int32_t mult_identity;
      int32_t mult_alpha;
      int shift_identity;
      int shift_alpha;
      int8_t table[256];
    };
    static LutCache cache = {};

    const bool needs_rebuild =
        (!cache.valid) || (cache.input_offset != input_offset) ||
        (cache.output_offset != output_offset) ||
        (cache.mult_identity != output_multiplier_identity) ||
        (cache.mult_alpha != output_multiplier_alpha) ||
        (cache.shift_identity != output_shift_identity) ||
        (cache.shift_alpha != output_shift_alpha);
    if (needs_rebuild) {
      for (int raw = std::numeric_limits<int8_t>::min();
           raw <= std::numeric_limits<int8_t>::max(); ++raw) {
        const int8_t q_val = static_cast<int8_t>(raw);
        const uint8_t lut_index = static_cast<uint8_t>(q_val);
        const int32_t input_value = static_cast<int32_t>(q_val) - input_offset;
        int32_t unclamped_output;
        if (input_value >= 0) {
          unclamped_output =
              output_offset + MultiplyByQuantizedMultiplier(
                                  input_value, output_multiplier_identity,
                                  output_shift_identity);
        } else {
          unclamped_output =
              output_offset +
              MultiplyByQuantizedMultiplier(
                  input_value, output_multiplier_alpha, output_shift_alpha);
        }
        const int32_t clamped_output =
            std::min(quantized_max, std::max(quantized_min, unclamped_output));
        cache.table[lut_index] = static_cast<int8_t>(clamped_output);
      }
      cache.valid = true;
      cache.input_offset = input_offset;
      cache.output_offset = output_offset;
      cache.mult_identity = output_multiplier_identity;
      cache.mult_alpha = output_multiplier_alpha;
      cache.shift_identity = output_shift_identity;
      cache.shift_alpha = output_shift_alpha;

      // Upload to CFU
      for (int i = 0; i < 256; i += 4) {
        uint32_t val = 0;
        val |= ((uint8_t)cache.table[i]);
        val |= ((uint8_t)cache.table[i + 1]) << 8;
        val |= ((uint8_t)cache.table[i + 2]) << 16;
        val |= ((uint8_t)cache.table[i + 3]) << 24;
        cfu_op5(0, i, val);
      }
    }

    const int8_t* input_ptr = reinterpret_cast<const int8_t*>(input_data);
    int8_t* output_ptr = reinterpret_cast<int8_t*>(output_data);
    const int8_t* lut = cache.table;

    int i = 0;
    const int limit4 = flat_size & ~3;
    for (; i < limit4; i += 4) {
      uint32_t in_val = *((const uint32_t*)(input_ptr + i));
      uint32_t out_val = cfu_op5(1, in_val, 0);
      *((uint32_t*)(output_ptr + i)) = out_val;
    }
    for (; i < flat_size; ++i) {
      output_ptr[i] = lut[static_cast<uint8_t>(input_ptr[i])];
    }
    return;
  }

  const T* input_ptr = input_data;
  T* output_ptr = output_data;
  const int limit4 = flat_size & ~3;
  int i = 0;
  for (; i < limit4; i += 4) {
    int32_t in0 = static_cast<int32_t>(input_ptr[0]) - input_offset;
    int32_t in1 = static_cast<int32_t>(input_ptr[1]) - input_offset;
    int32_t in2 = static_cast<int32_t>(input_ptr[2]) - input_offset;
    int32_t in3 = static_cast<int32_t>(input_ptr[3]) - input_offset;

    int32_t out0 =
        output_offset +
        MultiplyByQuantizedMultiplier(
            in0,
            (in0 >= 0) ? output_multiplier_identity : output_multiplier_alpha,
            (in0 >= 0) ? output_shift_identity : output_shift_alpha);
    int32_t out1 =
        output_offset +
        MultiplyByQuantizedMultiplier(
            in1,
            (in1 >= 0) ? output_multiplier_identity : output_multiplier_alpha,
            (in1 >= 0) ? output_shift_identity : output_shift_alpha);
    int32_t out2 =
        output_offset +
        MultiplyByQuantizedMultiplier(
            in2,
            (in2 >= 0) ? output_multiplier_identity : output_multiplier_alpha,
            (in2 >= 0) ? output_shift_identity : output_shift_alpha);
    int32_t out3 =
        output_offset +
        MultiplyByQuantizedMultiplier(
            in3,
            (in3 >= 0) ? output_multiplier_identity : output_multiplier_alpha,
            (in3 >= 0) ? output_shift_identity : output_shift_alpha);

    out0 = std::min(quantized_max, std::max(quantized_min, out0));
    out1 = std::min(quantized_max, std::max(quantized_min, out1));
    out2 = std::min(quantized_max, std::max(quantized_min, out2));
    out3 = std::min(quantized_max, std::max(quantized_min, out3));

    output_ptr[0] = static_cast<T>(out0);
    output_ptr[1] = static_cast<T>(out1);
    output_ptr[2] = static_cast<T>(out2);
    output_ptr[3] = static_cast<T>(out3);

    input_ptr += 4;
    output_ptr += 4;
  }

  for (; i < flat_size; ++i, ++input_ptr, ++output_ptr) {
    const int32_t input_value = static_cast<int32_t>(*input_ptr) - input_offset;
    const int32_t mult = (input_value >= 0) ? output_multiplier_identity
                                            : output_multiplier_alpha;
    const int shift =
        (input_value >= 0) ? output_shift_identity : output_shift_alpha;
    int32_t unclamped_output =
        output_offset + MultiplyByQuantizedMultiplier(input_value, mult, shift);
    const T clamped_output =
        std::min(quantized_max, std::max(quantized_min, unclamped_output));
    *output_ptr = static_cast<T>(clamped_output);
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LEAKY_RELU_H_
