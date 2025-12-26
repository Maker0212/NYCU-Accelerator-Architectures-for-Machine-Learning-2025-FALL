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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_LOGISTIC_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_LOGISTIC_H_

#include <algorithm>
#include <limits>

#include "cfu.h"
#include "perf.h"
#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_integer_ops {

inline void Logistic(int32_t input_zero_point, int32_t input_range_radius,
                     int32_t input_multiplier, int32_t input_left_shift,
                     int32_t input_size, const int8_t* input_data,
                     int8_t* output_data) {
  perf_enable_counter(6);
  // 使用 CFU 加速 int8 版本的 logistic，避免軟體重複做 LUT 近似

  // 只保留真的有用到的常數
  static constexpr int32_t kOutputIntegerBits = 8;  // 轉成 Q8 的整數位
  static constexpr int8_t kMinInt8 = std::numeric_limits<int8_t>::min();
  static constexpr int8_t kMaxInt8 = std::numeric_limits<int8_t>::max();
  static constexpr int32_t kOutputZeroPoint = -128;
  static constexpr uint32_t kUnusedOperand = 0;

  // 把輸入從量化中心 (zero_point) 平移回對稱範圍
  for (int i = 0; i < input_size; ++i) {
    const int32_t input =
        static_cast<int32_t>(input_data[i]) - input_zero_point;

    // 飽和邊界（保持原樣）
    if (input <= -input_range_radius) {
      output_data[i] = kMinInt8;
      continue;
    }
    if (input >= input_range_radius) {
      output_data[i] = kMaxInt8;
      continue;
    }

    // int8 -> Q4.27
    const int32_t input_in_q4 = MultiplyByQuantizedMultiplier(
        input, input_multiplier, input_left_shift);

    //    改成呼叫 CFU：function_id = 1, in: Q4.27, out: Q0.31
    //    這裡把運算交給硬體，以 Q0.31 回傳 σ(x) 的近似值
    const int32_t output_in_q0 = static_cast<int32_t>(cfu_op0(
        /*funct7=*/1, static_cast<uint32_t>(input_in_q4), kUnusedOperand));

    // Q0.31 -> Q8，加 zero point，再夾範圍（保持原樣）
    using gemmlowp::RoundingDivideByPOT;
    int32_t output_in_q23 =
        RoundingDivideByPOT(output_in_q0, 31 - kOutputIntegerBits);
    output_in_q23 = std::min(std::max(output_in_q23 + kOutputZeroPoint,
                                      static_cast<int32_t>(kMinInt8)),
                             static_cast<int32_t>(kMaxInt8));
    output_data[i] = static_cast<int8_t>(output_in_q23);
  }

  perf_disable_counter(6);
}

inline void Logistic(int32_t input_multiplier, int32_t input_left_shift,
                     int32_t input_size, const int16_t* ptr_input_data,
                     int16_t* ptr_output_data) {
  // We use the LUT for sigmoid and take into account, that
  // tanh(x) = 2*sigmoid(2*x) - 1

  // We scale by 3/4 to expand range [-8,8]->[-10.7,10.7].
  // In case of general parameter scale, multiplier 3 is taken into account
  // in TanhPrepare function and it is included in
  // input_multiplier already.

  TFLITE_DCHECK_GE(input_left_shift, 0);
  if (input_multiplier == 0) {  // power of two case
    input_multiplier = 3 << input_left_shift;
    input_left_shift = 0;
  }

  int32_t round = (input_left_shift > 0) ? 1 << (input_left_shift - 1) : 0;

  for (int i = 0; i < input_size; ++i, ptr_input_data++, ptr_output_data++) {
    int32_t input_data =
        ((*ptr_input_data) * input_multiplier + round) >> input_left_shift;

    // We do interpolation on unsigned values.
    uint32_t abs_input_data = abs(input_data);

    // We divide by 2 power of 9, because
    // we need to divide by 2 in power of 7 for
    // the input conversion + 1/4 from the scale above.

    // Define uh as uint32_t type not to make this function overflow.
    uint32_t uh = abs_input_data >> 9;
    uint32_t result;

    if (uh >= 255) {
      // Saturate to maximum.
      result = 0x7FFF << 10;
    } else {
      uint32_t ua = sigmoid_table_uint16[uh];
      uint32_t ub = sigmoid_table_uint16[uh + 1];
      uint32_t ut = abs_input_data & 0x1ff;
      // Interpolation is done using the fractional bit.
      result = (ua << 9) + ut * (ub - ua);
    }

    result = (input_data >= 0) ? (result + (1 << 9))
                               : ((1 << (16 + 9)) - result + (1 << 9) - 1);

    // Back to 16-bit.
    result >>= 10;

    *ptr_output_data = result;
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_LOGISTIC_H_
