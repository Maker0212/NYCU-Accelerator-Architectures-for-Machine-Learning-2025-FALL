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

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

#include "cfu.h"   // 讓你能呼叫 cfu_op1 / cfu_op2



namespace tflite {
namespace reference_integer_ops {

// 卷積運算：int8 x int8 -> int8 (per-channel quantization)
// 使用 CFU 來加速最內層的 MAC 計算
inline void ConvPerChannel(
    const ConvParams& params,
    const int32_t* output_multiplier,
    const int32_t* output_shift,
    const RuntimeShape& input_shape,   const int8_t*  input_data,
    const RuntimeShape& filter_shape,  const int8_t*  filter_data,
    const RuntimeShape& bias_shape,    const int32_t* bias_data,
    const RuntimeShape& output_shape,        int8_t*  output_data)
{
  // ====== 基本維度解析 (NHWC 輸入, OHWI 權重) ======
  const int batches       = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_h       = input_shape.Dims(1);
  const int input_w       = input_shape.Dims(2);
  const int input_depth   = input_shape.Dims(3);

  const int filter_out_ch = filter_shape.Dims(0);
  const int filter_h      = filter_shape.Dims(1);
  const int filter_w      = filter_shape.Dims(2);
  const int filter_in_ch  = filter_shape.Dims(3);  // 必須等於 input_depth

  const int output_h      = output_shape.Dims(1);
  const int output_w      = output_shape.Dims(2);
  const int output_depth  = output_shape.Dims(3);  // 必須等於 filter_out_ch

  // 驗證維度一致性
  TFLITE_DCHECK_EQ(filter_in_ch, input_depth);
  TFLITE_DCHECK_EQ(filter_out_ch, output_depth);

  // ====== 量化相關參數 ======
  const int32_t input_offset     = params.input_offset;   // 輸入 zero-point
  const int32_t output_offset    = params.output_offset;  // 輸出 zero-point
  const int32_t output_activation_min = params.quantized_activation_min; // ReLU6 下限
  const int32_t output_activation_max = params.quantized_activation_max; // ReLU6 上限

  // 卷積步長與 dilation / padding
  const int stride_h   = params.stride_height;
  const int stride_w   = params.stride_width;
  const int dilation_h = params.dilation_height_factor;
  const int dilation_w = params.dilation_width_factor;
  const int pad_h      = params.padding_values.height;
  const int pad_w      = params.padding_values.width;

  // ====== 初始化 CFU ======
  // FID=0 → 設定 input_offset，之後每次 MAC 計算都會自動加上 offset
  (void) cfu_op0(0, input_offset, 0);

  // ====== Fast path: 針對 1x1 kernel、stride=1、無 padding 的最佳化 ======
  const bool is_1x1_fast =
      (filter_h == 1 && filter_w == 1 &&
       stride_h == 1 && stride_w == 1 &&
       dilation_h == 1 && dilation_w == 1 &&
       pad_h == 0 && pad_w == 0);

  if (is_1x1_fast) {
    for (int b = 0; b < batches; ++b) {
      for (int y = 0; y < output_h; ++y) {
        // 輸入 y 與輸出 y 對齊（因為 stride=1, pad=0）
        const int in_y = y;
        const int in_y_base_in  = (b*input_h + in_y) * input_w * input_depth;
        const int out_y_base_out = (b*output_h + y) * output_w * output_depth;

        for (int x = 0; x < output_w; ++x) {
          const int in_x  = x;
          const int in_base  = in_y_base_in  + in_x  * input_depth;
          const int out_xy   = out_y_base_out + x    * output_depth;

          const uint8_t* in_ptr_base = reinterpret_cast<const uint8_t*>(&input_data[in_base]);

          // ====== 每個 output channel（oc）對應一組 filter ======
          for (int oc = 0; oc < output_depth; ++oc) {
            // (1) 清除累加器 → FID=2
            (void) cfu_op2(0,0,0);
            // (2) 若有 bias，先加到累加器 → FID=4
            if (bias_data) (void) cfu_op4(0, static_cast<uint32_t>(bias_data[oc]), 0);

            // 權重起點 (OHWI 格式: oc, fy=0, fx=0, ic)
            const int w_base = oc * filter_h * filter_w * filter_in_ch;
            const uint8_t* w_ptr = reinterpret_cast<const uint8_t*>(&filter_data[w_base]);

            // (3) 向量化 4-way dot product → FID=1
            int ic = 0;
            for (; ic + 3 < filter_in_ch; ic += 4) {
              uint32_t x_pack = *(const uint32_t*)(in_ptr_base + ic);
              uint32_t w_pack = *(const uint32_t*)(w_ptr       + ic);
              (void) cfu_op1(0, x_pack, w_pack);
            }
            // (4) 處理尾端不足 4 的元素 → FID=3 (scalar MAC)
            for (; ic < filter_in_ch; ++ic) {
              (void) cfu_op3(0, in_ptr_base[ic], w_ptr[ic]);
            }

            // (5) 取出累加器結果（同時清零）→ FID=2
            int32_t acc = (int32_t) cfu_op2(0,0,0);
            // 量化縮放（per-channel multiplier/shift）
            acc = tflite::MultiplyByQuantizedMultiplier(acc, output_multiplier[oc], output_shift[oc]);
            acc += output_offset;
            // 飽和裁剪
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);

            // (6) 寫回輸出
            output_data[out_xy + oc] = static_cast<int8_t>(acc);
          } // oc
        } // x
      } // y
    } // b
    return; // fast-path 已完成
  }

  // ====== 一般路徑：支援任意 kernel, stride, dilation, padding ======
  for (int b = 0; b < batches; ++b) {
    for (int out_y = 0; out_y < output_h; ++out_y) {
      const int in_y_origin = out_y * stride_h - pad_h;
      for (int out_x = 0; out_x < output_w; ++out_x) {
        const int in_x_origin = out_x * stride_w - pad_w;

        const int out_base = ((b*output_h + out_y)*output_w + out_x) * output_depth;

        for (int oc = 0; oc < output_depth; ++oc) {
          // (1) 清除累加器 → FID=2
          (void) cfu_op2(0,0,0);
          // (2) 加 bias → FID=4
          if (bias_data) (void) cfu_op4(0, static_cast<uint32_t>(bias_data[oc]), 0);

          // 權重基址（固定 oc，後面只在 fy, fx, ic 上掃）
          const int w_oc_base = oc * filter_h * filter_w * filter_in_ch;

          for (int fy = 0; fy < filter_h; ++fy) {
            const int in_y = in_y_origin + dilation_h * fy;
            if ((unsigned)in_y >= (unsigned)input_h) continue;

            const int in_row_base = ((b*input_h + in_y) * input_w) * input_depth;

            for (int fx = 0; fx < filter_w; ++fx) {
              const int in_x = in_x_origin + dilation_w * fx;
              if ((unsigned)in_x >= (unsigned)input_w) continue;

              const int in_base = in_row_base + in_x * input_depth;

              const uint8_t* in_ptr = reinterpret_cast<const uint8_t*>(&input_data[in_base]);
              const uint8_t* w_ptr  = reinterpret_cast<const uint8_t*>(
                  &filter_data[w_oc_base + (fy*filter_w + fx)*filter_in_ch]);

              // (3) 向量化 4-way dot → FID=1
              int ic = 0;
              for (; ic + 3 < filter_in_ch; ic += 4) {
                uint32_t x_pack = *(const uint32_t*)(in_ptr + ic);
                uint32_t w_pack = *(const uint32_t*)(w_ptr  + ic);
                (void) cfu_op1(0, x_pack, w_pack);
              }
              // (4) 處理尾端元素 → FID=3
              for (; ic < filter_in_ch; ++ic) {
                (void) cfu_op3(0, in_ptr[ic], w_ptr[ic]);
              }
            } // fx
          } // fy

          // (5) 取出累加器結果 → FID=2
          int32_t acc = (int32_t) cfu_op2(0,0,0);
          acc = tflite::MultiplyByQuantizedMultiplier(acc, output_multiplier[oc], output_shift[oc]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);

          // (6) 寫回輸出
          output_data[out_base + oc] = static_cast<int8_t>(acc);
        } // oc
      } // out_x
    } // out_y
  } // b
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
