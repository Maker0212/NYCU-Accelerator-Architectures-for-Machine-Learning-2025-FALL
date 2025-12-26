#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_

#include <algorithm>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

// 使用與 Conv 相同的 CFU 介面
#include "cfu.h"

namespace tflite {
namespace reference_integer_ops {

// 說明：per-channel 函式根據量化規格，weights 為對稱量化（zero_point=0）。
// per-tensor 版本為相容性保留，weights_offset 可能 ≠ 0。

// =========================
// int8 x int8 -> int8 (per-channel)  <<-- 使用 CFU 加速版本
// =========================
inline void FullyConnectedPerChannel(
    const FullyConnectedParams& params, const int32_t* output_multiplier,
    const int* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {

  // 量化參數：per-channel 中 weights_offset=0（對稱量化），只需修正 input_offset
  const int32_t input_offset = params.input_offset;       // (x + x0)
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // 形狀檢查（FC 的 output 是 [batch, out_c]）
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = output_shape.Dims(0);
  const int output_depth = output_shape.Dims(1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1); // 內積長度

  // (0) 將 input_offset 設定進 CFU（FID=0），之後 MAC 會自動加上 offset
  (void) cfu_op0(0, input_offset, 0);

  for (int b = 0; b < batches; ++b) {
    // 當前 batch 的輸入向量基址（長度為 accum_depth）
    const int base_in = b * accum_depth;
    const uint8_t* in_ptr_base =
        reinterpret_cast<const uint8_t*>(&input_data[base_in]);

    for (int out_c = 0; out_c < output_depth; ++out_c) {
      // (1) 清 CFU 累加器 → FID=2；(2) 先把 bias 加入累加器 → FID=4
      (void) cfu_op2(0, 0, 0); // 讀並清（此處回傳值可忽略）
      if (bias_data) {
        (void) cfu_op4(0, static_cast<uint32_t>(bias_data[out_c]), 0);
      }

      // 權重視為 [out_c, d] 的連續區塊（長度 accum_depth）
      const int base_w = out_c * accum_depth;
      const uint8_t* w_ptr_base =
          reinterpret_cast<const uint8_t*>(&filter_data[base_w]);

      // (3) 主體：4-way dot 向量化 → FID=1
      int d = 0;
      for (; d + 3 < accum_depth; d += 4) {
        // 以連續 4B 打包，避免索引乘法
        uint32_t x_pack = *(const uint32_t*)(in_ptr_base + d);
        uint32_t w_pack = *(const uint32_t*)(w_ptr_base + d);
        (void) cfu_op1(0, x_pack, w_pack);
      }
      // (4) 尾段（1~3 個元素）：scalar MAC → FID=3
      for (; d < accum_depth; ++d) {
        (void) cfu_op3(0, in_ptr_base[d], w_ptr_base[d]);
      }

      // (5) 取回累加結果（同時清零）→ FID=2
      int32_t acc = static_cast<int32_t>(cfu_op2(0, 0, 0));

      // (6) per-channel 量化縮放（對應 out_c），加 output_offset，做飽和
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[out_c],
                                          output_shift[out_c]);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);

      // (7) 寫回輸出：布局 [b, out_c]
      output_data[out_c + output_depth * b] = static_cast<int8_t>(acc);
    }
  }
}

// =========================
// int16 x int8 -> int16 (per-channel)  (原樣保留，未用 CFU)
// =========================
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

// =========================
// int8 x int8 -> int8（per-tensor）<<-- 使用 CFU，加速 & 正確處理 weights_offset
// =========================
inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {

  // per-tensor：input_offset 與 weights_offset 都可能 ≠ 0
  const int32_t input_offset  = params.input_offset;
  const int32_t filter_offset = params.weights_offset;   // 注意：per-tensor 校正要自己處理
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier; // per-tensor 量化
  const int      output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  const int filter_dim_count = filter_shape.DimensionsCount();
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches      = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth  = filter_shape.Dims(filter_dim_count - 1);

  // (0) 將 input_offset 設進 CFU（FID=0）→ CFU 在 MAC 時自動套用 (x + x0)
  (void) cfu_op0(0, input_offset, 0);

  // (重點) per-tensor weights_offset 的數學校正：
  //   sum_d ( (w_d + w0) * (x_d + x0) )
  // = sum_d [ w_d * (x_d + x0) ]  +  w0 * sum_d (x_d + x0)
  // 第一項 → 用 CFU (FID=1/3) 跑完；
  // 第二項 → 先在每個 batch 算 S = sum_d (x_d + x0)，之後每個 out_c 再加 w0*S。

  for (int b = 0; b < batches; ++b) {
    // --- (a) 計算 S = sum_d (x_d + input_offset) ---
    const int base_in = b * accum_depth;
    const uint8_t* in_ptr_base = reinterpret_cast<const uint8_t*>(&input_data[base_in]);

    // 先清 CFU 累加器 → FID=2
    (void) cfu_op2(0, 0, 0);

    // 以 4 為步長，令 w_pack = 0x01010101，相當於把四個 (x+offset) 相加
    const uint32_t ONE4 = 0x01010101u;
    int d = 0;
    for (; d + 3 < accum_depth; d += 4) {
      uint32_t x_pack = *(const uint32_t*)(in_ptr_base + d);
      (void) cfu_op1(0, x_pack, ONE4);   // FID=1：累加 (x+offset) * 1
    }
    // 尾段：scalar MAC，權重=1 → FID=3
    for (; d < accum_depth; ++d) {
      (void) cfu_op3(0, in_ptr_base[d], 1);
    }
    // 讀出 S 並清零 → FID=2
    int32_t S = static_cast<int32_t>(cfu_op2(0, 0, 0));

    // --- (b) 對每個 out_c 執行主計算 ---
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      // 清 acc + 加 bias（若有）→ FID=2 + FID=4
      (void) cfu_op2(0, 0, 0);
      if (bias_data) {
        (void) cfu_op4(0, static_cast<uint32_t>(bias_data[out_c]), 0);
      }

      const int base_w = out_c * accum_depth;
      const uint8_t* w_ptr_base =
          reinterpret_cast<const uint8_t*>(&filter_data[base_w]);

      // (1) 主體： sum_d [ w_d * (x_d + x0) ] → FID=1/3
      int k = 0;
      for (; k + 3 < accum_depth; ++k) {
        // 連續 4B 打包
        uint32_t x_pack = *(const uint32_t*)(in_ptr_base + k);
        uint32_t w_pack = *(const uint32_t*)(w_ptr_base + k);
        (void) cfu_op1(0, x_pack, w_pack); // 4-way dot
        k += 3; // for 末尾會再 ++k → 累計 +4
      }
      for (; k < accum_depth; ++k) {
        (void) cfu_op3(0, in_ptr_base[k], w_ptr_base[k]); // scalar 尾段
      }

      // (2) 校正：+ weights_offset * S  → FID=4
      if (filter_offset != 0) {
        int32_t corr = filter_offset * S; // int32 足夠
        (void) cfu_op4(0, static_cast<uint32_t>(corr), 0);
      }

      // (3) 取回 acc、做 per-tensor 量化、加 output_offset、做飽和
      int32_t acc = static_cast<int32_t>(cfu_op2(0, 0, 0));
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);

      // 輸出布局：[b, out_c]
      output_data[out_c + output_depth * b] = static_cast<int8_t>(acc);
    }
  }
}


// =========================
// int4 packed weights (原樣保留；先解包再呼叫上面的 FullyConnected)
// =========================
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

// =========================
// int16 x int8 -> int16（per-tensor）(原樣保留，未用 CFU)
// =========================
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

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_
