#include "tpu_helper.h"

namespace tpu_helper {

volatile uint32_t g_tpu_start_timeouts = 0;
volatile uint32_t g_tpu_finish_timeouts = 0;
ConvDebugInfo g_conv_debug_info = {};

void ResetTpuStats() {
  g_tpu_start_timeouts = 0;
  g_tpu_finish_timeouts = 0;
}

TpuStats GetTpuStats() {
  TpuStats stats;
  stats.start_timeouts = g_tpu_start_timeouts;
  stats.finish_timeouts = g_tpu_finish_timeouts;
  return stats;
}

void ResetConvDebugInfo() { g_conv_debug_info.recorded = 0; }

ConvDebugInfo GetConvDebugInfo() { return g_conv_debug_info; }

void RecordConvDebugInfo(const ConvDebugInfo& info) {
  g_conv_debug_info = info;
}

}  // namespace tpu_helper
