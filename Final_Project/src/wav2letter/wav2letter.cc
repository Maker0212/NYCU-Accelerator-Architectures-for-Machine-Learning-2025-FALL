#include "wav2letter.h"

#include <stdio.h>

#include <cstring>

#include "menu.h"
#include "model/wav2letter_pruned_int8.h"
#include "playground_util/console.h"
#include "test_data/test_input.h"
#include "test_data/test_output.h"
#include "tflite.h"
#include "tpu_helper.h"

static void wav2letter_pruned_init(void) {
  tflite_load_model(wav2letter_pruned_int8_tflite,
                    wav2letter_pruned_int8_tflite_len);
}

static void do_golden_tests() {
  printf("Running golden test...\n");

  printf("Setting model input...\n");
  tflite_set_input(g_test_input_data);

  printf("Running inference...\n");
  tpu_helper::ResetConvDebugInfo();
  tflite_classify();

  tpu_helper::ConvDebugInfo debug = tpu_helper::GetConvDebugInfo();
  if (debug.recorded) {
    printf(
        "[Conv debug] batch=%d y=%d x=%d ch=%d filter_sum=%d ref_filter_sum=%d "
        "hw_acc_no_bias=%d hw_acc_with_bias=%d ref_acc=%d\n",
        static_cast<int>(debug.batch), static_cast<int>(debug.out_y),
        static_cast<int>(debug.out_x), static_cast<int>(debug.out_channel),
        static_cast<int>(debug.filter_sum),
        static_cast<int>(debug.ref_filter_sum),
        static_cast<int>(debug.hw_acc_no_bias),
        static_cast<int>(debug.hw_acc_with_bias),
        static_cast<int>(debug.ref_acc));
  } else {
    // printf("[Conv debug] No instrumentation captured.\n");
  }

  int8_t* output = tflite_get_output();
  printf("Inference complete, comparing output...\n");

  bool passed = true;
  for (size_t i = 0; i < g_test_output_data_len; ++i) {
    int8_t expected_val = (int8_t)g_test_output_data[i];
    if (output[i] != expected_val) {
      printf("\n*** FAIL: Golden test failed.\n");
      printf("Mismatch at byte index %u:\n", (unsigned int)i);
      printf("  Actual:   %d\n", output[i]);
      printf("  Expected: %d\n", expected_val);

      passed = false;
      break;
    }
  }

  if (passed) {
    printf("\nOK   Golden tests passed!\n");
  }
}

static struct Menu MENU = {
    "Tests for wav2letter_pruned",
    "wav2letter",
    {
        MENU_ITEM('g', "Run golden tests", do_golden_tests),
        MENU_END,
    },
};

void wav2letter_pruned_menu() {
  wav2letter_pruned_init();
  menu_run(&MENU);
}