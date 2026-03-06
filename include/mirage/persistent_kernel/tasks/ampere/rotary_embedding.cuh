/* Copyright 2025 CMU
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
 */

#pragma once
#include "tasks/common/common_header.cuh"
#include <cutlass/arch/barrier.h>

namespace kernel {

template <typename T,
          typename InputSmem,
          int NUM_HEAD,
          int WINDOW_SIZE,
          int HEAD_DIM = 128>
__device__ __forceinline__ void rotary_embedding(InputSmem smem_input,
                                                 T const *cos_ptr,
                                                 T const *sin_ptr,
                                                 int token_offset = 0) {
  static_assert(HEAD_DIM % 2 == 0);
  int const tid = worker_thread_id();
#pragma unroll
  for (int win_idx = 0; win_idx < WINDOW_SIZE; ++win_idx) {

    int smem_seq_idx = token_offset + win_idx;

#pragma unroll
    for (int head_idx = 0; head_idx < NUM_HEAD; ++head_idx) {

      T const *cur_cos_ptr = cos_ptr + win_idx * HEAD_DIM;
      T const *cur_sin_ptr = sin_ptr + win_idx * HEAD_DIM;

#pragma unroll
      for (uint32_t i = tid; i < (HEAD_DIM / 2); i += NUM_THREADS) {
        int row = smem_seq_idx * NUM_HEAD + head_idx;
        int low_col = i;
        int high_col = i + HEAD_DIM / 2;

        float cos_low = static_cast<float>(cur_cos_ptr[low_col]);
        float sin_low = static_cast<float>(cur_sin_ptr[low_col]);
        float cos_high = static_cast<float>(cur_cos_ptr[high_col]);
        float sin_high = static_cast<float>(cur_sin_ptr[high_col]);

        float v_low = static_cast<float>(smem_input.at(row, low_col));
        float v_high = static_cast<float>(smem_input.at(row, high_col));

        // One thread handles a full rotary pair to avoid cross-thread
        // read-after-write hazards and barrier divergence when HEAD_DIM < 128.
        float out_low = v_low * cos_low - v_high * sin_low;
        float out_high = v_high * cos_high + v_low * sin_high;

        smem_input.at(row, low_col) = static_cast<T>(out_low);
        smem_input.at(row, high_col) = static_cast<T>(out_high);
      }
    }
  }
}

} // namespace kernel
