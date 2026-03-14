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
  static_assert(HEAD_DIM > 0);
  int const tid = worker_thread_id();
#pragma unroll
  for (int win_idx = 0; win_idx < WINDOW_SIZE; ++win_idx) {

    int smem_seq_idx = token_offset + win_idx;

#pragma unroll
    for (int head_idx = 0; head_idx < NUM_HEAD; ++head_idx) {

      T const *cur_cos_ptr = cos_ptr + win_idx * HEAD_DIM;
      T const *cur_sin_ptr = sin_ptr + win_idx * HEAD_DIM;

      if constexpr (HEAD_DIM % 2 == 0) {
        // Keep the original even-head implementation for the common 64/128-dim
        // Qwen-style path because it produces substantially better megakernel
        // codegen than the barrier-based compatibility path.
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

          float out_low = v_low * cos_low - v_high * sin_low;
          float out_high = v_high * cos_high + v_low * sin_high;

          smem_input.at(row, low_col) = static_cast<T>(out_low);
          smem_input.at(row, high_col) = static_cast<T>(out_high);
        }
      } else {
        // Compatibility fallback for odd head dims.
        constexpr int kIters = (HEAD_DIM + NUM_THREADS - 1) / NUM_THREADS;
#pragma unroll
        for (int it = 0; it < kIters; ++it) {
          int i = it * NUM_THREADS + tid;
          bool active = (i < HEAD_DIM);
          int row = smem_seq_idx * NUM_HEAD + head_idx;
          int col = active ? i : 0;

          float cos = 0.0f;
          float sin = 0.0f;
          if (active) {
            cos = static_cast<float>(cur_cos_ptr[col]);
            sin = static_cast<float>(cur_sin_ptr[col]);
          }

          wg_sync<WORKER_NUM_THREADS>(0);

          float v_rot = 0.0f;
          if (active) {
            int half = HEAD_DIM / 2;
            if (i < half) {
              float v1 = static_cast<float>(smem_input.at(row, col));
              float v2 = static_cast<float>(smem_input.at(row, col + half));
              v_rot = v1 * cos - v2 * sin;
            } else if (i < 2 * half) {
              float v1 = static_cast<float>(smem_input.at(row, col));
              float v2 = static_cast<float>(smem_input.at(row, col - half));
              v_rot = v1 * cos + v2 * sin;
            } else {
              // Keep tail element unchanged for odd HEAD_DIM.
              v_rot = static_cast<float>(smem_input.at(row, col));
            }
          }

          wg_sync<WORKER_NUM_THREADS>(0);
          if (active) {
            smem_input.at(row, col) = static_cast<T>(v_rot);
          }
        }
      }
    }
  }
}

} // namespace kernel
