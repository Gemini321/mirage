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

#include "profiler.h"
#include "tasks/common/copy_sm80.cuh"
#ifdef MPK_ENABLE_TMA
#include "tma.cuh"
#endif
#include "mpk_atoms.cuh"
#include "runtime_header.h"
#ifdef USE_NVSHMEM
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#endif
#include <algorithm>
#include <thread>
#include <unistd.h>
#include <vector>
#include <cuda_profiler_api.h>

#if defined(MIRAGE_GRACE_HOPPER)
#include "tasks/hopper/task_header.cuh"
#elif defined(MIRAGE_GRACE_BLACKWELL)
#include "tasks/blackwell/task_header.cuh"
#else
#include "tasks/ampere/task_header.cuh"
#endif

using bfloat16 = type::bfloat16_t;
using namespace mirage::runtime;
using namespace kernel;
// Configurations for the MPK runtime
// #define MPK_MAX_NUM_BATCHED_REQUESTS 16
// #define MPK_MAX_NUM_BATCHED_TOKENS 64
// #define MPK_MAX_NUM_PAGES 1024
// #define MPK_PAGE_SIZE 64

#if defined(MIRAGE_GRACE_HOPPER)
#define WORKER_NUM_THREADS 256
#define SINGLE_KERNEL_NUM_THREADS 256
#elif defined(MIRAGE_GRACE_BLACKWELL)
#define WORKER_NUM_THREADS 256
#define SINGLE_KERNEL_NUM_THREADS 256
#else
#define WORKER_NUM_THREADS 128
#define SINGLE_KERNEL_NUM_THREADS 128
#endif
#define INIT_NUM_THREADS 128

#ifndef MIRAGE_START_REG_TARGET
#define MIRAGE_START_REG_TARGET 128
#endif

#ifndef MIRAGE_IDLE_REG_TARGET
// Idle/control-path register target for worker warpgroup(s) when not executing
// a task. Tune via nvcc flag: -DMIRAGE_IDLE_REG_TARGET=<N>
#define MIRAGE_IDLE_REG_TARGET 112
#endif

#ifndef MIRAGE_FETCH_REG_TARGET
// Register target used for the "fetch task" control-path (loader group).
// Tune via nvcc flag: -DMIRAGE_FETCH_REG_TARGET=<N>
#define MIRAGE_FETCH_REG_TARGET 112
#endif

#ifndef MIRAGE_TRIGGER_REG_TARGET
// Register target used for the "trigger event" control-path (post-task).
// Tune via nvcc flag: -DMIRAGE_TRIGGER_REG_TARGET=<N>
#define MIRAGE_TRIGGER_REG_TARGET 112
#endif

#ifndef MIRAGE_PARK_REG_TARGET
// "Parking" register target used when a warp-group needs to temporarily shrink
// to free register budget for the other warp-group to admit a high-reg task.
//
// This should be low enough that:
//   MIRAGE_PARK_REG_TARGET <= MIRAGE_TOTAL_REG_BUDGET - max_task_reg_target
// otherwise admission can never succeed and both groups may deadlock waiting.
//
// Tune via nvcc flag: -DMIRAGE_PARK_REG_TARGET=<N>
#define MIRAGE_PARK_REG_TARGET 64
#endif

// Total register budget across both worker groups (regs/thread).
// Used as a simple admission control heuristic when both groups can execute
// concurrently on the same CTA/SM.
#ifndef MIRAGE_TOTAL_REG_BUDGET
#define MIRAGE_TOTAL_REG_BUDGET 256
#endif

// Per-task resource "lookup" should be compile-time (macros), not per-task
// device tables. These defaults can be overridden at compile time, or tasks can
// implement their own policy when calling into the runtime.
#ifndef MIRAGE_TASK_EXEC_REG_TARGET
#define MIRAGE_TASK_EXEC_REG_TARGET MIRAGE_START_REG_TARGET
#endif

#ifndef MIRAGE_TASK_EXEC_SMEM_BYTES
#define MIRAGE_TASK_EXEC_SMEM_BYTES 0u
#endif

#ifndef MIRAGE_TERMINATE_SCHED_MAX_SPINS
#define MIRAGE_TERMINATE_SCHED_MAX_SPINS (1u << 20)
#endif

#ifndef MIRAGE_HOST_DEBUG
#define MIRAGE_HOST_DEBUG 1
#endif

// Host-side synchronization strategy after launching split worker/scheduler
// kernels. `cudaDeviceSynchronize()` is the strictest, but in some environments
// it can appear to hang due to external factors (e.g., stdout/stderr piping).
// Set `-DMIRAGE_HOST_USE_DEVICE_SYNC=0` to use stream-level sync instead.
#ifndef MIRAGE_HOST_USE_DEVICE_SYNC
#define MIRAGE_HOST_USE_DEVICE_SYNC 1
#endif

#ifndef MIRAGE_PK_BUILD_STAMP
#define MIRAGE_PK_BUILD_STAMP __DATE__ " " __TIME__
#endif

#ifndef MIRAGE_ADMISSION_DEBUG
// Set `-DMIRAGE_ADMISSION_DEBUG=1` to enable device-side debug prints for
// worker scheduling/admission paths.
#define MIRAGE_ADMISSION_DEBUG 0
#endif

#ifndef MIRAGE_SCHED_LOG
// Set `-DMIRAGE_SCHED_LOG=1` to enable scheduler assignment logging.
#define MIRAGE_SCHED_LOG 0
#endif

#ifndef MIRAGE_WORKER_LOG
// Set `-DMIRAGE_WORKER_LOG=1` to enable worker task execution logging.
#define MIRAGE_WORKER_LOG 0
#endif

#if MIRAGE_ADMISSION_DEBUG
#define MIRAGE_ADMIT_DPRINTF(lane, fmt, ...)                                   \
  do {                                                                         \
    if ((lane) == 0) {                                                         \
      printf(fmt, ##__VA_ARGS__);                                               \
    }                                                                          \
  } while (0)
#else
#define MIRAGE_ADMIT_DPRINTF(lane, fmt, ...)                                   \
  do {                                                                         \
  } while (0)
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr,                                                          \
              "CUDA error at %s:%d: %s\n",                                     \
              __FILE__,                                                        \
              __LINE__,                                                        \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#endif

// #define MPK_ENABLE_VERBOSE
__device__ __forceinline__ void
    _execute_task(TaskDesc const *task_desc,
                  RuntimeConfig const &runtime_config,
                  void *exec_ctx,
                  int group_id,
                  char *smem_base,
                  uint32_t smem_capacity);

// Keep task execution in a separate noinline wrapper so that register liveness
// from the inlined task body does not cross the setmaxnreg transition points.
// This improves the chance that ptxas can honor wg_decrease_regs<> without
// emitting (C7507) "setmaxnreg ignored".
__device__ __forceinline__ void
    __mirage_execute_task_noinline(TaskDesc const *task_desc,
                                   RuntimeConfig const &runtime_config,
                                   void *exec_ctx,
                                   int group_id,
                                   char *smem_base,
                                   uint32_t smem_capacity) {
  _execute_task(
      task_desc, runtime_config, exec_ctx, group_id, smem_base, smem_capacity);
}

// Per-task resource request (registers + dynamic shared memory).
struct __mirage_task_resource {
  uint32_t reg_target;
  uint32_t smem_bytes;
};

struct __mirage_smem_block {
  int owner;       // -1: free; otherwise group_id
  uint32_t offset; // from aligned base
  uint32_t size;   // bytes
};

// Execution context shared by multiple warp-groups within a CTA.
struct __mirage_exec_ctx {
  // Per-group state
  uint32_t group_cur_regs[2];
  uint32_t group_smem_offset[2];
  int group_smem_handle[2];

  // Cached runtime reg targets (see __mirage_reg_slot).
  uint32_t reg_targets[4];

  // Cross-group arbitration
  int reg_lock;
  unsigned reg_epoch;

  // Cooperative reg-budget "parking" protocol.
  // If park_owner != -1, the other group should temporarily drop to a minimal
  // reg target to free budget for park_owner's high-reg admission.
  int park_owner;       // -1: none; otherwise requesting group_id

  // Cooperative cancellation flag (CTA-wide). When set, all warp-groups should
  // stop waiting/spinning and converge to a clean exit path to avoid deadlock.
  int terminate_all;

  // Shared spill slots for slow-path control flow (avoid local-memory spills).
  // Indexed by group_id (0/1).
  // uint32_t spill_target_regs[2];
  // uint32_t spill_req_smem[2];
  // uint32_t spill_smem_capacity[2];
  // uintptr_t spill_smem_base[2];

  // Shared SMEM allocator
  __mirage_smem_block smem_blocks[2];
  int smem_lock;
};

__device__ __forceinline__ void __mirage_exec_ctx_init(__mirage_exec_ctx *ctx,
                                                       RuntimeConfig const &cfg) {
  ctx->group_cur_regs[0] = MIRAGE_START_REG_TARGET;
  ctx->group_cur_regs[1] = MIRAGE_START_REG_TARGET;
  ctx->group_smem_offset[0] = 0;
  ctx->group_smem_offset[1] = 0;
  ctx->group_smem_handle[0] = -1;
  ctx->group_smem_handle[1] = -1;
  (void)cfg;
  // Use compile-time reg targets for all control paths. Override via nvcc flags:
  //   -DMIRAGE_IDLE_REG_TARGET=...
  //   -DMIRAGE_FETCH_REG_TARGET=...
  //   -DMIRAGE_TRIGGER_REG_TARGET=...
  //   -DMIRAGE_PARK_REG_TARGET=...
  ctx->reg_lock = 0;
  ctx->reg_epoch = 0;
  ctx->park_owner = -1;
  ctx->terminate_all = 0;
  // ctx->spill_target_regs[0] = 0;
  // ctx->spill_target_regs[1] = 0;
  // ctx->spill_req_smem[0] = 0;
  // ctx->spill_req_smem[1] = 0;
  // ctx->spill_smem_capacity[0] = 0;
  // ctx->spill_smem_capacity[1] = 0;
  // ctx->spill_smem_base[0] = 0;
  // ctx->spill_smem_base[1] = 0;
  ctx->smem_blocks[0].owner = -1;
  ctx->smem_blocks[0].offset = 0;
  ctx->smem_blocks[0].size = 0;
  ctx->smem_blocks[1].owner = -1;
  ctx->smem_blocks[1].offset = 0;
  ctx->smem_blocks[1].size = 0;
  ctx->smem_lock = 0;
}

__device__ __forceinline__ bool
    __mirage_try_alloc_smem(__mirage_smem_block blocks[2],
                            int *lock,
                            uint32_t smem_capacity,
                            uint32_t req_bytes,
                            int group_id,
                            uint32_t *out_offset,
                            int *out_handle);

__device__ __forceinline__ void __mirage_free_smem(
    __mirage_smem_block blocks[2], int *lock, int handle, int group_id);

__device__ __forceinline__ constexpr uint32_t
    __mirage_align_up_u32(uint32_t x, uint32_t a) {
  return (x + a - 1) / a * a;
}

__device__ __forceinline__ uint32_t __mirage_align_down_u32(uint32_t x,
                                                            uint32_t a) {
  return (x / a) * a;
}

// Smem-only admission: allocate/free dynamic shared memory per group without any
// register admission/parking logic. This is intended for simplified multigroup
// worker scheduling aligned with execute_worker().
__device__ __forceinline__ char *__mirage_task_enter_smem_only(
    __mirage_exec_ctx *ctx,
    int group_id,
    uint32_t req_smem,
    char *smem_base,
    uint32_t smem_capacity) {
  uint32_t off = 0;
  int handle = -1;
  if (threadIdx.x % WORKER_NUM_THREADS == 0) {
    while (!__mirage_try_alloc_smem(ctx->smem_blocks,
                                    &ctx->smem_lock,
                                    smem_capacity,
                                    req_smem,
                                    group_id,
                                    &off,
                                    &handle)) {
      __nanosleep(20);
    }
    ctx->group_smem_offset[group_id] = off;
    ctx->group_smem_handle[group_id] = handle;
  }
  wg_sync<WORKER_NUM_THREADS>(3);
  // return (req_smem == 0) ? smem_base : (smem_base + off);
  return smem_base;
}

__device__ __forceinline__ void __mirage_task_exit_smem_only(
    __mirage_exec_ctx *ctx, int group_id) {
  wg_sync<WORKER_NUM_THREADS>(3);
  if (threadIdx.x % WORKER_NUM_THREADS == 0) {
    int const handle = ctx->group_smem_handle[group_id];
    if (handle != -1) {
      __mirage_free_smem(ctx->smem_blocks, &ctx->smem_lock, handle, group_id);
      ctx->group_smem_handle[group_id] = -1;
      ctx->group_smem_offset[group_id] = 0;
    }
  }
}

__device__ __forceinline__ bool
    __mirage_try_alloc_smem(__mirage_smem_block blocks[2],
                            int *lock,
                            uint32_t smem_capacity,
                            uint32_t req_bytes,
                            int group_id,
                            uint32_t *out_offset,
                            int *out_handle) {
  if (req_bytes == 0) {
    *out_offset = 0;
    *out_handle = -1;
    return true;
  }
  req_bytes = __mirage_align_up_u32(req_bytes, 1024);

  while (atomicCAS(lock, 0, 1) != 0) {
    __nanosleep(10);
  }

  // Two-block allocator for at most two concurrent groups:
  // - blocks[0] grows from the left (offset 0 upward)
  // - blocks[1] grows from the right (offset smem_capacity downward)
  // All allocations are 1KB-aligned.
  uint32_t left_end = 0;
  if (blocks[0].owner != -1) {
    left_end = __mirage_align_up_u32(blocks[0].offset + blocks[0].size, 1024);
  }
  uint32_t right_start = smem_capacity;
  if (blocks[1].owner != -1) {
    right_start = blocks[1].offset;
  }

  // Maintain a valid ordering: [0, left_end) ... [right_start, smem_capacity)
  // If the two regions overlap (shouldn't happen), fail conservatively.
  if (left_end > right_start) {
    *out_offset = 0;
    *out_handle = -1;
    atomicExch(lock, 0);
    return false;
  }

  bool ok = false;
  int const prefer_left = ((group_id & 1) == 0) ? 1 : 0;
  for (int attempt = 0; attempt < 2 && !ok; ++attempt) {
    int do_left = prefer_left;
    if (attempt == 1) {
      do_left = 1 - do_left;
    }
    if (do_left) {
      // Left allocation at offset 0.
      if (blocks[0].owner == -1 && req_bytes <= right_start) {
        blocks[0].owner = group_id;
        blocks[0].offset = 0;
        blocks[0].size = req_bytes;
        *out_offset = 0;
        *out_handle = 0;
        ok = true;
      }
    } else {
      // Right allocation ending at right_start.
      if (blocks[1].owner == -1 && right_start >= req_bytes) {
        uint32_t candidate_offset =
            __mirage_align_down_u32(right_start - req_bytes, 1024);
        if (candidate_offset >= left_end) {
          blocks[1].owner = group_id;
          blocks[1].offset = candidate_offset;
          blocks[1].size = req_bytes;
          *out_offset = candidate_offset;
          *out_handle = 1;
          ok = true;
        }
      }
    }
  }
  if (!ok) {
    *out_offset = 0;
    *out_handle = -1;
    atomicExch(lock, 0);
    return false;
  }

  atomicExch(lock, 0);
  return true;
}

__device__ __forceinline__ void __mirage_free_smem(
    __mirage_smem_block blocks[2], int *lock, int handle, int group_id) {
  if (handle < 0) {
    return;
  }
  while (atomicCAS(lock, 0, 1) != 0) {
    __nanosleep(10);
  }
  if (blocks[handle].owner == group_id) {
    blocks[handle].owner = -1;
    blocks[handle].offset = 0;
    blocks[handle].size = 0;
  }
  atomicExch(lock, 0);
}

__device__ __forceinline__ bool is_termination_event(EventId event_id) {
  return (event_id == 0);
}

__device__ __forceinline__ bool is_nvshmem_event(EventId event_id) {
  return (event_id & EVENT_NVSHMEM_TAG) > 0;
}

__device__ __forceinline__ size_t get_event_gpu_id(EventId event_id) {
  return ((event_id >> 32) & 0xffff);
}

__device__ __forceinline__ size_t get_event_position_index(EventId event_id) {
  return (event_id & 0xffffffff);
}

__device__ __forceinline__ size_t get_task_iteration_num(TaskId task_id) {
  return (task_id >> 32);
}

__device__ __forceinline__ size_t get_task_position_index(TaskId task_id) {
  return (task_id & 0xffffffff);
}

__device__ __forceinline__ TaskId compute_task_id(size_t iteration_num,
                                                  size_t position_index) {
  return ((iteration_num << 32) | position_index);
}

__global__ void init_kernel(RuntimeConfig config) {
  assert(gridDim.x == 1);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Only a single thread that initializes everything
  if (threadIdx.x == 0) {
    // initialize metadata
#if defined(MODE_OFFLINE) || defined(MODE_ONLINE)
    for (int i = 0; i < config.total_num_requests; i++) {
      config.step[i] = 0;
    }
    *config.next_request_id = 0;
    for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
      config.request_ids[i] = -1;
    }
    for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS + 1; i++) {
      config.qo_indptr_buffer[i] = 0;
      config.paged_kv_indptr_buffer[i] = 0;
    }
    // Page manager
    *config.page_queue_head = 0;
    *config.page_queue_tail = MPK_MAX_NUM_PAGES;
    for (int i = 0; i < MPK_MAX_NUM_PAGES; i++) {
      config.page_queue[i] = i;
    }
#else
    // One-pass mode: request-batching metadata does not exist; leave meta
    // tensors initialization to the caller.
    (void)config;
#endif
  }
}

__global__ void prepare_kernel(RuntimeConfig config,
                               int end_of_task_graph_event_pos) {
  // Initialize worker queue last task id
  // Each worker now maintains a local and a remote worker queue
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < 2 * config.num_workers;
       i += blockDim.x * gridDim.x) {
    config.worker_queue_last_ready_task_id[i] = 0;
  }
  // Initialize scheduler queue last event id
  // We maintain one extra scheduler queue for the global scheduler
  int num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_schedulers + 1;
       i += blockDim.x * gridDim.x) {
    config.sched_queue_last_ready_event_id[i] = 0;
    config.sched_queue_next_free_event_id[i] = 0;
  }
  // Initialize all event counters
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < config.num_events;
       i += blockDim.x * gridDim.x) {
    config.all_event_counters[i] = 0;
  }
  // Send event to scheduler[0]
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    assert(config.all_events[end_of_task_graph_event_pos].event_type ==
           EVENT_END_OF_TASK_GRAPH);
    config.sched_queue_next_free_event_id[0] = 1;
    config.sched_queues[0][0] = end_of_task_graph_event_pos;
    config.sched_queue_last_ready_event_id[0] = 1;
  }
}

#ifdef MODE_OFFLINE
// TODO: parallelize this processing
__device__ __forceinline__ bool
    prepare_next_batch(RuntimeConfig const &config) {
  __shared__ int smem_kv_indices[MPK_MAX_NUM_PAGES];
  int page_queue_head = *config.page_queue_head;
  int page_queue_tail = *config.page_queue_tail;
  // Step 1: finalize previous batch
  for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    int16_t request_id = config.request_ids[i];
    if (request_id != -1) {
      // Step 1.1: move output_tokens to tokens
      int step = config.step[request_id];
      int qo_indptr = config.qo_indptr_buffer[i];
      int num_tokens = config.qo_indptr_buffer[i + 1] - qo_indptr;
      int prompt_len = config.prompt_length[request_id];
      for (int j = 0; j < num_tokens; j++) {
        if (step + j + 1 >= prompt_len &&
            step + j + 1 < config.max_seq_length) {
          config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step + j + 1] =
              config.output_tokens[qo_indptr + j];
        }
      }
      config.step[request_id] = step + num_tokens;
      if ((step + num_tokens + 1 >= config.max_seq_length) ||
          ((config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step +
                          num_tokens] == config.eos_token_id) &&
           (step + num_tokens >= prompt_len))) {
        // Request is done
        config.request_ids[i] = -1;
        // Free pages
        int kv_indptr = config.paged_kv_indptr_buffer[i];
        int num_pages = config.paged_kv_indptr_buffer[i + 1] - kv_indptr;
        for (int j = 0; j < num_pages; j++) {
          config.page_queue[page_queue_tail % MPK_MAX_NUM_PAGES] =
              config.paged_kv_indices_buffer[kv_indptr + j];
          page_queue_tail++;
        }
      }
    }
  }

  // Step 2: copy kv_indices to shared mem
  int num_pages = config.paged_kv_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];
  for (int i = 0; i < num_pages; i++) {
    smem_kv_indices[i] = config.paged_kv_indices_buffer[i];
  }

  // Step 3: prepare next batch
  int num_reqs = 0, num_tokens = 0;
  num_pages = 0;
  for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    int16_t request_id = config.request_ids[i];
    if (request_id != -1) {
      int kv_indptr = config.paged_kv_indptr_buffer[i];
      int num_old_pages = config.paged_kv_indptr_buffer[i + 1] - kv_indptr;
      config.request_ids[num_reqs] = request_id;
      config.qo_indptr_buffer[num_reqs] = num_tokens;
      config.paged_kv_indptr_buffer[num_reqs] = num_pages;
      int step = config.step[request_id];
      int num_new_tokens = config.prompt_length[request_id] - step;
      if (num_new_tokens > 0) {
        // Prefill requests
        num_new_tokens =
            min(num_new_tokens, MPK_MAX_NUM_BATCHED_TOKENS - num_tokens);
      } else {
        // Decode requests
        num_new_tokens = min(1, MPK_MAX_NUM_BATCHED_TOKENS - num_tokens);
      }
      // Move tokens to input_tokens
      for (int j = 0; j < num_new_tokens; j++) {
        config.input_tokens[num_tokens + j] =
            config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step + j];
      }
      // Prepare page indptrs
      int num_new_pages =
          (step + num_new_tokens + MPK_PAGE_SIZE - 1) / MPK_PAGE_SIZE;
      config.paged_kv_last_page_len_buffer[num_reqs] =
          (step + num_new_tokens) % MPK_PAGE_SIZE;
      for (int j = 0; j < num_old_pages; j++) {
        config.paged_kv_indices_buffer[num_pages + j] =
            smem_kv_indices[kv_indptr + j];
      }
      for (int j = num_old_pages; j < num_new_pages; j++) {
        config.paged_kv_indices_buffer[num_pages + j] =
            config.page_queue[page_queue_head % MPK_MAX_NUM_PAGES];
        page_queue_head++;
      }
      num_pages += num_new_pages;
      num_tokens += num_new_tokens;
      num_reqs++;
    }
  }

  // Add new prefill requests until we reach capacity
  while (num_reqs < MPK_MAX_NUM_BATCHED_REQUESTS &&
         num_tokens < MPK_MAX_NUM_BATCHED_TOKENS) {
    int next_request_id = *config.next_request_id;
    if (next_request_id >= config.total_num_requests) {
      break;
    }
    config.request_ids[num_reqs] = next_request_id;
    config.qo_indptr_buffer[num_reqs] = num_tokens;
    config.paged_kv_indptr_buffer[num_reqs] = num_pages;
    // Prefill request
    int num_new_tokens = min(config.prompt_length[next_request_id],
                             MPK_MAX_NUM_BATCHED_TOKENS - num_tokens);
    // Move tokens to input tokens
    for (int j = 0; j < num_new_tokens; j++) {
      config.input_tokens[num_tokens + j] =
          config.tokens[next_request_id * MPK_MAX_SEQ_LENGTH + j];
    }
    int num_new_pages = (num_new_tokens + MPK_PAGE_SIZE - 1) / MPK_PAGE_SIZE;
    config.paged_kv_last_page_len_buffer[num_reqs] =
        num_new_tokens % MPK_PAGE_SIZE;
    for (int j = 0; j < num_new_pages; j++) {
      config.paged_kv_indices_buffer[num_pages + j] =
          config.page_queue[page_queue_head % MPK_MAX_NUM_PAGES];
      page_queue_head++;
    }
    num_tokens += num_new_tokens;
    num_pages += num_new_pages;
    num_reqs++;
    *config.next_request_id = next_request_id + 1;
  }

  // Step 4: Update all unused requests slots
  for (int i = num_reqs; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    config.request_ids[i] = -1;
  }
  for (int i = num_reqs; i <= MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    config.qo_indptr_buffer[i] = num_tokens;
    config.paged_kv_indptr_buffer[i] = num_pages;
  }

  // Step 5: update page head tail
  *config.page_queue_head = page_queue_head;
  *config.page_queue_tail = page_queue_tail;

  // printf("Next batch: steps[%d %d %d %d] num_active_tokens(%d)\n",
  //        config.step[0],
  //        config.step[1],
  //        config.step[2],
  //        config.step[3],
  //        config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);

  if (num_tokens == 0) {
    return false;
  } else {
    return true;
  }
}
#endif

#ifdef MODE_ONLINE
__device__ __forceinline__ bool
    prepare_next_batch(RuntimeConfig const &config) {
  int step = config.step[0];
#ifdef MPK_ENABLE_VERBOSE
  printf("step: %d, new_token_num(%p): %d, new_token_ids:\n",
         step,
         config.new_token_nums,
         config.new_token_nums[0]);
  for (int i = 0; i < config.new_token_nums[0]; i++) {
    printf("%lld ", config.tokens[step + 1 + i]);
  }
  printf("\n");
#endif
  config.step[0] = step + config.new_token_nums[0];

  if ((step + 2 >= config.max_seq_length) ||
      (config.tokens[step + 1] == config.eos_token_id)) {
    return false;
  } else {
    return true;
  }
}
#endif

#ifdef MODE_ONEPASS
// One-pass mode does not use request batching; the scheduler controls
// "run exactly once" directly (see execute_scheduler()).
__device__ __forceinline__ bool prepare_next_batch(RuntimeConfig const &) {
  return false;
}
#endif

__device__ __forceinline__ int get_rand_sched_id(size_t event_index,
                                                 int worker_id,
                                                 int num_workers,
                                                 int num_schedulers) {
  // const size_t seed = 0xac4c1b51;
  // size_t x = event_index * seed;
  // x ^= x >> 17;
  // x *= worker_id;
  //  x *= 0xed5ad4bb;
  // x ^= x >> 11;
  size_t x = worker_id;
  return x / ((num_workers + num_schedulers - 1) / num_schedulers);
}

__device__ __forceinline__ void
    get_first_last_ids(unsigned long long int num_elements,
                       unsigned long long int num_workers,
                       unsigned long long int my_id,
                       unsigned long long int *my_first_element,
                       unsigned long long int *my_last_element) {
  unsigned long long int num_elements_per_worker = num_elements / num_workers;
  unsigned long long int reminder = num_elements % num_workers;
  if (my_id < reminder) {
    *my_first_element = (num_elements_per_worker + 1) * my_id;
    *my_last_element = *my_first_element + num_elements_per_worker + 1;
  } else {
    *my_first_element = num_elements_per_worker * my_id + reminder;
    *my_last_element = *my_first_element + num_elements_per_worker;
  }
}

__device__ __forceinline__ void terminate_schedulers(RuntimeConfig config) {
  // Event ID 0 is the termination event
  int num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  // Also terminate the global broadcast scheduler queue at index num_schedulers.
  for (int i = 0; i < num_schedulers + 1; i++) {
    // size_t last_event_id =
    //     atomicAdd(&config.sched_queue_next_free_event_id[i], 1);
    size_t last_event_id =
        atom_add_release_gpu_u64(&config.sched_queue_next_free_event_id[i], 1);
    st_relaxed_gpu_u64(
        &config.sched_queues[i][last_event_id % config.per_sched_queue_len], 0);
    // Use st.relaxed to make sure sched_queue updates are visible to scheduler
    // CTAs before incrementing its last_ready_event_id
    size_t old;
    unsigned spins = 0;
    while (true) {
      // old = atomicCAS(&config.sched_queue_last_ready_event_id[i],
      //                 last_event_id,
      //                 last_event_id + 1);
      old = atom_cas_release_gpu_u64(&config.sched_queue_last_ready_event_id[i],
                                     last_event_id,
                                     last_event_id + 1);
      if (old == last_event_id) {
        break;
      }
      if (++spins >= MIRAGE_TERMINATE_SCHED_MAX_SPINS) {
        // Fallback: fill any unpublished "holes" with termination events and
        // force-advance last_ready so scheduler CTAs can exit.
        unsigned long long cur_ready =
            ld_acquire_gpu_u64(&config.sched_queue_last_ready_event_id[i]);
        for (unsigned long long p = cur_ready; p <= (unsigned long long)last_event_id;
             ++p) {
          st_relaxed_gpu_u64(
              &config.sched_queues[i][p % config.per_sched_queue_len], 0);
        }
        // Force last_ready >= last_event_id+1 (monotonic).
        unsigned long long target = (unsigned long long)last_event_id + 1ull;
        while (cur_ready < target) {
          unsigned long long prev = atom_cas_release_gpu_u64(
              &config.sched_queue_last_ready_event_id[i], cur_ready, target);
          if (prev == cur_ready) {
            break;
          }
          cur_ready = prev;
        }
        break;
      }
    }
  }
}

__device__ __forceinline__ void worker_checker(RuntimeConfig config) {
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  // int num_schedulers =
  //    config.num_local_schedulers + config.num_remote_schedulers;

  assert(gridDim.x == config.num_workers);
  assert(config.num_workers <= MAX_NUM_WORKERS);
  // We will reinterpret TaskDesc as an array of integers to
  // collectively load it from device to shared memory
  static_assert(sizeof(TaskDesc) % sizeof(int) == 0);
}

__device__ __forceinline__ void scheduler_checker(RuntimeConfig config) {
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  // int num_schedulers =
  //    config.num_local_schedulers + config.num_remote_schedulers;

  assert(config.num_workers <= MAX_NUM_WORKERS);
}

__device__ __forceinline__ void persistent_checker(RuntimeConfig config) {
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  int const num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  int const num_schedulers_per_sm = std::min((int)blockDim.x / 32, 4);
  assert(num_schedulers % num_schedulers_per_sm == 0);
  assert(gridDim.x ==
         config.num_workers + num_schedulers / num_schedulers_per_sm);
  assert(config.num_workers <= MAX_NUM_WORKERS);
  // We will reinterpret TaskDesc as an array of integers to
  // collectively load it from device to shared memory
  static_assert(sizeof(TaskDesc) % sizeof(int) == 0);
  // assert(blockDim.x >= 128);
}

__device__ __forceinline__ void execute_worker(RuntimeConfig config) {
  // Make sure overall smem usage here do not exceed 3KB
  // last_task_pos: 2 * 8 = 16 B
  // next_task_pos: 2 * 8 = 16 B
  // worker_queue_ids: 2 * 4 = 8 B
  // worker_queues: 2 * 8 = 16 B
  // remaining: 3016 B

  constexpr int TASK_DESCS_BUFFER_LENGTH = std::min(
      (mirage::runtime::WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE - 56) /
          (int)(sizeof(TaskDesc) + sizeof(TaskId)),
      16);
  __shared__ TaskDesc task_descs[TASK_DESCS_BUFFER_LENGTH];
  __shared__ TaskId task_ids[TASK_DESCS_BUFFER_LENGTH];
  __shared__ TaskId *worker_queues[2];
  __shared__ int worker_queue_ids[2];
  __shared__ size_t next_task_pos[2];
  __shared__ size_t last_task_pos[2];
  __shared__ __mirage_exec_ctx __mirage_exec_ctx_single;
  extern __shared__ char __mirage_smem_base[];
  constexpr uint32_t __mirage_smem_capacity = 220 * 1024;

#ifdef MPK_ENABLE_PROFILING
  PROFILER_CLOSURE_PARAMS_DECL;
  PROFILER_INIT(static_cast<uint64_t *>(config.profiler_buffer),
                0,
                1,
                (threadIdx.x % WORKER_NUM_THREADS == 0));

#endif
  int const worker_id = blockIdx.x;
  worker_queues[0] = config.worker_queues[worker_id];
  worker_queue_ids[0] = worker_id;
  int num_worker_queues = 1;
  if (config.num_gpus > 1) {
    worker_queues[num_worker_queues] =
        config.worker_queues[worker_id + config.num_workers];
    worker_queue_ids[num_worker_queues] = worker_id + config.num_workers;
    num_worker_queues++;
  }

  if (threadIdx.x == 0) {
    for (int i = 0; i < 2; i++) {
      next_task_pos[i] = 0;
    }
    for (int i = 0; i < 2; i++) {
      last_task_pos[i] = 0;
    }
    // num_loaded_tasks = 0;
  }

  int queue_pos = 0, queue_len = 0;
#ifdef MPK_ENABLE_PROFILING
  size_t task_counter = 0;
#endif
  while (true) {
    // fetch next task from a task queue if task_descs is empty
    if (queue_pos == queue_len) {
      int queue_idx = 0;
      if (threadIdx.x == 0) {
        while (next_task_pos[queue_idx] == last_task_pos[queue_idx]) {
          last_task_pos[queue_idx] =
              ld_acquire_gpu_u64(&config.worker_queue_last_ready_task_id
                                      [worker_queue_ids[queue_idx]]);
          if (next_task_pos[queue_idx] < last_task_pos[queue_idx]) {
            break;
          } else {
            queue_idx =
                (queue_idx == num_worker_queues - 1) ? 0 : queue_idx + 1;
          }
          // nanosleep to avoid overwhelming I/O
          __nanosleep(10);
        }
        assert(next_task_pos[queue_idx] + config.per_worker_queue_len >
               last_task_pos[queue_idx]);
      }
      __syncthreads();
      int num_loaded_tasks =
          min((int)(last_task_pos[queue_idx] - next_task_pos[queue_idx]),
              TASK_DESCS_BUFFER_LENGTH);
      // Load task ids
      if (threadIdx.x < num_loaded_tasks) {
        task_ids[threadIdx.x] = ld_relaxed_gpu_u64(
            &worker_queues[queue_idx][(next_task_pos[queue_idx] + threadIdx.x) %
                                      config.per_worker_queue_len]);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
#ifdef MPK_ENABLE_VERBOSE
        for (int i = 0; i < num_loaded_tasks; i++) {
          printf(
              "[%d][FTCH] worker_id(%d) queue_idx(%d) next_task_pos(%llu, "
              "%llu) last_task_pos(%llu, %llu) "
              "task_id(%llu) task_type(%d) event_id(%llx) \n",
              config.my_gpu_id,
              worker_id,
              queue_idx,
              next_task_pos[0],
              next_task_pos[1],
              last_task_pos[0],
              last_task_pos[1],
              get_task_position_index(task_ids[i]),
              config.all_tasks[get_task_position_index(task_ids[i])].task_type,
              config.all_tasks[get_task_position_index(task_ids[i])]
                  .trigger_event);
        }
#endif
        next_task_pos[queue_idx] += num_loaded_tasks;
      }
      // Load task descs
      static_assert(sizeof(TaskDesc) % 16 == 0);
      constexpr int TASK_SIZE = sizeof(TaskDesc) / 16; // 128b copy-async
      for (int i = threadIdx.x; i < num_loaded_tasks * TASK_SIZE;
           i += WORKER_NUM_THREADS) {
        int task_idx = i / TASK_SIZE;
        int offset = i % TASK_SIZE;
        load_smem(reinterpret_cast<char *>(task_descs) + i * 16,
                  reinterpret_cast<char *>(
                      config.all_tasks +
                      get_task_position_index(task_ids[task_idx])) +
                      offset * 16);
      }
      kernel::cp_async_fence();
      kernel::cp_async_wait<0>();
      __syncthreads();
      queue_pos = 0;
      queue_len = num_loaded_tasks;
    }
    TaskDesc *task_desc = task_descs + queue_pos;
    // Make sure task is ready before start execution
    if (threadIdx.x == 0) {
      if (task_desc->dependent_event != EVENT_INVALID_ID) {
        // Wait until the event has been triggered enough times
        EventId event_id = task_desc->dependent_event;
        assert(get_event_gpu_id(event_id) == config.my_gpu_id);
        size_t event_index = get_event_position_index(event_id);
        EventCounter needed_counts =
            static_cast<EventCounter>(
                config.all_event_num_triggers[event_index]) *
            get_task_iteration_num(task_ids[queue_pos]);
        EventCounter actual_counts = 0;
        if (is_nvshmem_event(event_id)) {
#ifdef USE_NVSHMEM
          nvshmem_signal_wait_until(
              reinterpret_cast<uint64_t *>(
                  &config.all_event_counters[event_index]),
              NVSHMEM_CMP_EQ,
              needed_counts);
#endif
        } else {
          while (actual_counts < needed_counts) {
            actual_counts =
                ld_acquire_sys_u64(&config.all_event_counters[event_index]);
            __nanosleep(10);
          }
        }
      }
    }
    __syncthreads();

#ifdef MPK_ENABLE_PROFILING
    if (task_desc->task_type != TASK_TERMINATE) {
      PROFILER_EVENT_START(task_desc->task_type, task_counter);
    }
#endif

    // Successfully fetched a new task
    if (task_desc->task_type == TASK_TERMINATE) {
#ifdef MPK_ENABLE_PROFILING
      if (threadIdx.x == 0) {
        uint32_t const ev_no = static_cast<uint32_t>(task_counter++);
        PROFILER_EVENT_START(TASK_WORKER_EXIT, ev_no);
        // Extend the exit marker slightly for easier observation.
        __nanosleep(200); // ~O(100ns), architecture/clock dependent
        PROFILER_EVENT_END(TASK_WORKER_EXIT, ev_no);
      }
#endif
      // Terminate
      return;
    } else if (task_desc->task_type == TASK_BEGIN_TASK_GRAPH) {
      // Do nothing
    } else {
#ifdef MPK_ENABLE_VERBOSE
      if (threadIdx.x == 0) {
        printf("[worker] _execute_task EXECUTE_TASK %d\n",
               task_desc->task_type);
      }
#endif
      __mirage_execute_task_noinline(task_desc,
                                     config,
                                     &__mirage_exec_ctx_single,
                                     0,
                                     __mirage_smem_base,
                                     __mirage_smem_capacity);
    }
    __syncthreads();

#ifdef MPK_ENABLE_PROFILING
    if (task_desc->task_type != TASK_TERMINATE) {
      PROFILER_EVENT_END(task_desc->task_type, task_counter++);
    }
#endif

    // Trigger event
    if (threadIdx.x == 0) {
      EventId event_id = task_desc->trigger_event;
      size_t event_index = get_event_position_index(event_id);
      if (!is_nvshmem_event(event_id)) {
        size_t gpu_id = get_event_gpu_id(event_id);
        assert(gpu_id == config.my_gpu_id);
        // Case 1: Trigger a local non-nvshmem event
        // int count = atomicSub(&config.all_event_counters[event_index], 1);
        // Relaxed is sufficient: this counter is used only for atomicity; the
        // scheduler visibility is established by the later queue publish.
        EventCounter count = atom_add_relaxed_gpu_u64(
            &config.all_event_counters[event_index], 1);
        int num_triggers = config.all_event_num_triggers[event_index];
#ifdef MPK_ENABLE_VERBOSE
        printf("[%d][DONE] worker_id(%d) iter_num(%llu) task_idx(%llu) "
               "event_id(%llu) "
               "event_type(local) count(%llu)\n",
               config.my_gpu_id,
               worker_id,
               get_task_iteration_num(task_ids[queue_pos]),
               get_task_position_index(task_ids[queue_pos]),
               event_id,
               count);
#endif

        if ((count + 1) == static_cast<EventCounter>(num_triggers) *
                               get_task_iteration_num(task_ids[queue_pos])) {
#ifdef MPK_ENABLE_PROFILING
          PROFILER_EVENT_START(TASK_SCHD_EVENTS, task_counter);
#endif
          EventDesc event_desc = config.all_events[event_index];
          // The event has been triggered enough times
          // Refresh the event counter
          // atom_add_release_gpu_u64(&config.all_event_counters[event_index],
          //                       event_desc.num_triggers);
          // Add the event to the schedule_queue
          // Note that events launching massive tasks are scheduled
          // to the global sched_queue
          if (event_desc.event_type == EVENT_EMPTY) {
            // Do nothing for empty event
          } else {
            bool use_bcast_queue = false;
            if (event_desc.event_type == EVENT_LAUNCH_MASSIVE_TASKS ||
                event_desc.event_type == EVENT_LAUNCH_DEPENDENT_TASKS) {
              use_bcast_queue = true;
            }
            int sched_id =
                use_bcast_queue
                    ? config.num_local_schedulers + config.num_remote_schedulers
                    : get_rand_sched_id(event_index,
                                        worker_id,
                                        config.num_workers,
                                        config.num_local_schedulers);
            // Relaxed is sufficient: this is only a slot allocator.
            size_t last_event_pos = atom_add_relaxed_gpu_u64(
                &config.sched_queue_next_free_event_id[sched_id], 1);
            st_relaxed_gpu_u64(
                &config.sched_queues[sched_id][last_event_pos %
                                               config.per_sched_queue_len],
                event_index);
            // Use st.relaxed to make sure that the updated event_index is
            // visible to the scheduler CTA before updating its
            // last_ready_event_id
            size_t old;
            do {
              old = atom_cas_release_gpu_u64(
                  &config.sched_queue_last_ready_event_id[sched_id],
                  last_event_pos,
                  last_event_pos + 1);
            } while (old != last_event_pos);
          }
#ifdef MPK_ENABLE_PROFILING
          PROFILER_EVENT_END(TASK_SCHD_EVENTS, task_counter++);
#endif
        }
      } else {
        // Case 2: trigger a nvshmem event
        assert(task_desc->task_type == TASK_NVSHMEM_COPY);
        // Note that nvshmem copy task signal counter during data copy
        // we don't need to do anything here is the task type is NVSHMEM_COPY
#ifdef MPK_ENABLE_VERBOSE
        printf("[%d][DONE] worker_id(%d) task_id(%llu) event_id(%llx) "
               "event_type(remote)\n",
               config.my_gpu_id,
               worker_id,
               get_task_position_index(task_ids[queue_pos]),
               event_id);
#endif
      }
    }
    queue_pos += 1;
  }
}

// Worker control-path helper: fetch a TaskId from worker queue(s) and load its
// TaskDesc from `config.all_tasks` (no task execution).
__device__ __forceinline__ bool
    __mirage_worker_fetch_load_task_desc(RuntimeConfig const &config,
                                         int num_worker_queues,
                                         TaskId **worker_queues,
                                         int *worker_queue_ids,
                                         size_t *next_task_pos,
                                         size_t *last_task_pos,
                                         int *loader_queue_idx,
                                         TaskId *out_task_id,
                                         TaskDesc *out_task_desc) {
  int q = *loader_queue_idx;
  int scanned = 0;
  while (next_task_pos[q] == last_task_pos[q]) {
    last_task_pos[q] = ld_acquire_gpu_u64(
        &config.worker_queue_last_ready_task_id[worker_queue_ids[q]]);
    if (next_task_pos[q] < last_task_pos[q]) {
      break;
    }
    q = (q == num_worker_queues - 1) ? 0 : (q + 1);
    scanned++;
    if (scanned >= num_worker_queues) {
      *loader_queue_idx = q;
      return false;
    }
    __nanosleep(10);
  }

  if (next_task_pos[q] == last_task_pos[q]) {
    *loader_queue_idx = q;
    return false;
  }

  TaskId task_id = ld_relaxed_gpu_u64(
      &worker_queues[q][next_task_pos[q] % config.per_worker_queue_len]);
  size_t task_pos = get_task_position_index(task_id);
  *out_task_id = task_id;
  *out_task_desc = config.all_tasks[task_pos];
  next_task_pos[q] += 1;
  *loader_queue_idx = q;
  return true;
}

// Fetch a TaskId and its position index from worker queue(s), without loading
// the TaskDesc. This avoids default-constructing TaskDesc in device code.
__device__ __forceinline__ bool
    __mirage_worker_fetch_task_id(RuntimeConfig const &config,
                                  int num_worker_queues,
                                  TaskId **worker_queues,
                                  int *worker_queue_ids,
                                  size_t *next_task_pos,
                                  size_t *last_task_pos,
                                  int *loader_queue_idx,
                                  TaskId *out_task_id,
                                  size_t *out_task_pos) {
  int q = *loader_queue_idx;
  int scanned = 0;
  while (next_task_pos[q] == last_task_pos[q]) {
    last_task_pos[q] = ld_acquire_gpu_u64(
        &config.worker_queue_last_ready_task_id[worker_queue_ids[q]]);
    if (next_task_pos[q] < last_task_pos[q]) {
      break;
    }
    q = (q == num_worker_queues - 1) ? 0 : (q + 1);
    scanned++;
    if (scanned >= num_worker_queues) {
      *loader_queue_idx = q;
      return false;
    }
    __nanosleep(10);
  }

  if (next_task_pos[q] == last_task_pos[q]) {
    *loader_queue_idx = q;
    return false;
  }

  TaskId task_id = ld_relaxed_gpu_u64(
      &worker_queues[q][next_task_pos[q] % config.per_worker_queue_len]);
  size_t task_pos = get_task_position_index(task_id);
  *out_task_id = task_id;
  *out_task_pos = task_pos;
  next_task_pos[q] += 1;
  *loader_queue_idx = q;
  return true;
}

// Single-queue fast path: used by execute_worker_multi_group after stripping
// multi-GPU logic. Avoids arrays and round-robin scanning.
__device__ __forceinline__ bool __mirage_worker_fetch_task_id_single_queue(
    RuntimeConfig const &config,
    TaskId *worker_queue,
    int worker_queue_id,
    size_t *next_task_pos,
    size_t *last_task_pos,
    TaskId *out_task_id,
    size_t *out_task_pos) {
  if (*next_task_pos == *last_task_pos) {
    *last_task_pos =
        ld_acquire_gpu_u64(&config.worker_queue_last_ready_task_id[worker_queue_id]);
  }
  if (*next_task_pos == *last_task_pos) {
    __nanosleep(10);
    return false;
  }
  TaskId task_id = ld_relaxed_gpu_u64(
      &worker_queue[*next_task_pos % config.per_worker_queue_len]);
  *out_task_id = task_id;
  *out_task_pos = get_task_position_index(task_id);
  *next_task_pos += 1;
  return true;
}

// Entry-function regprobe so ptxas reliably reports register usage for the
// fetch/load control-path helper. This kernel is never launched in production.
extern "C" __global__
    __launch_bounds__(256) void __mirage_regprobe_worker_fetch_load_task_desc(
        RuntimeConfig const *config,
        int num_worker_queues,
        TaskId **worker_queues,
        int *worker_queue_ids,
        size_t *next_task_pos,
        size_t *last_task_pos,
        int *loader_queue_idx,
        TaskId *out_task_id,
        TaskDesc *out_task_desc) {
  bool ok = __mirage_worker_fetch_load_task_desc(*config,
                                                 num_worker_queues,
                                                 worker_queues,
                                                 worker_queue_ids,
                                                 next_task_pos,
                                                 last_task_pos,
                                                 loader_queue_idx,
                                                 out_task_id,
                                                 out_task_desc);
  asm volatile("" ::"r"((int)ok));
}

// Trigger-event logic as a standalone noinline "task" (post-task control path),
// so register usage is stable/predictable and can be controlled via setmaxnreg.
__device__ __forceinline__ void
    __mirage_trigger_event_noinline(RuntimeConfig const &config,
                                    int worker_id,
                                    TaskId task_id,
                                    TaskDesc const *task_desc) {
  EventId event_id = task_desc->trigger_event;
  if (event_id == EVENT_INVALID_ID) {
    return;
  }
  size_t event_index = get_event_position_index(event_id);
  if (is_nvshmem_event(event_id)) {
    // nvshmem copy task signals counter during data copy; nothing to do here.
    return;
  }
  size_t gpu_id = get_event_gpu_id(event_id);
  assert(gpu_id == config.my_gpu_id);
  // Relaxed is sufficient: the scheduler visibility is established by the later
  // queue publish (store + release update of last_ready).
  EventCounter count =
      atom_add_relaxed_gpu_u64(&config.all_event_counters[event_index], 1);
  int num_triggers = config.all_event_num_triggers[event_index];
  if ((count + 1) == static_cast<EventCounter>(num_triggers) *
                         get_task_iteration_num(task_id)) {
    EventDesc event_desc = config.all_events[event_index];
    if (event_desc.event_type == EVENT_EMPTY) {
      return;
    }
    bool use_bcast_queue = false;
    if (event_desc.event_type == EVENT_LAUNCH_MASSIVE_TASKS ||
        event_desc.event_type == EVENT_LAUNCH_DEPENDENT_TASKS) {
      use_bcast_queue = true;
    }
    int sched_id =
        use_bcast_queue
            ? config.num_local_schedulers + config.num_remote_schedulers
            : get_rand_sched_id(event_index,
                                worker_id,
                                config.num_workers,
                                config.num_local_schedulers);
    // Relaxed is sufficient: this is only a slot allocator.
    size_t last_event_pos = atom_add_relaxed_gpu_u64(
        &config.sched_queue_next_free_event_id[sched_id], 1);
    st_relaxed_gpu_u64(
        &config.sched_queues[sched_id]
                            [last_event_pos % config.per_sched_queue_len],
        event_index);
    size_t old;
    do {
      old = atom_cas_release_gpu_u64(
          &config.sched_queue_last_ready_event_id[sched_id],
          last_event_pos,
          last_event_pos + 1);
    } while (old != last_event_pos);
  }
}

extern "C" __global__ __launch_bounds__(
    256) void __mirage_regprobe_trigger_event(RuntimeConfig const *config,
                                              TaskId task_id,
                                              TaskDesc const *task_desc) {
  __mirage_trigger_event_noinline(*config, 0, task_id, task_desc);
  asm volatile("" ::"l"((unsigned long long)task_id));
}

// 2-group worker (single GPU), Scheme E (mailboxes + partitioned queue heads):
// - Each warp-group has its own 1-slot mailbox in shared memory (TaskId + TaskDesc).
// - If mailbox empty, group pops the next task index from worker_queue via a
//   per-group queue head `next_task_pos[group_id]` (even/odd), then loads its
//   TaskDesc into mailbox.
// - Groups execute tasks concurrently without any shared pool_state scanning.
__device__ __forceinline__ void execute_worker_multi_group_aligned(
    RuntimeConfig config) {
  // Single-GPU only: one worker queue per worker CTA.
  __shared__ TaskId *worker_queue;
  __shared__ int worker_queue_id;
  __shared__ size_t next_task_pos[2];
  __shared__ size_t last_task_pos;

  // Per-group mailbox.
  __shared__ TaskId slot_task_id[2];
  __shared__ TaskDesc slot_task_desc[2];
  __shared__ int slot_state[2]; // 0 empty, 1 full
  // Snapshot slot_state per group to keep control flow converged around wg_sync.
  __shared__ int slot_state_snapshot[2];

  __shared__ int terminate_all;
  // CTA-wide lock to serialize task execution when tasks reuse the full dynamic
  // shared memory region (Ampere tasks use `extern __shared__` and cannot run
  // concurrently across groups without per-group SMEM partitioning).
  __shared__ int task_exec_lock;

  // Per-group scratch for fetch+load.
  __shared__ int group_has_taskpos[2];
  __shared__ size_t group_task_pos[2];
  __shared__ TaskId group_task_id[2];

  // Per-group scratch for dependent-event waiting.
  __shared__ int dep_ok[2];

  // Converged termination snapshot (lane0 writes, group reads after wg_sync).
  __shared__ int terminate_snapshot[2];

  __shared__ __mirage_exec_ctx exec_ctx;

  extern __shared__ char __smem_base[];
  uintptr_t smem_base_u =
      (reinterpret_cast<uintptr_t>(__smem_base) + 1023) / 1024 * 1024;
  char *smem_base = reinterpret_cast<char *>(smem_base_u);
  uint32_t smem_capacity =
      static_cast<uint32_t>(config.worker_dynamic_smem_bytes) -
      static_cast<uint32_t>(smem_base_u -
                            reinterpret_cast<uintptr_t>(__smem_base));

  int const worker_id = blockIdx.x;
  int const group_id = threadIdx.x / WORKER_NUM_THREADS;
  int const lane = threadIdx.x % WORKER_NUM_THREADS;
  int const warp_id = lane >> 5;
  int const warp_lane = lane & 31;

#ifdef MPK_ENABLE_PROFILING
  PROFILER_CLOSURE_PARAMS_DECL;
  uint32_t const num_groups =
      static_cast<uint32_t>(blockDim.x / WORKER_NUM_THREADS);
  __shared__ uint32_t task_counter;
  // Per-group profiling state for the "fetch task" slow path (mailbox fill).
  __shared__ uint32_t fetch_task_ev_no[2];
  __shared__ int fetch_task_ev_active[2];
  uint32_t local_task_counter = 0;
  PROFILER_INIT(static_cast<uint64_t *>(config.profiler_buffer),
                static_cast<uint32_t>(group_id),
                num_groups,
                (threadIdx.x % WORKER_NUM_THREADS == 0));
#endif

  if (threadIdx.x == 0) {
    worker_queue = config.worker_queues[worker_id];
    worker_queue_id = worker_id;
    next_task_pos[0] = 0;
    next_task_pos[1] = 1;
    last_task_pos = 0;
    terminate_all = 0;
    task_exec_lock = 0;
    slot_state[0] = 0;
    slot_state[1] = 0;
    slot_state_snapshot[0] = 0;
    slot_state_snapshot[1] = 0;
    group_has_taskpos[0] = 0;
    group_has_taskpos[1] = 0;
    dep_ok[0] = 0;
    dep_ok[1] = 0;
    terminate_snapshot[0] = 0;
    terminate_snapshot[1] = 0;
    __mirage_exec_ctx_init(&exec_ctx, config);
#ifdef MPK_ENABLE_PROFILING
    task_counter = 0;
    fetch_task_ev_no[0] = 0;
    fetch_task_ev_no[1] = 0;
    fetch_task_ev_active[0] = 0;
    fetch_task_ev_active[1] = 0;
#endif
  }
  __syncthreads();

  while (true) {
    // Keep the warp-group in lockstep across loop iterations. Without an
    // unconditional group barrier here, different warps can observe different
    // values of shared flags (e.g., slot_state/terminate_all) and take
    // mismatched paths, which can strand threads at later wg_sync points.
    // wg_sync<WORKER_NUM_THREADS>(4);

    // Fill mailbox if empty. Read slot_state once (lane0), then use a converged
    // branch predicate so that either the entire group enters the block (and
    // hits wg_sync) or the entire group skips it.
    if (lane == 0) {
      slot_state_snapshot[group_id] = atomicAdd(&slot_state[group_id], 0);
      terminate_snapshot[group_id] = atomicAdd(&terminate_all, 0);
    }
    wg_sync<WORKER_NUM_THREADS>(3);
    if (terminate_snapshot[group_id] != 0) {
      // Keep the existing "extra" exit-path barrier safe by making the branch
      // predicate converged (terminate_snapshot is lane0-owned).
#if MIRAGE_ADMISSION_DEBUG
      if (lane == 0) {
        printf("[ADMIT][%d] worker=%d group=%d EXIT reason=terminate_all=1\n",
               (int)config.my_gpu_id,
               worker_id,
               group_id);
      }
#endif
      return;
    }
    if (slot_state_snapshot[group_id] == 0) {
      if (lane == 0) {
        group_has_taskpos[group_id] = 0;
#ifdef MPK_ENABLE_PROFILING
        fetch_task_ev_active[group_id] = 0;
#endif
        // Partitioned queue pop (no contention): each group advances its own
        // head by 2, so group0 consumes indices 0,2,4,... and group1 consumes
        // 1,3,5,... .
        if (atomicAdd(&terminate_all, 0) == 0) {
          // Each group must acquire-load last_ready itself to establish the
          // release/acquire chain that makes worker_queue[cur] visible.
          size_t last = static_cast<size_t>(ld_acquire_gpu_u64(
              &config.worker_queue_last_ready_task_id[worker_queue_id]));
          size_t cur = next_task_pos[group_id];
          if (cur < last) {
            TaskId tid = ld_relaxed_gpu_u64(
                &worker_queue[cur % config.per_worker_queue_len]);
#ifdef MPK_ENABLE_PROFILING
            unsigned long long const tid_iter = get_task_iteration_num(tid);
            if (tid_iter == 110ull) {
              fetch_task_ev_no[group_id] = atomicAdd(&task_counter, 1u);
              fetch_task_ev_active[group_id] = 1;
              PROFILER_EVENT_START(TASK_GET_NEXT_TASK,
                                   fetch_task_ev_no[group_id]);
            }
#endif
            size_t tpos = get_task_position_index(tid);
            group_task_id[group_id] = tid;
            group_task_pos[group_id] = tpos;
            group_has_taskpos[group_id] = 1;
            next_task_pos[group_id] = cur + 2;
          } else {
          }
        } else {
        }
      }
      wg_sync<WORKER_NUM_THREADS>(3);

      if (!group_has_taskpos[group_id]) {
        __nanosleep(50);
        continue;
      }

      TaskId const tid = group_task_id[group_id];
      size_t const tpos = group_task_pos[group_id];

      if (tid == 0ull) {
        if (lane == 0) {
          slot_task_id[group_id] = 0ull;
          TaskDesc &td = slot_task_desc[group_id];
          td.task_type = TASK_TERMINATE;
          td.variant_id = 0;
          td.dependent_event = EVENT_INVALID_ID;
          td.trigger_event = EVENT_INVALID_ID;
          __threadfence_block();
          atomicExch(&slot_state[group_id], 1);
#ifdef MPK_ENABLE_PROFILING
          if (fetch_task_ev_active[group_id]) {
            PROFILER_EVENT_END(TASK_GET_NEXT_TASK, fetch_task_ev_no[group_id]);
          }
#endif
        }
        wg_sync<WORKER_NUM_THREADS>(3);
      } else {
        static_assert(sizeof(TaskDesc) % 16 == 0);
        constexpr int TASK_SIZE = sizeof(TaskDesc) / 16; // 128b copy-async
        // Warp-based load without loops: each participating warp issues at most
        // one 16B cp.async per lane. Each warp must also fence/wait its own
        // cp.async queue (cp.async is warp-scoped).
        constexpr int LOAD_WARPS = (TASK_SIZE + 31) / 32;
        if (warp_id < LOAD_WARPS) {
          int const i = warp_id * 32 + warp_lane;
          if (i < TASK_SIZE) {
            load_smem(
                reinterpret_cast<char *>(&slot_task_desc[group_id]) + i * 16,
                reinterpret_cast<char *>(config.all_tasks + tpos) + i * 16);
          }
          kernel::cp_async_fence();
          kernel::cp_async_wait<0>();
        }
        wg_sync<WORKER_NUM_THREADS>(3);
        if (lane == 0) {
          slot_task_id[group_id] = tid;
          __threadfence_block();
          atomicExch(&slot_state[group_id], 1);
#ifdef MPK_ENABLE_PROFILING
          if (fetch_task_ev_active[group_id]) {
            PROFILER_EVENT_END(TASK_GET_NEXT_TASK, fetch_task_ev_no[group_id]);
          }
#endif
        }
        // Ensure slot_task_id/slot_task_desc publication is visible to all warps
        // before any thread proceeds to read/claim the mailbox.
        wg_sync<WORKER_NUM_THREADS>(3);
      }
    }

    TaskId const task_id = slot_task_id[group_id];
    unsigned long long const task_iter = get_task_iteration_num(task_id);
    TaskDesc const *task_desc = &slot_task_desc[group_id];
#ifdef MPK_ENABLE_PROFILING
    int do_profile = 0;
    if (lane == 0) {
      if (task_iter == 110ull) {
        do_profile = 1;
      }
    }
#endif

#if MIRAGE_WORKER_LOG
    if (blockIdx.x == 0 && lane == 0) {
      unsigned long long const task_pos =
          (unsigned long long)get_task_position_index(task_id);
      printf("[WORKER][EXEC] worker=%d group=%d task_pos=%llu task_type=%d variant=%u\n",
             worker_id,
             group_id,
             task_pos,
             static_cast<int>(task_desc->task_type),
             static_cast<unsigned>(task_desc->variant_id));
    }
#endif

    if (task_id == 0ull) {
      if (lane == 0) {
#if MIRAGE_ADMISSION_DEBUG
        printf("[ADMIT][%d] worker=%d group=%d EXIT reason=TERM-TOKEN 2\n",
               (int)config.my_gpu_id,
               worker_id,
               group_id);
#endif
        atomicExch(&terminate_all, 1);
        atomicExch(&slot_state[group_id], 0);
#ifdef MPK_ENABLE_PROFILING
        if (do_profile) {
          local_task_counter = atomicAdd(&task_counter, 1u);
          PROFILER_EVENT_START(TASK_WORKER_EXIT, local_task_counter);
          __nanosleep(200);
          PROFILER_EVENT_END(TASK_WORKER_EXIT, local_task_counter);
        }
#endif
      }
      wg_sync<WORKER_NUM_THREADS>(3);
      return;
    }

    if (task_desc->task_type == TASK_TERMINATE) {
      if (lane == 0) {
#if MIRAGE_ADMISSION_DEBUG
        printf("[ADMIT][%d] worker=%d group=%d EXIT reason=TASK_TERMINATE 3\n",
               (int)config.my_gpu_id,
               worker_id,
               group_id);
#endif
        atomicExch(&terminate_all, 1);
        atomicExch(&slot_state[group_id], 0);
#ifdef MPK_ENABLE_PROFILING
        if (do_profile) {
          local_task_counter = atomicAdd(&task_counter, 1u);
          PROFILER_EVENT_START(TASK_WORKER_EXIT, local_task_counter);
          __nanosleep(200);
          PROFILER_EVENT_END(TASK_WORKER_EXIT, local_task_counter);
        }
#endif
      }
      wg_sync<WORKER_NUM_THREADS>(3);
      return;
    }

    if (task_desc->dependent_event != EVENT_INVALID_ID) {
      if (lane == 0) {
        EventId event_id = task_desc->dependent_event;
        size_t event_index = get_event_position_index(event_id);
        EventCounter needed_counts =
            static_cast<EventCounter>(config.all_event_num_triggers[event_index]) *
            get_task_iteration_num(task_id);
        EventCounter actual_counts = 0;
        while (actual_counts < needed_counts) {
          if (atomicAdd(&terminate_all, 0) != 0) {
            break;
          }
          actual_counts = ld_acquire_gpu_u64(&config.all_event_counters[event_index]);
          __nanosleep(10);
        }
        dep_ok[group_id] = (actual_counts >= needed_counts) ? 1 : 0;
      }
      wg_sync<WORKER_NUM_THREADS>(3);
      if (!dep_ok[group_id]) {
        __nanosleep(50);
        continue;
      }
    }

#if MIRAGE_ADMISSION_DEBUG
    MIRAGE_ADMIT_DPRINTF(
        lane,
        "[ADMIT][%d] worker=%d group=%d TASK-BEGIN task_pos=%llu type=%d\n",
        (int)config.my_gpu_id,
        worker_id,
        group_id,
        (unsigned long long)get_task_position_index(task_id),
        (int)task_desc->task_type);
#endif

#ifdef MPK_ENABLE_PROFILING
    if (lane == 0 && do_profile) {
      local_task_counter = atomicAdd(&task_counter, 1u);
      PROFILER_EVENT_START(task_desc->task_type, local_task_counter);
    }
#endif

    if (task_desc->task_type != TASK_BEGIN_TASK_GRAPH) {
      __mirage_execute_task_noinline(
          task_desc, config, &exec_ctx, group_id, smem_base, smem_capacity);
    }
    wg_sync<WORKER_NUM_THREADS>(3);

#ifdef MPK_ENABLE_PROFILING
    if (lane == 0 && do_profile) {
      PROFILER_EVENT_END(task_desc->task_type, local_task_counter);
    }
#endif

#ifdef MPK_ENABLE_PROFILING
    if (lane == 0 && do_profile) {
      local_task_counter = atomicAdd(&task_counter, 1u);
      PROFILER_EVENT_START(TASK_SCHD_EVENTS, local_task_counter);
    }
#endif
    if (lane == 0) {
      __mirage_trigger_event_noinline(config, worker_id, task_id, task_desc);
#if MIRAGE_ADMISSION_DEBUG
      printf("[ADMIT][%d] worker=%d group=%d TRIG-DONE task_pos=%llu type=%d\n",
             (int)config.my_gpu_id,
             worker_id,
             group_id,
             (unsigned long long)get_task_position_index(task_id),
             (int)task_desc->task_type);
#endif
#ifdef MPK_ENABLE_PROFILING
      if (do_profile) {
        PROFILER_EVENT_END(TASK_SCHD_EVENTS, local_task_counter);
      }
#endif
      atomicExch(&slot_state[group_id], 0);
    }
    wg_sync<WORKER_NUM_THREADS>(3);
  }
}

// need to alter as there is only one warp per block
__device__ __forceinline__ void execute_scheduler(RuntimeConfig config,
                                                  int offset) {
  int const num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  int const warp_id = threadIdx.x / 32;
  // One scheduler per block: only warp 0 participates.
  if (threadIdx.x % 32 == 0 && warp_id == 0) {
    int const sched_id = blockIdx.x + offset;
    // if (threadIdx.x == 0) {
    //   int sched_id = (blockIdx.x - config.num_workers);
    size_t iteration_num = 0;
    EventId *sched_queue0 = config.sched_queues[sched_id];
    int sched_queue_id0 = sched_id;
    EventId *sched_queue1 = nullptr;
    int sched_queue_id1 = -1;
    bool has_queue1 = false;
    unsigned long long int my_first_worker, my_last_worker;

    if (sched_id < config.num_local_schedulers) {
      // local schedulers also (collectively) process events from
      // the global queue
      sched_queue1 = config.sched_queues[num_schedulers];
      sched_queue_id1 = num_schedulers;
      has_queue1 = true;
      get_first_last_ids(config.num_workers,
                         config.num_local_schedulers,
                         sched_id,
                         &my_first_worker,
                         &my_last_worker);
    } else {
      get_first_last_ids(config.num_workers,
                         config.num_remote_schedulers,
                         sched_id - config.num_local_schedulers,
                         &my_first_worker,
                         &my_last_worker);
      // Remote schedulers send tasks to remove worker queue
      // whose ids start from config.num_workers
      my_first_worker += config.num_workers;
      my_last_worker += config.num_workers;
    }

    // ONLY can run when comment this chunk
#ifdef MPK_ENABLE_VERBOSE
    printf("[SCHD] sched_id(%d) first_worker(%llu) last_worker(%llu)\n",
           sched_id,
	   my_first_worker,
	   my_last_worker);
#endif
    size_t cur_event_pos0 = 0, last_event_pos0 = 0;
    size_t cur_event_pos1 = 0, last_event_pos1 = 0;

    __shared__ size_t worker_queue_next_free_task_pos[MAX_WORKER_PER_SCHEDULER];
    for (int i = 0; i < MAX_WORKER_PER_SCHEDULER; i++) {
      worker_queue_next_free_task_pos[i] = 0;
    }

    // if (sched_id == 0) {
    //   worker_queue_next_free_task_pos[0] = 1;
    // }
    int next_worker = my_first_worker;
    int next_begin_worker = my_first_worker;
    int queue_idx = 0;
    while (true) {
      while (true) {
        if (queue_idx == 0) {
          if (cur_event_pos0 != last_event_pos0) {
            break;
          }
          last_event_pos0 = ld_acquire_gpu_u64(
              &config.sched_queue_last_ready_event_id[sched_queue_id0]);
          if (cur_event_pos0 < last_event_pos0) {
            break;
          }
          if (has_queue1) {
            queue_idx = 1;
          }
        } else {
          if (cur_event_pos1 != last_event_pos1) {
            break;
          }
          last_event_pos1 = ld_acquire_gpu_u64(
              &config.sched_queue_last_ready_event_id[sched_queue_id1]);
          if (cur_event_pos1 < last_event_pos1) {
            break;
          }
          queue_idx = 0;
        }
        __nanosleep(10);
      }
      // Make sure the schedule queue is not overflow
      size_t cur_event_pos = (queue_idx == 0) ? cur_event_pos0 : cur_event_pos1;
      size_t last_event_pos = (queue_idx == 0) ? last_event_pos0 : last_event_pos1;
      assert(cur_event_pos + config.per_sched_queue_len > last_event_pos);
      // Launch new tasks
      // Use ld.acquire to read latest events
      EventId *sched_queue = (queue_idx == 0) ? sched_queue0 : sched_queue1;
      EventId event_id = ld_relaxed_gpu_u64(
          &sched_queue[cur_event_pos % config.per_sched_queue_len]);
      if (is_termination_event(event_id)) {
        // terminate all workers
        if (sched_id < config.num_local_schedulers) {
          for (int i = my_first_worker; i < my_last_worker; i++) {
            size_t last_task_id =
                worker_queue_next_free_task_pos[i - my_first_worker]++;
            st_relaxed_gpu_u64(
                &config.worker_queues[i][last_task_id %
                                         config.per_worker_queue_len],
                0);
            atom_add_release_gpu_u64(&config.worker_queue_last_ready_task_id[i],
                                     1);
          }
        }
        return;
      }
      EventDesc const &e = config.all_events[event_id];
      // This is the ending task of the current task graph
      if (e.event_type == EVENT_END_OF_TASK_GRAPH) {
#ifdef MPK_ENABLE_VERBOSE
        printf("[SCHD] END_OF_TASK_GRAPH\n");
#endif
        // Check if we want to continue.
#ifdef MODE_ONEPASS
        // One-pass: the host seeds an END_OF_TASK_GRAPH event as a bootstrap.
        // When we observe it with iteration_num==0, we launch BEGIN_TASK_GRAPH
        // for iteration 1. When END_OF_TASK_GRAPH arrives again (after the real
        // graph completes), terminate.
        bool const continue_running = (iteration_num == 0);
#else
        bool const continue_running = prepare_next_batch(config);
#endif
        if (!continue_running) {
          terminate_schedulers(config);
        } else {
          // Launch task 1 (begin_task_graph) for the next iteration.
          int const begin_worker = next_begin_worker;
          size_t last_task_id =
              worker_queue_next_free_task_pos[begin_worker - my_first_worker]++;
          st_relaxed_gpu_u64(
              &config.worker_queues[begin_worker]
                                   [last_task_id % config.per_worker_queue_len],
              compute_task_id(iteration_num + 1, 1 /*begin_task_graph*/));
          // Use st.relaxed to make sure writes to worker_queues is visible to
          // worker CTAs before we increase its last_ready_task_id.
          atom_add_release_gpu_u64(
              &config.worker_queue_last_ready_task_id[begin_worker], 1);
#ifdef MPK_ENABLE_VERBOSE
          printf("[%d][SCHD]EVENT_END_OF_TASK_GRAPH schd_id(%d) "
                 "iter_num(%llu) task_idx(1) "
                 "worker_id(%d) "
                 "worker_last_ready_pos(%llu)\n",
                 config.my_gpu_id,
                 sched_id,
                 iteration_num + 1,
                 begin_worker,
                 last_task_id + 1);
#endif
          if (next_begin_worker == my_last_worker - 1) {
            next_begin_worker = my_first_worker;
          } else {
            next_begin_worker++;
          }
        }
      } else if (e.event_type == EVENT_LAUNCH_DEPENDENT_TASKS) {
        iteration_num = iteration_num + 1;
        // assign event in a round-robin fashion
        // Split event across local schedulers
        assert(sched_id < config.num_local_schedulers);
        for (size_t i = 0;
             i < (e.last_task_id - e.first_task_id + config.num_workers - 1) /
                     config.num_workers;
             i++) {
          for (size_t j = my_first_worker; j < my_last_worker; j++) {
            size_t position_index =
                e.first_task_id + i * config.num_workers + j;
            if (position_index < e.last_task_id) {
              size_t last_task_id =
                  worker_queue_next_free_task_pos[next_worker -
                                                  my_first_worker]++;
              st_relaxed_gpu_u64(
                  &config
                       .worker_queues[next_worker][last_task_id %
                                                   config.per_worker_queue_len],
                  compute_task_id(iteration_num, position_index));
              // Use st.relaxed to make sure writes to worker_queues is visible
              // to worker CTAs before we increase its last_ready_task_id
              atom_add_release_gpu_u64(
                  &config.worker_queue_last_ready_task_id[next_worker], 1);

#ifdef MPK_ENABLE_VERBOSE
              if (sched_id == 0) {
                printf("[%d][SCHD] EVENT_LAUNCH_DEPENDENT_TASKS schd_id(%d) "
                       "iter_num(%llu) task_idx(%llu) "
                       "worker_id(%d) "
                       "worker_last_ready_pos(%llu)"
                       "event_id(%llu)"
                       "event_range(%llu-%llu)\n",
                       config.my_gpu_id,
                       sched_id,
                       iteration_num,
                       position_index,
                       next_worker,
                       last_task_id + 1,
                       event_id,
                       e.first_task_id,
                       e.last_task_id);
              }
#endif
              next_worker = (next_worker == my_last_worker - 1)
                                ? my_first_worker
                                : next_worker + 1;
            }
          }
        }
      } else {
        TaskId my_first_task = e.first_task_id, my_last_task = e.last_task_id;
        if (e.event_type == EVENT_LAUNCH_MASSIVE_TASKS) {
          // Split event across local schedulers
          assert(sched_id < config.num_local_schedulers);
          get_first_last_ids(e.last_task_id - e.first_task_id,
                             config.num_local_schedulers,
                             sched_id,
                             &my_first_task,
                             &my_last_task);
          my_first_task += e.first_task_id;
          my_last_task += e.first_task_id;
        }
        for (size_t i = my_first_task; i < my_last_task; i++) {
          //  size_t last_task_id = atomicAdd(
          //      &(config.worker_queue_next_free_task_id[next_worker]), 1);
          //  size_t last_task_id = atom_add_release_gpu_u64(
          //     &(config.worker_queue_next_free_task_id[next_worker]), 1);
          size_t last_task_id =
              worker_queue_next_free_task_pos[next_worker - my_first_worker]++;
          st_relaxed_gpu_u64(
              &config.worker_queues[next_worker]
                                   [last_task_id % config.per_worker_queue_len],
              compute_task_id(iteration_num, i));
          // Use st.relaxed to make sure writes to worker_queues is visible to
          // worker CTAs before we increase its last_ready_task_id
          atom_add_release_gpu_u64(
              &config.worker_queue_last_ready_task_id[next_worker], 1);

#ifdef MPK_ENABLE_VERBOSE
          printf("[%d][SCHD] EXECUTE_TASK schd_id(%d) iter_num(%llu) "
                 "task_idx(%llu) "
                 "worker_id(%d) "
                 "worker_last_ready_pos(%llu)\n",
                 config.my_gpu_id,
                 sched_id,
                 iteration_num,
                 i,
                 next_worker,
                 last_task_id + 1);
#endif

          next_worker = (next_worker == my_last_worker - 1) ? my_first_worker
                                                            : next_worker + 1;
        }
      }
      if (queue_idx == 0) {
        cur_event_pos0 += 1;
      } else {
        cur_event_pos1 += 1;
      }
    }
  }
}

__device__ __forceinline__ void __mirage_sched_distribute_balanced_to_workers_warp(
    RuntimeConfig const &config,
    size_t *next_free_pos,
    int lane,
    int my_first_worker,
    int my_last_worker,
    unsigned long long iter_num,
    unsigned long long first_task_pos,
    unsigned long long last_task_pos_exclusive);

// Balanced scheduler variant based on execute_scheduler:
// - If task_count < num_workers: strict round-robin (0/1 per worker).
// - If num_workers <= task_count < 2*num_workers: one task each, then give the
//   remaining tasks as the "second task" to the first N workers.
// - If task_count >= 2*num_workers: contiguous blocks, with block sizes aligned
//   to 2 (even) as much as possible to favor 2-group pairing.
__device__ __forceinline__ void execute_scheduler_balanced(RuntimeConfig config,
                                                           int offset) {
  int const num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  int const warp_id = threadIdx.x / 32;
  if (threadIdx.x % 32 == 0) {
    int const sched_id = blockIdx.x + offset;
    size_t iteration_num = 0;
    EventId *sched_queue0 = config.sched_queues[sched_id];
    int sched_queue_id0 = sched_id;
    EventId *sched_queue1 = nullptr;
    int sched_queue_id1 = -1;
    bool has_queue1 = false;
    unsigned long long int my_first_worker, my_last_worker;

    if (sched_id < config.num_local_schedulers) {
      sched_queue1 = config.sched_queues[num_schedulers];
      sched_queue_id1 = num_schedulers;
      has_queue1 = true;
      get_first_last_ids(config.num_workers,
                         config.num_local_schedulers,
                         sched_id,
                         &my_first_worker,
                         &my_last_worker);
    } else {
      get_first_last_ids(config.num_workers,
                         config.num_remote_schedulers,
                         sched_id - config.num_local_schedulers,
                         &my_first_worker,
                         &my_last_worker);
      my_first_worker += config.num_workers;
      my_last_worker += config.num_workers;
    }

    size_t cur_event_pos0 = 0, last_event_pos0 = 0;
    size_t cur_event_pos1 = 0, last_event_pos1 = 0;

    __shared__ size_t worker_queue_next_free_task_pos[MAX_WORKER_PER_SCHEDULER];
    for (int i = 0; i < MAX_WORKER_PER_SCHEDULER; i++) {
      worker_queue_next_free_task_pos[i] = 0;
    }

    int next_begin_worker = my_first_worker;
    int queue_idx = 0;
    while (true) {
      while (true) {
        if (queue_idx == 0) {
          if (cur_event_pos0 != last_event_pos0) {
            break;
          }
          last_event_pos0 = ld_acquire_gpu_u64(
              &config.sched_queue_last_ready_event_id[sched_queue_id0]);
          if (cur_event_pos0 < last_event_pos0) {
            break;
          }
          if (has_queue1) {
            queue_idx = 1;
          }
        } else {
          if (cur_event_pos1 != last_event_pos1) {
            break;
          }
          last_event_pos1 = ld_acquire_gpu_u64(
              &config.sched_queue_last_ready_event_id[sched_queue_id1]);
          if (cur_event_pos1 < last_event_pos1) {
            break;
          }
          queue_idx = 0;
        }
        __nanosleep(10);
      }

      size_t cur_event_pos = (queue_idx == 0) ? cur_event_pos0 : cur_event_pos1;
      size_t last_event_pos =
          (queue_idx == 0) ? last_event_pos0 : last_event_pos1;
      assert(cur_event_pos + config.per_sched_queue_len > last_event_pos);

      EventId *sched_queue = (queue_idx == 0) ? sched_queue0 : sched_queue1;
      EventId event_id = ld_relaxed_gpu_u64(
          &sched_queue[cur_event_pos % config.per_sched_queue_len]);
      if (is_termination_event(event_id)) {
        if (sched_id < config.num_local_schedulers) {
          for (int i = my_first_worker; i < my_last_worker; i++) {
            size_t last_task_id =
                worker_queue_next_free_task_pos[i - my_first_worker]++;
            st_relaxed_gpu_u64(
                &config.worker_queues[i][last_task_id %
                                         config.per_worker_queue_len],
                0);
            atom_add_release_gpu_u64(&config.worker_queue_last_ready_task_id[i],
                                     1);
          }
        }
        return;
      }
      EventDesc const &e = config.all_events[event_id];
      if (e.event_type == EVENT_END_OF_TASK_GRAPH) {
#ifdef MODE_ONEPASS
        bool const continue_running = (iteration_num == 0);
#else
        bool const continue_running = prepare_next_batch(config);
#endif
        if (!continue_running) {
          terminate_schedulers(config);
        } else {
          int const begin_worker = next_begin_worker;
          size_t last_task_id =
              worker_queue_next_free_task_pos[begin_worker - my_first_worker]++;
          st_relaxed_gpu_u64(
              &config.worker_queues[begin_worker]
                                   [last_task_id % config.per_worker_queue_len],
              compute_task_id(iteration_num + 1, 1 /*begin_task_graph*/));
          atom_add_release_gpu_u64(
              &config.worker_queue_last_ready_task_id[begin_worker], 1);
#ifdef MPK_ENABLE_VERBOSE
          printf("[%d][SCHD]EVENT_END_OF_TASK_GRAPH schd_id(%d) "
                 "iter_num(%llu) task_idx(1) "
                 "worker_id(%d) "
                 "worker_last_ready_pos(%llu)\n",
                 config.my_gpu_id,
                 sched_id,
                 iteration_num + 1,
                 begin_worker,
                 last_task_id + 1);
#endif
          if (next_begin_worker == my_last_worker - 1) {
            next_begin_worker = my_first_worker;
          } else {
            next_begin_worker++;
          }
        }
      } else if (e.event_type == EVENT_LAUNCH_DEPENDENT_TASKS) {
        iteration_num = iteration_num + 1;
        assert(sched_id < config.num_local_schedulers);
        unsigned long long my_first_task = e.first_task_id;
        unsigned long long my_last_task = e.last_task_id;
        int const sched_count = config.num_local_schedulers;
        int const sched_index = sched_id;
        int const num_workers_this_sched =
            static_cast<int>(my_last_worker - my_first_worker);
        if (num_workers_this_sched > 0 && my_last_task > my_first_task) {
          unsigned long long kernel_pos = my_first_task;
          while (kernel_pos < my_last_task) {
            TaskDesc const &kernel_desc = config.all_tasks[kernel_pos];
            unsigned long long kernel_begin = kernel_desc.kernel_begin_task_id;
            unsigned long long kernel_end = kernel_desc.kernel_end_task_id;
            if (kernel_begin == TASK_INVALID_ID ||
                kernel_end == TASK_INVALID_ID ||
                kernel_end <= kernel_begin) {
              kernel_begin = kernel_pos;
              kernel_end = kernel_pos + 1;
            }
            if (kernel_begin < my_first_task) {
              kernel_begin = my_first_task;
            }
            if (kernel_end > my_last_task) {
              kernel_end = my_last_task;
            }
            if (kernel_end <= kernel_begin) {
              kernel_pos = (kernel_end > kernel_pos) ? kernel_end : (kernel_pos + 1);
              continue;
            }
            unsigned long long const total_tasks = kernel_end - kernel_begin;
            unsigned long long const k = 2ull;
            unsigned long long const total_blocks = total_tasks / k;
            unsigned long long const rem_tasks = total_tasks - total_blocks * k;
            unsigned long long const blocks_base =
                total_blocks / static_cast<unsigned long long>(sched_count);
            unsigned long long const blocks_rem =
                total_blocks % static_cast<unsigned long long>(sched_count);
            unsigned long long const sched_blocks =
                blocks_base +
                (static_cast<unsigned long long>(sched_index) < blocks_rem ? 1ull : 0ull);
            unsigned long long const blocks_before =
                blocks_base * static_cast<unsigned long long>(sched_index) +
                min(static_cast<unsigned long long>(sched_index), blocks_rem);
            unsigned long long const extra_before =
                min(static_cast<unsigned long long>(sched_index), rem_tasks);
            unsigned long long const sched_extra =
                (static_cast<unsigned long long>(sched_index) < rem_tasks) ? 1ull : 0ull;
            unsigned long long const sched_begin =
                kernel_begin + blocks_before * k + extra_before;
            unsigned long long const sched_end =
                min(kernel_end, sched_begin + sched_blocks * k + sched_extra);
            if (sched_begin >= sched_end) {
              kernel_pos = kernel_end;
              continue;
            }
            size_t worker_counts[MAX_WORKER_PER_SCHEDULER] = {0};
            unsigned long long const sched_tasks = sched_end - sched_begin;
            unsigned long long const worker_chunk =
                (sched_tasks + static_cast<unsigned long long>(num_workers_this_sched) - 1) /
                static_cast<unsigned long long>(num_workers_this_sched);
            for (int w = 0; w < num_workers_this_sched; ++w) {
              unsigned long long const w_begin =
                  sched_begin + worker_chunk * static_cast<unsigned long long>(w);
              unsigned long long const w_end =
                  min(sched_end, w_begin + worker_chunk);
              worker_counts[w] =
                  (w_end > w_begin) ? static_cast<size_t>(w_end - w_begin) : 0;
            }
            size_t worker_base[MAX_WORKER_PER_SCHEDULER] = {0};
            for (int w = 0; w < num_workers_this_sched; ++w) {
              if (worker_counts[w] == 0) {
                continue;
              }
              worker_base[w] = worker_queue_next_free_task_pos[w];
              worker_queue_next_free_task_pos[w] += worker_counts[w];
            }
            for (int w = 0; w < num_workers_this_sched; ++w) {
              worker_counts[w] = 0;
            }
            for (int w = 0; w < num_workers_this_sched; ++w) {
              unsigned long long const w_begin =
                  sched_begin + worker_chunk * static_cast<unsigned long long>(w);
              unsigned long long const w_end =
                  min(sched_end, w_begin + worker_chunk);
              if (w_begin >= w_end) {
                continue;
              }
              int const worker_id = my_first_worker + w;
              size_t const base = worker_base[w];
              for (unsigned long long task_pos = w_begin; task_pos < w_end;
                   ++task_pos) {
                size_t const last_task_id = base + worker_counts[w]++;
#if MIRAGE_SCHED_LOG
              {
                TaskDesc const &desc = config.all_tasks[task_pos];
                if (desc.task_type == TASK_LINEAR_WITH_RESIDUAL) {
                TaskId const task_id =
                    compute_task_id(iteration_num, task_pos);
                unsigned long long const task_in_evt =
                    task_pos -
                    static_cast<unsigned long long>(e.first_task_id);
                printf("[SCHD][ASSIGN] sched=%d first_worker=%llu evt=%llu type=%d variant=%u task=%llu iter=%llu task_pos=%llu task_in_evt=%llu worker=%d\n",
                       sched_id,
                       my_first_worker,
                       (unsigned long long)event_id,
                       static_cast<int>(desc.task_type),
                       static_cast<unsigned>(desc.variant_id),
                       (unsigned long long)task_id,
                       (unsigned long long)iteration_num,
                       (unsigned long long)task_pos,
                       (unsigned long long)task_in_evt,
                       worker_id);
                }
              }
#endif
              st_relaxed_gpu_u64(
                  &config.worker_queues[worker_id]
                                       [last_task_id %
                                        config.per_worker_queue_len],
                  compute_task_id(iteration_num, task_pos));
              }
            }
            for (int w = 0; w < num_workers_this_sched; ++w) {
              if (worker_counts[w] == 0) {
                continue;
              }
              atom_add_release_gpu_u64(
                  &config.worker_queue_last_ready_task_id[my_first_worker + w],
                  worker_counts[w]);
            }
            kernel_pos = kernel_end;
          }
        }
      } else {
        TaskId my_first_task = e.first_task_id, my_last_task = e.last_task_id;
        if (e.event_type == EVENT_LAUNCH_MASSIVE_TASKS) {
          assert(sched_id < config.num_local_schedulers);
          my_first_task = e.first_task_id;
          my_last_task = e.last_task_id;
#if MIRAGE_SCHED_LOG
          {
            printf("[SCHD][ASSIGN] EVENT_LAUNCH_MASSIVE_TASKS sched=%d first_task_id=%llu last_task_id=%llu\n",
                   sched_id,
                   (unsigned long long)e.first_task_id,
                   (unsigned long long)e.last_task_id);
          }
#endif
        }
        int const num_workers_this_sched =
            static_cast<int>(my_last_worker - my_first_worker);
        int const sched_count = (sched_id < config.num_local_schedulers)
                                    ? config.num_local_schedulers
                                    : config.num_remote_schedulers;
        int const sched_index =
            (sched_id < config.num_local_schedulers)
                ? sched_id
                : (sched_id - config.num_local_schedulers);
        if (num_workers_this_sched > 0 && my_last_task > my_first_task) {
          unsigned long long kernel_pos =
              static_cast<unsigned long long>(my_first_task);
          unsigned long long kernel_limit =
              static_cast<unsigned long long>(my_last_task);
          while (kernel_pos < kernel_limit) {
            TaskDesc const &kernel_desc = config.all_tasks[kernel_pos];
            unsigned long long kernel_begin = kernel_desc.kernel_begin_task_id;
            unsigned long long kernel_end = kernel_desc.kernel_end_task_id;
            if (kernel_begin == TASK_INVALID_ID ||
                kernel_end == TASK_INVALID_ID ||
                kernel_end <= kernel_begin) {
              kernel_begin = kernel_pos;
              kernel_end = kernel_pos + 1;
            }
            if (kernel_begin < static_cast<unsigned long long>(my_first_task)) {
              kernel_begin = static_cast<unsigned long long>(my_first_task);
            }
            if (kernel_end > kernel_limit) {
              kernel_end = kernel_limit;
            }
            if (kernel_end <= kernel_begin) {
              kernel_pos = (kernel_end > kernel_pos) ? kernel_end : (kernel_pos + 1);
              continue;
            }
            unsigned long long const total_tasks = kernel_end - kernel_begin;
            unsigned long long const k = 2ull;
            unsigned long long const total_blocks = total_tasks / k;
            unsigned long long const rem_tasks = total_tasks - total_blocks * k;
            unsigned long long const blocks_base =
                total_blocks / static_cast<unsigned long long>(sched_count);
            unsigned long long const blocks_rem =
                total_blocks % static_cast<unsigned long long>(sched_count);
            unsigned long long const sched_blocks =
                blocks_base +
                (static_cast<unsigned long long>(sched_index) < blocks_rem ? 1ull : 0ull);
            unsigned long long const blocks_before =
                blocks_base * static_cast<unsigned long long>(sched_index) +
                min(static_cast<unsigned long long>(sched_index), blocks_rem);
            unsigned long long const extra_before =
                min(static_cast<unsigned long long>(sched_index), rem_tasks);
            unsigned long long const sched_extra =
                (static_cast<unsigned long long>(sched_index) < rem_tasks) ? 1ull : 0ull;
            unsigned long long const sched_begin =
                kernel_begin + blocks_before * k + extra_before;
            unsigned long long const sched_end =
                min(kernel_end, sched_begin + sched_blocks * k + sched_extra);
            if (sched_begin >= sched_end) {
              kernel_pos = kernel_end;
              continue;
            }
            size_t worker_counts[MAX_WORKER_PER_SCHEDULER] = {0};
            unsigned long long const sched_tasks = sched_end - sched_begin;
            unsigned long long const worker_chunk =
                (sched_tasks + static_cast<unsigned long long>(num_workers_this_sched) - 1) /
                static_cast<unsigned long long>(num_workers_this_sched);
            for (int w = 0; w < num_workers_this_sched; ++w) {
              unsigned long long const w_begin =
                  sched_begin + worker_chunk * static_cast<unsigned long long>(w);
              unsigned long long const w_end =
                  min(sched_end, w_begin + worker_chunk);
              worker_counts[w] =
                  (w_end > w_begin) ? static_cast<size_t>(w_end - w_begin) : 0;
            }
            size_t worker_base[MAX_WORKER_PER_SCHEDULER] = {0};
            for (int w = 0; w < num_workers_this_sched; ++w) {
              if (worker_counts[w] == 0) {
                continue;
              }
              worker_base[w] = worker_queue_next_free_task_pos[w];
              worker_queue_next_free_task_pos[w] += worker_counts[w];
            }
            for (int w = 0; w < num_workers_this_sched; ++w) {
              worker_counts[w] = 0;
            }
            for (int w = 0; w < num_workers_this_sched; ++w) {
              unsigned long long const w_begin =
                  sched_begin + worker_chunk * static_cast<unsigned long long>(w);
              unsigned long long const w_end =
                  min(sched_end, w_begin + worker_chunk);
              if (w_begin >= w_end) {
                continue;
              }
              int const worker_id = my_first_worker + w;
              size_t const base = worker_base[w];
              for (unsigned long long task_pos = w_begin; task_pos < w_end;
                   ++task_pos) {
                size_t const last_task_id = base + worker_counts[w]++;
#if MIRAGE_SCHED_LOG
              {
                TaskDesc const &desc = config.all_tasks[task_pos];
                if (desc.task_type == TASK_LINEAR_WITH_RESIDUAL) {
                TaskId const task_id =
                    compute_task_id(iteration_num, task_pos);
                unsigned long long const task_in_evt =
                    task_pos -
                    static_cast<unsigned long long>(e.first_task_id);
                printf("[SCHD][ASSIGN] sched=%d first_worker=%llu evt=%llu type=%d variant=%u task=%llu iter=%llu task_pos=%llu task_in_evt=%llu worker=%d\n",
                       sched_id,
                       my_first_worker,
                       (unsigned long long)event_id,
                       static_cast<int>(desc.task_type),
                       static_cast<unsigned>(desc.variant_id),
                       (unsigned long long)task_id,
                       (unsigned long long)iteration_num,
                       (unsigned long long)task_pos,
                       (unsigned long long)task_in_evt,
                       worker_id);
                }
              }
#endif
              st_relaxed_gpu_u64(
                  &config.worker_queues[worker_id]
                                       [last_task_id %
                                        config.per_worker_queue_len],
                  compute_task_id(iteration_num, task_pos));
              }
            }
            for (int w = 0; w < num_workers_this_sched; ++w) {
              if (worker_counts[w] == 0) {
                continue;
              }
              atom_add_release_gpu_u64(
                  &config.worker_queue_last_ready_task_id[my_first_worker + w],
                  worker_counts[w]);
            }
            kernel_pos = kernel_end;
          }
        }
      }
      if (queue_idx == 0) {
        cur_event_pos0 += 1;
      } else {
        cur_event_pos1 += 1;
      }
    }
  }
}

// Batched scheduler variant:
// - For each worker, reserve a contiguous slot range once, then let the whole
//   warp cooperatively write the task IDs (scheme B: warp-parallel writes).
// - Publish the batch with a single release increment of last_ready.
__device__ __forceinline__ void __mirage_sched_publish_tasks_to_worker_warp(
    RuntimeConfig const &config,
    size_t *next_free_pos,
    int lane,
    int my_first_worker,
    int worker_id,
    unsigned long long iter_num,
    unsigned long long task_pos_begin,
    unsigned long long task_count) {
  if (task_count == 0) {
    return;
  }

  unsigned long long base_slot = 0;
  int const worker_slot = worker_id - my_first_worker;
  if (lane == 0) {
    base_slot = next_free_pos[worker_slot];
    next_free_pos[worker_slot] = base_slot + task_count;
  }
  base_slot = __shfl_sync(0xffffffff, base_slot, 0);

  TaskId *q = config.worker_queues[worker_id];
  unsigned long long const qlen =
      static_cast<unsigned long long>(config.per_worker_queue_len);

  for (unsigned long long t = static_cast<unsigned long long>(lane);
       t < task_count;
       t += 32ull) {
    unsigned long long const pos = base_slot + t;
    unsigned long long const task_pos = task_pos_begin + t;
    st_relaxed_gpu_u64(&q[pos % qlen], compute_task_id(iter_num, task_pos));
  }
  __syncwarp();

  if (lane == 0) {
    atom_add_release_gpu_u64(&config.worker_queue_last_ready_task_id[worker_id],
                             task_count);
  }
  __syncwarp();
}

// Balanced scheduler variant:
// - If task_count < num_workers: strict round-robin (0/1 per worker).
// - If num_workers <= task_count < 2*num_workers: one task each, then give the
//   remaining tasks as the "second task" to the first N workers.
// - If task_count >= 2*num_workers: contiguous blocks, with block sizes aligned
//   to 2 (even) as much as possible to favor 2-group pairing.
__device__ __forceinline__ void __mirage_sched_distribute_balanced_to_workers_warp(
    RuntimeConfig const &config,
    size_t *next_free_pos,
    int lane,
    int my_first_worker,
    int my_last_worker,
    unsigned long long iter_num,
    unsigned long long first_task_pos,
    unsigned long long last_task_pos_exclusive) {
  unsigned long long total = 0;
  if (last_task_pos_exclusive > first_task_pos) {
    total = last_task_pos_exclusive - first_task_pos;
  }
  int const num_workers_this_sched = my_last_worker - my_first_worker;
  if (num_workers_this_sched <= 0 || total == 0) {
    return;
  }

  if (total < static_cast<unsigned long long>(num_workers_this_sched)) {
    for (unsigned long long t = 0; t < total; ++t) {
      int const worker_id = my_first_worker + static_cast<int>(t);
      __mirage_sched_publish_tasks_to_worker_warp(config,
                                                  next_free_pos,
                                                  lane,
                                                  my_first_worker,
                                                  worker_id,
                                                  iter_num,
                                                  first_task_pos + t,
                                                  1ull);
    }
    return;
  }

  if (total < static_cast<unsigned long long>(num_workers_this_sched * 2)) {
    for (int w = 0; w < num_workers_this_sched; ++w) {
      int const worker_id = my_first_worker + w;
      __mirage_sched_publish_tasks_to_worker_warp(config,
                                                  next_free_pos,
                                                  lane,
                                                  my_first_worker,
                                                  worker_id,
                                                  iter_num,
                                                  first_task_pos +
                                                      static_cast<unsigned long long>(w),
                                                  1ull);
    }
    unsigned long long rem =
        total - static_cast<unsigned long long>(num_workers_this_sched);
    for (unsigned long long w = 0; w < rem; ++w) {
      int const worker_id = my_first_worker + static_cast<int>(w);
      __mirage_sched_publish_tasks_to_worker_warp(config,
                                                  next_free_pos,
                                                  lane,
                                                  my_first_worker,
                                                  worker_id,
                                                  iter_num,
                                                  first_task_pos +
                                                      static_cast<unsigned long long>(num_workers_this_sched) +
                                                      w,
                                                  1ull);
    }
    return;
  }

  unsigned long long const base =
      total / static_cast<unsigned long long>(num_workers_this_sched);
  unsigned long long const base_even = (base / 2ull) * 2ull;
  unsigned long long remaining =
      total - base_even * static_cast<unsigned long long>(num_workers_this_sched);
  unsigned long long extra_pairs = remaining / 2ull;
  unsigned long long extra_odd = remaining & 1ull;

  unsigned long long cur = first_task_pos;
  for (int w = 0; w < num_workers_this_sched; ++w) {
    unsigned long long cnt = base_even;
    if (extra_pairs > 0) {
      cnt += 2ull;
      extra_pairs -= 1ull;
    } else if (extra_odd > 0) {
      cnt += 1ull;
      extra_odd = 0ull;
    }
    int const worker_id = my_first_worker + w;
    if (cnt != 0ull) {
      __mirage_sched_publish_tasks_to_worker_warp(config,
                                                  next_free_pos,
                                                  lane,
                                                  my_first_worker,
                                                  worker_id,
                                                  iter_num,
                                                  cur,
                                                  cnt);
      cur += cnt;
    }
  }
}

// __global__ __launch_bounds__(WORKER_NUM_THREADS,
//                              1) void persistent_kernel(RuntimeConfig config) {
//   persistent_checker(config);
//   if (blockIdx.x < config.num_workers) {
//     execute_worker(config);
//   } else {
//     execute_scheduler(config, -(4 * config.num_workers));
//   }
// }

__global__ __launch_bounds__(WORKER_NUM_THREADS * 2,
                             1) void worker_kernel(RuntimeConfig config) {
  worker_checker(config);
  execute_worker_multi_group_aligned(config);
  // execute_worker(config);
}

__global__ void scheduler_kernel(RuntimeConfig config) {
  scheduler_checker(config);
  // execute_scheduler(config, 0);
  execute_scheduler_balanced(config, 0);
}

template <typename DT>
DT *gpu_malloc(size_t size) {
  void *dst_ptr;
#ifdef USE_NVSHMEM
  dst_ptr = nvshmem_malloc(size);
#else
  cudaMalloc(&dst_ptr, size);
#endif
  return static_cast<DT *>(dst_ptr);
}

void gpu_free(void *ptr) {
#ifdef USE_NVSHMEM
  nvshmem_free(ptr);
#else
  cudaFree(ptr);
#endif
}

// The following function will be generated by the transpiler
static void _init_persistent_kernel(std::vector<FullTaskDesc> &all_tasks,
                                    std::vector<EventDesc> &all_events,
                                    std::vector<TaskId> &first_tasks,
                                    int num_gpus,
                                    int my_gpu_id);

static RuntimeConfig global_runtime_config;

// meta_tensors[0]: seq_length
// meta_tensors[1]: tokens
// meta_tensors[2]: input_tokens
// meta_tensors[3]: output_tokens
// meta_tensors[4]: new_tokens_nums
// meta_tensors[5]: prompt_length
// meta_tensors[6]: qo_indptr_buffer
// meta_tensors[7]: paged_kv_indptr_buffer
// meta_tensors[8]: paged_kv_indices_buffer
// meta_tensors[9]: paged_kv_last_page_len_buffer

extern "C" void init_persistent_kernel(std::vector<void *> meta_tensors,
                                       void *profiler_buffer,
                                       int my_rank,
                                       int num_workers,
                                       int num_local_schedulers,
                                       int num_remote_schedulers,
                                       int max_seq_length,
                                       int total_num_requests,
                                       long long eos_token_id) {
  assert(meta_tensors.size() == 10);
  global_runtime_config.step = static_cast<int *>(meta_tensors[0]);
  global_runtime_config.tokens = static_cast<long long *>(meta_tensors[1]);
  global_runtime_config.input_tokens =
      static_cast<long long *>(meta_tensors[2]);
  global_runtime_config.output_tokens =
      static_cast<long long *>(meta_tensors[3]);
  global_runtime_config.new_token_nums = static_cast<int *>(meta_tensors[4]);
  global_runtime_config.qo_indptr_buffer = static_cast<int *>(meta_tensors[6]);
  global_runtime_config.paged_kv_indptr_buffer =
      static_cast<int *>(meta_tensors[7]);
  global_runtime_config.paged_kv_indices_buffer =
      static_cast<int *>(meta_tensors[8]);
  global_runtime_config.paged_kv_last_page_len_buffer =
      static_cast<int *>(meta_tensors[9]);
  global_runtime_config.num_workers = num_workers;
  global_runtime_config.num_local_schedulers = num_local_schedulers;
  global_runtime_config.num_remote_schedulers = num_remote_schedulers;
  global_runtime_config.max_seq_length = max_seq_length;
  global_runtime_config.eos_token_id = eos_token_id;
  global_runtime_config.profiler_buffer = profiler_buffer;
  int num_schedulers = num_local_schedulers + num_remote_schedulers;

  // Initialize nvshmem
  cudaSetDevice(my_rank);

#ifdef USE_NVSHMEM
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  nvshmem_barrier_all();
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  printf("mype(%d) npes(%d) mype_node(%d)\n", mype, npes, mype_node);
#else
  int mype = 0;
  int npes = 1;
#endif

#if defined(MODE_OFFLINE) || defined(MODE_ONLINE)
  global_runtime_config.prompt_length = static_cast<int *>(meta_tensors[5]);
  global_runtime_config.request_ids =
      gpu_malloc<int>(sizeof(int) * (MPK_MAX_NUM_BATCHED_REQUESTS + 1));
  global_runtime_config.next_request_id = gpu_malloc<int>(sizeof(int));
  global_runtime_config.page_queue =
      gpu_malloc<int>(MPK_MAX_NUM_PAGES * sizeof(int));
  global_runtime_config.page_queue_head = gpu_malloc<int>(sizeof(int));
  global_runtime_config.page_queue_tail = gpu_malloc<int>(sizeof(int));
  global_runtime_config.total_num_requests = total_num_requests;
#endif
  global_runtime_config.per_worker_queue_len = 1024;
  global_runtime_config.per_sched_queue_len = 1024;
  global_runtime_config.num_gpus = npes;
  global_runtime_config.my_gpu_id = mype;
  global_runtime_config.num_graphs = 1;
  global_runtime_config.split_worker_scheduler = true;
  // Runtime reg targets (device code reads from RuntimeConfig::reg_targets).
  // Indices match __mirage_reg_slot (defined under MIRAGE_GRACE_HOPPER).
  // global_runtime_config.reg_targets[0] = MIRAGE_IDLE_REG_TARGET;
  // global_runtime_config.reg_targets[1] = MIRAGE_FETCH_REG_TARGET;
  // global_runtime_config.reg_targets[2] = MIRAGE_TRIGGER_REG_TARGET;
  // global_runtime_config.reg_targets[3] = MIRAGE_PARK_REG_TARGET;

  std::vector<FullTaskDesc> all_fulltasks;
  std::vector<EventDesc> all_events;
  std::vector<TaskId> first_tasks;
  _init_persistent_kernel(all_fulltasks, all_events, first_tasks, npes, mype);
  std::vector<TaskDesc> all_tasks;
  for (auto const &ft : all_fulltasks) {
    TaskDesc task_desc(ft);
    // if (ft.task_type == TASK_PAGED_ATTENTION_SPLIT_KV_SM100 || ft.task_type
    // == TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100) {
    //   printf("ft.kv_idx %d\n", ft.kv_idx);
    //   printf("ft.merge_task_offset %d\n", ft.merge_task_offset);
    // }
    // Reinterpret part of TaskDesc to save xfer_size information
    if (ft.task_type == TASK_NVSHMEM_COPY) {
      int size_in_bytes = 2;
      for (int i = 0; i < ft.inputs[0].num_dims; i++) {
        size_in_bytes *= ft.inputs[0].dim[i];
      }
      task_desc.task_metadata.xfer_size_in_bytes = size_in_bytes;
    }
    all_tasks.push_back(task_desc);
  }

  // Initialize worker queue last task id
  // Each worker now maintains a local and a remote worker queue
  global_runtime_config.worker_queue_last_ready_task_id =
      gpu_malloc<unsigned long long int>((num_workers * 2) *
                                         sizeof(unsigned long long int));
  // std::vector<unsigned long long int> host_worker_queue_last_task_id;
  // for (int i = 0; i < 2 * num_workers; i++) {
  //   host_worker_queue_last_task_id.push_back(0);
  // }
  // cudaMemcpy(global_runtime_config.worker_queue_last_ready_task_id,
  //            host_worker_queue_last_task_id.data(),
  //            (num_workers * 2) * sizeof(unsigned long long int),
  //            cudaMemcpyHostToDevice);
  //  Initialize scheduler queue last event id
  //  We maintain one extra scheduler queue for the global scheduler
  global_runtime_config.sched_queue_last_ready_event_id =
      gpu_malloc<unsigned long long int>((num_schedulers + 1) *
                                         sizeof(unsigned long long int));
  global_runtime_config.sched_queue_next_free_event_id =
      gpu_malloc<unsigned long long int>((num_schedulers + 1) *
                                         sizeof(unsigned long long int));

  // std::vector<unsigned long long int> host_sched_queue_last_event_id;
  // for (int i = 0; i < (num_schedulers + 1); i++) {
  //   host_sched_queue_last_event_id.push_back(0);
  // }
  // cudaMemcpy(global_runtime_config.sched_queue_last_ready_event_id,
  //            host_sched_queue_last_event_id.data(),
  //            (num_schedulers + 1) * sizeof(unsigned long long int),
  //            cudaMemcpyHostToDevice);
  // cudaMemcpy(global_runtime_config.sched_queue_next_free_event_id,
  //            host_sched_queue_last_event_id.data(),
  //            (num_schedulers + 1) * sizeof(unsigned long long int),
  //            cudaMemcpyHostToDevice);
  //  Initialize all event counters
  global_runtime_config.all_event_counters =
      gpu_malloc<EventCounter>(all_events.size() * sizeof(EventCounter));
  global_runtime_config.all_event_num_triggers =
      gpu_malloc<int>(all_events.size() * sizeof(int));
  std::vector<int> host_all_event_counters;
  for (size_t i = 0; i < all_events.size(); i++) {
    host_all_event_counters.push_back(all_events.at(i).num_triggers);
  }
  cudaMemcpy(global_runtime_config.all_event_num_triggers,
             host_all_event_counters.data(),
             all_events.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  // cudaMemset(global_runtime_config.all_event_counters,
  //            0,
  //            all_events.size() * sizeof(EventCounter));
  //  Initialize all tasks
  global_runtime_config.all_tasks =
      gpu_malloc<TaskDesc>(all_tasks.size() * sizeof(TaskDesc));
  cudaMemcpy(global_runtime_config.all_tasks,
             all_tasks.data(),
             all_tasks.size() * sizeof(TaskDesc),
             cudaMemcpyHostToDevice);
  // Initialize all events
  global_runtime_config.num_events = (int)all_events.size();
  global_runtime_config.all_events =
      gpu_malloc<EventDesc>(all_events.size() * sizeof(EventDesc));
  cudaMemcpy(global_runtime_config.all_events,
             all_events.data(),
             all_events.size() * sizeof(EventDesc),
             cudaMemcpyHostToDevice);
  // Initialize worker queues
  {
    std::vector<TaskId *> host_worker_queues;
    for (int i = 0; i < (num_workers * 2); i++) {
      TaskId *worker_queue = gpu_malloc<TaskId>(
          global_runtime_config.per_worker_queue_len * sizeof(TaskId));
      host_worker_queues.push_back(worker_queue);
    }
    global_runtime_config.worker_queues =
        gpu_malloc<TaskId *>((num_workers * 2) * sizeof(TaskId *));
    cudaMemcpy(global_runtime_config.worker_queues,
               host_worker_queues.data(),
               (num_workers * 2) * sizeof(TaskId *),
               cudaMemcpyHostToDevice);
  }
  // Initialize scheduler queues
  {
    std::vector<EventId *> host_sched_queues;
    for (int i = 0; i < (num_schedulers + 1); i++) {
      EventId *sched_queue = gpu_malloc<EventId>(
          global_runtime_config.per_sched_queue_len * sizeof(EventId));
      host_sched_queues.push_back(sched_queue);
    }
    global_runtime_config.sched_queues =
        gpu_malloc<EventId *>((num_schedulers + 1) * sizeof(EventId *));
    cudaMemcpy(global_runtime_config.sched_queues,
               host_sched_queues.data(),
               (num_schedulers + 1) * sizeof(EventId *),
               cudaMemcpyHostToDevice);
  }
  // Initialize first tasks
  {
    global_runtime_config.first_tasks =
        gpu_malloc<TaskId>(first_tasks.size() * sizeof(TaskId));
    cudaMemcpy(global_runtime_config.first_tasks,
               first_tasks.data(),
               first_tasks.size() * sizeof(TaskId),
               cudaMemcpyHostToDevice);
  }

  // Set configuration for kernels
  cudaFuncSetAttribute(worker_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       MAX_DYNAMIC_SHARED_MEMORY_SIZE);
  cudaFuncSetAttribute(scheduler_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       MAX_DYNAMIC_SHARED_MEMORY_SIZE);
  // cudaFuncSetAttribute(persistent_kernel,
  //                      cudaFuncAttributeMaxDynamicSharedMemorySize,
  //                      MAX_DYNAMIC_SHARED_MEMORY_SIZE);
  global_runtime_config.worker_dynamic_smem_bytes =
      MAX_DYNAMIC_SHARED_MEMORY_SIZE;
  // Create worker and scheduler streams
  cudaStreamCreate(&global_runtime_config.worker_stream);
  cudaStreamCreate(&global_runtime_config.scheduler_stream);

  // launch init kernel
  init_kernel<<<dim3(1, 1, 1), dim3(INIT_NUM_THREADS, 1, 1)>>>(
      global_runtime_config);
  cudaDeviceSynchronize();
#ifdef USE_NVSHMEM
  // Add a global barrier for all init_kernel to complete
  nvshmem_barrier_all();
#endif
}

// Entry point for C/C++
// TODO: change launch config
extern "C" void launch_persistent_kernel() {
  // int device;
  // cudaGetDevice(&device);
  // int sm_count;
  // cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  //  Prepare next persistent kernel by resetting queue pointers
  {
    int end_of_task_graph_event_pos = global_runtime_config.num_events - 1;
    prepare_kernel<<<dim3(global_runtime_config.num_workers, 1, 1),
                     dim3(128, 1, 1)>>>(global_runtime_config,
                                        end_of_task_graph_event_pos);
    cudaDeviceSynchronize();
#ifdef USE_NVSHMEM
    nvshmem_barrier_all();
#endif
  }
  int num_schedulers = global_runtime_config.num_local_schedulers +
                       global_runtime_config.num_remote_schedulers;
	  if (global_runtime_config.split_worker_scheduler) {
    // printf("worker kernel: %d & scheduler kernel: %d\n", global_runtime_config.num_workers, num_schedulers);
    // printf("smem size: %d\n", MAX_DYNAMIC_SHARED_MEMORY_SIZE);
	    int worker_threads = WORKER_NUM_THREADS;
	    worker_threads = WORKER_NUM_THREADS * 2;
	    // printf("worker threads: %d, scheduler threads: %d\n", worker_threads, 32);

      cudaProfilerStart();
      // #define MIRAGE_PROFILE_TIME
#if defined(MIRAGE_PROFILE_TIME)
	    // Precise timing (GPU events) for split worker/scheduler kernels.
	    cudaEvent_t worker_start, worker_end, sched_start, sched_end, total_end;
	    CUDA_CHECK(cudaEventCreate(&worker_start));
	    CUDA_CHECK(cudaEventCreate(&worker_end));
	    CUDA_CHECK(cudaEventCreate(&sched_start));
	    CUDA_CHECK(cudaEventCreate(&sched_end));
	    CUDA_CHECK(cudaEventCreate(&total_end));

	    // The split kernel does not support NVSHMEM because
	    // nvshmemx_collective_launch launches kernels sequentially, which blocks
	    // the interaction between the worker kernel and the scheduler kernel
	    CUDA_CHECK(cudaEventRecord(worker_start, global_runtime_config.worker_stream));
#endif
      // Launch worker kernel
	    worker_kernel<<<dim3(global_runtime_config.num_workers, 1, 1),
	                    dim3(worker_threads, 1, 1),
	                    MAX_DYNAMIC_SHARED_MEMORY_SIZE /*smem*/,
	                    global_runtime_config.worker_stream>>>(
	        global_runtime_config);

#if defined(MIRAGE_PROFILE_TIME)
	    CUDA_CHECK(cudaGetLastError());
	    CUDA_CHECK(cudaEventRecord(worker_end, global_runtime_config.worker_stream));

	    CUDA_CHECK(cudaEventRecord(sched_start, global_runtime_config.scheduler_stream));
#endif

      // Launch scheduler kernel
	    scheduler_kernel<<<dim3(global_runtime_config.num_local_schedulers, 1, 1),
	                       dim3(32, 1, 1),
	                       0 /*smem*/,
	                       global_runtime_config.scheduler_stream>>>(
	        global_runtime_config);

// #if MIRAGE_ADMISSION_DEBUG
//       // Independently track completion of worker/scheduler kernels by stream
//       // events, so we can tell which side is stuck if the final sync hangs.
//       // Rate limit: poll every 1ms for up to 30s.
//       cudaEvent_t worker_done_evt, sched_done_evt;
//       CUDA_CHECK(cudaEventCreateWithFlags(&worker_done_evt, cudaEventDisableTiming));
//       CUDA_CHECK(cudaEventCreateWithFlags(&sched_done_evt, cudaEventDisableTiming));
//       CUDA_CHECK(cudaEventRecord(worker_done_evt, global_runtime_config.worker_stream));
//       CUDA_CHECK(cudaEventRecord(sched_done_evt, global_runtime_config.scheduler_stream));

// 	      bool worker_done = false;
// 	      bool sched_done = false;
// 	      unsigned long long waited_us = 0;
// 	      unsigned worker_prints = 0;
// 	      unsigned sched_prints = 0;
// 	      while (!(worker_done && sched_done) &&
// 	             waited_us < (1000 * 1000ull * 1000ull)) {
// 	        if (!worker_done) {
// 	          cudaError_t q = cudaEventQuery(worker_done_evt);
// 	          if (q == cudaSuccess) {
// 	            worker_done = true;
// 	            printf("[HOST][EVENT] worker_kernel done\n");
// 	          } else if (q != cudaErrorNotReady) {
// 	            printf("[HOST][EVENT] worker_kernel query error: %s\n",
// 	                   cudaGetErrorString(q));
// 	            break;
// 	          }
// 	          else if ((waited_us == 0) || ((++worker_prints % 10u) == 0u)) {
// 	            printf("[HOST][EVENT] worker_kernel not done yet (error: %s, waited=%llu us)\n",
//                      cudaGetErrorString(q),
// 	                   waited_us);
// 	          }
// 	        }
// 	        if (!sched_done) {
// 	          cudaError_t q = cudaEventQuery(sched_done_evt);
// 	          if (q == cudaSuccess) {
// 	            sched_done = true;
// 	            printf("[HOST][EVENT] scheduler_kernel done\n");
// 	          } else if (q != cudaErrorNotReady) {
// 	            printf("[HOST][EVENT] scheduler_kernel query error: %s\n",
// 	                   cudaGetErrorString(q));
// 	            break;
// 	          }
// 	          else if ((waited_us == 0) || ((++sched_prints % 10u) == 0u)) {
// 	            printf("[HOST][EVENT] scheduler_kernel not done yet (waited=%llu us)\n",
// 	                   waited_us);
// 	          }
// 	        }
// 	        if (!(worker_done && sched_done)) {
// 	          usleep(100000u);
// 	          waited_us += 100000ull;
// 	        }
//           printf("continue checking worker/scheduler status...\n");
//           fflush(stdout);
// 	      }
// 	      if (!(worker_done && sched_done)) {
// 	        printf("[HOST][EVENT] timeout after %llu us (worker_done=%d sched_done=%d)\n",
// 	               waited_us,
// 	               (int)worker_done,
// 	               (int)sched_done);
// 	      }
// 	      CUDA_CHECK(cudaEventDestroy(worker_done_evt));
// 	      CUDA_CHECK(cudaEventDestroy(sched_done_evt));
// #endif

#if defined(MIRAGE_PROFILE_TIME)
	    CUDA_CHECK(cudaGetLastError());
	    CUDA_CHECK(cudaEventRecord(sched_end, global_runtime_config.scheduler_stream));

	    // Total time: from worker_start until both kernels complete.
	    CUDA_CHECK(cudaStreamWaitEvent(global_runtime_config.worker_stream, sched_end, 0));
	    CUDA_CHECK(cudaEventRecord(total_end, global_runtime_config.worker_stream));
	    CUDA_CHECK(cudaEventSynchronize(total_end));

	    float worker_ms = 0.0f;
	    float sched_ms = 0.0f;
	    float total_ms = 0.0f;
	    CUDA_CHECK(cudaEventElapsedTime(&worker_ms, worker_start, worker_end));
	    CUDA_CHECK(cudaEventElapsedTime(&sched_ms, sched_start, sched_end));
	    CUDA_CHECK(cudaEventElapsedTime(&total_ms, worker_start, total_end));
	    printf("[TIME][%d workers] worker_kernel=%.3f ms scheduler_kernel=%.3f ms total=%.3f ms\n",
              global_runtime_config.num_workers,
	           (double)worker_ms,
	           (double)sched_ms,
	           (double)total_ms);

	    CUDA_CHECK(cudaEventDestroy(worker_start));
	    CUDA_CHECK(cudaEventDestroy(worker_end));
	    CUDA_CHECK(cudaEventDestroy(sched_start));
	    CUDA_CHECK(cudaEventDestroy(sched_end));
	    CUDA_CHECK(cudaEventDestroy(total_end));
#endif

	    // Keep legacy sync+error check for compatibility with existing call sites.
	    cudaError_t err = cudaDeviceSynchronize();
      cudaProfilerStop();
      if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
      }
// #if MIRAGE_ADMISSION_DEBUG
//       CUDA_CHECK(cudaEventDestroy(worker_done_evt));
//       CUDA_CHECK(cudaEventDestroy(sched_done_evt));
// #endif
      // printf("Finished Launch Persistent Kernel\n");
		  } else {
//     printf("a single persistent kernel\n");
//     int num_sms_to_use = global_runtime_config.num_workers + num_schedulers / 4;
// #ifdef USE_NVSHMEM
//     void *args[] = {&global_runtime_config};
//     nvshmemx_collective_launch((void const *)persistent_kernel,
//                                dim3(num_sms_to_use, 1, 1),
//                                dim3(SINGLE_KERNEL_NUM_THREADS, 1, 1),
//                                args,
//                                MAX_DYNAMIC_SHARED_MEMORY_SIZE /*sharedmem*/,
//                                0 /*stream*/);
// #else
//     persistent_kernel<<<dim3(num_sms_to_use, 1, 1),
//                         dim3(SINGLE_KERNEL_NUM_THREADS, 1, 1),
//                         MAX_DYNAMIC_SHARED_MEMORY_SIZE /*smem*/>>>(
//         global_runtime_config);
// #endif
// 	    cudaError_t err = cudaDeviceSynchronize();
// 	    if (err != cudaSuccess) {
// 	      printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
// 	    }
// 	    printf("Finished Launch Persistent Kernel\n");
// 	    fflush(stdout);
	  }

}

extern "C" void finalize_persistent_kernel() {
  gpu_free(global_runtime_config.worker_queue_last_ready_task_id);
  gpu_free(global_runtime_config.sched_queue_last_ready_event_id);
  gpu_free(global_runtime_config.sched_queue_next_free_event_id);
  gpu_free(global_runtime_config.all_event_counters);
  gpu_free(global_runtime_config.all_event_num_triggers);
  gpu_free(global_runtime_config.all_tasks);
  gpu_free(global_runtime_config.all_events);
#if defined(MODE_OFFLINE) || defined(MODE_ONLINE)
  gpu_free(global_runtime_config.next_request_id);
  gpu_free(global_runtime_config.page_queue);
  gpu_free(global_runtime_config.page_queue_head);
  gpu_free(global_runtime_config.page_queue_tail);
#endif
  int num_workers = global_runtime_config.num_workers;
  std::vector<TaskId *> host_worker_queues(num_workers * 2);
  cudaMemcpy(host_worker_queues.data(),
             global_runtime_config.worker_queues,
             (num_workers * 2) * sizeof(TaskId *),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < 2 * num_workers; i++) {
    gpu_free(host_worker_queues[i]);
  }
  gpu_free(global_runtime_config.worker_queues);
  int num_schedulers = global_runtime_config.num_local_schedulers +
                       global_runtime_config.num_remote_schedulers;
  std::vector<EventId *> host_sched_queues(num_schedulers + 1);
  cudaMemcpy(host_sched_queues.data(),
             global_runtime_config.sched_queues,
             (num_schedulers + 1) * sizeof(EventId *),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < num_schedulers + 1; i++) {
    gpu_free(host_sched_queues[i]);
  }
  gpu_free(global_runtime_config.sched_queues);
  gpu_free(global_runtime_config.first_tasks);
#ifdef USE_NVSHMEM
  nvshmem_barrier_all();
  nvshmem_finalize();
#endif
  // Free worker and scheduler streams
  cudaStreamDestroy(global_runtime_config.worker_stream);
  cudaStreamDestroy(global_runtime_config.scheduler_stream);
}
