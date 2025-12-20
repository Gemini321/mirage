#!/usr/bin/env python3
"""
Patch an MPK-generated megakernel shared library (.so) by rewriting the embedded
device ELF (cubin) metadata for *worker_kernel only*.

This is an unsafe workflow if you DOWN-patch regcount below real usage:
it may lead to undefined behavior at runtime. The goal here is to bypass
launch-time "too many resources requested for launch" checks for experiments.

What this script does:
  1) Locate CUDA fatbin sections in the host .so (.nv_fatbin/.nvFatBinSegment).
  2) Find embedded ELF64 images (cubin) inside those sections.
  3) If a cubin contains a function symbol whose name includes "worker_kernel",
     patch:
        - the corresponding .text.<symbol> section header sh_info regcount
        - the .nv.info regcount attribute (attr=0x2F04) for that symidx
  4) Write a patched .so.
  5) Dump a patched cubin and disassemble it to SASS (nvdisasm).
  6) Optionally compile+run a no-sync CUDA driver launcher to validate that
     cuLaunchKernel no longer returns OUT_OF_RESOURCES for a larger block size.
"""

from __future__ import annotations

import argparse
import os
import re
import io
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable



ELF_MAGIC = b"\x7fELF"
EI_CLASS_64 = 2
EI_DATA_LSB = 1

# ==========================================
# CUDA ELF Attribute ID Definitions
# ==========================================
EIATTR_MAX_THREADS = 0x1b04      # __launch_bounds__ 的上限
EIATTR_MIN_CTA_PER_SM = 0x1804   # __launch_bounds__ 的下限
EIATTR_REG_COUNT = 0x2f04        # 寄存器数量 (我们要修改的目标)
EIATTR_FRAME_SIZE = 0x1104       # Local Memory Frame Size
EIATTR_EXTERN_FRAME_SIZE = 0x1204
EIATTR_REQNTID = 0x1004          # Required Block Dimensions (强限制)
EIATTR_FLAT_CTA_ID = 0x1304
EIATTR_COOP_GROUP = 0x1a04

@dataclass(frozen=True)
class Elf64_Ehdr:
    e_ident: bytes
    e_type: int
    e_machine: int
    e_version: int
    e_entry: int
    e_phoff: int
    e_shoff: int
    e_flags: int
    e_ehsize: int
    e_phentsize: int
    e_phnum: int
    e_shentsize: int
    e_shnum: int
    e_shstrndx: int


@dataclass(frozen=True)
class Elf64_Shdr:
    sh_name: int
    sh_type: int
    sh_flags: int
    sh_addr: int
    sh_offset: int
    sh_size: int
    sh_link: int
    sh_info: int
    sh_addralign: int
    sh_entsize: int


@dataclass(frozen=True)
class Elf64_Sym:
    st_name: int
    st_info: int
    st_other: int
    st_shndx: int
    st_value: int
    st_size: int

    @property
    def st_type(self) -> int:
        return self.st_info & 0x0F


def _cstr_at(buf: bytes, off: int) -> str:
    end = buf.find(b"\x00", off)
    if end < 0:
        end = len(buf)
    return buf[off:end].decode("utf-8", errors="replace")


def _parse_ehdr(buf: bytes, base: int) -> Elf64_Ehdr | None:
    if base + 64 > len(buf):
        return None
    e_ident = buf[base : base + 16]
    if e_ident[:4] != ELF_MAGIC or e_ident[4] != EI_CLASS_64 or e_ident[5] != EI_DATA_LSB:
        return None
    (
        e_type,
        e_machine,
        e_version,
        e_entry,
        e_phoff,
        e_shoff,
        e_flags,
        e_ehsize,
        e_phentsize,
        e_phnum,
        e_shentsize,
        e_shnum,
        e_shstrndx,
    ) = struct.unpack_from("<HHIQQQIHHHHHH", buf, base + 16)
    return Elf64_Ehdr(
        e_ident=e_ident,
        e_type=e_type,
        e_machine=e_machine,
        e_version=e_version,
        e_entry=e_entry,
        e_phoff=e_phoff,
        e_shoff=e_shoff,
        e_flags=e_flags,
        e_ehsize=e_ehsize,
        e_phentsize=e_phentsize,
        e_phnum=e_phnum,
        e_shentsize=e_shentsize,
        e_shnum=e_shnum,
        e_shstrndx=e_shstrndx,
    )


def _parse_shdr(buf: bytes, base: int, off: int) -> Elf64_Shdr:
    (
        sh_name,
        sh_type,
        sh_flags,
        sh_addr,
        sh_offset,
        sh_size,
        sh_link,
        sh_info,
        sh_addralign,
        sh_entsize,
    ) = struct.unpack_from("<IIQQQQIIQQ", buf, base + off)
    return Elf64_Shdr(
        sh_name=sh_name,
        sh_type=sh_type,
        sh_flags=sh_flags,
        sh_addr=sh_addr,
        sh_offset=sh_offset,
        sh_size=sh_size,
        sh_link=sh_link,
        sh_info=sh_info,
        sh_addralign=sh_addralign,
        sh_entsize=sh_entsize,
    )


def _parse_sym(buf: bytes, off: int) -> Elf64_Sym:
    st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack_from(
        "<IBBHQQ", buf, off
    )
    return Elf64_Sym(
        st_name=st_name,
        st_info=st_info,
        st_other=st_other,
        st_shndx=st_shndx,
        st_value=st_value,
        st_size=st_size,
    )


def _find_cuda_bin_dir() -> Path | None:
    # Prefer HPC-style install root candidates (often contains CUDA 12.x+ which
    # is required for sm_90a/sm_100* tooling like nvdisasm).
    root = Path("/APP/u22/ai_x86/CUDA")
    if root.exists():
        for cand in sorted(root.glob("*/bin/nvcc"), key=lambda p: p.parent.parent.name, reverse=True):
            return cand.parent

    # Next prefer explicit environment variables.
    for p in [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"]:
        if not p:
            continue
        cand = Path(p) / "bin" / "nvcc"
        if cand.exists():
            return cand.parent

    # Finally fall back to PATH.
    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).resolve().parent
    return None


def _tool_path(name: str, cuda_bin: Path | None) -> str:
    if cuda_bin is not None:
        cand = cuda_bin / name
        if cand.exists():
            return str(cand)
    return shutil.which(name) or name


def _compute_embedded_elf_size(elf_bytes: bytes) -> int | None:
    eh = _parse_ehdr(elf_bytes, 0)
    if eh is None:
        return None
    if eh.e_shoff == 0 or eh.e_shentsize == 0 or eh.e_shnum == 0:
        return None
    if eh.e_shentsize != 64:
        return None
    sh_table_end = eh.e_shoff + eh.e_shentsize * eh.e_shnum
    # Some CUDA toolchains place the program header table *after* the section
    # header table. nvdisasm/readelf will treat the file as truncated if we
    # don't include it.
    ph_table_end = eh.e_phoff + eh.e_phentsize * eh.e_phnum if eh.e_phoff and eh.e_phentsize else 0
    if sh_table_end > len(elf_bytes):
        return None
    if ph_table_end and ph_table_end > len(elf_bytes):
        return None
    shdrs = [_parse_shdr(elf_bytes, 0, eh.e_shoff + i * eh.e_shentsize) for i in range(eh.e_shnum)]
    max_end = max(sh_table_end, ph_table_end)
    for sh in shdrs:
        end = sh.sh_offset + sh.sh_size
        if end > max_end:
            max_end = end
    if max_end <= 0 or max_end > len(elf_bytes):
        return None
    return int(max_end)


def _iter_embedded_elfs(section_payload: bytes) -> Iterable[tuple[int, int]]:
    """
    Yield (offset, size) for each embedded ELF64 image within `section_payload`.
    """
    i = 0
    while True:
        j = section_payload.find(ELF_MAGIC, i)
        if j < 0:
            return
        # Need at least an ELF header to validate.
        if j + 64 > len(section_payload):
            return
        eh = _parse_ehdr(section_payload, j)
        if eh is None:
            i = j + 4
            continue
        # Compute size using internal section table.
        size = _compute_embedded_elf_size(section_payload[j:])
        if size is None or j + size > len(section_payload):
            i = j + 4
            continue
        yield j, size
        i = j + size


def _patch_cubin_worker_kernel_regcount(
    cubin: bytearray,
    new_reg: int,
    kernel_substr: str,
) -> tuple[bool, list[str]]:
    """
    Patch regcount metadata for symbols whose names contain `kernel_substr`.
    Returns (patched?, patched_symbol_names).
    """
    if not (0 <= new_reg <= 255):
        raise ValueError("regcount must fit in 0..255 for ELF sh_info encoding")
    if new_reg == 0:
        raise ValueError("regcount must be > 0")

    eh = _parse_ehdr(cubin, 0)
    if eh is None:
        return False, []
    if eh.e_shoff == 0 or eh.e_shentsize == 0 or eh.e_shnum == 0:
        return False, []
    if eh.e_shentsize != 64:
        return False, []
    shdrs = [_parse_shdr(cubin, 0, eh.e_shoff + i * eh.e_shentsize) for i in range(eh.e_shnum)]
    if eh.e_shstrndx >= len(shdrs):
        return False, []
    shstr = shdrs[eh.e_shstrndx]
    shstrtab = bytes(cubin[shstr.sh_offset : shstr.sh_offset + shstr.sh_size])
    sec_names = [_cstr_at(shstrtab, sh.sh_name) if sh.sh_name else "" for sh in shdrs]

    def find_section(name: str) -> tuple[int, Elf64_Shdr] | None:
        for idx, sh in enumerate(shdrs):
            if sec_names[idx] == name:
                return idx, sh
        return None

    symtab_s = find_section(".symtab")
    strtab_s = find_section(".strtab")
    if symtab_s is None or strtab_s is None:
        return False, []
    symtab = symtab_s[1]
    strtab = strtab_s[1]
    if symtab.sh_entsize == 0:
        return False, []
    sym_blob = bytes(cubin[symtab.sh_offset : symtab.sh_offset + symtab.sh_size])
    str_blob = bytes(cubin[strtab.sh_offset : strtab.sh_offset + strtab.sh_size])

    # Discover candidate function symbols.
    FUNC = 2
    candidates: list[tuple[int, str]] = []
    count = symtab.sh_size // symtab.sh_entsize
    for sym_idx in range(count):
        off = sym_idx * symtab.sh_entsize
        sym = _parse_sym(sym_blob, off)
        if sym.st_name == 0:
            continue
        if sym.st_type != FUNC:
            continue
        name = _cstr_at(str_blob, sym.st_name)
        if kernel_substr not in name:
            continue
        candidates.append((sym_idx, name))

    if not candidates:
        return False, []

    patched_symbols: list[str] = []

    # Patch sh_info in .text.* section headers.
    for sym_idx, sym_name in candidates:
        # Prefer exact section name .text.<symbol>.
        text_sec = f".text.{sym_name}"
        found = find_section(text_sec)
        if found is None:
            # Fallback: any .text.* whose name contains the symbol name.
            found_idx = None
            for idx, sec_name in enumerate(sec_names):
                if sec_name.startswith(".text.") and sym_name in sec_name:
                    found_idx = idx
                    break
            if found_idx is None:
                continue
            found = (found_idx, shdrs[found_idx])
        idx, sh = found

        old_info = sh.sh_info
        new_info = (old_info & 0x00FFFFFF) | (new_reg << 24)
        shoff = eh.e_shoff + idx * eh.e_shentsize
        sh_info_off = shoff + 44  # offsetof(Elf64_Shdr, sh_info)
        struct.pack_into("<I", cubin, sh_info_off, new_info)
        patched_symbols.append(sym_name)

    if not patched_symbols:
        return False, []

    # Patch .nv.info regcount entries (attr=0x2F04, payload=(symidx, reg)).
    # Also patch max threads per CTA (__launch_bounds__) to avoid launch-time
    # invalid-argument when the driver enforces the encoded upper bound.
    nvinfo_s = find_section(".nv.info")
    if nvinfo_s is not None:
        _, nvinfo = nvinfo_s
        blob = bytes(cubin[nvinfo.sh_offset : nvinfo.sh_offset + nvinfo.sh_size])
        num_reg = 65536  # H100 register file size (32-bit registers) per SM
        max_threads = (num_reg // new_reg) // 32 * 32
        max_threads = max(32, min(1024, max_threads))
        off = 0
        while off + 4 <= len(blob):
            attr, size = struct.unpack_from("<HH", blob, off)
            payload_off = off + 4
            payload = blob[payload_off : payload_off + size]
            next_off = (payload_off + size + 3) & ~3
            if size == 8 and (attr == EIATTR_REG_COUNT or attr == EIATTR_MAX_THREADS):
                symidx, _old_reg = struct.unpack_from("<II", payload, 0)
                for cand_symidx, _ in candidates:
                    if symidx == cand_symidx:
                        write_off = nvinfo.sh_offset + payload_off + 4  # offsetof(val) in (symidx,val)
                        if attr == EIATTR_REG_COUNT:
                            struct.pack_into("<I", cubin, write_off, new_reg)
                        else:
                            struct.pack_into("<I", cubin, write_off, max_threads)
                        break
            off = next_off

    return True, patched_symbols

def analyze_symidx_names(elf_bytes):
    """
    Robustly analyze .nv.info section using TLV parsing.
    Correctly identifies Tags, Sizes, and Symbol Indices even with variable-length attributes.
    """
    print("\n[DEBUG] --- Robust Analyzing .nv.info ---")
    
    eh = _parse_ehdr(elf_bytes, 0)
    if not eh:
        print("Failed to parse ELF header.")
        return

    shdrs = [_parse_shdr(elf_bytes, 0, eh.e_shoff + i * eh.e_shentsize) for i in range(eh.e_shnum)]
    shstr = shdrs[eh.e_shstrndx]
    shstrtab = bytes(elf_bytes[shstr.sh_offset : shstr.sh_offset + shstr.sh_size])
    sec_names = [_cstr_at(shstrtab, sh.sh_name) if sh.sh_name else "" for sh in shdrs]

    # 定位符号表和字符串表
    try:
        symtab_idx = next(i for i, n in enumerate(sec_names) if n == ".symtab")
        strtab_idx = next(i for i, n in enumerate(sec_names) if n == ".strtab")
        nvinfo_idx = next(i for i, n in enumerate(sec_names) if n == ".nv.info")
    except StopIteration:
        print("Required sections (.symtab, .strtab, or .nv.info) not found.")
        return

    sym_sh = shdrs[symtab_idx]
    str_sh = shdrs[strtab_idx]
    nv_sh = shdrs[nvinfo_idx]

    sym_blob = bytes(elf_bytes[sym_sh.sh_offset : sym_sh.sh_offset + sym_sh.sh_size])
    str_blob = bytes(elf_bytes[str_sh.sh_offset : str_sh.sh_offset + str_sh.sh_size])
    nv_blob = bytes(elf_bytes[nv_sh.sh_offset : nv_sh.sh_offset + nv_sh.sh_size])

    # 预解析符号名映射
    sym_map = {}
    for i in range(sym_sh.sh_size // 24):
        sym = _parse_sym(sym_blob, i * 24)
        name = _cstr_at(str_blob, sym.st_name)
        sym_map[i] = name

    # TLV 遍历解析 .nv.info
    offset = 0
    size = len(nv_blob)
    
    print(f"{'Offset':<8} | {'Tag':<8} | {'Size':<4} | {'Value':<8} | {'SymIdx':<6} -> Symbol Name")
    print("-" * 80)

    while offset + 4 <= size:
        # 1. 读取 Header (Tag 2字节, Size 2字节)
        tag, length = struct.unpack_from("<HH", nv_blob, offset)
        
        # 2. 识别常见的 0xXX04 格式 (4字节 SymIdx + 4字节 Value)
        # 大部分属性如 RegCount(0x2f04), MaxThreads(0x1b04) 都属于此类，长度为 8
        if length == 8 and offset + 12 <= size:
            sym_idx, val = struct.unpack_from("<II", nv_blob, offset + 4)
            name = sym_map.get(sym_idx, f"unknown_{sym_idx}")
            
            # 格式化输出，方便对比
            print(f"0x{offset:04x}   | 0x{tag:04x}   | {length:<4} | {val:<8} | {sym_idx:<6} -> {name}")
        
        # 3. 处理非 8 字节长度的属性 (例如 MIN_CTA_PER_SM 或其他)
        else:
            # 如果是其他长度，仅打印摘要
            print(f"0x{offset:04x}   | 0x{tag:04x}   | {length:<4} | [Other Payload Length]")

        # 4. 计算下一个条目的偏移
        # 公式：Header(4) + Payload Length，然后向上 4 字节对齐
        offset = (offset + 4 + length + 3) & ~3

    print("[DEBUG] --- Analysis End ---\n")

def _parse_host_elf_sections(host: bytes) -> dict[str, tuple[int, int]]:
    """
    Return mapping {section_name: (file_offset, size)} for ELF64 host binary.
    """
    eh = _parse_ehdr(host, 0)
    if eh is None:
        raise SystemExit("input is not an ELF64 file")
    if eh.e_shoff == 0 or eh.e_shentsize == 0 or eh.e_shnum == 0:
        raise SystemExit("ELF has no section header table")
    if eh.e_shentsize != 64:
        raise SystemExit(f"unexpected ELF64 shentsize={eh.e_shentsize} (expected 64)")
    shdrs = [_parse_shdr(host, 0, eh.e_shoff + i * eh.e_shentsize) for i in range(eh.e_shnum)]
    if eh.e_shstrndx >= len(shdrs):
        raise SystemExit("invalid e_shstrndx")
    shstr = shdrs[eh.e_shstrndx]
    shstrtab = bytes(host[shstr.sh_offset : shstr.sh_offset + shstr.sh_size])
    sec_names = [_cstr_at(shstrtab, sh.sh_name) if sh.sh_name else "" for sh in shdrs]
    out: dict[str, tuple[int, int]] = {}
    for name, sh in zip(sec_names, shdrs):
        if not name:
            continue
        out[name] = (int(sh.sh_offset), int(sh.sh_size))
    return out


def _compile_nosync_launcher(out_path: Path, cuda_bin: Path | None) -> Path:
    """
    Build a minimal CUDA-driver launcher that calls cuLaunchKernel and exits
    immediately without synchronization (to avoid hanging on persistent kernels).
    """
    cc = shutil.which("g++") or shutil.which("c++")
    if cc is None:
        raise SystemExit("missing tool: g++")

    src = r"""
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

#include "mirage/persistent_kernel/runtime_header.h"

using PFN_cuInit = CUresult (*)(unsigned int);
using PFN_cuDeviceGet = CUresult (*)(CUdevice*, int);
using PFN_cuCtxCreate = CUresult (*)(CUcontext*, unsigned int, CUdevice);
using PFN_cuModuleLoad = CUresult (*)(CUmodule*, const char*);
using PFN_cuModuleGetFunction = CUresult (*)(CUfunction*, CUmodule, const char*);
using PFN_cuLaunchKernel = CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                                       unsigned int, unsigned int, unsigned int, CUstream, void**, void**);
using PFN_cuFuncGetAttribute = CUresult (*)(int*, CUfunction_attribute, CUfunction);
using PFN_cuGetErrorName = CUresult (*)(CUresult, const char**);
using PFN_cuGetErrorString = CUresult (*)(CUresult, const char**);

static PFN_cuInit p_cuInit = nullptr;
static PFN_cuDeviceGet p_cuDeviceGet = nullptr;
static PFN_cuCtxCreate p_cuCtxCreate = nullptr;
static PFN_cuModuleLoad p_cuModuleLoad = nullptr;
static PFN_cuModuleGetFunction p_cuModuleGetFunction = nullptr;
static PFN_cuLaunchKernel p_cuLaunchKernel = nullptr;
static PFN_cuFuncGetAttribute p_cuFuncGetAttribute = nullptr;
static PFN_cuGetErrorName p_cuGetErrorName = nullptr;
static PFN_cuGetErrorString p_cuGetErrorString = nullptr;

static void cu_check(CUresult r, const char* call, const char* file, int line) {
  if (r == CUDA_SUCCESS) return;
  const char* name = nullptr;
  const char* desc = nullptr;
  if (p_cuGetErrorName) p_cuGetErrorName(r, &name);
  if (p_cuGetErrorString) p_cuGetErrorString(r, &desc);
  std::fprintf(stderr, "CUDA driver error %s:%d: %s failed: %s (%s)\n",
               file, line, call, name ? name : "?", desc ? desc : "?");
  std::exit(1);
}
#define CU_CHECK(x) cu_check((x), #x, __FILE__, __LINE__)

static void* must_dlopen() {
  void* h = dlopen("libcuda.so.1", RTLD_NOW);
  if (!h) h = dlopen("libcuda.so", RTLD_NOW);
  if (!h) {
    std::fprintf(stderr, "dlopen(libcuda) failed: %s\n", dlerror());
    return nullptr;
  }
  return h;
}

template <typename T>
static T must_sym(void* h, const char* name) {
  dlerror();
  void* p = dlsym(h, name);
  const char* e = dlerror();
  if (e || !p) {
    std::fprintf(stderr, "dlsym(%s) failed: %s\n", name, e ? e : "null");
    std::exit(1);
  }
  return reinterpret_cast<T>(p);
}

template <typename T>
static T sym_oneof(void* h, const char* name0, const char* name1) {
  dlerror();
  void* p = dlsym(h, name0);
  if (p) return reinterpret_cast<T>(p);
  dlerror();
  p = dlsym(h, name1);
  if (p) return reinterpret_cast<T>(p);
  std::fprintf(stderr, "dlsym(%s or %s) failed\n", name0, name1);
  std::exit(1);
}

static void load_cuda_driver_syms() {
  void* h = must_dlopen();
  if (!h) std::exit(0);

  p_cuInit = must_sym<PFN_cuInit>(h, "cuInit");
  p_cuDeviceGet = must_sym<PFN_cuDeviceGet>(h, "cuDeviceGet");

  // cuCtxCreate is macro-mapped to cuCtxCreate_v2 in modern headers/toolkits, but
  // some environments may still export cuCtxCreate.
  p_cuCtxCreate = sym_oneof<PFN_cuCtxCreate>(h, "cuCtxCreate_v2", "cuCtxCreate");
  p_cuModuleLoad = must_sym<PFN_cuModuleLoad>(h, "cuModuleLoad");
  p_cuModuleGetFunction = must_sym<PFN_cuModuleGetFunction>(h, "cuModuleGetFunction");
  p_cuLaunchKernel = must_sym<PFN_cuLaunchKernel>(h, "cuLaunchKernel");
  p_cuFuncGetAttribute = must_sym<PFN_cuFuncGetAttribute>(h, "cuFuncGetAttribute");
  p_cuGetErrorName = must_sym<PFN_cuGetErrorName>(h, "cuGetErrorName");
  p_cuGetErrorString = must_sym<PFN_cuGetErrorString>(h, "cuGetErrorString");
}

int main(int argc, char** argv) {
  load_cuda_driver_syms();
  // We intentionally use _Exit(0) to avoid any teardown that might synchronize
  // with a persistent kernel. Make stdout/stderr unbuffered so messages still
  // appear.
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::setvbuf(stderr, nullptr, _IONBF, 0);
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <cubin> <kernel> [grid=1] [block=1024]\n", argv[0]);
    return 2;
  }
  const char* cubin_path = argv[1];
  const char* kernel = argv[2];
  int grid = (argc >= 4) ? std::atoi(argv[3]) : 1;
  int block = (argc >= 5) ? std::atoi(argv[4]) : 1024;
  if (grid <= 0) grid = 1;
  if (block <= 0) block = 1;

  CUresult init = p_cuInit(0);
  if (init != CUDA_SUCCESS) {
    const char* name = nullptr;
    const char* desc = nullptr;
    p_cuGetErrorName(init, &name);
    p_cuGetErrorString(init, &desc);
    std::fprintf(stderr, "cuInit failed: %s (%s)\n", name ? name : "?", desc ? desc : "?");
    return 0;
  }

  CUdevice dev{};
  CU_CHECK(p_cuDeviceGet(&dev, 0));
  CUcontext ctx{};
  CU_CHECK(p_cuCtxCreate(&ctx, 0, dev));

  CUmodule mod{};
  CU_CHECK(p_cuModuleLoad(&mod, cubin_path));
  CUfunction fun{};
  CU_CHECK(p_cuModuleGetFunction(&fun, mod, kernel));

  int max_threads = 0;
  int num_regs = 0;
  int static_smem = 0;
  if (p_cuFuncGetAttribute) {
    (void)p_cuFuncGetAttribute(&max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fun);
    (void)p_cuFuncGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fun);
    (void)p_cuFuncGetAttribute(&static_smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fun);
    std::printf("kernel attrs: max_threads_per_block=%d num_regs=%d static_smem=%d\n",
                max_threads, num_regs, static_smem);
  }

  // worker_kernel takes a by-value RuntimeConfig. Use the real type so the
  // driver sees a correctly-sized argument buffer (avoid INVALID_VALUE).
  mirage::runtime::RuntimeConfig cfg{};
  void* kernelParams[] = {&cfg};

  CUresult launch = p_cuLaunchKernel(fun,
                                     (unsigned)grid, 1, 1,
                                     (unsigned)block, 1, 1,
                                     0,
                                     /*stream=*/0,
                                     /*kernelParams=*/kernelParams,
                                     /*extra=*/nullptr);
  if (launch == CUDA_SUCCESS) {
    std::printf("cuLaunchKernel: SUCCESS (grid=%d block=%d)\n", grid, block);
  } else {
    const char* name = nullptr;
    const char* desc = nullptr;
    p_cuGetErrorName(launch, &name);
    p_cuGetErrorString(launch, &desc);
    std::printf("cuLaunchKernel: %s (%s)\n", name ? name : "?", desc ? desc : "?");
  }

  // Exit immediately without synchronizing, to avoid hanging on persistent kernels.
  std::fflush(nullptr);
  std::_Exit(0);
}
"""

    tmp_dir = out_path.parent
    src_path = tmp_dir / "launch_cubin_nosync.cpp"
    src_path.write_text(src)

    include_flags: list[str] = []
    if cuda_bin is not None:
        cuda_root = cuda_bin.parent
        cuda_inc = cuda_root / "include"
        if cuda_inc.exists():
            include_flags += [f"-I{cuda_inc}"]
    # Needed for mirage/persistent_kernel/runtime_header.h
    repo_root = Path(__file__).resolve().parent
    include_flags += [f"-I{repo_root / 'include'}"]

    cmd = (
        [cc, "-O2", "-DMIRAGE_BACKEND_USE_CUDA"]
        + include_flags
        + [str(src_path), "-ldl", "-o", str(out_path)]
    )
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"failed to build launcher: {' '.join(cmd)}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", type=Path, default=None, help="Input megakernel .so (ELF shared library).")
    ap.add_argument("--search-dir", type=Path, default=None, help="Search directory for megakernel_rank*.so if --so omitted.")
    ap.add_argument("--out", type=Path, default=None, help="Output patched .so path (default: <so>.patched.so).")
    ap.add_argument("--reg", type=int, required=True, help="Regcount to write into metadata (0..255).")
    ap.add_argument("--kernel-substr", type=str, default="worker_kernel", help="Substring to match worker kernel symbols.")
    ap.add_argument("--dump-dir", type=Path, default=Path("mpk_patch_dump"), help="Where to write extracted cubin + SASS.")
    ap.add_argument("--no-sass", action="store_true", help="Skip nvdisasm SASS generation.")
    ap.add_argument("--no-launch-test", action="store_true", help="Skip launch test.")
    ap.add_argument("--grid", type=int, default=1, help="Grid for launch test.")
    ap.add_argument("--block", type=int, default=512, help="Block for launch test.")
    args = ap.parse_args()

    so_path: Path | None = args.so
    if so_path is None:
        search_dir = args.search_dir or Path(".")
        cands = sorted(search_dir.rglob("megakernel_rank*.so"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            raise SystemExit("could not find megakernel_rank*.so; pass --so or re-run mpk.compile(output_dir=...)")
        so_path = cands[0]

    if not so_path.exists():
        raise SystemExit(f"missing input .so: {so_path}")

    out_path = args.out or so_path.with_suffix(so_path.suffix + ".patched.so")
    dump_dir = args.dump_dir
    dump_dir.mkdir(parents=True, exist_ok=True)

    host = bytearray(so_path.read_bytes())
    host_sections = _parse_host_elf_sections(bytes(host))
    fat_sections = []
    for sec in (".nv_fatbin", ".nvFatBinSegment"):
        if sec in host_sections:
            fat_sections.append((sec, host_sections[sec][0], host_sections[sec][1]))
    if not fat_sections:
        raise SystemExit("no CUDA fatbin sections found (.nv_fatbin/.nvFatBinSegment)")

    any_patched = False
    dumped_cubins: list[tuple[Path, str]] = []
    patched_syms_all: list[str] = []

    for sec_name, sec_off, sec_size in fat_sections:
        payload = bytes(host[sec_off : sec_off + sec_size])
        for elf_rel_off, elf_size in _iter_embedded_elfs(payload):
            elf_abs_off = sec_off + elf_rel_off
            elf_bytes = bytearray(host[elf_abs_off : elf_abs_off + elf_size])
            patched, patched_syms = _patch_cubin_worker_kernel_regcount(
                elf_bytes, new_reg=args.reg, kernel_substr=args.kernel_substr
            )
            # analyze_symidx_names(elf_bytes)
            if not patched:
                continue
            any_patched = True
            patched_syms_all.extend(patched_syms)
            host[elf_abs_off : elf_abs_off + elf_size] = elf_bytes

            # Dump a copy for inspection.
            cubin_path = dump_dir / f"worker_kernel.image{len(dumped_cubins)}.cubin"
            cubin_path.write_bytes(bytes(elf_bytes))
            dumped_cubins.append((cubin_path, patched_syms[0] if patched_syms else ""))

    if not any_patched:
        raise SystemExit(f"no embedded cubin contained a FUNC symbol matching '{args.kernel_substr}'")

    out_path.write_bytes(host)
    print(f"Patched .so written to: {out_path}")
    print(f"Patched symbols (may include mangled names): {sorted(set(patched_syms_all))}")

    cuda_bin = _find_cuda_bin_dir()
    nvdisasm = _tool_path("nvdisasm", cuda_bin)

    if not args.no_sass:
        for cubin_path, _sym in dumped_cubins[:1]:
            sass_path = cubin_path.with_suffix(".sass")
            with open(sass_path, "w") as f:
                proc = subprocess.run(
                    [nvdisasm, "--print-code", "--print-raw", str(cubin_path)],
                    text=True,
                    stdout=f,
                    stderr=subprocess.PIPE,
                )
            if proc.returncode != 0:
                sys.stderr.write(proc.stderr)
                print(f"Warning: nvdisasm failed for {cubin_path}")
            else:
                print(f"SASS written to: {sass_path}")

    if not args.no_launch_test:
        # Launch-test on the dumped cubin (not the .so), because it's simpler to load.
        # This only checks whether cuLaunchKernel returns OUT_OF_RESOURCES at the given
        # launch shape; it exits immediately without sync to avoid hanging on persistent kernels.
        launcher = dump_dir / "launch_cubin_nosync"
        _compile_nosync_launcher(launcher, cuda_bin)
        test_cubin, test_sym = dumped_cubins[0]
        if not test_sym:
            test_sym = args.kernel_substr
        proc = subprocess.run(
            [str(launcher), str(test_cubin), test_sym, str(args.grid), str(args.block)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
