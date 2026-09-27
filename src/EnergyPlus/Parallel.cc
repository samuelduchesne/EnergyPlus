// EnergyPlus, Copyright (c) 1996-present, The Board of Trustees of the University of Illinois,
// The Regents of the University of California, through Lawrence Berkeley National Laboratory
// (subject to receipt of any required approvals from the U.S. Dept. of Energy), Oak Ridge
// National Laboratory, managed by UT-Battelle, Alliance for Energy Innovation, LLC, and other
// contributors. All rights reserved.
//
// NOTICE: This Software was developed under funding from the U.S. Department of Energy and the
// U.S. Government consequently retains certain rights. As such, the U.S. Government has been
// granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable,
// worldwide license in the Software to reproduce, distribute copies to the public, prepare
// derivative works, and perform publicly and display publicly, and to permit others to do so.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted
// provided that the following conditions are met:
//
// (1) Redistributions of source code must retain the above copyright notice, this list of
//     conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright notice, this list of
//     conditions and the following disclaimer in the documentation and/or other materials
//     provided with the distribution.
//
// (3) Neither the name of the University of California, Lawrence Berkeley National Laboratory,
//     the University of Illinois, U.S. Dept. of Energy nor the names of its contributors may be
//     used to endorse or promote products derived from this software without specific prior
//     written permission.
//
// (4) Use of EnergyPlus(TM) Name. If Licensee (i) distributes the software in stand-alone form
//     without changes from the version obtained under this License, or (ii) Licensee makes a
//     reference solely to the software portion of its product, Licensee must refer to the
//     software as "EnergyPlus version X" software, where "X" is the version number Licensee
//     obtained under this License and may not use a different name for the software. Except as
//     specifically required in this Section (4), Licensee shall not use in a company name, a
//     product name, in advertising, publicity, or other promotional activities any name, trade
//     name, trademark, logo, or other designation of "EnergyPlus", "E+", "e+" or confusingly
//     similar designation, without the U.S. Department of Energy's prior written consent.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// C++ Headers
#include <algorithm>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define EP_CPU_RELAX() _mm_pause()
#elif defined(__aarch64__) && !defined(_MSC_VER)
#define EP_CPU_RELAX() asm volatile("yield" ::: "memory")
#else
#define EP_CPU_RELAX() std::this_thread::yield()
#endif

// EnergyPlus Headers
#include <EnergyPlus/Data/EnergyPlusData.hh>
#include <EnergyPlus/DataGlobals.hh>
#include <EnergyPlus/Parallel.hh>

namespace EnergyPlus::Parallel {

namespace {
    // Idle workers spin for about a millisecond, yield for about another millisecond, then park.
    constexpr int kSpinIterations = 20000;
    constexpr int kYieldIterations = 1000;
} // namespace

ThreadPool::ThreadPool(int numThreads) : numThreads_(std::max(1, numThreads))
{
    workers_.reserve(numThreads_ - 1);
    for (int tid = 1; tid < numThreads_; ++tid) {
        workers_.emplace_back([this, tid] { workerLoop(tid); });
    }
}

ThreadPool::~ThreadPool()
{
    stop_.store(true, std::memory_order_seq_cst);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cv_.notify_all();
    }
    for (auto &w : workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
}

void ThreadPool::workerLoop(int tid)
{
    std::uint64_t seen = 0;
    for (;;) {
        std::uint64_t g = generation_.load(std::memory_order_seq_cst);
        int spins = 0;
        while (g == seen) {
            if (stop_.load(std::memory_order_relaxed)) {
                return;
            }
            if (spins < kSpinIterations) {
                EP_CPU_RELAX();
                ++spins;
            } else if (spins < kSpinIterations + kYieldIterations) {
                std::this_thread::yield();
                ++spins;
            } else {
                std::unique_lock<std::mutex> lock(mutex_);
                parked_.fetch_add(1, std::memory_order_seq_cst);
                cv_.wait(lock, [&] { return generation_.load(std::memory_order_seq_cst) != seen || stop_.load(std::memory_order_relaxed); });
                parked_.fetch_sub(1, std::memory_order_seq_cst);
            }
            g = generation_.load(std::memory_order_seq_cst);
        }
        seen = g;
        try {
            thunk_(ctx_, tid);
        } catch (...) {
            std::lock_guard<std::mutex> lock(exceptionMutex_);
            if (!exception_) {
                exception_ = std::current_exception();
            }
        }
        remaining_.fetch_sub(1, std::memory_order_acq_rel);
    }
}

void ThreadPool::dispatch(Thunk thunk, void *ctx)
{
    thunk_ = thunk;
    ctx_ = ctx;
    remaining_.store(numThreads_ - 1, std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_seq_cst);
    if (parked_.load(std::memory_order_seq_cst) > 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        cv_.notify_all();
    }
    try {
        thunk(ctx, 0);
    } catch (...) {
        std::lock_guard<std::mutex> lock(exceptionMutex_);
        if (!exception_) {
            exception_ = std::current_exception();
        }
    }
    int spins = 0;
    while (remaining_.load(std::memory_order_acquire) > 0) {
        if (spins < kSpinIterations) {
            EP_CPU_RELAX();
            ++spins;
        } else {
            std::this_thread::yield();
        }
    }
    if (exception_) {
        std::exception_ptr e;
        {
            std::lock_guard<std::mutex> lock(exceptionMutex_);
            e = exception_;
            exception_ = nullptr;
        }
        std::rethrow_exception(e);
    }
}

void initialize(EnergyPlusData &state)
{
    int requested = std::max(1, state.dataGlobal->numThread);
    int const hardware = static_cast<int>(std::thread::hardware_concurrency());
    if (hardware > 0) {
        requested = std::min(requested, hardware);
    }
    auto &d = *state.dataParallel;
    if (d.pool && d.numThreads == requested) {
        return;
    }
    d.pool.reset();
    d.numThreads = requested;
    d.pool = std::make_unique<ThreadPool>(requested);
}

ThreadPool &pool(EnergyPlusData &state)
{
    if (!state.dataParallel->pool) {
        initialize(state);
    }
    return *state.dataParallel->pool;
}

} // namespace EnergyPlus::Parallel
