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

#ifndef Parallel_hh_INCLUDED
#define Parallel_hh_INCLUDED

// C++ Headers
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

// EnergyPlus Headers
#include <EnergyPlus/Data/BaseData.hh>
#include <EnergyPlus/EnergyPlus.hh>

namespace EnergyPlus {

struct EnergyPlusData;

namespace Parallel {

    // A persistent fork/join thread pool owned by one EnergyPlusData instance.
    //
    // The pool has numThreads threads in total: the calling thread (thread 0) plus numThreads - 1
    // workers. A region is executed by all threads and the caller returns only when every thread
    // has finished, so a region behaves like a plain loop to its caller. Workers spin briefly after
    // a region and then park on a condition variable, which keeps fork/join in the microsecond range
    // for the many small regions issued within one timestep while leaving cores free between them.
    //
    // Regions must not perform reductions across threads; callers store per-item results and combine
    // them serially afterwards so that results do not depend on the number of threads. Exceptions
    // thrown inside a region are captured and rethrown on the calling thread after the join.
    class ThreadPool
    {
    public:
        explicit ThreadPool(int numThreads);
        ~ThreadPool();
        ThreadPool(ThreadPool const &) = delete;
        ThreadPool &operator=(ThreadPool const &) = delete;

        int numThreads() const
        {
            return numThreads_;
        }

        // Runs fn(tid) once on every thread of the pool (the caller is tid 0) and waits for all.
        template <typename F> void runOnAll(F &&fn)
        {
            if (numThreads_ == 1) {
                fn(0);
                return;
            }
            using Fn = std::remove_reference_t<F>;
            dispatch([](void *ctx, int tid) { (*static_cast<Fn *>(ctx))(tid); }, static_cast<void *>(&fn));
        }

        // Runs fn(i, tid) for i in [begin, end) with a static contiguous partition over the threads.
        template <typename F> void parallelFor(int begin, int end, F &&fn)
        {
            int const n = end - begin;
            if (n <= 0) {
                return;
            }
            if (numThreads_ == 1 || n == 1) {
                for (int i = begin; i < end; ++i) {
                    fn(i, 0);
                }
                return;
            }
            int const T = numThreads_;
            runOnAll([&](int tid) {
                int const lo = begin + static_cast<int>((static_cast<long long>(n) * tid) / T);
                int const hi = begin + static_cast<int>((static_cast<long long>(n) * (tid + 1)) / T);
                for (int i = lo; i < hi; ++i) {
                    fn(i, tid);
                }
            });
        }

        // Runs fn(chunkBegin, chunkEnd, tid) with explicit chunk boundaries (bounds.size() == numThreads + 1),
        // typically produced by balancedBounds so that unequal item costs are spread evenly.
        template <typename F> void parallelChunks(std::vector<int> const &bounds, F &&fn)
        {
            runOnAll([&](int tid) { fn(bounds[tid], bounds[tid + 1], tid); });
        }

        // Splits [begin, end) into numThreads contiguous chunks of roughly equal summed cost(i).
        template <typename CostFn> std::vector<int> balancedBounds(int begin, int end, CostFn &&cost) const
        {
            std::vector<int> bounds(numThreads_ + 1, end);
            bounds[0] = begin;
            double total = 0.0;
            for (int i = begin; i < end; ++i) {
                total += cost(i);
            }
            double const target = total / numThreads_;
            double acc = 0.0;
            int chunk = 1;
            for (int i = begin; i < end && chunk < numThreads_; ++i) {
                acc += cost(i);
                if (acc >= target * chunk) {
                    bounds[chunk++] = i + 1;
                }
            }
            return bounds;
        }

    private:
        using Thunk = void (*)(void *, int);
        void dispatch(Thunk thunk, void *ctx);
        void workerLoop(int tid);

        int numThreads_;
        std::vector<std::thread> workers_;

        Thunk thunk_ = nullptr;
        void *ctx_ = nullptr;
        alignas(64) std::atomic<std::uint64_t> generation_{0};
        alignas(64) std::atomic<int> remaining_{0};
        alignas(64) std::atomic<int> parked_{0};
        std::atomic<bool> stop_{false};
        std::mutex mutex_;
        std::condition_variable cv_;
        std::mutex exceptionMutex_;
        std::exception_ptr exception_;
    };

    // Creates the pool for this state from dataGlobal->numThread (set by --threads / -j, or
    // ENERGYPLUS_NUM_THREADS). Safe to call more than once; it is a no-op when the pool exists
    // with the requested size.
    void initialize(EnergyPlusData &state);

    // The pool for this state, created on first use if initialize() has not been called.
    ThreadPool &pool(EnergyPlusData &state);

} // namespace Parallel

struct ParallelData : BaseGlobalStruct
{
    int numThreads = 1; // threads used by the pool (1 = serial)
    std::unique_ptr<Parallel::ThreadPool> pool;

    void init_constant_state([[maybe_unused]] EnergyPlusData &state) override
    {
    }

    void init_state([[maybe_unused]] EnergyPlusData &state) override
    {
    }

    void clear_state() override
    {
        pool.reset();
        numThreads = 1;
    }
};

} // namespace EnergyPlus

#endif
