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
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
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

        // Runs fn(chunkBegin, chunkEnd, tid) with explicit chunk boundaries (bounds.size() == numThreads + 1),
        // typically produced by balancedBounds so that unequal item costs are spread evenly.
        template <typename F> void parallelChunks(std::vector<int> const &bounds, F &&fn)
        {
            assert(static_cast<int>(bounds.size()) == numThreads_ + 1);
            runOnAll([&](int tid) { fn(bounds[tid], bounds[tid + 1], tid); });
        }

        // Splits [begin, end) into at most numThreads contiguous chunks so that the largest summed cost(i) of a chunk
        // is as small as possible (the chunk that finishes last bounds the region's time). The optimum is found by
        // bisection on the per-chunk limit; packing items for a given limit is greedy and exact. Trailing chunks
        // that are not needed are empty. Intended to be called once per thread count, not per region.
        template <typename CostFn> std::vector<int> balancedBounds(int begin, int end, CostFn &&cost) const
        {
            int const n = std::max(0, end - begin);
            std::vector<double> costs(n);
            double total = 0.0;
            double maxCost = 0.0;
            for (int i = 0; i < n; ++i) {
                costs[i] = cost(begin + i);
                total += costs[i];
                maxCost = std::max(maxCost, costs[i]);
            }
            // Packs the items into chunks whose cost does not exceed limit; false when that takes more than numThreads chunks.
            auto pack = [&](double const limit, std::vector<int> *bounds) -> bool {
                if (bounds != nullptr) {
                    bounds->assign(numThreads_ + 1, end);
                    (*bounds)[0] = begin;
                }
                int chunk = 1;
                double acc = 0.0;
                for (int i = 0; i < n; ++i) {
                    if (acc + costs[i] > limit && acc > 0.0) { // start a new chunk before item i
                        if (chunk >= numThreads_) {
                            return false;
                        }
                        if (bounds != nullptr) {
                            (*bounds)[chunk] = begin + i;
                        }
                        ++chunk;
                        acc = 0.0;
                    }
                    acc += costs[i];
                }
                return true;
            };
            double lo = maxCost; // no limit below the heaviest item can be feasible
            double hi = total;   // one chunk always fits
            for (int iter = 0; iter < 64 && hi - lo > 1.0e-9 * std::max(1.0, total); ++iter) {
                double const mid = 0.5 * (lo + hi);
                if (pack(mid, nullptr)) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            std::vector<int> bounds;
            pack(hi, &bounds);
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

    // Chooses between a serial and a parallel variant of a region. Whether the parallel variant pays for its
    // fork/join and cold caches depends on the model, the machine and what else is running, so calls are timed
    // alternately serial and parallel, the faster variant is kept for a while, and the comparison is repeated
    // periodically. Only the scheduling changes; both variants must compute identical results.
    //
    //   bool const parallel = choice.chooseParallel(); // before the region
    //   ... run the region with the chosen variant ...
    //   if (choice.recordSample()) { ... a comparison just completed; decision() says which variant won ... }
    class AdaptiveChoice
    {
    public:
        static constexpr int samplesPerVariant = 64;         // timed calls of each variant per comparison
        static constexpr int callsBetweenComparisons = 4096; // calls the decision is kept before comparing again
        static constexpr double requiredGain = 0.9;          // parallel is kept only when it is at least 10% faster

        // Returns true when the next call should run the parallel variant; starts the timer when the call is a sample.
        bool chooseParallel();
        // Call after the region. Returns true when this sample completed a comparison and a new decision was made.
        bool recordSample();
        // 0 = still measuring, 1 = parallel chosen, -1 = serial chosen
        int decision() const
        {
            return decision_;
        }
        double serialSecondsPerCall() const
        {
            return serialSeconds_ / samplesPerVariant;
        }
        double parallelSecondsPerCall() const
        {
            return parallelSeconds_ / samplesPerVariant;
        }
        void reset();

    private:
        int decision_ = 0;
        int samples_ = 0; // timed samples so far while measuring, or decided calls since the last comparison
        double serialSeconds_ = 0.0;
        double parallelSeconds_ = 0.0;
        bool timing_ = false;   // the call chosen last is a timed sample
        bool parallel_ = false; // the variant chosen last
        std::chrono::steady_clock::time_point start_;
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
