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
#include <format>
#include <string>

// EnergyPlus Headers
#include <EnergyPlus/Data/EnergyPlusData.hh>
#include <EnergyPlus/DataGlobals.hh>
#include <EnergyPlus/DataSystemVariables.hh>
#include <EnergyPlus/DisplayRoutines.hh>
#include <EnergyPlus/IOFiles.hh>
#include <EnergyPlus/Parallel.hh>
#include <EnergyPlus/PerformanceTimers.hh>

namespace EnergyPlus::Perf {

namespace {
    double secondsBetween(ScopedTimer::Clock::time_point a, ScopedTimer::Clock::time_point b)
    {
        return std::chrono::duration<double>(b - a).count();
    }
} // namespace

ScopedTimer::ScopedTimer(EnergyPlusData &state, Timer t) : state_(state), idx_(static_cast<int>(t)), active_(state.dataSysVars->TimingFlag)
{
    if (active_) {
        state_.dataPerf->childStack.push_back(0.0);
        start_ = Clock::now();
    }
}

ScopedTimer::~ScopedTimer()
{
    if (!active_) {
        return;
    }
    double const dur = secondsBetween(start_, Clock::now());
    auto &d = *state_.dataPerf;
    double child = 0.0;
    if (!d.childStack.empty()) {
        child = d.childStack.back();
        d.childStack.pop_back();
    }
    d.inclusive[idx_] += dur;
    d.exclusive[idx_] += dur - child;
    ++d.calls[idx_];
    if (!d.childStack.empty()) {
        d.childStack.back() += dur;
    }
}

void addItems(EnergyPlusData &state, Timer t, long long n)
{
    if (state.dataSysVars->TimingFlag) {
        state.dataPerf->items[static_cast<int>(t)] += n;
    }
}

void writeReport(EnergyPlusData &state)
{
    if (!state.dataSysVars->TimingFlag) {
        return;
    }
    auto &d = *state.dataPerf;
    double const wall = secondsBetween(d.processStart, PerfTimerData::Clock::now());
    int const threads = state.dataParallel->numThreads;

    auto perfFile = state.files.perf.try_open();
    if (perfFile.good()) {
        print(perfFile, "{{\n  \"wall_seconds\": {:.6f},\n  \"threads\": {},\n  \"timers\": [\n", wall, threads);
        for (int i = 0; i < numTimers; ++i) {
            print(perfFile,
                  "    {{\"name\": \"{}\", \"inclusive_s\": {:.6f}, \"exclusive_s\": {:.6f}, \"calls\": {}, \"items\": {}}}{}\n",
                  timerNames[i],
                  d.inclusive[i],
                  d.exclusive[i],
                  d.calls[i],
                  d.items[i],
                  (i + 1 < numTimers) ? "," : "");
        }
        print(perfFile, "  ]\n}}\n");
        perfFile.close();
    }

    // Console summary, largest exclusive time first
    std::array<int, numTimers> order{};
    for (int i = 0; i < numTimers; ++i) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&d](int a, int b) { return d.exclusive[a] > d.exclusive[b]; });
    DisplayString(state, std::format("Timing summary: wall {:.3f} s, {} thread(s)", wall, threads));
    DisplayString(state, std::format("  {:<20} {:>10} {:>10} {:>7} {:>10} {:>10}", "phase", "inclusive", "exclusive", "%wall", "calls", "items"));
    for (int i : order) {
        if (d.calls[i] == 0) {
            continue;
        }
        DisplayString(state,
                      std::format("  {:<20} {:>10.3f} {:>10.3f} {:>6.1f}% {:>10} {:>10}",
                                  timerNames[i],
                                  d.inclusive[i],
                                  d.exclusive[i],
                                  wall > 0.0 ? 100.0 * d.exclusive[i] / wall : 0.0,
                                  d.calls[i],
                                  d.items[i]));
    }
}

} // namespace EnergyPlus::Perf
