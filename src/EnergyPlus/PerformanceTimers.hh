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

#ifndef PerformanceTimers_hh_INCLUDED
#define PerformanceTimers_hh_INCLUDED

// C++ Headers
#include <array>
#include <chrono>
#include <string_view>
#include <vector>

// EnergyPlus Headers
#include <EnergyPlus/Data/BaseData.hh>
#include <EnergyPlus/Data/EnergyPlusData.hh>
#include <EnergyPlus/DataSystemVariables.hh>
#include <EnergyPlus/EnergyPlus.hh>

namespace EnergyPlus {

// Scoped wall-clock timers for the main simulation phases.
//
// Timing is off by default and costs one boolean test per scope. It is enabled by the --timings
// command line flag, the TimingFlag environment variable, or Output:Diagnostics,TimingFlag. When
// enabled, each timed scope records inclusive time, exclusive time (inclusive minus the time spent
// in nested timed scopes), a call count and an optional item count, and the totals are written to
// eplusout.perf (JSON) and echoed to the console at the end of the run. Timers are never written
// to the eio or err files, so enabling them does not change any regression-compared output.

namespace Perf {

    enum class Timer : int
    {
        Sizing,
        Weather,
        HeatBalance,
        InitSurfaceHB,
        SolarShading,
        InitSolarHeatGains,
        Daylighting,
        InternalGains,
        OutsideHB,
        InsideHB,
        IntRadExchange,
        WindowHB,
        ConvectionInside,
        ConvectionOutside,
        ThermalHistories,
        AirHeatBalance,
        ZoneAirUpdates,
        HVAC,
        SimHVAC,
        AirLoops,
        ZoneEquipment,
        NonZoneEquipment,
        ElectricService,
        Plant,
        AirflowNetwork,
        ReportSurfaceHB,
        ReportHeatBalance,
        ReportAirHB,
        NodeInfo,
        UpdateDataandReport,
        Num
    };

    constexpr int numTimers = static_cast<int>(Timer::Num);

    constexpr std::array<std::string_view, numTimers> timerNames = {
        "Sizing",             // ManageSizing
        "Weather",            // ManageWeather
        "HeatBalance",        // ManageHeatBalance (zone timestep, includes everything below down to ReportHeatBalance)
        "InitSurfaceHB",      // InitSurfaceHeatBalance
        "SolarShading",       // PerformSolarCalculations (shadowing, sunlit fractions)
        "InitSolarHeatGains", // InitSolarHeatGains
        "Daylighting",        // manageDaylighting
        "InternalGains",      // ManageInternalHeatGains
        "OutsideHB",          // CalcHeatBalanceOutsideSurf
        "InsideHB",           // CalcHeatBalanceInsideSurf (items = inside iterations)
        "IntRadExchange",     // CalcInteriorRadExchange (all callers)
        "WindowHB",           // CalcWindowHeatBalance (per window)
        "ConvectionInside",   // InitIntConvCoeff
        "ConvectionOutside",  // InitExtConvCoeff
        "ThermalHistories",   // UpdateThermalHistories
        "AirHeatBalance",     // ManageAirHeatBalance
        "ZoneAirUpdates",     // ManageZoneAirUpdates (predictor and corrector)
        "HVAC",               // ManageHVAC (system timestep loop, includes SimHVAC and HVAC reporting)
        "SimHVAC",            // SimHVAC (one converged HVAC solution)
        "AirLoops",           // ManageAirLoops
        "ZoneEquipment",      // ManageZoneEquipment
        "NonZoneEquipment",   // ManageNonZoneEquipment
        "ElectricService",    // manageElectricPowerService
        "Plant",              // ManagePlantLoops
        "AirflowNetwork",     // AirflowNetwork manage_balance
        "ReportSurfaceHB",    // ReportSurfaceHeatBalance
        "ReportHeatBalance",  // ReportHeatBalance
        "ReportAirHB",        // ReportAirHeatBalance
        "NodeInfo",           // CalcMoreNodeInfo
        "UpdateDataandReport" // UpdateDataandReport (output processor, ESO/MTR/SQL writers)
    };

    // Adds n to the item counter of a timer (for example iterations) when timing is enabled.
    void addItems(EnergyPlusData &state, Timer t, long long n);

    // Writes eplusout.perf and a console summary when timing is enabled. No-op otherwise.
    void writeReport(EnergyPlusData &state);

    // RAII scope timer. Construct at the top of the function or block to be timed.
    class ScopedTimer
    {
    public:
        using Clock = std::chrono::steady_clock;

        ScopedTimer(EnergyPlusData &state, Timer t);
        ~ScopedTimer();
        ScopedTimer(ScopedTimer const &) = delete;
        ScopedTimer &operator=(ScopedTimer const &) = delete;

    private:
        EnergyPlusData &state_;
        int idx_;
        bool active_;
        Clock::time_point start_;
    };

} // namespace Perf

struct PerfTimerData : BaseGlobalStruct
{
    using Clock = std::chrono::steady_clock;

    Clock::time_point processStart = Clock::now(); // wall clock reference for the report
    std::array<double, Perf::numTimers> inclusive{};
    std::array<double, Perf::numTimers> exclusive{};
    std::array<long long, Perf::numTimers> calls{};
    std::array<long long, Perf::numTimers> items{};
    std::vector<double> childStack; // time spent in nested timed scopes, one entry per open scope

    void init_constant_state([[maybe_unused]] EnergyPlusData &state) override
    {
    }

    void init_state([[maybe_unused]] EnergyPlusData &state) override
    {
    }

    void clear_state() override
    {
        new (this) PerfTimerData();
    }
};

} // namespace EnergyPlus

#endif
