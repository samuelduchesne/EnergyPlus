Faster multizone / multi-plant simulations on multi-core CPUs
==============================================================

**samuelduchesne/EnergyPlus fork — engineering plan, drafted with Claude Code**

 - Original date: 2026-09-27
 - Code base: EnergyPlus 26.2.0 (`e2fd2334`)
 - Status: plan for review. Phase 0, the first two Phase 1 items and the Phase 2 pilot region are
   implemented on branch `claude/sleepy-goldberg-bqjqqq`; see section 13, "Implementation log".

## Summary ##

EnergyPlus is a single-threaded program. There is no `#pragma omp`, `std::thread`,
`std::mutex` or `std::atomic` anywhere in `src/EnergyPlus`; the `-j/--jobs` command-line
option stores a value that nothing reads, and the eio report "Program Control
Information:Threads/Parallel Sims" is a leftover of an OpenMP experiment that was removed
before V8.6. A 130-zone hospital with 6 air loops and 4 plant loops therefore uses one core
of a machine that typically has 8 to 64.

This plan treats the problem as two coupled problems:

1. **A building-physics problem.** How the building is decomposed and how the pieces are
   coupled in time decides what can be computed concurrently without changing the answer.
   EnergyPlus already uses lagged (explicit, Jacobi-style) coupling between surfaces and
   zone air, between adjacent zones through interzone surfaces, between zone air and HVAC,
   and between plant loops through interconnect flags. Those lags are the seams along which
   the work can be split. Where the coupling is a Gauss-Seidel sweep (air loops → zone
   equipment → plant inside one HVAC iteration), we can still run the independent members
   of each stage concurrently, and we can attack the *number* of sweeps, which is where most
   of the HVAC time goes today.

2. **A computer-science problem.** The code carries ~250 module-state structs inside
   `EnergyPlusData`, hundreds of "current object" globals (`CurSysNum`, `CurZoneEqNum`,
   `TurnFansOn`, …), memoising caches that are written on every call (psychrometrics,
   glycols, curves, tables), an error subsystem whose output order is compared as text in
   regression testing, and a reporting subsystem that is one sequential writer. Threads
   cannot be introduced until those are made per-thread, per-call, or deterministic. Many
   of those changes are also worthwhile serial speedups and bug fixes on their own.

The plan is organised as six phases. Phase 0 builds the measurement harness; Phase 1 takes
bit-identical serial wins; Phase 2 builds the threading runtime and hardens shared state;
Phase 3 parallelises the envelope (surfaces, radiation, windows, shading, daylighting);
Phase 4 reduces HVAC iteration counts; Phase 5 parallelises air loops, zone equipment,
plant-loop clusters and reporting; Phase 6 lists research items (per-enclosure convergence,
design-day parallelism, sparse AirflowNetwork solvers). Every phase has an exit criterion
that is measured on a fixed benchmark set, and the default behaviour (`--threads 1`) must
stay bit-identical with today's results throughout.

**Recommended first four weeks.** (1) Land the scoped timers and the benchmark script so every
later claim has a number (Phase 0). (2) Fix the bugs found while mapping the code: the static
`MyEnvrnFlagLocal` in `WaterUse.cc`, the leap-year static in `SQLiteProcedures.cc`, the
by-value `WarmupSurfTemp` counter, the two non-idempotent window-gain accumulations. (3) Take
the two cheapest bit-identical wins: skip the duplicate "Outside" radiant-exchange call and
remove the whole-array copies in the inside-surface iteration. (4) Prototype the thread pool
on the per-enclosure radiant exchange (region P2) to measure fork/join overhead on real
timestep sizes before committing to the wider design.

## 1. Justification ##

 - **Latency of a single large model is the bottleneck** for interactive design, calibration
   loops and sensitivity studies. Portfolio work already parallelises across processes; a
   single 100+ zone model with several plant loops does not.
 - **Cores are cheap and idle.** Laptops have 8–16 cores and cloud instances 32–96. A 4× wall
   clock reduction on an 8-core machine turns a 40-minute hospital into 10 minutes.
 - **The existing "go faster" knob trades accuracy.** `PerformancePrecisionTradeoffs`
   (`SimulationManager.cc:1260-1439`) speeds up by coarsening timesteps, forcing Euler,
   shortening warm-up and loosening convergence (the documented example goes from 100% to
   16% run time, but with an order-of-magnitude more oscillating hours). We want speed that
   preserves the answer.
 - **The upstream architecture is now ready for it.** The multi-year global-state refactor
   put nearly all state into `EnergyPlusData` (251 module structs, `Data/EnergyPlusData.hh`),
   so per-thread contexts and per-instance thread pools are feasible in a way they were not
   when the original OpenMP work was abandoned.

## 2. Where the time goes today ##

### 2.1 Cost model per zone timestep ###

For one zone timestep (`SimulationManager.cc:569`) the work is:

```
T_step ≈ T_weather
       + T_surfHB      = T_outsideHB(all surfaces)
                       + N_insideIter × [ Σ_encl N_e² (ScriptF)  +  per-surface CTF update ]
                       + T_windows (iteration 0 only)  + T_convection (every 30 iterations)
       + T_solar       = interior solar distribution (per timestep) + shading recompute (every 20 days)
       + T_daylighting
       + N_sysSteps × [ T_predict + T_SimHVAC + T_correct ]
       + T_report      = UpdateDataandReport × (1 + N_sysSteps)
```

and inside `SimHVAC` (`HVACManager.cc:691-1773`):

```
T_SimHVAC ≈ (1 + N_hvacIter) × [ N_airPasses × ( T_airLoops + T_zoneEquip + T_AFN )
                                 + T_nonZone + 2 × T_electric
                                 + N_plantSweeps × Σ_halfLoops T_halfLoop ]
```

The multipliers are large: `N_hvacIter` ≤ 21 (`while ... HVACManageIteration <= MaxIter`,
`HVACManager.cc:884`), `N_airPasses` ≤ 6 (`MaxAir = 5`, `:1797`), `N_plantSweeps` ≤ 9 with a
forced minimum of 2 (7 when any loop has a common pipe, `PlantManager.cc:185-191`), and each
air-loop controller iterate re-simulates the *entire* air loop (`SimAirServingZones.cc:3017`).
When the corrector reports a zone temperature change above `MaxZoneTempDiff` (0.3 K) the
system timestep is subdivided and the whole `SimHVAC` is repeated for each sub-step *and the
full-step result is discarded* (`HVACManager.cc:280` vs `:333-405`).

The documented `PerformancePrecisionTradeoffs` study is consistent with this model: forcing
1 zone timestep/hour (Mode01) removed 62% of run time, and forcing the minimum system
timestep to 1 hour (Mode05) halved what was left
(`doc/input-output-reference/src/overview/group-simulation-parameters.tex:885-887`). The
HVAC iteration structure, not the envelope, dominates HVAC-heavy models.

### 2.2 Measured baseline (this container: 4 vCPU, GCC 13.3, `-O3`, no `-march`) ###

Annual runs, single thread, outputs as shipped in the test files, one run each (timings are
indicative; the Phase 0 harness will produce medians on a quiet runner):

| Model | Zones | Opaque + window surfaces | Air loops | Plant + condenser loops | Timesteps/h | Wall time |
|---|---|---|---|---|---|---|
| `performance_tests/15zonevav_no_reports.idf` | 15 | 94 + 12 | 1 | 2 + 1 | 4 | 12.6 s |
| `performance_tests/15zonevav.idf` | 15 | 94 + 12 | 1 | 2 + 1 | 4 | 13.7 s |
| `performance_tests/30zonevav.idf` | 30 | 188 + 24 | 1 | 2 + 1 | 4 | 25.5 s |
| `testfiles/RefBldgLargeOfficeNew2004_Chicago.idf` | 19 | 130 + 12 | 4 | 3 + 1 | 6 | 34.3 s |
| `performance_tests/45zonevav.idf` | 45 | 282 + 36 | 1 | 2 + 1 | 4 | 37.3 s |
| `testfiles/ASHRAE901_Hospital_STD2019_Denver.idf` | 58 | 402 + 57 | 8 | 6 + 1 | 4 | 120.1 s |
| `performance_tests/BenchmarkHospitalNew_USA_CA_SAN_FRANCISCO.idf` | 55 | 402 + 40 | 2 | 3 + 1 | 12 | 141.2 s |
| `testfiles/ASHRAE901_OutPatientHealthCare_STD2019_Denver.idf` | 119 | 1400 + 117 | 2 | 2 + 0 | 4 | 151.5 s |
| `testfiles/HospitalBaseline.idf` | 130 | 1101 + 125 | 6 | 3 + 1 | 6 | 186.6 s |

Two observations from these runs feed directly into the plan:

 - Run time grows with HVAC complexity much faster than with zone count: 45 zones on one
   VAV loop cost 37 s, while 58 zones on 8 air loops and 7 plant/condenser loops cost 120 s.
 - Reporting is not free but is not dominant either: the 15-zone file with tabular,
   monthly and utility-cost reports costs 9% more than the same file without them. (The
   Standard 90.1 hospital counts 1.19 million warnings, nearly all recurring; every one of
   them passes through the shared error-tracking counters that threads will have to
   handle — see 5.2.)

Callgrind instruction-count profiles (percent of total instructions, inclusive unless
marked exclusive). Design-day runs include input processing and sizing, which inflates
those two rows relative to an annual run; the annual 15-zone profile is the steady-state view.

| Function (inclusive) | 45zonevav, design days | HospitalBaseline, design days | 15zonevav, annual |
|---|---|---|---|
| `ManageSimulation` | 88.8% | 96.6% | 98.2% |
| `ManageSurfaceHeatBalance` minus the nested `ManageAirHeatBalance` (envelope) | 32.5% | 32.8% | 31.8% |
| ├ `CalcHeatBalanceInsideSurf` (iterative inside balance) | 16.0% | 13.9% | 16.0% |
| ├ `InitSurfaceHeatBalance` (rad exchange "Main", solar, daylighting init, convection) | 12.4% | 13.7% | 10.9% |
| ├ `CalcInteriorRadExchange` (all call sites) | 7.3% | 7.5% | 6.1% |
| ├ `CalcWindowHeatBalance` | — | 2.6% | 6.0% |
| ├ `CalcHeatBalanceOutsideSurf` | — | 3.0% | 2.1% |
| `ManageHVAC` | 30.8% | 42.3% | 53.4% |
| ├ `SimZoneEquipment` (VAV terminals → reheat coils → `ControlCompOutput`) | 20.7% | 25.6% | 18.1% |
| │ └ `SimulateWaterCoilComponents` | 14.2% | 24.1% | 10.2% |
| │   └ `ControlCompOutput` (interval-halving controller) | 10.0% | 11.5% | 10.2% |
| ├ `ManagePlantLoops` (2 plant + 1 condenser loop in the 15-zone file) | 5.7% | 11.2% | 17.2% |
| ├ `SimAirLoops` | — | 7.8% | 3.5% |
| ├ `ManageZoneAirUpdates` (predictor/corrector) | — | 3.5% | 3.8% |
| ├ `CalcMoreNodeInfo` (per-node psychrometric report values, every system timestep) | — | — | 3.8% |
| ├ `manageElectricPowerService` (7 meter scans × 2 per iteration) | — | — | 2.5% |
| `ReportHeatBalance` (`UpdateDataandReport` 3.9%, tabular/monthly gathering 3.5%) | — | — | 7.5% |
| `ReportZoneMeanAirTemp` | — | — | 2.1% |
| `PerformSolarCalculations` incl. `CalcDayltgCoefficients` | 12.0% | — | 1.8% |
| `ManageSizing` (design-day runs only) | 27.7% | 25.3% | 1.8% |
| Input processing (`processInput`, JSON/valijson) | 5.4% + 5.7% | 3.4% | ~2.5% |
| **Exclusive:** `CalcInteriorRadExchange` | 5.2% | 5.4% | 4.9% |
| **Exclusive:** `pow`/`exp`/`log`/`sincos` (libm) | 8.7% | 7.2% | 8.8% |
| **Exclusive:** `__dynamic_cast` | 1.1% | 1.0% | 3.5% |
| **Exclusive:** `CalcHeatBalanceInsideSurf2CTFOnly` (the kernel itself) | 3.3% | 3.1% | 3.0% |
| **Exclusive:** `PsyTsatFnPb_raw` + `PsyPsatFnTemp_raw` (cache misses) | — | — | 5.8% incl. / 1.0% excl. |
| **Exclusive:** `malloc`/`free` family | 4.9% | 8.1% | < 1% |
| **Exclusive:** `ObjexxFCL::Array<double>` copy constructor | 1.6% | 4.0% | < 0.5% |
| **Exclusive:** `memset` | 3.7% | 2.9% | 1.3% |

### 2.3 Profile-driven hot spots ###

1. **The envelope is a steady third of the work; the inside-surface iteration and long-wave
   exchange are most of it.** The envelope share is 32–33% on all three profiles.
   `CalcHeatBalanceInsideSurf` alone is 16% of the annual run, `CalcInteriorRadExchange`
   6% (and the single largest *exclusive* function at 4.9–5.4%), and the window heat
   balance 6% (`SolveForWindowTemperatures` 4.1%, a 100-iteration Newton solve per window
   per timestep). `CalcInteriorRadExchange` is called 2 + N_iter times per zone timestep
   (section 4.2). These are the prime parallel targets (regions P2, P4, P5) and they carry
   easy serial waste (duplicate "Outside" call, whole-array copies and zeroing every
   iteration).
2. **HVAC is half of an annual run, and zone equipment leads it through water-coil reheat
   control.** `ManageHVAC` is 53% of the annual 15-zone run and 42% of the hospital
   design days. `SimZoneEquipment` (18–26%) is almost entirely VAV reheat terminals →
   `SimulateWaterCoilComponents` → `ControlCompOutput` (10–11.5% on every profile), the
   interval-halving controller that re-simulates the coil up to 25 times per terminal per
   call. Zones are independent here (region P12), and a direct or secant coil solution
   ("Use Coil Direct Solutions" only covers DX coils today) would remove most of those
   evaluations serially (5.3 item 9b).
3. **Plant is 17% of the annual run on a model with only two plant loops and one condenser
   loop, 11% on the hospital design days, and its sweeps are mostly forced, not converged.**
   The iteration-count output variables (annual runs, `detailed` frequency) show:

   | | 45zonevav (2 loops + 1 condenser) | ASHRAE 90.1 Hospital (6 + 1) |
   |---|---|---|
   | System timesteps per zone timestep | 1.74 | 1.17 |
   | "HVAC System Solver Iteration Count", mean / max | 1.00 / 1 | 3.04 / 6 (473 steps hit the file's limit of 5) |
   | "Air System Solver Iteration Count", mean / max | 1.18 / 6 | 7.2 / 48 |
   | "Plant Solver Sub Iteration Count", mean / max | 4.00 / 4 (constant) | 7.8 / 16 (8 in 85% of steps) |
   | "Plant Solver Half Loop Calls Count", mean | 24 (constant) | 109 (14 half-loops × ~8 sweeps) |

   On the 45-zone model the plant counts never vary: the plant does exactly the forced
   minimum sweeps at every system timestep. On the hospital the counts are quantised in
   steps of 2 sweeps × 14 half-loops and take the same value in 85% of all system
   timesteps regardless of load, which is what forced minimum sweeps on each of the ~3
   `ManagePlantLoops` calls per `SimHVAC` produce. This is item 4 of section 4.6, and it is
   why plant-cluster parallelism (P13) *and* a convergence-based sweep count both matter.
4. **Reporting is ~13% of the annual run, and it is serial.** `ReportHeatBalance` (7.5%:
   `UpdateDataandReport` 3.9% plus tabular/monthly gathering 3.5%), `CalcMoreNodeInfo`
   (3.8%: psychrometric report values for every node at every system timestep, whether
   or not anything reports them) and `ReportZoneMeanAirTemp` (2.1%). `__dynamic_cast` is
   3.5% of all instructions on its own — one cast per output variable per
   `UpdateDataandReport` call (5.2, reporting row). This is the Amdahl residue that the
   asynchronous writer (P14) and the serial reporting fixes (5.3 item 7) exist for.
5. **Psychrometric cache misses are visible.** `PsyTsatFnPb_raw` and `PsyPsatFnTemp_raw`
   are 5.8% of the annual run inclusive: the hashed caches (~96 MiB per instance) are
   missing often enough that the raw iterative solves dominate. Cache design, not just
   thread safety, is on the table in 5.2.
6. **libm is 7–9%.** `pow`, `exp`, `log` and `sincos` come from psychrometrics, the T⁴ terms
   in radiant exchange, glycol property fits and solar geometry. Vectorised or fused
   evaluation and the psychrometric cache redesign in 5.2 address this; it is also the part
   that benefits from `-march`.
7. **Allocation and copying are a sizing-phase problem.** On the design-day profiles
   `malloc`/`free`, the `ObjexxFCL` array copy constructor and `memset` are 10–15% of
   exclusive cost, driven by `BaseSizer::initializeWithinEP` copying entire `ZoneSizingData`
   structs (10% of the hospital design-day run by itself), `CalcScriptF`'s per-call `N×N`
   matrices and the whole-array zeroing in the radiant exchange. In the annual run they fall
   below 2%. Fixes are bit-identical (5.3) and matter most for sizing-heavy workflows
   (design-day-only runs, sizing iteration in optimisation loops).
8. **Downstepping is common.** The 45-zone VAV run logs 58,565 system timesteps for
   33,614 zone timesteps (1.74 per zone timestep); every downstepped zone timestep repeats
   `SimHVAC` for each sub-step and discards the full-step solve (section 4.6 item 3).

These numbers are reproduced by `scripts/dev/perf/bench.sh` (Phase 0 deliverable); the
benchmark set is section 7.1.

## 3. Guiding principles ##

1. **Measure before and after.** No optimisation lands without a before/after number on the
   benchmark set and a diff classification from `energyplus-regressions`.
2. **`--threads 1` is bit-identical to today.** All serial optimisations in Phases 1 and 4a
   must produce identical ESO/MTR/SQL output; behaviour-changing algorithms are opt-in
   through input fields.
3. **`--threads N` is bit-identical to `--threads 1`.** Parallel loops write per-item results
   into pre-sized arrays; every reduction (zone sums, enclosure sums, max-ΔT) is performed
   serially in the original order after the join. Floating-point atomics and unordered
   reductions are forbidden.
4. **The physics chooses the unit of work.** Surface for conduction and convection, radiant
   *enclosure* (not zone: enclosures span air-boundary zones, `DataViewFactorInformation.hh:64-96`)
   for long-wave exchange, space/zone for air and zone equipment, air loop for the air
   system, *connected cluster* for plant, building-global for AirflowNetwork and electric
   service.
5. **Do the serial work first.** Amdahl's law says a 20% serial residue caps speedup at 5×.
   Reporting, error handling, AFN and electric service are serial today; they are dealt with
   before the parallel loops are widened.
6. **Per-instance, not per-process.** Thread pools, scratch buffers and caches hang off
   `EnergyPlusData`, so the C/Python API (`api/state.cc`) keeps working with several
   instances in one process. `thread_local` is avoided for simulation data.
7. **Small, upstreamable changes.** This fork tracks NREL `develop`; each step is a PR-sized
   change (bug fix, serial optimisation, infrastructure behind an option, one parallel
   region) to keep the merge burden low and to give NREL the option to adopt pieces.

## 4. The physics: decomposition and coupling ##

### 4.1 Envelope conduction and inside-surface heat balance ###

*How it is solved today.* `CalcHeatBalanceOutsideSurf` (`HeatBalanceSurfaceManager.cc:7243`)
is a single sweep over all heat-transfer surfaces using previous-timestep inside
temperatures. `CalcHeatBalanceInsideSurf2` / `…2CTFOnly` (`:8130`, `:8964`) iterate: long-wave
exchange → per-surface update reading `SurfTempInsOld` (a **Jacobi** update, `:8295-8611`)
→ interzone copy of the partner's new inside temperature into this surface's outside
history (`:8797-8805`) → global max-ΔT reduction (`:8810-8821`) against
`MaxAllowedDelTemp` = 0.002 K. Windows are solved only at iteration 0 (`:8670`).

*What that means physically.* Within one iteration every surface depends only on the
previous iterate. Adjacent zones see each other one iteration late. The building-wide
convergence test couples all enclosures only through the stopping rule.

*Decomposition.* Parallel over spaces (contiguous surface ranges
`Space.HTSurfaceFirst..Last`, `DataHeatBalance.hh:482-494`) inside each iteration, with
three serial phases per iteration: (a) the interzone copy, (b) the radiant-system and
TDD cross-writes (`:8529-8548`, `:8766-8782`), (c) the max-ΔT reduction. Because the
update is already Jacobi, this changes nothing numerically. The per-iteration region is
small (hundreds of microseconds for a 1000-surface model), so it needs a persistent pool
with cheap fork/join (section 5.1), and the whole-array copies at `:8233-8236` must go first
(section 5.3).

*Later option (behaviour change).* Iterate each enclosure group (enclosures joined by
interzone surfaces) to its own convergence and couple groups once per timestep. This is
physically defensible (interzone coupling is already lagged) and removes the per-iteration
barrier, but it changes results; it goes behind a `PerformancePrecisionTradeoffs` field in
Phase 6.

### 4.2 Long-wave radiant exchange ###

`CalcInteriorRadExchange` (`HeatBalanceIntRadExchange.cc:105-426`) costs
Σ_enclosures N_e² per call with ScriptF, and is called 2 + N_insideIter times per timestep
("Main" at `HeatBalanceSurfaceManager.cc:510`, "Outside" at `:7336`, "Inside" per
iteration). Enclosures are independent: each surface belongs to exactly one radiant
enclosure and the writes to `NetLWRadToSurf` / `SurfWinIRfromParentZone` are disjoint. The
only obstacles are the shared scratch arrays `SurfaceTempRad`, `SurfaceTempInKto4th`,
`SurfaceEmiss` (`:142-150`) and the per-call `N×N` heap allocation in `CalcScriptF`
(`:1803`, `CalcMatrixInverse :1902`).

The "Outside" call is a candidate duplicate of "Main": both pass `SurfInsideTempHist(1)`
and iteration 0, and nothing between them changes surface temperatures. Daylighting
(`initDaylighting :508`, `manageDaylighting :525`) can change window shade state between the
two calls, so the fix is "skip the second call when no shade/insulation state changed",
which keeps results identical.

`CarrollMRT` (O(N) per enclosure) already exists as an input option and is the right
recommendation for very large enclosures; it is not a substitute for parallelism because it
changes results.

### 4.3 Solar, shading and daylighting ###

Shadowing is precomputed for 24 h × timesteps every 20 days (`SolarShading.cc:463-474`) and
looked up per timestep; the expensive part is the recompute day, in which each sun position
is independent. `SHADOW` (`:5762-5951`) uses module-global clipping stacks (`HCX/HCY/HCA…`,
`SolarShading.hh:380-463`), so the natural parallel unit is *the hour/timestep* with a
thread-private clipping workspace; alternatively the receiving surface. `PixelCounting`
(Penumbra, GPU) exists but is bound to the OpenGL context thread; it stays on the main
thread.

Per timestep, `CalcInteriorSolarDistribution` (`:6543-8646`) accumulates into other
surfaces and other enclosures (`SurfOpaqAI(BackSurfNum) += …`, `EnclSolDBIntWin(adj) += …`),
so it is a per-*solar enclosure* unit with a serial merge of the interzone-window terms.
Daylighting coefficients (`DaylightingManager.cc:325`, ray tracing per control × window ×
sun position) are independent per daylighting control.

`ComputeDifSolExcZonesWIZWindows` (`HeatBalanceSurfaceManager.cc:4685-4856`) runs every
timestep when any interzone window exists and nests up to five loops over enclosures
(O(N_encl⁵) worst case). It should be restructured to a sparse graph traversal before anyone
worries about threading it.

### 4.4 Zone air heat balance ###

`PredictSystemLoads` (`ZoneTempPredictorCorrector.cc:2913`, per-zone loop `:3155-3167`) and
`correctZoneAirTemps` (`:3874`) treat zones independently within a step: interzone mixing and
AFN flows are fixed inputs computed before the predictor (`HVACManager.cc:243-247`). Parallel
over zones is safe once the two accumulations flagged in the code map are made idempotent:
`SurfWinHeatGain += SurfWinRetHeatGainToZoneAir` inside `calcSumHAT` (`:5405`) and
`ZoneWinHeatGain +=` in `UpdateIntermediateSurfaceHeatBalanceResults`
(`HeatBalanceSurfaceManager.cc:5216`). Both are pre-existing double-counting hazards on
re-simulation and should be fixed regardless.

### 4.5 HVAC: air loops, zone equipment, plant ###

*Structure.* Zone air temperatures are **frozen** inside `SimHVAC`; the loop is a nested
Gauss-Seidel fixed point over {air loops, zone equipment} × plant × electric, driven by
node-tolerance flags (`HVACInterfaceManager.cc:87-458`: 0.01 kg/s, 0.01 K, 10 W, …).

*Independent work inside one pass.*

 - **Air loops** (`SimAirLoops`, `SimAirServingZones.cc:2523-2591`): node sets are disjoint;
   controllers and root finders are per-controller. Couplings to the rest of the model are
   (a) plant demand-node requests via `SetComponentFlowRate` (`PlantUtilities.cc:124-295`,
   465 call sites), which write per-node data plus an OR into `SimLoopSideNeeded`, (b) the
   zone-side interface nodes, (c) a DOAS second sweep (`:2593-2678`) and (d) AirflowNetwork.
   Air loops can run concurrently within a pass once the ~20 "current object" globals they
   write are made per-thread (section 5.2).
 - **Zone equipment** (`SimZoneEquipment`, `ZoneEquipmentManager.cc:3663-4149`): zones are
   independent after the forward supply-path pass; equipment *within* a zone is sequential by
   design (sequential load distribution, `:4139-4145`). Cross-zone couplings that force
   serial groups: VRF terminal units sharing a condenser (run when the *last* TU is
   simulated, `HVACVariableRefrigerantFlow.cc:236-278`), refrigeration chiller sets,
   heat-pump water heaters sharing tanks, exhaust systems, zones served by several air
   loops. A one-time dependency analysis at initialisation builds "zone groups" that are
   scheduled as units.
 - **Plant** (`ManagePlantLoops`, `PlantManager.cc:149-276`): the half-loop is *not* the unit
   (`SetupLoopFlowRequest` scans both sides; `UpdatePlantLoopInterface` writes the sibling).
   The unit is the **connected cluster**: loops linked through `LoopSide.Connected[]`
   (`InterConnectTwoPlantLoopSides`, `PlantUtilities.cc:1175-1228`, registered by chillers,
   heat exchangers, heat-recovery and water-to-water heat pumps). A hospital with
   HW, CHW, CW, SHW and heat-recovery loops typically has 2–3 clusters. Within a cluster
   the revised calling order (`RevisePlantCallingOrder :3638-3740`) is kept. Clusters are
   solved concurrently, sweep by sweep, with the `SimAirLoopsNeeded/SimZoneEquipNeeded`
   flags OR-merged after each sweep.
 - **Serial stages.** AirflowNetwork (global network; skyline LU plus a dense O(N³) inverse
   for the thermal/contaminant balances, `AirflowNetwork/src/Solver.cpp:8664`), electric
   service (reads 7 facility meters, `ElectricPowerServiceManager.cc:111-190`), EMS and
   Python-plugin calling points, and reporting.

*Pipelining plant against air (Jacobi across stages)* would give more concurrency but
changes numerics and probably iteration counts. It is a Phase 6 option.

### 4.6 Doing less HVAC work (multi-rate and iteration control) ###

These are physics/numerics changes, mostly serial, and several are bigger wins than threads:

1. **Warm starts.** Every system timestep cold-starts: air-loop flows are reset to design
   flow on the first HVAC iteration (`SimAirServingZones.cc:2224-2228`, `:2264-2287`),
   `ResetTerminalUnitFlowLimits` runs, controllers cold-start (`:2917-2934`) and plant
   `MassFlowRateRequest` is zeroed (`PlantManager.cc:2670`, `~:2958`). Starting from the
   previous system timestep's converged state (with the existing rigidity/convergence
   histories as guards) should cut the first-pass and total iteration counts.
2. **Selective re-simulation.** `ManageAirLoops` re-runs *every* air loop when any needs it
   (`SimAirServingZones.cc:179-182`); `ManageZoneEquipment` re-runs every zone. Per-loop
   convergence records already exist (`AirLoopConvergence(AirLoopNum)`,
   `ZoneInletConvergence`). Re-simulate only loops/zones whose interface nodes moved.
3. **Reuse the discarded solve on downstep.** When `ZoneTempChange > MaxZoneTempDiff`
   forces N sub-steps, the full-step `SimHVAC` result is thrown away. Use it as the warm
   start for sub-step 1, and consider a per-zone (rather than building-wide) downstep
   decision as an option.
4. **Plant forced minimum sweeps.** 2 (or 7 with a common pipe) full sweeps every call and an
   unconditional second `ManagePlantLoops` at `HVACManager.cc:875`. Replace with
   convergence-based early exit guarded by the existing interconnect criteria.
5. **Controller iterate scope.** `SolveAirLoopControllers` re-simulates the whole air loop per
   controller iterate (`:3017`). Simulate from the controller's sensor/actuator span
   downstream only, when the loop topology allows (most OA-system and coil controllers).
6. **Diagnostics that run unconditionally.** `UpdateZoneInletConvergenceLog` every
   iteration (`HVACManager.cc:2989-3019`), the electric-service meter scans twice per
   iteration, `InitLoadDistribution` from every half-loop (`Plant/LoopSide.cc:109-125`).

Items 1, 2 and 4 change results slightly (they change where the fixed-point iteration
stops) and are therefore delivered behind `PerformancePrecisionTradeoffs` fields with
"Small Diffs"-level validation; item 3's warm start and items 5–6 can be made result-neutral.

### 4.7 Time parallelism ###

The zone-sizing design days are independent simulations from identical initial state, and
the annual run is one long dependent chain (warm-up, history). Running design days on
separate threads requires a cloneable `EnergyPlusData` (blocked today by
`AirflowNetwork::Solver` holding a reference to the state, `Solver.hpp:734`, and by raw
`Which` pointers in every output variable, `OutputProcessor.hh:~482`). Splitting a run
period into chunks with re-warm-up is a known approximation used by other tools; both are
Phase 6 research.

## 5. The software: runtime, hardening and serial wins ##

### 5.1 Threading runtime ###

A small facility, `EnergyPlus::Parallel`, owned by the state (`state.dataParallel`):

 - `parallel_for(begin, end, grain, fn(i, tid))` with static partitioning and a persistent
   pool of `--threads - 1` workers plus the calling thread. Workers spin briefly then park,
   so fork/join costs a few microseconds; inside-HB iterations issue hundreds of
   sub-millisecond regions per timestep and the old per-region OpenMP spawn is exactly what
   made the V7–V8 interior-radiation experiment unprofitable.
 - `ThreadContext` array (one per worker): scratch buffers (window solver, shading clipping
   stacks, radiant-exchange temperaries, `TmpRealARR`, `PrioritySimOrder`), the "current
   object" register (`CurSysNum`, `CurZoneEqNum`, `CurBranchNum`, `CurOASysNum`,
   `TurnFansOn/Off`, `OnOffFanPartLoadFraction`, module `CoolingLoad/HeatingLoad`), a
   psychrometric/glycol cache slice, and an error queue.
 - `parallel_reduce` is deliberately absent; callers store per-item values and reduce serially.
 - Backend: `std::thread` + condition variable/atomics. OpenMP is not used because MSVC ships
   OpenMP 2.0, because an OpenMP runtime is process-global (bad for multi-instance API use),
   and because we want explicit ownership of thread contexts. `Threads::Threads` is already
   linked (`CMakeLists.txt:82-84`).
 - Control: `--threads N` (alias of the existing dead `-j/--jobs`, `CommandLineInterface.cc:172-194`,
   `dataGlobal->numThread`), env `ENERGYPLUS_NUM_THREADS`, and `api/runtime.h` setter. Default 1
   until Phase 5 exit criteria are met, then `min(hardware, 8)`.
 - `ENABLE_SANITIZER_THREAD` (`cmake/Sanitizers.cmake`) becomes a CI configuration for the
   unit tests and a short list of integration files.

### 5.2 Thread-safety hardening (prerequisites, each valuable alone) ###

| Area | Today | Change |
|---|---|---|
| Psychrometrics caches (`PsychCacheData.hh:162-176`, ~96 MiB per instance) | Every call may write shared entries; `PsyTsatFnHPb` (`Psychrometrics.hh:1077-1117`) and `PsyTsatFnPb` (`:1493-1515`) store values computed at the *exact* input, so the cached value depends on call order | Compute from the quantised key (removes order dependence, tiny result change to be validated) or drop those two caches; per-thread cache slices sized by measurement (e.g. 2¹⁶ entries) for the hot `Twb`/`Psat` caches; memos `last_Patm/Press_Save` (`:1336-1339`) into `ThreadContext` |
| Psychrometric / glycol error counters (`iPsyErrIndex :1707`; `FluidProperties.hh:183,356`) | Shared `int&` indices into recurring-error table | Per-thread error queue (below) |
| Glycol `getSpecificHeat` cache (`FluidProperties.cc:4064-4094`), search hints `Lo*TempIdxLast` (`.hh:313-352`) | Order-dependent cache; racy read-modify-write hints | Same treatment as psychrometrics; hints per thread |
| `Curve::value` (`CurveManager.cc:147-233`) writes `inputs[]`/`output` and calls `commonEnvironInit` per call | Last-writer-wins report values; env reset on first call | `const` evaluation path for hot loops; per-curve report value written from the serial phase; env reset moved to `BeginEnvrn` |
| btwxt tables (`:2894-2970`) mutate interpolator internals and a **static** logger (`CurveManager.cc:114`) with a dangling stack pointer | Race and per-call allocations | Per-thread interpolator copies (lazy) or upstream a `const` evaluate; remove the static logger |
| Error reporting (`UtilityRoutines.cc:832-1400`) | `++Total*Errors`, linear search + `redimension` in `StoreRecurringErrorMessage`, unbuffered `err` stream, SQLite "append to newest row" UPDATE (`SQLiteProcedures.cc:1215-1219`) | Inside a parallel region messages go to the thread's queue; at the join they are flushed in item order, so recurring indices, counts and the err text are identical to serial. Fatal errors set a flag; the region exits and the main thread throws. `BufferedErrFile` behaviour becomes the default with flush on severe/fatal |
| Module scratch used as globals: `dataWindowManager` (`WindowManager.hh:472-509`), TARCOG (`TARCOGMain.hh:177-187`), shading stacks (`SolarShading.hh:380-463`), `dataHeatBalIntRadExchg` temporaries, `dataSimAirServingZones` (`SimAirServingZones.hh:243-327`), `TmpRealARR`, `PrioritySimOrder` (`ZoneEquipmentManager.hh:277`), `ControlCompOutput` scratch (`GeneralRoutines.hh:224-225`) | Written by every call | Move into per-call structs or `ThreadContext` |
| "Current object" globals: `dataSize->CurSysNum` (221 uses), `CurZoneEqNum` (340), `CurOASysNum` (53), `CurBranchNum`, `DataCoolCoilCap`, `TurnFansOn/TurnFansOff/NightVentOn`, `OnOffFanPartLoadFraction` (80), `dataUnitarySystems->CoolingLoad/HeatingLoad` (96/90) | Implicit arguments | Route through `ThreadContext` behind accessor functions first (mechanical refactor), then thread it |
| Root finder algorithm (`state.dataRootFinder->rootAlgo`) swapped by `SolveRoot2` (`General.cc:~330`) and `ExtendedHeatIndex.cc:545-556`; `thread_local SolveRootStats` (`WaterCoils.cc:5840`) makes results depend on call history | Race and non-determinism | Pass the algorithm as an argument; make the adaptive statistics per component |
| True process globals: `WaterUse.cc:1727` static `MyEnvrnFlagLocal` (shadows the state member), `SQLiteProcedures.cc:1572-1574` static `lastDayOfMonth` mutated for leap years, PCM singleton (`PCMThermalStorage.cc:81,308,461`), `rand()` in AFN (`Solver.cpp:13389`), ObjexxFCL RNG, WCE gas singletons, Kiva `MEMOIZE` static maps (`third_party/kiva/src/libkiva/Algorithms.hpp:13-28`) | Bugs and races | Fix in Phase 0/2; Kiva memoisation replaced by per-instance caches (upstream PR) |
| Reporting (`UpdateDataandReport`, `OutputProcessor.cc:3335-3834`) | Single writer, `dynamic_cast` per variable per call, `std::format` string per value, one `sqlite3_step` per value | Snapshot values on the main thread in fixed order (cheap), format/write/SQLite on a dedicated writer thread with a bounded queue; file order unchanged |
| EMS / Python plugin calling points | Arbitrary user code | Components with EMS actuators/sensors or plugin hooks in a parallel stage force their group onto the calling thread (dependency analysis) |

### 5.3 Serial optimisations (bit-identical, Phase 1) ###

Ordered by expected value / risk; each gets its own PR and benchmark row.

1. Skip the duplicate "Outside" `CalcInteriorRadExchange` when nothing changed (4.2).
2. Remove the O(TotSurfaces) whole-array copies per inside iteration
   (`HeatBalanceSurfaceManager.cc:8233-8236`, `:9115-9116`) and the `TotSurfaces` zeroing of
   `NetLWRadToSurf` per call (`HeatBalanceIntRadExchange.cc:189-192`) in favour of
   per-enclosure ranges; hoist `surfTempForRadiation` (`:8229`) out of the per-call heap.
3. Move the `allInsideSourceSurfaceList` loop out of the space loop in the CTF-only path
   (`:9073-9076`).
4. Pre-size `CalcScriptF` matrices per enclosure (`:1803`, `:1902`).
5. `WindowManagerExteriorThermal.cc:100-103` builds a Tarcog system per window per call;
   cache per construction. `DayltgInteriorIllum` per-call nested allocations
   (`DaylightingManager.cc:5953-5956`).
6. `ComputeDifSolExcZonesWIZWindows` O(N_encl⁵) → sparse traversal (4.3).
7. Reporting: replace `dynamic_cast<OutVarReal*>` per variable with a typed pointer or
   `std::variant` (3.5% of the annual run); write ESO/MTR values with `dtoa` straight into
   a buffer instead of a `std::format` temporary; batch `ResultsFramework` map lookups;
   compute `CalcMoreNodeInfo` (`NodeInputManager.cc`, 3.8%) only for nodes whose report
   variables are requested or metered, and only at the reporting frequency.
8. Electric service: cache the 7 facility-meter sums per iteration instead of scanning all
   meter sources twice per iteration.
9. `UpdateZoneInletConvergenceLog` only when `DisplayExtraWarnings` or the not-converged
   path needs it; `InitLoadDistribution` once per timestep instead of per half-loop.
9a. Sizing: `BaseSizer::initializeWithinEP` (`Autosizing/Base.cc`) copies whole
    `ZoneSizingData` / `TermUnitZoneSizingCommonData` structs on every sizer call — 10% of
    the hospital design-day run in the profile (`ObjexxFCL::Array` copy constructor 4%
    exclusive). Hold references instead of copies.
9b. Water-coil control: `ControlCompOutput` (`GeneralRoutines.cc:126`) interval-halves the
    reheat coil water flow with up to 25 coil evaluations per terminal per call (11.5% of the
    hospital profile). A secant/Brent step on the coil's near-linear capacity-vs-flow curve,
    or a direct solution for the simple heating coil (the analogue of "Use Coil Direct
    Solutions" for DX coils), removes most evaluations. The tolerance-equivalent version is
    result-neutral to within the existing controller tolerance and is validated as "Small
    Diffs"; the exact-tolerance version (same stopping rule, better bracketing) is
    bit-identical only if it lands on the same iterate, so it is measured, not assumed.
10. Weather: avoid the daily full-array copy `wvarsHrTsToday = wvarsHrTsTomorrow`
    (`WeatherManager.cc:2009`) and the three `std::format` date strings per timestep
    (`:2075-2079`).
11. Build flags: turn on IPO/LTO for Release (`cmake/CompilerFlags.cmake` has none), evaluate
    `-march=x86-64-v3` for the fork's own builds (with `-ffp-contract=off` already set,
    vectorisation does not reassociate, so results should stay identical; verify with the
    regression suite), and a PGO pipeline trained on the benchmark set. Fix the Kiva
    `-fopen` typo (`third_party/kiva/src/libkiva/CMakeLists.txt:191-194`) so `ENABLE_OPENMP`
    means something.

### 5.4 Parallel regions, in order of introduction ###

| # | Region | Unit | Writes | Serial phases needed | Phase |
|---|---|---|---|---|---|
| P1 | `UpdateThermalHistories` (`HeatBalanceSurfaceManager.cc:5517-5836`) | surface | own history arrays | none | 2 (pilot) |
| P2 | `CalcInteriorRadExchange` enclosure body (`HeatBalanceIntRadExchange.cc:195-414`) | radiant enclosure | disjoint `NetLWRadToSurf` | ScriptF recompute on shade change (allocation) | 2 (pilot) |
| P3 | `CalcHeatBalanceOutsideSurf` main loop (`:7344-8011`) | surface | own outside arrays | OSC/OSCM, vented cavity, EcoRoof shared objects grouped | 3 |
| P4 | Inside-HB per-iteration kernel (`:8295-8611`, `:9172-9495`) | space | own inside arrays | interzone copy, radiant/TDD cross-writes, max-ΔT | 3 |
| P5 | `CalcWindowHeatBalance` (`WindowManager.cc:2059`) | window | own window arrays | after de-globalising `dataWindowManager`/TARCOG/WCE | 3 |
| P6 | `InitIntConvCoeff` / `InitExtConvCoeff` (`ConvectionCoefficients.cc:142`, `:394`) | surface | `SurfHConvInt/Ext` | one-time input fetches hoisted | 3 |
| P7 | Shadow recompute (`CalcPerSolarBeam`, `SolarShading.cc:4913`) | hour × timestep | `SurfSunlitFrac(hr,ts,·)`, back-surface arrays | thread-private clipping stacks; Penumbra stays on main thread | 3 |
| P8 | Daylighting coefficients (`DaylightingManager.cc:745`, `:1097`) | daylighting control | own factors | `sunAngles`/flags per thread | 3 |
| P9 | Kiva instances (`HeatBalanceKivaManager.cc:1158`) | instance | own domain | Kiva memoise maps fixed | 3 |
| P10 | Zone predictor/corrector (`ZoneTempPredictorCorrector.cc:3155`, `:3874`) | zone | own zone arrays | idempotent window-gain fix | 5 |
| P11 | `SimAirLoops` (`SimAirServingZones.cc:2523`) | air loop | own nodes, plant requests (per node) | DOAS sweep, AFN, flag OR-merge | 5 |
| P12 | `SimZoneEquipment` zone loop (`ZoneEquipmentManager.cc:3663`) | zone group | own nodes/equipment | supply paths, exhaust, mass balance, return paths | 5 |
| P13 | `ManagePlantLoops` sweep (`PlantManager.cc:203-235`) | connected cluster | own loops | flag OR-merge between sweeps | 5 |
| P14 | Reporting writer (`OutputProcessor.cc:3335`) | — | files/SQLite | snapshot on main thread | 5 |

## 6. Phased roadmap ##

Effort is in engineer-weeks for one experienced EnergyPlus developer; ranges reflect the
unknowns that Phase 0 will resolve.

### Phase 0 — Measure (2–3 weeks) ###

 - Scoped `steady_clock` timers behind the already-parsed but unused `TimingFlag`
   (`DataSystemVariables.cc`) for: weather, outside HB, inside HB (and iteration count),
   radiant exchange, windows, convection, solar/shading, daylighting, predictor/corrector,
   air loops, zone equipment, plant, AFN, electric, reporting. Written to `eplusout.perf`
   (JSON) and summarised in the eio. Wire `EP_Count_Calls` (`DataTimings.hh`) into CMake.
 - Benchmark set and script (`scripts/dev/perf/`), annual runs, wall time plus the existing
   iteration-count output variables ("HVAC System Solver Iteration Count", "Air System
   Solver Iteration Count", "Plant Solver Sub Iteration Count"). Nightly job on the
   self-hosted runner (`self_hosted_build_and_test.yml`) with `BUILD_PERFORMANCE_TESTS=ON`,
   `TEST_ANNUAL_SIMULATION=ON`, `PERF_STAT_ANALYZE_PERFORMANCE_TESTS=ON`; results archived
   as CSV; alert on >5% regressions.
 - Fix the true-global bugs (5.2, "True process globals" row) and the two non-idempotent
   accumulations (4.4).
 - Exit: baseline table (section 2.2) filled from CI, not a laptop; profile top-20 per model.

### Phase 1 — Serial wins, bit-identical (4–6 weeks) ###

Items 5.3.1–5.3.11 (9b's "Small Diffs" variant is deferred to Phase 4). Exit: zero diffs on
the full regression suite; ≥15% wall-time reduction
on envelope-heavy models (OutPatient, 100-zone overhang file) and ≥10% on HVAC-heavy models
(Hospital, 45zonevav); LTO/PGO numbers reported separately.

### Phase 2 — Threading runtime and hardening (6–8 weeks) ###

`EnergyPlus::Parallel`, `ThreadContext`, `--threads`, per-thread error queues, psychrometric
and glycol cache changes, curve/btwxt `const` evaluation, "current object" accessor refactor
(mechanical, large diff, no behaviour change), TSan CI job. Pilot regions P1 and P2.
Exit: `--threads 1` bit-identical to Phase 1; `--threads N` bit-identical to `--threads 1`
on the whole regression suite; TSan clean on the pilot regions; P2 shows ≥3× on the
radiant-exchange timer at 4 threads for the hospital models.

### Phase 3 — Envelope parallelism (8–10 weeks) ###

Regions P3–P9. Exit: envelope timers scale ≥0.6× linear to 8 threads on
`ASHRAE901_OutPatientHealthCare` (1400 surfaces) and the 100-zone overhang model; whole-run
speedup ≥1.5× at 8 threads on those two; no change on HVAC-heavy models beyond noise;
bit-identical outputs.

### Phase 4 — HVAC iteration reduction (8–12 weeks) ###

Items 4.6.1–4.6.6, result-neutral parts first (downstep warm start, diagnostics, controller
scope), then the opt-in ones behind new `PerformancePrecisionTradeoffs` fields
("HVAC Warm Start", "Selective Air Loop Resimulation", "Plant Convergence-Based Minimum
Iterations"). Exit: ≥30% fewer `SimSelectedEquipment` component evaluations on the hospital
models with the options on, differences classified "Small Diffs" or better, and
oscillating-hour counts (already tracked by `_perflog.csv`) not worse than baseline.

### Phase 5 — HVAC/plant parallelism and asynchronous reporting (10–14 weeks) ###

Regions P10–P14 with the zone-group and plant-cluster dependency analyses, then flip the
default thread count. Exit: ≥2.5× whole-run speedup at 8 threads on `HospitalBaseline`,
`ASHRAE901_Hospital` and `ASHRAE901_OfficeLarge`; bit-identical outputs; no more than 5%
slowdown at `--threads 1` versus Phase 4.

### Phase 6 — Research options ###

Per-enclosure-group convergence (4.1), Jacobi pipelining of plant against air (4.5),
cloneable state and design-day parallelism (4.7), a sparse solver for the AFN thermal and
contaminant balances (replace `mrxinv`, `Solver.cpp:8664`, with the existing skyline LU or
Eigen sparse), GPU radiant exchange for very large enclosures.

## 7. Testing and validation ##

### 7.1 Benchmark set ###

Annual runs unless noted; all present in the repository.

 - Envelope-heavy: `performance_tests/10x_incr_100zones1win10overhangsFullExIn.idf`,
   `testfiles/ASHRAE901_OutPatientHealthCare_STD2019_Denver.idf`,
   `testfiles/ASHRAE901_ApartmentHighRise_STD2019_Denver.idf` (230 zones, 80 air loops).
 - HVAC/plant-heavy: `testfiles/HospitalBaseline.idf`, `testfiles/HospitalLowEnergy.idf`
   (4 plant + 4 condenser loops), `testfiles/ASHRAE901_Hospital_STD2019_Denver.idf`,
   `testfiles/ASHRAE901_OfficeLarge_STD2019_Denver.idf`, `performance_tests/45zonevav.idf`,
   `performance_tests/BenchmarkHospitalNew_USA_CA_SAN_FRANCISCO.idf`.
 - Reporting-heavy: `15zonevav.idf` vs `15zonevav_no_reports.idf`, and
   `RefBldgMediumOfficeNew2004_Chicago_JSON_Outputs.idf`.
 - Special features: an AirflowNetwork file, a Kiva file (`ZoneCoupledKivaRefBldgMediumOffice`),
   a daylighting + `PixelCounting` file, a Python-plugin file.

### 7.2 Correctness gates ###

 - Full regression run through `scripts/dev/gha_regressions.py` against the pre-change
   baseline for every PR (already in `test_pull_requests.yml`); "Big Diffs" fail; text diffs
   in err/eio/audit fail for Phases 1–3 and 5.
 - New CI job: run the benchmark subset with `--threads 1` and `--threads 4`, assert
   byte-identical ESO/MTR/SQL/err; run twice at `--threads 4` to catch scheduling
   non-determinism.
 - `ENABLE_SANITIZER_THREAD` job for unit tests plus the benchmark subset design-day runs.
 - `ENABLE_REVERSE_DD_TESTING` (`cmake/RunReverseDD.cmake`) on the benchmark subset: it detects
   order-dependent state, which is exactly what per-thread caches and warm starts can introduce.
 - Unit tests: `Parallel` fork/join and ordering, per-thread error flush order and recurring
   index assignment, psychrometric cache equivalence (quantised vs exact), dependency
   analyses for zone groups and plant clusters on synthetic topologies.

### 7.3 Performance gates ###

Nightly benchmark on the self-hosted runner; wall time (median of 3), iteration counts, and
the scoped timers; regressions >5% block the merge queue; speedups are recorded in the
design document's changelog.

## 8. Input, output and API changes ##

 - **Command line:** `--threads N` (the existing `-j/--jobs` becomes its alias; help text
   updated), env `ENERGYPLUS_NUM_THREADS`, `--timings`.
 - **API:** `setNumberOfThreads(state, n)` in `api/runtime.h`; documented that each
   `EnergyPlusData` instance owns its pool.
 - **IDD (Phase 4 only):** new optional fields on `PerformancePrecisionTradeoffs` — `HVAC Warm
   Start` (Yes/No, default No), `Selective Air Loop Resimulation` (Yes/No, default No), `Plant
   Minimum Iterations Mode` (Fixed/ConvergenceBased, default Fixed). Appending fields needs no
   transition rule; the Input Output Reference and Engineering Reference sections on HVAC
   convergence are updated accordingly.
 - **eio:** the "Program Control Information:Threads/Parallel Sims" report
   (`SimulationManager.cc:1924-1987`) prints the real thread count and pool size instead of
   the hard-coded "No"; `eplusout.perf` timing summary.
 - **Output variables:** "Surface Heat Balance Inside Iteration Count" (zone timestep),
   alongside the existing HVAC/plant iteration counters.

## 9. Risks and mitigations ##

| Risk | Mitigation |
|---|---|
| Amdahl: reporting, AFN, electric and EMS stay serial | Async reporting writer (P14) and the serial reporting optimisations first; measure the serial residue per model in Phase 0 and publish it with the speedup |
| Floating-point non-determinism from parallel reductions | No reductions in parallel regions; per-item stores plus serial ordered sums; CI double-run check |
| Memory growth from per-thread caches (psychrometrics is ~96 MiB per instance today) | Smaller per-thread slices sized from hit-rate measurements; shared read-mostly tables for the order-independent caches |
| Third-party code that is not thread-safe (Kiva memoisation, WCE singletons, Penumbra GL context, btwxt logger) | Kiva/btwxt fixes upstreamed; WCE evaluated per thread; Penumbra pinned to the main thread |
| EMS/Python callbacks inside parallel stages | Dependency analysis forces affected groups serial; documented limitation |
| Regression tool compares err/eio as text | Deterministic error flush order makes the files identical; where message *timing* legitimately changes (warm starts), those PRs are reviewed with the text diffs explained |
| Divergence from NREL `develop` | PR-sized changes; infrastructure behind an option; bug fixes and serial optimisations offered upstream first |
| Effort under-estimate on the "current object" refactor (~1000 use sites) | Do it mechanically with clang-tidy/regex tooling in one PR per module; it is a no-behaviour-change refactor and can be split by module |

## 10. Appendix A — shared-state inventory for the envelope path ##

Collected from `HeatBalanceSurfaceManager.cc`, `HeatBalanceIntRadExchange.cc`,
`WindowManager.cc`, `ConvectionCoefficients.cc`, `SolarShading.cc`, `DaylightingManager.cc`.

 - **Scratch used as globals:** `dataHeatBalIntRadExchg->SurfaceTempRad/SurfaceTempInKto4th/SurfaceEmiss`
   (`HeatBalanceIntRadExchange.cc:142-150`); `dataWindowManager` per-call scalars and arrays
   (`WindowManager.hh:472-509`); TARCOG module state (`TARCOGMain.hh:177-187`); shading
   clipping stacks and counters (`SolarShading.hh:380-463`); `SurfWinAbsBeam*`
   (`SolarShading.cc:7704`); `dl->DaylIllum`, `dl->sunAngles` (`DaylightingManager.cc:5947-5950`, `:2922`).
 - **Counters/flags mutated in loops:** `InsideSurfIterations` (`HeatBalanceSurfaceManager.cc:8807`),
   `CondFDRelaxFactor` (`:8834-8841`), `calcHeatBalInsideSurfErrCount` (`:8856`),
   `dataThermalComforts->ZoneNum` (`:5979`), ~25 convection recurring-error indices
   (`ConvectionCoefficients.cc:5447-5520`), `dataHVACGlobal->ShortenTimeStepSysRoomAir`
   written from `predictSystemLoad` (`ZoneTempPredictorCorrector.cc:3237`).
 - **Accumulations needing ordered reductions:** `ZoneAESum`, `EnclRadInfo.sumAE/sumAET`
   (`:5863-5876`, `:5905`); `ZoneWinHeatGain` (`:5216`); `zoneHeatBalance.SumHmA*` (`:8909-8958`);
   `ZoneOpaqSurf*FaceCond` (`:7158`); `SumSurfaceHeatEmission` (`:6965`); `EnclSolQSWRad`,
   `EnclSolQSDifSol` (`:3146-3158`, `:4031-4047`); `SurfOpaqAI(back/base)` and
   `EnclSolDBIntWin(adj)` (`SolarShading.cc:7669`, `:8096`).
 - **Cross-surface writes (ordering):** radiant-system interzone coefficient copies
   (`:8534-8548`, `:9293-9307`), TDD dome/diffuser (`:8766-8782`), interzone TH11 update
   (`:8797-8805`), OSC/OSCM objects shared by several surfaces (`:7507-7545`), `ExtVentedCavity`
   (`:10060-10123`), EcoRoof module scalars (`EcoRoofManager.hh:100-147`), Kiva
   `surfaceConvMap[SurfNum]` (`operator[]` may insert), `QRadSysSource(partner)`
   (`LowTempRadiantSystem.cc:3826-3828`).
 - **Error calls inside hot loops:** `TestSurfTempCalcHeatBalanceInsideSurf` (`:9586-9763`,
   including `ShowFatalError`), inside-HB non-convergence (`:8854-8885`), `CalcScriptF`
   emissivity warning (`HeatBalanceIntRadExchange.cc:1862`), window solver
   (`WindowManager.cc:3602-3620`), WCE (`WindowManagerExteriorThermal.cc:123`), convection
   correlations, psychrometric range warnings.
 - **Latent bugs found on the way:** `TestSurfTempCalcHeatBalanceInsideSurf` takes
   `WarmupSurfTemp` by value so its increment is lost (`:9586`); the CTF-only non-convergence
   warning prints `MaxAllowedDelTempCondFD` instead of `MaxAllowedDelTemp` (`:9557`); the IDD
   note for `MaxAllowedDelTemp` says it shortens the HVAC timestep but it is the inside-surface
   convergence tolerance (`:8824`, `:9536`).

## 11. Appendix B — shared-state inventory for the HVAC path ##

 - **Implicit arguments:** `dataSize->CurSysNum` (221 uses), `CurZoneEqNum` (340),
   `CurOASysNum` (53), `CurBranchNum`, `CurDuctType`, `CurTermUnitSizingNum`, `DataCoolCoilCap`;
   `dataHVACGlobal->TurnFansOn/TurnFansOff/NightVentOn`, `OnOffFanPartLoadFraction` (80),
   `DXElecCoolingPower` and siblings (`DataHVACGlobals.hh:487-500`); `UnbalExhMassFlow`,
   `BalancedExhMassFlow`, `PlenumInducedMassFlow` (`ZoneEquipmentManager.cc:3698-3708`, `:4115-4119`);
   `dataUnitarySystems->CoolingLoad/HeatingLoad`, `dataFurnaces->CoolingLoad`.
 - **Scratch:** `dataSimAirServingZones` (`SimAirServingZones.hh:243-327`),
   `dataHVACInterfaceMgr->TmpRealARR`, `PrioritySimOrder` (`ZoneEquipmentManager.hh:277`),
   `dataGeneralRoutines->ZoneInterHalf/ZoneController` (`GeneralRoutines.hh:224-225`),
   `dataPlantUtilities->CriteriaChecks` (growable, `PlantUtilities.cc:870-955`).
 - **Statistics with `+=`:** `salIterMax/salIterTot/NumCallsTot` (`SimAirServingZones.cc`),
   `PlantManageHalfLoopCalls`, `PlantManageSubIterations`, controller statistics under
   `TRACK_AIRLOOP`.
 - **Cross-object couplings:** `HeatReclaim*` arrays read by desuperheaters in other systems;
   VRF condenser triggered by the last terminal unit (`HVACVariableRefrigerantFlow.cc:236-278`);
   refrigeration chiller sets; HPWH shared tanks; `SeriesActive` plant branches scanned by
   `SetComponentFlowRate` (`PlantUtilities.cc:207-246`).
 - **Order-dependent algorithm selection:** `state.dataRootFinder->rootAlgo` swapped by
   `SolveRoot2` (`General.cc:~330`) and `ExtendedHeatIndex.cc:545-556`; `thread_local
   SolveRootStats` (`WaterCoils.cc:5840`). 195 `General::SolveRoot` call sites in 31 files
   depend on it.

## 12. References ##

 - EnergyPlus Input Output Reference, "PerformancePrecisionTradeoffs"
   (`doc/input-output-reference/src/overview/group-simulation-parameters.tex`).
 - EnergyPlus Engineering Reference, "Inside Heat Balance", "Interior Long-Wave Radiation
   Exchange" (ScriptF, Carroll MRT), "HVAC Manager" and "Plant Manager" iteration schemes.
 - Historical `ProgramControl` object, "Number of Threads Allowed" (V7.2–V8.5,
   `idd/versions/V8-4-0-Energy+.idd:786-792`, marked "currently disabled"), and the eio
   report it left behind (`doc/output-details-and-examples/src/output-files/eplusout-eio.tex`).
 - J. A. Carroll, "An MRT Method of Computing Radiant Energy Exchange in Rooms", 1980
   (basis of the `CarrollMRT` option).
 - Big Ladder Software, Penumbra (GPU pixel-counting shading) and Kiva (ground heat
   transfer), both bundled in `third_party/`.
 - G. Amdahl, "Validity of the single processor approach to achieving large scale computing
   capabilities", 1967.

## 13. Implementation log ##

Everything below is on branch `claude/sleepy-goldberg-bqjqqq`, one commit per item, validated with
`scripts/dev/perf/bench.py` against a baseline build of `e2fd2334` on the same container (4 vCPU,
GCC 13.3, `-O3`). "Identical" means byte-identical ESO, MTR, EIO, ERR, RDD and MDD after removing
the time-stamped version lines. The validation set covers the code paths touched: `5ZoneAirCooled`,
`1ZoneUncontrolled_Win_ASH55_Thermal_Comfort`, `DaylightingDeviceShelf` (glare shading control),
`AirflowWindowsAndBetweenGlassBlinds`, `MovableIntInsulationLightsLowE`, `RadLoHydrHeatCoolAuto`
(zone re-simulation), `EquivalentLayerWindow`, `5ZoneEndUses` (water use), `CarrollMRT-RefBldgLargeOffice`,
`ZoneCoupledKivaRefBldgMediumOffice` and `45zonevav` (`1ZoneUncontrolled_win_1` replaces the
thermal-comfort model, whose weather file is not in the repository).

**Correction (2026-09-28).** Until that date every "baseline" run of the harness loaded the *new*
library (section 13.7), so the identity claims and the baseline-vs-new wall times recorded before
it were self-comparisons. Everything below has been re-established against the true `e2fd2334`
library: the identity statements now hold on the 11-model set above plus the 44-model sweep of
13.7, and the wall-time tables carry the re-measured numbers where they were affected. One real
regression that the defective harness had hidden (movable insulation, 13.7) is fixed.

### 13.1 Phase 0 — bugs (commit "Fix process-global state and repeated window-gain accumulation") ###

The five bugs listed in the summary are fixed: the `WaterUse.cc` static flag, the SQLite leap-year
static, the by-value warm-up counter, `ZoneWinHeatGain` accumulation on zone re-simulation and the
`SurfWinHeatGain += SurfWinRetHeatGainToZoneAir` repetition per predictor/corrector call (now guarded
by a per-window flag reset when the return-air gain is recomputed). Identical on the validation set;
the last two change results only for models with airflow windows returning to zones without return
air, or radiant systems re-simulating zones with windows.

The warm-up counter fix deserves a release-note line of its own. Because the counter was never
incremented, the fatal exit in `TestSurfTempCalcHeatBalanceInsideSurf` for more than 10 out-of-bounds
surface temperatures during warm-up (3 with enforced view-factor reciprocity) has been unreachable
since the C++ port, and a model that exceeds that count today completes silently. With the counter
counting, as the finite-difference path already does, such a model terminates with "Program terminates
due to preceding conditions". None of the 44 validation models is affected; the full regression suite
should be run with the fix before release, and any model that newly terminates is a model with a real
warm-up problem that the message was written for.

### 13.2 Phase 0 — measurement harness (commit "Add phase timers, --threads plumbing and a benchmark runner") ###

 - `Perf::ScopedTimer` in 29 entry points plus the AirflowNetwork calls; inclusive/exclusive seconds,
   call and item counts; `--timings` (or `TimingFlag`, or `Output:Diagnostics,TimingFlag`) writes
   `eplusout.perf` and a console table. Off by default; nothing is written to eio or err, so timed
   runs keep their regression outputs.
 - `Parallel::ThreadPool` (persistent fork/join pool per `EnergyPlusData`, workers stay hot for
   50 ms then park; exceptions rethrown on the caller; cost-balanced chunking), created at the start
   of `ManageSimulation` from `--threads`/`-j`/`ENERGYPLUS_NUM_THREADS`; default 1.
 - `scripts/dev/perf/bench.py`: medians over repeats, phase timers, byte-identity check across builds
   and thread counts, CSV/JSON output.
 - The eio "Threads/Parallel Sims" line reports real counts only when more than one thread is used.

Measured fork/join cost of the pool (standalone micro-benchmark, idle machine): 0.4 µs at 2 threads,
0.7 µs at 4 threads; a 17 µs region of 130 items runs 1.8× faster at 2 threads and 2.8× at 4. With a
compiler running on the same 4 vCPUs the same region ran 20–60× *slower* than serial: a spinning
fork/join pool must never be oversubscribed.

### 13.3 Phase 1 — items 5.3.1 and 5.3.2 ###

Skipping the repeated "Outside" `CalcInteriorRadExchange` (the "Main" call records the shading flags,
construction indices, inside thermal absorptances, window effective inside temperatures and the
surface temperature history of all heat transfer surfaces; the full "Outside" call is skipped when
none of them changed, and a zone re-simulation is never skipped) and removing the per-iteration
whole-array copies and the per-call heap allocation of the radiation temperature array are both
identical on the validation set. Wall time, true `e2fd2334` baseline → all commits of this log
through 13.7 (Phase 0–1, the pilot at `--threads 1`, and the latency items of 13.5), measured on
2026-09-28 after the harness fix (annual: single run each, same VM, back to back; design day:
median of 3):

| Model | Annual baseline | Annual now | Change | Design day baseline | Design day now | Change |
|---|---|---|---|---|---|---|
| `ASHRAE901_OutPatientHealthCare` | 166.9 s | 148.4 s | −11.1% | 9.47 s | 9.00 s | −5.0% |
| `HospitalBaseline` | 190.5 s | 178.6 s | −6.3% | 17.18 s | 16.40 s | −4.5% |
| `45zonevav` | 37.1 s | 35.0 s | −5.4% | 2.78 s | 2.40 s | −13% (min-to-min −7.5%) |

The table replaces the one recorded before the harness fix, which had compared the new build with
itself and reported noise. The Phase 1 share of the saving is one of the 2 + N_iter
radiant-exchange calls per timestep (about a quarter of the 4.9–5.4% that `CalcInteriorRadExchange`
costs); the rest is the reporting and cast work of 13.5. The array-copy item is a prerequisite for
the per-space parallel iteration (P4) rather than a serial win.

The annual phase timers of the final build (1 thread) put numbers on section 2.3 for these models:

| Timer (inclusive s) | OutPatient (152.9 s) | HospitalBaseline (178.1 s) | 45zonevav (34.6 s) |
|---|---|---|---|
| HVAC (system timestep loop) | 66.1 | 100.3 | 15.8 |
| ZoneEquipment | 20.9 | 42.4 | 6.4 |
| Plant | 5.6 | 21.9 | 3.0 |
| AirLoops | 6.7 | 12.7 | 1.1 |
| InsideHB (incl. windows, radiant exchange) | 20.4 | 20.6 | 5.7 |
| IntRadExchange | 9.0 (295,802 calls) | 6.5 (340,393 calls) | 1.4 (241,583 calls) |
| WindowHB | 3.9 (4.26 M calls) | 6.6 (7.09 M calls) | 2.6 |
| OutsideHB | 3.0 | 3.6 | 0.7 |
| ReportHeatBalance | 27.0 | 11.5 | 2.6 |
| UpdateDataandReport | 12.6 | 9.0 | 1.7 |
| NodeInfo (CalcMoreNodeInfo) | 1.7 | 5.0 | 0.8 |

`ReportHeatBalance` at 17.6% of the OutPatient run (it has 1400 surfaces and many surface output
variables) is larger than any single envelope phase and moves reporting up the Phase 1 list.

### 13.4 Phase 2 pilot — region P2 (commit "Run the interior radiant exchange enclosure loop on the thread pool") ###

`CalcInteriorRadExchange` is split into a serial phase (shade/insulation change detection and
emissivity/ScriptF recompute, which allocates and can warn) and a per-enclosure phase that runs on
the pool with per-thread scratch. Outputs are byte-identical at 1, 3 and 4 threads, twice each, on
all validation models, and on the annual runs below. The performance result is negative for this
region on its own:

| Model, annual, 1 → 3 threads | Radiant-exchange calls | Serial µs/call | Parallel µs/call (3) | Variant chosen | Radiant timer | Wall |
|---|---|---|---|---|---|---|
| `HospitalBaseline` (1050 surfaces, ~8 per enclosure) | 340,393 | 22.6 | 37.2 | serial 69 of 81 comparisons | 6.45 → 7.06 s | 178.1 → 179.5 s |
| `ASHRAE901_OutPatientHealthCare` (1400 surfaces) | 295,802 | 34.0 | 21.2 | parallel 61 of 70 | 8.99 → 7.41 s | 152.9 → 146.9 s (−3.9%) |
| `45zonevav` | 241,583 | 7.7 | 18.7 | serial 58 of 58 | 1.38 → 1.48 s | 34.6 → 34.9 s |

Design-day runs (1 → 4 threads) told the same story: HospitalBaseline 16.5 → 16.8 s, OutPatient
8.9 → 9.5 s, `ASHRAE901_ApartmentHighRise` 11.5 → 11.8 s.

Three findings, all of which change the plan for Phases 2 and 3:

1. **The regions are too small.** A whole-building radiant exchange is 5–30 µs; a whole inside-surface
   heat balance call is only ~170 µs on the hospital (2.0 s over 11,800 timesteps), and a whole
   timestep is ~1.4 ms. Fork/join at 0.7 µs is not the problem; cold caches are. Workers that did
   not run the preceding serial code must pull every `Surface`, `Construct` and `ScriptF` row from
   another core's cache, which for an 8-surface enclosure costs more than the 64 multiply-adds it
   saves. Only the OutPatient model (larger enclosures) showed a gain (28 → 21 µs with 3 threads,
   2.1× in isolated samples).
2. **Kept-hot workers cost the main thread 2–3%.** With `--threads 4` on a 4-vCPU VM, wall time
   rose 2–6% even when the exchange ran serially, because the spinning workers compete with the main
   thread for the host's execution resources. Between timesteps the HVAC phase is only ~0.6 ms, so
   a block time short enough to park the workers (1 ms) makes every timestep pay a 15–45 µs wake-up
   instead. Recommendation: `--threads` ≤ physical cores − 1 on shared machines, and the default
   stays 1 until Phase 3 regions exist.
3. **Adaptive scheduling is necessary.** `Parallel::AdaptiveChoice` times the first 64 full calls
   alternately serial and parallel, keeps the faster variant for 4096 calls, then re-compares. With this the radiant-exchange
   timer is never more than noise above serial at any thread count, and results are unaffected
   (scheduling only). The early decision was wrong for the hospital (sizing-period samples chose
   parallel; the run as a whole was slower) until the periodic re-comparison was added.

**Revised plan.** Region P2 stays as the pilot and as the mechanism for very large enclosures, but
Phase 3 should not add P3–P9 one region at a time. The unit of parallel work has to be one *space
(or enclosure group) for the whole envelope step* — outside balance, radiant exchange, window
balance, convection and inside iteration kernel for the same surfaces on the same thread — so that
each thread works on data it just touched and a timestep issues a handful of ~100 µs regions rather
than dozens of 10 µs ones. That is the P4 design in section 4.1 with P2, P3, P5 and P6 fused into
it, and it needs the shared-state hardening of section 5.2 first (windows: `dataWindowManager`,
TARCOG; convection: one-time input fetches). The Phase 2 exit criterion "P2 shows ≥3× on the
radiant-exchange timer at 4 threads" is withdrawn as unrealistic for 8-surface enclosures; the
Phase 3 criterion (envelope timers ≥0.6× linear to 8 threads) stands but is to be measured on the
fused kernel.

### 13.5 Latency work (Tier B, section 14.2) ###

Measured with a one-week `ASHRAE901_OutPatientHealthCare` run under callgrind (46 G instructions,
32% of it sizing) and the annual timers. Findings that drove the changes:

 - `std::format` for the ESO records: 733,000 calls per simulated week at about 2,000 instructions
   each (3.3%), i.e. most of the cost of writing a value. Replaced by direct assembly of
   `<id>,<value>\n` for the ESO and MTR value records (same characters).
 - `dynamic_cast` per output variable per timestep: 2.4–3.7%. Replaced by `static_cast` on the
   already-known variable type.
 - Small `memset`s: 10 million per week from per-space, per-layer zeroing loops in
   `InitSolarHeatGains` (1%) and the whole-array zeroing of `NetLWRadToSurf` in the radiant
   exchange (1%). The radiant exchange now zeroes only enclosure surfaces (equivalent, since
   nothing else writes the others); the solar loops are left for the kernel-table work.
 - `pow`: 14 million calls (3.8%), 5.6 million from the TARP natural convection correlation
   (`x^(1/3)`) and 5.5 million from water-coil UA and simple-heating-coil correlations. These are
   real arithmetic; `cbrt` would not be bit-identical, so they stay.
 - Heap allocation: 12 million `operator new` per week (about 5% with `free`), 3.1 million of them
   `Array<double>` copies from `ZoneSizingData` copies in `BaseSizer::initializeWithinEP` (sizing
   only, plan item 9a), most of the rest `std::string` temporaries. The three sizing vectors
   (`FinalZoneSizing`, `TermUnitFinalZoneSizing`, `FinalSysSizing`, 3.4% of the one-week
   instructions and 4.5% of its level-1 data misses) are now read-only views on the state arrays
   (`SizingDataView`); a sizer initialized from the API still owns its own storage. The other
   vectors the sizer copies stay copies: `CoolingCapacitySizing` writes into its copy of
   `PrimaryAirSystems`, so sharing it would change results.
 - `GetInstantMeterValue`, `GatherMonthlyResultsForTimestep`: 3.7% inclusive, driven by nine
   `Output:Table:Monthly` objects evaluated per zone and system timestep through ObjexxFCL indexing
   (13,000 column visits per timestep on this model). The time stamp that every visit rebuilt (two
   calls) is now computed once per timestep; the column walk itself is left as is.
 - `CalcMoreNodeInfo`: three glycol property lookups per water node per system timestep, one of
   them (density at the standard temperature) constant; now evaluated once.
 - Iterative psychrometric solvers probe the saturation-pressure table with a fresh guess on every
   iteration. Evaluating the raw function at the table's quantized temperature instead (identical
   value) was tried and reverted: on a two-day `OutPatient` run with cache simulation it removed
   3.4 M level-1 data misses inside `PsyTsatFnPb` but added 139 M instructions, a wash at best.

Link-time optimization (`-flto=auto`, new `ENABLE_LTO` CMake option, off by default): identical
outputs on the three benchmark models; design-day wall time −0.3% (`HospitalBaseline`), −3.1%
(`OutPatient`), −5.4% (`45zonevav`), median of 3.

### 13.5a Review pass over the branch ###

A review of the whole branch against `develop` led to these changes, all scheduling, harness or
robustness fixes with no effect on results (identical on the 11-model and 44-model sets):

 - `ThreadPool::balancedBounds` now computes the exact minimax contiguous partition (bisection on
   the per-chunk cost limit with greedy packing) instead of closing a chunk once its cost crossed
   the average, which put a heavy enclosure at a boundary into the previous chunk and left later
   threads idle. Unused trailing chunks are empty. The unused `parallelFor` is removed and
   `parallelChunks` asserts the bounds size.
 - The serial/parallel A/B state machine moved out of the radiant-exchange routine into
   `Parallel::AdaptiveChoice`, so the next parallel region reuses it rather than copying it.
 - The "Outside" call skip records and compares five inputs per surface (above) instead of two, so
   a future writer inserted between the two calls makes the skip fall back to recomputing rather
   than reusing a stale exchange.
 - `SizingDataView` has explicit copy and move operations that rebind a copy to its own storage;
   the implicit ones left a copy of an API-initialized sizer pointing at the source's buffer.
 - `PerfTimerData::clear_state` resets its members instead of placement-new over a live object.
 - `bench.py` matches `add_simulation_test` case-insensitively (the CMake list spells it in lower
   case, so the IDF-to-weather mapping never matched and every model ran with Chicago weather), and
   its exit status also requires every run to have completed, so two builds that fail identically
   no longer pass the identity gate. `chunked_run.py` parses each chunk ESO once and reports a
   missing `RunPeriod` as an error instead of a traceback.
 - 25 files reformatted to `src/.clang-format`.

### 13.6 Not yet done ###

Items 5.3.3–5.3.11 (Phase 1), the hardening table (5.2), regions P3–P14, the HVAC iteration work
(Phase 4) and the nightly CI job. The next step with the best value/risk is 5.3.9b/9a (water-coil
controller and sizing copies, 10–11.5% of the hospital profiles) followed by the reporting items
(5.3.7).

### 13.7 Measurement-harness defect, and what re-validation found ###

`bench.py` ran the copied baseline executable without setting `LD_LIBRARY_PATH`. The executable's
`RUNPATH` is the build directory it was linked in, so `baseline/energyplus` loaded
`build/Products/libenergyplusapi.so`, i.e. the library under test. Every baseline-vs-new comparison
made with the harness before 2026-09-28 (identity checks and wall times in 13.1, 13.3 and 13.5)
compared the new build with itself; only the thread-count comparisons (13.4), the LTO comparison
(a separate build directory with its own `RUNPATH`) and the chunked-run comparisons (14.1) were
real. Found when a callgrind run of "both" binaries returned the same instruction count to five
digits. The runner now sets `LD_LIBRARY_PATH` to each executable's directory and prints the
resolved library for every build; the validation scripts were rerun.

Re-validation against the true baseline library:

 - Identity holds on 10 of the 11 models of the validation set and, after the fix below, on all 11
   and on a 44-model sweep chosen for feature coverage (CondFD, PCM, AirflowNetwork, complex
   fenestration, TDD, Kiva and ground domains, radiant systems, movable insulation inside and
   outside, swimming pool, ice and chilled-water storage, VRF, PIU, fan coils, DOAS, PV and BIPVT,
   green roof, EMS, economics, resilience reports, room air models, night ventilation).
 - `MovableIntInsulationLightsLowE` differed: the inside surface heat balance no longer converged.
   Commit "Drop per-iteration whole-array copies in the inside surface heat balance" had removed the
   per-iteration copy `SurfTempInTmpOld = SurfTempInTmp` as write-only. It is read for surfaces with
   interior movable insulation (their damped temperature update and their convergence check). The
   copy is restored for exactly those surfaces (`intMovInsulSurfNums`), which is what the whole-array
   copy amounted to at every reading site.
 - Instruction counts (callgrind, one-week `OutPatient`, sizing included), true baseline → current:
   46.71 G → 44.85 G (−4.0%). `__dynamic_cast` 1.09 G → 0.03 G, `memset` 1.53 G → 1.01 G,
   `CalcInteriorRadExchange` 4.55 G → 3.92 G (the skipped "Outside" call). `std::vformat` is
   unchanged at 0.23 G: the ESO record writer was not the `std::format` consumer the profile
   suggested, so that part of the 13.5 entry is withdrawn. The remaining 0.73 M `std::format`
   calls per week (3.3% inclusive) are the EMS trace lines that this model requests with
   `Output:EnergyManagementSystem, Verbose`: `WriteTrace` built the time-stamp string for every
   traced instruction. It is now built once per timestep and the record is written without
   `std::format` (same bytes in the edd file, which the harness now compares as well).
 - Cache simulation (callgrind, two-day `OutPatient`, true baseline → current): 37.76 G → 36.68 G
   instructions, 1.169 G → 1.161 G level-1 data misses, 5.6 M last-level misses in both. The
   last-level miss count is tiny (one per 6,700 instructions): on a core with a large L3 this code
   is not DRAM-latency bound, it is L1-miss and dependency bound. The level-1 misses concentrate in
   the envelope kernels (`CalcHeatBalanceInsideSurf2CTFOnly` 10.8%, `CalcInteriorRadExchange` 8.4%,
   `CalculateZoneMRT` 2.6%, `InitSurfaceHeatBalance` 2.2%, `ReportSurfaceHeatBalance` 2.0%) and in
   the meter walk `GetInstantMeterValue` (3.1%), i.e. in the per-surface arrays that are indexed
   through several indirections per surface. That is the data-layout work of 5.3.9 and 14.2 (fused
   per-space kernels over surface-ordered tables); nothing smaller will move it.

## 14. Reassessment: what can make a run an order of magnitude faster ##

The measurements above change the emphasis of this plan. An annual run is a chain of 35,000 to
100,000 timesteps of about 1.4 ms each on a 1,000-surface hospital (0.5 ms envelope, 0.6 ms HVAC,
the rest reporting). No phase is compute-bound: the whole-building radiant exchange is 8,000
multiply-adds in 20 µs, about 1% of one core's arithmetic peak. The time is spent in latency
(indirect array lookups per surface and node, psychrometric cache probes, `dynamic_cast` per
output variable). Splitting 10–30 µs latency-bound regions across cores makes them slower, as the
pilot showed, and even the fused envelope kernel of section 13.4 is a 1.2× whole-run story.
Three tiers follow, distinguished by what "same results" means.

### 14.1 Tier A — same results as EnergyPlus already defines them: time-parallel run periods ###

Every run period starts with warm-up days that repeat the first day until the state forgets its
initial condition, to the tolerances of the `Building` object (defaults 0.4 K on zone temperature,
4% on loads). Every annual result relies on that definition. It allows the run period to be split
into K chunks (months or weeks), each run as its own process from a generated input with the
chunk's `RunPeriod`, the model's own warm-up settings, and an overlap of a few real days with the
previous chunk. The overlap is the correctness instrument: zone temperatures and meters on the
overlapping days are compared with the previous chunk, the maximum deviation is reported, and the
warm-up is extended and the chunk rerun when it exceeds the warm-up tolerance. That is a stronger
statement than today's, where nobody checks that warm-up converged for the first day of the run
period.

From the annual measurements on `HospitalBaseline` (178 s, about 16 s of it sizing and
initialisation, 0.45 s per simulated day):

| Chunks | Cores | Time per chunk (sizing + 7 warm-up days + chunk) | Speedup |
|---|---|---|---|
| 3 × 4 months | 4 | 16 + 3.2 + 55 = 74 s | 2.4× |
| 12 months | 12 | 16 + 3.2 + 14 = 33 s | 5.4× |
| 52 weeks | 52 | 16 + 3.2 + 3.2 = 22 s | 8× |

Sizing is redone identically in every chunk and becomes the floor; sizing periods are independent
environments and can run concurrently, which moves the floor toward 10 s and the weekly case past
10×. None of this needs threads in the C++: an orchestrator, a `RunPeriod` rewriter, and output
merging (ESO/MTR concatenation, meter sums, SQLite merge, and the annual tabular reports rebuilt
from merged meter data, which is the one substantial piece).

**Measured (prototype `scripts/dev/perf/chunked_run.py`, `HospitalBaseline`, 12 monthly chunks and
4 quarterly chunks, 4 processes at a time on 4 vCPUs, reference year run with the same added
hourly outputs).**

 - Zone mean air temperature, hourly, every zone, over the chunk bodies: maximum deviation from the
   full-year run 0.033 K, 99th percentile 0.0005 K, mean below 0.0001 K, against the model's own
   0.4 K warm-up tolerance. January is bit-identical once the shading calculation period is
   aligned (see below); the other months carry 6–8 warm-up days plus a 3-day overlap.
 - Natural gas, cooling and heating energy transfer: monthly totals within ±0.001%, annual totals
   within 0.0005%.
 - Electricity: annual −0.034% (monthly chunks) and 0.0000% (quarterly chunks); February −0.6%.
   The February difference is a single plant pump of about 190 kW that runs continuously in the
   full-year run from 25 January to 7 February with zero cooling load and does not run in the
   chunk. A 10-day overlap did not change it (the year-run trajectory is a path-dependent discrete
   control state, not slow thermal memory), so the overlap check reported it as a 2–6% daily
   electricity deviation on the overlap days, which is the intended behaviour: the model has two
   valid trajectories and the tool tells the user which meter and which days disagree. The pump
   behaviour itself is worth a look as a model or engine quirk.
 - Alignment matters: with the model's 7-day periodic shading calculation, a chunk whose length is
   not a multiple of 7 days averages a different shading day at its end than the year run does
   (January differed by 0.026 K for that reason alone). The orchestrator must start and end chunks
   on the year run's shading period boundaries, or the comparison run must use daily updates as
   the prototype does.
 - Wall time on 4 vCPUs: 4 quarterly chunks 68 s against 178 s for the year (2.6×); 12 monthly
   chunks in three waves 107 s (1.7×, because each chunk repeats the 16 s of sizing). Per-chunk
   times were 33–37 s for a month and 60–68 s for a quarter while sharing the machine four ways.

Detected from the input, and either refused or run serially: components with seasonal memory
(ground heat exchangers with g-function load history, borefields, seasonal storage), EMS or Python
plugins with persistent state or long trend windows, demand limiting on billing periods that do not
align with chunk boundaries. Daily-cycle storage (ice, water heaters, PCM) reconverges within
warm-up as thermal mass does. Multi-year runs and `RunPeriod` objects with actual weather years are
chunked per year first.

### 14.2 Tier B — bit-identical: make each timestep cheap ###

 - Reporting on a second thread (13–18% of an annual run): snapshot values in fixed order on the
   main thread, format and write on the writer thread. Same bytes, same order.
 - Exact memoisation of HVAC components: within one HVAC iteration, air loops and zone equipment
   are re-simulated because a flag was raised elsewhere, and most components see identical inlet
   nodes, control signals and schedules. A component that is a pure function of its inputs returns
   the cached result. Enumerating the implicit inputs ("current object" globals) is the same
   hardening that threading needs.
 - Envelope data layout: per-surface kernel descriptors (temperature source, emissivity, CTF
   terms) so the inside iteration is a tight loop over contiguous arrays instead of several
   indirections per surface. This is what the radiant-exchange pilot was really pointing at.
 - Psychrometrics without cache probes in hot paths (bit-identical only where the cache is exact
   today; needs the equivalence check of section 5.2).
 - LTO, PGO on the benchmark set, `-march` with contraction off (section 5.3.11).

### 14.3 Tier C — same equations and tolerances, different solver path ###

Warm starts of the HVAC solution across system timesteps, convergence-based plant sweeps, and a
direct solution for the simple heating coil (section 4.6, Phase 4). Measured on the hospital
models: quantised load-independent plant sweep counts, 1.17–1.74 system timesteps per zone
timestep with the full-step solve discarded on every downstep, controllers cold-started every
system timestep, up to 25 coil evaluations per terminal per call. Estimated 1.5–2× on the HVAC
half; differences at the tolerance level, behind `PerformancePrecisionTradeoffs` fields.

### 14.4 Revised order of work ###

1. Tier A orchestrator: the validation half exists (`chunked_run.py`) and the warm-up argument
   holds on `HospitalBaseline` to 0.03 K and 0.001% on gas and thermal meters. Next: chunk
   alignment to the shading period, detection of long-memory objects, sizing periods run once and
   shared, then output merging (ESO/MTR, SQLite, tabular reports from merged meters).
2. Tier B reporting thread and envelope data layout, which compound with Tier A and never risk a
   result.
3. Tier C behind input options.
4. In-process fork/join threading only where it has a region worth its cost: the fused per-space
   envelope kernel (section 13.4) and the reporting thread.
