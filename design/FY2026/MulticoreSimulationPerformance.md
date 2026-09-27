Faster multizone / multi-plant simulations on multi-core CPUs
==============================================================

**samuelduchesne/EnergyPlus fork — engineering plan, drafted with Claude Code**

 - Original date: 2026-09-27
 - Code base: EnergyPlus 26.2.0 (`e2fd2334`)
 - Status: plan for review; nothing in this document has been implemented yet

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
| `ManageSimulation` | 88.8% | 96.6% | _pending_ |
| `ManageSurfaceHeatBalance` minus the nested `ManageAirHeatBalance` (envelope) | 32.5% | 32.8% | _pending_ |
| ├ `CalcHeatBalanceInsideSurf` (iterative inside balance) | 16.0% | 13.9% | _pending_ |
| ├ `InitSurfaceHeatBalance` (rad exchange "Main", solar, daylighting init, convection) | 12.4% | 13.7% | _pending_ |
| ├ `CalcInteriorRadExchange` (all call sites) | 7.3% | 7.5% | _pending_ |
| ├ `CalcHeatBalanceOutsideSurf` | — | 3.0% | _pending_ |
| ├ `CalcWindowHeatBalance` | — | 2.6% | _pending_ |
| `ManageHVAC` | 30.8% | 42.3% | _pending_ |
| ├ `SimZoneEquipment` (VAV terminals → reheat coils → `ControlCompOutput`) | 20.7% | 25.6% | _pending_ |
| │ └ `SimulateWaterCoilComponents` | 14.2% | 24.1% | _pending_ |
| │   └ `ControlCompOutput` (interval-halving controller) | 10.0% | 11.5% | _pending_ |
| ├ `ManagePlantLoops` | 5.7% | 11.2% | _pending_ |
| ├ `SimAirLoops` | — | 7.8% | _pending_ |
| ├ `ManageZoneAirUpdates` (predictor/corrector) | — | 3.5% | _pending_ |
| `PerformSolarCalculations` incl. `CalcDayltgCoefficients` | 12.0% | — | _pending_ |
| `ManageSizing` (design-day runs only) | 27.7% | 25.3% | n/a |
| Input processing (`processInput`, JSON/valijson) | 5.4% + 5.7% | 3.4% | n/a |
| **Exclusive:** `CalcInteriorRadExchange` | 5.2% | 5.4% | _pending_ |
| **Exclusive:** `pow`/`exp`/`sincos` (libm) | 8.7% | 7.2% | _pending_ |
| **Exclusive:** `malloc`/`free` family | 4.9% | 8.1% | _pending_ |
| **Exclusive:** `ObjexxFCL::Array<double>` copy constructor | 1.6% | 4.0% | _pending_ |
| **Exclusive:** `memset` | 3.7% | 2.9% | _pending_ |
| **Exclusive:** `__dynamic_cast` | 1.1% | 1.0% | _pending_ |

### 2.3 Profile-driven hot spots ###

1. **The inside-surface iteration and long-wave exchange are the envelope cost.** Together
   `CalcHeatBalanceInsideSurf` and `CalcInteriorRadExchange` are ~20% of a design-day run
   on both models, and `CalcInteriorRadExchange` is the single largest *exclusive* function
   (5.2–5.4%). It is called 2 + N_iter times per zone timestep (section 4.2). Both are the
   prime parallel targets (regions P2 and P4) and both carry easy serial waste (duplicate
   "Outside" call, whole-array copies and zeroing every iteration).
2. **Zone equipment dominates HVAC, through water-coil reheat control.** On the hospital,
   26% of all instructions are in `SimZoneEquipment`, almost all of it in VAV reheat
   terminals → `SimulateWaterCoilComponents` (24%) → `ControlCompOutput` (11.5%), the
   interval-halving controller that re-simulates the coil up to 25 times per terminal per
   call. Zones are independent here (region P12), and a direct or secant coil solution
   ("Use Coil Direct Solutions" only covers DX coils today) would remove most of those
   evaluations serially.
3. **Plant is 11% on the hospital and its sweeps are mostly forced, not converged.** The
   iteration-count output variables (annual runs, `detailed` frequency) show:

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
4. **Allocation and copying are 8–12% of instructions.** `malloc`/`free`, the `ObjexxFCL`
   array copy constructor and `memset` together account for 10% (45-zone) to 15%
   (hospital) of exclusive cost: per-call `Array1D` temporaries in the inside balance
   and `CalcScriptF`, `BaseSizer::initializeWithinEP` copying entire `ZoneSizingData`
   structs (10% of the hospital design-day run by itself), and the whole-array zeroing in
   the radiant exchange. These are bit-identical serial fixes (5.3).
5. **libm is 7–9%.** `pow`, `exp` and `sincos` come from psychrometrics, the T⁴ terms in
   radiant exchange and solar geometry. Vectorised or fused evaluation and the psychrometric
   cache redesign in 5.2 address this; it is also the part that benefits from `-march`.
6. **Downstepping is common.** The 45-zone VAV run logs 58,565 system timesteps for
   33,614 zone timesteps (1.74 per zone timestep); every downstepped zone timestep repeats
   `SimHVAC` for each sub-step and discards the full-step solve (section 4.6 item 3).
7. **`__dynamic_cast` shows up at all** (1%): that is `UpdateDataandReport` casting every
   output variable on every call (5.2, reporting row).

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
   `std::variant`; write ESO/MTR values with `dtoa` straight into a buffer instead of a
   `std::format` temporary; batch `ResultsFramework` map lookups.
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
