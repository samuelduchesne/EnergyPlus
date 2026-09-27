# Performance benchmarking

`bench.py` runs a list of IDF files with one or more EnergyPlus builds, repeats each run,
records the median wall time, the per-phase timers written by `--timings` to `eplusout.perf`,
and verifies that ESO, MTR, EIO, ERR (time stamps excluded), RDD and MDD outputs are
byte-identical across builds and thread counts. The exit status is non-zero when any output
differs, so the script doubles as the correctness gate for performance changes.

```
scripts/dev/perf/bench.py \
    --build baseline=/opt/eplus-baseline/energyplus \
    --build new=build/Products/energyplus \
    --threads 1 4 --repeat 3 --annual --out /tmp/bench \
    performance_tests/45zonevav.idf testfiles/HospitalBaseline.idf
```

Timers are enabled with `--timings` (or `TimingFlag=Yes` in the environment, or
`Output:Diagnostics,TimingFlag`). They are never written to the eio or err files, so a timed run
produces the same regression-compared outputs as an untimed one.

Benchmark set (see `design/FY2026/MulticoreSimulationPerformance.md`, section 7.1): envelope-heavy
(`performance_tests/10x_incr_100zones1win10overhangsFullExIn.idf`, the OutPatient and
ApartmentHighRise 90.1 prototypes), HVAC/plant-heavy (`HospitalBaseline`, `HospitalLowEnergy`,
90.1 Hospital and OfficeLarge, `45zonevav`, `BenchmarkHospitalNew`), and reporting-heavy
(`15zonevav` vs `15zonevav_no_reports`).
