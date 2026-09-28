#!/usr/bin/env python3
"""Benchmark runner for EnergyPlus performance work.

Runs a set of IDF files with one or more EnergyPlus builds, records wall time
(median of N repeats), the phase timers from eplusout.perf (when the build
supports --timings), the solver iteration counts, and checks that the outputs
of every build are byte-identical to the first build's outputs.

Usage examples
--------------
  # Time three models with two builds, 3 repeats each, 1 and 4 threads
  bench.py --build baseline=/path/to/baseline/energyplus \
           --build new=/path/to/new/energyplus \
           --threads 1 4 --repeat 3 --annual \
           --out /tmp/bench \
           testfiles/5ZoneAirCooled.idf performance_tests/45zonevav.idf

  # Design-day runs, default weather from testfiles/CMakeLists.txt when known
  bench.py --build new=./build/Products/energyplus --design-day testfiles/HospitalBaseline.idf

Only the Python standard library is used.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_EPW = REPO / "weather" / "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"

# Files whose content is compared between builds. Lines carrying the program version or time stamps are ignored.
COMPARED_OUTPUTS = ["eplusout.eso", "eplusout.mtr", "eplusout.eio", "eplusout.err", "eplusout.rdd", "eplusout.mdd", "eplusout.edd"]
ERR_LINE_FILTER = re.compile(
    r"(Elapsed Time|EnergyPlus, Version|YMD=|Program Version|Started at|Simulation Time|\*\*\*\*\*|Program Control Information:Threads/Parallel Sims)"
)


def weather_for(idf: Path) -> Path:
    """Look up the weather file mapped to an IDF in testfiles/CMakeLists.txt, else Chicago."""
    cmake = REPO / "testfiles" / "CMakeLists.txt"
    if cmake.exists():
        pat = re.compile(r'ADD_SIMULATION_TEST\(IDF_FILE\s+(\S+)\s+EPW_FILE\s+(\S+)')
        for m in pat.finditer(cmake.read_text(errors="replace")):
            if Path(m.group(1)).name == idf.name:
                return REPO / "weather" / m.group(2)
    return DEFAULT_EPW


def needs_expand_objects(idf: Path) -> bool:
    """HVACTemplate objects need ExpandObjects (energyplus -x), which must sit next to each executable."""
    try:
        with idf.open(errors="replace") as f:
            return any(line.lstrip().lower().startswith("hvactemplate:") for line in f)
    except OSError:
        return False


def file_digest(path: Path) -> str:
    if not path.exists():
        return "missing"
    # Every compared file starts with a "Program Version" line that carries the run's time stamp
    h = hashlib.sha256()
    with path.open("rb") as f:
        for line in f:
            if ERR_LINE_FILTER.search(line.decode(errors="replace")):
                continue
            h.update(line)
    return h.hexdigest()[:16]


def run_once(exe: Path, idf: Path, epw: Path, outdir: Path, threads: int, mode: str, timings: bool, extra_env: dict) -> dict:
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    cmd = [str(exe), "-w", str(epw), "-d", str(outdir)]
    if mode == "annual":
        cmd.append("-a")
    elif mode == "design-day":
        cmd.append("-D")
    if threads != 1:
        cmd += ["--threads", str(threads)]
    if timings:
        cmd.append("--timings")
    if needs_expand_objects(idf):
        cmd.append("-x")
    cmd.append(str(idf))
    env = dict(os.environ)
    env.update(extra_env)
    # Load the shared library that sits next to the executable. The executable's RUNPATH points at the
    # directory it was built in, so a copied baseline binary would otherwise silently pick up the library
    # of the current build and every "baseline" comparison would compare the new build against itself.
    lib_dir = str(exe.resolve().parent)
    env["LD_LIBRARY_PATH"] = lib_dir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(outdir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    wall = time.perf_counter() - t0
    (outdir / "stdout.txt").write_text(proc.stdout)
    result = {"wall": wall, "rc": proc.returncode, "completed": "EnergyPlus Completed Successfully" in proc.stdout}
    perf = outdir / "eplusout.perf"
    if perf.exists():
        try:
            result["perf"] = json.loads(perf.read_text())
        except json.JSONDecodeError:
            result["perf"] = None
    result["digests"] = {name: file_digest(outdir / name) for name in COMPARED_OUTPUTS}
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("idf", nargs="+", type=Path, help="IDF files to run")
    ap.add_argument("--build", action="append", required=True, metavar="NAME=EXE", help="named EnergyPlus executable; repeatable")
    ap.add_argument("--threads", nargs="+", type=int, default=[1], help="thread counts to run (default 1)")
    ap.add_argument("--repeat", type=int, default=1, help="repeats per case; wall time is the median")
    ap.add_argument("--annual", action="store_true", help="force an annual run (-a)")
    ap.add_argument("--design-day", action="store_true", help="force design-day only (-D)")
    ap.add_argument("--epw", type=Path, help="weather file for all runs (default: testfiles mapping or Chicago)")
    ap.add_argument("--out", type=Path, default=Path("bench_out"), help="output directory")
    ap.add_argument("--no-timings", action="store_true", help="do not pass --timings (for builds that do not support it)")
    args = ap.parse_args()

    mode = "annual" if args.annual else ("design-day" if args.design_day else "default")
    builds = []
    for spec in args.build:
        name, _, exe = spec.partition("=")
        if not exe:
            ap.error(f"--build needs NAME=EXE, got {spec!r}")
        builds.append((name, Path(exe).resolve()))
    # Show which shared library each executable will load so that a mis-resolved baseline is visible in the log.
    for name, exe in builds:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = str(exe.parent) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        try:
            ldd = subprocess.run(["ldd", str(exe)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env).stdout
            libs = [line.split("=>")[1].split("(")[0].strip() for line in ldd.splitlines() if "libenergyplusapi" in line and "=>" in line]
        except OSError:
            libs = []
        print(f"build {name}: {exe} -> {', '.join(libs) if libs else 'static or unresolved library'}")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    identical = True
    for idf in args.idf:
        idf = idf.resolve()
        epw = args.epw.resolve() if args.epw else weather_for(idf)
        reference = None
        for name, exe in builds:
            for threads in args.threads:
                runs = []
                for rep in range(args.repeat):
                    outdir = args.out / idf.stem / f"{name}_t{threads}_r{rep}"
                    r = run_once(exe, idf, epw, outdir, threads, mode, not args.no_timings, {})
                    runs.append(r)
                    status = "ok" if r["completed"] else f"FAILED rc={r['rc']}"
                    print(f"{idf.name:50s} {name:12s} threads={threads} rep={rep}: {r['wall']:8.2f} s  {status}", flush=True)
                walls = [r["wall"] for r in runs]
                row = {
                    "idf": idf.name,
                    "build": name,
                    "threads": threads,
                    "mode": mode,
                    "wall_median_s": round(statistics.median(walls), 3),
                    "wall_min_s": round(min(walls), 3),
                    "completed": all(r["completed"] for r in runs),
                }
                digests = runs[0]["digests"]
                for r in runs[1:]:
                    if r["digests"] != digests:
                        identical = False
                        print(f"  !! outputs differ between repeats of {name} threads={threads}", flush=True)
                if reference is None:
                    reference = digests
                    row["outputs_match_reference"] = True
                else:
                    diff = [k for k in digests if digests[k] != reference[k]]
                    row["outputs_match_reference"] = not diff
                    if diff:
                        identical = False
                        print(f"  !! {name} threads={threads}: differs from reference in {', '.join(diff)}", flush=True)
                perf = runs[0].get("perf")
                if perf:
                    for t in perf["timers"]:
                        row[f"t_{t['name']}_incl_s"] = round(t["inclusive_s"], 3)
                        row[f"t_{t['name']}_excl_s"] = round(t["exclusive_s"], 3)
                        row[f"n_{t['name']}"] = t["calls"]
                        if t["items"]:
                            row[f"items_{t['name']}"] = t["items"]
                rows.append(row)

    fieldnames = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    csv_path = args.out / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    (args.out / "results.json").write_text(json.dumps(rows, indent=1))

    print()
    print(f"{'idf':40s} {'build':12s} {'thr':>3s} {'median s':>9s} {'match':>5s}")
    for row in rows:
        print(f"{row['idf'][:40]:40s} {row['build']:12s} {row['threads']:3d} {row['wall_median_s']:9.2f} {'yes' if row['outputs_match_reference'] else 'NO':>5s}")
    print(f"\nResults: {csv_path}")
    print("All compared outputs identical across builds and thread counts." if identical else "OUTPUT DIFFERENCES FOUND (see above).")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
