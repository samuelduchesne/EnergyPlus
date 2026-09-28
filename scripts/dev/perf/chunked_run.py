#!/usr/bin/env python3
"""Prototype of time-parallel annual runs (design document section 14.1).

Splits the RunPeriod of an IDF into K chunks (months), each preceded by an overlap of a few real
days with the previous chunk, runs every chunk as its own EnergyPlus process (J at a time), runs
the unmodified year as the reference, and compares hourly zone mean air temperatures and facility
meters on the overlap days and on the chunk bodies against the reference.

This is the validation half of the orchestrator: no output merging yet. It answers the question
"does the model forget its initial state within the warm-up, to what accuracy?" for a given model.

Example:
  chunked_run.py --exe build/Products/energyplus --epw weather/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw \
      --jobs 4 --overlap-days 3 --year 2017 --out /tmp/chunks testfiles/HospitalBaseline.idf
"""

import argparse
import calendar
import concurrent.futures
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ADDED_OUTPUTS = """
! --- added by chunked_run.py for chunk validation ---
Output:Variable,*,Zone Mean Air Temperature,Hourly;
Output:Meter,Electricity:Facility,Hourly;
Output:Meter,NaturalGas:Facility,Hourly;
Output:Meter,Cooling:EnergyTransfer,Hourly;
Output:Meter,Heating:EnergyTransfer,Hourly;
"""

RUNPERIOD_RE = re.compile(r"^\s*RunPeriod\s*,.*?;", re.M | re.S)


def strip_comments(obj: str) -> list[str]:
    fields = []
    for part in obj.split(","):
        part = re.sub(r"!.*", "", part).strip().rstrip(";").strip()
        fields.append(part)
    return fields


def rewrite_runperiod(idf_text: str, begin: tuple[int, int], end: tuple[int, int], year: int) -> str:
    matches = list(RUNPERIOD_RE.finditer(idf_text))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one RunPeriod object, found {len(matches)}")
    fields = strip_comments(matches[0].group(0))
    # fields: RunPeriod, Name, BeginMonth, BeginDay, BeginYear, EndMonth, EndDay, EndYear, DayOfWeek, ...rest
    rest = fields[9:]
    new = ["RunPeriod", fields[1], str(begin[0]), str(begin[1]), str(year), str(end[0]), str(end[1]), str(year), ""] + rest
    obj = ",\n    ".join(new) + ";"
    return idf_text[: matches[0].start()] + obj + idf_text[matches[0].end() :]


SIMCONTROL_RE = re.compile(r"^\s*SimulationControl\s*,.*?;", re.M | re.S)
SHADOWCALC_RE = re.compile(r"^\s*ShadowCalculation\s*,.*?;", re.M | re.S)


def set_shading_update_days(idf_text: str, days: int) -> str:
    """Sets ShadowCalculation 'Shading Calculation Update Frequency' (field 3). A chunk whose length is not a
    multiple of the update period sees a different averaged shading day at its end than the full year does;
    daily updates remove that effect for the comparison (the orchestrator will align chunks instead)."""
    m = SHADOWCALC_RE.search(idf_text)
    if not m:
        return idf_text + f"\nShadowCalculation,PolygonClipping,Periodic,{days};\n"
    fields = strip_comments(m.group(0))
    while len(fields) < 4:
        fields.append("")
    fields[3] = str(days)
    return idf_text[: m.start()] + ",\n    ".join(fields) + ";" + idf_text[m.end() :]


def enable_runperiod_simulation(idf_text: str) -> str:
    """Sets SimulationControl 'Run Simulation for Weather File Run Periods' to Yes (test files rely on -a)."""
    m = SIMCONTROL_RE.search(idf_text)
    if not m:
        return idf_text
    fields = strip_comments(m.group(0))
    while len(fields) < 6:
        fields.append("")
    fields[5] = "Yes"
    return idf_text[: m.start()] + ",\n    ".join(fields) + ";" + idf_text[m.end() :]


def make_chunks(months_per_chunk: int, overlap_days: int, year: int):
    chunks = []
    m = 1
    while m <= 12:
        m_end = min(12, m + months_per_chunk - 1)
        body_begin = (m, 1)
        body_end = (m_end, calendar.monthrange(year, m_end)[1])
        if m == 1:
            run_begin = body_begin
        else:
            prev_month = m - 1
            prev_days = calendar.monthrange(year, prev_month)[1]
            run_begin = (prev_month, prev_days - overlap_days + 1)
        chunks.append({"name": f"chunk_{m:02d}_{m_end:02d}", "run_begin": run_begin, "run_end": body_end, "body_begin": body_begin, "body_end": body_end})
        m = m_end + 1
    return chunks


def run_eplus(exe: Path, idf: Path, epw: Path, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.run([str(exe), "-w", str(epw), "-d", str(outdir), str(idf)], cwd=str(outdir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wall = time.perf_counter() - t0
    (outdir / "stdout.txt").write_text(proc.stdout)
    warmup = None
    eio = outdir / "eplusout.eio"
    if eio.exists():
        for line in eio.read_text(errors="replace").splitlines():
            if line.startswith("Environment:WarmupDays"):
                warmup = int(line.split(",")[1])
    return {"wall": wall, "ok": "EnergyPlus Completed Successfully" in proc.stdout, "warmup_days": warmup}


def parse_eso(path: Path, run_period_name: str) -> dict:
    """Returns {(key, variable): {(month, day, hour): value}} for hourly records of the run period."""
    names = {}
    data = {}
    in_dict = True
    in_runperiod = False
    stamp = None
    with path.open(errors="replace") as f:
        for line in f:
            if in_dict:
                if line.startswith("End of Data Dictionary"):
                    in_dict = False
                    continue
                p = line.split(",", 2)
                if len(p) == 3 and p[0].isdigit() and int(p[0]) > 5:
                    rest = p[2].split("!")[0].strip()
                    if "," in rest:
                        key, _, var = rest.partition(",")
                    else:  # meters have no key
                        key, var = "", rest
                    names[int(p[0])] = (key.strip(), var.strip())
                continue
            if line.startswith("End of Data"):
                break
            p = line.rstrip("\n").split(",")
            rid = int(p[0])
            if rid == 1:
                in_runperiod = p[1].strip().upper() == run_period_name.upper()
                stamp = None
            elif rid == 2:
                stamp = (int(p[2]), int(p[3]), int(p[5]))  # month, day of month, hour
            elif rid in (3, 4, 5):
                stamp = None
            elif in_runperiod and stamp is not None and rid in names:
                data.setdefault(names[rid], {})[stamp] = float(p[1])
    return data


def in_range(md: tuple[int, int], begin: tuple[int, int], end: tuple[int, int]) -> bool:
    return begin <= md <= end


def compare(ref: dict, chunk: dict, begin, end) -> dict:
    """Deviation statistics of chunk vs reference over days begin..end (inclusive)."""
    temp_abs = []
    meters = {}
    for (key, var), series in chunk.items():
        if (key, var) not in ref:
            continue
        rseries = ref[(key, var)]
        if var.startswith("Zone Mean Air Temperature"):
            for (m, d, h), v in series.items():
                if in_range((m, d), begin, end) and (m, d, h) in rseries:
                    temp_abs.append(abs(v - rseries[(m, d, h)]))
        elif key == "":
            tot_c = tot_r = 0.0
            hourly_rel = []
            for (m, d, h), v in series.items():
                if in_range((m, d), begin, end) and (m, d, h) in rseries:
                    r = rseries[(m, d, h)]
                    tot_c += v
                    tot_r += r
                    if abs(r) > 1.0e6:  # ignore hours below 1 MJ for the relative measure
                        hourly_rel.append(abs(v - r) / abs(r))
            if tot_r != 0.0:
                meters[var] = {"total_rel_diff": (tot_c - tot_r) / tot_r, "hourly_rel_max": max(hourly_rel) if hourly_rel else 0.0, "hourly_rel_mean": statistics.fmean(hourly_rel) if hourly_rel else 0.0}
    return {
        "zone_temp_hours": len(temp_abs),
        "zone_temp_max_abs_dev_K": max(temp_abs) if temp_abs else None,
        "zone_temp_mean_abs_dev_K": statistics.fmean(temp_abs) if temp_abs else None,
        "zone_temp_p99_abs_dev_K": sorted(temp_abs)[int(0.99 * (len(temp_abs) - 1))] if temp_abs else None,
        "meters": meters,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("idf", type=Path)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--epw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4, help="concurrent EnergyPlus processes")
    ap.add_argument("--months-per-chunk", type=int, default=1)
    ap.add_argument("--overlap-days", type=int, default=3)
    ap.add_argument("--year", type=int, default=2017, help="calendar year fixed in every RunPeriod (weekday alignment)")
    ap.add_argument("--min-warmup-days", type=int, default=None, help="override Building minimum warm-up days for chunks")
    ap.add_argument("--skip-reference", action="store_true", help="reuse an existing reference run in OUT/reference")
    ap.add_argument("--shading-update-days", type=int, default=None, help="override the shading calculation update frequency in all inputs")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.exe, args.epw, args.idf, args.out = (p.resolve() for p in (args.exe, args.epw, args.idf, args.out))
    text = enable_runperiod_simulation(args.idf.read_text(errors="replace"))
    if args.shading_update_days is not None:
        text = set_shading_update_days(text, args.shading_update_days)
    fields = strip_comments(RUNPERIOD_RE.search(text).group(0))
    run_period_name = fields[1]

    # Reference: full year, same fixed year and the same added outputs
    ref_text = rewrite_runperiod(text, (1, 1), (12, 31), args.year) + ADDED_OUTPUTS
    ref_idf = args.out / "reference.idf"
    ref_idf.write_text(ref_text)

    chunks = make_chunks(args.months_per_chunk, args.overlap_days, args.year)
    jobs = []
    if not args.skip_reference:
        jobs.append(("reference", ref_idf, args.out / "reference"))
    for c in chunks:
        ctext = rewrite_runperiod(text, c["run_begin"], c["run_end"], args.year) + ADDED_OUTPUTS
        if args.min_warmup_days is not None:
            ctext = re.sub(r"(^\s*Building\s*,.*?;)", lambda m: set_min_warmup(m.group(1), args.min_warmup_days), ctext, count=1, flags=re.M | re.S)
        cidf = args.out / f"{c['name']}.idf"
        cidf.write_text(ctext)
        jobs.append((c["name"], cidf, args.out / c["name"]))

    results = {}
    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_eplus, args.exe, idf, args.epw, outdir): name for name, idf, outdir in jobs}
        for fut in concurrent.futures.as_completed(futs):
            name = futs[fut]
            results[name] = fut.result()
            r = results[name]
            print(f"{name:14s} {r['wall']:7.1f} s  warmup days={r['warmup_days']}  {'ok' if r['ok'] else 'FAILED'}", flush=True)
    wall_total = time.perf_counter() - t_start

    ref = parse_eso(args.out / "reference" / "eplusout.eso", run_period_name)
    report = {"model": args.idf.name, "chunks": [], "runs": results, "wall_all_jobs_s": wall_total}
    print(f"\n{'chunk':14s} {'overlap maxΔT':>13s} {'body maxΔT':>10s} {'body p99ΔT':>10s} {'body meanΔT':>11s}  {'elec total':>10s} {'gas total':>10s} {'elec hourly max':>15s}")
    for c in chunks:
        cd = parse_eso(args.out / c["name"] / "eplusout.eso", run_period_name)
        entry = dict(c)
        if c["run_begin"] != c["body_begin"]:
            ob_end = (c["body_begin"][0] - 1, calendar.monthrange(args.year, c["body_begin"][0] - 1)[1])
            entry["overlap"] = compare(ref, cd, c["run_begin"], ob_end)
        entry["body"] = compare(ref, cd, c["body_begin"], c["body_end"])
        report["chunks"].append(entry)
        b = entry["body"]
        o = entry.get("overlap")
        me = b["meters"].get("Electricity:Facility [J]", {})
        mg = b["meters"].get("NaturalGas:Facility [J]", {})
        fmt = lambda x: f"{x:.4f}" if x is not None else "-"
        print(f"{c['name']:14s} {fmt(o['zone_temp_max_abs_dev_K']) if o else '-':>13s} {fmt(b['zone_temp_max_abs_dev_K']):>10s} {fmt(b['zone_temp_p99_abs_dev_K']):>10s} {fmt(b['zone_temp_mean_abs_dev_K']):>11s}  {me.get('total_rel_diff', 0)*100:9.3f}% {mg.get('total_rel_diff', 0)*100:9.3f}% {me.get('hourly_rel_max', 0)*100:14.3f}%")

    # Whole-year totals from chunk bodies vs reference
    print()
    for var in ["Electricity:Facility [J]", "NaturalGas:Facility [J]", "Cooling:EnergyTransfer [J]", "Heating:EnergyTransfer [J]"]:
        rs = ref.get(("", var), {})
        ref_total = sum(rs.values())
        chunk_total = 0.0
        for c in chunks:
            cd = parse_eso(args.out / c["name"] / "eplusout.eso", run_period_name)
            for (m, d, h), v in cd.get(("", var), {}).items():
                if in_range((m, d), c["body_begin"], c["body_end"]):
                    chunk_total += v
        if ref_total:
            report.setdefault("annual_totals", {})[var] = {"reference": ref_total, "chunks": chunk_total, "rel_diff": (chunk_total - ref_total) / ref_total}
            print(f"annual {var:32s} reference {ref_total/3.6e9:12.1f} MWh  chunks {chunk_total/3.6e9:12.1f} MWh  diff {100*(chunk_total-ref_total)/ref_total:+.4f}%")

    ref_wall = results.get("reference", {}).get("wall")
    chunk_walls = [results[c["name"]]["wall"] for c in chunks if c["name"] in results]
    if chunk_walls:
        print(f"\nchunk walls: max {max(chunk_walls):.1f} s, mean {statistics.fmean(chunk_walls):.1f} s, {len(chunk_walls)} chunks run {args.jobs} at a time")
        if args.skip_reference:
            print(f"all chunks wall {wall_total:.1f} s (chunks only)")
            report["chunks_wall_s"] = wall_total
        if ref_wall:
            print(f"reference wall {ref_wall:.1f} s")
    (args.out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    return 0


def set_min_warmup(building_obj: str, min_days: int) -> str:
    fields = strip_comments(building_obj)
    # Building: Name, North Axis, Terrain, Loads Tol, Temp Tol, Solar Dist, Max Warmup Days, Min Warmup Days
    while len(fields) < 9:
        fields.append("")
    fields[8] = str(min_days)
    return ",\n    ".join(fields[:9]) + ";"


if __name__ == "__main__":
    sys.exit(main())
