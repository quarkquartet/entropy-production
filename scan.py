"""Calculate the Z2 scan with the general EOS and lambda_s = 1.

Run ``python generate_scan_points.py`` followed by ``python scan.py``.
``--workers`` sets the number of independent parameter points in parallel.
Energies are in GeV and wall widths in GeV^-1. WallGo comparisons are
computed separately. Model conventions: https://arxiv.org/abs/2009.14295.
"""

import argparse
import contextlib
import csv
import os
from multiprocessing import Pool
from pathlib import Path

from Z2_model import make_model_ls, v2
from generate_scan_points import output_dir


def empty_result(mass, lambda_hs):
    fields = ("vc", "Tc", "vn", "Tn", "alpha_n", "v_LTE", "v_ours",
              "vw_nLTE", "vw_ball", "type_ours", "vJ", "L_ours",
              "status", "error", "v_detonation", "status_detonation")
    return dict(mass=mass, lambda_hs=lambda_hs, **dict.fromkeys(fields, ""))


def wall_velocities(model, row):
    """Evaluate each entropy source and search for a higher detonation root."""
    row.update(vJ=model.find_vJ(), L_ours=model.wall_width())
    for limit, column in (("interpolated", "v_ours"),
                          ("near_lte", "vw_nLTE"), ("ballistic", "vw_ball")):
        # The production scan extends the model's default search down to 0.001.
        lte, row[column], mode = model.find_vw(L=row["L_ours"], limit=limit, vw_min=1e-3)
        if limit == "interpolated":
            row.update(v_LTE=lte, type_ours=mode)
    row["status"] = next((f"no_{name}" for name in
        ("v_ours", "vw_nLTE", "vw_ball") if row[name] is None), "ok")
    if row["type_ours"] in ("runaway", "det"):
        row.update(v_detonation=row["v_ours"], status_detonation=f"original_{row['type_ours']}")
    elif row["type_ours"] in ("def", "hyb") and row["v_ours"] is not None:
        row["v_detonation"] = model.find_detonation_solution(
            L=row["L_ours"], vw_min=row["v_ours"])
        row["status_detonation"] = model.detonation_status


def run_point(mass, lambda_hs):
    row = empty_result(mass, lambda_hs)
    try:
        if mass**2 >= 0.5 * lambda_hs * v2:
            row["status"] = "no_singlet_minimum"
            return row
        model = make_model_ls(mass, lambda_hs, 1.0)
        transition = model.find_Tn()
        if transition is None:
            row["status"] = "no_nucleation"
            return row
        high_h, high_s = abs(transition["high_vev"])
        low_h, low_s = abs(transition["low_vev"])
        row.update(Tc=model.Tc, Tn=model.Tn, vn=float(low_h),
                   vc=abs(float(transition["crit_trans"]["low_vev"][0])))
        # Select singlet-to-Higgs endpoints at nucleation using the scan's GeV cuts.
        if not (high_h < 1.0 and high_s > 10.0 and low_h > 10.0 and low_s < 1.0):
            row["status"] = "wrong_endpoints"
            return row
        row["alpha_n"] = model.alpha_n()
        wall_velocities(model, row)
    except Exception as error:
        row.update(status="error", error=f"{type(error).__name__}: {error}")
    return row


def quiet_point(point):
    with open(os.devnull, "w") as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
        return run_point(*point)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=empty_result(0.0, 0.0))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-file", type=Path, default=output_dir / "uniform_rectangle_points.csv")
    parser.add_argument("--output", type=Path, default=output_dir / "vw_scan.csv")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    with args.points_file.open(newline="") as handle:
        points = sorted((float(row["mass"]), float(row["lambda_hs"]))
                        for row in csv.DictReader(handle))
    with Pool(args.workers) as pool:
        write_rows(args.output, pool.imap(quiet_point, points))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
