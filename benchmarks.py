"""Compute BP1 and BP2 with the same numerical settings as the Z2 scan.

Run ``python benchmarks.py`` to write ``output/benchmarks.csv``.
The calculation uses the general EOS, LTE and all three entropy sources,
and searches for a higher detonation root. Energies are in GeV and widths
in GeV^-1. Model conventions: https://arxiv.org/abs/2009.14295.
"""

from Z2_model import make_model_ls
from generate_scan_points import output_dir
from scan import empty_result, wall_velocities, write_rows


def main():
    rows = []
    for name, lambda_hs in (("BP1", 0.8), ("BP2", 0.9)):
        model = make_model_ls(100.0, lambda_hs, 1.0)
        transition = model.find_Tn()
        row = empty_result(100.0, lambda_hs)
        row.update(Tc=model.Tc, Tn=model.Tn,
                   vc=abs(float(transition["crit_trans"]["low_vev"][0])),
                   vn=abs(float(transition["low_vev"][0])), alpha_n=model.alpha_n())
        wall_velocities(model, row)
        rows.append(row)
        print(f"{name}: Tn={model.Tn:.8f} GeV, "
              f"vw={row['v_ours']} ({row['type_ours']}), "
              f"higher detonation={row['v_detonation']} ({row['status_detonation']})")
    write_rows(output_dir / "benchmarks.csv", rows)


if __name__ == "__main__":
    main()
