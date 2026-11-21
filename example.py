"""BP1: phase transition, wall width, and entropy-production wall velocity.

Run with ``python example.py``. Energies and temperatures are in GeV,
wall widths in GeV^-1, and velocities in units of the speed of light.
The model conventions follow https://arxiv.org/abs/2009.14295.
"""

from Z2_model import make_model_ls


def main():
    # BP1: (m_S, lambda_hs, lambda_s) = (100 GeV, 0.8, 1).
    model = make_model_ls(100.0, 0.8, 1.0)

    # Trace the phases and solve the nucleation condition S_3(T_n)/T_n = 140.
    transition = model.find_Tn()
    if transition is None:
        raise RuntimeError("the benchmark has no nucleation transition")

    # Evaluate the Higgs wall-width moment along the nucleation path.
    width = model.wall_width()

    # Use the effective-potential EOS and the interpolated entropy source.
    vw_lte, vw, mode = model.find_vw(eos="general", L=width)
    if vw is None:
        raise RuntimeError("the benchmark wall-velocity search is unresolved")

    print(f"T_c = {model.Tc:.8f} GeV")
    print(f"T_n = {model.Tn:.8f} GeV")
    print(f"L_h = {width:.8f} GeV^-1")
    print(f"v_w (LTE) = {vw_lte:.8f}")
    print(f"v_w (entropy production) = {vw:.8f} ({mode})")


if __name__ == "__main__":
    main()
