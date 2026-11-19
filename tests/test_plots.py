from sdge_microgrid.eval.plots import plot_trajectory_72h


def test_plot_trajectory_writes_file(tmp_path):
    infos = [
        {
            "soc": 0.5,
            "p_pv": 1.0,
            "u_shed": 0.0,
            "g": 1.0 if t < 40 else 0.0,
            "p_batt": 0.0,
        }
        for t in range(72)
    ]
    out = tmp_path / "traj.png"
    plot_trajectory_72h(infos, out, title="test")
    assert out.exists() and out.stat().st_size > 0
