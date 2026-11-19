from sdge_microgrid.tariffs import buy_price, grid_cost


def test_buy_price_on_peak_and_off_peak():
    assert buy_price(16) == 0.691
    assert buy_price(20) == 0.691
    assert buy_price(15) == 0.374
    assert buy_price(21) == 0.374
    assert buy_price(3) == 0.374


def test_grid_cost_import_and_export():
    assert abs(grid_cost(2.0, 0.691) - 1.382) < 1e-9
    assert abs(grid_cost(-2.0, 0.691) - (-0.10)) < 1e-9
    assert grid_cost(0.0, 0.691) == 0.0
