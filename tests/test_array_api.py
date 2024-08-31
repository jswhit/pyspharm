import pytest

import spharm

array_api = pytest.importorskip("numpy.array_api")

test_data = [array_api.ones((32, 64), dtype=array_api.float32)]

try:
    import xarray as xr
except ImportError:
    pass
else:
    test_data.append(xr.DataArray(test_data[0], dims=["lat", "lon"]))

transform = spharm.Spharmt(64, 32)


@pytest.mark.parametrize("data", test_data)
def test_grdtospec(data):
    result = transform.grdtospec(data, 21)


@pytest.mark.parametrize("data", test_data)
def test_regrid(data):
    result = spharm.regrid(transform, transform, data, 21)
