import itertools

import numpy as np
import pytest

import spharm

hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies")
hy_np = pytest.importorskip("hypothesis.extra.numpy")


NTRUNC = [21, 42]
NLATS = 64

MAX_MAGNITUDE = np.float32(1e20)


@pytest.fixture(
    params=itertools.product(["regular"], ["stored"])  # "gaussian",  # "computed",
)
def spharmt(request):
    """Create a (collection of) Spharmt instances for a test."""
    nlon = NLATS * 2
    transform = spharm.Spharmt(
        nlon, NLATS, gridtype=request.param[0], legfunc=request.param[1]
    )
    return transform


@pytest.mark.parametrize("ntrunc", NTRUNC)
class TestSpharmt:

    @hypothesis.given(
        hy_np.arrays(
            np.dtype("f4"),
            (NLATS, NLATS * 2),
            elements=hy_np.from_dtype(
                np.dtype("f4"),
                allow_infinity=False,
                allow_nan=False,
                allow_subnormal=False,
                max_value=MAX_MAGNITUDE,
                min_value=-MAX_MAGNITUDE,
            ),
        )
    )
    @hypothesis.settings(
        suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture]
    )
    def test_roundtrip_spectral_grid(self, spharmt, ntrunc, test_data):
        coeffs = spharmt.grdtospec(test_data, ntrunc)
        smoothed_test_data = spharmt.spectogrd(coeffs)
        recalculated_coeffs = spharmt.grdtospec(smoothed_test_data, ntrunc)
        atol = 1e-6 * test_data.size * np.abs(test_data).max()
        assert recalculated_coeffs == pytest.approx(coeffs, abs=atol, rel=1e-5)
        recalculated_test_data = spharmt.spectogrd(recalculated_coeffs)
        assert recalculated_test_data == pytest.approx(
            smoothed_test_data, abs=atol, rel=1e-5
        )

    @hypothesis.given(
        *(
            2
            * (
                hy_np.arrays(
                    np.dtype("f4"),
                    (NLATS, NLATS * 2),
                    elements=hy_np.from_dtype(
                        np.dtype("f4"),
                        allow_infinity=False,
                        allow_nan=False,
                        allow_subnormal=False,
                        max_value=MAX_MAGNITUDE,
                        min_value=-MAX_MAGNITUDE,
                    ),
                ),
            )
        )
    )
    @hypothesis.settings(
        suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture]
    )
    def test_roundtrip_winds(self, spharmt, ntrunc, test_u, test_v):
        if ntrunc > spharmt.nlat / 2:
            pytest.skip("ntrunc > nlat/2")
        expected_vrt, expected_div = spharmt.getvrtdivspec(test_u, test_v, ntrunc)
        test_u, test_v = spharmt.getuv(expected_vrt, expected_div)
        actual_vrt, actual_div = spharmt.getvrtdivspec(test_u, test_v, ntrunc)
        atol = 1e-6 * test_u.size * max(test_u.max(), test_v.max())
        assert actual_vrt == pytest.approx(expected_vrt, abs=atol, rel=3e-6)
        assert actual_div == pytest.approx(expected_div, abs=atol, rel=3e-6)


@hypothesis.given(
    hy_np.arrays(
        np.dtype("f4"),
        (NLATS, NLATS * 2),
        elements=hy_np.from_dtype(
            np.dtype("f4"),
            allow_infinity=False,
            allow_nan=False,
            allow_subnormal=False,
            max_value=MAX_MAGNITUDE,
            min_value=-MAX_MAGNITUDE,
        ),
    )
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture]
)
def test_roundtrip_regrid(spharmt, test_data):
    transform = spharm.Spharmt(192, 96)
    expected_data = spharm.regrid(spharmt, spharmt, test_data)
    hires_data = spharm.regrid(spharmt, transform, expected_data)
    recovered_data = spharm.regrid(transform, spharmt, hires_data)
    atol = 1e-6 * test_data.size * np.abs(test_data).max()
    assert recovered_data == pytest.approx(expected_data, abs=atol, rel=1e-6)
