import itertools

import numpy as np
import pytest
import spharm

# T799 transforms result in nans
# T382 on a 512x1024 grid seems to work fine
NTRUNC = [21, 42, 63]  # , 85]
NLATS = [32, 64, 96]  # , 128]


@pytest.fixture(
    params=itertools.product(["gaussian", "regular"], ["computed", "stored"], NLATS)
)
def spharmt(request):
    nlat = request.param[2]
    nlon = nlat * 2
    transform = spharm.Spharmt(
        nlon, nlat, gridtype=request.param[0], legfunc=request.param[1]
    )
    return transform


@pytest.mark.parametrize("transform_multiple", [False, True])
@pytest.mark.parametrize("ntrunc", NTRUNC)
class TestSpharmt:
    """Test the Spharmt class.

    longitude 0-360
    latitude 90 -- -90
    """

    def test_grdtospec(self, spharmt, transform_multiple, ntrunc):
        if ntrunc > spharmt.nlat - 1:
            pytest.skip("ntrunc larger than nlat")
        if transform_multiple:
            shape = (spharmt.nlat, spharmt.nlon, 5)
        else:
            shape = (spharmt.nlat, spharmt.nlon)
        test_data = np.ones(shape, dtype="f4")
        coeffs = spharmt.grdtospec(test_data, ntrunc)
        ncoeffs = (ntrunc + 1) * (ntrunc + 2) // 2
        assert coeffs.shape[0] == ncoeffs
        if transform_multiple:
            assert coeffs.shape[1] == 5
        assert np.all(np.isfinite(coeffs))
        assert coeffs[0] == pytest.approx(np.sqrt(2))
        assert coeffs[1:] == pytest.approx(0, abs=1e-6 * ncoeffs)

    def test_spectogrd(self, spharmt, transform_multiple, ntrunc):
        if ntrunc > spharmt.nlat - 1:
            pytest.skip("ntrunc larger than nlat")
        ncoeffs = (ntrunc + 1) * (ntrunc + 2) // 2
        if transform_multiple:
            shape = (ncoeffs, 5)
        else:
            shape = (ncoeffs,)
        test_data = np.ones(shape, dtype="c8")
        grid = spharmt.spectogrd(test_data)
        assert grid.shape[0] == spharmt.nlat
        assert grid.shape[1] == spharmt.nlon
        if transform_multiple:
            assert grid.shape[2] == 5
        assert np.all(np.isfinite(grid))

    def test_roundtrip_spectral_grid(self, spharmt, transform_multiple, ntrunc):
        if ntrunc > spharmt.nlat - 1:
            pytest.skip("ntrunc larger than nlat")
        elif spharmt.gridtype == "regular" and ntrunc > 1 * (spharmt.nlat + 1) / 2:
            pytest.skip("ntrunc larger than nlat on regular grid")
        ncoeffs = (ntrunc + 1) * (ntrunc + 2) // 2
        if transform_multiple:
            shape = (ncoeffs, 5)
        else:
            shape = (ncoeffs,)
        size = np.prod(shape)
        test_data = np.arange(size, dtype="c8").reshape(shape)
        grid = spharmt.spectogrd(test_data)
        coeffs = spharmt.grdtospec(grid, ntrunc)
        assert coeffs == pytest.approx(test_data, abs=1e-6 * size, rel=7e-6)

    def test_getuv(self, spharmt, transform_multiple, ntrunc):
        if ntrunc > spharmt.nlat - 1:
            pytest.skip("ntrunc larger than nlat")
        ncoeffs = (ntrunc + 1) * (ntrunc + 2) // 2
        shape = (ncoeffs,)
        if transform_multiple:
            shape = shape + (5,)
        vrt = div = np.ones(shape, "c8")
        ugrid_vgrid = spharmt.getuv(vrt, div)
        for grid in ugrid_vgrid:
            assert grid.shape[0] == spharmt.nlat
            assert grid.shape[1] == spharmt.nlon
            if transform_multiple:
                assert grid.shape[2] == 5
            assert np.all(np.isfinite(grid))

    def test_getvrtdivspec(self, spharmt, transform_multiple, ntrunc):
        if ntrunc > spharmt.nlat - 1:
            pytest.skip("ntrunc larger than nlat")
        shape = (spharmt.nlat, spharmt.nlon)
        if transform_multiple:
            shape = shape + (5,)
        ugrid = vgrid = np.ones(shape, dtype="f4")
        vrt_div = spharmt.getvrtdivspec(ugrid, vgrid, ntrunc)
        ncoeffs = (ntrunc + 1) * (ntrunc + 2) // 2
        for coeffs in vrt_div:
            assert coeffs.shape[0] == ncoeffs
            if transform_multiple:
                assert coeffs.shape[1] == 5
            assert np.all(np.isfinite(coeffs))

    @pytest.mark.xfail
    @pytest.mark.parametrize("scenario", [1, 2, 3])
    def test_roundtrip_winds(self, spharmt, transform_multiple, ntrunc, scenario):
        if ntrunc > spharmt.nlat / 2:
            pytest.skip("ntrunc larger than nlat")
        ncoeffs = (ntrunc + 1) * (ntrunc + 2) // 2
        shape = (ncoeffs,)
        if transform_multiple:
            shape = shape + (5,)
        size = np.prod(shape)
        test_coeffs = np.arange(size, dtype="c8").reshape(shape)
        zero_coeffs = np.zeros_like(test_coeffs)
        if scenario & 1:
            div = test_coeffs
        else:
            div = zero_coeffs
        if scenario & 2:
            vrt = test_coeffs
        else:
            vrt = zero_coeffs
        ugrid, vgrid = spharmt.getuv(vrt, div)
        actual_vrt, actual_div = spharmt.getvrtdivspec(ugrid, vgrid, ntrunc)
        # TODO: Test with original code and no slicing
        assert actual_vrt[4:] == pytest.approx(vrt[4:], abs=1e-6 * size)
        assert actual_div[4:] == pytest.approx(div[4:], abs=1e-6 * size)

    def test_getgrad(self, spharmt, transform_multiple, ntrunc):
        if ntrunc > spharmt.nlat - 1:
            pytest.skip("ntrunc larger than nlat")
        ncoeffs = (ntrunc + 1) * (ntrunc + 2) // 2
        shape = (ncoeffs,)
        if transform_multiple:
            shape = shape + (5,)
        test_data = np.ones(shape, dtype="c8")
        uchi_vchi = spharmt.getgrad(test_data)
        for grid in uchi_vchi:
            assert grid.shape[0] == spharmt.nlat
            assert grid.shape[1] == spharmt.nlon
            if transform_multiple:
                assert grid.shape[2] == 5
            assert grid == pytest.approx(0.0, abs=1e-6 * max(ncoeffs, np.prod(grid.shape[:2])))

    def test_getpsichi(self, spharmt, transform_multiple, ntrunc):
        if ntrunc > spharmt.nlat - 1:
            pytest.skip("ntrunc larger than nlat")
        shape = (spharmt.nlat, spharmt.nlon)
        if transform_multiple:
            shape = shape + (5,)
        test_data = np.ones(shape, dtype="f4")
        psi_chi = spharmt.getpsichi(test_data, test_data, ntrunc)
        ncoeffs = (ntrunc + 1) * (ntrunc + 2) // 2
        for coeffs in psi_chi:
            assert coeffs.shape[:len(shape)] == shape
            if transform_multiple:
                assert coeffs.shape[2] == 5
            assert np.all(np.isfinite(coeffs))

    def test_specsmooth(self, spharmt, transform_multiple, ntrunc):
        if ntrunc > spharmt.nlat - 1:
            pytest.skip("ntrunc larger than nlat")
        shape = (spharmt.nlat, spharmt.nlon)
        if transform_multiple:
            shape = shape + (5,)
        test_data = np.ones(shape, dtype="f4")
        smoothing_coeffs = np.ones(shape[0], dtype="f4")
        smoothed = spharmt.specsmooth(test_data, smoothing_coeffs)
        assert np.squeeze(smoothed) == pytest.approx(test_data, abs=1e-6 * shape[0])


@pytest.mark.parametrize("smooth", [None, True])
def test_regrid(spharmt, smooth):
    ntrunc = 25
    sourcegrid = spharm.Spharmt(54, 27)
    test_data = np.ones((27, 54), dtype="f4")
    if smooth is not None:
        smooth = np.ones(spharmt.nlat)
    hires_data = spharm.regrid(sourcegrid, spharmt, test_data, ntrunc, smooth)
    lowres_data = spharm.regrid(spharmt, sourcegrid, hires_data, ntrunc)
    assert np.squeeze(lowres_data) == pytest.approx(test_data, abs=1e-6 * spharmt.nlat)


@pytest.mark.parametrize("nlat", NLATS)
def test_gaussian_lats_wts(nlat):
    lats, weights = spharm.gaussian_lats_wts(nlat)
    assert np.all(lats <= 90)
    assert np.all(lats >= -90)
    assert np.all(weights >= 0)
    assert np.sum(weights) == pytest.approx(2)


@pytest.mark.parametrize("ntrunc", NTRUNC)
def test_getspecindx(ntrunc):
    wavenumber, degree = spharm.getspecindx(ntrunc)
    assert np.all(wavenumber <= ntrunc)
    assert np.all(degree <= ntrunc)
    assert np.all(wavenumber <= degree)
    assert np.all(degree >= 0)
    assert np.all(wavenumber >= 0)
    assert wavenumber.shape[0] == (ntrunc + 1) * (ntrunc + 2) // 2


@pytest.mark.parametrize("lat", np.linspace(-90, 90, 7))
@pytest.mark.parametrize("ntrunc", NTRUNC)
def test_legendre(lat, ntrunc):
    pnm = spharm.legendre(lat, ntrunc)
    ncoeffs = (ntrunc + 1) * (ntrunc + 2) // 2
    assert pnm.shape[0] == ncoeffs
    assert np.all(np.isfinite(pnm))


@pytest.mark.parametrize("m", range(1, 20))
def test_getgeodesicpts(m):
    lats, lons = spharm.getgeodesicpts(m)
    assert np.all(lats >= -90)
    assert np.all(lats <= 90)
    assert np.all(lons >= 0)
    assert np.all(lons <= 360)
    npoints = 10 * (m - 1) ** 2 + 2
    assert lats.shape[0] == npoints
    # Only other test I can think of involves breaking out the
    # heaviside formula and checking each pair of points, which is
    # O(m**4)


@pytest.mark.parametrize("lon", np.arange(0, 360, 30))
@pytest.mark.parametrize("lat", np.linspace(-90, 90, 7))
@pytest.mark.parametrize("ntrunc", NTRUNC)
def test_specintrp(lon, lat, ntrunc):
    ncoeffs = (ntrunc + 1) * (ntrunc + 2) // 2
    test_coeffs = np.ones(ncoeffs, dtype="c8")
    legfuncs = spharm.legendre(lat, ntrunc)
    interpolated = spharm.specintrp(lon, test_coeffs, legfuncs)
    assert np.isfinite(interpolated)