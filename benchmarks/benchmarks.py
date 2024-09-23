# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import itertools

import numpy as np

import spharm

TEST_SIZES = {
    21: (32, 64),
    42: (64, 128),
    63: (96, 182),
    84: (128, 256),
    127: (196, 384),
    168: (256, 512),
    252: (384, 768),
    # 336: (512, 1024),
}


class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """

    params = (list(TEST_SIZES.keys()), ["computed", "stored"])
    param_names = ("ntrunc", "method")

    def setup(self, ntrunc, method):
        data_size = TEST_SIZES[ntrunc]
        self.transform = spharm.Spharmt(data_size[1], data_size[0], legfunc=method)
        self.data = np.ones(data_size, dtype="f4")
        ncoeffs = (ntrunc + 1) * (ntrunc + 2) // 2
        self.coeffs = np.ones(ncoeffs, dtype="c8")

    def time_grdtospec(self, ntrunc, method):
        self.transform.grdtospec(self.data, ntrunc)

    def time_spectogrd(self, ntrunc, method):
        self.transform.spectogrd(self.coeffs)
