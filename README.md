Requires: Numpy (http://numeric.scipy.org),
and a fortran compiler supported by numpy.f2py,

Now works for python 3.12 (build system updated to use meson instead of f2py)

Please read LICENSE.spherepack

Installation: 

python setup.py install

(to change default fortran compiler you can use e.g.
 python setup.py build config_fc --fcompiler=g95)

View documentation by pointing your browser to html/index.html.

Example programs are provided in the examples directory.

Copyright: (applies only to python binding, Spherepack fortran
source code licensing is in LICENSE.spherepack)

Permission to use, copy, modify, and distribute this software and its
documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appear in all copies and that
both that copyright notice and this permission notice appear in
supporting documentation.
THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO
EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR
CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF
USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.

-- Jeff Whitaker <Jeffrey.S.Whitaker@noaa.gov>
