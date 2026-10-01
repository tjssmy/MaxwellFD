"""Pin BLAS to one thread unless the environment already chose a count.

A small 3D SuperLU factor does not speed up across every core. OpenBLAS
at the full machine width makes that factor slower, so the suite leaves
the count at one when it is unset.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
