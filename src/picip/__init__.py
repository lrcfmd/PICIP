from picip.picip import PICIP, Sample, Prediction, Suggestions
from picip.phase_field import Phase_Field
from picip.spread import Spread
from picip.visualise_cube import make_plotter

# picip.evaluation is deliberately *not* imported here.  It caps BLAS/OpenMP
# threads at import time (needed for its parallel trial running), and that must
# stay an opt-in side effect of `import picip.evaluation`, not of `import picip`.
