"""Keep native dependency imports shallow in the frozen prediction worker."""
import sys
if '--individual-worker' in sys.argv:
    import faulthandler
    if sys.stderr is not None:
        faulthandler.enable(all_threads=True)
    # pandas' nested native imports can exhaust the Windows stack when first
    # reached through the nnU-Net -> preprocessor -> resampling import chain.
    import pandas
