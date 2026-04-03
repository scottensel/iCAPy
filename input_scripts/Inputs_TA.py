# Total activation related information
####################################################
def setup_ta_params():
    """
    Returns
    -------
    param : dict
        Only TA-specific keys. Meant to be merged into an
        existing param dict from setup_data_params().
    """
    param = {}

    # ── Force flags ──────────────────────────────────────────────────────────
    param['force_TA_on_real'] = 1
    param['force_TA_on_surrogate'] = 0

    # ── Parallelism / acceleration flags ─────────────────────────────────────

    # Run temporal regularization using parallel CPU processes.
    # Highly recommended — substantial speedup at the cost of 100% CPU usage
    # during the temporal step.
    param['runParallel'] = 1

    # Use Numba JIT compilation for the spatial TA operations.
    # Requires: pip install numba
    # First run will be slower due to JIT compilation; subsequent runs use
    # the cached compiled code automatically.
    # I DO NOT RECOMMEND THIS. GPU IS MUCH FASTER AND THIS DOES NOT GIVE MUCH SPEED
    param['use_numba'] = 0

    # Number of time points per Numba chunk.
    # Controls peak RAM usage during the spatial step.
    # Set to None to process all T time points at once (fine if you have
    # plenty of RAM).
    param['numba_chunk_size'] = None

    # Use GPU (CuPy) for spatial TA array operations.
    # Requires: pip install cupy-cuda12x  (match your CUDA version)
    # If use_gpu=1 and CuPy is not installed, falls back to NumPy with a warning.
    # use_gpu takes precedence over use_numba if both are set to 1.
    param['use_gpu'] = 1

    # Number of time points to process per GPU chunk.
    # The full (X x Y x Z x T) volume requires ~64GB of GPU working memory
    # across all MyProx internal arrays, which exceeds any consumer GPU.
    # Chunking along T processes a subset of time points at a time.
    #
    # Start with a value around 120 if you have an 8 GB GPU.
    # If you get an OutOfMemoryError, reduce by 20 until it works.
    # If you have more VRAM, increase for fewer GPU transfers and faster runs.
    # Set to None only if GPU has A LOT OF VRAM (e.g. A100 80 GB).
    param['gpu_chunk_size'] = 120

    # ── Checkpointing ─────────────────────────────────────────────────────────
    # Save the full loop state (TC_OUT, xT, xS, param, iteration index) every
    # N outer iterations so the run can be resumed if it is interrupted.
    #
    # The checkpoint is saved as a single rolling file:
    #   <outDir_TA>/ta_checkpoint.h5
    # Each new checkpoint overwrites the previous one — only one file ever
    # exists at a time. The file is deleted automatically on successful
    # completion so no extra files accumulate.
    #
    # On restart the pipeline automatically detects the checkpoint and resumes
    # from the next iteration — no manual steps needed.
    #
    # Options:
    #   1  — save after every outer iteration (safest, ~300 MB per save)
    #   2  — save every other iteration (good balance for Nit=5)
    #   0  — disable checkpointing entirely (fastest, no recovery possible)
    #
    # Performance: each save takes a few seconds — negligible compared to
    # the cost of a full TA iteration which takes minutes.
    param['checkpoint_every'] = 1

    # ── HRF and regularization ────────────────────────────────────────────────

    # Type of assumed hemodynamic response function
    # Options: 'bold', 'spmhrf', 'mion'
    param['HRF'] = 'bold'

    # Number of outer iterations for the forward-backward scheme
    param['Nit'] = 5

    # Number of iterations for temporal and spatial regularization schemes
    param['NitTemp'] = 500
    param['NitSpat'] = 400

    # Tolerance threshold for convergence of TA methods
    param['tol'] = 1e-6

    # Weighting parameters for temporal and spatial TA schemes:
    # 3/4 to 1/4 trade-off as recommended by Karahanoglu and Farouj
    param['weights'] = [0.75, 0.25]

    # Regularization weight multiplier for temporal regularization
    param['LambdaTempCoef'] = 1 / 0.8095

    # Regularization weight for spatial regularization
    param['LambdaSpat'] = 6

    # Weight matrix generation parameter for gray matter map
    param['sigma'] = 0.5

    return param
