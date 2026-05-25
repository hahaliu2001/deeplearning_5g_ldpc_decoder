# Deep Learning 5G LDPC Decoder

This repository implements and evaluates 5G NR LDPC encoding, rate matching,
channel simulation, rate recovery, decoding, and deep-learning-assisted decoder
parameter optimization across three backends:

- `py5gphy/`: Python reference implementation.
- `c_lib/`: C encoder, channel helper, and LDPC decoders exposed through a shared library.
- `CUDA/`: CUDA encoder and decoder implementations for GPU experiments.
- `deep_learning/`: Training and evaluation utilities for learned LDPC decoder parameters.

The project also includes simulation scripts for BLER/BER sweeps and
deep-learning utilities for training or evaluating learned decoder parameters.

The Python LDPC implementation presented here originates from a prior repository, [`hahaliu2001/python_5gtoolbox.git`](https://github.com/hahaliu2001/python_5gtoolbox.git). Code correctness and strict adherence to the 5G standard were confirmed through extensive simulation utilizing over 60,000 generated test vectors.

## LDPC Deep Learning Notes

The deep-learning part of this project is used for offline estimation of LDPC
decoder parameters such as `alpha` and `beta`. These learned parameters can be
exported into decoder tables or used to evaluate fixed-point/min-sum decoder
variants.

After applying deep-learning-estimated parameters, the decoder performance is
not expected to be better than belief propagation (BP). BP is the reference
algorithm being approximated by min-sum-style decoders, so papers or results
claiming learned min-sum parameters outperform BP should be checked carefully.

## Features

- 5G NR LDPC encoder and decoder flow based on TS 38.212 concepts.
- CRC16, CRC24A, CRC24B, and CRC24C support.
- Base graph 1 and base graph 2 table support.
- Rate matching and rate recovery with modulation-order interleaving for
  `Qm = 1, 2, 4, 6, 8`.
- C decoder variants:
  - Floating-point BP.
  - Fixed-point min-sum.
  - Trained-table min-sum.
  - Degree-dependent min-sum.
  - Floating-point min-sum.
- CUDA decoder variants:
  - Fixed-point min-sum.
  - Floating-point BP/min-sum path exposed through the CUDA wrapper.
- Python, C, and CUDA simulation entry points that save result pickle files.

## Repository Layout

```text
.
├── py5gphy/                  Python reference encoder, decoder, CRC, modulation, LDPC helpers
├── c_lib/                    C LDPC encoder/decoder/channel library and C tests
├── CUDA/                     CUDA LDPC encoder/decoder library and CUDA tests
├── scripts/                  BLER/BER simulation entry points
├── deep_learning/            Training and evaluation utilities for learned LDPC decoder parameters
├── tables/                   Generated BG1/BG2 LDPC table text files
├── out/                      Simulation result pickle files
├── deeplearning_out/         Saved trained-model artifacts
├── test/                     Pytest tests for Python/C consistency checks
└── requirements.txt          Python environment snapshot
```

## Development Environment

This project was developed and tested in:

- Ubuntu 24.04 on WSL.
- Python 3.12.3.
- GCC/Make for the C backend.
- NVIDIA CUDA toolkit 13.0 (`nvcc` release 13.0, V13.0.88) for the CUDA backend.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

For a smaller runtime-only environment, the core Python path mainly needs
`numpy` and `scipy`. Plotting, notebooks, tests, CUDA Python packages, and deep
learning workflows use the larger dependency set in `requirements.txt`.

## Build

Build the C shared library and C test executables:

```bash
make -C c_lib
```

This creates:

- `c_lib/libnrLDPC.so`
- `c_lib/test_ldpc_5g_encoder`
- `c_lib/test_ldpc_5g_decoder`

Build the CUDA shared library and CUDA test executables:

```bash
make -C CUDA
```

This creates:

- `CUDA/libnrLDPC_cuda.so`
- `CUDA/test_ldpc_5g_encoder_cuda`
- `CUDA/test_ldpc_5g_decoder_cuda`
- `CUDA/test_ldpc_encoder_cuda`
- `CUDA/test_nr_crc_cuda`

CUDA execution requires a compatible NVIDIA driver and CUDA runtime. The CUDA
library may compile even on a machine where runtime tests cannot execute.

## Run Tests

Run the Python/C tests:

```bash
python3 -m pytest -q test
```

Run C test executables:

```bash
LD_LIBRARY_PATH=c_lib ./c_lib/test_ldpc_5g_encoder
LD_LIBRARY_PATH=c_lib ./c_lib/test_ldpc_5g_decoder
```

Run CUDA test executables after building CUDA:

```bash
LD_LIBRARY_PATH=CUDA ./CUDA/test_ldpc_5g_encoder_cuda
LD_LIBRARY_PATH=CUDA ./CUDA/test_ldpc_5g_decoder_cuda
LD_LIBRARY_PATH=CUDA ./CUDA/test_ldpc_encoder_cuda
LD_LIBRARY_PATH=CUDA ./CUDA/test_nr_crc_cuda
```

## Run Simulations

The simulation scripts are in `scripts/`. They define LDPC grid parameters,
decoder configurations, SNR ranges, and Monte Carlo trial counts near the top
of each file.

Run the C backend simulation:

```bash
python3 scripts/sim_c_ldpc_encoder_decoder.py
```

Run the CUDA backend simulation:

```bash
python3 scripts/sim_cuda_ldpc_encoder_decoder.py
```

Run the pure Python reference simulation:

```bash
python3 scripts/sim_py_ldpc_encoder_decoder.py
```

Results are saved as pickle files under `out/`. Each result stores:

- simulation result rows,
- decoder configuration list,
- SNR list,
- number of Monte Carlo trials.

The shared helper `scripts/ldpc_c_sim_common.py` validates grid combinations
before running long sweeps. Unsupported combinations are skipped with a warning,
and an all-invalid grid raises `ValueError`.

## Decoder Configuration

For the C simulation, decoder rows use:

```text
[algo, max_iterations, alpha, beta, filling_bit_value, max_llr, min_llr]
```

The C algorithm IDs are:

```text
0  Floating-point BP
1  Fixed-point min-sum
2  Trained-table min-sum
3  Degree-dependent min-sum
4  Floating-point min-sum
```

`alpha = 22936` represents approximately `0.7` in fixed-point normalized
min-sum configurations.

## Deep Learning Utilities

The `deep_learning/` folder contains scripts and helpers for training and
evaluating learned LDPC decoder parameters. The saved artifacts currently live
under `deeplearning_out/`.

Typical entry points include:

```bash
python3 deep_learning/train_single_alpha_ldpc_model.py
python3 deep_learning/train_degree_alpha_ldpc_model.py
python3 deep_learning/train_checkpoint_alpha_ldpc_model.py
```

Check the script-level constants before launching training jobs, since training
parameters and output paths are configured inside those files.

## Regenerate LDPC C Tables

`py5gphy/ldpc_c_parameters.py` contains helpers for generating C table text
files under `tables/`, including BG1/BG2 HBG index arrays and shift-value
arrays.

Run it directly when you need to regenerate the table files:

```bash
python3 py5gphy/ldpc_c_parameters.py
```

## Debugging C From Python

When debugging a C function called from Python through `ctypes`, Linux may
block debugger attachment. For the current session, allow GDB attachment with:

```bash
echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
```

Then:

1. Start the Python script and pause before the C call.
2. Print or inspect the Python process ID, for example with `os.getpid()`.
3. Attach GDB to that process from another terminal.

## Cleaning Build Artifacts

Clean C build outputs:

```bash
make -C c_lib clean
```

Clean CUDA build outputs:

```bash
make -C CUDA clean
```

## Additional Notes

- The C implementation of the LDPC encoder and decoder is about 500x faster
  than the Python reference implementation in the tested configurations.

- In my experiments, deep-learning-assisted optimization of LDPC decoder
  parameters does not show a clear performance advantage over carefully tuned
  non-trained decoder configurations.

- Many papers report BER versus Eb/N0 curves, but BLER is often the more useful
  metric for practical communication-system evaluation. BER and BLER do not have
  a fixed one-to-one relationship: BLER also depends on code block length, error
  clustering, CRC detection, and decoder behavior.

## C-Lib LDPC Decoder Performance

The following BLER curves are generated by
`scripts/plot_analysis_c_lib_ldpc_performance.ipynb` from the C backend
simulation results.

![C-lib LDPC decoder BLER performance, algo 0](img/c_lib_ldpc_decoder_performance_algo0.png)

![C-lib LDPC decoder BLER performance, algo 1](img/c_lib_ldpc_decoder_performance_algo1.png)

## CUDA LDPC Decoder Performance

The following BLER curves are generated by
`scripts/plot_analysis_cuda_ldpc_performance.ipynb` from the CUDA backend
simulation results.

![CUDA LDPC decoder BLER performance, algo 0](img/cuda_ldpc_decoder_performance_algo0.png)

![CUDA LDPC decoder BLER performance, algo 1](img/cuda_ldpc_decoder_performance_algo1.png)

## License

This project is released under the MIT License. See `LICENSE` for details.
