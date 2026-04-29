# introduction

This project is to generate LDPC encoder and decoder using Python and C

# file info
- GEO_BP_minsum_LDPC_decoder.py
  - BP and min-sum ldpc decoder using py5gphy toolbox
- GEO_layered_min_sum_ldpc_decoder.py
  - layered min-sum ldpc decoder
- GEO_LDPC_encoder.py
  - LDPC encoder,includes CRC encoder, LDPC encoder, rate matching
- LDPC_parameter.py
  - generate ldpc parameters, such as Zc, E, filling bit info
- plot_ldpc_decoder_performance.ipynb
  - plot ldpc decoder performance curve 
- plot_result.py
  - define functions used to plot result
- sim_GEObackhaul_LDPC.py
  - run LDPC simulation, result write to out folder
# how to setup environment

* create python .venv

# how to use it

# install packages
``` shell
pip install --upgrade pip
pip install numpy
pip install scipy
pip install pandas
pip install seaborn
pip install jupyter nbconvert ipykernel
```

# some cmds

* write all packages into requirements.txt
``` shell
pip freeze > requirements.txt
```

* load from requirement.txt

``` shell
pip install --upgrade pip
pip install -r requirements.txt
```

# run ipynb from terminal

1. convert ipynb to py
``` shell
jupyter nbconvert --to python my_notebook.ipynb
```

2. run it
``` shell
python my_notebook.py
```