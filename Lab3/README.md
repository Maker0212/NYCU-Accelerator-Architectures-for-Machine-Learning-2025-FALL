# AAML2025 Lab3


[Lab introduction of Lab3](https://nycu-caslab.github.io/AAML2025/labs/lab_3.html)


## Directory Structure
```bash
.
├── data_generator.py
├── Makefile
├── Makefile_ncverilog
├── Makefile_vcs
├── README.md
├── RTL
│   ├── global_buffer.v
│   └── TPU.v
└── TESTBENCH
    ├── PATTERN.v
    └── TESTBENCH.v
```

- `RTL`: The source code of your design
- `TESTBENCH`: The testbench to test your design.
- `data_generator.py`: The generator to generate the test case.
- `dump.(vcd|fsdb)`: the waveform after running any test.

## Makefile
- `make verif1`
    - Run the code with #1 test case.
- `make verif2`
    - Run the code with #2 test case.
- `make verif3`
    - Run the code with #3 test case.
- `make verif4`
    - RUn the code with #4 test case.


