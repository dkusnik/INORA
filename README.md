# INORA implementation

This software implements the INORA algoritm. Code is written in CUDA.

# Requirements
- CUDA SDK (optional, recommended)
- OpenMP (if CUDA not provided, recommended)
- libpng

# Installation
You can define in Makefile if you will use the CUDA framework or not, by default no. To use the CUDA framework add `CUDA=1` to the make command

Run:

```
make lib
make main
```

# Usage
Sample usage:
`./main_inora <reference image { rgb }> <noisy image {rgb}> <block_radius> <h> <iter>`
where:

Reference image - original, not noisy image

Noisy image - image which will be filtered

block_radius - radius of the processing block B

h - smoothing parameter

iter - number of iteration for limiting of fliter execution


# Acknowledgment

This code uses parts of the Fourier 0.8 library by Emre Celebi licensed on GPL avalaible here:

http://sourceforge.net/projects/fourier-ipal

http://www.lsus.edu/faculty/~ecelebi/fourier.htm

