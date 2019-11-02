# Laboratory Practice I
This is my own programs for Laboratory Practice I for Computer Engineering [ 2015 Batch ] . This consist of programs for openMp and cuda depending on the problem statements given by SPPU.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

What things you need to install the software and how to install them.

If you have NVIDIA Graphics card and Ubuntu 18.04 LTS follow the steps in the given below:
https://linoxide.com/linux-how-to/install-cuda-ubuntu/

If you don't have a NVIDIA Graphics card then you can use google colab to run the cuda programs online , but you get a new instance storage is not persistent so make sure to download the files before exiting colab.

Steps for compiling cuda programs : 

```
nvcc filename.cu
```
```
./a.out
```
Steps for compiling openMp programs :

```
gcc -fopenmp filename.c
```
```
./a.out
```
