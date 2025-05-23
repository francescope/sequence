# sequence
Multicore Systems Programming project


EduHPC 2024: Peachy assignment

(c) 2023-2024 Arturo Gonzalez-Escribano, Diego García-Álvarez, Jesús Cámara 
Group Trasgo, Grupo GAMUVa, Universidad de Valladolid (Spain)

--------------------------------------------------------------

Read the handout and use the sequential code as reference to study.
Use the other source files to parallelize with the proper programming model.

Edit the first lines in the Makefile to set your preferred compilers and flags
for both the sequential code and for each parallel programming model: 
OpenMP, MPI, and CUDA.

To see a description of the Makefile options execute:
$ make help 

Use the following program arguments for your first tests.
Students are encouraged to generate their own program arguments for more 
complete tests. See a description of the program arguments in the handout.


Example tests
==============

1) Basic test:
--------------
300 0.1 0.3 0.35 100 5 5 300 150 50 150 80 M 609823


2) Simple tests for race conditions:
------------------------------------
1000 0.35 0.2 0.25 0 0 0 20000 10 0 500 0 M 4353435

10000 0.35 0.2 0.25 0 0 0 10000 9000 9000 50 100 M 4353435


3) Check that the program works for sequences longest than INT_MAX:
-------------------------------------------------------------------
4294967300 0.35 0.2 0.25 0 0 0 1 1 0 4294967298 0 M 683224

