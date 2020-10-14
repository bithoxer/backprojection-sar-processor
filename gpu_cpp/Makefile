#********************************************************************************************************************
#
# Makefile for Backprojection SAR Processor implemented in CUDA
#
#********************************************************************************************************************

#====================================================================================================================
# DEFINITIONS
#====================================================================================================================

# CUDA Directory
CUDA_ROOT_DIR = C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.25.28610

# CUDA Directories
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib/x64 # CUDA library directory
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include # CUDA include directory
CUDA_cuFFT_LIB = -lcufft # cuFFT library
CUDA_LINK_LIBS= -lcudart # CUDA linking libraries

# LIBS = -lm

# Project File Structure
PROJ_INC_DIR = ./include
PROJ_OBJ_DIR = ./obj
PROJ_SRC_DIR = ./src
PROJ_LIB_DIR = ./lib

# Compiler Options
# CC = g++
NVCC = nvcc # nvcc compiler. Uses cl.exe to compile host code.

# Compiler Flags
# CFLAGS = -I$(PROJ_INC_DIR)
NFLAGS = -I$(PROJ_INC_DIR)  # for adding include directory as search path when compiling

# C++ Header Files
# _DEPS_H = 
# DEPS_H = $(patsubst %,$(IDIR)/%,$(_DEPS_H))

# CUDA Header Files (to be used to list all header/dependancy files for compilation)
CUDA_HEADER_FILES = gpu_backprojection.cuh
# patsubst finds whitespace-separated words in TEXT that match PATTERN and replaces them with REPLACEMENT.
# Here PATTERN may contain a % which acts as a wildcard, matching any number of any characters within a word.
CUDA_DEPS = $(patsubst %,$(PROJ_INC_DIR)/%,$(CUDA_HEADER_FILES))

# Object Files (to be used to list all the object files to link)
OBJ_FILES = test.o # gpu_backprojection.o
OBJ = $(patsubst %,$(PROJ_OBJ_DIR)/%,$(OBJ_FILES))

#====================================================================================================================
# COMPILATION
#====================================================================================================================

# Compile C++ files
# $(ODIR)/%.o: $(SDIR)/%.cpp $(DEPS_H)
	# $(CC) -c -o $@ $< $(CFLAGS)

# Compile CUDA files
# go through ./src, find everything with .cu and compile it to .obj stored in ./obj
# $(NFLAGS) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) are all that's needed here are search paths not libraries. Libraries specified in linking stage.	
# all: library.cpp main.cpp
# $@ evaluates to all
# $< evaluates to library.cpp. Used here because compiling each .cu file one by one
# $^ evaluates to library.cpp main.cpp
# -c means compile
# -o means output
$(PROJ_OBJ_DIR)/%.o: $(PROJ_SRC_DIR)/%.cu $(CUDA_DEPS)
	$(NVCC) -c -o $@ $< $(NFLAGS) $(CUDA_INC_DIR) $(CUDA_LIB_DIR)

#====================================================================================================================
# LINKING
#====================================================================================================================

# Link object files into one directory
main: $(OBJ)
	$(NVCC) -o $@ $^ $(NFLAGS) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $(CUDA_cuFFT_LIB)
	make clean


#====================================================================================================================
# OTHER ROUTINES
#====================================================================================================================

run:
	.\main

.PHONY: clean

clean:
	rm -f $(PROJ_OBJ_DIR)/*.o *~ core $(INCDIR)/*~
