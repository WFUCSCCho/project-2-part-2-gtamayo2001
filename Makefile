# Compiler
NVCC = nvcc

# Compiler flags
# -O3 for optimization
NVCC_FLAGS = -O3

# Target executables
TARGETS = blur blur_orig

# Default target: build all programs
all: $(TARGETS)

# Individual build rules
blur: blur.cu
	$(NVCC) $(NVCC_FLAGS) blur.cu -o blur

blur_orig: blur_orig.cu
	$(NVCC) $(NVCC_FLAGS) blur_orig.cu -o blur_orig

# Clean up generated files
clean:
	rm -f $(TARGETS)
