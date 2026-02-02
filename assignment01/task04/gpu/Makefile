# Windows-friendly Makefile (use mingw32-make in PowerShell)

NVCC := nvcc

# Compile-time tunable: `mingw32-make BLOCK=16`
BLOCK ?= 16

NVCCFLAGS := -std=c++17 -O2 -g -lineinfo -DBLOCK_SIZE=$(BLOCK)
# Keep warnings-as-errors, but suppress MSVC warning C4211 from nvcc-generated stub code.
HOSTWARN  := /W4 /WX /wd4211

BUILDDIR := build
SRC := src/main.cu src/io.cpp src/ops.cu src/cuda_utils.cu src/matrix.cpp
OBJ := $(BUILDDIR)/main.obj $(BUILDDIR)/io.obj $(BUILDDIR)/ops.obj $(BUILDDIR)/cuda_utils.obj $(BUILDDIR)/matrix.obj
TARGET := matrix_add_cuda.exe

.PHONY: all clean

all: $(TARGET)

$(BUILDDIR):
	@if not exist $(BUILDDIR) mkdir $(BUILDDIR)

$(BUILDDIR)/%.obj: src/%.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -Xcompiler="$(HOSTWARN)" -c $< -o $@

$(BUILDDIR)/%.obj: src/%.cpp | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -Xcompiler="$(HOSTWARN)" -c $< -o $@

$(TARGET): $(BUILDDIR) $(OBJ)
	$(NVCC) $(NVCCFLAGS) -Xcompiler="$(HOSTWARN)" -o $@ $(OBJ)

clean:
	@if exist $(BUILDDIR) rmdir /S /Q $(BUILDDIR)
	@if exist $(TARGET) del /Q $(TARGET)
