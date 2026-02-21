NVCC := nvcc
BLOCK ?= 16
NVCCFLAGS := -std=c++17 -O2 -g -lineinfo -DBLOCK_SIZE=$(BLOCK)
HOSTWARN  := /W4 /WX /wd4211

BUILDDIR := build
OBJ := $(BUILDDIR)/main.obj $(BUILDDIR)/io.obj $(BUILDDIR)/ops.obj $(BUILDDIR)/cuda_utils.obj $(BUILDDIR)/matrix.obj
TARGET := matmul_cuda.exe

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
