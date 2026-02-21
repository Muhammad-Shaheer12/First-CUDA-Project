CXX := g++
CXXFLAGS := -std=c++17 -O2 -g -Wall -Wextra -Wpedantic -Werror -Wconversion -Wsign-conversion

BUILDDIR := build
OBJ := $(BUILDDIR)/main.o $(BUILDDIR)/io.o $(BUILDDIR)/ops.o $(BUILDDIR)/matrix.o
TARGET := matmul_cpu.exe

.PHONY: all clean

all: $(TARGET)

$(BUILDDIR):
	@if not exist $(BUILDDIR) mkdir $(BUILDDIR)

$(BUILDDIR)/%.o: src/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(BUILDDIR) $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ)

clean:
	@if exist $(BUILDDIR) rmdir /S /Q $(BUILDDIR)
	@if exist $(TARGET) del /Q $(TARGET)
