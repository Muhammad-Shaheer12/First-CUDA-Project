CXX := g++
CXXFLAGS := -std=c++17 -O2 -g -Wall -Wextra -Wpedantic -Werror -Wconversion -Wsign-conversion

BUILDDIR := build
SRC := src/main.cpp src/matrix.cpp src/io.cpp src/ops.cpp
OBJ := $(patsubst src/%.cpp,$(BUILDDIR)/%.o,$(SRC))
TARGET := matrix_add_cpu.exe

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
