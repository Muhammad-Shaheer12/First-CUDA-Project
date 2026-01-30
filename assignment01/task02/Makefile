CXX := g++
CXXFLAGS := -std=c++17 -O2 -g -Wall -Wextra -Wpedantic -Werror -Wconversion -Wsign-conversion

SRC := src/main.cpp src/matrix.cpp src/io.cpp src/ops.cpp
BUILDDIR := build
OBJ := $(patsubst src/%.cpp,$(BUILDDIR)/%.o,$(SRC))
TARGET := matrix_add.exe

.PHONY: all clean

all: $(TARGET)


$(BUILDDIR):
	@if not exist $(BUILDDIR) mkdir $(BUILDDIR)

$(TARGET): $(BUILDDIR) $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ)

$(BUILDDIR)/%.o: src/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@if exist $(BUILDDIR)\\*.o del /Q $(BUILDDIR)\\*.o
	@if exist $(BUILDDIR) rmdir /S /Q $(BUILDDIR)
	@if exist $(TARGET) del /Q $(TARGET)
