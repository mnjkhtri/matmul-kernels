# Define the C compiler and common flags
CC = gcc
CFLAGS = -O3 -march=native -fPIC -shared

# Define the build and source directories
BUILD_DIR = impl/cpu_impl/build
SRC_DIR = impl/cpu_impl/src

# Define the source files
NAIVE_SRC = $(SRC_DIR)/naive.c
ORDER_SRC = $(SRC_DIR)/order.c
TILED_SRC = $(SRC_DIR)/tiled.c

# Define the output shared object files
NAIVE_SO = $(BUILD_DIR)/naive.so
ORDER_SO = $(BUILD_DIR)/order.so
TILED_SO = $(BUILD_DIR)/tiled.so

# A "phony" target that creates the build directory if it doesn't exist
.PHONY: all
all: $(NAIVE_SO) $(ORDER_SO) $(TILED_SO)

# Rule to compile naive_matmul
$(NAIVE_SO): $(NAIVE_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $< -o $@

# Rule to compile order_matmul
$(ORDER_SO): $(ORDER_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $< -o $@

# Rule to compile tiled_matmul
$(TILED_SO): $(TILED_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $< -o $@

# Clean up all compiled files
.PHONY: clean
clean:
	rm -f $(NAIVE_SO) $(ORDER_SO) $(TILED_SO)