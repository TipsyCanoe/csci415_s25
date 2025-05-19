# Compiler and flags
NVCC = nvcc
CFLAGS = -O3
TARGET = jacobi

# Files
SRC = jacobi.cu
TEST_SMALL = test_small.txt
TEST_LARGE = test_large.txt

# Default rule
all: $(TARGET)

# Compile CUDA code
$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

# Clean up binary and object files
clean:
	rm -f $(TARGET) *.dat *.png $(TEST_SMALL) $(TEST_LARGE)

# Generate test files
test-files: $(TEST_SMALL) $(TEST_LARGE)

$(TEST_SMALL):
	@echo "Creating small test file (10x10)..."
	@echo "10 0.01" > $(TEST_SMALL)
	@echo "0.001 0 0 8 0" >> $(TEST_SMALL)
	@echo "0.001 0 9 8 9" >> $(TEST_SMALL)
	@echo "0.001 0 1 0 8" >> $(TEST_SMALL)
	@echo "1.0 9 0 9 9" >> $(TEST_SMALL)

$(TEST_LARGE):
	@echo "Creating large test file (1024x1024)..."
	@echo "1024 0.0001" > $(TEST_LARGE)
	@echo "0.0 0 0 1023 0" >> $(TEST_LARGE)
	@echo "0.0 1023 0 1023 1023" >> $(TEST_LARGE)
	@echo "0.0 0 1 0 1022" >> $(TEST_LARGE)
	@echo "0.0 1022 1 1022 1022" >> $(TEST_LARGE)
	@echo "1.5 245 245 255 255" >> $(TEST_LARGE)

# Run timing tests
timing-tests: $(TARGET) $(TEST_SMALL) $(TEST_LARGE)
	@echo "=== Small Grid Timing Tests ==="
	@for i in 1 5 10 50 100; do \
		echo "\n--- Using -i $$i ---"; \
		./$(TARGET) -T -i $$i $(TEST_SMALL); \
	done
	
	@echo "\n=== Large Grid Timing Tests ==="
	@for i in 1 5 10 50 100; do \
		echo "\n--- Using -i $$i ---"; \
		./$(TARGET) -T -i $$i $(TEST_LARGE); \
	done


# Generate visualizations
visualize: 
	@echo "Generating heat map for small test..."
	gnuplot -e "set terminal png size 800,600; set output '$(TEST_SMALL)_heat.png'; \
		set view map; set title 'Jacobi Results (10x10)'; \
		splot '$(TEST_SMALL).dat' using 1:2:3 with pm3d"
		
	@echo "Generating heat map for large test..."
	gnuplot -e "set terminal png size 1200,1000; set output '$(TEST_LARGE)_heat.png'; \
		set view map; set title 'Jacobi Results (1024x1024)'; \
		splot '$(TEST_LARGE).dat' using 1:2:3 with pm3d"
		
	@echo "Generating 3D plot for assignment submission..."
	gnuplot -e "set terminal png size 1200,1000; set output '$(TEST_LARGE)_3d.png'; \
		set title 'Jacobi 3D Results (1024x1024)'; \
		splot '$(TEST_LARGE).dat' using 1:2:3 with pm3d"

# Run everything needed for the assignment
assignment: $(TARGET) test-files timing-tests visualize
	@echo "\n=== Assignment tasks completed ==="
	@echo "1. Code compiled"
	@echo "2. Test files created"
	@echo "3. Timing tests completed"
	@echo "4. Visualizations generated"

