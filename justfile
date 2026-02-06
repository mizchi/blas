# MoonBit BLAS Library Commands

# Default target (native only)
target := "native"

# Default task: check and test
default: check test

# Format code
fmt:
    moon fmt

# Type check
check:
    moon check --target {{target}}

# Run tests
test:
    moon test --target {{target}}

# Update snapshot tests
test-update:
    moon test --update --target {{target}}

# Run benchmark
bench:
    moon run src/bench --target {{target}}

# Run benchmark with custom parameters
bench-large:
    moon run src/bench --target {{target}} -- --batch 256 --iters 500

# Generate type definition files
info:
    moon info

# Clean build artifacts
clean:
    moon clean

# Pre-release check
release-check: fmt info check test

# Docker build for Linux testing
docker-build:
    docker build -t mizchi/blas .

# Docker test
docker-test:
    docker run --rm mizchi/blas

# Docker benchmark
docker-bench:
    docker run --rm mizchi/blas sh -c "moon run src/bench --target native"
