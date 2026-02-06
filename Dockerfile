# Linux build environment with OpenBLAS
FROM ubuntu:24.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install MoonBit
RUN curl -fsSL https://cli.moonbitlang.com/install/unix.sh | bash
ENV PATH="/root/.moon/bin:${PATH}"

WORKDIR /app

# Copy project files
COPY . .

# Patch moon.pkg for Linux (replace Accelerate with OpenBLAS)
RUN sed -i 's/-framework Accelerate/-lopenblas -lm/' src/moon.pkg

# Build and test
CMD ["sh", "-c", "moon check --target native && moon test --target native"]
