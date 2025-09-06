# docker/client/Dockerfile
# Multi-stage build: build with vcpkg + CMake, then ship a slim runtime
ARG UBUNTU_VERSION=22.04
FROM ubuntu:${UBUNTU_VERSION} AS build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl unzip zip tar pkg-config ninja-build ca-certificates \
    python3 \
    libvulkan1 mesa-vulkan-drivers \
    ocl-icd-opencl-dev \
    pocl-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

# Install vcpkg
ENV VCPKG_ROOT=/opt/vcpkg
RUN git clone --depth=1 https://github.com/microsoft/vcpkg.git ${VCPKG_ROOT} \
 && ${VCPKG_ROOT}/bootstrap-vcpkg.sh -disableMetrics

# FIXED: Define VCPKG_TRIPLET as ARG and ENV
ARG VCPKG_TRIPLET=x64-linux
ENV VCPKG_TRIPLET=${VCPKG_TRIPLET}

# Install only what we need – rely on system libvulkan loader
RUN ${VCPKG_ROOT}/vcpkg install --triplet=${VCPKG_TRIPLET} \
    boost-system \
    boost-url \
    boost-beast \
    boost-asio \
    openssl \
    nlohmann-json \
    shaderc \
    vulkan-headers

# Build client
WORKDIR /src
COPY . /src

RUN find /opt/vcpkg/installed/x64-linux/share -maxdepth 3 -type f -iname "*shaderc*" -print

RUN ls -la ${VCPKG_ROOT}/installed/${VCPKG_TRIPLET}/lib | grep -i shaderc || true

ARG BUILD_TYPE=Debug
RUN cmake -S . -B build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake \
    -DVCPKG_TARGET_TRIPLET=${VCPKG_TRIPLET} \
    -DVulkan_INCLUDE_DIR=${VCPKG_ROOT}/installed/${VCPKG_TRIPLET}/include \
    -DVulkan_LIBRARY=/lib/x86_64-linux-gnu/libvulkan.so.1 \
    -DOPENCL_INCLUDE_DIR=/usr/include \
    -DOPENCL_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libOpenCL.so \
    -DENABLE_OPENCL=ON \
    -DENABLE_VULKAN=ON \
    -DENABLE_CUDA=OFF \
 && cmake --build build --parallel

# Runtime stage
FROM ubuntu:${UBUNTU_VERSION} AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libstdc++6 \
    libvulkan1 mesa-vulkan-drivers \
    ocl-icd-libopencl1 pocl-opencl-icd clinfo \
    python3 \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

# FIXED: Set VCPKG_TRIPLET in runtime stage
ARG VCPKG_TRIPLET=x64-linux
ENV VCPKG_TRIPLET=${VCPKG_TRIPLET}

# Copy built application
COPY --from=build /src/build/native_client /opt/mfc/bin/native_client
COPY --from=build /opt/vcpkg/installed /opt/vcpkg/installed

# Copy certificate files from build stage if they exist
COPY --from=build /src /tmp/src-files
RUN if [ -f /tmp/src-files/server.crt ]; then cp /tmp/src-files/server.crt /opt/mfc/bin/; chmod 644 /opt/mfc/bin/server.crt; fi && \
    if [ -f /tmp/src-files/server.key ]; then cp /tmp/src-files/server.key /opt/mfc/bin/; chmod 600 /opt/mfc/bin/server.key; fi && \
    if [ -f /tmp/src-files/generate_cert.sh ]; then cp /tmp/src-files/generate_cert.sh /opt/mfc/bin/; chmod +x /opt/mfc/bin/generate_cert.sh; fi && \
    rm -rf /tmp/src-files

# Set environment variables for runtime
ENV ROOTSYS=/opt/root
ENV PATH=$ROOTSYS/bin:$PATH

# FIXED: Set LD_LIBRARY_PATH properly with fallback and ROOT libraries
ENV LD_LIBRARY_PATH=/opt/vcpkg/installed/${VCPKG_TRIPLET}/lib:/opt/vcpkg/installed/${VCPKG_TRIPLET}/lib/manual-link:/opt/root/lib:/usr/local/lib:/usr/lib

WORKDIR /opt/mfc/bin

# Set the entrypoint to the native client binary
ENTRYPOINT ["./native_client"]

# Example environment variables (uncomment as needed)
#ENV FRAMEWORK=vulkan
#ENV CLIENT_TYPE=vulkan
#ENV SERVER_URL=https://localhost:3000
