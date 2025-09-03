
# native-client-min

A minimal native client with exactly two responsibilities:

1) **Binary exec from server**: connect to the central server via WebSocket (`/ws-native`) and execute self-contained binaries (workload-level or chunk-level) via `fork/execve` with IPC (stdin/stdout).  
2) **Native source from browser**: run a **local WebSocket server** (127.0.0.1:8787) that accepts JSON jobs from the browser to compile+run native GPU code (CUDA via NVRTC implemented; OpenCL/Vulkan stubs).

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
cmake --build build -j
```

## Run

### Connect to server for binary execution
```bash
./build/native_client --mode server --url wss://localhost:3000 --insecure
```

### Run local WebSocket for browser→native CUDA
```bash
./build/native_client --mode local
# Browser posts to ws://127.0.0.1:8787/native
```

## Protocols

### Server binary (inbound)
- `workload:new` or `workload:chunk_assign`
  - fields: `id` or `{parentId,chunkId}`, `executable` (path or base64 blob), `args` (array), `stdin` (base64), `env` (object)
- Replies:
  - `workload:done` or `workload:chunk_done_enhanced` with base64 outputs and processing time

### Local browser→native (inbound)
- JSON:
```json
{
  "action": "compile_and_run",
  "framework": "cuda", 
  "source": "extern "C" __global__ void ...",
  "entry": "kernel_name",
  "grid": [gx,gy,gz],
  "block": [bx,by,bz],
  "uniforms": [ ...numbers... ],
  "inputs": [{"data": "<base64>", "size": N}, ...],
  "outputSizes": [bytes0, bytes1, ...]
}
```
- Reply:
```json
{"ok": true, "outputs": ["<base64>", ...], "processingTimeMs": 12.3}
```


## Optional backends

- **CUDA (NVRTC)**: `-DENABLE_CUDA=ON` (requires CUDA driver + nvrtc)
- **OpenCL**: `-DENABLE_OPENCL=ON` (requires OpenCL headers + libOpenCL)
- **Vulkan (shaderc)**: `-DENABLE_VULKAN=ON` (requires Vulkan SDK + shaderc)

Example:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DENABLE_OPENCL=ON -DENABLE_VULKAN=ON
cmake --build build -j
```

### Local WS requests
- CUDA:
  - `framework: "cuda"`, `source` (CUDA C), `entry`, `grid`, `block`, `uniforms`, `inputs`, `outputSizes`
- OpenCL:
  - `framework: "opencl"`, `source` (OpenCL C), `entry`, `global`, `local`, `uniforms`, `inputs`, `outputSizes`
- Vulkan:
  - `framework: "vulkan"`, either `source_glsl` (GLSL compute, needs shaderc) **or** `spirv` (base64 SPIR-V),
  - `uniforms`, `inputs`, `outputSizes`, and `groups` (workgroup dispatch counts).

Binding order for all three: **uniforms (by value/UBO) → inputs (storage) → outputs (storage)**.
