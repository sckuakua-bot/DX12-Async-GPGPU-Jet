#include <iostream>
#include <vector>
#include <chrono>
#include <print>
#include <execution>
#include <algorithm>
#include "GPU.hpp"

inline uint32_t cpu_rotl(uint32_t x, uint32_t n) { return (x << n) | (x >> (32 - n)); }

uint32_t ReferenceCompute(uint32_t val, uint32_t iters) {
    uint32_t x = val, y = val ^ 0xDEADBEEF;
    for (uint32_t i = 0; i < iters; i++) {
        x = x + y; y = cpu_rotl(y ^ x, 7); x = cpu_rotl(x + i, 13) ^ 0x55555555;
    }
    return x ^ y;
}

int main() {
    std::print("\n{:^85}\n", "=== GPU GPGPU 深度性能分析系統 V6.7 ===");

    GPU myGPU;
    const size_t bridgeSize = 256ULL * 1024 * 1024;
    if (!myGPU.Init(bridgeSize)) return -1;

    std::print("\n[壓力曲線與數據精確度校驗]\n");
    std::print("| 迭代輪數 | CPU (ms) | GPU (ms) | 加速比 | 數據驗證 |\n");
    std::print("|----------|----------|----------|--------|----------|\n");

    for (uint32_t iters : {1, 64, 512}) {
        // 1. 準備原始數據 (0xAAAAAAAA)
        std::vector<uint32_t> initial(bridgeSize / 4, 0xAAAAAAAA);

        // 2. CPU 運算
        auto sc = std::chrono::high_resolution_clock::now();
        std::vector<uint32_t> cpuRes(bridgeSize / 4);
        std::transform(std::execution::par, initial.begin(), initial.end(), cpuRes.begin(), [iters](uint32_t v) {
            return ReferenceCompute(v, iters);
            });
        double tc = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - sc).count();

        // 3. GPU 運算
        std::memcpy(myGPU.GetVramAddress(), initial.data(), bridgeSize);
        myGPU.RecordUploadOnly(0, 0, bridgeSize, 0);
        myGPU.ExecuteCopyAndSignal(); // 確保數據先搬進去
        myGPU.Wait();

        auto sg = std::chrono::high_resolution_clock::now();
        myGPU.ResetCommandList();
        myGPU.RecordXorShader(0, bridgeSize, iters);
        myGPU.ExecuteAndSignal();
        myGPU.Wait(); // 這裡必須停下來等 GPU
        double tg = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - sg).count();

        // 4. 下載驗證
        myGPU.DownloadFromVram(0, 1024);
        myGPU.Wait(); // 確保下載也完成
        uint32_t* gpuData = (uint32_t*)myGPU.GetReadbackAddress();

        bool isMatch = (gpuData[0] == cpuRes[0]);
        std::print("| {:>8} | {:>8.1f} | {:>8.1f} | {:>5.1f}x | {:>8} |\n",
            iters, tc, tg, tc / tg, isMatch ? "MATCH" : "MISMATCH");

        if (!isMatch) {
            std::print("   [!] 錯誤: CPU=0x{:08X} vs GPU=0x{:08X}\n", cpuRes[0], gpuData[0]);
        }
    }

    myGPU.ReleaseResources();
    std::print("\n測試結束。按任意鍵離開...");
    std::cin.get();
    return 0;
}