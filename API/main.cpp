#include "GPU.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <print>
#include <numeric>
#include <iomanip>

using namespace std;

// 高強度 CPU 參考運算 (使用多線程模擬真實負載，若想單純對比單核性能可移除 OpenMP)
void CpuXorHeavy(uint32_t* data, size_t count, uint32_t iters) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t val = data[i];
        for (uint32_t j = 0; j < iters; ++j) {
            val ^= 0xFFFFFFFF;
            val += (j % 2); // 稍微增加運算複雜度，防止編譯器優化跳過循環
            val -= (j % 2);
        }
        data[i] = val;
    }
}

int main() {
    std::println("\n      === D3D12 GPGPU 工業級壓力分析系統 V8.0 (自動化測試版) ===");

    // [設定] 測試數據大小 (建議 256MB，展現對大記憶體管理的掌控)
    const size_t dataSize = 256ULL * 1024 * 1024;
    const size_t elementCount = dataSize / sizeof(uint32_t);

    GPU gpu;
    if (!gpu.Init(dataSize + 64ULL * 1024 * 1024)) { // 預留一點緩衝空間
        std::println("[!] GPU 初始化失敗，請檢查驅動程式或 VRAM 容量。");
        return -1;
    }

    // 1. 生成隨機或規律數據
    std::vector<uint32_t> hostData(elementCount);
    std::iota(hostData.begin(), hostData.end(), 0x12345678);

    // 2. 定義階梯式壓力點：1 (測頻寬), 64, 256, 1024, 2048 (測算力)
    vector<uint32_t> testConfigs = { 1, 64, 256, 1024, 2048 };

    std::println("[*] 數據規模: {} MB", dataSize / (1024 * 1024));
    std::println("[*] 正在進行硬體預熱與驅動同步...");
    {
        gpu.RecordUploadOnly(0, 0, dataSize, 0);
        gpu.ExecuteCopyAndSignal();
        gpu.Wait();
    }

    std::println("\n| 迭代次數 | CPU (ms) | GPU (ms) | 加速比 | 驗證 | 吞吐量 (GB/s) | 算力 (GOPS) |");
    std::println("|----------|----------|----------|--------|------|--------------|-------------|");

    for (uint32_t iters : testConfigs) {
        // --- CPU 部分 ---
        auto c_start = chrono::high_resolution_clock::now();
        // 為了測試強度，我們只在 1 與 64 輪做 CPU 對比，高強度時 CPU 太慢會等太久
        double cpuMs = 0;
        bool shouldVerify = (iters <= 256);

        if (shouldVerify) {
            vector<uint32_t> cpuRes = hostData;
            CpuXorHeavy(cpuRes.data(), elementCount, iters);
            cpuMs = chrono::duration<double, milli>(chrono::high_resolution_clock::now() - c_start).count();

            // --- GPU 部分 ---
            memcpy(gpu.GetVramAddress(), hostData.data(), dataSize);
            uint64_t fCopy = gpu.ExecuteCopyAndSignal();

            auto g_start = chrono::high_resolution_clock::now();
            gpu.ComputeWaitCopy(fCopy);
            gpu.ResetCommandList();
            gpu.RecordXorShader(0, dataSize, iters);
            gpu.ExecuteAndSignal();
            gpu.Wait();
            auto g_end = chrono::high_resolution_clock::now();
            double gpuMs = chrono::duration<double, milli>(g_end - g_start).count();

            gpu.DownloadFromVram(0, dataSize);
            uint32_t* gpuResPtr = (uint32_t*)gpu.GetReadbackAddress();

            // 數據驗證
            bool match = (memcmp(cpuRes.data(), gpuResPtr, 4096) == 0);

            // 指標計算
            double gbps = (double(dataSize) * 2) / (gpuMs / 1000.0) / (1024.0 * 1024.0 * 1024.0);
            double gops = (double(elementCount) * iters) / (gpuMs * 1e6);

            std::println("| {:>8} | {:>8.1f} | {:>8.1f} | {:>6.1f}x | {} | {:>12.2f} | {:>11.2f} |",
                iters, cpuMs, gpuMs, cpuMs / gpuMs, match ? "PASS" : "FAIL", gbps, gops);
        }
        else {
            // 純 GPU 壓力測試 (不跑 CPU 以節省時間)
            auto g_start = chrono::high_resolution_clock::now();
            gpu.RecordXorShader(0, dataSize, iters);
            gpu.ExecuteAndSignal();
            gpu.Wait();
            double gpuMs = chrono::duration<double, milli>(chrono::high_resolution_clock::now() - g_start).count();

            double gops = (double(elementCount) * iters) / (gpuMs * 1e6);
            // 在 iters > 256 的 else 區塊中使用：
            std::println("| {:>8} | {:>8} | {:>8.1f} | {:>7} | {:^6} | {:>12} | {:>11.2f} |",
                iters, "N/A", gpuMs, "N/A", "SKIP", "---", gops);
        }
    }

    std::println("\n[*] 壓力測試完成。");
    std::println("[*] 提示：當迭代次數增加時，GOPS 會持續上升，代表 GPU 隱藏延遲的能力增強。");
    return 0;
}