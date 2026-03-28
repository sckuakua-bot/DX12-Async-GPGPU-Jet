#pragma once
#include <expected>
#include <memory>
#include <vector>
#include <string>

enum class gpu_error {
    DeviceCreationFailed = 5546,
    QueueCreationFailed = 5547,
    FenceCreationFailed = 5548,
    EventCreationFailed = 5549,
    AllocatorCreationFailed = 5550,
    ResourceCreationFailed = 5551,
    MappingFailed = 5552,
    ListCreationFailed = 5553,
    RootSignatureCreationFailed = 5554
};

class GPU {
public:
    GPU();
    ~GPU();

    auto Init(size_t bridgeSize) -> std::expected<void, gpu_error>;

    // --- 核心執行指令 ---
    void RecordUploadOnly(size_t vramOffset, size_t uploadOffset, size_t batchSize, int heapIndex);
    uint64_t ExecuteCopyAndSignal();  // 提交搬運並回傳 m_copyFenceValue

    void ComputeWaitCopy(uint64_t copyValue); // 讓計算隊列在硬體層等搬運隊列
    void WaitForCopyFence(uint64_t copyValue); // CPU 等待搬運隊列

    void ResetCommandList();
    void RecordXorShader(size_t vramOffset, size_t size, uint32_t iters);
    uint64_t ExecuteAndSignal();      // 提交計算並回傳 m_fenceValue

    void WaitForSpecificFence(uint64_t value);
    void Wait(); // 等待

    void DownloadFromVram(size_t vramOffset, size_t batchSize);
    void ReleaseResources();

    // --- 記憶體存取 ---
    void* GetVramAddress() const;
    void* GetVramAddressB() const;
    void* GetReadbackAddress() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_pImpl;
};