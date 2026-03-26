#pragma once
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>
#include <expected>
#include <vector>
#include <string>

using Microsoft::WRL::ComPtr;

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

struct GpuBuffer {
    ComPtr<ID3D12Resource> m_resource;
    void* m_cpuAddress = nullptr;
    size_t m_size = 0;

    static auto Create(ID3D12Device* device, size_t size, D3D12_HEAP_TYPE type) -> std::expected<GpuBuffer, gpu_error>;
    ID3D12Resource* GetResource() const { return m_resource.Get(); }
    void* GetCpuAddress() const { return m_cpuAddress; }
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

    void ResetCommandList();
    void RecordXorShader(size_t vramOffset, size_t size, uint32_t iters);
    uint64_t ExecuteAndSignal();      // 提交計算並回傳 m_fenceValue

    void WaitForSpecificFence(uint64_t value);
    void Wait(); // 全體大等待

    void DownloadFromVram(size_t vramOffset, size_t batchSize);
    void ReleaseResources();

    // --- 記憶體存取 ---
    void* GetVramAddress() const;
    void* GetVramAddressB() const;
    void* GetReadbackAddress() const;

private:
    ComPtr<ID3D12Device> m_device;

    // Compute Engine (運算)
    ComPtr<ID3D12CommandQueue> m_computeQueue;
    ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    ComPtr<ID3D12GraphicsCommandList> m_commandList;
    ComPtr<ID3D12Fence> m_fence;
    uint64_t m_fenceValue;
    HANDLE m_fenceEvent;

    // Copy Engine (搬運 - 修正點：新增獨立 Fence)
    ComPtr<ID3D12CommandQueue> m_copyQueue;
    ComPtr<ID3D12CommandAllocator> m_copyAllocator;
    ComPtr<ID3D12GraphicsCommandList> m_copyList;
    ComPtr<ID3D12Fence> m_copyFence;      // <--- 這裡定義 m_copyFence
    uint64_t m_copyFenceValue;           // <--- 這裡定義 m_copyFenceValue

    // 資源緩衝區
    GpuBuffer m_uploadHeap, m_uploadHeapB, m_vramTemp, m_readbackHeap;
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12PipelineState> m_computePSO;

    ComPtr<ID3DBlob> CompileShader(const std::wstring& filename, const std::string& entrypoint, const std::string& target);
};