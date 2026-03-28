#include "GPU.hpp"
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <d3d12sdklayers.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <print>
#include <cstring>
#include <wrl.h>

#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;
//GpuBuffer
struct GpuBuffer {
    ComPtr<ID3D12Resource> m_resource;
    void* m_cpuAddress = nullptr;
    size_t m_size = 0;

    static auto Create(ID3D12Device* device, size_t size, D3D12_HEAP_TYPE type) -> std::expected<GpuBuffer, gpu_error>;
    ID3D12Resource* GetResource() const { return m_resource.Get(); }
    void* GetCpuAddress() const { return m_cpuAddress; }
};
//基本變數
struct GPU::Impl {
    ComPtr<ID3D12Device> m_device;

    // Compute Engine (運算)
    ComPtr<ID3D12CommandQueue> m_computeQueue;
    ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    ComPtr<ID3D12GraphicsCommandList> m_commandList;
    ComPtr<ID3D12Fence> m_fence;   
    uint64_t m_fenceValue = 0;
    HANDLE m_fenceEvent = nullptr;

    // Copy Engine (搬運 - 修正點：新增獨立 Fence)
    ComPtr<ID3D12CommandQueue> m_copyQueue;
    ComPtr<ID3D12CommandAllocator> m_copyAllocator;
    ComPtr<ID3D12GraphicsCommandList> m_copyList;
    ComPtr<ID3D12Fence> m_copyFence;      // <--- 這裡定義 m_copyFence
    uint64_t m_copyFenceValue = 0;      // <--- 這裡定義 m_copyFenceValue
                            

    // 資源緩衝區
    GpuBuffer m_uploadHeap, m_uploadHeapB, m_vramTemp, m_readbackHeap;
    GpuBuffer m_constantBuffer;  // For shader parameters
    ComPtr<ID3D12DescriptorHeap> m_cbvHeap;  // Descriptor heap for CBV
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12PipelineState> m_computePSO;

    // --- 其餘輔助函數 ---
    ComPtr<ID3DBlob> CompileShader(const std::wstring& f, const std::string& e, const std::string& t) {
        ComPtr<ID3DBlob> b, err;
        UINT flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_IEEE_STRICTNESS;
        #ifdef _DEBUG
        flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
        #endif

        HRESULT hr = D3DCompileFromFile(f.c_str(), nullptr, nullptr, e.c_str(), t.c_str(), flags, 0, &b, &err);
        if (FAILED(hr) && err) {
            std::println("[!] HLSL 錯誤: {}", (char*)err->GetBufferPointer());
        }
        return b;
    }
};

// --- 錯誤轉譯輔助 ---
template<typename E>
auto TranslateHR(HRESULT hr, E error_code) -> std::expected<void, E> {
    if (FAILED(hr)) return std::unexpected(error_code);
    return {};
}

GPU::GPU() : m_pImpl(std::make_unique<Impl>()) // 1. 先建立內部實現的實體
{

}

GPU::~GPU() {
    // 1. 先執行你原本的資源釋放邏輯（確保 GPU 任務已完成）
    ReleaseResources();

    // 2. 存取隱藏在 Impl 裡的成員
    if (m_pImpl && m_pImpl->m_fenceEvent) {
        CloseHandle(m_pImpl->m_fenceEvent);
    }

    // 注意：你不需要手動 delete m_pImpl，
    // 因為 std::unique_ptr 會在 GPU 析構結束時自動幫你釋放記憶體。
}

// --- GpuBuffer 實作 ---
auto GpuBuffer::Create(ID3D12Device* device, size_t size, D3D12_HEAP_TYPE type) -> std::expected<GpuBuffer, gpu_error> {
    GpuBuffer buffer;

    // 強制 64KB 對齊，這是 D3D12 緩衝區的最佳實踐
    const size_t alignment = 64 * 1024;
    buffer.m_size = (size + alignment - 1) & ~(alignment - 1);

    D3D12_HEAP_PROPERTIES heapProps = { type };

    D3D12_RESOURCE_DESC resDesc = {};
    resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resDesc.Width = buffer.m_size;
    resDesc.Height = 1;
    resDesc.DepthOrArraySize = 1;
    resDesc.MipLevels = 1;
    resDesc.Format = DXGI_FORMAT_UNKNOWN;
    resDesc.SampleDesc.Count = 1;
    resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    // 確保 VRAM 支援 UAV，這是執行 Compute Shader 的關鍵
    resDesc.Flags = (type == D3D12_HEAP_TYPE_DEFAULT) ?
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS :
        D3D12_RESOURCE_FLAG_NONE;

    D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON;
    if (type == D3D12_HEAP_TYPE_UPLOAD) initialState = D3D12_RESOURCE_STATE_GENERIC_READ;
    else if (type == D3D12_HEAP_TYPE_READBACK) initialState = D3D12_RESOURCE_STATE_COPY_DEST;

    HRESULT hr = device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &resDesc,
        initialState,
        nullptr,
        IID_PPV_ARGS(&buffer.m_resource)
    );

    if (FAILED(hr)) return std::unexpected(gpu_error::ResourceCreationFailed);

    if (type == D3D12_HEAP_TYPE_UPLOAD || type == D3D12_HEAP_TYPE_READBACK) {
        D3D12_RANGE readRange = (type == D3D12_HEAP_TYPE_READBACK) ?
            D3D12_RANGE{ 0, buffer.m_size } :
            D3D12_RANGE{ 0, 0 };
        buffer.m_resource->Map(0, &readRange, &buffer.m_cpuAddress);
    }
    return buffer;
}

// --- GPU 初始化 (雙引擎版) ---
auto GPU::Init(size_t bridgeSize) -> std::expected<void, gpu_error> {
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return std::unexpected(gpu_error::DeviceCreationFailed);

    ComPtr<ID3D12Debug> debugController;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
        debugController->EnableDebugLayer();
        std::println("[*] D3D12 調試層已啟動 - 準備捕捉非法存取...");
    }

    // [Step 1] 尋找硬體適配器與建立設備
    ComPtr<IDXGIAdapter1> adapter;
    bool found = false;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_pImpl->m_device)))) {
            std::wcout << L"[*] 成功掛載硬體核心: " << desc.Description << std::endl;
            found = true; break;
        }
    }
    if (!found) return std::unexpected(gpu_error::DeviceCreationFailed);

    // [Step 2] 建立雙引擎隊列與同步物件
    return TranslateHR(S_OK, gpu_error::QueueCreationFailed)
        .and_then([&]() {
        m_pImpl->m_copyFenceValue = 0;
        return TranslateHR(m_pImpl->m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_pImpl->m_copyFence)), gpu_error::FenceCreationFailed);
            })
        .and_then([&]() {
        D3D12_COMMAND_QUEUE_DESC computeDesc = { D3D12_COMMAND_LIST_TYPE_COMPUTE, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0 };
        return TranslateHR(m_pImpl->m_device->CreateCommandQueue(&computeDesc, IID_PPV_ARGS(&m_pImpl->m_computeQueue)), gpu_error::QueueCreationFailed);
            })
        .and_then([&]() {
        D3D12_COMMAND_QUEUE_DESC copyDesc = { D3D12_COMMAND_LIST_TYPE_COPY, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0 };
        return TranslateHR(m_pImpl->m_device->CreateCommandQueue(&copyDesc, IID_PPV_ARGS(&m_pImpl->m_copyQueue)), gpu_error::QueueCreationFailed);
            })
        .and_then([&]() {
        return TranslateHR(m_pImpl->m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_pImpl->m_fence)), gpu_error::FenceCreationFailed);
            })
        .and_then([&]() -> std::expected<void, gpu_error> {
        m_pImpl->m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        return m_pImpl->m_fenceEvent ? std::expected<void, gpu_error>({}) : std::unexpected(gpu_error::EventCreationFailed);
            })
        // [Step 3] 分配器與指令列表
        .and_then([&]() {
        return TranslateHR(m_pImpl->m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_pImpl->m_commandAllocator)), gpu_error::AllocatorCreationFailed);
            })
        .and_then([&]() {
        return TranslateHR(m_pImpl->m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&m_pImpl->m_copyAllocator)), gpu_error::AllocatorCreationFailed);
            })
        .and_then([&]() {
        return TranslateHR(m_pImpl->m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_pImpl->m_commandAllocator.Get(), nullptr, IID_PPV_ARGS(&m_pImpl->m_commandList)), gpu_error::ListCreationFailed);
            })
        .and_then([&]() {
        return TranslateHR(m_pImpl->m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, m_pImpl->m_copyAllocator.Get(), nullptr, IID_PPV_ARGS(&m_pImpl->m_copyList)), gpu_error::ListCreationFailed);
            })
        .and_then([&]() -> std::expected<void, gpu_error> {
        m_pImpl->m_commandList->Close();
        m_pImpl->m_copyList->Close();
        return {};
            })
        // [Step 4] 資源緩衝區分配 (VRAM + Upload/Readback)
        .and_then([&]() -> std::expected<void, gpu_error> {
        const size_t vramSize = 512ULL * 1024 * 1024;
        return GpuBuffer::Create(m_pImpl->m_device.Get(), bridgeSize, D3D12_HEAP_TYPE_UPLOAD)
            .and_then([&](GpuBuffer&& bA) -> std::expected<GpuBuffer, gpu_error> {
            m_pImpl->m_uploadHeap = std::move(bA);
            return GpuBuffer::Create(m_pImpl->m_device.Get(), bridgeSize, D3D12_HEAP_TYPE_UPLOAD);
                })
            .and_then([&](GpuBuffer&& bB) -> std::expected<GpuBuffer, gpu_error> {
            m_pImpl->m_uploadHeapB = std::move(bB);
            return GpuBuffer::Create(m_pImpl->m_device.Get(), vramSize, D3D12_HEAP_TYPE_DEFAULT);
                })
            .and_then([&](GpuBuffer&& v) -> std::expected<GpuBuffer, gpu_error> {
            m_pImpl->m_vramTemp = std::move(v);
            return GpuBuffer::Create(m_pImpl->m_device.Get(), bridgeSize, D3D12_HEAP_TYPE_READBACK);
                })
            .and_then([&](GpuBuffer&& rb) -> std::expected<void, gpu_error> {
            m_pImpl->m_readbackHeap = std::move(rb);
            return {};
                });
            })
        // [Step 5] 建立 Descriptor Heap (解決 0xC0000005 關鍵點)
        .and_then([&]() -> std::expected<void, gpu_error> {
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.NumDescriptors = 1;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        return TranslateHR(m_pImpl->m_device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_pImpl->m_cbvHeap)), gpu_error::DeviceCreationFailed);
            })
        // [Step 6] 建立 Root Signature (3-Slot 結構)
        .and_then([&]() -> std::expected<void, gpu_error> {
        // 定義 Slot 1 要用的 Table Range
        D3D12_DESCRIPTOR_RANGE cbvRange = {};
        cbvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
        cbvRange.NumDescriptors = 1;
        cbvRange.BaseShaderRegister = 1; // register(b1)
        cbvRange.RegisterSpace = 0;
        cbvRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

        D3D12_ROOT_PARAMETER rootParams[3] = {};

        // Slot 0: Root Constants (b0) -> 存放 iters, size
        rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        rootParams[0].Constants.ShaderRegister = 0;
        rootParams[0].Constants.Num32BitValues = 4;

        // Slot 1: Descriptor Table (b1) -> 指向 cbvHeap
        rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParams[1].DescriptorTable.NumDescriptorRanges = 1;
        rootParams[1].DescriptorTable.pDescriptorRanges = &cbvRange;

        // Slot 2: Root UAV (u0) -> 直接綁定 VRAM 地址
        rootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rootParams[2].Descriptor.ShaderRegister = 0;
        rootParams[2].Descriptor.RegisterSpace = 0;

        D3D12_ROOT_SIGNATURE_DESC rsDesc = { 3, rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE };

        ComPtr<ID3DBlob> sig, err;
        if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err))) {
            if (err) std::println("[!] RootSig 序列化失敗: {}", (char*)err->GetBufferPointer());
            return std::unexpected(gpu_error::RootSignatureCreationFailed);
        }
        return TranslateHR(m_pImpl->m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&m_pImpl->m_rootSignature)), gpu_error::RootSignatureCreationFailed);
            })
        // [Step 7] 編譯 Shader 與建立 PSO
        .and_then([&]() -> std::expected<void, gpu_error> {
        auto cs = m_pImpl->CompileShader(L"SuperCompute.hlsl", "main", "cs_5_1");
        if (!cs) return std::unexpected(gpu_error::DeviceCreationFailed);

        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_pImpl->m_rootSignature.Get();
        psoDesc.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
        return TranslateHR(m_pImpl->m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_pImpl->m_computePSO)), gpu_error::DeviceCreationFailed);
            });
}

// --- 雙引擎核心操作 ---
void GPU::RecordUploadOnly(size_t vramOffset, size_t uploadOffset, size_t batchSize, int heapIndex)
{
    m_pImpl->m_copyAllocator->Reset();
    m_pImpl->m_copyList->Reset(m_pImpl->m_copyAllocator.Get(), nullptr);

    ID3D12Resource* src =
        (heapIndex == 0) ? m_pImpl->m_uploadHeap.GetResource()
        : m_pImpl->m_uploadHeapB.GetResource();

    // COMMON → COPY_DEST
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = m_pImpl->m_vramTemp.GetResource();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    m_pImpl->m_copyList->ResourceBarrier(1, &barrier);

    m_pImpl->m_copyList->CopyBufferRegion(
        m_pImpl->m_vramTemp.GetResource(),
        vramOffset,
        src,
        uploadOffset,
        batchSize
    );

    // COPY_DEST → COMMON
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;

    m_pImpl->m_copyList->ResourceBarrier(1, &barrier);
}

void GPU::RecordXorShader(size_t vramOffset, size_t size, uint32_t iters) {
    m_pImpl->m_commandList->SetComputeRootSignature(m_pImpl->m_rootSignature.Get());
    m_pImpl->m_commandList->SetPipelineState(m_pImpl->m_computePSO.Get());

    // 必須先設置 Heap，否則 Descriptor Table 會失效
    ID3D12DescriptorHeap* heaps[] = { m_pImpl->m_cbvHeap.Get() };
    m_pImpl->m_commandList->SetDescriptorHeaps(1, heaps);

    // [Slot 0] Constants: 確保 size 傳遞正確
    uint32_t params[4] = { static_cast<uint32_t>(size), iters, 0, 0 };
    m_pImpl->m_commandList->SetComputeRoot32BitConstants(0, 4, params, 0);

    // [Slot 1] Table: 雖然這套測試主要用 UAV，但 Slot 1 必須綁定以符合 RootSig
    m_pImpl->m_commandList->SetComputeRootDescriptorTable(1, m_pImpl->m_cbvHeap->GetGPUDescriptorHandleForHeapStart());

    // [Slot 2] UAV: 直接綁定運算目標位址
    D3D12_GPU_VIRTUAL_ADDRESS gpuAddr = m_pImpl->m_vramTemp.GetResource()->GetGPUVirtualAddress() + vramOffset;
    m_pImpl->m_commandList->SetComputeRootUnorderedAccessView(2, gpuAddr);

    // 狀態轉換：COMMON -> UAV
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = m_pImpl->m_vramTemp.GetResource();
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pImpl->m_commandList->ResourceBarrier(1, &b);

    uint32_t elementCount = static_cast<uint32_t>(size >> 2);
    m_pImpl->m_commandList->Dispatch((elementCount + 255) / 256, 1, 1);

    // 狀態轉換：UAV -> COMMON (這一步很重要，否則後續 Copy Engine 讀不到最新數據)
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    m_pImpl->m_commandList->ResourceBarrier(1, &b);
}

// --- 修正：對外窗口 1 (搬運引擎) ---
uint64_t GPU::ExecuteCopyAndSignal() {
    // 實際上是呼叫隱藏在 m_pImpl 裡面的邏輯
    m_pImpl->m_copyList->Close();

    ID3D12CommandList* lists[] = { m_pImpl->m_copyList.Get() };
    m_pImpl->m_copyQueue->ExecuteCommandLists(1, lists);

    m_pImpl->m_copyFenceValue++;
    m_pImpl->m_copyQueue->Signal(m_pImpl->m_copyFence.Get(), m_pImpl->m_copyFenceValue);

    return m_pImpl->m_copyFenceValue;
}

// --- 修正：對外窗口 2 (運算引擎) ---
uint64_t GPU::ExecuteAndSignal() {
    m_pImpl->m_commandList->Close();

    ID3D12CommandList* lists[] = { m_pImpl->m_commandList.Get() };
    m_pImpl->m_computeQueue->ExecuteCommandLists(1, lists);

    m_pImpl->m_fenceValue++;
    m_pImpl->m_computeQueue->Signal(m_pImpl->m_fence.Get(), m_pImpl->m_fenceValue);

    return m_pImpl->m_fenceValue;
}

void GPU::Wait() {
    WaitForSpecificFence(ExecuteAndSignal());
}

void GPU::WaitForSpecificFence(uint64_t value) {
    if (m_pImpl->m_fence->GetCompletedValue() < value) {
        m_pImpl->m_fence->SetEventOnCompletion(value, m_pImpl->m_fenceEvent);
        WaitForSingleObject(m_pImpl->m_fenceEvent, INFINITE);
    }
}

void GPU::ResetCommandList() {
    m_pImpl->m_commandAllocator->Reset();
    m_pImpl->m_commandList->Reset(m_pImpl->m_commandAllocator.Get(), nullptr);
}

void GPU::DownloadFromVram(size_t offset, size_t size) {
    // 1. 確保之前的 Compute 任務完全結束
    Wait();

    // 2. 重置指令表進行搬運
    m_pImpl->m_commandAllocator->Reset();
    m_pImpl->m_commandList->Reset(m_pImpl->m_commandAllocator.Get(), nullptr);

    // 狀態轉換：COMMON -> COPY_SOURCE
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = m_pImpl->m_vramTemp.GetResource();
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_pImpl->m_commandList->ResourceBarrier(1, &b);

    // 關鍵：從 VRAM 的 offset 位置搬運到 ReadbackHeap 的開頭 (0)
    m_pImpl->m_commandList->CopyBufferRegion(
        m_pImpl->m_readbackHeap.GetResource(), 0,
        m_pImpl->m_vramTemp.GetResource(), offset,
        size
    );

    // 狀態還原
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    m_pImpl->m_commandList->ResourceBarrier(1, &b);

    // 3. 提交並強制等待完成
    ExecuteAndSignal();
    Wait();

    // 4. 刷新 CPU 可見記憶體
    D3D12_RANGE readRange = { 0, size };
    void* pMappedData = nullptr;
    // 先 Unmap 再 Map 能強制讓驅動程式刷新 Readback 緩衝區
    m_pImpl->m_readbackHeap.GetResource()->Unmap(0, nullptr);
    m_pImpl->m_readbackHeap.GetResource()->Map(0, &readRange, &pMappedData);
    m_pImpl->m_readbackHeap.m_cpuAddress = pMappedData;
}

void GPU::ReleaseResources() {
    if (m_pImpl->m_fence) { // 確保 device 還在時才 Wait
        Wait();
    }
    if (m_pImpl->m_fenceEvent) {
        CloseHandle(m_pImpl->m_fenceEvent);
        m_pImpl->m_fenceEvent = nullptr;
    }
    m_pImpl->m_vramTemp = GpuBuffer(); m_pImpl->m_uploadHeap = GpuBuffer(); m_pImpl->m_uploadHeapB = GpuBuffer(); m_pImpl->m_readbackHeap = GpuBuffer();
    std::cout << "[System] 雙引擎關閉，資源已安全回收。" << std::endl;
}

void GPU::ComputeWaitCopy(uint64_t val) {
    m_pImpl->m_computeQueue->Wait(m_pImpl->m_copyFence.Get(), val);
}

void GPU::WaitForCopyFence(uint64_t val) {
    // 1. 檢查是否已經完成，若完成則直接返回，避免建立不必要的 Event
    if (m_pImpl->m_copyFence->GetCompletedValue() >= val) return;

    // 2. 建立 Event 並檢查有效性
    HANDLE copyEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (copyEvent == nullptr) {
        // 這裡可以拋出異常或記錄錯誤
        std::println("[!] 建立同步事件失敗！錯誤碼: {}", GetLastError());
        return;
    }

    // 3. 設定 Fence 觸發
    if (SUCCEEDED(m_pImpl->m_copyFence->SetEventOnCompletion(val, copyEvent))) {
        WaitForSingleObject(copyEvent, INFINITE);
    }

    // 4. 安全關閉
    CloseHandle(copyEvent);
}

void* GPU::GetVramAddress() const { return m_pImpl->m_uploadHeap.GetCpuAddress(); }
void* GPU::GetVramAddressB() const { return m_pImpl->m_uploadHeapB.GetCpuAddress(); }
void* GPU::GetReadbackAddress() const { return m_pImpl->m_readbackHeap.GetCpuAddress(); }