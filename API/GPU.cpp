#include "GPU.hpp"
#include <vector>
#include <dxgi1_6.h>
#include <iostream>
#include <d3dcompiler.h>
#include <algorithm>
#include <d3d12sdklayers.h>
#include <print>

#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

// --- 錯誤轉譯輔助 ---
template<typename E>
auto TranslateHR(HRESULT hr, E error_code) -> std::expected<void, E> {
    if (FAILED(hr)) return std::unexpected(error_code);
    return {};
}

GPU::GPU()
    : m_fenceEvent(nullptr)
    , m_fenceValue(0)
    , m_copyFenceValue(0)  // 修正：顯式初始化
{
    // 構造函數主體
}

GPU::~GPU() {
    if (m_fenceEvent) CloseHandle(m_fenceEvent);
}

// --- GpuBuffer 實作 ---
auto GpuBuffer::Create(ID3D12Device* device, size_t size, D3D12_HEAP_TYPE type) -> std::expected<GpuBuffer, gpu_error> {
    GpuBuffer buffer;
    buffer.m_size = size;

    D3D12_HEAP_PROPERTIES heapProps = { type };
    D3D12_RESOURCE_DESC resDesc = {};
    resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resDesc.Width = size;
    resDesc.Height = 1;
    resDesc.DepthOrArraySize = 1;
    resDesc.MipLevels = 1;
    resDesc.Format = DXGI_FORMAT_UNKNOWN;
    resDesc.SampleDesc.Count = 1;
    resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resDesc.Flags = (type == D3D12_HEAP_TYPE_DEFAULT) ?
        (D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) :
        D3D12_RESOURCE_FLAG_NONE;

    // 確保 Default Heap 初始狀態絕對是 COMMON
    D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON;
    if (type == D3D12_HEAP_TYPE_UPLOAD) initialState = D3D12_RESOURCE_STATE_GENERIC_READ;
    else if (type == D3D12_HEAP_TYPE_READBACK) initialState = D3D12_RESOURCE_STATE_COPY_DEST;

    if (FAILED(device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resDesc, initialState, nullptr, IID_PPV_ARGS(&buffer.m_resource))))
        return std::unexpected(gpu_error::ResourceCreationFailed);

    if (type == D3D12_HEAP_TYPE_UPLOAD || type == D3D12_HEAP_TYPE_READBACK) {
        D3D12_RANGE readRange = (type == D3D12_HEAP_TYPE_READBACK) ? D3D12_RANGE{ 0, size } : D3D12_RANGE{ 0, 0 };
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

    // [Error: 5546] 尋找硬體適配器
    ComPtr<IDXGIAdapter1> adapter;
    bool found = false;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_device)))) {
            std::wcout << L"[*] 成功掛載硬體核心: " << desc.Description << std::endl;
            found = true; break;
        }
    }
    if (!found) return std::unexpected(gpu_error::DeviceCreationFailed);

    // 開始鏈條式初始化
    return TranslateHR(S_OK, gpu_error::QueueCreationFailed)
    .and_then([&]() {
        // 初始化搬運專用的 Fence
        m_copyFenceValue = 0;
        return TranslateHR(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_copyFence)),
            gpu_error::FenceCreationFailed);
    })
        // [5547] 建立雙引擎隊列 (Compute + Copy)
    .and_then([&]() {
        D3D12_COMMAND_QUEUE_DESC computeDesc = { D3D12_COMMAND_LIST_TYPE_COMPUTE, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0 };
        return TranslateHR(m_device->CreateCommandQueue(&computeDesc, IID_PPV_ARGS(&m_computeQueue)), gpu_error::QueueCreationFailed);
    })
    .and_then([&]() {
        D3D12_COMMAND_QUEUE_DESC copyDesc = { D3D12_COMMAND_LIST_TYPE_COPY, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0 };
        return TranslateHR(m_device->CreateCommandQueue(&copyDesc, IID_PPV_ARGS(&m_copyQueue)), gpu_error::QueueCreationFailed);
    })
        // [5548/5549] 同步物件
    .and_then([&]() {
        return TranslateHR(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)), gpu_error::FenceCreationFailed);
    })
    .and_then([&]() -> std::expected<void, gpu_error> {
        m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        return m_fenceEvent ? std::expected<void, gpu_error>({}) : std::unexpected(gpu_error::EventCreationFailed);
    })
        // [5550] 分配器 (雙份)
    .and_then([&]() {
        return TranslateHR(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_commandAllocator)), gpu_error::AllocatorCreationFailed);
    })
    .and_then([&]() {
        return TranslateHR(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&m_copyAllocator)), gpu_error::AllocatorCreationFailed);
    })
        // [5551/5552] 資源分配 (2.8GB VRAM + 雙橋樑)
    .and_then([&]() -> std::expected<void, gpu_error> {
        const size_t vramSize = 2800ULL * 1024 * 1024;
        return GpuBuffer::Create(m_device.Get(), bridgeSize, D3D12_HEAP_TYPE_UPLOAD)
            .and_then([&](GpuBuffer&& bA) -> std::expected<GpuBuffer, gpu_error> {
            m_uploadHeap = std::move(bA);
            return GpuBuffer::Create(m_device.Get(), bridgeSize, D3D12_HEAP_TYPE_UPLOAD);
                })
            .and_then([&](GpuBuffer&& bB) -> std::expected<GpuBuffer, gpu_error> {
            m_uploadHeapB = std::move(bB);
            return GpuBuffer::Create(m_device.Get(), vramSize, D3D12_HEAP_TYPE_DEFAULT);
                })
            .and_then([&](GpuBuffer&& v) -> std::expected<GpuBuffer, gpu_error> {
            m_vramTemp = std::move(v);
            return GpuBuffer::Create(m_device.Get(), bridgeSize, D3D12_HEAP_TYPE_READBACK);
                })
            .and_then([&](GpuBuffer&& rb) -> std::expected<void, gpu_error> {
            m_readbackHeap = std::move(rb);
            return {};
                });
    })
        // [5553] 指令列表 (雙份)
    .and_then([&]() {
        return TranslateHR(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, m_commandAllocator.Get(), nullptr, IID_PPV_ARGS(&m_commandList)), gpu_error::ListCreationFailed);
    })
    .and_then([&]() {
        return TranslateHR(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, m_copyAllocator.Get(), nullptr, IID_PPV_ARGS(&m_copyList)), gpu_error::ListCreationFailed);
    })
        // 預設關閉指令表與編譯 Shader
    .and_then([&]() -> std::expected<void, gpu_error> {
        m_commandList->Close();
        m_copyList->Close();
        return {};
    })
    .and_then([&]() -> std::expected<void, gpu_error> {
        // --- Root Signature ---
        D3D12_ROOT_PARAMETER rootParams[2] = {};
        rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        rootParams[0].Constants.ShaderRegister = 0;
        rootParams[0].Constants.Num32BitValues = 2;
        rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rootParams[1].Descriptor.ShaderRegister = 0;

        D3D12_ROOT_SIGNATURE_DESC rsDesc = { 2, rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE };
        ComPtr<ID3DBlob> sig, err;
        if (FAILED(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err))) return std::unexpected(gpu_error::RootSignatureCreationFailed);
        return TranslateHR(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&m_rootSignature)), gpu_error::RootSignatureCreationFailed);
    })
    .and_then([&]() -> std::expected<void, gpu_error> {
        // --- PSO ---
        auto cs = CompileShader(L"SuperCompute.hlsl", "main", "cs_5_0");
        if (!cs) return std::unexpected(gpu_error::DeviceCreationFailed);

        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_rootSignature.Get();
        psoDesc.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
        return TranslateHR(m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_computePSO)), gpu_error::DeviceCreationFailed);
    });
}

// --- 雙引擎核心操作 ---

void GPU::RecordUploadOnly(size_t vramOffset, size_t uploadOffset, size_t batchSize, int heapIndex) {
    m_copyAllocator->Reset();
    m_copyList->Reset(m_copyAllocator.Get(), nullptr);

    ID3D12Resource* src = (heapIndex == 0) ? m_uploadHeap.GetResource() : m_uploadHeapB.GetResource();

    // Copy Queue 執行時，資源必須處於 COMMON 狀態
    m_copyList->CopyBufferRegion(m_vramTemp.GetResource(), vramOffset, src, uploadOffset, batchSize);

    m_copyList->Close();
}

uint64_t GPU::ExecuteCopyAndSignal() {
    m_copyList->Close();
    ID3D12CommandList* lists[] = { m_copyList.Get() };
    m_copyQueue->ExecuteCommandLists(1, lists);
    m_copyFenceValue++;
    m_copyQueue->Signal(m_copyFence.Get(), m_copyFenceValue);
    return m_copyFenceValue;
}

void GPU::RecordXorShader(size_t vramOffset, size_t size, uint32_t iters) {
    // [A] 強制狀態切換：COMMON -> UAV
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = m_vramTemp.GetResource();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    m_commandList->ResourceBarrier(1, &barrier);

    // [B] 設置管線狀態
    m_commandList->SetComputeRootSignature(m_rootSignature.Get());
    m_commandList->SetPipelineState(m_computePSO.Get());

    // [C] 關鍵：傳遞 32-bit Constants
    // 這裡的 0 代表 RootParameter 的第 0 個槽位
    uint32_t params[4] = { (uint32_t)size, iters, 0, 0 };
    m_commandList->SetComputeRoot32BitConstants(0, 4, params, 0);

    // [D] 關鍵：設置 UAV 視圖
    // 這裡的 1 代表 RootParameter 的第 1 個槽位
    m_commandList->SetComputeRootUnorderedAccessView(1,
        m_vramTemp.GetResource()->GetGPUVirtualAddress() + vramOffset);

    // [E] 分派任務
    uint32_t threadGroups = ((uint32_t)(size / 4) + 255) / 256;
    m_commandList->Dispatch(threadGroups, 1, 1);

    // [F] UAV 屏障：確保寫入落地
    D3D12_RESOURCE_BARRIER uavBarrier = {};
    uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = m_vramTemp.GetResource();
    m_commandList->ResourceBarrier(1, &uavBarrier);

    // [G] 狀態切換回 COMMON，以便 Readback 讀取
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    m_commandList->ResourceBarrier(1, &barrier);
}

uint64_t GPU::ExecuteAndSignal() {
    m_commandList->Close();
    ID3D12CommandList* lists[] = { m_commandList.Get() };
    m_computeQueue->ExecuteCommandLists(1, lists);
    m_fenceValue++;
    m_computeQueue->Signal(m_fence.Get(), m_fenceValue);
    return m_fenceValue;
}

void GPU::Wait() {
    const UINT64 fence = ++m_fenceValue;
    m_computeQueue->Signal(m_fence.Get(), fence);
    WaitForSpecificFence(fence);
}

void GPU::WaitForSpecificFence(uint64_t value) {
    if (m_fence->GetCompletedValue() < value) {
        m_fence->SetEventOnCompletion(value, m_fenceEvent);
        WaitForSingleObject(m_fenceEvent, INFINITE);
    }
}

void GPU::ResetCommandList() {
    m_commandAllocator->Reset();
    m_commandList->Reset(m_commandAllocator.Get(), nullptr);
}

// --- 其餘輔助函數 ---
ComPtr<ID3DBlob> GPU::CompileShader(const std::wstring& f, const std::string& e, const std::string& t) {
    ComPtr<ID3DBlob> b, err;
    D3DCompileFromFile(f.c_str(), nullptr, nullptr, e.c_str(), t.c_str(), 0, 0, &b, &err);
    return b;
}

void GPU::DownloadFromVram(size_t offset, size_t size) {
    // 關鍵：在下載前先同步，確保前一個 Compute 任務的所有數據都已進 VRAM
    Wait();

    ResetCommandList();
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = m_vramTemp.GetResource();
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    m_commandList->ResourceBarrier(1, &b);

    m_commandList->CopyBufferRegion(m_readbackHeap.GetResource(), 0, m_vramTemp.GetResource(), offset, size);

    // 下載完轉回 COMMON
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    m_commandList->ResourceBarrier(1, &b);

    ExecuteAndSignal();
    Wait();
}

void GPU::ReleaseResources() {
    Wait();
    if (m_fenceEvent) {
        CloseHandle(m_fenceEvent);
        m_fenceEvent = nullptr; // 關鍵：防止重複釋放
    }
    m_vramTemp = GpuBuffer(); m_uploadHeap = GpuBuffer(); m_uploadHeapB = GpuBuffer(); m_readbackHeap = GpuBuffer();
    std::cout << "[System] 雙引擎關閉，資源已安全回收。" << std::endl;
}

void GPU::ComputeWaitCopy(uint64_t val) {
    m_computeQueue->Wait(m_copyFence.Get(), val);
}

void* GPU::GetVramAddress() const { return m_uploadHeap.GetCpuAddress(); }
void* GPU::GetVramAddressB() const { return m_uploadHeapB.GetCpuAddress(); }
void* GPU::GetReadbackAddress() const { return m_readbackHeap.GetCpuAddress(); }