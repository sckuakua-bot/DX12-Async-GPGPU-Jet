// CommonCompute.hlsl
RWByteAddressBuffer DataBuffer : register(u0);

// 嚴格對齊 16-Byte 邊界
cbuffer Params : register(b0)
{
    uint g_MaxAddress; // Offset 0
    uint g_Iterations; // Offset 4
    uint g_Mode; // Offset 8
    uint g_Reserved; // Offset 12 (Padding)
};

// 旋轉位移：GPU 硬體級指令
uint rotl(uint x, uint n)
{
    return (x << n) | (x >> (32 - n));
}

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint byteAddr = DTid.x * 4;
    
    // 邊界檢查
    if (byteAddr >= g_MaxAddress)
        return;

    // 讀取原始數據
    uint x = DataBuffer.Load(byteAddr);
    uint y = x ^ 0xDEADBEEF;

    // 嚴格循環邏輯
    for (uint i = 0; i < g_Iterations; i++)
    {
        x = x + y; // 自動溢出繞回 (Modulo 2^32)
        y = rotl(y ^ x, 7);
        x = rotl(x + i, 13) ^ 0x55555555;
    }

    // 寫回結果
    DataBuffer.Store(byteAddr, x ^ y);
}