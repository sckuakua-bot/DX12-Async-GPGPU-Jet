// SuperCompute.hlsl
cbuffer Params : register(b0)
{
    uint g_size; // 對應 C++ params[0]
    uint g_iters; // 對應 C++ params[1]
    uint g_pad1; // 對應 C++ params[2]
    uint g_pad2; // 對應 C++ params[3]
};

RWByteAddressBuffer Data : register(u0);

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint byteAddr = DTid.x * 4;
    
    // 如果 g_size 因為對齊問題讀到 0，這裡就會直接 Return，導致 GPU=0
    if (byteAddr >= g_size)
        return;

    uint val = Data.Load(byteAddr);
    
    // 執行運算
    for (uint i = 0; i < g_iters; i++)
    {
        val ^= 0xFFFFFFFF;
    }

    Data.Store(byteAddr, val);
}