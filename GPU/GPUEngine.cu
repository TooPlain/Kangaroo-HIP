/*
* This file is part of the BTCCollider distribution (https://github.com/JeanLucPons/Kangaroo).
* Copyright (c) 2020 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef WIN64
#include <unistd.h>
#include <stdio.h>
#endif

#include "GPUEngine.h"
#include <hip/hip_common.h>
#include <hip/hip_runtime.h>

#include <stdint.h>
#include "../Timer.h"

#include "GPUMath.h"
#include "GPUCompute.h"

// ---------------------------------------------------------------------------------------

__global__ void comp_kangaroos(uint64_t *kangaroos,uint32_t maxFound,uint32_t *found,uint64_t dpMask) __attribute__((amdgpu_flat_work_group_size(1, 512))) {

  int xPtr = (hipBlockIdx_x*hipBlockDim_x*GPU_GRP_SIZE) * KSIZE; // x[4] , y[4] , d[2], lastJump
  ComputeKangaroos(kangaroos + xPtr,maxFound,found,dpMask);

}

// ---------------------------------------------------------------------------------------
//#define GPU_CHECK
#ifdef GPU_CHECK
__global__ void check_gpu() {

  // Check ModInv
  uint64_t N[5] = { 0x0BE3D7593BE1147CULL,0x4952AAF512875655ULL,0x08884CCAACCB9B53ULL,0x9EAE2E2225044292ULL,0ULL };
  uint64_t I[5];
  uint64_t R[5];
  bool ok = true;

  /*
  for(uint64_t i=0;i<10000 && ok;i++) {

    Load(R,N);
    _ModInv(R);
    Load(I,R);
    _ModMult(R,N);
    SubP(R);
    if(!_IsOne(R)) {
      ok = false;
      printf("ModInv wrong %d\n",(int)i);
      printf("N = %016llx %016llx %016llx %016llx %016llx\n",N[4],N[3],N[2],N[1],N[0]);
      printf("I = %016llx %016llx %016llx %016llx %016llx\n",I[4],I[3],I[2],I[1],I[0]);
      printf("R = %016llx %016llx %016llx %016llx %016llx\n",R[4],R[3],R[2],R[1],R[0]);
    }

    N[0]++;

  }
  */
  I[4] = 0;
  R[4] = 0;
  for(uint64_t i = 0; i < 100000 && ok; i++) {

    _ModSqr(I,N);
    _ModMult(R,N,N);
    if(!_IsEqual(I,R)) {
      ok = false;
      printf("_ModSqr wrong %d\n",(int)i);
      printf("N = %016llx %016llx %016llx %016llx %016llx\n",N[4],N[3],N[2],N[1],N[0]);
      printf("I = %016llx %016llx %016llx %016llx %016llx\n",I[4],I[3],I[2],I[1],I[0]);
      printf("R = %016llx %016llx %016llx %016llx %016llx\n",R[4],R[3],R[2],R[1],R[0]);
    }

    N[0]++;

  }

}
#endif

// ---------------------------------------------------------------------------------------

using namespace std;

int _ConvertSMVer2Cores(int major,int minor) {

	return 60; // since I have 60 CUs I hope thats the purpose of this lol will find out soon.
	////Returning static value for testing only will implement some sort of function to properly count for amd gpus
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  /*typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
             // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
    { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
    { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
    { 0x30, 192 },
    { 0x32, 192 },
    { 0x35, 192 },
    { 0x37, 192 },
    { 0x50, 128 },
    { 0x52, 128 },
    { 0x53, 128 },
    { 0x60,  64 },
    { 0x61, 128 },
    { 0x62, 128 },
    { 0x70,  64 },
    { 0x72,  64 },
    { 0x75,  64 },
    { -1, -1 } };

  int index = 0;

  while(nGpuArchCoresPerSM[index].SM != -1) {
    if(nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  return 0;*/

}

void GPUEngine::SetWildOffset(Int* offset) {
  wildOffset.Set(offset);
}

GPUEngine::GPUEngine(int nbThreadGroup,int nbThreadPerGroup,int gpuId,uint32_t maxFound) {

  // Initialise CUDA
  this->nbThreadPerGroup = nbThreadPerGroup;
  initialised = false;
  hipError_t err;

  int deviceCount = 0;
  hipError_t error_id = hipGetDeviceCount(&deviceCount);

  if(error_id != hipSuccess) {
    printf("GPUEngine: hipGetDeviceCount %s\n",hipGetErrorString(error_id));
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if(deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  err = hipSetDevice(gpuId);
  if(err != hipSuccess) {
    printf("GPUEngine: %s\n",hipGetErrorString(err));
    return;
  }

  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp,gpuId);

  this->nbThread = nbThreadGroup * nbThreadPerGroup;
  this->maxFound = maxFound;
  this->outputSize = (maxFound*ITEM_SIZE + 4);

  char tmp[512];
  sprintf(tmp,"GPU #%d %s (%dx%d cores) Grid(%dx%d)",
    gpuId,deviceProp.name,deviceProp.multiProcessorCount,
    _ConvertSMVer2Cores(deviceProp.major,deviceProp.minor),
    nbThread / nbThreadPerGroup,
    nbThreadPerGroup);
  deviceName = std::string(tmp);

  // Prefer L1 (We do not use __shared__ at all)
  err = hipDeviceSetCacheConfig(hipFuncCachePreferL1);
  if(err != hipSuccess) {
    printf("GPUEngine: %s\n",hipGetErrorString(err));
    return;
  }

  // Allocate memory
  inputKangaroo = NULL;
  inputKangarooPinned = NULL;
  outputItem = NULL;
  outputItemPinned = NULL;
  jumpPinned = NULL;

  // Input kangaroos
  kangarooSize = nbThread * GPU_GRP_SIZE * KSIZE * 8;
  err = hipMalloc((void **)&inputKangaroo,kangarooSize);
  if(err != hipSuccess) {
    printf("GPUEngine: Allocate input memory: %s\n",hipGetErrorString(err));
    return;
  }
  kangarooSizePinned = nbThreadPerGroup * GPU_GRP_SIZE *  KSIZE * 8;
  err = hipHostRegister(&inputKangarooPinned,kangarooSizePinned,hipHostRegisterMapped);
  if(err != hipSuccess) {
    printf("GPUEngine: Allocate input pinned memory: %s\n",hipGetErrorString(err));
    return;
  }

  // OutputHash
  err = hipMalloc((void **)&outputItem,outputSize);
  if(err != hipSuccess) {
    printf("GPUEngine: Allocate output memory: %s\n",hipGetErrorString(err));
    return;
  }
  err = hipHostRegister(&outputItemPinned,outputSize,hipHostRegisterMapped);
  if(err != hipSuccess) {
    printf("GPUEngine: Allocate output pinned memory: %s\n",hipGetErrorString(err));
    return;
  }

  // Jump array
  jumpSize = NB_JUMP * 8 * 4;
  err = hipHostRegister(&jumpPinned,jumpSize,hipHostRegisterMapped);
  if(err != hipSuccess) {
    printf("GPUEngine: Allocate jump pinned memory: %s\n",hipGetErrorString(err));
    return;
  }

  lostWarning = false;
  initialised = true;
  wildOffset.SetInt32(0);

#ifdef GPU_CHECK

  double minT = 1e9;
  for(int i=0;i<5;i++) {
    double t0 = Timer::get_tick();
    check_gpu<<<1,1>>>();
    hipThreadSynchronize();
    double t1 = Timer::get_tick();
    if( (t1-t0)<minT ) minT = (t1-t0);
  }
  printf("Cuda: %.3f ms\n",minT*1000.0);
  exit(0);

#endif

}

GPUEngine::~GPUEngine() {

  if(inputKangaroo) hipFree(inputKangaroo);
  if(outputItem) hipFree(outputItem);
  if(inputKangarooPinned) hipHostFree(inputKangarooPinned);
  if(outputItemPinned) hipHostFree(outputItemPinned);
  if(jumpPinned) hipHostFree(jumpPinned);

}


int GPUEngine::GetMemory() {
  return kangarooSize + outputSize + jumpSize;
}


int GPUEngine::GetGroupSize() {
  return GPU_GRP_SIZE;
}

bool GPUEngine::GetGridSize(int gpuId,int *x,int *y) {

  if(*x <= 0 || *y <= 0) {

    int deviceCount = 0;
    hipError_t error_id = hipGetDeviceCount(&deviceCount);

    if(error_id != hipSuccess) {
      printf("GPUEngine: hipGetDeviceCount %s\n",hipGetErrorString(error_id));
      return false;
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if(deviceCount == 0) {
      printf("GPUEngine: There are no available device(s) that support HIPAMD\n");
      return false;
    }

    if(gpuId >= deviceCount) {
      printf("GPUEngine::GetGridSize() Invalid gpuId\n");
      return false;
    }

    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp,gpuId);

    if(*x <= 0) *x = 2 * deviceProp.multiProcessorCount;
    if(*y <= 0) *y = 2 * _ConvertSMVer2Cores(deviceProp.major,deviceProp.minor);
    if(*y <= 0) *y = 128;

  }

  return true;

}

void *GPUEngine::AllocatePinnedMemory(size_t size) {

  void *buff;

  hipError_t err = hipHostRegister(&buff,size,hipHostRegisterPortable);
  if(err != hipSuccess) {
    printf("GPUEngine: AllocatePinnedMemory: %s\n",hipGetErrorString(err));
    return NULL;
  }

  return buff;

}

void GPUEngine::FreePinnedMemory(void *buff) {
  hipHostFree(buff);
}

void GPUEngine::PrintCudaInfo() {

  hipError_t err;

  const char *sComputeMode[] =
  {
    "Multiple host threads",
    "Only one host thread",
    "No host thread",
    "Multiple process threads",
    "Unknown",
    NULL
  };

  int deviceCount = 0;
  hipError_t error_id = hipGetDeviceCount(&deviceCount);

  if(error_id != hipSuccess) {
    printf("GPUEngine: hipGetDeviceCount %s\n",hipGetErrorString(error_id));
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if(deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  for(int i = 0; i<deviceCount; i++) {

    err = hipSetDevice(i);
    if(err != hipSuccess) {
      printf("GPUEngine: hipSetDevice(%d) %s\n",i,hipGetErrorString(err));
      return;
    }

    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp,i);
    printf("GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB) (%s)\n",
      i,deviceProp.name,deviceProp.multiProcessorCount,
      _ConvertSMVer2Cores(deviceProp.major,deviceProp.minor),
      deviceProp.major,deviceProp.minor,(double)deviceProp.totalGlobalMem / 1048576.0,
      sComputeMode[deviceProp.computeMode]);

  }

}

int GPUEngine::GetNbThread() {
  return nbThread;
}

void GPUEngine::SetKangaroos(Int *px,Int *py,Int *d) {

  // Sets the kangaroos of each thread
  int gSize = KSIZE * GPU_GRP_SIZE;
  int strideSize = nbThreadPerGroup * KSIZE;
  int nbBlock = nbThread / nbThreadPerGroup;
  int blockSize = nbThreadPerGroup * gSize;
  int idx = 0;

  for(int b = 0; b < nbBlock; b++) {
    for(int g = 0; g < GPU_GRP_SIZE; g++) {
      for(int t = 0; t < nbThreadPerGroup; t++) {

        // X
        inputKangarooPinned[g * strideSize + t + 0 * nbThreadPerGroup] = px[idx].bits64[0];
        inputKangarooPinned[g * strideSize + t + 1 * nbThreadPerGroup] = px[idx].bits64[1];
        inputKangarooPinned[g * strideSize + t + 2 * nbThreadPerGroup] = px[idx].bits64[2];
        inputKangarooPinned[g * strideSize + t + 3 * nbThreadPerGroup] = px[idx].bits64[3];

        // Y
        inputKangarooPinned[g * strideSize + t + 4 * nbThreadPerGroup] = py[idx].bits64[0];
        inputKangarooPinned[g * strideSize + t + 5 * nbThreadPerGroup] = py[idx].bits64[1];
        inputKangarooPinned[g * strideSize + t + 6 * nbThreadPerGroup] = py[idx].bits64[2];
        inputKangarooPinned[g * strideSize + t + 7 * nbThreadPerGroup] = py[idx].bits64[3];

        // Distance
        Int dOff;
        dOff.Set(&d[idx]);
        if(idx % 2 == WILD) dOff.ModAddK1order(&wildOffset);
        inputKangarooPinned[g * strideSize + t + 8 * nbThreadPerGroup] = dOff.bits64[0];
        inputKangarooPinned[g * strideSize + t + 9 * nbThreadPerGroup] = dOff.bits64[1];

#ifdef USE_SYMMETRY
        // Last jump
        inputKangarooPinned[t + 10 * nbThreadPerGroup] = (uint64_t)NB_JUMP;
#endif

        idx++;
      }

    }

    uint32_t offset = b * blockSize;
    hipMemcpy(inputKangaroo + offset,inputKangarooPinned,kangarooSizePinned,hipMemcpyHostToDevice);

  }

  hipError_t err = hipGetLastError();
  if(err != hipSuccess) {
    printf("GPUEngine: SetKangaroos: %s\n",hipGetErrorString(err));
  }

}

void GPUEngine::GetKangaroos(Int *px,Int *py,Int *d) {

  if(inputKangarooPinned==NULL ) {
    printf("GPUEngine: GetKangaroos: Cannot retreive kangaroos, mem has been freed\n");
    return;
  }

  // Sets the kangaroos of each thread
  int gSize = KSIZE * GPU_GRP_SIZE;
  int strideSize = nbThreadPerGroup * KSIZE;
  int nbBlock = nbThread / nbThreadPerGroup;
  int blockSize = nbThreadPerGroup * gSize;
  int idx = 0;

  for(int b = 0; b < nbBlock; b++) {

    uint32_t offset = b * blockSize;
    hipMemcpy(inputKangarooPinned,inputKangaroo + offset,kangarooSizePinned,hipMemcpyDeviceToHost);

    for(int g = 0; g < GPU_GRP_SIZE; g++) {

      for(int t = 0; t < nbThreadPerGroup; t++) {

        // X
        px[idx].bits64[0] = inputKangarooPinned[g * strideSize + t + 0 * nbThreadPerGroup];
        px[idx].bits64[1] = inputKangarooPinned[g * strideSize + t + 1 * nbThreadPerGroup];
        px[idx].bits64[2] = inputKangarooPinned[g * strideSize + t + 2 * nbThreadPerGroup];
        px[idx].bits64[3] = inputKangarooPinned[g * strideSize + t + 3 * nbThreadPerGroup];
        px[idx].bits64[4] = 0;

        // Y
        py[idx].bits64[0] = inputKangarooPinned[g * strideSize + t + 4 * nbThreadPerGroup];
        py[idx].bits64[1] = inputKangarooPinned[g * strideSize + t + 5 * nbThreadPerGroup];
        py[idx].bits64[2] = inputKangarooPinned[g * strideSize + t + 6 * nbThreadPerGroup];
        py[idx].bits64[3] = inputKangarooPinned[g * strideSize + t + 7 * nbThreadPerGroup];
        py[idx].bits64[4] = 0;

        // Distance
        Int dOff;
        dOff.SetInt32(0);
        dOff.bits64[0] = inputKangarooPinned[g * strideSize + t + 8 * nbThreadPerGroup];
        dOff.bits64[1] = inputKangarooPinned[g * strideSize + t + 9 * nbThreadPerGroup];
        if(idx % 2 == WILD) dOff.ModSubK1order(&wildOffset);
        d[idx].Set(&dOff);

        idx++;
      }

    }
  }

  hipError_t err = hipGetLastError();
  if(err != hipSuccess) {
    printf("GPUEngine: GetKangaroos: %s\n",hipGetErrorString(err));
  }

}

void GPUEngine::SetKangaroo(uint64_t kIdx,Int *px,Int *py,Int *d) {

  int gSize = KSIZE * GPU_GRP_SIZE;
  int strideSize = nbThreadPerGroup * KSIZE;
  int blockSize = nbThreadPerGroup * gSize;

  uint64_t t = kIdx % nbThreadPerGroup;
  uint64_t g = (kIdx / nbThreadPerGroup) % GPU_GRP_SIZE;
  uint64_t b = kIdx / (nbThreadPerGroup*GPU_GRP_SIZE);

  // X
  inputKangarooPinned[0] = px->bits64[0];
  hipMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 0 * nbThreadPerGroup),inputKangarooPinned,8,hipMemcpyHostToDevice);
  inputKangarooPinned[0] = px->bits64[1];
  hipMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 1 * nbThreadPerGroup),inputKangarooPinned,8,hipMemcpyHostToDevice);
  inputKangarooPinned[0] = px->bits64[2];
  hipMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 2 * nbThreadPerGroup),inputKangarooPinned,8,hipMemcpyHostToDevice);
  inputKangarooPinned[0] = px->bits64[3];
  hipMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 3 * nbThreadPerGroup),inputKangarooPinned,8,hipMemcpyHostToDevice);

  // Y
  inputKangarooPinned[0] = py->bits64[0];
  hipMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 4 * nbThreadPerGroup),inputKangarooPinned,8,hipMemcpyHostToDevice);
  inputKangarooPinned[0] = py->bits64[1];
  hipMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 5 * nbThreadPerGroup),inputKangarooPinned,8,hipMemcpyHostToDevice);
  inputKangarooPinned[0] = py->bits64[2];
  hipMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 6 * nbThreadPerGroup),inputKangarooPinned,8,hipMemcpyHostToDevice);
  inputKangarooPinned[0] = py->bits64[3];
  hipMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 7 * nbThreadPerGroup),inputKangarooPinned,8,hipMemcpyHostToDevice);

  // D
  Int dOff;
  dOff.Set(d);
  if(kIdx % 2 == WILD) dOff.ModAddK1order(&wildOffset);
  inputKangarooPinned[0] = dOff.bits64[0];
  hipMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 8 * nbThreadPerGroup),inputKangarooPinned,8,hipMemcpyHostToDevice);
  inputKangarooPinned[0] = dOff.bits64[1];
  hipMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 9 * nbThreadPerGroup),inputKangarooPinned,8,hipMemcpyHostToDevice);

#ifdef USE_SYMMETRY
  // Last jump
  inputKangarooPinned[0] = (uint64_t)NB_JUMP;
  hipMemcpy(inputKangaroo + (b * blockSize + g * strideSize + t + 10 * nbThreadPerGroup),inputKangarooPinned,8,hipMemcpyHostToDevice);
#endif

}

bool GPUEngine::callKernel() {

  // Reset nbFound
  hipMemset(outputItem,0,4);

  // Call the kernel (Perform STEP_SIZE keys per thread)
  //comp_kangaroos << < nbThread / nbThreadPerGroup,nbThreadPerGroup >> > (inputKangaroo,maxFound,outputItem,dpMask);

  //Should work insead of using the Cuda syntax
  hipLaunchKernelGGL(comp_kangaroos, dim3(nbThread / nbThreadPerGroup), dim3(nbThreadPerGroup), 0, 0,inputKangaroo,maxFound,outputItem,dpMask);
  
  hipError_t err = hipGetLastError();
  if(err != hipSuccess) {
    printf("GPUEngine: Kernel: %s\n",hipGetErrorString(err));
    return false;
  }

  return true;

}

void GPUEngine::SetParams(uint64_t dpMask,Int *distance,Int *px,Int *py) {
  
  this->dpMask = dpMask;

  for(int i=0;i< NB_JUMP;i++)
    memcpy(jumpPinned + 2*i,distance[i].bits64,16);
  hipMemcpyToSymbol(jD,jumpPinned,jumpSize/2);
  hipError_t err = hipGetLastError();
  if(err != hipSuccess) {
    printf("GPUEngine: SetParams: Failed to copy to constant memory: %s\n",hipGetErrorString(err));
    return;
  }

  for(int i = 0; i < NB_JUMP; i++)
    memcpy(jumpPinned + 4 * i,px[i].bits64,32);
  hipMemcpyToSymbol(jPx,jumpPinned,jumpSize);
  err = hipGetLastError();
  if(err != hipSuccess) {
    printf("GPUEngine: SetParams: Failed to copy to constant memory: %s\n",hipGetErrorString(err));
    return;
  }

  for(int i = 0; i < NB_JUMP; i++)
    memcpy(jumpPinned + 4 * i,py[i].bits64,32);
  hipMemcpyToSymbol(jPy,jumpPinned,jumpSize);
  err = hipGetLastError();
  if(err != hipSuccess) {
    printf("GPUEngine: SetParams: Failed to copy to constant memory: %s\n",hipGetErrorString(err));
    return;
  }

}

bool GPUEngine::callKernelAndWait() {

  // Debug function
  callKernel();
  hipMemcpy(outputItemPinned,outputItem,outputSize,hipMemcpyDeviceToHost);
  hipError_t err = hipGetLastError();
  if(err != hipSuccess) {
    printf("GPUEngine: callKernelAndWait: %s\n",hipGetErrorString(err));
    return false;
  }

  return true;

}

bool GPUEngine::Launch(std::vector<ITEM> &hashFound,bool spinWait) {


  hashFound.clear();

  // Get the result

  if(spinWait) {

    hipMemcpy(outputItemPinned,outputItem,outputSize,hipMemcpyDeviceToHost);

  } else {

    // Use hipMemcpyAsync to avoid default spin wait of hipMemcpy wich takes 100% CPU
    hipEvent_t evt;
    hipEventCreate(&evt);
    hipMemcpyAsync(outputItemPinned,outputItem,4,hipMemcpyDeviceToHost,0);
    hipEventRecord(evt,0);
    while(hipEventQuery(evt) == hipErrorNotReady) {
      // Sleep 1 ms to free the CPU
      Timer::SleepMillis(1);
    }
    hipEventDestroy(evt);

  }

  hipError_t err = hipGetLastError();
  if(err != hipSuccess) {
    printf("GPUEngine: Launch: %s\n",hipGetErrorString(err));
    return false;
  }

  // Look for prefix found
  uint32_t nbFound = outputItemPinned[0];
  if(nbFound > maxFound) {
    // prefix has been lost
    if(!lostWarning) {
      printf("\nWarning, %d items lost\nHint: Search with less threads (-g) or increse dp (-d)\n",(nbFound - maxFound));
      lostWarning = true;
    }
    nbFound = maxFound;
  }

  // When can perform a standard copy, the kernel is eneded
  hipMemcpy(outputItemPinned,outputItem,nbFound*ITEM_SIZE + 4,hipMemcpyDeviceToHost);

  for(uint32_t i = 0; i < nbFound; i++) {
    uint32_t *itemPtr = outputItemPinned + (i*ITEM_SIZE32 + 1);
    ITEM it;

    it.kIdx = *((uint64_t*)(itemPtr + 12));

    uint64_t *x = (uint64_t *)itemPtr;
    it.x.bits64[0] = x[0];
    it.x.bits64[1] = x[1];
    it.x.bits64[2] = x[2];
    it.x.bits64[3] = x[3];
    it.x.bits64[4] = 0;

    uint64_t *d = (uint64_t *)(itemPtr + 8);
    it.d.bits64[0] = d[0];
    it.d.bits64[1] = d[1];
    it.d.bits64[2] = 0;
    it.d.bits64[3] = 0;
    it.d.bits64[4] = 0;
    if(it.kIdx % 2 == WILD) it.d.ModSubK1order(&wildOffset);

    hashFound.push_back(it);
  }

  return callKernel();

}
