//
//  kommivoyazherShader.metal
//  Kommivoyazher
//
//  Created by Rustam Khakhuk on 21.03.2022.
//

#include <metal_stdlib>


using namespace metal;

//получение такого индекса iter, что в массиве from на данном индексе находится index-я true
int getValue(bool from[], int index) {
    int localIndex = -1;
    int iter = 0;
    while(true) {
        localIndex += from[iter] ? 1 : 0;
        if (localIndex == index) {
            return iter;
        }
        iter++;
    }
    
    return iter;
}

kernel void kommivoyazher(device float &waysCount    [[buffer(0)]],
                          device float &vertexCount  [[buffer(1)]],
                          device float &threadsCount [[buffer(2)]],
                          device float *graph        [[buffer(3)]],
                          device float *output       [[buffer(4)]],
                          device float *outputIndex  [[buffer(5)]],
                          ushort2 sumIndex [[thread_position_in_grid]]) {
    int index = sumIndex.x;
    
    float localMin = MAXFLOAT;
    float localMinIndex = MAXFLOAT;

    float delta = waysCount / threadsCount;
    int start = (int)(delta * index);
    int end = (int)(delta * (index + 1));
    
    if (start == end) return;

    for(int iter = start; iter < end; iter++) {

        if (iter > waysCount) {
            break;
        }
        
        bool nums[15];
        float array[15];
        float wayLenght = 0;

        for(int i = 0; i < vertexCount; i++) {
            nums[i] = true;
        }

        int localIndex = iter;
        int localSize = vertexCount - 1;
        int localCount = waysCount;

        for(int i = 0; i < vertexCount; i++) {
            int iter2 = getValue(nums, localIndex / (localCount / localSize));
            array[i] = iter2 + 1;

            nums[iter2] = false;

            localCount /= localSize;
            localIndex %= localCount;
            localSize--;
        }

        wayLenght += graph[int(0 * vertexCount + array[0])];
        wayLenght += graph[int(array[int(vertexCount - 2)] * vertexCount + 0)];

        for(int i = 0; i < vertexCount - 2; i++) {
            wayLenght += graph[int(array[i] * vertexCount + array[i + 1])];
        }

        if (wayLenght < localMin) {
            localMin = wayLenght;
            localMinIndex = iter;
        }
    }
    
    output[index] = localMin;
    outputIndex[index] = localMinIndex;

}
