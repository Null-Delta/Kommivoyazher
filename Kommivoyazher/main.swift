//
//  main.swift
//  Kommivoyazher
//
//  Created by Rustam Khakhuk on 21.03.2022.
//

import Foundation
import Metal
import MetalKit

var vertexCount = 10

var graph: [[Double]] = []

var result = Double.infinity
var way: [Int] = []

//факториал
func fact(value: Double) -> Double {
    var result: Double = 1
    for i in 1...Int(value) {
        result *= Double(i)
    }
    return result
}


//MARK: Рекурсивный алгоритм решения задачи Коммивояжера
func kommivoyazherRecursion(targetVertex: Int, wayLenght: Double, usedVertexs: [Int]) {
    if usedVertexs.count == vertexCount {
        if wayLenght + graph[targetVertex][0] < result {
            result = wayLenght + graph[targetVertex][0]
            way = usedVertexs
            way.append(0)
        }
    } else {
        for index in 0..<vertexCount where !usedVertexs.contains(index) {
            var newUsedVertex = usedVertexs;
            newUsedVertex.append(index);
            kommivoyazherRecursion(targetVertex: index, wayLenght: wayLenght + graph[targetVertex][index], usedVertexs: newUsedVertex)
        }
    }
}

//Генерация подстановки длины size с порядковым номером index
func getSubstitution(index: Int, size: Int) -> [Int] {
    var result: [Int] = []
    var nums: [Int] = Array(0..<size)
    
    var localIndex = index
    var localSize = size
    var localCount = Int(fact(value: Double(size)))

    for _ in 0..<size {
        result.append(nums.remove(at: localIndex / (localCount / localSize)))
        
        localCount /= localSize
        localIndex %= localCount
        localSize -= 1
    }
        
    return result
}

//MARK: алгоритм решения задачи Коммивояжера без рекурсии
func kommivoyazherFront() {
    for index in 0..<Int(fact(value: Double(vertexCount - 1))) {
        var wayLenght: Double = 0
        
        var checkWay: [Int] = [0]
        checkWay.append(contentsOf: getSubstitution(index: index, size: vertexCount - 1).map({v in v + 1}))
        checkWay.append(0)
    
        for edge in 0..<vertexCount {
            wayLenght += graph[checkWay[edge]][checkWay[edge + 1]]
        }
        
        if wayLenght < result {
            result = wayLenght
            way = checkWay
        }
    }
}

//MARK: Реализация алгоритма решения задачи Коммивояжера с использованием многопоточности
func kommivoyazherConcurrent() {
    let coresCount = Foundation.ProcessInfo.processInfo.processorCount
    
    var mins: [Double] = [Double].init(repeating: .infinity, count: coresCount)
    var minWays: [[Int]] = [[Int]].init(repeating: [], count: coresCount)

    let count = fact(value: Double(vertexCount - 1))
    let delta = count / Double(coresCount)
    
    DispatchQueue.concurrentPerform(iterations: coresCount, execute: { coreIndex in
        let start = Int(delta * Double(coreIndex))
        let end = Int(delta * Double(coreIndex + 1))

        if start != end {
            for index in start..<end {
                var wayLenght: Double = 0

                var checkWay: [Int] = [0]
                checkWay.append(contentsOf: getSubstitution(index: index, size: vertexCount - 1).map({v in v + 1}))
                checkWay.append(0)

                for edge in 0..<vertexCount {
                    wayLenght += graph[checkWay[edge]][checkWay[edge + 1]]
                }

                if wayLenght < mins[coreIndex] {
                    mins[coreIndex] = wayLenght
                    minWays[coreIndex] = checkWay
                }
            }
        }
    })
    
    let minIndex = mins.firstIndex(of: mins.min()!)!
    
    result = mins[minIndex]
    way = minWays[minIndex]
}

//MARK: Реализация алгоритма решения задачи Коммивояжера с использованием графического процессора
func kommivoyazherShader() {
    let device = MTLCreateSystemDefaultDevice()!
    let queue = device.makeCommandQueue()!
    let buffer = queue.makeCommandBuffer()!
    let encoder = buffer.makeComputeCommandEncoder()!
    
    let pipeline = try! device.makeComputePipelineState(function: device.makeDefaultLibrary()!.makeFunction(name: "kommivoyazher")!)
    
    encoder.setComputePipelineState(pipeline)
    
    var graphArray: [Float] = []
    var vertCount = Float(vertexCount)
    var wayCount = Float(fact(value: Double(vertCount - 1)))
    var threadsCount: Float = 512
    
    let output: [Float] = [Float].init(repeating: Float.infinity, count: 512)
    let output2: [Float] = [Float].init(repeating: Float.infinity, count: 512)

    let outputBuffer = device.makeBuffer(bytes: output, length: MemoryLayout<Float>.stride * output.count, options: [.storageModeShared])!

    let outputIndexBuffer = device.makeBuffer(bytes: output2, length: MemoryLayout<Float>.stride * output.count, options: [.storageModeShared])!

    graph.forEach({ arr in
        graphArray.append(contentsOf: arr.map({d in Float(d)}))
    })
        
    encoder.setBuffer(device.makeBuffer(bytes: &wayCount, length: MemoryLayout<Float>.stride, options: []), offset: 0, index: 0)

    encoder.setBuffer(device.makeBuffer(bytes: &vertCount, length: MemoryLayout<Float>.stride, options: []), offset: 0, index: 1)
    encoder.setBuffer(device.makeBuffer(bytes: &threadsCount, length: MemoryLayout<Float>.stride, options: []), offset: 0, index: 2)

    encoder.setBuffer(device.makeBuffer(bytes: graphArray, length: MemoryLayout<Float>.stride * graphArray.count, options: []), offset: 0, index: 3)

    encoder.setBuffer(outputBuffer, offset: 0, index: 4)
    encoder.setBuffer(outputIndexBuffer, offset: 0, index: 5)

    let threadGroupSize = MTLSize(width: 1, height: 1, depth: 1)
    let threadGroupCount = MTLSize(width: 512, height: 1, depth: 1)
    
    encoder.dispatchThreadgroups(threadGroupCount, threadsPerThreadgroup: threadGroupSize)

    encoder.endEncoding()
    
    buffer.commit()
    buffer.waitUntilCompleted()

    var resultArray: [Float] = []
    var resultIndexArray: [Float] = []
    for i in 0..<512 {
        resultArray.append(outputBuffer.contents().load(fromByteOffset: MemoryLayout<Float>.stride * i, as: Float.self))
        resultIndexArray.append(outputIndexBuffer.contents().load(fromByteOffset: MemoryLayout<Float>.stride * i, as: Float.self))
    }
    
    let index = resultArray.firstIndex(of: resultArray.min()!)!
    let wayIndex = outputIndexBuffer.contents().load(fromByteOffset: MemoryLayout<Float>.stride * index, as: Float.self)
    
    var targetWay = getSubstitution(index: Int(wayIndex), size: vertexCount - 1).map({i in i + 1})
    targetWay.insert(0, at: 0)
    targetWay.append(0)
    
    result = Double(resultArray.min()!)
    way = targetWay
}

print("Введите количество вершин в графе:")
vertexCount = Int(readLine()!)!

for i in 0..<vertexCount {
    graph.append([])
    for _ in 0..<vertexCount {
        graph[i].append(Double(Int.random(in: 1...100)))
    }
}

var start = Date()

start = Date()
kommivoyazherFront()
print("Время выполнения алгоритма ( без рекукрсии ): \(abs(start.timeIntervalSinceNow))")
print("Самый выгодный маршрут: \(way.map({ i in i}))")
print("Длина маршрута: \(result)")
print()

result = Double.infinity

start = Date()
kommivoyazherConcurrent()
print("Время выполнения алгоритма ( с многопоточностью ): \(abs(start.timeIntervalSinceNow))")
print("Самый выгодный маршрут: \(way.map({ i in i}))")
print("Длина маршрута: \(result)")
print()

result = Double.infinity

start = Date()
kommivoyazherShader()
print("Время выполнения алгоритма ( шейдер ): \(abs(start.timeIntervalSinceNow))")
print("Самый выгодный маршрут: \(way.map({ i in i}))")
print("Длина маршрута: \(result)")
print()
