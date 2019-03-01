//
//  Swift_Neural_Network.swift
//  Swift_NN
//
//  Created by Christopher Weaver on 1/30/19.
//  Copyright © 2019 Netronix. All rights reserved.
//

import Foundation

enum Layer_Types: String {
    case Convoluted = "Convoluted"
    case Recurrent = "Recurrent"
    case Maxpool = "Maxpool"
    case Flatten = "Flatten"
    case Input = "Input"
    case Fully_Connected = "Fully_Connected"
}

enum Activation_Function : String {
    case Sigmoid = "sigmoid"
    case Relu = "relu"
    case Leaky_Relu = "leadky relu"
    case Tanh = "tahn"
    case SoftMax = "softmax"
    case Pass = "pass"
    case Input = "input"
    case Maxpool = "maxpool"
    case Flatten = "flatten"
    
    func run(input : Matrix) -> Matrix  {
        switch self {
        case .Relu:
            return input
        case .Sigmoid:
            return 1.0 / (1.0 + exp(value:-input))
        case .Pass:
            return input
        case .Input:
            return input
        case .SoftMax:
            let result = input[0]
            let exps = exp(value: result)
            let sum_of_exps = sum(value: exps)
            return [exps / sum_of_exps]
        default:
            return [[]]
        }
    }
}

func sum(value : [Matrix]) -> Double {
    var summation : Double = 0.0
    for dimension in value {
        summation += sum(value: dimension)
    }
    return summation
}

func return2DMatrixSlice(value: Matrix, startIndexX: Int, startIndexY: Int, slice : Int) -> Matrix {
    var returnMatrix = Matrix(repeating:  Array(repeating: 0.0, count: slice), count: slice)
    for second_dimension in startIndexY..<(startIndexY + slice) {
        for first_diemsion in startIndexX..<(startIndexX + slice) {
            returnMatrix[second_dimension - startIndexY][first_diemsion - startIndexX] = value[second_dimension][first_diemsion]
        }
    }
    
    return returnMatrix
}

func return3DMatrixSlice(value : [Matrix], startIndexX: Int, startIndexY: Int, slice : Int) -> [Matrix] {
    var returnMatrix = [Matrix](repeating: Array(repeating: Array(repeating: 0.0, count: 3), count: 3), count: value.count)
    for third_dimension in 0..<(value.count) {
        for second_dimension in startIndexY..<(startIndexY + slice) {
            for first_diemsion in startIndexX..<(startIndexX + slice) {
                returnMatrix[third_dimension][second_dimension - startIndexY][first_diemsion - startIndexX] = value[third_dimension][second_dimension][first_diemsion]
            }
        }
    }
    
    return returnMatrix
}

func sum(value : Matrix) -> Double {
    var summation : Double = 0.0
    for row in value {
        for column in row {
            summation += column
        }
    }
    return summation
}

func sum(value : [Double]) -> Double {
    var summation : Double = 0.0
    for number in value {
        summation += number
    }
    return summation
}

func / (lhs: [Double], rhs: Double) -> [Double] {
    return lhs.map { $0 / rhs }
}

func / (lhs: Double, rhs: [Double]) -> [Double] {
    return rhs.map { lhs / $0 }
}

func / (lhs: Double, rhs: Matrix) -> Matrix {
    return rhs.map { lhs / $0 }
}

func + (lhs: Double, rhs: [Double]) -> [Double] {
    return rhs.map { lhs + $0 }
}

func + (lhs: Double, rhs: Matrix) -> Matrix {
    return rhs.map { lhs + $0 }
}

func exp(value: [Double]) -> [Double] {
    return value.map { pow(2.718281, $0) }
}

func exp(value: Matrix) -> Matrix {
    return value.map { exp(value: $0) }
}

func max(value : [Double]) -> Double {
    var maxvalue : Double = 0.0
    for v in value {
        if v > maxvalue {
            maxvalue = v
        }
    }
    return maxvalue
}

func max(value : Matrix) -> Double {
    var maxValue : Double = 0.0
    for row in value {
        let maxInRow = max(value: row)
        if maxInRow > maxValue {
            maxValue = maxInRow
        }
    }
    return maxValue
}

func - (lhs: Double, rhs: [Double]) -> [Double] {
    return rhs.map { lhs - $0 }
}

func - (lhs: Double, rhs: Matrix) -> Matrix {
    return rhs.map {lhs - $0}
}

prefix func - (value: [Double]) -> [Double] {
    return value.map { -$0 }
}

prefix func - (value: Matrix) -> Matrix {
    return value.map { -$0 }
}

func * (lhs : [Double], rhs : [Double]) -> [Double] {
    var returnArray : [Double] = []
    for value in 0 ..< lhs.count {
        returnArray.append(lhs[value] * rhs[value])
    }
    return returnArray
}

typealias Matrix = [[Double]]

class Swift_Neural_Network {
    
    var weights : [Matrix] = []
    var layers : [Layer_Types] = []
    var stride : [Int] = []
    var filters : [[[Matrix]]] = []
    var hidden_layer_results : [[Double]] = []
    var activation_functions : [Activation_Function] = []

    func convertToActivationFunctions(activationFunctions : [String]) -> [Activation_Function] {
        var returnArray : [Activation_Function] = []
        for activationFunction in activationFunctions {
            switch activationFunction {
            case "sigmoid":
                returnArray.append(Activation_Function.Sigmoid)
            case "relu":
                returnArray.append(Activation_Function.Relu)
            case "leaky relu":
                returnArray.append(Activation_Function.Leaky_Relu)
            case "tahn":
                returnArray.append(Activation_Function.Tanh)
            case "softmax":
                returnArray.append(Activation_Function.SoftMax)
            case "pass":
                returnArray.append(Activation_Function.Pass)
            case "input":
                returnArray.append(Activation_Function.Input)
            case "maxpool":
                returnArray.append(Activation_Function.Maxpool)
            case "flatten":
                returnArray.append(Activation_Function.Flatten)
            default:
                print("failed to find activiation function type")
            }
        }
        return returnArray
    }

    func convertToLayerTypes(layers : [String]) -> [Layer_Types] {
        var returnArray : [Layer_Types] = []
        for layer in layers {
            switch layer {
            case "Fully_Connected":
                returnArray.append(Layer_Types.Fully_Connected)
            case "Input":
                returnArray.append(Layer_Types.Input)
            case "Flatten":
                returnArray.append(Layer_Types.Flatten)
            case "Maxpool":
                returnArray.append(Layer_Types.Maxpool)
            case "Convoluted":
                returnArray.append(Layer_Types.Convoluted)
            case "Recurrent":
                returnArray.append(Layer_Types.Recurrent)
            default:
                print("oh well")
            }
        }
        return returnArray
    }
    
    func multiply( _ matrixA:[Matrix], _ matrixB:[Matrix]) -> [Matrix] {
        var results : [Matrix] = [Matrix](repeating: Array(repeating: Array(repeating: 0.0, count: matrixA[0].count), count: matrixA[0].count), count: matrixA.count)
        for tensor_row in 0 ..< results.count {
            results[tensor_row] = multiplyNotDot(matrixA[tensor_row], matrixB[tensor_row])
        }
        return results
    }
    
    func multiplyNotDot( _ matrixA:[[Double]], _ matrixB:[[Double]]) -> Matrix {
        var result:[[Double]] = [[Double]]( repeating: [Double]( repeating: 0, count: matrixB[0].count ), count: matrixA.count )
        for result_row in 0 ..< result.count {
            for result_column in 0 ..< result[0].count {
                result[result_row][result_column] = matrixA[result_row][result_column] * matrixB[result_row][result_column]
            }
        }
        return result
    }
    
    func multiply( _ matrixA:[[Double]], _ matrixB:[[Double]]) -> [[Double]] {
        if matrixA[0].count != matrixB.count {
            print( "Illegal matrix dimensions! \(matrixA[0].count) * \(matrixB[0].count)" )
            return [[]]
        }
        var result:[[Double]] = [[Double]]( repeating: [Double]( repeating: 0, count: matrixB[0].count ), count: matrixA.count )
        for result_row in 0 ..< result.count {
            for result_column in 0 ..< result[0].count {
                var transposedColumn : [Double] = []
                for matBColumn in matrixB {
                    transposedColumn.append(matBColumn[result_column])
                }
                result[result_row][result_column] = sum(value: matrixA[result_row] * transposedColumn)
            }
        }
        
        return result
    }
    
    func prettyPrintMatrix( _ matrix:[[Double]] ) {
        for array in matrix {
            print( array )
        }
    }
    
    func executeNeuralNetwork(input : [Matrix]) -> Any {
        var current_results : Any!
        var layerIndex : Int = 0
        var filterIndex : Int = 0
        for layer in self.layers {
            switch layer {
            case .Convoluted:
                current_results = convolution(input_data: current_results as! [Matrix], filters: self.filters[filterIndex], stride: self.stride[filterIndex], activation_fuction: self.activation_functions[layerIndex])
                filterIndex += 1
                print("Convolutional")
            case .Maxpool:
                current_results = maxpool(input_data: current_results as! [Matrix], window_size: 2, stride: self.stride[filterIndex])
                filterIndex += 1
                print("Maxpool")
            case .Flatten:
                current_results = flatten(input_data: current_results as! [Matrix])
                filterIndex += 1
                print("Flatten")
            case .Input:
                print("Input")
                current_results = input
                layerIndex -= 1
            case .Fully_Connected:
                if (current_results as? Matrix) != nil {
                } else {
                    current_results = [current_results]
                }
                current_results = forwardPassDenseLayer(input: current_results as! Matrix, activation_function: self.activation_functions[layerIndex], weights: self.weights[layerIndex - filterIndex])
                print("Dense")
            case .Recurrent:
                print("Recurrent")
            }
            layerIndex += 1
        }
        return current_results
    }
    
    func forwardPassDenseLayer(input : Matrix, activation_function : Activation_Function, weights : Matrix) -> Matrix {
        let dot_results = multiply(input, weights)
        let hidden_layer_results = activation_function.run(input: dot_results)
        return hidden_layer_results
    }

    func convolution(input_data : [Matrix], filters : [[Matrix]], stride : Int, activation_fuction : Activation_Function) -> [Matrix] {
        let filter_height = filters[0][0].count
        let image_height = input_data[0].count
        let outputWidthAndHeight = ((image_height - filter_height) / stride) + 1
        var output = [Matrix](repeating: Array(repeating: Array(repeating: 0.0, count: outputWidthAndHeight), count: outputWidthAndHeight), count: filters.count)
        var filter_count = 0
        for current_filter in filters {
            var current_y = 0
            var out_y = 0
            while current_y + filter_height <= image_height {
                var current_x = 0
                var out_x = 0
                while current_x + filter_height <= image_height {
                    let s = return3DMatrixSlice(value: input_data, startIndexX: current_x, startIndexY: current_y, slice: filter_height)
                    let r = multiply(current_filter, s)
                    let rr = sum(value: r)
                    output[filter_count][out_y][out_x] = max(0.0, rr)
                    current_x += stride
                    out_x += 1
                }
                current_y += stride
                out_y += 1
            }
            filter_count += 1
        }
        return output
    }
    
    func maxpool(input_data : [Matrix], window_size : Int, stride : Int) -> [Matrix] {
        let number_of_channels = input_data.count
        let input_height = input_data[0].count
        let height = ((input_height - window_size) / stride) + 1
        
        var downsampled = [Matrix](repeating: Array(repeating: Array(repeating: 0.0, count: height), count: height), count: number_of_channels)
        var channel_count = 0
        for _ in input_data {
            var current_y = 0
            var out_y = 0
            while current_y + window_size <= input_height {
                var current_x = 0
                var out_x = 0
                while current_x + window_size <= input_height {
                    let input_slice = return2DMatrixSlice(value: input_data[channel_count], startIndexX: current_x, startIndexY: current_y, slice: window_size)
                    downsampled[channel_count][out_y][out_x] = max(value: input_slice)
                    current_x += stride
                    out_x += 1
                }
                current_y += stride
                out_y += 1
            }
            channel_count += 1
        }
        return downsampled
    }
    
    func flatten(input_data : [Matrix]) -> [Double] {
        var returnArray : [Double] = []
        for channel in input_data {
            for row in channel {
                for column in row {
                    returnArray.append(column)
                }
            }
        }
        return returnArray
    }
}


