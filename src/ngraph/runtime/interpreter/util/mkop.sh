#!/bin/bash

function emit {
    header=$(<header.txt)
    echo ${1,,}
    h2="${header//CAMEL_CASE_NAME/$1}"
    h2="${h2//LOWER_CASE_NAME/$2}"
    echo "${h2}" > /mnt/c/dev/ngraph/src/ngraph/runtime/interpreter/op/$2.hpp
}

emit Abs abs
emit Acos acos
emit Add add
emit AllReduce allreduce
emit And and
emit Asin asin
emit Atan atan
emit AvgPool avg_pool
emit BatchNorm batch_norm
emit Broadcast broadcast
emit Ceiling ceiling
emit Concat concat
emit Constant constant
emit Convert convert
emit Convolution convolution
emit Cos cos
emit Cosh cosh
emit Divide divide
emit Dot dot
emit Equal equal
emit Exp exp
emit Floor floor
emit FunctionCall function_call
emit GetOutputElement get_output_element
emit Greater greater
emit GreaterEq greater_eq
emit Less less
emit LessEq less_eq
emit Log log
emit Max max
emit Maximum maximum
emit MaxPool max_pool
emit Min min
emit Minimum minimum
emit Multiply multiply
emit Negative negative
emit Not not
emit NotEqual not_equal
emit OneHot one_hot
emit Or or
emit Pad pad
emit Parameter parameter
emit Power power
emit Product product
emit Reduce reduce
emit ReduceWindow reduce_window
emit Relu relu
emit ReluBackprop relubackprop
emit Remainder remainder
emit ReplaceSlice replace_slice
emit Reshape reshape
emit Result result
emit Reverse reverse
emit ReverseSequence reverse_sequence
emit Select select
emit SelectAndScatter select_and_scatter
emit Sigmoid sigmoid
emit Sign sign
emit Sin sin
emit Sinh sinh
emit Slice slice
emit Softmax softmax
emit Sqrt sqrt
emit StopGradient stop_gradient
emit Subtract subtract
emit Sum sum
emit Tan tan
emit Tanh tanh