/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the \"License\");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an \"AS IS\" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/runtime/interpreter/exec_node.hpp"
#include "op/abs.hpp"
#include "op/acos.hpp"
#include "op/add.hpp"
#include "op/allreduce.hpp"
#include "op/and.hpp"
#include "op/asin.hpp"
#include "op/atan.hpp"
#include "op/avg_pool.hpp"
#include "op/batch_norm.hpp"
#include "op/broadcast.hpp"
#include "op/ceiling.hpp"
#include "op/concat.hpp"
#include "op/constant.hpp"
#include "op/convert.hpp"
#include "op/convolution.hpp"
#include "op/cos.hpp"
#include "op/cosh.hpp"
#include "op/divide.hpp"
#include "op/dot.hpp"
#include "op/equal.hpp"
#include "op/exp.hpp"
#include "op/floor.hpp"
#include "op/function_call.hpp"
#include "op/get_output_element.hpp"
#include "op/greater.hpp"
#include "op/greater_eq.hpp"
#include "op/less.hpp"
#include "op/less_eq.hpp"
#include "op/log.hpp"
#include "op/max.hpp"
#include "op/max_pool.hpp"
#include "op/maximum.hpp"
#include "op/min.hpp"
#include "op/minimum.hpp"
#include "op/multiply.hpp"
#include "op/negative.hpp"
#include "op/not.hpp"
#include "op/not_equal.hpp"
#include "op/one_hot.hpp"
#include "op/or.hpp"
#include "op/pad.hpp"
#include "op/parameter.hpp"
#include "op/power.hpp"
#include "op/product.hpp"
#include "op/reduce.hpp"
#include "op/reduce_window.hpp"
#include "op/relu.hpp"
#include "op/remainder.hpp"
#include "op/replace_slice.hpp"
#include "op/reshape.hpp"
#include "op/result.hpp"
#include "op/reverse.hpp"
#include "op/reverse_sequence.hpp"
#include "op/select.hpp"
#include "op/select_and_scatter.hpp"
#include "op/sigmoid.hpp"
#include "op/sign.hpp"
#include "op/sin.hpp"
#include "op/sinh.hpp"
#include "op/slice.hpp"
#include "op/softmax.hpp"
#include "op/sqrt.hpp"
#include "op/stop_gradient.hpp"
#include "op/subtract.hpp"
#include "op/sum.hpp"
#include "op/tan.hpp"
#include "op/tanh.hpp"

using namespace std;
using namespace ngraph;

#define TID(a) type_index(typeid(a))

unordered_map<type_index, ngraph::runtime::interpreter::ExecNode::create_t>
    runtime::interpreter::ExecNode::s_list = {
        {TID(op::Abs), &runtime::interpreter::AbsExec::create},
        {TID(op::Acos), &runtime::interpreter::AcosExec::create},
        {TID(op::Add), &runtime::interpreter::AddExec::create},
        {TID(op::AllReduce), &runtime::interpreter::AllReduceExec::create},
        {TID(op::And), &runtime::interpreter::AndExec::create},
        {TID(op::Asin), &runtime::interpreter::AsinExec::create},
        {TID(op::Atan), &runtime::interpreter::AtanExec::create},
        {TID(op::AvgPool), &runtime::interpreter::AvgPoolExec::create},
        {TID(op::BatchNorm), &runtime::interpreter::BatchNormExec::create},
        {TID(op::Broadcast), &runtime::interpreter::BroadcastExec::create},
        {TID(op::Ceiling), &runtime::interpreter::CeilingExec::create},
        {TID(op::Concat), &runtime::interpreter::ConcatExec::create},
        {TID(op::Constant), &runtime::interpreter::ConstantExec::create},
        {TID(op::Convert), &runtime::interpreter::ConvertExec::create},
        {TID(op::Convolution), &runtime::interpreter::ConvolutionExec::create},
        {TID(op::Cos), &runtime::interpreter::CosExec::create},
        {TID(op::Cosh), &runtime::interpreter::CoshExec::create},
        {TID(op::Divide), &runtime::interpreter::DivideExec::create},
        {TID(op::Dot), &runtime::interpreter::DotExec::create},
        {TID(op::Equal), &runtime::interpreter::EqualExec::create},
        {TID(op::Exp), &runtime::interpreter::ExpExec::create},
        {TID(op::Floor), &runtime::interpreter::FloorExec::create},
        {TID(op::FunctionCall), &runtime::interpreter::FunctionCallExec::create},
        {TID(op::GetOutputElement), &runtime::interpreter::GetOutputElementExec::create},
        {TID(op::Greater), &runtime::interpreter::GreaterExec::create},
        {TID(op::GreaterEq), &runtime::interpreter::GreaterEqExec::create},
        {TID(op::Less), &runtime::interpreter::LessExec::create},
        {TID(op::LessEq), &runtime::interpreter::LessEqExec::create},
        {TID(op::Log), &runtime::interpreter::LogExec::create},
        {TID(op::Max), &runtime::interpreter::MaxExec::create},
        {TID(op::Maximum), &runtime::interpreter::MaximumExec::create},
        {TID(op::MaxPool), &runtime::interpreter::MaxPoolExec::create},
        {TID(op::Min), &runtime::interpreter::MinExec::create},
        {TID(op::Minimum), &runtime::interpreter::MinimumExec::create},
        {TID(op::Multiply), &runtime::interpreter::MultiplyExec::create},
        {TID(op::Negative), &runtime::interpreter::NegativeExec::create},
        {TID(op::Not), &runtime::interpreter::NotExec::create},
        {TID(op::NotEqual), &runtime::interpreter::NotEqualExec::create},
        {TID(op::OneHot), &runtime::interpreter::OneHotExec::create},
        {TID(op::Or), &runtime::interpreter::OrExec::create},
        {TID(op::Pad), &runtime::interpreter::PadExec::create},
        {TID(op::Parameter), &runtime::interpreter::ParameterExec::create},
        {TID(op::Power), &runtime::interpreter::PowerExec::create},
        {TID(op::Product), &runtime::interpreter::ProductExec::create},
        {TID(op::Reduce), &runtime::interpreter::ReduceExec::create},
        {TID(op::ReduceWindow), &runtime::interpreter::ReduceWindowExec::create},
        {TID(op::Relu), &runtime::interpreter::ReluExec::create},
        {TID(op::Remainder), &runtime::interpreter::RemainderExec::create},
        {TID(op::ReplaceSlice), &runtime::interpreter::ReplaceSliceExec::create},
        {TID(op::Reshape), &runtime::interpreter::ReshapeExec::create},
        {TID(op::Result), &runtime::interpreter::ResultExec::create},
        {TID(op::Reverse), &runtime::interpreter::ReverseExec::create},
        {TID(op::ReverseSequence), &runtime::interpreter::ReverseSequenceExec::create},
        {TID(op::Select), &runtime::interpreter::SelectExec::create},
        {TID(op::SelectAndScatter), &runtime::interpreter::SelectAndScatterExec::create},
        {TID(op::Sigmoid), &runtime::interpreter::SigmoidExec::create},
        {TID(op::Sign), &runtime::interpreter::SignExec::create},
        {TID(op::Sin), &runtime::interpreter::SinExec::create},
        {TID(op::Sinh), &runtime::interpreter::SinhExec::create},
        {TID(op::Slice), &runtime::interpreter::SliceExec::create},
        {TID(op::Softmax), &runtime::interpreter::SoftmaxExec::create},
        {TID(op::Sqrt), &runtime::interpreter::SqrtExec::create},
        {TID(op::StopGradient), &runtime::interpreter::StopGradientExec::create},
        {TID(op::Subtract), &runtime::interpreter::SubtractExec::create},
        {TID(op::Sum), &runtime::interpreter::SumExec::create},
        {TID(op::Tan), &runtime::interpreter::TanExec::create},
        {TID(op::Tanh), &runtime::interpreter::TanhExec::create}};

runtime::interpreter::ExecNode runtime::interpreter::ExecNode::create_exec(const Node* node)
{
    const Node& n = *node;
    return s_list.at(type_index(typeid(n)))(node);
}
