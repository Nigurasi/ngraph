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

#pragma once

#include <memory>
#include <vector>

#include "ngraph/op/negative.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"
#include "ngraph/runtime/interpreter/exec_node.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class NegativeExec;
        }
    }
}

class ngraph::runtime::interpreter::NegativeExec : public ExecNode
{
public:
    static ExecNode create(const ngraph::Node* node)
    {
        return NegativeExec(node);
    }

    NegativeExec(const ngraph::Node* node)
        : m_node{dynamic_cast<const ngraph::op::Negative*>(node)}
    {
        (void)m_node; // Silence compiler warning
    }

    template <typename T>
    void execute(const std::vector<std::shared_ptr<HostTensorView>>& out,
                 const std::vector<std::shared_ptr<HostTensorView>>& args)
    {
    }

private:
    const ngraph::op::Negative* m_node;
};
