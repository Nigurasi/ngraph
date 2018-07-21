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
#include <iostream>

#include "ngraph/op/add.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"
#include "ngraph/runtime/interpreter/exec_node.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class AddExec;
        }
    }
}

class ngraph::runtime::interpreter::AddExec : public ExecNode
{
public:
    static ExecNode create(const ngraph::Node* node)
    {
        std::cout << "create Add" << std::endl;
        return AddExec(node);
    }

    AddExec(const ngraph::Node* node)
        : m_node{dynamic_cast<const ngraph::op::Add*>(node)}
    {
        (void)m_node; // Silence compiler warning

        std::cout << "Add ctor" << std::endl;
    }

    template <typename T>
    void execute(const std::vector<std::shared_ptr<HostTensorView>>& out,
                 const std::vector<std::shared_ptr<HostTensorView>>& args)
    {
        std::cout << "execute Add" << std::endl;
    }

private:
    const ngraph::op::Add* m_node;
};
