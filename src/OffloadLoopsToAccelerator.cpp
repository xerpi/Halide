#include <memory>

#include "Closure.h"
#include "CodeGen_Accelerator_Dev.h"
#include "CodeGen_CIRCT_Xilinx_Dev.h"
#include "ExprUsesVar.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "IRPrinter.h"
#include "InjectHostDevBufferCopies.h"
#include "OffloadLoopsToAccelerator.h"
#include "Util.h"

namespace Halide {
namespace Internal {

using std::map;
using std::string;
using std::unique_ptr;
using std::vector;

namespace {

class InjectAcceleratorOffload : public IRMutator {
    /** Child code generator for device kernels. */
    map<DeviceAPI, unique_ptr<CodeGen_Accelerator_Dev>> cgdev;

    map<string, bool> state_needed;

    const Target &target;

    using IRMutator::visit;

    Stmt visit(const For *loop) override {
        // We're in the loop over outermost block dimension
        debug(2) << "Creating kernel for loop " << loop->name << "\n";

        // Compute a closure over the state passed into the kernel
        HostClosure c;
        c.include(loop, loop->name);

        // Determine the arguments that must be passed into the halide function
        vector<DeviceArgument> closure_args = c.arguments();

        // Compile the kernel
        string kernel_name = unique_name("kernel_" + loop->name);

        CodeGen_Accelerator_Dev *acc_codegen = cgdev[target.get_required_device_api()].get();
        user_assert(acc_codegen != nullptr)
            << "Invalid device api " << int(target.get_required_device_api())
            << " . Target " << target.to_string() << "\n";

        debug(1) << "Kernel " << kernel_name << " has " << closure_args.size() << " arguments:\n";
        int i = 0;
        for (const auto &arg : closure_args) {
            static const char *const type_code_names[] = {
                "int",
                "uint",
                "float",
                "handle",
                "bfloat",
            };
            debug(1) << "\targ[" << i << "]: name: " << arg.name << ",  "
                     << (arg.is_buffer ? "buffer" : "scalar") << ", "
                     << "type: " << type_code_names[arg.type.code()] << "\n";
            i++;
        }

        acc_codegen->add_kernel(loop, kernel_name, closure_args);

        // get the actual name of the generated kernel for this loop
        kernel_name = acc_codegen->get_current_kernel_name();
        debug(2) << "Compiled kernel \"" << kernel_name << "\"\n";

        // Generate the kernel launch arguments lists
        vector<Expr> args, arg_types, arg_is_buffer;
        for (const auto &arg : closure_args) {
            Expr val;
            if (arg.is_buffer) {
                val = Variable::make(Handle(), arg.name + ".buffer");
            } else {
                val = Variable::make(arg.type, arg.name);
                val = Call::make(type_of<void *>(), Call::make_struct, {val}, Call::Intrinsic);
            }
            args.emplace_back(val);
            arg_types.emplace_back(((halide_type_t)arg.type).as_u32());
            arg_is_buffer.emplace_back(cast<uint8_t>(arg.is_buffer));
        }

        // nullptr-terminate the lists
        args.emplace_back(reinterpret(Handle(), cast<uint64_t>(0)));
        internal_assert(sizeof(halide_type_t) == sizeof(uint32_t));
        arg_types.emplace_back(cast<uint32_t>(0));
        arg_is_buffer.emplace_back(cast<uint8_t>(0));

        string api_unique_name = acc_codegen->api_unique_name();
        vector<Expr> run_args = {
            get_state_var(api_unique_name, state_needed[api_unique_name]),
            kernel_name,
            Call::make(Handle(), Call::make_struct, arg_types, Call::Intrinsic),
            Call::make(Handle(), Call::make_struct, args, Call::Intrinsic),
            Call::make(Handle(), Call::make_struct, arg_is_buffer, Call::Intrinsic),
        };
        return call_extern_and_assert("halide_" + api_unique_name + "_run", run_args);
    }

public:
    InjectAcceleratorOffload(const Target &target)
        : target(target) {
        if (target.has_feature(Target::CIRCT)) {
            cgdev[DeviceAPI::XRT] = new_CodeGen_CIRCT_Xilinx_Dev(target);
        }

        internal_assert(!cgdev.empty()) << "Requested unknown accelerator target: " << target.to_string() << "\n";
    }

    Stmt inject(const Stmt &s) {
        // Create a new module for all of the kernels we find in this function.
        for (auto &i : cgdev) {
            i.second->init_module();
        }

        Stmt result = mutate(s);

        for (auto &i : cgdev) {
            string api_unique_name = i.second->api_unique_name();

            Expr state_ptr = make_state_var(api_unique_name);
            Expr state_ptr_var = Variable::make(type_of<void *>(), api_unique_name);

            debug(2) << "Generating init_kernels for " << api_unique_name << "\n";
            string kernel_n = i.second->get_current_kernel_name();
            Expr kernel_name = make_string_ptr(kernel_n, api_unique_name + "_kernel_name");

            string init_kernels_name = "halide_" + api_unique_name + "_initialize_kernels";
            vector<Expr> init_args = {state_ptr_var, kernel_name};
            Stmt init_kernels = call_extern_and_assert(init_kernels_name, init_args);

            string destructor_name = "halide_" + api_unique_name + "_finalize_kernels";
            vector<Expr> finalize_args = {Expr(destructor_name), get_state_var(api_unique_name, state_needed[api_unique_name])};
            Stmt register_destructor = Evaluate::make(
                Call::make(Handle(), Call::register_destructor, finalize_args, Call::Intrinsic));

            result = LetStmt::make(api_unique_name, state_ptr, Block::make({init_kernels, register_destructor, result}));
        }
        return result;
    }
};

}  // namespace

Stmt inject_accelerator_offload(const Stmt &s, const Target &host_target) {
    return InjectAcceleratorOffload(host_target).inject(s);
}

}  // namespace Internal
}  // namespace Halide
