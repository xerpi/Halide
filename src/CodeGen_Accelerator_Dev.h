#ifndef HALIDE_CODEGEN_ACCELERATOR_DEV_H
#define HALIDE_CODEGEN_ACCELERATOR_DEV_H

/** \file
 * Defines the code-generator interface for producing accelerator device code
 */
#include <string>
#include <vector>

#include "DeviceArgument.h"
#include "Expr.h"

namespace Halide {
namespace Internal {

/** A code generator that emits accelerator code from a given Halide stmt. */
struct CodeGen_Accelerator_Dev {
    /** Compile a accelerator kernel into the module. This may be called many times
     * with different kernels, which will all be accumulated into a single
     * source module shared by a given Halide pipeline. */
    virtual void add_kernel(Stmt stmt,
                            const std::string &name,
                            const std::vector<DeviceArgument> &args) = 0;

    /** (Re)initialize the accelerator kernel module. This is separate from compile,
     * since a accelerator device module will often have many kernels compiled into it
     * for a single pipeline. */
    virtual void init_module() = 0;

    virtual std::string get_current_kernel_name() = 0;

    /** This routine returns the accelerator API name that is combined into
     *  runtime routine names to ensure each accelerator API has a unique
     *  name.
     */
    virtual std::string api_unique_name() = 0;

    /** Returns the specified name transformed by the variable naming rules
     * for the accelerator language backend. Used to determine the name of a parameter
     * during host codegen. */
    virtual std::string print_accelerator_name(const std::string &name) = 0;
};

}  // namespace Internal
}  // namespace Halide

#endif
