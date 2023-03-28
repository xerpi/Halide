#ifndef HALIDE_CODEGEN_CIRCT_XILINX_DEV_H
#define HALIDE_CODEGEN_CIRCT_XILINX_DEV_H

/** \file
 * Defines the code-generator for producing CIRCT MLIR code targeting Xilinx devices
 */

#include <memory>

namespace Halide {

struct Target;

namespace Internal {

struct CodeGen_Accelerator_Dev;

std::unique_ptr<CodeGen_Accelerator_Dev> new_CodeGen_CIRCT_Xilinx_Dev(const Target &target);

}  // namespace Internal
}  // namespace Halide

#endif
