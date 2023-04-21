#ifndef HALIDE_CODEGEN_CIRCT_DEV_H
#define HALIDE_CODEGEN_CIRCT_DEV_H

/** \file
 * Defines the code-generator for producing CIRCT code
 */

#include "DeviceArgument.h"
#include "IRVisitor.h"
#include "Scope.h"

#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/MLIRContext.h>

#include <string>
#include <vector>

namespace Halide {

struct Target;

namespace Internal {

class CodeGen_CIRCT_Dev {
public:
    bool compile(mlir::LocationAttr &loc, mlir::ModuleOp &mlir_module, Stmt stmt,
                 const std::string &name, const std::vector<DeviceArgument> &args,
                 std::string &calyxOutput, int axiDataWidth);
};

}  // namespace Internal
}  // namespace Halide

#endif
