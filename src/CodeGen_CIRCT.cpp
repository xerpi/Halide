#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/Transforms/Passes.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <circt/Conversion/CalyxToFSM.h>
#include <circt/Conversion/CalyxToHW.h>
#include <circt/Conversion/SCFToCalyx.h>
#include <circt/Dialect/Calyx/CalyxEmitter.h>
#include <circt/Dialect/Calyx/CalyxPasses.h>

#include "CodeGen_CIRCT.h"
#include "CodeGen_Internal.h"
#include "CodeGen_MLIR.h"
#include "Debug.h"
#include "IROperator.h"

namespace Halide {

namespace Internal {

bool CodeGen_CIRCT_Dev::compile(mlir::LocationAttr &loc, mlir::ModuleOp &mlir_module, Stmt stmt, const std::string &name,
                                const std::vector<DeviceArgument> &args, std::string &calyxOutput, int axiDataWidth) {
    debug(1) << "[Generating CIRCT for kernel '" << name << "']\n";

    CodeGen_MLIR cgMlir;
    debug(1) << "[Generating MLIR for kernel '" << name << "']\n";
    bool ret = cgMlir.compile(loc, mlir_module, stmt, name, args);

    debug(1) << "Original MLIR:\n";
    mlir_module.dump();

    internal_assert(ret) << "Generation of MLIR for kernel '" << name << "' failed!\n";

    // Add port names to the function arguments (for Calyx)
    mlir_module.walk([&](mlir::func::FuncOp funcOp) {
        if (funcOp.getName() == name) {
            mlir::ImplicitLocOpBuilder builder = mlir::ImplicitLocOpBuilder::atBlockEnd(loc, mlir_module.getBody());
            for (auto [index, arg] : llvm::enumerate(args))
                funcOp.setArgAttr(index, circt::scfToCalyx::sPortNameAttr, builder.getStringAttr(arg.name));

            // Memrefs are at the end
            size_t index = 0;
            for (const auto &arg : args) {
                if (arg.is_buffer) {
                    funcOp.setArgAttr(args.size() + index, circt::scfToCalyx::sPortNameAttr, builder.getStringAttr(arg.name + ".buffer"));
                    funcOp.setArgAttr(args.size() + index, circt::scfToCalyx::sSequentialReads, builder.getBoolAttr(true));
                    funcOp.setArgAttr(args.size() + index, circt::scfToCalyx::sDataBusWidth, builder.getI32IntegerAttr(axiDataWidth));
                    index++;
                }
            }
        }
    });

    // Set "sequential reads" attribute to all allocated memories
    mlir_module.walk([&](mlir::memref::AllocOp allocOp) {
        mlir::ImplicitLocOpBuilder builder = mlir::ImplicitLocOpBuilder::atBlockEnd(loc, mlir_module.getBody());
        allocOp->setAttr(circt::scfToCalyx::sSequentialReads, builder.getBoolAttr(true));
    });

    // Create and run passes
    debug(1) << "[SCF to Calyx] Start.\n";
    mlir::PassManager pmSCFToCalyx(mlir_module.getContext());
    pmSCFToCalyx.enableIRPrinting();
    pmSCFToCalyx.addPass(mlir::createForToWhileLoopPass());
    pmSCFToCalyx.addPass(mlir::createCanonicalizerPass());
    pmSCFToCalyx.addPass(circt::createSCFToCalyxPass());
    pmSCFToCalyx.addPass(mlir::createCanonicalizerPass());

    auto pmSCFToCalyxRunResult = pmSCFToCalyx.run(mlir_module);
    debug(1) << "[SCF to Calyx] Result: " << pmSCFToCalyxRunResult.succeeded() << "\n";
    if (!pmSCFToCalyxRunResult.succeeded()) {
        debug(1) << "[SCF to Calyx] MLIR:\n";
        mlir_module.dump();
        return false;
    }

    // Generate Calyx (for debugging purposes)
    if (0) {
        llvm::raw_string_ostream os(calyxOutput);
        debug(1) << "[Exporting Calyx]\n";
        auto exportCalyxResult = circt::calyx::exportCalyx(mlir_module, os);
        debug(1) << "[Export Calyx] Result: " << exportCalyxResult.succeeded() << "\n";
    }

    debug(1) << "[Calyx to FSM] Start.\n";
    mlir::PassManager pmCalyxToFSM(mlir_module.getContext());
    pmCalyxToFSM.enableIRPrinting();
    pmCalyxToFSM.nest<circt::calyx::ComponentOp>().addPass(circt::calyx::createRemoveCombGroupsPass());
    pmCalyxToFSM.addPass(mlir::createCanonicalizerPass());
    pmCalyxToFSM.nest<circt::calyx::ComponentOp>().addPass(circt::createCalyxToFSMPass());
    pmCalyxToFSM.addPass(mlir::createCanonicalizerPass());
    pmCalyxToFSM.nest<circt::calyx::ComponentOp>().addPass(circt::createMaterializeCalyxToFSMPass());
    pmCalyxToFSM.addPass(mlir::createCanonicalizerPass());
    pmCalyxToFSM.nest<circt::calyx::ComponentOp>().addPass(circt::createRemoveGroupsFromFSMPass());
    pmCalyxToFSM.nest<circt::calyx::ComponentOp>().addPass(circt::calyx::createClkInsertionPass());
    pmCalyxToFSM.nest<circt::calyx::ComponentOp>().addPass(circt::calyx::createResetInsertionPass());
    pmCalyxToFSM.addPass(mlir::createCanonicalizerPass());
    pmCalyxToFSM.addPass(circt::createCalyxToHWPass());
    pmCalyxToFSM.addPass(mlir::createCanonicalizerPass());

    auto pmCalyxToFSMRunResult = pmCalyxToFSM.run(mlir_module);
    debug(1) << "[Calyx to FSM] Result: " << pmCalyxToFSMRunResult.succeeded() << "\n";
    if (!pmCalyxToFSMRunResult.succeeded()) {
        debug(1) << "[Calyx to FSM] MLIR:\n";
        mlir_module.dump();
        return false;
    }

    debug(1) << "[CIRCT for kernel '" << name << " generated successfully']\n";
    return true;
}

}  // namespace Internal
}  // namespace Halide
