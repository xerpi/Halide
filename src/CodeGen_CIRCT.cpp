#include <fstream>
#include <vector>

#include <tinyxml2.h>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/ToolOutputFile.h>

#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Passes.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Transforms/Passes.h>

#include <circt/Conversion/CalyxToFSM.h>
#include <circt/Conversion/CalyxToHW.h>
#include <circt/Conversion/ExportVerilog.h>
#include <circt/Conversion/FSMToSV.h>
#include <circt/Conversion/SCFToCalyx.h>
#include <circt/Dialect/Calyx/CalyxDialect.h>
#include <circt/Dialect/Calyx/CalyxEmitter.h>
#include <circt/Dialect/Calyx/CalyxOps.h>
#include <circt/Dialect/Calyx/CalyxPasses.h>
#include <circt/Dialect/FSM/FSMDialect.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/HW/HWPasses.h>
#include <circt/Dialect/SV/SVAttributes.h>
#include <circt/Dialect/SV/SVOps.h>
#include <circt/Dialect/Seq/SeqDialect.h>
#include <circt/Dialect/Seq/SeqOps.h>
#include <circt/Dialect/Seq/SeqPasses.h>
#include <circt/Support/LoweringOptions.h>

#include "CodeGen_CIRCT.h"
#include "CodeGen_Internal.h"
#include "Debug.h"
#include "IROperator.h"
#include "Util.h"

using namespace tinyxml2;

namespace Halide {

namespace Internal {

CodeGen_CIRCT::CodeGen_CIRCT() {
    mlir_context.loadDialect<circt::calyx::CalyxDialect>();
    mlir_context.loadDialect<circt::comb::CombDialect>();
    mlir_context.loadDialect<circt::fsm::FSMDialect>();
    mlir_context.loadDialect<circt::hw::HWDialect>();
    mlir_context.loadDialect<circt::seq::SeqDialect>();
    mlir_context.loadDialect<mlir::arith::ArithDialect>();
    mlir_context.loadDialect<mlir::cf::ControlFlowDialect>();
    mlir_context.loadDialect<mlir::func::FuncDialect>();
    mlir_context.loadDialect<mlir::memref::MemRefDialect>();
    mlir_context.loadDialect<mlir::scf::SCFDialect>();
}

void CodeGen_CIRCT::compile(const Module &module) {
    debug(1) << "Generating CIRCT MLIR IR for module " << module.name() << "\n";

    const std::string outputDir = "generated_" + module.name();

    circt::LoweringOptions opts;
    opts.emittedLineLength = 200;
    opts.disallowPortDeclSharing = true;
    opts.printDebugInfo = true;
    opts.wireSpillingNamehintTermLimit = 1000;
    opts.maximumNumberOfTermsPerExpression = 1000;
    opts.emitBindComments = true;

    mlir::LocationAttr loc = mlir::UnknownLoc::get(&mlir_context);
    mlir::ModuleOp mlir_module = mlir::ModuleOp::create(loc, {});
    opts.setAsAttribute(mlir_module);
    mlir::ImplicitLocOpBuilder builder = mlir::ImplicitLocOpBuilder::atBlockEnd(loc, mlir_module.getBody());

    const std::vector<Internal::LoweredFunc> &functions = module.functions();
    assert(functions.size() == 1);
    const auto &function = functions[0];

    std::cout << "Generating CIRCT MLIR IR for function " << function.name << std::endl;

    mlir::SmallVector<mlir::Type> inputs;
    std::vector<std::string> inputNames;
    mlir::SmallVector<mlir::Type> results;
    mlir::SmallVector<mlir::NamedAttribute> attrs;
    mlir::SmallVector<mlir::DictionaryAttr> argAttrs;

    FlattenedKernelArgs flattenedKernelArgs;
    flattenKernelArguments(function.args, flattenedKernelArgs);

    for (const auto &arg : flattenedKernelArgs) {
        inputs.push_back(builder.getIntegerType(arg.getHWBits()));
        inputNames.push_back(arg.name);
        argAttrs.push_back(builder.getDictionaryAttr(
            builder.getNamedAttr(circt::scfToCalyx::sPortNameAttr, builder.getStringAttr(arg.name))));
    }

    for (const auto &arg : flattenedKernelArgs) {
        if (arg.isPointer) {
            // Add memref to the arguments. Treat buffers as 1D
            inputs.push_back(mlir::MemRefType::get({0}, builder.getIntegerType(arg.type.bits())));
            inputNames.push_back(arg.name + ".buffer");
            argAttrs.push_back(builder.getDictionaryAttr(
                builder.getNamedAttr(circt::scfToCalyx::sSequentialReads, builder.getBoolAttr(true))));
        }
    }

    mlir::FunctionType functionType = builder.getFunctionType(inputs, results);
    mlir::func::FuncOp functionOp = builder.create<mlir::func::FuncOp>(builder.getStringAttr(function.name), functionType, attrs, argAttrs);
    builder.setInsertionPointToStart(functionOp.addEntryBlock());

    CodeGen_CIRCT::Visitor visitor(builder, inputNames);
    function.body.accept(&visitor);

    builder.create<mlir::func::ReturnOp>();

    // Print MLIR before running passes
    std::cout << "Original MLIR" << std::endl;
    mlir_module.dump();

    // Verify module (before running passes)
    auto moduleVerifyResult = mlir::verify(mlir_module);
    std::cout << "Module verify (before passess) result: " << moduleVerifyResult.succeeded() << std::endl;
    internal_assert(moduleVerifyResult.succeeded());

    // Create and run passes
    std::cout << "[SCF to Calyx] Start." << std::endl;
    mlir::PassManager pmSCFToCalyx(mlir_module.getContext());
    pmSCFToCalyx.addPass(mlir::createForToWhileLoopPass());
    pmSCFToCalyx.addPass(mlir::createCanonicalizerPass());
    pmSCFToCalyx.addPass(circt::createSCFToCalyxPass());
    pmSCFToCalyx.addPass(mlir::createCanonicalizerPass());

    auto pmSCFToCalyxRunResult = pmSCFToCalyx.run(mlir_module);
    std::cout << "[SCF to Calyx] Result: " << pmSCFToCalyxRunResult.succeeded() << std::endl;
    if (!pmSCFToCalyxRunResult.succeeded()) {
        std::cout << "[SCF to Calyx] MLIR:" << std::endl;
        mlir_module.dump();
    }
    internal_assert(pmSCFToCalyxRunResult.succeeded());

    // Create output directory (if it doesn't exist)
    llvm::sys::fs::create_directories(outputDir);

    // Emit Calyx
    if (pmSCFToCalyxRunResult.succeeded()) {
        std::string str;
        llvm::raw_string_ostream os(str);
        std::cout << "[Exporting Calyx]" << std::endl;
        auto exportVerilogResult = circt::calyx::exportCalyx(mlir_module, os);
        std::cout << "[Export Calyx] Result: " << exportVerilogResult.succeeded() << std::endl;

        auto output = mlir::openOutputFile(outputDir + "/" + function.name + ".futil");
        if (output) {
            output->os() << str;
            output->keep();
        }
    }

    std::cout << "[Calyx to FSM] Start." << std::endl;
    mlir::PassManager pmCalyxToFSM(mlir_module.getContext());
    pmCalyxToFSM.nest<circt::calyx::ComponentOp>().addPass(circt::calyx::createRemoveCombGroupsPass());
    pmCalyxToFSM.addPass(mlir::createCanonicalizerPass());
    pmCalyxToFSM.nest<circt::calyx::ComponentOp>().addPass(circt::createCalyxToFSMPass());
    pmCalyxToFSM.addPass(mlir::createCanonicalizerPass());
    pmCalyxToFSM.nest<circt::calyx::ComponentOp>().addPass(circt::createMaterializeCalyxToFSMPass());
    pmCalyxToFSM.addPass(mlir::createCanonicalizerPass());
    pmCalyxToFSM.nest<circt::calyx::ComponentOp>().addPass(circt::createRemoveGroupsFromFSMPass());
    pmCalyxToFSM.addPass(mlir::createCanonicalizerPass());
    pmCalyxToFSM.addPass(circt::createCalyxToHWPass());
    pmCalyxToFSM.addPass(mlir::createCanonicalizerPass());

    auto pmCalyxToFSMRunResult = pmCalyxToFSM.run(mlir_module);
    std::cout << "[Calyx to FSM] Result: " << pmCalyxToFSMRunResult.succeeded() << std::endl;
    if (!pmCalyxToFSMRunResult.succeeded()) {
        std::cout << "[Calyx to FSM] MLIR:" << std::endl;
        mlir_module.dump();
    }
    internal_assert(pmCalyxToFSMRunResult.succeeded());

    // Create Calyx external memory to AXI interface
    builder = mlir::ImplicitLocOpBuilder::atBlockEnd(loc, mlir_module.getBody());
    std::cout << "[Adding CalyxExtMemToAxi]" << std::endl;
    generateCalyxExtMemToAxi(builder);

    // Add AXI control
    builder = mlir::ImplicitLocOpBuilder::atBlockEnd(loc, mlir_module.getBody());
    std::cout << "[Adding Control AXI]" << std::endl;
    generateControlAxi(builder, flattenedKernelArgs);

    // Add toplevel
    builder = mlir::ImplicitLocOpBuilder::atBlockEnd(loc, mlir_module.getBody());
    std::cout << "[Adding Toplevel]" << std::endl;
    generateToplevel(builder, function.name, flattenedKernelArgs);

    std::cout << "[FSM to SV] Start." << std::endl;
    mlir::PassManager pmFSMtoSV(mlir_module.getContext());
    pmFSMtoSV.addPass(circt::createConvertFSMToSVPass());
    pmFSMtoSV.addPass(mlir::createCanonicalizerPass());
    pmFSMtoSV.addPass(circt::seq::createSeqLowerToSVPass());
    pmFSMtoSV.addPass(mlir::createCanonicalizerPass());

    auto pmFSMtoSVRunResult = pmFSMtoSV.run(mlir_module);
    std::cout << "[FSM to SV] Result: " << pmFSMtoSVRunResult.succeeded() << std::endl;
    if (!pmFSMtoSVRunResult.succeeded()) {
        std::cout << "[FSM to SV] MLIR:" << std::endl;
        mlir_module.dump();
    }
    internal_assert(pmFSMtoSVRunResult.succeeded());

    // Emit Verilog
    if (pmFSMtoSVRunResult.succeeded()) {
        std::cout << "[Exporting Verilog]" << std::endl;
        auto exportVerilogResult = circt::exportSplitVerilog(mlir_module, outputDir);
        std::cout << "[Export Verilog] Result: " << exportVerilogResult.succeeded() << std::endl;

        std::cout << "[Generating kernel.xml]" << std::endl;
        auto output = mlir::openOutputFile(outputDir + "/kernel.xml");
        if (output) {
            generateKernelXml(output->os(), function.name, flattenedKernelArgs);
            output->keep();
        }
    }

    std::cout << "Done!" << std::endl;
}

void CodeGen_CIRCT::flattenKernelArguments(const std::vector<LoweredArgument> &inArgs, FlattenedKernelArgs &args) {
    for (const auto &arg : inArgs) {
        static const char *const kind_names[] = {
            "InputScalar",
            "InputBuffer",
            "OutputBuffer",
        };
        static const char *const type_code_names[] = {
            "int",
            "uint",
            "float",
            "handle",
            "bfloat",
        };

        debug(1) << "\t\tArg: " << arg.name << "\n";
        debug(1) << "\t\t\tKind: " << kind_names[arg.kind] << "\n";
        debug(1) << "\t\t\tDimensions: " << int(arg.dimensions) << "\n";
        debug(1) << "\t\t\tType: " << type_code_names[arg.type.code()] << "\n";
        debug(1) << "\t\t\tType bits: " << arg.type.bits() << "\n";
        debug(1) << "\t\t\tType lanes: " << arg.type.lanes() << "\n";

        if (arg.is_scalar() && arg.type.is_int_or_uint()) {
            args.push_back({arg.name, arg.type});
        } else if (arg.is_buffer()) {
            // A pointer to the start of the data in main memory (offset of the buffer into the AXI4 master interface)
            args.push_back({arg.name, arg.type, /*isPointer=*/true});
            // The dimensionality of the buffer
            args.push_back({arg.name + "_dimensions", Int(32)});

            // Buffer dimensions
            for (int i = 0; i < arg.dimensions; i++) {
                args.push_back({arg.name + "_dim_" + std::to_string(i) + "_min", Int(32)});
                args.push_back({arg.name + "_dim_" + std::to_string(i) + "_extent", Int(32)});
                args.push_back({arg.name + "_dim_" + std::to_string(i) + "_stride", Int(32)});
            }
        }
    }
}

void CodeGen_CIRCT::generateCalyxExtMemToAxi(mlir::ImplicitLocOpBuilder &builder) {
    // AXI4 Manager signals
    // araddr, awaddr, wdata and rdata are connected directly outside the FSM module
    mlir::SmallVector<circt::hw::PortInfo> axiSignals;
    portsAddAXI4ManagerSignalsPrefix(builder, AXI_MANAGER_PREFIX, M_AXI_ADDR_WIDTH, M_AXI_DATA_WIDTH, axiSignals);

    mlir::SmallVector<mlir::Type> inputs, outputs;
    mlir::SmallVector<mlir::Attribute> inputNames, outputNames;

    // First add AXI signals
    for (const auto &signal : axiSignals) {
        if (signal.direction == circt::hw::PortDirection::INPUT) {
            inputs.push_back(signal.type);
            inputNames.push_back(builder.getStringAttr(signal.name.str()));
        } else {
            outputs.push_back(signal.type);
            outputNames.push_back(builder.getStringAttr(signal.name.str()));
        }
    }

    // Then add Calyx external memory interface signals
    size_t CalyxExtMemToAxiReadEnSignalIndex = inputs.size();
    inputs.push_back(builder.getI1Type());
    inputNames.push_back(builder.getStringAttr("calyx_read_en"));
    size_t CalyxExtMemToAxiWriteEnSignalIndex = inputs.size();
    inputs.push_back(builder.getI1Type());
    inputNames.push_back(builder.getStringAttr("calyx_write_en"));
    size_t calyxDoneSignalIndex = outputs.size();
    outputs.push_back(builder.getI1Type());
    outputNames.push_back(builder.getStringAttr("calyx_done"));

    mlir::FunctionType fsmFunctionType = builder.getFunctionType(inputs, outputs);
    circt::fsm::MachineOp machineOp = builder.create<circt::fsm::MachineOp>("CalyxExtMemToAxi", "IDLE", fsmFunctionType);
    machineOp.setArgNamesAttr(builder.getArrayAttr(inputNames));
    machineOp.setResNamesAttr(builder.getArrayAttr(outputNames));

    mlir::Region &fsmBody = machineOp.getBody();
    mlir::ImplicitLocOpBuilder fsmBuilder = mlir::ImplicitLocOpBuilder::atBlockEnd(fsmBody.getLoc(), &fsmBody.front());

    mlir::Value value0 = fsmBuilder.create<circt::hw::ConstantOp>(fsmBuilder.getBoolAttr(false));
    mlir::SmallVector<mlir::Value> outputValues(machineOp.getNumResults());

    auto getAxiInputValue = [&](const std::string &name) {
        for (unsigned int i = 0; i < machineOp.getNumArguments(); i++) {
            if (machineOp.getArgName(i) == toFullAxiManagerSignalName(name))
                return machineOp.getArgument(i);
        }
        return mlir::BlockArgument();
    };

    auto getAxiOutputIndex = [&](const std::string &name) {
        unsigned int i = 0;
        for (i = 0; i < machineOp.getNumResults(); i++) {
            if (machineOp.getResName(i) == toFullAxiManagerSignalName(name))
                break;
        }
        return i;
    };

    auto getAxiOutputType = [&](const std::string &name) {
        for (unsigned int i = 0; i < machineOp.getNumResults(); i++) {
            if (machineOp.getResName(i) == toFullAxiManagerSignalName(name))
                return machineOp.getResultTypes()[i];
        }
        return mlir::Type();
    };

    auto createAxiOutputConstantOp = [&](const std::string &name, int64_t value) {
        return builder.create<circt::hw::ConstantOp>(getAxiOutputType(name), value);
    };

    auto setAxiOutputConstant = [&](const std::string &name, int64_t value) {
        outputValues[getAxiOutputIndex(name)] = createAxiOutputConstantOp(name, value);
    };

    // Constant outputs
    setAxiOutputConstant("arlen", 0);  // 1 transfer
    setAxiOutputConstant("awlen", 0);  // 1 transfer
    setAxiOutputConstant("wstrb", 0);
    setAxiOutputConstant("wlast", 1);

    {
        circt::fsm::StateOp idleState = fsmBuilder.create<circt::fsm::StateOp>("IDLE");
        {
            setAxiOutputConstant("arvalid", 0);
            setAxiOutputConstant("rready", 0);
            setAxiOutputConstant("awvalid", 0);
            setAxiOutputConstant("wvalid", 0);
            setAxiOutputConstant("bready", 0);
            outputValues[calyxDoneSignalIndex] = value0;
            idleState.getOutputOp()->setOperands(outputValues);
        }
        mlir::Region &transitions = idleState.getTransitions();
        mlir::ImplicitLocOpBuilder transitionsBuilder = mlir::ImplicitLocOpBuilder::atBlockBegin(transitions.getLoc(), &transitions.front());
        {
            {
                circt::fsm::TransitionOp transition = transitionsBuilder.create<circt::fsm::TransitionOp>("AW_HANDSHAKE");
                transition.ensureGuard(transitionsBuilder);
                circt::fsm::ReturnOp returnOp = transition.getGuardReturn();
                returnOp.setOperand(machineOp.getArgument(CalyxExtMemToAxiWriteEnSignalIndex));
            }

            {
                circt::fsm::TransitionOp transition = transitionsBuilder.create<circt::fsm::TransitionOp>("AR_HANDSHAKE");
                transition.ensureGuard(transitionsBuilder);
                circt::fsm::ReturnOp returnOp = transition.getGuardReturn();
                returnOp.setOperand(machineOp.getArgument(CalyxExtMemToAxiReadEnSignalIndex));
            }
        }
    }

    {
        circt::fsm::StateOp idleState = fsmBuilder.create<circt::fsm::StateOp>("AW_HANDSHAKE");
        {
            setAxiOutputConstant("arvalid", 0);
            setAxiOutputConstant("rready", 0);
            setAxiOutputConstant("awvalid", 1);
            setAxiOutputConstant("wvalid", 0);
            setAxiOutputConstant("bready", 0);
            outputValues[calyxDoneSignalIndex] = value0;
            idleState.getOutputOp()->setOperands(outputValues);
        }
        mlir::Region &transitions = idleState.getTransitions();
        mlir::ImplicitLocOpBuilder transitionsBuilder = mlir::ImplicitLocOpBuilder::atBlockBegin(transitions.getLoc(), &transitions.front());
        {
            {
                circt::fsm::TransitionOp transition = transitionsBuilder.create<circt::fsm::TransitionOp>("W_HANDSHAKE");
                transition.ensureGuard(transitionsBuilder);
                circt::fsm::ReturnOp returnOp = transition.getGuardReturn();
                returnOp.setOperand(getAxiInputValue("awready"));
            }
        }
    }

    {
        circt::fsm::StateOp idleState = fsmBuilder.create<circt::fsm::StateOp>("W_HANDSHAKE");
        {
            setAxiOutputConstant("arvalid", 0);
            setAxiOutputConstant("rready", 0);
            setAxiOutputConstant("awvalid", 0);
            setAxiOutputConstant("wvalid", 1);
            setAxiOutputConstant("bready", 0);
            outputValues[calyxDoneSignalIndex] = value0;
            idleState.getOutputOp()->setOperands(outputValues);
        }
        mlir::Region &transitions = idleState.getTransitions();
        mlir::ImplicitLocOpBuilder transitionsBuilder = mlir::ImplicitLocOpBuilder::atBlockBegin(transitions.getLoc(), &transitions.front());
        {
            {
                circt::fsm::TransitionOp transition = transitionsBuilder.create<circt::fsm::TransitionOp>("B_WAIT");
                transition.ensureGuard(transitionsBuilder);
                circt::fsm::ReturnOp returnOp = transition.getGuardReturn();
                returnOp.setOperand(getAxiInputValue("wready"));
            }
        }
    }

    {
        circt::fsm::StateOp idleState = fsmBuilder.create<circt::fsm::StateOp>("B_WAIT");
        {
            setAxiOutputConstant("arvalid", 0);
            setAxiOutputConstant("rready", 0);
            setAxiOutputConstant("awvalid", 0);
            setAxiOutputConstant("wvalid", 0);
            setAxiOutputConstant("bready", 1);
            outputValues[calyxDoneSignalIndex] = getAxiInputValue("bvalid");
            idleState.getOutputOp()->setOperands(outputValues);
        }
        mlir::Region &transitions = idleState.getTransitions();
        mlir::ImplicitLocOpBuilder transitionsBuilder = mlir::ImplicitLocOpBuilder::atBlockBegin(transitions.getLoc(), &transitions.front());
        {
            {
                circt::fsm::TransitionOp transition = transitionsBuilder.create<circt::fsm::TransitionOp>("IDLE");
                transition.ensureGuard(transitionsBuilder);
                circt::fsm::ReturnOp returnOp = transition.getGuardReturn();
                returnOp.setOperand(getAxiInputValue("bvalid"));
            }
        }
    }

    {
        circt::fsm::StateOp idleState = fsmBuilder.create<circt::fsm::StateOp>("AR_HANDSHAKE");
        {
            setAxiOutputConstant("arvalid", 1);
            setAxiOutputConstant("rready", 0);
            setAxiOutputConstant("awvalid", 0);
            setAxiOutputConstant("wvalid", 0);
            setAxiOutputConstant("bready", 0);
            outputValues[calyxDoneSignalIndex] = value0;
            idleState.getOutputOp()->setOperands(outputValues);
        }
        mlir::Region &transitions = idleState.getTransitions();
        mlir::ImplicitLocOpBuilder transitionsBuilder = mlir::ImplicitLocOpBuilder::atBlockBegin(transitions.getLoc(), &transitions.front());
        {
            {
                circt::fsm::TransitionOp transition = transitionsBuilder.create<circt::fsm::TransitionOp>("R_HANDSHAKE");
                transition.ensureGuard(transitionsBuilder);
                circt::fsm::ReturnOp returnOp = transition.getGuardReturn();
                returnOp.setOperand(getAxiInputValue("arready"));
            }
        }
    }

    {
        circt::fsm::StateOp idleState = fsmBuilder.create<circt::fsm::StateOp>("R_HANDSHAKE");
        {
            setAxiOutputConstant("arvalid", 0);
            setAxiOutputConstant("rready", 1);
            setAxiOutputConstant("awvalid", 0);
            setAxiOutputConstant("wvalid", 0);
            setAxiOutputConstant("bready", 0);
            outputValues[calyxDoneSignalIndex] = getAxiInputValue("rvalid");
            idleState.getOutputOp()->setOperands(outputValues);
        }
        mlir::Region &transitions = idleState.getTransitions();
        mlir::ImplicitLocOpBuilder transitionsBuilder = mlir::ImplicitLocOpBuilder::atBlockBegin(transitions.getLoc(), &transitions.front());
        {
            {
                circt::fsm::TransitionOp transition = transitionsBuilder.create<circt::fsm::TransitionOp>("IDLE");
                transition.ensureGuard(transitionsBuilder);
                circt::fsm::ReturnOp returnOp = transition.getGuardReturn();
                returnOp.setOperand(getAxiInputValue("rvalid"));
            }
        }
    }
}

void CodeGen_CIRCT::generateControlAxi(mlir::ImplicitLocOpBuilder &builder, const FlattenedKernelArgs &kernelArgs) {
    mlir::Type axiAddrWidthType = builder.getIntegerType(S_AXI_ADDR_WIDTH);
    mlir::Type axiDataWidthType = builder.getIntegerType(S_AXI_DATA_WIDTH);

    // Module inputs and outputs
    mlir::SmallVector<circt::hw::PortInfo> ports;

    // Clock and reset signals
    ports.push_back(circt::hw::PortInfo{builder.getStringAttr("clock"), circt::hw::PortDirection::INPUT, builder.getI1Type()});
    ports.push_back(circt::hw::PortInfo{builder.getStringAttr("reset"), circt::hw::PortDirection::INPUT, builder.getI1Type()});

    // ap_start and ap_done
    ports.push_back(circt::hw::PortInfo{builder.getStringAttr("ap_start"), circt::hw::PortDirection::OUTPUT, builder.getI1Type()});
    ports.push_back(circt::hw::PortInfo{builder.getStringAttr("ap_done"), circt::hw::PortDirection::INPUT, builder.getI1Type()});

    // AXI signals
    portsAddAXI4LiteSubordinateSignals(builder, S_AXI_ADDR_WIDTH, S_AXI_DATA_WIDTH, ports);

    // Kernel arguments
    for (const auto &arg : kernelArgs) {
        ports.push_back(circt::hw::PortInfo{builder.getStringAttr(arg.name),
                                            circt::hw::PortDirection::OUTPUT,
                                            builder.getIntegerType(arg.getHWBits())});
    }

    // Create ControllerAXI HW module
    circt::hw::HWModuleOp hwModuleOp = builder.create<circt::hw::HWModuleOp>(builder.getStringAttr("ControlAXI"), ports);

    // This will hold the list of all the output Values
    mlir::SmallVector<mlir::Value> hwModuleOutputValues(hwModuleOp.getNumResults());

    // Helpers
    auto moduleGetInputValue = [&](const std::string &name) {
        const auto &names = hwModuleOp.getArgNames();
        for (unsigned int i = 0; i < hwModuleOp.getNumArguments(); i++) {
            if (names[i].cast<mlir::StringAttr>().str() == name)
                return hwModuleOp.getArgument(i);
        }
        assert(0);
    };

    auto moduleGetAxiInputValue = [&](const std::string &name) {
        return moduleGetInputValue(toFullAxiSubordinateSignalName(name));
    };

    auto moduleGetOutputIndex = [&](const std::string &name) {
        unsigned int i;
        const auto &names = hwModuleOp.getResultNames();
        for (i = 0; i < hwModuleOp.getNumResults(); i++) {
            if (names[i].cast<mlir::StringAttr>().str() == name)
                break;
        }
        assert(i < hwModuleOp.getNumResults());
        return i;
    };

    auto moduleGetAxiOutputIndex = [&](const std::string &name) {
        return moduleGetOutputIndex(toFullAxiSubordinateSignalName(name));
    };

    mlir::Value clock = moduleGetInputValue("clock");
    mlir::Value reset = moduleGetInputValue("reset");

    // ReadState FSM
    mlir::SmallVector<std::pair<std::string, mlir::Value>> readStateFsmInputs{
        {"arvalid", moduleGetAxiInputValue("arvalid")},
        {"rready", moduleGetAxiInputValue("rready")}};
    mlir::SmallVector<std::pair<std::string, unsigned int>> readStateFsmOutputs{
        {"arready", moduleGetAxiOutputIndex("arready")},
        {"rvalid", moduleGetAxiOutputIndex("rvalid")}};
    mlir::SmallVector<mlir::Value> readStateFsmInputValues;
    mlir::SmallVector<mlir::Type> readStateFsmOutputTypes;
    circt::fsm::MachineOp readFsmMachineOp;
    {
        mlir::SmallVector<mlir::Type> fsmInputTypes;
        mlir::SmallVector<mlir::Attribute> fsmInputNames, fsmOutputNames;

        for (const auto &input : readStateFsmInputs) {
            fsmInputTypes.push_back(input.second.getType());
            readStateFsmInputValues.push_back(input.second);
            fsmInputNames.push_back(builder.getStringAttr(toFullAxiSubordinateSignalName(input.first)));
        }

        for (const auto &output : readStateFsmOutputs) {
            readStateFsmOutputTypes.push_back(hwModuleOp.getResultTypes()[output.second]);
            fsmOutputNames.push_back(builder.getStringAttr(toFullAxiSubordinateSignalName(output.first)));
        }

        mlir::FunctionType fsmFunctionType = builder.getFunctionType(fsmInputTypes, readStateFsmOutputTypes);
        readFsmMachineOp = builder.create<circt::fsm::MachineOp>("ControlAXI_ReadFSM", "IDLE", fsmFunctionType);
        readFsmMachineOp.setArgNamesAttr(builder.getArrayAttr(fsmInputNames));
        readFsmMachineOp.setResNamesAttr(builder.getArrayAttr(fsmOutputNames));
        mlir::SmallVector<mlir::Value> fsmOutputValues(readFsmMachineOp.getNumResults());

        auto &fsmBody = readFsmMachineOp.getBody();
        auto fsmBuilder = mlir::ImplicitLocOpBuilder::atBlockEnd(fsmBody.getLoc(), &fsmBody.front());
        mlir::Value value0 = fsmBuilder.create<circt::hw::ConstantOp>(builder.getBoolAttr(false));
        mlir::Value value1 = fsmBuilder.create<circt::hw::ConstantOp>(builder.getBoolAttr(true));
        {
            auto state = fsmBuilder.create<circt::fsm::StateOp>("IDLE");
            {
                fsmOutputValues[0] = value1;
                fsmOutputValues[1] = value0;
                state.getOutputOp()->setOperands(fsmOutputValues);
            }
            auto &transitions = state.getTransitions();
            auto transitionsBuilder = mlir::ImplicitLocOpBuilder::atBlockBegin(transitions.getLoc(), &transitions.front());
            {
                auto transition = transitionsBuilder.create<circt::fsm::TransitionOp>("DATA");
                transition.ensureGuard(transitionsBuilder);
                transition.getGuardReturn().setOperand(readFsmMachineOp.getArgument(0));
            }
        }
        {
            auto state = fsmBuilder.create<circt::fsm::StateOp>("DATA");
            {
                fsmOutputValues[0] = value0;
                fsmOutputValues[1] = value1;
                state.getOutputOp()->setOperands(fsmOutputValues);
            }
            auto &transitions = state.getTransitions();
            auto transitionsBuilder = mlir::ImplicitLocOpBuilder::atBlockBegin(transitions.getLoc(), &transitions.front());
            {
                auto transition = transitionsBuilder.create<circt::fsm::TransitionOp>("IDLE");
                transition.ensureGuard(transitionsBuilder);
                transition.getGuardReturn().setOperand(readFsmMachineOp.getArgument(1));
            }
        }
    }

    // WriteState FSM
    mlir::SmallVector<std::pair<std::string, mlir::Value>> writeStateFsmInputs{
        {"awvalid", moduleGetAxiInputValue("awvalid")},
        {"wvalid", moduleGetAxiInputValue("wvalid")},
        {"bready", moduleGetAxiInputValue("bready")}};
    mlir::SmallVector<std::pair<std::string, unsigned int>> writeStateFsmOutputs{
        {"awready", moduleGetAxiOutputIndex("awready")},
        {"wready", moduleGetAxiOutputIndex("wready")},
        {"bvalid", moduleGetAxiOutputIndex("bvalid")}};
    mlir::SmallVector<mlir::Value> writeStateFsmInputValues;
    mlir::SmallVector<mlir::Type> writeStateFsmOutputTypes;
    circt::fsm::MachineOp writeFsmMachineOp;
    {
        mlir::SmallVector<mlir::Type> fsmInputTypes;
        mlir::SmallVector<mlir::Attribute> fsmInputNames, fsmOutputNames;

        for (const auto &input : writeStateFsmInputs) {
            fsmInputTypes.push_back(input.second.getType());
            writeStateFsmInputValues.push_back(input.second);
            fsmInputNames.push_back(builder.getStringAttr(toFullAxiSubordinateSignalName(input.first)));
        }

        for (const auto &output : writeStateFsmOutputs) {
            writeStateFsmOutputTypes.push_back(hwModuleOp.getResultTypes()[output.second]);
            fsmOutputNames.push_back(builder.getStringAttr(toFullAxiSubordinateSignalName(output.first)));
        }

        mlir::FunctionType fsmFunctionType = builder.getFunctionType(fsmInputTypes, writeStateFsmOutputTypes);
        writeFsmMachineOp = builder.create<circt::fsm::MachineOp>("ControlAXI_WriteFSM", "IDLE", fsmFunctionType);
        writeFsmMachineOp.setArgNamesAttr(builder.getArrayAttr(fsmInputNames));
        writeFsmMachineOp.setResNamesAttr(builder.getArrayAttr(fsmOutputNames));
        mlir::SmallVector<mlir::Value> fsmOutputValues(writeFsmMachineOp.getNumResults());

        auto &fsmBody = writeFsmMachineOp.getBody();
        auto fsmBuilder = mlir::ImplicitLocOpBuilder::atBlockEnd(fsmBody.getLoc(), &fsmBody.front());
        mlir::Value value0 = fsmBuilder.create<circt::hw::ConstantOp>(builder.getBoolAttr(false));
        mlir::Value value1 = fsmBuilder.create<circt::hw::ConstantOp>(builder.getBoolAttr(true));
        {
            auto state = fsmBuilder.create<circt::fsm::StateOp>("IDLE");
            {
                fsmOutputValues[0] = value1;
                fsmOutputValues[1] = value0;
                fsmOutputValues[2] = value0;
                state.getOutputOp()->setOperands(fsmOutputValues);
            }
            auto &transitions = state.getTransitions();
            auto transitionsBuilder = mlir::ImplicitLocOpBuilder::atBlockBegin(transitions.getLoc(), &transitions.front());
            {
                auto transition = transitionsBuilder.create<circt::fsm::TransitionOp>("DATA");
                transition.ensureGuard(transitionsBuilder);
                transition.getGuardReturn().setOperand(writeFsmMachineOp.getArgument(0));
            }
        }
        {
            auto state = fsmBuilder.create<circt::fsm::StateOp>("DATA");
            {
                fsmOutputValues[0] = value0;
                fsmOutputValues[1] = value1;
                fsmOutputValues[2] = value0;
                state.getOutputOp()->setOperands(fsmOutputValues);
            }
            auto &transitions = state.getTransitions();
            auto transitionsBuilder = mlir::ImplicitLocOpBuilder::atBlockBegin(transitions.getLoc(), &transitions.front());
            {
                auto transition = transitionsBuilder.create<circt::fsm::TransitionOp>("RESP");
                transition.ensureGuard(transitionsBuilder);
                transition.getGuardReturn().setOperand(writeFsmMachineOp.getArgument(1));
            }
        }
        {
            auto state = fsmBuilder.create<circt::fsm::StateOp>("RESP");
            {
                fsmOutputValues[0] = value0;
                fsmOutputValues[1] = value0;
                fsmOutputValues[2] = value1;
                state.getOutputOp()->setOperands(fsmOutputValues);
            }
            auto &transitions = state.getTransitions();
            auto transitionsBuilder = mlir::ImplicitLocOpBuilder::atBlockBegin(transitions.getLoc(), &transitions.front());
            {
                auto transition = transitionsBuilder.create<circt::fsm::TransitionOp>("IDLE");
                transition.ensureGuard(transitionsBuilder);
                transition.getGuardReturn().setOperand(writeFsmMachineOp.getArgument(2));
            }
        }
    }

    // Instantiate FSMs
    builder.setInsertionPointToStart(hwModuleOp.getBodyBlock());
    mlir::Value value0 = builder.create<circt::hw::ConstantOp>(builder.getBoolAttr(false));
    mlir::Value value1 = builder.create<circt::hw::ConstantOp>(builder.getBoolAttr(true));

    auto readFsmInstanceOp =
        builder.create<circt::fsm::HWInstanceOp>(readStateFsmOutputTypes, "ReadFSM", readFsmMachineOp.getSymName(),
                                                 readStateFsmInputValues, clock, reset);

    for (unsigned int i = 0; i < readFsmInstanceOp.getNumResults(); i++)
        hwModuleOutputValues[moduleGetAxiOutputIndex(readStateFsmOutputs[i].first)] =
            readFsmInstanceOp.getResult(i);

    auto writeFsmInstanceOp =
        builder.create<circt::fsm::HWInstanceOp>(writeStateFsmOutputTypes, "WriteFSM", writeFsmMachineOp.getSymName(),
                                                 writeStateFsmInputValues, clock, reset);

    for (unsigned int i = 0; i < writeFsmMachineOp.getNumResults(); i++)
        hwModuleOutputValues[moduleGetAxiOutputIndex(writeStateFsmOutputs[i].first)] =
            writeFsmInstanceOp.getResult(i);

    // ControlAXI logic
    hwModuleOutputValues[moduleGetAxiOutputIndex("rresp")] = builder.create<circt::hw::ConstantOp>(builder.getI2Type(), 0);
    hwModuleOutputValues[moduleGetAxiOutputIndex("bresp")] = builder.create<circt::hw::ConstantOp>(builder.getI2Type(), 0);

    // Write address. Store it into a register
    mlir::Value awaddr_next = builder.create<circt::sv::LogicOp>(axiAddrWidthType, "awaddr_next");
    mlir::Value awaddr_next_read = builder.create<circt::sv::ReadInOutOp>(awaddr_next);
    mlir::Value awaddr_reg = builder.create<circt::seq::CompRegOp>(awaddr_next_read, clock, reset,
                                                                   builder.create<circt::hw::ConstantOp>(axiAddrWidthType, 0), "awaddr_reg");
    mlir::Value writeToAwaddrReg = builder.create<circt::comb::AndOp>(moduleGetAxiInputValue("awvalid"),
                                                                      hwModuleOutputValues[moduleGetAxiOutputIndex("awready")]);
    builder.create<circt::sv::AlwaysCombOp>(/*bodyCtor*/ [&]() {
        builder.create<circt::sv::IfOp>(
            writeToAwaddrReg,
            /*thenCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(awaddr_next, moduleGetAxiInputValue("awaddr")); },
            /*elseCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(awaddr_next, awaddr_reg); });
    });

    const mlir::Value isWriteValidReady = builder.create<circt::comb::AndOp>(moduleGetAxiInputValue("wvalid"),
                                                                             hwModuleOutputValues[moduleGetAxiOutputIndex("wready")]);
    const mlir::Value isAreadValidReady = builder.create<circt::comb::AndOp>(moduleGetAxiInputValue("arvalid"),
                                                                             hwModuleOutputValues[moduleGetAxiOutputIndex("arready")]);
    const mlir::Value isReadValidReady = builder.create<circt::comb::AndOp>(moduleGetAxiInputValue("rready"),
                                                                            hwModuleOutputValues[moduleGetAxiOutputIndex("rvalid")]);
    // Control Register Signals (offset 0x00)
    mlir::Value int_ap_start_next = builder.create<circt::sv::LogicOp>(builder.getI1Type(), "int_ap_start_next");
    mlir::Value int_ap_start_next_read = builder.create<circt::sv::ReadInOutOp>(int_ap_start_next);
    mlir::Value int_ap_start = builder.create<circt::seq::CompRegOp>(int_ap_start_next_read, clock, reset, value0, "int_ap_start_reg");
    hwModuleOutputValues[moduleGetOutputIndex("ap_start")] = int_ap_start;
    mlir::Value isWaddr0x00 = builder.create<circt::comb::ICmpOp>(circt::comb::ICmpPredicate::eq, awaddr_reg,
                                                                  builder.create<circt::hw::ConstantOp>(axiAddrWidthType, 0));

    builder.create<circt::sv::AlwaysCombOp>(/*bodyCtor*/ [&]() {
        builder.create<circt::sv::IfOp>(
            builder.create<circt::comb::AndOp>(isWriteValidReady, isWaddr0x00),
            /*thenCtor*/ [&]() {
                mlir::Value apStartBit = builder.create<circt::comb::ExtractOp>(moduleGetAxiInputValue("wdata"), 0, 1);
                builder.create<circt::sv::BPAssignOp>(int_ap_start_next, apStartBit); },
            /*elseCtor*/ [&]() { builder.create<circt::sv::IfOp>(
                                     moduleGetInputValue("ap_done"),
                                     /*thenCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(int_ap_start_next, value0); },
                                     /*elseCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(int_ap_start_next, int_ap_start); }); });
    });

    mlir::Value int_ap_done_next = builder.create<circt::sv::LogicOp>(builder.getI1Type(), "int_ap_done_next");
    mlir::Value int_ap_done_next_read = builder.create<circt::sv::ReadInOutOp>(int_ap_done_next);
    mlir::Value int_ap_done = builder.create<circt::seq::CompRegOp>(int_ap_done_next_read, clock, reset, value0, "int_ap_done_reg");
    mlir::Value isRaddr0x00 = builder.create<circt::comb::ICmpOp>(circt::comb::ICmpPredicate::eq,
                                                                  moduleGetAxiInputValue("araddr"),
                                                                  builder.create<circt::hw::ConstantOp>(axiAddrWidthType, 0));

    builder.create<circt::sv::AlwaysCombOp>(/*bodyCtor*/ [&]() {
        builder.create<circt::sv::IfOp>(
            builder.create<circt::comb::AndOp>(isReadValidReady, isRaddr0x00),
            /*thenCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(int_ap_done_next, value0); },
            /*elseCtor*/ [&]() { builder.create<circt::sv::IfOp>(
                                     moduleGetInputValue("ap_done"),
                                     /*thenCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(int_ap_done_next, value1); },
                                     /*elseCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(int_ap_done_next, int_ap_done); }); });
    });

    mlir::Value int_ap_idle_next = builder.create<circt::sv::LogicOp>(builder.getI1Type(), "int_ap_idle_next");
    mlir::Value int_ap_idle_next_read = builder.create<circt::sv::ReadInOutOp>(int_ap_idle_next);
    mlir::Value int_ap_idle = builder.create<circt::seq::CompRegOp>(int_ap_idle_next_read, clock, reset, value1, "int_ap_idle_reg");

    builder.create<circt::sv::AlwaysCombOp>(/*bodyCtor*/ [&]() {
        builder.create<circt::sv::IfOp>(
            moduleGetInputValue("ap_done"),
            /*thenCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(int_ap_idle_next, value1); },
            /*elseCtor*/ [&]() { builder.create<circt::sv::IfOp>(
                                     hwModuleOutputValues[moduleGetOutputIndex("ap_start")],
                                     /*thenCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(int_ap_idle_next, value0); },
                                     /*elseCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(int_ap_idle_next, int_ap_idle); }); });
    });

    mlir::Value int_gie_next = builder.create<circt::sv::LogicOp>(builder.getI1Type(), "int_gie_next");
    mlir::Value int_gie_read = builder.create<circt::sv::ReadInOutOp>(int_gie_next);
    mlir::Value int_gie = builder.create<circt::seq::CompRegOp>(int_gie_read, clock, reset, value0, "int_gie_reg");
    mlir::Value isWaddr0x04 = builder.create<circt::comb::ICmpOp>(circt::comb::ICmpPredicate::eq, awaddr_reg,
                                                                  builder.create<circt::hw::ConstantOp>(axiAddrWidthType, 0x04));

    builder.create<circt::sv::AlwaysCombOp>(/*bodyCtor*/ [&]() {
        builder.create<circt::sv::IfOp>(
            builder.create<circt::comb::AndOp>(isWriteValidReady, isWaddr0x04),
            /*thenCtor*/ [&]() {
                mlir::Value gieBit = builder.create<circt::comb::ExtractOp>(moduleGetAxiInputValue("wdata"), 0, 1);
                builder.create<circt::sv::BPAssignOp>(int_gie_next, gieBit); },
            /*elseCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(int_gie_next, int_gie); });
    });

    mlir::Value int_ier_next = builder.create<circt::sv::LogicOp>(builder.getI2Type(), "int_ier_next");
    mlir::Value int_ier_read = builder.create<circt::sv::ReadInOutOp>(int_ier_next);
    mlir::Value int_ier = builder.create<circt::seq::CompRegOp>(int_ier_read, clock, reset,
                                                                builder.create<circt::hw::ConstantOp>(builder.getI2Type(), 0),
                                                                "int_ier_reg");
    mlir::Value isWaddr0x08 = builder.create<circt::comb::ICmpOp>(circt::comb::ICmpPredicate::eq, awaddr_reg,
                                                                  builder.create<circt::hw::ConstantOp>(axiAddrWidthType, 0x08));

    builder.create<circt::sv::AlwaysCombOp>(/*bodyCtor*/ [&]() {
        builder.create<circt::sv::IfOp>(
            builder.create<circt::comb::AndOp>(isWriteValidReady, isWaddr0x08),
            /*thenCtor*/ [&]() {
                mlir::Value ierBits = builder.create<circt::comb::ExtractOp>(moduleGetAxiInputValue("wdata"), 0, 2);
                builder.create<circt::sv::BPAssignOp>(int_ier_next, ierBits); },
            /*elseCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(int_ier_next, int_ier); });
    });

    mlir::Value int_isr_done_next = builder.create<circt::sv::LogicOp>(builder.getI1Type(), "int_isr_done_next");
    mlir::Value int_isr_done_read = builder.create<circt::sv::ReadInOutOp>(int_isr_done_next);
    mlir::Value int_isr_done = builder.create<circt::seq::CompRegOp>(int_isr_done_read, clock, reset, value0, "int_isr_done_reg");
    mlir::Value int_isr_ready_next = builder.create<circt::sv::LogicOp>(builder.getI1Type(), "int_isr_ready_next");
    mlir::Value int_isr_ready_read = builder.create<circt::sv::ReadInOutOp>(int_isr_ready_next);
    mlir::Value int_isr_ready = builder.create<circt::seq::CompRegOp>(int_isr_ready_read, clock, reset, value0, "int_isr_ready_reg");
    mlir::Value isWaddr0x0C = builder.create<circt::comb::ICmpOp>(circt::comb::ICmpPredicate::eq, awaddr_reg,
                                                                  builder.create<circt::hw::ConstantOp>(axiAddrWidthType, 0x0C));

    builder.create<circt::sv::AlwaysCombOp>(/*bodyCtor*/ [&]() {
        builder.create<circt::sv::IfOp>(
            builder.create<circt::comb::AndOp>(isWriteValidReady, isWaddr0x0C),
            /*thenCtor*/ [&]() {
                mlir::Value isrDoneBit = builder.create<circt::comb::ExtractOp>(moduleGetAxiInputValue("wdata"), 0, 1);
                mlir::Value isrReadyBit = builder.create<circt::comb::ExtractOp>(moduleGetAxiInputValue("wdata"), 1, 1);
                builder.create<circt::sv::BPAssignOp>(int_isr_done_next, isrDoneBit);
                builder.create<circt::sv::BPAssignOp>(int_isr_ready_next, isrReadyBit); },
            /*elseCtor*/ [&]() {
                builder.create<circt::sv::BPAssignOp>(int_isr_done_next, int_isr_done);
                builder.create<circt::sv::BPAssignOp>(int_isr_ready_next, int_isr_ready); });
    });

    // Create registers storing kernel arguments (scalars and buffer pointers)
    uint32_t argOffset = XRT_KERNEL_ARGS_OFFSET;
    for (const auto &arg : kernelArgs) {
        mlir::Type type = builder.getIntegerType(arg.getHWBits());
        mlir::Value zero = builder.create<circt::hw::ConstantOp>(type, 0);
        mlir::Value argRegNext = builder.create<circt::sv::LogicOp>(type, arg.name + "_next");
        mlir::Value argRegNextRead = builder.create<circt::sv::ReadInOutOp>(argRegNext);
        mlir::Value argReg = builder.create<circt::seq::CompRegOp>(argRegNextRead, clock, reset, zero, arg.name + "_reg");

        // Implement the store logic
        int size = 0;
        uint32_t subArgOffset = argOffset;
        while (size < arg.getHWBits()) {
            mlir::Value isWaddrToSubArgAddr = builder.create<circt::comb::ICmpOp>(circt::comb::ICmpPredicate::eq, awaddr_reg,
                                                                                  builder.create<circt::hw::ConstantOp>(axiAddrWidthType, subArgOffset));
            mlir::Value startBit = builder.create<circt::hw::ConstantOp>(axiAddrWidthType, size);
            mlir::Value bitsToUpdate = builder.create<circt::sv::IndexedPartSelectInOutOp>(argRegNext, startBit, 32);
            builder.create<circt::sv::AlwaysCombOp>(/*bodyCtor*/ [&]() {
                builder.create<circt::sv::IfOp>(
                    builder.create<circt::comb::AndOp>(isWriteValidReady, isWaddrToSubArgAddr),
                    /*thenCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(bitsToUpdate, moduleGetAxiInputValue("wdata")); },
                    /*elseCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(bitsToUpdate,
                                                                               builder.create<circt::sv::IndexedPartSelectOp>(argReg, startBit, 32)); });
            });

            size += 32;
            subArgOffset += 4;
        }
        hwModuleOutputValues[moduleGetOutputIndex(arg.name)] = argReg;
        argOffset += 8;
    }

    // Holds the data that the host requested to read
    mlir::Value rdata_next = builder.create<circt::sv::LogicOp>(axiDataWidthType, "rdata_next");
    mlir::Value rdata_next_read = builder.create<circt::sv::ReadInOutOp>(rdata_next);
    mlir::Value rdata = builder.create<circt::seq::CompRegOp>(rdata_next_read, clock, reset,
                                                              builder.create<circt::hw::ConstantOp>(axiDataWidthType, 0),
                                                              "rdata_reg");
    hwModuleOutputValues[moduleGetAxiOutputIndex("rdata")] = rdata;

    // XRT registers + number of kernel arguments (each is considered to be have 8 bytes)
    const size_t numCases = XRT_KERNEL_ARGS_OFFSET / 4 + kernelArgs.size() * 2;

    builder.create<circt::sv::AlwaysCombOp>(/*bodyCtor*/ [&]() {
        builder.create<circt::sv::IfOp>(
            isAreadValidReady,
            /*thenCtor*/ [&]() {
                mlir::Value index = builder.create<circt::comb::ExtractOp>(moduleGetAxiInputValue("araddr"), 0, 12);
                builder.create<circt::sv::CaseOp>(
                    CaseStmtType::CaseStmt, index, numCases + 1,
                    [&](size_t caseIdx) -> std::unique_ptr<circt::sv::CasePattern> {
                        bool isDefault = caseIdx == numCases;
                        std::unique_ptr<circt::sv::CasePattern> pattern;
                        mlir::Value value;

                        if (isDefault) {
                            pattern = std::make_unique<circt::sv::CaseDefaultPattern>(builder.getContext());
                        } else {
                            const uint32_t offset = caseIdx << 2;
                            pattern = std::make_unique<circt::sv::CaseBitPattern>(mlir::APInt(/*numBits=*/12, offset),
                                                                                  builder.getContext());
                            switch (offset) {
                            case 0x00:
                                value = builder.create<circt::comb::ConcatOp>(mlir::ValueRange{
                                    builder.create<circt::hw::ConstantOp>(builder.getIntegerType(29), 0),
                                    int_ap_idle, int_ap_done, int_ap_start});
                                break;
                            case 0x04:
                                value = builder.create<circt::comb::ConcatOp>(mlir::ValueRange{
                                    builder.create<circt::hw::ConstantOp>(builder.getIntegerType(31), 0),
                                    int_gie});
                                break;
                            case 0x08:
                                value = builder.create<circt::comb::ConcatOp>(mlir::ValueRange{
                                    builder.create<circt::hw::ConstantOp>(builder.getIntegerType(30), 0),
                                    int_ier});
                                break;
                            case 0x0C:
                                value = builder.create<circt::comb::ConcatOp>(mlir::ValueRange{
                                    builder.create<circt::hw::ConstantOp>(builder.getIntegerType(30), 0),
                                    int_isr_ready, int_isr_done});
                                break;
                            default:
                                const KernelArg &arg = kernelArgs[(offset - XRT_KERNEL_ARGS_OFFSET) >> 3];
                                if ((offset % 8) == 0 || arg.getHWBits() > 32) {
                                    value = builder.create<circt::comb::ExtractOp>(hwModuleOutputValues[moduleGetOutputIndex(arg.name)],
                                                                                   (offset % 8) * 8, 32);
                                }
                                break;
                            }
                        }
                        if (!value)
                            value = builder.create<circt::hw::ConstantOp>(axiDataWidthType, 0);
                        builder.create<circt::sv::BPAssignOp>(rdata_next, value);
                        return pattern;
                    }); },
            /*elseCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(rdata_next, rdata); });
    });

    // Set module output operands
    auto outputOp = hwModuleOp.getBodyBlock()->getTerminator();
    outputOp->setOperands(hwModuleOutputValues);
}

void CodeGen_CIRCT::generateToplevel(mlir::ImplicitLocOpBuilder &builder, const std::string &kernelName, const FlattenedKernelArgs &kernelArgs) {
    // Module inputs and outputs
    mlir::SmallVector<circt::hw::PortInfo> ports;

    // Clock and reset signals
    ports.push_back(circt::hw::PortInfo{builder.getStringAttr("ap_clk"), circt::hw::PortDirection::INPUT, builder.getI1Type()});
    ports.push_back(circt::hw::PortInfo{builder.getStringAttr("ap_rst_n"), circt::hw::PortDirection::INPUT, builder.getI1Type()});

    // AXI4 lite subordinate control signals
    mlir::SmallVector<circt::hw::PortInfo> axi4LiteSubordinateSignals;
    portsAddAXI4LiteSubordinateSignals(builder, S_AXI_ADDR_WIDTH, S_AXI_DATA_WIDTH, axi4LiteSubordinateSignals);
    ports.append(axi4LiteSubordinateSignals);

    // AXI4 lite subordinate control signals
    mlir::SmallVector<mlir::SmallVector<circt::hw::PortInfo>> axi4ManagerSignals;

    // Signals for CalyxExtMemToAxi for each kernel buffer argument
    unsigned numBufferArgs = 0;
    for (unsigned i = 0; i < kernelArgs.size(); i++) {
        if (kernelArgs[i].isPointer) {
            mlir::SmallVector<circt::hw::PortInfo> signals;
            portsAddAXI4ManagerSignalsPrefix(builder, getAxiManagerSignalNamePrefixId(numBufferArgs) + "_",
                                             M_AXI_ADDR_WIDTH, M_AXI_DATA_WIDTH, signals);
            axi4ManagerSignals.push_back(signals);
            ports.append(signals);

            // araddr, awaddr, rdata and wdata
            ports.push_back(circt::hw::PortInfo{builder.getStringAttr(toFullAxiManagerSignalNameId(numBufferArgs, "araddr")),
                                                circt::hw::PortDirection::OUTPUT, builder.getIntegerType(M_AXI_ADDR_WIDTH)});
            ports.push_back(circt::hw::PortInfo{builder.getStringAttr(toFullAxiManagerSignalNameId(numBufferArgs, "awaddr")),
                                                circt::hw::PortDirection::OUTPUT, builder.getIntegerType(M_AXI_ADDR_WIDTH)});
            ports.push_back(circt::hw::PortInfo{builder.getStringAttr(toFullAxiManagerSignalNameId(numBufferArgs, "rdata")),
                                                circt::hw::PortDirection::INPUT, builder.getIntegerType(M_AXI_DATA_WIDTH)});
            ports.push_back(circt::hw::PortInfo{builder.getStringAttr(toFullAxiManagerSignalNameId(numBufferArgs, "wdata")),
                                                circt::hw::PortDirection::OUTPUT, builder.getIntegerType(M_AXI_DATA_WIDTH)});
            numBufferArgs++;
        }
    }

    // Create toplevel HW module
    circt::hw::HWModuleOp hwModuleOp = builder.create<circt::hw::HWModuleOp>(builder.getStringAttr("toplevel"), ports);
    builder.setInsertionPointToStart(hwModuleOp.getBodyBlock());

    // This will hold the list of all the output Values
    mlir::SmallVector<mlir::Value> hwModuleOutputWires(hwModuleOp.getNumResults());
    mlir::SmallVector<mlir::Value> hwModuleOutputValues(hwModuleOp.getNumResults());
    for (unsigned i = 0; i < hwModuleOp.getNumResults(); i++) {
        auto name = hwModuleOp.getResultNames()[i].cast<mlir::StringAttr>().str();
        auto type = hwModuleOp.getResultTypes()[i];
        hwModuleOutputWires[i] = builder.create<circt::sv::WireOp>(type, name);
        hwModuleOutputValues[i] = builder.create<circt::sv::ReadInOutOp>(hwModuleOutputWires[i]);
    }

    // Module access helpers
    auto moduleGetInputValue = [&](const std::string &name) {
        const auto &names = hwModuleOp.getArgNames();
        for (unsigned int i = 0; i < hwModuleOp.getNumArguments(); i++) {
            if (names[i].cast<mlir::StringAttr>().str() == name)
                return hwModuleOp.getArgument(i);
        }
        assert(0);
    };

    auto moduleGetOutputIndex = [](auto &mod, const std::string &name) {
        unsigned int i;
        const auto &names = mod.getResultNames();
        for (i = 0; i < mod.getNumResults(); i++) {
            if (names[i].template cast<mlir::StringAttr>().str() == name)
                break;
        }
        assert(i < mod.getNumResults());
        return i;
    };

    mlir::Value clock = moduleGetInputValue("ap_clk");
    // Create reset signal from ap_rst_n
    mlir::Value reset = builder.create<circt::comb::XorOp>(moduleGetInputValue("ap_rst_n"),
                                                           builder.create<circt::hw::ConstantOp>(builder.getBoolAttr(1)));

    mlir::Value apDone = builder.create<circt::sv::WireOp>(builder.getI1Type(), "ap_done");
    mlir::Value apDoneRead = builder.create<circt::sv::ReadInOutOp>(apDone);

    // Control AXI-slave subordinate
    mlir::SmallVector<mlir::Value> controlAxiInputs;
    mlir::SmallVector<mlir::Type> controlAxiResultTypes;
    mlir::SmallVector<mlir::Attribute> controlAxiArgNames, controlAxiResultNames;

    controlAxiInputs.push_back(clock);
    controlAxiArgNames.push_back(builder.getStringAttr("clock"));
    controlAxiInputs.push_back(reset);
    controlAxiArgNames.push_back(builder.getStringAttr("reset"));
    controlAxiInputs.push_back(apDoneRead);
    controlAxiArgNames.push_back(builder.getStringAttr("ap_done"));

    controlAxiResultTypes.push_back(builder.getI1Type());
    controlAxiResultNames.push_back(builder.getStringAttr("ap_start"));

    for (const auto &signal : axi4LiteSubordinateSignals) {
        if (signal.direction == circt::hw::PortDirection::INPUT) {
            controlAxiInputs.push_back(moduleGetInputValue(signal.name.str()));
            controlAxiArgNames.push_back(signal.name);
        } else {
            controlAxiResultTypes.push_back(signal.type);
            controlAxiResultNames.push_back(signal.name);
        }
    }

    // Instance a CalyxExtMemToAxi for each kernel buffer argument
    mlir::SmallVector<circt::hw::InstanceOp> CalyxExtMemToAxiInstances(numBufferArgs);
    mlir::SmallVector<circt::sv::WireOp> CalyxExtMemToAxiReadEnWires(numBufferArgs);
    mlir::SmallVector<circt::sv::WireOp> CalyxExtMemToAxiWriteEnWires(numBufferArgs);
    mlir::SmallVector<mlir::Value> CalyxExtMemToAxiReadEnWiresRead(numBufferArgs);
    mlir::SmallVector<mlir::Value> CalyxExtMemToAxiWriteEnWiresRead(numBufferArgs);

    for (unsigned i = 0; i < numBufferArgs; i++) {
        mlir::SmallVector<mlir::Value> CalyxExtMemToAxiInputs;
        mlir::SmallVector<mlir::Type> CalyxExtMemToAxiResultTypes;
        mlir::SmallVector<mlir::Attribute> CalyxExtMemToAxiArgNames, CalyxExtMemToAxiResultNames;

        CalyxExtMemToAxiReadEnWires[i] = builder.create<circt::sv::WireOp>(builder.getI1Type(), "calyx_read_en_" + std::to_string(i));
        CalyxExtMemToAxiReadEnWiresRead[i] = builder.create<circt::sv::ReadInOutOp>(CalyxExtMemToAxiReadEnWires[i]);
        CalyxExtMemToAxiWriteEnWires[i] = builder.create<circt::sv::WireOp>(builder.getI1Type(), "calyx_write_en_" + std::to_string(i));
        CalyxExtMemToAxiWriteEnWiresRead[i] = builder.create<circt::sv::ReadInOutOp>(CalyxExtMemToAxiWriteEnWires[i]);

        for (const auto &signal : axi4ManagerSignals[i]) {
            std::string basename = fullAxiSignalNameIdGetBasename(signal.name.str());
            mlir::StringAttr nameAttr = builder.getStringAttr(toFullAxiManagerSignalName(basename));
            if (signal.direction == circt::hw::PortDirection::INPUT) {
                CalyxExtMemToAxiInputs.push_back(moduleGetInputValue(signal.name.str()));
                CalyxExtMemToAxiArgNames.push_back(nameAttr);
            } else {
                CalyxExtMemToAxiResultTypes.push_back(signal.type);
                CalyxExtMemToAxiResultNames.push_back(nameAttr);
            }
        }

        CalyxExtMemToAxiInputs.push_back(CalyxExtMemToAxiReadEnWiresRead[i]);
        CalyxExtMemToAxiArgNames.push_back(builder.getStringAttr("calyx_read_en"));
        CalyxExtMemToAxiInputs.push_back(CalyxExtMemToAxiWriteEnWiresRead[i]);
        CalyxExtMemToAxiArgNames.push_back(builder.getStringAttr("calyx_write_en"));
        CalyxExtMemToAxiInputs.push_back(clock);
        CalyxExtMemToAxiArgNames.push_back(builder.getStringAttr("clk"));
        CalyxExtMemToAxiInputs.push_back(reset);
        CalyxExtMemToAxiArgNames.push_back(builder.getStringAttr("rst"));

        CalyxExtMemToAxiResultTypes.push_back(builder.getI1Type());
        CalyxExtMemToAxiResultNames.push_back(builder.getStringAttr("calyx_done"));

        mlir::StringAttr nameAttr = builder.getStringAttr("calyx_ext_mem_to_axi_" + std::to_string(i));
        CalyxExtMemToAxiInstances[i] =
            builder.create<circt::hw::InstanceOp>(CalyxExtMemToAxiResultTypes,
                                                  nameAttr, "CalyxExtMemToAxi",
                                                  CalyxExtMemToAxiInputs,
                                                  builder.getArrayAttr(CalyxExtMemToAxiArgNames),
                                                  builder.getArrayAttr(CalyxExtMemToAxiResultNames),
                                                  /*parameters=*/builder.getArrayAttr({}),
                                                  /*sym_name=*/nameAttr);

        for (const auto &signal : axi4ManagerSignals[i]) {
            if (signal.direction == circt::hw::PortDirection::OUTPUT) {
                std::string basename = fullAxiSignalNameIdGetBasename(signal.name.str());
                int hwModIdx = moduleGetOutputIndex(hwModuleOp, signal.name.str());
                int memIdx = moduleGetOutputIndex(CalyxExtMemToAxiInstances[i], toFullAxiManagerSignalName(basename));
                builder.create<circt::sv::AssignOp>(hwModuleOutputWires[hwModIdx], CalyxExtMemToAxiInstances[i].getResult(memIdx));
            }
        }
    }

    // Kernel arguments
    mlir::SmallVector<circt::sv::WireOp> kernelArgsWires(kernelArgs.size());
    mlir::SmallVector<mlir::Value> kernelArgsWiresRead(kernelArgs.size());
    for (unsigned i = 0; i < kernelArgs.size(); i++) {
        mlir::Type type = builder.getIntegerType(kernelArgs[i].getHWBits());
        auto name = kernelArgs[i].name;

        kernelArgsWires[i] = builder.create<circt::sv::WireOp>(type, name);
        kernelArgsWiresRead[i] = builder.create<circt::sv::ReadInOutOp>(kernelArgsWires[i]);

        controlAxiResultTypes.push_back(type);
        controlAxiResultNames.push_back(builder.getStringAttr(name));
    }

    circt::hw::InstanceOp controlAxiInstance =
        builder.create<circt::hw::InstanceOp>(controlAxiResultTypes,
                                              builder.getStringAttr("control_axi"), "ControlAXI",
                                              controlAxiInputs,
                                              builder.getArrayAttr(controlAxiArgNames),
                                              builder.getArrayAttr(controlAxiResultNames),
                                              /*parameters=*/builder.getArrayAttr({}),
                                              /*sym_name=*/builder.getStringAttr("control_axi"));

    for (const auto &signal : axi4LiteSubordinateSignals) {
        if (signal.direction == circt::hw::PortDirection::OUTPUT) {
            int i = moduleGetOutputIndex(hwModuleOp, signal.name.str());
            int j = moduleGetOutputIndex(controlAxiInstance, signal.name.str());
            builder.create<circt::sv::AssignOp>(hwModuleOutputWires[i], controlAxiInstance.getResult(j));
        }
    }

    // Kernel instance
    mlir::SmallVector<mlir::Value> kernelInputs;
    mlir::SmallVector<mlir::Type> kernelResultTypes;
    mlir::SmallVector<mlir::Attribute> kernelArgNames, kernelResultNames;

    for (const auto &arg : kernelArgs) {
        int idx = moduleGetOutputIndex(controlAxiInstance, arg.name);
        kernelInputs.push_back(controlAxiInstance.getResult(idx));
        kernelArgNames.push_back(builder.getStringAttr(arg.name));
    }

    for (unsigned i = 0; i < numBufferArgs; i++) {
        kernelInputs.push_back(moduleGetInputValue(toFullAxiManagerSignalNameId(i, "rdata")));
        kernelArgNames.push_back(builder.getStringAttr("ext_mem" + std::to_string(i) + "_read_data"));
        kernelInputs.push_back(CalyxExtMemToAxiInstances[i].getResult(
            moduleGetOutputIndex(CalyxExtMemToAxiInstances[i], "calyx_done")));
        kernelArgNames.push_back(builder.getStringAttr("ext_mem" + std::to_string(i) + "_done"));

        kernelResultTypes.push_back(builder.getIntegerType(M_AXI_DATA_WIDTH));
        kernelResultNames.push_back(builder.getStringAttr("ext_mem" + std::to_string(i) + "_write_data"));
        kernelResultTypes.push_back(builder.getIntegerType(M_AXI_ADDR_WIDTH));
        kernelResultNames.push_back(builder.getStringAttr("ext_mem" + std::to_string(i) + "_addr0"));
        kernelResultTypes.push_back(builder.getI1Type());
        kernelResultNames.push_back(builder.getStringAttr("ext_mem" + std::to_string(i) + "_write_en"));
        kernelResultTypes.push_back(builder.getI1Type());
        kernelResultNames.push_back(builder.getStringAttr("ext_mem" + std::to_string(i) + "_read_en"));
    }

    kernelInputs.push_back(clock);
    kernelArgNames.push_back(builder.getStringAttr("clk"));
    kernelInputs.push_back(reset);
    kernelArgNames.push_back(builder.getStringAttr("reset"));
    kernelInputs.push_back(controlAxiInstance.getResult(moduleGetOutputIndex(controlAxiInstance, "ap_start")));
    kernelArgNames.push_back(builder.getStringAttr("go"));

    kernelResultTypes.push_back(builder.getI1Type());
    kernelResultNames.push_back(builder.getStringAttr("done"));

    circt::hw::InstanceOp kernelInstance =
        builder.create<circt::hw::InstanceOp>(kernelResultTypes,
                                              builder.getStringAttr("kernel"), kernelName,
                                              kernelInputs,
                                              builder.getArrayAttr(kernelArgNames),
                                              builder.getArrayAttr(kernelResultNames),
                                              /*parameters=*/builder.getArrayAttr({}),
                                              /*sym_name=*/builder.getStringAttr("kernel"));

    for (unsigned i = 0; i < numBufferArgs; i++) {
        int idxIn, idxOut;
        idxIn = moduleGetOutputIndex(kernelInstance, "ext_mem" + std::to_string(i) + "_addr0");
        idxOut = moduleGetOutputIndex(hwModuleOp, toFullAxiManagerSignalNameId(i, "awaddr"));
        builder.create<circt::sv::AssignOp>(hwModuleOutputWires[idxOut], kernelInstance.getResult(idxIn));
        idxOut = moduleGetOutputIndex(hwModuleOp, toFullAxiManagerSignalNameId(i, "araddr"));
        builder.create<circt::sv::AssignOp>(hwModuleOutputWires[idxOut], kernelInstance.getResult(idxIn));

        idxIn = moduleGetOutputIndex(kernelInstance, "ext_mem" + std::to_string(i) + "_write_data");
        idxOut = moduleGetOutputIndex(hwModuleOp, toFullAxiManagerSignalNameId(i, "wdata"));
        builder.create<circt::sv::AssignOp>(hwModuleOutputWires[idxOut], kernelInstance.getResult(idxIn));

        idxIn = moduleGetOutputIndex(kernelInstance, "ext_mem" + std::to_string(i) + "_read_en");
        builder.create<circt::sv::AssignOp>(CalyxExtMemToAxiReadEnWires[i], kernelInstance.getResult(idxIn));

        idxIn = moduleGetOutputIndex(kernelInstance, "ext_mem" + std::to_string(i) + "_write_en");
        builder.create<circt::sv::AssignOp>(CalyxExtMemToAxiWriteEnWires[i], kernelInstance.getResult(idxIn));
    }

    builder.create<circt::sv::AssignOp>(apDone,
                                        kernelInstance.getResult(moduleGetOutputIndex(kernelInstance, "done")));

    // Set module output operands
    auto outputOp = hwModuleOp.getBodyBlock()->getTerminator();
    outputOp->setOperands(hwModuleOutputValues);
}

void CodeGen_CIRCT::generateKernelXml(llvm::raw_ostream &os, const std::string &kernelName, const FlattenedKernelArgs &kernelArgs) {
    XMLDocument doc;
    doc.InsertFirstChild(doc.NewDeclaration());

    XMLElement *pRoot = doc.NewElement("root");
    pRoot->SetAttribute("versionMajor", 1);
    pRoot->SetAttribute("versionMinor", 6);
    doc.InsertEndChild(pRoot);

    XMLElement *pKernel = doc.NewElement("kernel");
    pKernel->SetAttribute("name", "toplevel");
    pKernel->SetAttribute("language", "ip_c");
    pKernel->SetAttribute("vlnv", std::string("halide-lang.org:" + kernelName + ":toplevel:1.0").c_str());
    pKernel->SetAttribute("attributes", "");
    pKernel->SetAttribute("preferredWorkGroupSizeMultiple", 0);
    pKernel->SetAttribute("workGroupSize", 1);
    pKernel->SetAttribute("interrupt", "false");
    pKernel->SetAttribute("hwControlProtocol", "ap_ctrl_hs");
    pRoot->InsertEndChild(pKernel);

    auto toHexStr = [](uint64_t value) {
        std::stringstream ss;
        ss << "0x" << std::uppercase << std::hex << value;
        return ss.str();
    };

    auto genPort = [&](std::string name, std::string mode, uint64_t range, int dataWidth) {
        XMLElement *pPort = doc.NewElement("port");
        pPort->SetAttribute("name", name.c_str());
        pPort->SetAttribute("mode", mode.c_str());
        pPort->SetAttribute("range", toHexStr(range).c_str());
        pPort->SetAttribute("dataWidth", dataWidth);
        pPort->SetAttribute("portType", "addressable");
        pPort->SetAttribute("base", toHexStr(0).c_str());
        return pPort;
    };

    auto genArg = [&](std::string name, int addressQualifier, int id, std::string port, uint64_t size, uint64_t offset,
                      std::string type, uint64_t hostOffset, uint64_t hostSize) {
        XMLElement *pArg = doc.NewElement("arg");
        pArg->SetAttribute("name", name.c_str());
        pArg->SetAttribute("addressQualifier", addressQualifier);
        pArg->SetAttribute("id", id);
        pArg->SetAttribute("port", port.c_str());
        pArg->SetAttribute("size", toHexStr(size).c_str());
        pArg->SetAttribute("offset", toHexStr(offset).c_str());
        pArg->SetAttribute("type", type.c_str());
        pArg->SetAttribute("hostOffset", toHexStr(hostOffset).c_str());
        pArg->SetAttribute("hostSize", toHexStr(hostSize).c_str());
        return pArg;
    };

    auto genTypeStr = [](Type type) {
        switch (type.code()) {
        case Type::Int:
        default:
            return std::string("int");
        case Type::UInt:
            return std::string("uint");
        case Type::Float:
            return std::string("float");
        }
    };

    XMLElement *pPorts = doc.NewElement("ports");
    XMLElement *pArgs = doc.NewElement("args");

    pPorts->InsertEndChild(genPort(AXI_CONTROL_PREFIX, "slave", 0x1000, 32));

    uint64_t bufCnt = 0;
    uint64_t argIdx = 0;
    uint64_t argOffset = XRT_KERNEL_ARGS_OFFSET;
    for (const auto &arg : kernelArgs) {
        if (arg.isPointer) {
            pPorts->InsertEndChild(genPort(getAxiManagerSignalNamePrefixId(bufCnt), "master", std::numeric_limits<uint64_t>::max(), M_AXI_DATA_WIDTH));
            pArgs->InsertEndChild(genArg(arg.name, 1, argIdx, getAxiManagerSignalNamePrefixId(bufCnt), 8, argOffset, genTypeStr(arg.type) + "*", 0, 8));
            bufCnt++;
        } else {
            pArgs->InsertEndChild(genArg(arg.name, 0, argIdx, AXI_CONTROL_PREFIX, 4, argOffset, genTypeStr(arg.type), 0, arg.type.bytes()));
        }
        argOffset += 8;
        argIdx++;
    }

    pKernel->InsertEndChild(pPorts);
    pKernel->InsertEndChild(pArgs);

    XMLPrinter printer;
    doc.Print(&printer);
    os << printer.CStr();
}

void CodeGen_CIRCT::portsAddAXI4ManagerSignalsPrefix(mlir::ImplicitLocOpBuilder &builder, const std::string &prefix,
                                                     int addrWidth, int dataWidth,
                                                     mlir::SmallVector<circt::hw::PortInfo> &ports) {
    struct SignalInfo {
        std::string name;
        int size;
        bool input;
    };

    const mlir::SmallVector<SignalInfo> signals = {
        // Read address channel
        {"arvalid", 1, false},
        {"arready", 1, true},
        {"arlen", 8, false},
        // Read data channel
        {"rvalid", 1, true},
        {"rready", 1, false},
        {"rlast", 1, true},
        // Write address channel
        {"awvalid", 1, false},
        {"awready", 1, true},
        {"awlen", 8, false},
        // Write data channel
        {"wvalid", 1, false},
        {"wready", 1, true},
        {"wstrb", dataWidth / 8, false},
        {"wlast", 1, false},
        // Write response channel
        {"bvalid", 1, true},
        {"bready", 1, false},
    };

    for (const auto &signal : signals) {
        ports.push_back(circt::hw::PortInfo{builder.getStringAttr(prefix + signal.name),
                                            signal.input ? circt::hw::PortDirection::INPUT : circt::hw::PortDirection::OUTPUT,
                                            builder.getIntegerType(signal.size)});
    }
}

void CodeGen_CIRCT::portsAddAXI4LiteSubordinateSignals(mlir::ImplicitLocOpBuilder &builder, int addrWidth, int dataWidth, mlir::SmallVector<circt::hw::PortInfo> &ports) {
    struct SignalInfo {
        std::string name;
        int size;
        bool input;
    };
    const mlir::SmallVector<SignalInfo> signals = {
        // Read address channel
        {"arvalid", 1, true},
        {"arready", 1, false},
        {"araddr", addrWidth, true},
        // Read data channel
        {"rvalid", 1, false},
        {"rready", 1, true},
        {"rdata", dataWidth, false},
        {"rresp", 2, false},
        // Write address channel
        {"awvalid", 1, true},
        {"awready", 1, false},
        {"awaddr", addrWidth, true},
        // Write data channel
        {"wvalid", 1, true},
        {"wready", 1, false},
        {"wdata", dataWidth, true},
        {"wstrb", dataWidth / 8, true},
        // Write response channel
        {"bvalid", 1, false},
        {"bready", 1, true},
        {"bresp", 2, false},
    };

    for (const auto &signal : signals) {
        ports.push_back(circt::hw::PortInfo{builder.getStringAttr(toFullAxiSubordinateSignalName(signal.name)),
                                            signal.input ? circt::hw::PortDirection::INPUT : circt::hw::PortDirection::OUTPUT,
                                            builder.getIntegerType(signal.size)});
    }
}

CodeGen_CIRCT::Visitor::Visitor(mlir::ImplicitLocOpBuilder &builder, const std::vector<std::string> &inputNames)
    : builder(builder) {

    mlir::func::FuncOp funcOp = cast<mlir::func::FuncOp>(builder.getBlock()->getParentOp());

    // Add function arguments to the symbol table
    for (unsigned int i = 0; i < funcOp.getFunctionType().getNumInputs(); i++) {
        std::string name = inputNames[i];
        sym_push(name, funcOp.getArgument(i));
    }
}

mlir::Value CodeGen_CIRCT::Visitor::codegen(const Expr &e) {
    internal_assert(e.defined());
    debug(4) << "Codegen (E): " << e.type() << ", " << e << "\n";
    value = mlir::Value();
    e.accept(this);
    internal_assert(value) << "Codegen of an expr did not produce a MLIR value\n"
                           << e;
    return value;
}

void CodeGen_CIRCT::Visitor::codegen(const Stmt &s) {
    internal_assert(s.defined());
    debug(4) << "Codegen (S): " << s << "\n";
    value = mlir::Value();
    s.accept(this);
}

void CodeGen_CIRCT::Visitor::visit(const IntImm *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    mlir::Type type = builder.getIntegerType(op->type.bits());
    value = builder.create<mlir::arith::ConstantOp>(type, builder.getIntegerAttr(type, op->value));
}

void CodeGen_CIRCT::Visitor::visit(const UIntImm *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    mlir::Type type = builder.getIntegerType(op->type.bits());
    value = builder.create<mlir::arith::ConstantOp>(type, builder.getIntegerAttr(type, op->value));
}

void CodeGen_CIRCT::Visitor::visit(const FloatImm *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const StringImm *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const Cast *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    Halide::Type src = op->value.type();
    Halide::Type dst = op->type;
    mlir::Type dstType = builder.getIntegerType(dst.bits());

    debug(1) << "\tSrc type: " << src << "\n";
    debug(1) << "\tDst type: " << dst << "\n";

    value = codegen(op->value);

    if (src.is_int_or_uint() && dst.is_int_or_uint()) {
        if (dst.bits() > src.bits()) {
            if (src.is_int())
                value = builder.create<mlir::arith::ExtSIOp>(dstType, value);
            else
                value = builder.create<mlir::arith::ExtUIOp>(dstType, value);
        } else {
            value = builder.create<mlir::arith::TruncIOp>(dstType, value);
        }
    } else {
        assert(0);
    }
}

void CodeGen_CIRCT::Visitor::visit(const Reinterpret *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const Variable *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    debug(1) << "\tname: " << op->name << "\n";
    value = sym_get(op->name, true);
}

void CodeGen_CIRCT::Visitor::visit(const Add *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<mlir::arith::AddIOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_CIRCT::Visitor::visit(const Sub *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<mlir::arith::SubIOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_CIRCT::Visitor::visit(const Mul *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<mlir::arith::MulIOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_CIRCT::Visitor::visit(const Div *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    if (op->type.is_int()) {
        value = builder.create<mlir::arith::DivSIOp>(codegen(op->a), codegen(op->b));
    } else {
        value = builder.create<mlir::arith::DivUIOp>(codegen(op->a), codegen(op->b));
    }
}

void CodeGen_CIRCT::Visitor::visit(const Mod *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    if (op->type.is_int()) {
        value = builder.create<mlir::arith::RemSIOp>(codegen(op->a), codegen(op->b));
    } else {
        value = builder.create<mlir::arith::RemUIOp>(codegen(op->a), codegen(op->b));
    }
}

void CodeGen_CIRCT::Visitor::visit(const Min *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    std::string a_name = unique_name('a');
    std::string b_name = unique_name('b');
    Expr a = Variable::make(op->a.type(), a_name);
    Expr b = Variable::make(op->b.type(), b_name);
    value = codegen(Let::make(a_name, op->a, Let::make(b_name, op->b, select(a < b, a, b))));
}

void CodeGen_CIRCT::Visitor::visit(const Max *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    std::string a_name = unique_name('a');
    std::string b_name = unique_name('b');
    Expr a = Variable::make(op->a.type(), a_name);
    Expr b = Variable::make(op->b.type(), b_name);
    value = codegen(Let::make(a_name, op->a, Let::make(b_name, op->b, select(a > b, a, b))));
}

void CodeGen_CIRCT::Visitor::visit(const EQ *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::eq, codegen(op->a), codegen(op->b));
}

void CodeGen_CIRCT::Visitor::visit(const NE *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::ne, codegen(op->a), codegen(op->b));
}

void CodeGen_CIRCT::Visitor::visit(const LT *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    mlir::arith::CmpIPredicate predicate = op->type.is_int() ? mlir::arith::CmpIPredicate::slt :
                                                               mlir::arith::CmpIPredicate::ult;
    value = builder.create<mlir::arith::CmpIOp>(predicate, codegen(op->a), codegen(op->b));
}

void CodeGen_CIRCT::Visitor::visit(const LE *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    mlir::arith::CmpIPredicate predicate = op->type.is_int() ? mlir::arith::CmpIPredicate::sle :
                                                               mlir::arith::CmpIPredicate::ule;
    value = builder.create<mlir::arith::CmpIOp>(predicate, codegen(op->a), codegen(op->b));
}

void CodeGen_CIRCT::Visitor::visit(const GT *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    mlir::arith::CmpIPredicate predicate = op->type.is_int() ? mlir::arith::CmpIPredicate::sgt :
                                                               mlir::arith::CmpIPredicate::ugt;
    value = builder.create<mlir::arith::CmpIOp>(predicate, codegen(op->a), codegen(op->b));
}

void CodeGen_CIRCT::Visitor::visit(const GE *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    mlir::arith::CmpIPredicate predicate = op->type.is_int() ? mlir::arith::CmpIPredicate::sge :
                                                               mlir::arith::CmpIPredicate::uge;
    value = builder.create<mlir::arith::CmpIOp>(predicate, codegen(op->a), codegen(op->b));
}

void CodeGen_CIRCT::Visitor::visit(const And *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    // value = builder.create<mlir::arith::CmpIOp>(predicate, codegen(op->a), codegen(op->b));
}

void CodeGen_CIRCT::Visitor::visit(const Or *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const Not *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    mlir::Value a = codegen(op->a);
    mlir::Value allZeroes = builder.create<mlir::arith::ConstantOp>(a.getType(), builder.getIntegerAttr(a.getType(), 0));
    value = builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::eq, a, allZeroes);
}

void CodeGen_CIRCT::Visitor::visit(const Select *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    // mlir::Value trueValue = codegen(op->true_value);
    // mlir::Value falseValue = codegen(op->false_value);

    // mlir::Block *trueBlock = builder.createBlock();

    // builder.getLoc()
#if 0
    value = builder.create<mlir::cf::CondBranchOp>(
        codegen(op->condition),
        /*thenBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          b.create<mlir::scf::YieldOp>(loc, trueValue);
        },
        /*elseBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          b.create<mlir::scf::YieldOp>(loc, falseValue);
        }).getResult(0);
#endif
#if 0
    mlir::IntegerAttr allOnesAttr = builder.getIntegerAttr(condition.getType(),
        llvm::APInt::getAllOnes(condition.getType().getIntOrFloatBitWidth()));
    mlir::Value allOnes = builder.create<mlir::arith::ConstantOp>(allOnesAttr);
    mlir::Value conditionNeg = builder.create<circt::comb::XorOp>(condition, allOnes);

    builder.create<circt::calyx::AssignOp>(value, trueValue, condition);
    builder.create<circt::calyx::AssignOp>(value, falseValue, conditionNeg);
#endif
#if 0
    value = builder.create<mlir::scf::IfOp>(
        codegen(op->condition),
        /*thenBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          b.create<mlir::scf::YieldOp>(loc, trueValue);
        },
        /*elseBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          b.create<mlir::scf::YieldOp>(loc, falseValue);
        }).getResult(0);
#endif
}

void CodeGen_CIRCT::Visitor::visit(const Load *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    debug(1) << "\tName: " << op->name << "\n";

    mlir::Value baseAddr = sym_get(op->name);
    mlir::Value index = builder.create<mlir::arith::ExtUIOp>(builder.getI64Type(), codegen(op->index));
    mlir::Value elementSize = builder.create<mlir::arith::ConstantOp>(builder.getI64IntegerAttr(op->type.bytes()));
    mlir::Value offset = builder.create<mlir::arith::MulIOp>(index, elementSize);
    mlir::Value loadAddress = builder.create<mlir::arith::AddIOp>(baseAddr, offset);
    mlir::Value loadAddressAsIndex = builder.create<mlir::arith::IndexCastOp>(builder.getIndexType(), loadAddress);

    value = builder.create<mlir::memref::LoadOp>(sym_get(op->name + ".buffer"), mlir::ValueRange{loadAddressAsIndex});
}

void CodeGen_CIRCT::Visitor::visit(const Ramp *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const Broadcast *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const Call *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    debug(1) << "\tName: " << op->name << "\n";
    debug(1) << "\tCall type: " << op->call_type << "\n";
    for (const Expr &e : op->args)
        debug(1) << "\tArg: " << e << "\n";

    mlir::Type op_type = builder.getIntegerType(op->type.bits());

    if (op->name == Call::buffer_get_host) {
        auto name = op->args[0].as<Variable>()->name;
        name = name.substr(0, name.find(".buffer"));
        value = sym_get(name);
    } else if (op->name == Call::buffer_get_min) {
        auto name = op->args[0].as<Variable>()->name;
        name = name.substr(0, name.find(".buffer"));
        int32_t d = op->args[1].as<IntImm>()->value;
        value = sym_get(name + "_dim_" + std::to_string(d) + "_min");
    } else if (op->name == Call::buffer_get_extent) {
        auto name = op->args[0].as<Variable>()->name;
        name = name.substr(0, name.find(".buffer"));
        int32_t d = op->args[1].as<IntImm>()->value;
        value = sym_get(name + "_dim_" + std::to_string(d) + "_extent");
    } else if (op->name == Call::buffer_get_stride) {
        auto name = op->args[0].as<Variable>()->name;
        name = name.substr(0, name.find(".buffer"));
        int32_t d = op->args[1].as<IntImm>()->value;
        value = sym_get(name + "_dim_" + std::to_string(d) + "_stride");
    } else {
        // Just return 1 for now
        value = builder.create<mlir::arith::ConstantOp>(op_type, builder.getIntegerAttr(op_type, 1));
    }
}

void CodeGen_CIRCT::Visitor::visit(const Let *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    sym_push(op->name, codegen(op->value));
    value = codegen(op->body);
    sym_pop(op->name);
}

void CodeGen_CIRCT::Visitor::visit(const LetStmt *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    debug(1) << "Contents:"
             << "\n";
    debug(1) << "\tName: " << op->name << "\n";
    sym_push(op->name, codegen(op->value));
    codegen(op->body);
    sym_pop(op->name);
}

void CodeGen_CIRCT::Visitor::visit(const AssertStmt *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const ProducerConsumer *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    debug(1) << "\tName: " << op->name << "\n";
    debug(1) << "\tIs producer: " << op->is_producer << "\n";
    codegen(op->body);
}

void CodeGen_CIRCT::Visitor::visit(const For *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    debug(1) << "\tName: " << op->name << "\n";
    debug(1) << "\tMin: " << op->min << "\n";
    debug(1) << "\tExtent: " << op->extent << "\n";
    static const char *for_types[] = {
        "Serial",
        "Parallel",
        "Vectorized",
        "Unrolled",
        "Extern",
        "GPUBlock",
        "GPUThread",
        "GPULane",
    };
    debug(1) << "\tForType: " << for_types[unsigned(op->for_type)] << "\n";

    mlir::Value min = codegen(op->min);
    mlir::Value extent = codegen(op->extent);
    mlir::Value max = builder.create<mlir::arith::AddIOp>(min, extent);
    mlir::Value lb = builder.create<mlir::arith::IndexCastOp>(builder.getIndexType(), min);
    mlir::Value ub = builder.create<mlir::arith::IndexCastOp>(builder.getIndexType(), max);
    mlir::Value step = builder.create<mlir::arith::ConstantIndexOp>(1);

    mlir::scf::ForOp forOp = builder.create<mlir::scf::ForOp>(lb, ub, step);
    mlir::Region &forBody = forOp.getLoopBody();

    mlir::ImplicitLocOpBuilder prevBuilder = builder;
    builder = mlir::ImplicitLocOpBuilder::atBlockBegin(forBody.getLoc(), &forBody.front());

    sym_push(op->name, builder.create<mlir::arith::IndexCastOp>(max.getType(), forOp.getInductionVar()));
    codegen(op->body);
    sym_pop(op->name);

    builder = prevBuilder;
}

void CodeGen_CIRCT::Visitor::visit(const Store *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    debug(1) << "\tName: " << op->name << "\n";

    mlir::Value baseAddr = sym_get(op->name);
    mlir::Value index = builder.create<mlir::arith::ExtUIOp>(builder.getI64Type(), codegen(op->index));
    mlir::Value elementSize = builder.create<mlir::arith::ConstantOp>(builder.getI64IntegerAttr(op->value.type().bytes()));
    mlir::Value offset = builder.create<mlir::arith::MulIOp>(index, elementSize);
    mlir::Value storeAddress = builder.create<mlir::arith::AddIOp>(baseAddr, offset);
    mlir::Value storeAddressAsIndex = builder.create<mlir::arith::IndexCastOp>(builder.getIndexType(), storeAddress);

    builder.create<mlir::memref::StoreOp>(codegen(op->value), sym_get(op->name + ".buffer"), mlir::ValueRange{storeAddressAsIndex});
}

void CodeGen_CIRCT::Visitor::visit(const Provide *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const Allocate *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    int32_t size = op->constant_allocation_size();

    debug(1) << "  name: " << op->name << "\n";
    debug(1) << "  type: " << op->type << "\n";
    debug(1) << "  memory_type: " << int(op->memory_type) << "\n";
    debug(1) << "  size: " << size << "\n";

    for (auto &ext : op->extents) {
        debug(1) << "  ext: " << ext << "\n";
    }

    codegen(op->body);
}

void CodeGen_CIRCT::Visitor::visit(const Free *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const Realize *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const Block *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    // Peel blocks of assertions with pure conditions
    const AssertStmt *a = op->first.as<AssertStmt>();
    if (a && is_pure(a->condition)) {
        std::vector<const AssertStmt *> asserts;
        asserts.push_back(a);
        Stmt s = op->rest;
        while ((op = s.as<Block>()) && (a = op->first.as<AssertStmt>()) && is_pure(a->condition) && asserts.size() < 63) {
            asserts.push_back(a);
            s = op->rest;
        }
        // codegen_asserts(asserts);
        codegen(s);
    } else {
        codegen(op->first);
        codegen(op->rest);
    }
}

void CodeGen_CIRCT::Visitor::visit(const IfThenElse *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    debug(1) << "Contents:"
             << "\n";
    debug(1) << "\tcondition: " << op->condition << "\n";
    debug(1) << "\tthen_case: " << op->then_case << "\n";
    debug(1) << "\telse_case: " << op->else_case << "\n";

    codegen(op->condition);
    codegen(op->then_case);

    if (op->else_case.defined()) {
        codegen(op->else_case);
        // halide_buffer_t
    } else {
    }
}

void CodeGen_CIRCT::Visitor::visit(const Evaluate *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    codegen(op->value);

    // Discard result
    value = mlir::Value();
}

void CodeGen_CIRCT::Visitor::visit(const Shuffle *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const VectorReduce *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const Prefetch *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const Fork *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const Acquire *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const Atomic *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::sym_push(const std::string &name, mlir::Value value) {
    // value.getDefiningOp()->setAttr("sv.namehint", builder.getStringAttr(name));
    symbol_table.push(name, value);
}

void CodeGen_CIRCT::Visitor::sym_pop(const std::string &name) {
    symbol_table.pop(name);
}

mlir::Value CodeGen_CIRCT::Visitor::sym_get(const std::string &name, bool must_succeed) const {
    // look in the symbol table
    if (!symbol_table.contains(name)) {
        if (must_succeed) {
            std::ostringstream err;
            err << "Symbol not found: " << name << "\n";

            if (debug::debug_level() > 0) {
                err << "The following names are in scope:\n"
                    << symbol_table << "\n";
            }

            internal_error << err.str();
        } else {
            return nullptr;
        }
    }
    return symbol_table.get(name);
}

}  // namespace Internal
}  // namespace Halide
