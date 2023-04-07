#include <fstream>
#include <string>
#include <vector>

#include <tinyxml2.h>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/ToolOutputFile.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Transforms/Passes.h>

#include <circt/Conversion/ExportVerilog.h>
#include <circt/Conversion/FSMToSV.h>
#include <circt/Dialect/Calyx/CalyxDialect.h>
#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/FSM/FSMDialect.h>
#include <circt/Dialect/FSM/FSMOps.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/HW/HWOps.h>
#include <circt/Dialect/HW/HWPasses.h>
#include <circt/Dialect/SV/SVAttributes.h>
#include <circt/Dialect/SV/SVOps.h>
#include <circt/Dialect/Seq/SeqDialect.h>
#include <circt/Dialect/Seq/SeqOps.h>
#include <circt/Dialect/Seq/SeqPasses.h>
#include <circt/Support/LoweringOptions.h>

#include "CodeGen_Accelerator_Dev.h"
#include "CodeGen_CIRCT_Dev.h"
#include "CodeGen_CIRCT_Xilinx_Dev.h"
#include "Debug.h"
#include "IROperator.h"
#include "IRVisitor.h"
#include "Module.h"
#include "Scope.h"
#include "Util.h"

using namespace tinyxml2;

namespace Halide {

namespace Internal {

namespace {

class CodeGen_CIRCT_Xilinx_Dev : public CodeGen_Accelerator_Dev {
public:
    CodeGen_CIRCT_Xilinx_Dev(const Target &target);

    void add_kernel(Stmt stmt,
                    const std::string &name,
                    const std::vector<DeviceArgument> &args) override;

    void init_module() override {
    }

    std::string get_current_kernel_name() override {
        return currentKernelName;
    }

    std::string api_unique_name() override {
        return "xrt";
    }

    std::string print_accelerator_name(const std::string &name) override {
        return name;
    }

private:
    struct SignalInfo {
        std::string name;
        int size;
        bool input;
    };

    // XRT-Managed Kernels Control Requirements
    // See https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/Control-Requirements-for-XRT-Managed-Kernels
    static constexpr uint64_t XRT_KERNEL_ARGS_OFFSET = 0x10;
    static constexpr int M_AXI_ADDR_WIDTH = 64;
    static constexpr int M_AXI_DATA_WIDTH = 512;
    static constexpr int S_AXI_ADDR_WIDTH = 12;
    static constexpr int S_AXI_DATA_WIDTH = 32;

    static constexpr char AXI_MANAGER_PREFIX[] = "m_axi_";
    static constexpr char AXI_CONTROL_PREFIX[] = "s_axi_control";

    static void generateKernelXml(llvm::raw_ostream &os, const std::string &kernelName, const std::vector<DeviceArgument> &kernelArgs);
    static void generateCalyxExtMemToAxi(mlir::ImplicitLocOpBuilder &builder);
    static void generateControlAxi(mlir::ImplicitLocOpBuilder &builder, const std::vector<DeviceArgument> &kernelArgs);
    static void generateToplevel(mlir::ImplicitLocOpBuilder &builder, const std::string &kernelName, const std::vector<DeviceArgument> &kernelArgs);

    static void portsAddAXI4ManagerSignalsPrefix(mlir::ImplicitLocOpBuilder &builder, const std::string &prefix,
                                                 int addrWidth, int dataWidth,
                                                 mlir::SmallVector<circt::hw::PortInfo> &ports);
    static void portsAddAXI4LiteSubordinateSignals(mlir::ImplicitLocOpBuilder &builder, int addrWidth, int dataWidth,
                                                   mlir::SmallVector<circt::hw::PortInfo> &ports);

    static void portsAddCalyxExtMemInterfaceSubordinateSignals(mlir::ImplicitLocOpBuilder &builder,
                                                               int addrWidth, int dataWidth, mlir::SmallVector<circt::hw::PortInfo> &ports);

    static std::string getAxiManagerSignalNamePrefixId(int id) {
        return "m" + std::string(id < 10 ? "0" : "") + std::to_string(id) + "_axi";
    }

    static std::string toFullAxiManagerSignalName(const std::string &name) {
        return std::string(AXI_MANAGER_PREFIX) + name;
    }

    static std::string toFullAxiManagerSignalNameId(int id, const std::string &name) {
        return getAxiManagerSignalNamePrefixId(id) + "_" + name;
    }

    static std::string toFullAxiSubordinateSignalName(const std::string &name) {
        return std::string(AXI_CONTROL_PREFIX) + "_" + name;
    }

    static std::string fullAxiSignalNameIdGetBasename(const std::string &name) {
        std::string token = "axi_";
        return name.substr(name.find(token) + token.size());
    }

    template<typename ModType>
    static mlir::Value hwModuleGetInputValue(ModType &mod, const std::string &name) {
        const auto &names = mod.getArgNames();
        for (unsigned int i = 0; i < mod.getNumArguments(); i++) {
            if (names[i].template cast<mlir::StringAttr>().str() == name)
                return mod.getArgument(i);
        }
        assert(0);
    }

    static mlir::Value hwModuleGetAxiInputValue(circt::hw::HWModuleOp &mod, const std::string &name) {
        return hwModuleGetInputValue(mod, toFullAxiSubordinateSignalName(name));
    };

    template<typename ModType>
    static unsigned int hwModuleGetOutputIndex(ModType &mod, const std::string &name) {
        unsigned int i;
        const auto &names = mod.getResultNames();
        for (i = 0; i < mod.getNumResults(); i++) {
            if (names[i].template cast<mlir::StringAttr>().str() == name)
                break;
        }
        assert(i < mod.getNumResults());
        return i;
    };

    static unsigned int hwModuleGetAxiOutputIndex(circt::hw::HWModuleOp &mod, const std::string &name) {
        return hwModuleGetOutputIndex(mod, toFullAxiSubordinateSignalName(name));
    };

    static mlir::Value fsmGetInputValue(circt::fsm::MachineOp &fsm, const std::string &name) {
        for (unsigned int i = 0; i < fsm.getNumArguments(); i++) {
            if (fsm.getArgName(i) == name)
                return fsm.getArgument(i);
        }
        return mlir::BlockArgument();
    }

    static mlir::Value fsmGetAxiInputValue(circt::fsm::MachineOp &fsm, const std::string &name) {
        return fsmGetInputValue(fsm, toFullAxiManagerSignalName(name));
    }

    static unsigned int fsmGetOutputIndex(circt::fsm::MachineOp &fsm, const std::string &name) {
        unsigned int i = 0;
        for (i = 0; i < fsm.getNumResults(); i++) {
            if (fsm.getResName(i) == name)
                break;
        }
        assert(i < fsm.getNumResults());
        return i;
    };

    static unsigned int fsmGetAxiOutputIndex(circt::fsm::MachineOp &fsm, const std::string &name) {
        return fsmGetOutputIndex(fsm, toFullAxiManagerSignalName(name));
    };

    mlir::MLIRContext mlir_context;
    std::string currentKernelName;
};

CodeGen_CIRCT_Xilinx_Dev::CodeGen_CIRCT_Xilinx_Dev(const Target &target) {
    mlir_context.disableMultithreading();
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
    mlir_context.loadDialect<mlir::vector::VectorDialect>();
}

void CodeGen_CIRCT_Xilinx_Dev::add_kernel(Stmt stmt, const std::string &name, const std::vector<DeviceArgument> &args) {
    debug(1) << "[Adding kernel '" << name << "']\n";

    const std::string outputDir = "generated_" + name;

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

    CodeGen_CIRCT_Dev cg;
    std::string calyxOutput;
    bool ret = cg.compile(loc, mlir_module, stmt, name, args, calyxOutput, M_AXI_DATA_WIDTH);
    internal_assert(ret) << "Compilation of " << name << " failed\n";

    // Create output directory (if it doesn't exist)
    llvm::sys::fs::create_directories(outputDir);

    // Emit Calyx
    auto output = mlir::openOutputFile(outputDir + "/" + name + ".futil");
    if (output) {
        output->os() << calyxOutput;
        output->keep();
    }

    // Create Calyx external memory to AXI interface
    builder = mlir::ImplicitLocOpBuilder::atBlockEnd(loc, mlir_module.getBody());
    debug(1) << "[Adding CalyxExtMemToAxi]\n";
    generateCalyxExtMemToAxi(builder);

    // Add AXI control
    builder = mlir::ImplicitLocOpBuilder::atBlockEnd(loc, mlir_module.getBody());
    debug(1) << "[Adding Control AXI]\n";
    generateControlAxi(builder, args);

    // Add toplevel
    builder = mlir::ImplicitLocOpBuilder::atBlockEnd(loc, mlir_module.getBody());
    debug(1) << "[Adding Toplevel]\n";
    generateToplevel(builder, name, args);

    debug(1) << "[FSM to SV] Start.\n";
    mlir::PassManager pmFSMtoSV(mlir_module.getContext());
    pmFSMtoSV.addPass(circt::createConvertFSMToSVPass());
    pmFSMtoSV.addPass(mlir::createCanonicalizerPass());
    pmFSMtoSV.addPass(circt::seq::createSeqLowerToSVPass());
    pmFSMtoSV.addPass(mlir::createCanonicalizerPass());

    auto pmFSMtoSVRunResult = pmFSMtoSV.run(mlir_module);
    debug(1) << "[FSM to SV] Result: " << pmFSMtoSVRunResult.succeeded() << "\n";
    if (!pmFSMtoSVRunResult.succeeded()) {
        debug(1) << "[FSM to SV] MLIR:\n";
        mlir_module.dump();
    }
    internal_assert(pmFSMtoSVRunResult.succeeded());

    // Emit Verilog
    if (pmFSMtoSVRunResult.succeeded()) {
        debug(1) << "[Exporting Verilog]\n";
        auto exportVerilogResult = circt::exportSplitVerilog(mlir_module, outputDir);
        debug(1) << "[Export Verilog] Result: " << exportVerilogResult.succeeded() << "\n";

        debug(1) << "[Generating kernel.xml]\n";
        auto output = mlir::openOutputFile(outputDir + "/kernel.xml");
        if (output) {
            generateKernelXml(output->os(), name, args);
            output->keep();
        }
    }

    debug(1) << "[Added kernel '" << name << "']\n";

    currentKernelName = name;
}

void CodeGen_CIRCT_Xilinx_Dev::generateCalyxExtMemToAxi(mlir::ImplicitLocOpBuilder &builder) {
    // AXI4 Manager signals
    mlir::SmallVector<circt::hw::PortInfo> axiSignals;
    portsAddAXI4ManagerSignalsPrefix(builder, AXI_MANAGER_PREFIX, M_AXI_ADDR_WIDTH, M_AXI_DATA_WIDTH, axiSignals);
    // Calyx external memory interface signals
    mlir::SmallVector<circt::hw::PortInfo> calyxExtMemSignals;
    portsAddCalyxExtMemInterfaceSubordinateSignals(builder, M_AXI_ADDR_WIDTH, M_AXI_DATA_WIDTH, calyxExtMemSignals);

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
    for (const auto &signal : calyxExtMemSignals) {
        if (signal.direction == circt::hw::PortDirection::INPUT) {
            inputs.push_back(signal.type);
            inputNames.push_back(builder.getStringAttr(signal.name.str()));
        } else {
            outputs.push_back(signal.type);
            outputNames.push_back(builder.getStringAttr(signal.name.str()));
        }
    }

    mlir::FunctionType fsmFunctionType = builder.getFunctionType(inputs, outputs);
    circt::fsm::MachineOp machineOp = builder.create<circt::fsm::MachineOp>("CalyxExtMemToAxi", "IDLE", fsmFunctionType);
    machineOp.setArgNamesAttr(builder.getArrayAttr(inputNames));
    machineOp.setResNamesAttr(builder.getArrayAttr(outputNames));

    mlir::Region &fsmBody = machineOp.getBody();
    mlir::ImplicitLocOpBuilder fsmBuilder = mlir::ImplicitLocOpBuilder::atBlockEnd(fsmBody.getLoc(), &fsmBody.front());

    mlir::Value value0 = fsmBuilder.create<circt::hw::ConstantOp>(fsmBuilder.getBoolAttr(false));
    mlir::SmallVector<mlir::Value> outputValues(machineOp.getNumResults());

    auto createAxiOutputConstantOp = [&](const std::string &name, int64_t value) {
        unsigned int idx = fsmGetAxiOutputIndex(machineOp, name);
        mlir::Type type = machineOp.getResultTypes()[idx];
        return builder.create<circt::hw::ConstantOp>(type, value);
    };

    auto setAxiOutputConstant = [&](const std::string &name, int64_t value) {
        outputValues[fsmGetAxiOutputIndex(machineOp, name)] = createAxiOutputConstantOp(name, value);
    };

    // Constant outputs
    setAxiOutputConstant("arlen", 0);   // 1 transfer
    setAxiOutputConstant("awlen", 0);   // 1 transfer
    setAxiOutputConstant("wstrb", -1);  // All bytes valid
    setAxiOutputConstant("wlast", 1);   // Last transfer

    auto addr0 = fsmGetInputValue(machineOp, "calyx_addr0");
    outputValues[fsmGetAxiOutputIndex(machineOp, "araddr")] = addr0;
    outputValues[fsmGetAxiOutputIndex(machineOp, "awaddr")] = addr0;
    outputValues[fsmGetAxiOutputIndex(machineOp, "wdata")] = fsmGetInputValue(machineOp, "calyx_write_data");

    // TODO
    outputValues[fsmGetOutputIndex(machineOp, "calyx_read_data")] = fsmGetAxiInputValue(machineOp, "rdata");

    size_t calyxDoneSignalIndex = fsmGetOutputIndex(machineOp, "calyx_done");
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
                returnOp.setOperand(fsmGetInputValue(machineOp, "calyx_write_en"));
            }

            {
                circt::fsm::TransitionOp transition = transitionsBuilder.create<circt::fsm::TransitionOp>("AR_HANDSHAKE");
                transition.ensureGuard(transitionsBuilder);
                circt::fsm::ReturnOp returnOp = transition.getGuardReturn();
                returnOp.setOperand(fsmGetInputValue(machineOp, "calyx_read_en"));
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
                returnOp.setOperand(fsmGetAxiInputValue(machineOp, "awready"));
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
                returnOp.setOperand(fsmGetAxiInputValue(machineOp, "wready"));
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
            outputValues[calyxDoneSignalIndex] = fsmGetAxiInputValue(machineOp, "bvalid");
            idleState.getOutputOp()->setOperands(outputValues);
        }
        mlir::Region &transitions = idleState.getTransitions();
        mlir::ImplicitLocOpBuilder transitionsBuilder = mlir::ImplicitLocOpBuilder::atBlockBegin(transitions.getLoc(), &transitions.front());
        {
            {
                circt::fsm::TransitionOp transition = transitionsBuilder.create<circt::fsm::TransitionOp>("IDLE");
                transition.ensureGuard(transitionsBuilder);
                circt::fsm::ReturnOp returnOp = transition.getGuardReturn();
                returnOp.setOperand(fsmGetAxiInputValue(machineOp, "bvalid"));
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
                returnOp.setOperand(fsmGetAxiInputValue(machineOp, "arready"));
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
            outputValues[calyxDoneSignalIndex] = fsmGetAxiInputValue(machineOp, "rvalid");
            idleState.getOutputOp()->setOperands(outputValues);
        }
        mlir::Region &transitions = idleState.getTransitions();
        mlir::ImplicitLocOpBuilder transitionsBuilder = mlir::ImplicitLocOpBuilder::atBlockBegin(transitions.getLoc(), &transitions.front());
        {
            {
                circt::fsm::TransitionOp transition = transitionsBuilder.create<circt::fsm::TransitionOp>("IDLE");
                transition.ensureGuard(transitionsBuilder);
                circt::fsm::ReturnOp returnOp = transition.getGuardReturn();
                returnOp.setOperand(fsmGetAxiInputValue(machineOp, "rvalid"));
            }
        }
    }
}

void CodeGen_CIRCT_Xilinx_Dev::generateControlAxi(mlir::ImplicitLocOpBuilder &builder, const std::vector<DeviceArgument> &kernelArgs) {
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
                                            builder.getIntegerType(argGetHWBits(arg))});
    }

    // Create ControllerAXI HW module
    circt::hw::HWModuleOp mod = builder.create<circt::hw::HWModuleOp>(builder.getStringAttr("ControlAXI"), ports);

    // This will hold the list of all the output Values
    mlir::SmallVector<mlir::Value> hwModuleOutputValues(mod.getNumResults());

    mlir::Value clock = hwModuleGetInputValue(mod, "clock");
    mlir::Value reset = hwModuleGetInputValue(mod, "reset");

    // ReadState FSM
    mlir::SmallVector<std::pair<std::string, mlir::Value>> readStateFsmInputs{
        {"arvalid", hwModuleGetAxiInputValue(mod, "arvalid")},
        {"rready", hwModuleGetAxiInputValue(mod, "rready")}};
    mlir::SmallVector<std::pair<std::string, unsigned int>> readStateFsmOutputs{
        {"arready", hwModuleGetAxiOutputIndex(mod, "arready")},
        {"rvalid", hwModuleGetAxiOutputIndex(mod, "rvalid")}};
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
            readStateFsmOutputTypes.push_back(mod.getResultTypes()[output.second]);
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
        {"awvalid", hwModuleGetAxiInputValue(mod, "awvalid")},
        {"wvalid", hwModuleGetAxiInputValue(mod, "wvalid")},
        {"bready", hwModuleGetAxiInputValue(mod, "bready")}};
    mlir::SmallVector<std::pair<std::string, unsigned int>> writeStateFsmOutputs{
        {"awready", hwModuleGetAxiOutputIndex(mod, "awready")},
        {"wready", hwModuleGetAxiOutputIndex(mod, "wready")},
        {"bvalid", hwModuleGetAxiOutputIndex(mod, "bvalid")}};
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
            writeStateFsmOutputTypes.push_back(mod.getResultTypes()[output.second]);
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
    builder.setInsertionPointToStart(mod.getBodyBlock());
    mlir::Value value0 = builder.create<circt::hw::ConstantOp>(builder.getBoolAttr(false));
    mlir::Value value1 = builder.create<circt::hw::ConstantOp>(builder.getBoolAttr(true));

    auto readFsmInstanceOp =
        builder.create<circt::fsm::HWInstanceOp>(readStateFsmOutputTypes, "ReadFSM", readFsmMachineOp.getSymName(),
                                                 readStateFsmInputValues, clock, reset);

    for (unsigned int i = 0; i < readFsmInstanceOp.getNumResults(); i++)
        hwModuleOutputValues[hwModuleGetAxiOutputIndex(mod, readStateFsmOutputs[i].first)] =
            readFsmInstanceOp.getResult(i);

    auto writeFsmInstanceOp =
        builder.create<circt::fsm::HWInstanceOp>(writeStateFsmOutputTypes, "WriteFSM", writeFsmMachineOp.getSymName(),
                                                 writeStateFsmInputValues, clock, reset);

    for (unsigned int i = 0; i < writeFsmMachineOp.getNumResults(); i++)
        hwModuleOutputValues[hwModuleGetAxiOutputIndex(mod, writeStateFsmOutputs[i].first)] =
            writeFsmInstanceOp.getResult(i);

    // ControlAXI logic
    hwModuleOutputValues[hwModuleGetAxiOutputIndex(mod, "rresp")] = builder.create<circt::hw::ConstantOp>(builder.getI2Type(), 0);
    hwModuleOutputValues[hwModuleGetAxiOutputIndex(mod, "bresp")] = builder.create<circt::hw::ConstantOp>(builder.getI2Type(), 0);

    // Write address. Store it into a register
    mlir::Value awaddr_next = builder.create<circt::sv::LogicOp>(axiAddrWidthType, "awaddr_next");
    mlir::Value awaddr_next_read = builder.create<circt::sv::ReadInOutOp>(awaddr_next);
    mlir::Value awaddr_reg = builder.create<circt::seq::CompRegOp>(awaddr_next_read, clock, reset,
                                                                   builder.create<circt::hw::ConstantOp>(axiAddrWidthType, 0), "awaddr_reg");
    mlir::Value writeToAwaddrReg = builder.create<circt::comb::AndOp>(hwModuleGetAxiInputValue(mod, "awvalid"),
                                                                      hwModuleOutputValues[hwModuleGetAxiOutputIndex(mod, "awready")]);
    builder.create<circt::sv::AlwaysCombOp>(/*bodyCtor*/ [&]() {
        builder.create<circt::sv::IfOp>(
            writeToAwaddrReg,
            /*thenCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(awaddr_next, hwModuleGetAxiInputValue(mod, "awaddr")); },
            /*elseCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(awaddr_next, awaddr_reg); });
    });

    const mlir::Value isWriteValidReady = builder.create<circt::comb::AndOp>(hwModuleGetAxiInputValue(mod, "wvalid"),
                                                                             hwModuleOutputValues[hwModuleGetAxiOutputIndex(mod, "wready")]);
    const mlir::Value isAreadValidReady = builder.create<circt::comb::AndOp>(hwModuleGetAxiInputValue(mod, "arvalid"),
                                                                             hwModuleOutputValues[hwModuleGetAxiOutputIndex(mod, "arready")]);
    const mlir::Value isReadValidReady = builder.create<circt::comb::AndOp>(hwModuleGetAxiInputValue(mod, "rready"),
                                                                            hwModuleOutputValues[hwModuleGetAxiOutputIndex(mod, "rvalid")]);
    // Control Register Signals (offset 0x00)
    mlir::Value int_ap_start_next = builder.create<circt::sv::LogicOp>(builder.getI1Type(), "int_ap_start_next");
    mlir::Value int_ap_start_next_read = builder.create<circt::sv::ReadInOutOp>(int_ap_start_next);
    mlir::Value int_ap_start = builder.create<circt::seq::CompRegOp>(int_ap_start_next_read, clock, reset, value0, "int_ap_start_reg");
    hwModuleOutputValues[hwModuleGetOutputIndex(mod, "ap_start")] = int_ap_start;
    mlir::Value isWaddr0x00 = builder.create<circt::comb::ICmpOp>(circt::comb::ICmpPredicate::eq, awaddr_reg,
                                                                  builder.create<circt::hw::ConstantOp>(axiAddrWidthType, 0));

    builder.create<circt::sv::AlwaysCombOp>(/*bodyCtor*/ [&]() {
        builder.create<circt::sv::IfOp>(
            builder.create<circt::comb::AndOp>(isWriteValidReady, isWaddr0x00),
            /*thenCtor*/ [&]() {
                mlir::Value apStartBit = builder.create<circt::comb::ExtractOp>(hwModuleGetAxiInputValue(mod, "wdata"), 0, 1);
                builder.create<circt::sv::BPAssignOp>(int_ap_start_next, apStartBit); },
            /*elseCtor*/ [&]() { builder.create<circt::sv::IfOp>(
                                     hwModuleGetInputValue(mod, "ap_done"),
                                     /*thenCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(int_ap_start_next, value0); },
                                     /*elseCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(int_ap_start_next, int_ap_start); }); });
    });

    mlir::Value int_ap_done_next = builder.create<circt::sv::LogicOp>(builder.getI1Type(), "int_ap_done_next");
    mlir::Value int_ap_done_next_read = builder.create<circt::sv::ReadInOutOp>(int_ap_done_next);
    mlir::Value int_ap_done = builder.create<circt::seq::CompRegOp>(int_ap_done_next_read, clock, reset, value0, "int_ap_done_reg");
    mlir::Value isRaddr0x00 = builder.create<circt::comb::ICmpOp>(circt::comb::ICmpPredicate::eq,
                                                                  hwModuleGetAxiInputValue(mod, "araddr"),
                                                                  builder.create<circt::hw::ConstantOp>(axiAddrWidthType, 0));

    builder.create<circt::sv::AlwaysCombOp>(/*bodyCtor*/ [&]() {
        builder.create<circt::sv::IfOp>(
            builder.create<circt::comb::AndOp>(isReadValidReady, isRaddr0x00),
            /*thenCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(int_ap_done_next, value0); },
            /*elseCtor*/ [&]() { builder.create<circt::sv::IfOp>(
                                     hwModuleGetInputValue(mod, "ap_done"),
                                     /*thenCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(int_ap_done_next, value1); },
                                     /*elseCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(int_ap_done_next, int_ap_done); }); });
    });

    mlir::Value int_ap_idle_next = builder.create<circt::sv::LogicOp>(builder.getI1Type(), "int_ap_idle_next");
    mlir::Value int_ap_idle_next_read = builder.create<circt::sv::ReadInOutOp>(int_ap_idle_next);
    mlir::Value int_ap_idle = builder.create<circt::seq::CompRegOp>(int_ap_idle_next_read, clock, reset, value1, "int_ap_idle_reg");

    builder.create<circt::sv::AlwaysCombOp>(/*bodyCtor*/ [&]() {
        builder.create<circt::sv::IfOp>(
            hwModuleGetInputValue(mod, "ap_done"),
            /*thenCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(int_ap_idle_next, value1); },
            /*elseCtor*/ [&]() { builder.create<circt::sv::IfOp>(
                                     hwModuleOutputValues[hwModuleGetOutputIndex(mod, "ap_start")],
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
                mlir::Value gieBit = builder.create<circt::comb::ExtractOp>(hwModuleGetAxiInputValue(mod, "wdata"), 0, 1);
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
                mlir::Value ierBits = builder.create<circt::comb::ExtractOp>(hwModuleGetAxiInputValue(mod, "wdata"), 0, 2);
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
                mlir::Value isrDoneBit = builder.create<circt::comb::ExtractOp>(hwModuleGetAxiInputValue(mod, "wdata"), 0, 1);
                mlir::Value isrReadyBit = builder.create<circt::comb::ExtractOp>(hwModuleGetAxiInputValue(mod, "wdata"), 1, 1);
                builder.create<circt::sv::BPAssignOp>(int_isr_done_next, isrDoneBit);
                builder.create<circt::sv::BPAssignOp>(int_isr_ready_next, isrReadyBit); },
            /*elseCtor*/ [&]() {
                builder.create<circt::sv::BPAssignOp>(int_isr_done_next, int_isr_done);
                builder.create<circt::sv::BPAssignOp>(int_isr_ready_next, int_isr_ready); });
    });

    // Create registers storing kernel arguments (scalars and buffer pointers)
    uint32_t argOffset = XRT_KERNEL_ARGS_OFFSET;
    for (const auto &arg : kernelArgs) {
        mlir::Type type = builder.getIntegerType(argGetHWBits(arg));
        mlir::Value zero = builder.create<circt::hw::ConstantOp>(type, 0);
        mlir::Value argRegNext = builder.create<circt::sv::LogicOp>(type, arg.name + "_next");
        mlir::Value argRegNextRead = builder.create<circt::sv::ReadInOutOp>(argRegNext);
        mlir::Value argReg = builder.create<circt::seq::CompRegOp>(argRegNextRead, clock, reset, zero, arg.name + "_reg");

        // Implement the store logic
        int size = 0;
        uint32_t subArgOffset = argOffset;
        while (size < argGetHWBits(arg)) {
            mlir::Value isWaddrToSubArgAddr = builder.create<circt::comb::ICmpOp>(circt::comb::ICmpPredicate::eq, awaddr_reg,
                                                                                  builder.create<circt::hw::ConstantOp>(axiAddrWidthType, subArgOffset));
            mlir::Value startBit = builder.create<circt::hw::ConstantOp>(axiAddrWidthType, size);
            mlir::Value bitsToUpdate = builder.create<circt::sv::IndexedPartSelectInOutOp>(argRegNext, startBit, 32);
            builder.create<circt::sv::AlwaysCombOp>(/*bodyCtor*/ [&]() {
                builder.create<circt::sv::IfOp>(
                    builder.create<circt::comb::AndOp>(isWriteValidReady, isWaddrToSubArgAddr),
                    /*thenCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(bitsToUpdate, hwModuleGetAxiInputValue(mod, "wdata")); },
                    /*elseCtor*/ [&]() { builder.create<circt::sv::BPAssignOp>(bitsToUpdate,
                                                                               builder.create<circt::sv::IndexedPartSelectOp>(argReg, startBit, 32)); });
            });

            size += 32;
            subArgOffset += 4;
        }
        hwModuleOutputValues[hwModuleGetOutputIndex(mod, arg.name)] = argReg;
        argOffset += 8;
    }

    // Holds the data that the host requested to read
    mlir::Value rdata_next = builder.create<circt::sv::LogicOp>(axiDataWidthType, "rdata_next");
    mlir::Value rdata_next_read = builder.create<circt::sv::ReadInOutOp>(rdata_next);
    mlir::Value rdata = builder.create<circt::seq::CompRegOp>(rdata_next_read, clock, reset,
                                                              builder.create<circt::hw::ConstantOp>(axiDataWidthType, 0),
                                                              "rdata_reg");
    hwModuleOutputValues[hwModuleGetAxiOutputIndex(mod, "rdata")] = rdata;

    // XRT registers + number of kernel arguments (each is considered to be have 8 bytes)
    const size_t numCases = XRT_KERNEL_ARGS_OFFSET / 4 + kernelArgs.size() * 2;

    builder.create<circt::sv::AlwaysCombOp>(/*bodyCtor*/ [&]() {
        builder.create<circt::sv::IfOp>(
            isAreadValidReady,
            /*thenCtor*/ [&]() {
                mlir::Value index = builder.create<circt::comb::ExtractOp>(hwModuleGetAxiInputValue(mod, "araddr"), 0, 12);
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
                                const DeviceArgument &arg = kernelArgs[(offset - XRT_KERNEL_ARGS_OFFSET) >> 3];
                                if ((offset % 8) == 0 || argGetHWBits(arg) > 32) {
                                    value = builder.create<circt::comb::ExtractOp>(hwModuleOutputValues[hwModuleGetOutputIndex(mod, arg.name)],
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
    auto outputOp = mod.getBodyBlock()->getTerminator();
    outputOp->setOperands(hwModuleOutputValues);
}

void CodeGen_CIRCT_Xilinx_Dev::generateToplevel(mlir::ImplicitLocOpBuilder &builder, const std::string &kernelName,
                                                const std::vector<DeviceArgument> &kernelArgs) {
    // Module inputs and outputs
    mlir::SmallVector<circt::hw::PortInfo> ports;

    // Clock and reset signals
    ports.push_back(circt::hw::PortInfo{builder.getStringAttr("ap_clk"), circt::hw::PortDirection::INPUT, builder.getI1Type()});
    ports.push_back(circt::hw::PortInfo{builder.getStringAttr("ap_rst_n"), circt::hw::PortDirection::INPUT, builder.getI1Type()});

    // AXI4 lite subordinate control signals
    mlir::SmallVector<circt::hw::PortInfo> axi4LiteSubordinateSignals;
    portsAddAXI4LiteSubordinateSignals(builder, S_AXI_ADDR_WIDTH, S_AXI_DATA_WIDTH, axi4LiteSubordinateSignals);
    ports.append(axi4LiteSubordinateSignals);

    // AXI4 manager control signals
    mlir::SmallVector<mlir::SmallVector<circt::hw::PortInfo>> axi4ManagerSignals;

    // Calyx external memory interface signals
    mlir::SmallVector<circt::hw::PortInfo> calyxExtMemSignals;
    portsAddCalyxExtMemInterfaceSubordinateSignals(builder, M_AXI_ADDR_WIDTH, M_AXI_DATA_WIDTH, calyxExtMemSignals);

    // Signals for CalyxExtMemToAxi for each kernel buffer argument
    unsigned numBufferArgs = 0;
    for (unsigned i = 0; i < kernelArgs.size(); i++) {
        if (kernelArgs[i].is_buffer) {
            mlir::SmallVector<circt::hw::PortInfo> signals;
            portsAddAXI4ManagerSignalsPrefix(builder, getAxiManagerSignalNamePrefixId(numBufferArgs) + "_",
                                             M_AXI_ADDR_WIDTH, M_AXI_DATA_WIDTH, signals);
            axi4ManagerSignals.push_back(signals);
            ports.append(signals);
            // Connected directly to the AXI interface
            ports.push_back(circt::hw::PortInfo{builder.getStringAttr(toFullAxiManagerSignalNameId(numBufferArgs, "arsize")),
                                                circt::hw::PortDirection::OUTPUT, builder.getIntegerType(3)});
            ports.push_back(circt::hw::PortInfo{builder.getStringAttr(toFullAxiManagerSignalNameId(numBufferArgs, "awsize")),
                                                circt::hw::PortDirection::OUTPUT, builder.getIntegerType(3)});
            numBufferArgs++;
        }
    }

    // Create toplevel HW module
    circt::hw::HWModuleOp mod = builder.create<circt::hw::HWModuleOp>(builder.getStringAttr("toplevel"), ports);
    builder.setInsertionPointToStart(mod.getBodyBlock());

    // This will hold the list of all the output Values
    mlir::SmallVector<mlir::Value> hwModuleOutputWires(mod.getNumResults());
    mlir::SmallVector<mlir::Value> hwModuleOutputValues(mod.getNumResults());
    for (unsigned i = 0; i < mod.getNumResults(); i++) {
        auto name = mod.getResultNames()[i].cast<mlir::StringAttr>().str();
        auto type = mod.getResultTypes()[i];
        hwModuleOutputWires[i] = builder.create<circt::sv::WireOp>(type, name);
        hwModuleOutputValues[i] = builder.create<circt::sv::ReadInOutOp>(hwModuleOutputWires[i]);
    }

    mlir::Value clock = hwModuleGetInputValue(mod, "ap_clk");
    mlir::Value reset = circt::comb::createOrFoldNot(hwModuleGetInputValue(mod, "ap_rst_n"), builder);

    mlir::Value apDone = builder.create<circt::sv::WireOp>(builder.getI1Type(), "ap_done");
    mlir::Value apDoneRead = builder.create<circt::sv::ReadInOutOp>(apDone);
    // Kernel done signal also enables reset to the kernel and the CalyxExtMemToAxi FSMs
    mlir::Value resetOrApDone = builder.create<circt::comb::OrOp>(reset, apDoneRead);

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
            controlAxiInputs.push_back(hwModuleGetInputValue(mod, signal.name.str()));
            controlAxiArgNames.push_back(signal.name);
        } else {
            controlAxiResultTypes.push_back(signal.type);
            controlAxiResultNames.push_back(signal.name);
        }
    }

    // Instance a CalyxExtMemToAxi for each kernel buffer argument
    mlir::SmallVector<circt::hw::InstanceOp> CalyxExtMemToAxiInstances(numBufferArgs);
    mlir::SmallVector<mlir::SmallVector<circt::sv::WireOp>> CalyxExtMemToAxiInstanceWires(numBufferArgs);

    for (unsigned i = 0; i < numBufferArgs; i++) {
        mlir::SmallVector<mlir::Value> CalyxExtMemToAxiInputs;
        mlir::SmallVector<mlir::Type> CalyxExtMemToAxiResultTypes;
        mlir::SmallVector<mlir::Attribute> CalyxExtMemToAxiArgNames, CalyxExtMemToAxiResultNames;

        for (const auto &signal : axi4ManagerSignals[i]) {
            std::string basename = fullAxiSignalNameIdGetBasename(signal.name.str());
            mlir::StringAttr nameAttr = builder.getStringAttr(toFullAxiManagerSignalName(basename));
            if (signal.direction == circt::hw::PortDirection::INPUT) {
                CalyxExtMemToAxiInputs.push_back(hwModuleGetInputValue(mod, signal.name.str()));
                CalyxExtMemToAxiArgNames.push_back(nameAttr);
            } else {
                CalyxExtMemToAxiResultTypes.push_back(signal.type);
                CalyxExtMemToAxiResultNames.push_back(nameAttr);
            }
        }

        for (const auto &signal : calyxExtMemSignals) {
            if (signal.direction == circt::hw::PortDirection::INPUT) {
                circt::sv::WireOp wire = builder.create<circt::sv::WireOp>(signal.type, signal.name.str() + "_" + std::to_string(i));
                mlir::Value wireRead = builder.create<circt::sv::ReadInOutOp>(wire);
                CalyxExtMemToAxiInputs.push_back(wireRead);
                CalyxExtMemToAxiArgNames.push_back(signal.name);
                CalyxExtMemToAxiInstanceWires[i].push_back(wire);
            } else {
                CalyxExtMemToAxiResultTypes.push_back(signal.type);
                CalyxExtMemToAxiResultNames.push_back(signal.name);
            }
        }

        CalyxExtMemToAxiInputs.push_back(clock);
        CalyxExtMemToAxiArgNames.push_back(builder.getStringAttr("clk"));
        CalyxExtMemToAxiInputs.push_back(resetOrApDone);
        CalyxExtMemToAxiArgNames.push_back(builder.getStringAttr("rst"));

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
                int hwModIdx = hwModuleGetOutputIndex(mod, signal.name.str());
                int memIdx = hwModuleGetOutputIndex(CalyxExtMemToAxiInstances[i], toFullAxiManagerSignalName(basename));
                builder.create<circt::sv::AssignOp>(hwModuleOutputWires[hwModIdx], CalyxExtMemToAxiInstances[i].getResult(memIdx));
            }
        }
    }

    // Kernel arguments
    mlir::SmallVector<circt::sv::WireOp> kernelArgsWires(kernelArgs.size());
    mlir::SmallVector<mlir::Value> kernelArgsWiresRead(kernelArgs.size());
    for (unsigned i = 0; i < kernelArgs.size(); i++) {
        mlir::Type type = builder.getIntegerType(argGetHWBits(kernelArgs[i]));
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
            int i = hwModuleGetOutputIndex(mod, signal.name.str());
            int j = hwModuleGetOutputIndex(controlAxiInstance, signal.name.str());
            builder.create<circt::sv::AssignOp>(hwModuleOutputWires[i], controlAxiInstance.getResult(j));
        }
    }

    // Kernel instance
    mlir::SmallVector<mlir::Value> kernelInputs;
    mlir::SmallVector<mlir::Type> kernelResultTypes;
    mlir::SmallVector<mlir::Attribute> kernelArgNames, kernelResultNames;

    for (const auto &arg : kernelArgs) {
        int idx = hwModuleGetOutputIndex(controlAxiInstance, arg.name);
        kernelInputs.push_back(controlAxiInstance.getResult(idx));
        kernelArgNames.push_back(builder.getStringAttr(arg.name));
    }

    for (unsigned i = 0; i < numBufferArgs; i++) {
        for (const auto &signal : calyxExtMemSignals) {
            std::string signalNameSuffix = signal.name.str().substr(std::string("calyx_").length());
            if (signal.direction == circt::hw::PortDirection::INPUT) {
                kernelResultTypes.push_back(signal.type);
                kernelResultNames.push_back(builder.getStringAttr("ext_mem" + std::to_string(i) + "_" + signalNameSuffix));
            } else {
                kernelInputs.push_back(CalyxExtMemToAxiInstances[i].getResult(
                    hwModuleGetOutputIndex(CalyxExtMemToAxiInstances[i], signal.name.str())));
                kernelArgNames.push_back(builder.getStringAttr("ext_mem" + std::to_string(i) + "_" + signalNameSuffix));
            }
        }
    }

    kernelInputs.push_back(clock);
    kernelArgNames.push_back(builder.getStringAttr("clk"));
    kernelInputs.push_back(resetOrApDone);
    kernelArgNames.push_back(builder.getStringAttr("reset"));
    kernelInputs.push_back(controlAxiInstance.getResult(hwModuleGetOutputIndex(controlAxiInstance, "ap_start")));
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

        idxIn = hwModuleGetOutputIndex(kernelInstance, "ext_mem" + std::to_string(i) + "_access_size");
        idxOut = hwModuleGetOutputIndex(mod, toFullAxiManagerSignalNameId(i, "arsize"));
        builder.create<circt::sv::AssignOp>(hwModuleOutputWires[idxOut], kernelInstance.getResult(idxIn));
        idxOut = hwModuleGetOutputIndex(mod, toFullAxiManagerSignalNameId(i, "awsize"));
        builder.create<circt::sv::AssignOp>(hwModuleOutputWires[idxOut], kernelInstance.getResult(idxIn));

        int j = 0;
        for (const auto &signal : calyxExtMemSignals) {
            if (signal.direction == circt::hw::PortDirection::INPUT) {
                std::string signalNameSuffix = signal.name.str().substr(std::string("calyx_").length());
                idxIn = hwModuleGetOutputIndex(kernelInstance, "ext_mem" + std::to_string(i) + "_" + signalNameSuffix);
                builder.create<circt::sv::AssignOp>(CalyxExtMemToAxiInstanceWires[i][j++], kernelInstance.getResult(idxIn));
            }
        }
    }

    builder.create<circt::sv::AssignOp>(apDone,
                                        kernelInstance.getResult(hwModuleGetOutputIndex(kernelInstance, "done")));

    // Set module output operands
    auto outputOp = mod.getBodyBlock()->getTerminator();
    outputOp->setOperands(hwModuleOutputValues);
}

void CodeGen_CIRCT_Xilinx_Dev::generateKernelXml(llvm::raw_ostream &os, const std::string &kernelName, const std::vector<DeviceArgument> &kernelArgs) {
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
        if (arg.is_buffer) {
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

void CodeGen_CIRCT_Xilinx_Dev::portsAddAXI4ManagerSignalsPrefix(mlir::ImplicitLocOpBuilder &builder, const std::string &prefix,
                                                                int addrWidth, int dataWidth,
                                                                mlir::SmallVector<circt::hw::PortInfo> &ports) {
    const mlir::SmallVector<SignalInfo> signals = {
        // Read address channel
        {"araddr", addrWidth, false},
        {"arvalid", 1, false},
        {"arready", 1, true},
        {"arlen", 8, false},
        // Read data channel
        {"rdata", dataWidth, true},
        {"rvalid", 1, true},
        {"rready", 1, false},
        {"rlast", 1, true},
        // Write address channel
        {"awaddr", addrWidth, false},
        {"awvalid", 1, false},
        {"awready", 1, true},
        {"awlen", 8, false},
        // Write data channel
        {"wdata", dataWidth, false},
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

void CodeGen_CIRCT_Xilinx_Dev::portsAddAXI4LiteSubordinateSignals(mlir::ImplicitLocOpBuilder &builder,
                                                                  int addrWidth, int dataWidth,
                                                                  mlir::SmallVector<circt::hw::PortInfo> &ports) {
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

void CodeGen_CIRCT_Xilinx_Dev::portsAddCalyxExtMemInterfaceSubordinateSignals(mlir::ImplicitLocOpBuilder &builder,
                                                                              int addrWidth, int dataWidth,
                                                                              mlir::SmallVector<circt::hw::PortInfo> &ports) {
    // Must follow same order as SCFToCalyx's appendPortsForExternalMemref
    const mlir::SmallVector<SignalInfo> signals = {
        {"calyx_write_data", dataWidth, true},
        {"calyx_addr0", addrWidth, true},
        {"calyx_write_en", 1, true},
        {"calyx_read_en", 1, true},
        {"calyx_access_size", 3, true},
        {"calyx_read_data", dataWidth, false},
        {"calyx_done", 1, false},
    };

    for (const auto &signal : signals) {
        ports.push_back(circt::hw::PortInfo{builder.getStringAttr(signal.name),
                                            signal.input ? circt::hw::PortDirection::INPUT : circt::hw::PortDirection::OUTPUT,
                                            builder.getIntegerType(signal.size)});
    }
}

}  // namespace

std::unique_ptr<CodeGen_Accelerator_Dev> new_CodeGen_CIRCT_Xilinx_Dev(const Target &target) {
    return std::make_unique<CodeGen_CIRCT_Xilinx_Dev>(target);
}

}  // namespace Internal
}  // namespace Halide
