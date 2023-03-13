#include <fstream>
#include <vector>

#include <circt/Conversion/ExportVerilog.h>
#include <circt/Conversion/FSMToSV.h>
#include <circt/Conversion/HWArithToHW.h>
#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/FSM/FSMDialect.h>
#include <circt/Dialect/FSM/FSMOps.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/HW/HWOps.h>
#include <circt/Dialect/HWArith/HWArithDialect.h>
#include <circt/Dialect/HWArith/HWArithOps.h>
#include <circt/Dialect/Seq/SeqDialect.h>
#include <circt/Dialect/Seq/SeqOps.h>
#include <circt/Dialect/Seq/SeqPasses.h>
#include <circt/Dialect/SV/SVDialect.h>
#include <circt/Dialect/SV/SVOps.h>
#include <circt/Dialect/SV/SVPasses.h>
#include <circt/Support/LoweringOptions.h>

#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>

#include "CodeGen_CIRCT.h"
#include "CodeGen_Internal.h"
#include "Debug.h"
#include "IROperator.h"
#include "Util.h"

namespace Halide {

namespace Internal {


CodeGen_CIRCT::CodeGen_CIRCT() {
    mlir_context.loadDialect<circt::comb::CombDialect>();
    mlir_context.loadDialect<circt::fsm::FSMDialect>();
    mlir_context.loadDialect<circt::hw::HWDialect>();
    mlir_context.loadDialect<circt::hwarith::HWArithDialect>();
    mlir_context.loadDialect<circt::seq::SeqDialect>();
    mlir_context.loadDialect<circt::sv::SVDialect>();
}

void CodeGen_CIRCT::compile(const Module &input) {
    debug(1) << "Generating CIRCT MLIR IR...\n";

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
    CirctGlobalTypes globalTypes;

    // Create top level ("builtin.module") common definitions for all the HW modules to use
    create_circt_definitions(globalTypes, builder);

    // Translate each function into a CIRCT HWModuleOp
    for (const auto &function : input.functions()) {
        std::cout << "Generating CIRCT MLIR IR for function " << function.name << std::endl;
        CodeGen_CIRCT::Visitor visitor(builder, globalTypes, function);
        function.body.accept(&visitor);
    }

    // Print MLIR before running passes
    std::cout << "Original MLIR" << std::endl;
    mlir_module.dump();

    // Verify module (before running passes)
    auto moduleVerifyResult = mlir::verify(mlir_module);
    std::cout << "Module verify (before passess) result: " << moduleVerifyResult.succeeded() << std::endl;
    internal_assert(moduleVerifyResult.succeeded());

    // Create and run passes
    std::cout << "Running passes." << std::endl;
    mlir::PassManager pm(mlir_module.getContext());
    pm.addPass(circt::createHWArithToHWPass());
    pm.addPass(circt::createConvertFSMToSVPass());
    pm.addPass(circt::seq::createSeqLowerToSVPass());
    pm.nest<circt::hw::HWModuleOp>().addPass(circt::sv::createHWCleanupPass());
    pm.nest<circt::hw::HWModuleOp>().addPass(circt::sv::createHWLegalizeModulesPass());
    auto &modulePM = pm.nest<circt::hw::HWModuleOp>();
    modulePM.addPass(circt::sv::createPrettifyVerilogPass());

    auto pmRunResult = pm.run(mlir_module);
    std::cout << "Run passes result: " << pmRunResult.succeeded() << std::endl;
    //std::cout << "Module inputs: " << top.getNumInputs() << ", outputs: " << top.getNumOutputs() << std::endl;

    // Print MLIR after running passes
    std::cout << "MLIR after running passes" << std::endl;
    mlir_module.dump();

    // Verify module (after running passes)
    moduleVerifyResult = mlir::verify(mlir_module);
    std::cout << "Module verify (after passes) result: " << moduleVerifyResult.succeeded() << std::endl;
    internal_assert(moduleVerifyResult.succeeded());

    // Generate Verilog
    std::cout << "Exporting Verilog." << std::endl;
    auto exportVerilogResult = circt::exportSplitVerilog(mlir_module, "gen/");
    std::cout << "Export Verilog result: " << exportVerilogResult.succeeded() << std::endl;

    std::cout << "Done!" << std::endl;
}

void CodeGen_CIRCT::create_circt_definitions(CirctGlobalTypes &globalTypes, mlir::ImplicitLocOpBuilder &builder) {
    // Create MemAccessFSM machine definition
    {
        // Inputs: {enable, mem_access_started, mem_access_finished}
        mlir::SmallVector<mlir::Type> inputs{builder.getI1Type(), builder.getI1Type(), builder.getI1Type()};
        // Outputs: {valid, done}
        mlir::SmallVector<mlir::Type> results{builder.getI1Type(), builder.getI1Type()};
        mlir::FunctionType function_type = builder.getFunctionType(inputs, results);
        globalTypes.memAccessFSM = builder.create<circt::fsm::MachineOp>("MemAccessFSM", "IDLE", function_type);
        mlir::ArrayAttr argNames = builder.getArrayAttr({builder.getStringAttr("enable"),
                                                         builder.getStringAttr("mem_access_started"),
                                                         builder.getStringAttr("mem_access_finished")});
        globalTypes.memAccessFSM.setArgNamesAttr(argNames);
        mlir::ArrayAttr resNames = builder.getArrayAttr({builder.getStringAttr("valid"),
                                                         builder.getStringAttr("done")});
        globalTypes.memAccessFSM.setResNamesAttr(resNames);
        mlir::Region &fsm_body = globalTypes.memAccessFSM.getBody();
        mlir::ImplicitLocOpBuilder fsm_builder = mlir::ImplicitLocOpBuilder::atBlockEnd(fsm_body.getLoc(), &fsm_body.front());
        mlir::Value val0 = fsm_builder.create<circt::hw::ConstantOp>(fsm_builder.getBoolAttr(false));
        mlir::Value val1 = fsm_builder.create<circt::hw::ConstantOp>(fsm_builder.getBoolAttr(true));

        circt::fsm::StateOp idle_state = fsm_builder.create<circt::fsm::StateOp>("IDLE");
        {
            {
                idle_state.getOutputOp()->setOperands(mlir::ValueRange{val0, val0});
            }
            mlir::Region &transitions = idle_state.getTransitions();
            mlir::ImplicitLocOpBuilder transitions_builder = mlir::ImplicitLocOpBuilder::atBlockBegin(transitions.getLoc(), &transitions.front());
            {
                circt::fsm::TransitionOp transition = transitions_builder.create<circt::fsm::TransitionOp>("READY");
                transition.ensureGuard(transitions_builder);
                circt::fsm::ReturnOp returnOp = transition.getGuardReturn();
                returnOp.setOperand(globalTypes.memAccessFSM.getArgument(0));
            }
        }
        circt::fsm::StateOp ready_state = fsm_builder.create<circt::fsm::StateOp>("READY");
        {
            {
                ready_state.getOutputOp()->setOperands(mlir::ValueRange{val1, val0});
            }
            mlir::Region &transitions = ready_state.getTransitions();
            mlir::ImplicitLocOpBuilder transitions_builder = mlir::ImplicitLocOpBuilder::atBlockBegin(transitions.getLoc(), &transitions.front());
            {
                circt::fsm::TransitionOp transition = transitions_builder.create<circt::fsm::TransitionOp>("BUSY");
                transition.ensureGuard(transitions_builder);
                circt::fsm::ReturnOp returnOp = transition.getGuardReturn();
                returnOp.setOperand(globalTypes.memAccessFSM.getArgument(1));
            }
        }
        circt::fsm::StateOp busy_state = fsm_builder.create<circt::fsm::StateOp>("BUSY");
        {
            {
                busy_state.getOutputOp()->setOperands(mlir::ValueRange{val0, val0});
            }
            mlir::Region &transitions = busy_state.getTransitions();
            mlir::ImplicitLocOpBuilder transitions_builder = mlir::ImplicitLocOpBuilder::atBlockBegin(transitions.getLoc(), &transitions.front());
            {
                circt::fsm::TransitionOp transition = transitions_builder.create<circt::fsm::TransitionOp>("DONE");
                transition.ensureGuard(transitions_builder);
                circt::fsm::ReturnOp returnOp = transition.getGuardReturn();
                returnOp.setOperand(globalTypes.memAccessFSM.getArgument(2));
            }
        }
        circt::fsm::StateOp done_state = fsm_builder.create<circt::fsm::StateOp>("DONE");
        {
            {
                done_state.getOutputOp()->setOperands(mlir::ValueRange{val0, val1});
            }
            mlir::Region &transitions = done_state.getTransitions();
            mlir::ImplicitLocOpBuilder transitions_builder = mlir::ImplicitLocOpBuilder::atBlockBegin(transitions.getLoc(), &transitions.front());
            {
                transitions_builder.create<circt::fsm::TransitionOp>("IDLE");
            }
        }
    }
}

CodeGen_CIRCT::Visitor::Visitor(mlir::ImplicitLocOpBuilder &builder, CirctGlobalTypes &globalTypes, const Internal::LoweredFunc &function)
    : builder(builder), globalTypes(globalTypes) {
    // Generate module ports (inputs and outputs)
    mlir::SmallVector<circt::hw::PortInfo> ports;

    // Clock and reset signals
    ports.push_back(circt::hw::PortInfo{builder.getStringAttr("clk"), circt::hw::PortDirection::INPUT, builder.getI1Type(), 0});
    ports.push_back(circt::hw::PortInfo{builder.getStringAttr("reset"), circt::hw::PortDirection::INPUT, builder.getI1Type(), 0});

    // Convert function arguments to module ports
    //std::vector<LoweredArgument> scalar_arguments;
    std::vector<LoweredArgument> buffer_arguments;
    debug(1) << "\tArg count: " << function.args.size() << "\n";

    for (const auto &arg: function.args) {
        static const char *const kind_names[] = {
            "halide_argument_kind_input_scalar",
            "halide_argument_kind_input_buffer",
            "halide_argument_kind_output_buffer",
        };

        static const char *const type_code_names[] = {
            "halide_type_int",
            "halide_type_uint",
            "halide_type_float",
            "halide_type_handle",
            "halide_type_bfloat",
        };

        debug(1) << "\t\tArg: " << arg.name << "\n";
        debug(1) << "\t\t\tKind: " << kind_names[arg.kind] << "\n";
        debug(1) << "\t\t\tDimensions: " << int(arg.dimensions) << "\n";
        debug(1) << "\t\t\tType: " << type_code_names[arg.type.code()] << "\n";
        debug(1) << "\t\t\tType bits: " << arg.type.bits() << "\n";
        debug(1) << "\t\t\tType lanes: " << arg.type.lanes() << "\n";

        mlir::Type type;

        switch (arg.kind) {
        case Argument::Kind::InputScalar:
            switch (arg.type.code()) {
            case Type::Int:
            case Type::UInt:
                type = builder.getIntegerType(arg.type.bits(), arg.type.is_int());
                ports.push_back(circt::hw::PortInfo{builder.getStringAttr(arg.name), circt::hw::PortDirection::INPUT, type, 0});
                break;
            case Type::Float:
            case Type::BFloat:
                assert(0 && "TODO");
                break;
            case Type::Handle:
                // TODO: Ignore for now
                break;
            }
            //scalar_arguments.push_back(arg);
            break;
        case Argument::Kind::InputBuffer:
        case Argument::Kind::OutputBuffer:
            struct BufferDescriptorEntry {
                std::string suffix;
                mlir::Type type;
            };

            std::vector<BufferDescriptorEntry> entries;
            mlir::Type si32 = builder.getIntegerType(32, true);
            mlir::Type ui64 = builder.getIntegerType(64, true);

            // A pointer to the start of the data in main memory (offset of the buffer into the AXI4 master interface)
            entries.push_back({"host", ui64});
            // The dimensionality of the buffer
            entries.push_back({"dimensions", si32});

            // Buffer dimensions
            for (int i = 0; i < arg.dimensions; i++) {
                entries.push_back({"dim_" + std::to_string(i) + "_min", si32});
                entries.push_back({"dim_" + std::to_string(i) + "_extent", si32});
                entries.push_back({"dim_" + std::to_string(i) + "_stride", si32});
            }

            for (const auto &entry: entries) {
                ports.push_back(circt::hw::PortInfo{builder.getStringAttr(arg.name + "_" + entry.suffix),
                                circt::hw::PortDirection::INPUT, entry.type, 0});
            }

            buffer_arguments.push_back(arg);
            break;
        }
    }

    // For each buffer argument, we use a different AXI4 master interface (up to 16), named [m00_axi, ..., m16_axi]
    for (size_t i = 0; i < buffer_arguments.size(); i++) {
        const std::vector<std::tuple<std::string, mlir::Type, circt::hw::PortDirection>> axi_signals = {
            {"awvalid", builder.getI1Type(), circt::hw::PortDirection::OUTPUT},
            {"awready", builder.getI1Type(), circt::hw::PortDirection::INPUT},
            {"awaddr",  builder.getI64Type(), circt::hw::PortDirection::OUTPUT},
            {"awlen",   builder.getI8Type(), circt::hw::PortDirection::OUTPUT},
            {"wvalid",  builder.getI1Type(), circt::hw::PortDirection::OUTPUT},
            {"wready",  builder.getI1Type(), circt::hw::PortDirection::INPUT},
            {"wdata",   builder.getI1Type(), circt::hw::PortDirection::OUTPUT},
            {"wstrb",   builder.getI1Type(), circt::hw::PortDirection::OUTPUT},
            {"wlast",   builder.getI1Type(), circt::hw::PortDirection::OUTPUT},
            {"bvalid",  builder.getI1Type(), circt::hw::PortDirection::INPUT},
            {"bready",  builder.getI1Type(), circt::hw::PortDirection::OUTPUT},
            {"arvalid", builder.getI1Type(), circt::hw::PortDirection::OUTPUT},
            {"arready", builder.getI1Type(), circt::hw::PortDirection::INPUT},
            {"araddr",  builder.getI64Type(), circt::hw::PortDirection::OUTPUT},
            {"arlen",   builder.getI1Type(), circt::hw::PortDirection::OUTPUT},
            {"rvalid",  builder.getI1Type(), circt::hw::PortDirection::INPUT},
            {"rready",  builder.getI1Type(), circt::hw::PortDirection::OUTPUT},
            {"rdata",   builder.getI1Type(), circt::hw::PortDirection::INPUT},
            {"rlast",   builder.getI1Type(), circt::hw::PortDirection::INPUT},
        };

        for (const auto &signal: axi_signals) {
            ports.push_back(circt::hw::PortInfo{builder.getStringAttr("m" + std::string(i < 10 ? "0" : "") +
                                                std::to_string(i) + "_" + std::get<0>(signal)),
                                                std::get<2>(signal), std::get<1>(signal)});
        }
    }

    // Create module top
    circt::hw::HWModuleOp top = builder.create<circt::hw::HWModuleOp>(builder.getStringAttr(function.name), ports);
    builder.setInsertionPointToStart(top.getBodyBlock());

    // Add function arguments to the symbol table
    for (unsigned i = 0; i < top.getNumArguments(); i++) {
        std::string name = top.getArgNames()[i].cast<mlir::StringAttr>().str();
        sym_push(name, top.getArgument(i));
    }

    // Outputs
    mlir::SmallVector<mlir::Value> outputs;
    for (unsigned i = 0; i < top.getNumResults(); i++) {
        std::string name = top.getResultNames()[i].cast<mlir::StringAttr>().str();
        mlir::Type type = top.getResultTypes()[i];
        mlir::Value wire = builder.create<circt::sv::WireOp>(type);
        mlir::Value rdata = builder.create<circt::sv::ReadInOutOp>(wire);
        sym_push(name, rdata);
        outputs.push_back(rdata);
    }

    // Set module output operands
    auto outputOp = top.getBodyBlock()->getTerminator();
    outputOp->setOperands(outputs);

    // Create buffer type to facilitate access
    int axi_interface = 0;
    for (const auto &buf: buffer_arguments) {
        mlir::Type si32 = builder.getIntegerType(32, true);
        mlir::Type ui64 = builder.getIntegerType(64, true);
        circt::hw::StructType dim_type = builder.getType<circt::hw::StructType>(mlir::SmallVector({
            circt::hw::StructType::FieldInfo{builder.getStringAttr("min"), si32},
            circt::hw::StructType::FieldInfo{builder.getStringAttr("extent"), si32},
            circt::hw::StructType::FieldInfo{builder.getStringAttr("stride"), si32}
        }));
        circt::hw::ArrayType dim_array_type = circt::hw::ArrayType::get(dim_type, buf.dimensions);
        circt::hw::StructType buf_type  = builder.getType<circt::hw::StructType>(mlir::SmallVector({
            circt::hw::StructType::FieldInfo{builder.getStringAttr("host"), ui64},
            circt::hw::StructType::FieldInfo{builder.getStringAttr("dimensions"), si32},
            circt::hw::StructType::FieldInfo{builder.getStringAttr("dim"), dim_array_type},
        }));

        mlir::SmallVector<mlir::Value> dim_array_elements;
        for (int i = 0; i < buf.dimensions; i++) {
            mlir::SmallVector<mlir::Value> struct_elements;
            for (auto &elem: dim_type.getElements())
                struct_elements.push_back(sym_get(buf.name + "_dim_" + std::to_string(i) + "_" + elem.name.str()));

            dim_array_elements.push_back(builder.create<circt::hw::StructCreateOp>(dim_type, struct_elements));
        }
        mlir::Value host = sym_get(buf.name + "_host");
        mlir::Value dimensions = sym_get(buf.name + "_dimensions");
        mlir::Value dim = builder.create<circt::hw::ArrayCreateOp>(dim_array_type, dim_array_elements);

        llvm::SmallVector<mlir::NamedAttribute> attributes = {
            mlir::NamedAttribute(builder.getStringAttr("axi_interface"), builder.getI32IntegerAttr(axi_interface)),
        };

        mlir::Value buffer = builder.create<circt::hw::StructCreateOp>(buf_type, mlir::ValueRange({host, dimensions, dim}), attributes);
        sym_push(buf.name + ".buffer", buffer);

        axi_interface++;
    }
}

mlir::Value CodeGen_CIRCT::Visitor::codegen(const Expr &e) {
    internal_assert(e.defined());
    debug(4) << "Codegen (E): " << e.type() << ", " << e << "\n";
    value = mlir::Value();
    e.accept(this);
    internal_assert(value) << "Codegen of an expr did not produce a MLIR value\n" << e;
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
    mlir::Type type = builder.getIntegerType(op->type.bits(), true);
    value = builder.create<circt::hwarith::ConstantOp>(type, builder.getIntegerAttr(type, op->value));
}

void CodeGen_CIRCT::Visitor::visit(const UIntImm *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    mlir::Type type = builder.getIntegerType(op->type.bits(), false);
    value = builder.create<circt::hwarith::ConstantOp>(type, builder.getIntegerAttr(type, op->value));
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

    debug(1) << "\tSrc type: " << src << "\n";
    debug(1) << "\tDst type: " << dst << "\n";

    value = codegen(op->value);

    if (src.is_int_or_uint() && dst.is_int_or_uint()) {
        value = builder.create<circt::hwarith::CastOp>(builder.getIntegerType(dst.bits(), dst.is_int()), value);
        // circt::hw::BitcastOp
    } else {
        assert(0);
    }
}

void CodeGen_CIRCT::Visitor::visit(const Reinterpret *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    value = codegen(op->value);
    value = builder.create<circt::hwarith::CastOp>(builder.getIntegerType(op->type.bits(), op->type.is_int()), value);
}

void CodeGen_CIRCT::Visitor::visit(const Variable *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    debug(1) << "\tname: " << op->name << "\n";
    value = sym_get(op->name, true);
}

void CodeGen_CIRCT::Visitor::visit(const Add *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<circt::hwarith::AddOp>(mlir::ValueRange({codegen(op->a), codegen(op->b)}));
    value = truncate_int(value, op->type.bits());
}

void CodeGen_CIRCT::Visitor::visit(const Sub *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<circt::hwarith::SubOp>(mlir::ValueRange({codegen(op->a), codegen(op->b)}));
    value = truncate_int(value, op->type.bits());
}

void CodeGen_CIRCT::Visitor::visit(const Mul *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<circt::hwarith::MulOp>(mlir::ValueRange({codegen(op->a), codegen(op->b)}));
    value = truncate_int(value, op->type.bits());
}

void CodeGen_CIRCT::Visitor::visit(const Div *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<circt::hwarith::DivOp>(mlir::ValueRange({codegen(op->a), codegen(op->b)}));
    value = truncate_int(value, op->type.bits());
}

void CodeGen_CIRCT::Visitor::visit(const Mod *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    value = codegen(lower_int_uint_mod(op->a, op->b));
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

    mlir::Value a = codegen(op->a);
    mlir::Value b = codegen(op->b);
    value = builder.create<circt::hwarith::ICmpOp>(circt::hwarith::ICmpPredicate::eq, a, b);
}

void CodeGen_CIRCT::Visitor::visit(const NE *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    mlir::Value a = codegen(op->a);
    mlir::Value b = codegen(op->b);
    value = builder.create<circt::hwarith::ICmpOp>(circt::hwarith::ICmpPredicate::ne, a, b);
}

void CodeGen_CIRCT::Visitor::visit(const LT *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    mlir::Value a = codegen(op->a);
    mlir::Value b = codegen(op->b);
    value = builder.create<circt::hwarith::ICmpOp>(circt::hwarith::ICmpPredicate::lt, a, b);
}

void CodeGen_CIRCT::Visitor::visit(const LE *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    mlir::Value a = codegen(op->a);
    mlir::Value b = codegen(op->b);
    value = builder.create<circt::hwarith::ICmpOp>(circt::hwarith::ICmpPredicate::le, a, b);
}

void CodeGen_CIRCT::Visitor::visit(const GT *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    mlir::Value a = codegen(op->a);
    mlir::Value b = codegen(op->b);
    value = builder.create<circt::hwarith::ICmpOp>(circt::hwarith::ICmpPredicate::gt, a, b);
}

void CodeGen_CIRCT::Visitor::visit(const GE *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    mlir::Value a = codegen(op->a);
    mlir::Value b = codegen(op->b);
    value = builder.create<circt::hwarith::ICmpOp>(circt::hwarith::ICmpPredicate::ge, a, b);
}

void CodeGen_CIRCT::Visitor::visit(const And *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    visit_and_or<And, circt::comb::AndOp>(op);
}

void CodeGen_CIRCT::Visitor::visit(const Or *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    visit_and_or<Or, circt::comb::OrOp>(op);
}

void CodeGen_CIRCT::Visitor::visit(const Not *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    mlir::Value a = codegen(op->a);
    circt::hwarith::ConstantOp allzeroes = builder.create<circt::hwarith::ConstantOp>(a.getType(), builder.getIntegerAttr(a.getType(), 0));
    value = builder.create<circt::hwarith::ICmpOp>(circt::hwarith::ICmpPredicate::eq, a, allzeroes);
}

void CodeGen_CIRCT::Visitor::visit(const Select *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    mlir::Value cond = codegen(op->condition);
    mlir::Value a = codegen(op->true_value);
    mlir::Value b = codegen(op->false_value);
    value = builder.create<circt::comb::MuxOp>(to_signless(cond), a, b);
}

void CodeGen_CIRCT::Visitor::visit(const Load *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    debug(1) << "\tname: " << op->name << "\n";

    //mlir::Value predicate = codegen(op->predicate);
    //mlir::Value index = codegen(op->index);

    mlir::Type type = builder.getIntegerType(op->type.bits(), op->type.is_int());
    value = builder.create<circt::hwarith::ConstantOp>(type, builder.getIntegerAttr(type, 0x1234));
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
    for (const Expr &e: op->args)
        debug(1) << "\tArg: " << e << "\n";

    mlir::Type op_type = builder.getIntegerType(op->type.bits(), op->type.is_int());

    auto buffer_extract_dim_field = [this](const mlir::Value &buf, const mlir::Value &d, const std::string &field) {
        circt::hw::ArrayType buf_dim_type = buf.getType().cast<circt::hw::StructType>().getFieldType("dim").cast<circt::hw::ArrayType>();
        mlir::Type idx_type = builder.getIntegerType(llvm::Log2_64_Ceil(buf_dim_type.getSize()));
        mlir::Value d_value = builder.create<circt::hwarith::CastOp>(idx_type, d);
        mlir::Value dim_value = builder.create<circt::hw::StructExtractOp>(buf, builder.getStringAttr("dim"));
        mlir::Value dim_d_value = builder.create<circt::hw::ArrayGetOp>(dim_value, to_signless(d_value));
        return builder.create<circt::hw::StructExtractOp>(dim_d_value, builder.getStringAttr(field));
    };

    if (op->name == Call::buffer_get_host) {
        mlir::Value buf = codegen(op->args[0]);
        mlir::Attribute axi_interface = buf.getDefiningOp()->getAttr("axi_interface");
        value = builder.create<circt::hw::StructExtractOp>(buf, builder.getStringAttr("host"));
        value.getDefiningOp()->setAttr(builder.getStringAttr("axi_interface"), axi_interface);
    } else if (op->name == Call::buffer_is_bounds_query) {
        value = builder.create<circt::hwarith::ConstantOp>(op_type, builder.getIntegerAttr(op_type, 1));
    } else if(op->name == Call::buffer_get_min) {
        value = buffer_extract_dim_field(codegen(op->args[0]), codegen(op->args[1]), "min");
    } else if(op->name == Call::buffer_get_extent) {
        value = buffer_extract_dim_field(codegen(op->args[0]), codegen(op->args[1]), "extent");
    } else if(op->name == Call::buffer_get_stride) {
        value = buffer_extract_dim_field(codegen(op->args[0]), codegen(op->args[1]), "stride");
    } else {
        // Just return 1 for now
        value = builder.create<circt::hwarith::ConstantOp>(op_type, builder.getIntegerAttr(op_type, 1));
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
    debug(1) << "Contents:" << "\n";
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
        "Serial", "Parallel", "Vectorized", "Unrolled",
        "Extern", "GPUBlock", "GPUThread", "GPULane",
    };
    debug(1) << "\tForType: " << for_types[unsigned(op->for_type)] << "\n";

    mlir::Value clk = sym_get("clk");
    mlir::Value reset = sym_get("reset");
    mlir::Value clk_en = builder.create<circt::sv::LogicOp>(builder.getI1Type(), op->name + "_next_en");
    mlir::Value clk_en_read = builder.create<circt::sv::ReadInOutOp>(clk_en);

    mlir::Value min = codegen(op->min);
    mlir::Value extent = codegen(op->extent);
    mlir::Value max = builder.create<circt::hwarith::AddOp>(mlir::ValueRange{min, extent});
    mlir::Type type = builder.getIntegerType(max.getType().getIntOrFloatBitWidth());
    mlir::Value min_signless = builder.create<circt::hwarith::CastOp>(type, min);

    mlir::Value loop_var_next = builder.create<circt::sv::LogicOp>(type, op->name + "_next_val");
    mlir::Value loop_var_next_read = builder.create<circt::sv::ReadInOutOp>(loop_var_next);
    mlir::Value loop_var = builder.create<circt::seq::CompRegClockEnabledOp>(loop_var_next_read, clk, clk_en_read, reset, min_signless, op->name);
    mlir::Value loop_var_with_sign = builder.create<circt::hwarith::CastOp>(max.getType(), loop_var);

    // Increment loop variable by 1
    mlir::Value const_1 = builder.create<circt::hw::ConstantOp>(type, builder.getIntegerAttr(type, 1));
    mlir::Value loop_var_add_1 = builder.create<circt::comb::AddOp>(loop_var, const_1);
    builder.create<circt::sv::AssignOp>(loop_var_next, loop_var_add_1);

    // Inner loops will update this signal
    loop_done = builder.create<circt::hw::ConstantOp>(builder.getBoolAttr(true));

    sym_push(op->name, loop_var_with_sign);
    codegen(op->body);
    sym_pop(op->name);

    // An iteration can advance when: 1) all the inner loops have finished and 2) all memory accesses in the loop body have finished
    builder.create<circt::sv::AssignOp>(clk_en, loop_done);

    // Update signal for outer loops to know this inner loop has finished
    loop_done = builder.create<circt::comb::ICmpOp>(circt::comb::ICmpPredicate::eq, loop_var, to_signless(max));
    // TODO: loop_done = builder.create<circt::comb::AndOp>(loop_done, mem_accesses_finished);
}

void CodeGen_CIRCT::Visitor::visit(const Store *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    debug(1) << "\tName: " << op->name << "\n";

    //mlir::Value predicate = codegen(op->predicate);
    mlir::Value value = codegen(op->value);
    mlir::Value index = codegen(op->index);
    mlir::Value buf = sym_get(op->name);
    int axi_interface = buf.getDefiningOp()->getAttrOfType<mlir::IntegerAttr>("axi_interface").getInt();
    debug(1) << "\taxi_interface: " << axi_interface << "\n";

    // TODO: Use shift instead of multiplication
    uint32_t element_size_bytes = (value.getType().getIntOrFloatBitWidth() + 7) / 8;
    mlir::Value element_size = builder.create<circt::hwarith::ConstantOp>(builder.getIntegerType(32, false),
                                                                          builder.getUI32IntegerAttr(element_size_bytes));
    mlir::Value offset = builder.create<circt::hwarith::MulOp>(mlir::ValueRange{index, element_size});
    mlir::Value addr = builder.create<circt::hwarith::AddOp>(mlir::ValueRange{buf, truncate_int(offset, 64)});

    mlir::TypeRange outputs = globalTypes.memAccessFSM.getFunctionType().getResults();
    mlir::Value foo = builder.create<circt::hw::ConstantOp>(builder.getBoolAttr(false));
    circt::fsm::HWInstanceOp fsmInstance = builder.create<circt::fsm::HWInstanceOp>(outputs, op->name,
                                                                                    globalTypes.memAccessFSM.getSymName(),
                                                                                    mlir::ValueRange({foo, foo, foo}),
                                                                                    sym_get("clk"), sym_get("reset"));

    // if (predicate) buffer[op->name].store(index, value);
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

    for (auto &ext: op->extents) {
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
        //codegen_asserts(asserts);
        codegen(s);
    } else {
        codegen(op->first);
        codegen(op->rest);
    }
}

void CodeGen_CIRCT::Visitor::visit(const IfThenElse *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    debug(1) << "Contents:" << "\n";
    debug(1) << "\tcondition: " << op->condition << "\n";
    debug(1) << "\tthen_case: " << op->then_case << "\n";
    debug(1) << "\telse_case: " << op->else_case << "\n";

    codegen(op->condition);
    codegen(op->then_case);

    if (op->else_case.defined()) {
        codegen(op->else_case);
        //halide_buffer_t
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

template<typename T, typename CirctOp>
void CodeGen_CIRCT::Visitor::visit_and_or(const T *op) {
    mlir::Value a = codegen(op->a);
    mlir::Value b = codegen(op->b);
    circt::hwarith::ConstantOp allzeroes_a = builder.create<circt::hwarith::ConstantOp>(a.getType(), builder.getIntegerAttr(a.getType(), 0));
    circt::hwarith::ConstantOp allzeroes_b = builder.create<circt::hwarith::ConstantOp>(b.getType(), builder.getIntegerAttr(b.getType(), 0));
    mlir::Value isnotzero_a = builder.create<circt::hwarith::ICmpOp>(circt::hwarith::ICmpPredicate::ne, a, allzeroes_a);
    mlir::Value isnotzero_b = builder.create<circt::hwarith::ICmpOp>(circt::hwarith::ICmpPredicate::ne, b, allzeroes_b);
    value = to_unsigned(builder.create<CirctOp>(to_signless(isnotzero_a), to_signless(isnotzero_b)));
}

void CodeGen_CIRCT::Visitor::sym_push(const std::string &name, mlir::Value value) {
    //value.getDefiningOp()->setAttr("sv.namehint", builder.getStringAttr(name));
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

mlir::Value CodeGen_CIRCT::Visitor::truncate_int(const mlir::Value &value, int size) {
    if (value.getType().getIntOrFloatBitWidth() != unsigned(size)) {
        mlir::Type new_type = builder.getIntegerType(size, value.getType().isSignedInteger());
        return builder.create<circt::hwarith::CastOp>(new_type, value);
    }
    return value;
}

mlir::Value CodeGen_CIRCT::Visitor::to_sign(const mlir::Value &value, bool isSigned) {
    if (value.getType().isSignlessInteger()) {
        mlir::Type new_type = builder.getIntegerType(value.getType().getIntOrFloatBitWidth(), isSigned);
        return builder.create<circt::hwarith::CastOp>(new_type, value);
    }
    return value;
}

mlir::Value CodeGen_CIRCT::Visitor::to_signed(const mlir::Value &value) {
    return to_sign(value, true);
}

mlir::Value CodeGen_CIRCT::Visitor::to_unsigned(const mlir::Value &value) {
    return to_sign(value, false);
}

mlir::Value CodeGen_CIRCT::Visitor::to_signless(const mlir::Value &value) {
    if (!value.getType().isSignlessInteger()) {
        mlir::Type new_type = builder.getIntegerType(value.getType().getIntOrFloatBitWidth());
        return builder.create<circt::hwarith::CastOp>(new_type, value);
    }
    return value;
}

}  // namespace Internal
}  // namespace Halide
