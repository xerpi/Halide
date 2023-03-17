#include <fstream>
#include <vector>

#include <tinyxml2.h>

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
#include <circt/Dialect/Seq/SeqDialect.h>
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
    mlir::Type i32 = builder.getI32Type();
    mlir::Type i64 = builder.getI64Type();

    // Translate each function into a Calyx component
    for (const auto &function : module.functions()) {
        std::cout << "Generating CIRCT MLIR IR for function " << function.name << std::endl;

        mlir::SmallVector<mlir::Type> inputs;
        std::vector<std::string> inputNames;
        mlir::SmallVector<mlir::Type> results;
        mlir::SmallVector<mlir::NamedAttribute> attrs;
        mlir::SmallVector<mlir::DictionaryAttr> argAttrs;

        for (const auto &arg: function.args) {
            static const char *const kind_names[] = {
                "InputScalar", "InputBuffer", "OutputBuffer",
            };
            static const char *const type_code_names[] = {
                "int", "uint", "float", "handle", "bfloat",
            };

            debug(1) << "\t\tArg: " << arg.name << "\n";
            debug(1) << "\t\t\tKind: " << kind_names[arg.kind] << "\n";
            debug(1) << "\t\t\tDimensions: " << int(arg.dimensions) << "\n";
            debug(1) << "\t\t\tType: " << type_code_names[arg.type.code()] << "\n";
            debug(1) << "\t\t\tType bits: " << arg.type.bits() << "\n";
            debug(1) << "\t\t\tType lanes: " << arg.type.lanes() << "\n";

            if (arg.is_scalar() && arg.type.is_int_or_uint()) {
                inputs.push_back(builder.getIntegerType(arg.type.bits(), arg.type.is_int()));
                inputNames.push_back(arg.name);
            } else if (arg.is_buffer()) {
                struct BufferDescriptorEntry {
                    std::string suffix;
                    mlir::Type type;
                };

                std::vector<BufferDescriptorEntry> entries;

                // A pointer to the start of the data in main memory (offset of the buffer into the AXI4 master interface)
                entries.push_back({"host", i64});
                // The dimensionality of the buffer
                entries.push_back({"dimensions", i32});

                // Buffer dimensions
                for (int i = 0; i < arg.dimensions; i++) {
                    entries.push_back({"dim_" + std::to_string(i) + "_min", i32});
                    entries.push_back({"dim_" + std::to_string(i) + "_extent", i32});
                    entries.push_back({"dim_" + std::to_string(i) + "_stride", i32});
                }

                for (const auto &entry: entries) {
                    inputs.push_back(entry.type);
                    const std::string name = arg.name + "_" + entry.suffix;
                    inputNames.push_back(name);
                    argAttrs.push_back(builder.getDictionaryAttr(builder.getNamedAttr(circt::scfToCalyx::sPortNameAttr, builder.getStringAttr(name))));
                }

                // Treat buffers as 1D
                inputs.push_back(mlir::MemRefType::get({0}, builder.getIntegerType(arg.type.bits())));
                const std::string name = arg.name + ".buffer";
                inputNames.push_back(name);
                argAttrs.push_back(builder.getDictionaryAttr(builder.getNamedAttr(circt::scfToCalyx::sPortNameAttr, builder.getStringAttr(name))));
            }
        }

        mlir::FunctionType functionType = builder.getFunctionType(inputs, results);
        mlir::func::FuncOp functionOp = builder.create<mlir::func::FuncOp>(builder.getStringAttr(function.name), functionType, attrs, argAttrs);
        builder.setInsertionPointToStart(functionOp.addEntryBlock());

        CodeGen_CIRCT::Visitor visitor(builder, inputNames);
        function.body.accept(&visitor);

        builder.create<mlir::func::ReturnOp>();

        generateKernelXml(function);
    }

    // Print MLIR before running passes
    std::cout << "Original MLIR" << std::endl;
    mlir_module.dump();

    // Verify module (before running passes)
    auto moduleVerifyResult = mlir::verify(mlir_module);
    std::cout << "Module verify (before passess) result: " << moduleVerifyResult.succeeded() << std::endl;
    internal_assert(moduleVerifyResult.succeeded());

    // Create and run passes
    std::cout << "Running passes to Calyx." << std::endl;
    mlir::PassManager pmToCalyx(mlir_module.getContext());
    pmToCalyx.addPass(mlir::createForToWhileLoopPass());
    pmToCalyx.addPass(mlir::createCanonicalizerPass());
    pmToCalyx.addPass(circt::createSCFToCalyxPass());
    pmToCalyx.addPass(mlir::createCanonicalizerPass());

    auto pmToCalyxRunResult = pmToCalyx.run(mlir_module);
    std::cout << "Passes to Calyx result: " << pmToCalyxRunResult.succeeded() << std::endl;

    // Print MLIR after running passes
    std::cout << "MLIR after running passes to Calyx" << std::endl;
    mlir_module.dump();
    internal_assert(pmToCalyxRunResult.succeeded());

    // Emit Calyx
    if (pmToCalyxRunResult.succeeded()) {
        std::string str;
        llvm::raw_string_ostream os(str);
        std::cout << "Exporting Calyx." << std::endl;
        auto exportVerilogResult = circt::calyx::exportCalyx(mlir_module, os);
        std::cout << "Export Calyx result: " << exportVerilogResult.succeeded() << std::endl;
        std::cout << str << std::endl;
    }

    std::cout << "Running passes to SystemVerilog." << std::endl;
    mlir::PassManager pmToSV(mlir_module.getContext());
    pmToSV.nest<circt::calyx::ComponentOp>().addPass(circt::calyx::createRemoveCombGroupsPass());
    pmToSV.addPass(mlir::createCanonicalizerPass());
    pmToSV.nest<circt::calyx::ComponentOp>().addPass(circt::createCalyxToFSMPass());
    pmToSV.addPass(mlir::createCanonicalizerPass());
    pmToSV.nest<circt::calyx::ComponentOp>().addPass(circt::createMaterializeCalyxToFSMPass());
    pmToSV.addPass(mlir::createCanonicalizerPass());
    pmToSV.nest<circt::calyx::ComponentOp>().addPass(circt::createRemoveGroupsFromFSMPass());
    pmToSV.addPass(mlir::createCanonicalizerPass());
    pmToSV.addPass(circt::createCalyxToHWPass());
    pmToSV.addPass(mlir::createCanonicalizerPass());
    pmToSV.addPass(circt::createConvertFSMToSVPass());
    pmToSV.addPass(mlir::createCanonicalizerPass());
    pmToSV.addPass(circt::seq::createSeqLowerToSVPass());
    pmToSV.addPass(mlir::createCanonicalizerPass());

    auto pmToSVRunResult = pmToSV.run(mlir_module);
    std::cout << "Passes to SystemVerilog result: " << pmToSVRunResult.succeeded() << std::endl;

    // Print MLIR after running passes
    std::cout << "MLIR after running passes to SystemVerilog" << std::endl;
    mlir_module.dump();
    internal_assert(pmToSVRunResult.succeeded());

    // Verify module (after running passes)
    moduleVerifyResult = mlir::verify(mlir_module);
    std::cout << "Module verify (after passes) result: " << moduleVerifyResult.succeeded() << std::endl;
    internal_assert(moduleVerifyResult.succeeded());

    // Emit Verilog
    if (pmToCalyxRunResult.succeeded()) {
        std::cout << "Exporting Verilog." << std::endl;
        auto exportVerilogResult = circt::exportSplitVerilog(mlir_module, module.name() + "_generated");
        std::cout << "Export Verilog result: " << exportVerilogResult.succeeded() << std::endl;
    }

    std::cout << "Done!" << std::endl;
}

void CodeGen_CIRCT::generateKernelXml(const Internal::LoweredFunc &function) {
    XMLDocument doc;
    doc.InsertFirstChild(doc.NewDeclaration());

    XMLElement *pRoot = doc.NewElement("root");
    pRoot->SetAttribute("versionMajor", 1);
    pRoot->SetAttribute("versionMinor", 6);
    doc.InsertEndChild(pRoot);

    XMLElement *pKernel = doc.NewElement("kernel");
    pKernel->SetAttribute("name", function.name.c_str());
    pKernel->SetAttribute("language", "ip_c");
    pKernel->SetAttribute("vlnv", std::string("halide-lang.org:kernel:" + function.name + ":1.0").c_str());
    pKernel->SetAttribute("attributes", "");
    pKernel->SetAttribute("preferredWorkGroupSizeMultiple", 0);
    pKernel->SetAttribute("workGroupSize", 1);
    pKernel->SetAttribute("interrupt", 1);
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

    auto genAxiPortName = [](int id) {
        return "m" + std::string(id < 10 ? "0" : "") + std::to_string(id) + "_axi";
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
        }
    };

    XMLElement *pPorts = doc.NewElement("ports");
    XMLElement *pArgs = doc.NewElement("args");

    pPorts->InsertEndChild(genPort("s_axi_control", "slave", 0x1000, 32));

    uint64_t bufCnt = 0;
    uint64_t argIdx = 0;
    uint64_t argOffset = 0x10; // XRT-Managed Kernels Control Requirements
    for (const auto &arg: function.args) {
        if (arg.is_scalar() && arg.type.is_int_or_uint()) {
            pArgs->InsertEndChild(genArg(arg.name, 0, argIdx++, "s_axi_control", arg.type.bytes(),
                                  argOffset += 8, genTypeStr(arg.type), 0, arg.type.bytes()));
        } else if (arg.is_buffer()) {
            pPorts->InsertEndChild(genPort(genAxiPortName(bufCnt), "master", std::numeric_limits<uint64_t>::max(), 512));
            pArgs->InsertEndChild(genArg(arg.name + "_host", 1, argIdx++, genAxiPortName(bufCnt), 8,
                                  argOffset += 8, genTypeStr(arg.type) + "*", 0, 8));
            pArgs->InsertEndChild(genArg(arg.name + "_dimensions", 0, argIdx++, "s_axi_control", 4, argOffset += 8, "int", 0, 4));
            for (int i = 0; i < arg.dimensions; i++) {
                pArgs->InsertEndChild(genArg(arg.name + "_dim_" + std::to_string(i) + "_min", 0,
                                      argIdx++, "s_axi_control", 4, argOffset += 8, "int", 0, 4));
                pArgs->InsertEndChild(genArg(arg.name + "_dim_" + std::to_string(i) + "_extent", 0,
                                      argIdx++, "s_axi_control", 4, argOffset += 8, "int", 0, 4));
                pArgs->InsertEndChild(genArg(arg.name + "_dim_" + std::to_string(i) + "_stride", 0,
                                      argIdx++, "s_axi_control", 4, argOffset += 8, "int", 0, 4));
            }
            bufCnt++;
        }
    }

    pRoot->InsertEndChild(pPorts);
    pRoot->InsertEndChild(pArgs);

    XMLPrinter printer;
    doc.Print(&printer);
    std::cout << printer.CStr() << std::endl;
}

CodeGen_CIRCT::Visitor::Visitor(mlir::ImplicitLocOpBuilder &builder, const std::vector<std::string> &inputNames)
    : builder(builder) {

    mlir::func::FuncOp funcOp = cast<mlir::func::FuncOp>(builder.getBlock()->getParentOp());

    // Add function arguments to the symbol table
    for (unsigned int i = 0; i < funcOp.getFunctionType().getNumInputs(); i++) {
        std::string name = inputNames[i];
        sym_push(name, funcOp.getArgument(i));
    }

#if 0
     /*   circt::hw::StructType dim_type = builder.getType<circt::hw::StructType>(mlir::SmallVector({
            circt::hw::StructType::FieldInfo{builder.getStringAttr("min"), i32},
            circt::hw::StructType::FieldInfo{builder.getStringAttr("extent"), i32},
            circt::hw::StructType::FieldInfo{builder.getStringAttr("stride"), i32}
        }));*/

       // builder.create<circt::hw::StructCreateOp>(dim_type, {builder.create<mlir::arith::ConstantOp>(type, builder.getIntegerAttr(type, op->value));
    mlir::Type i32 = builder.getI32Type();
    mlir::Type i64 = builder.getI64Type();
    mlir::TupleType type = builder.getTupleType({i32, i64, i32});

    builder.create<mlir::arith::ConstantOp>(type, builder.getIntegerAttr(type, 43));
#endif
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
    debug(1) << __PRETTY_FUNCTION__ << "\n";{}

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

    //value = builder.create<mlir::arith::CmpIOp>(predicate, codegen(op->a), codegen(op->b));
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

    mlir::Value trueValue = codegen(op->true_value);
    mlir::Value falseValue = codegen(op->false_value);

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

    mlir::Value baseAddr = sym_get(op->name + "_host");
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
    for (const Expr &e: op->args)
        debug(1) << "\tArg: " << e << "\n";

    mlir::Type op_type = builder.getIntegerType(op->type.bits());

    if (op->name == Call::buffer_get_host) {
        auto name = op->args[0].as<Variable>()->name;
        name = name.substr(0, name.find(".buffer"));
        value = sym_get(name + "_host");
    } else if(op->name == Call::buffer_get_min) {
        auto name = op->args[0].as<Variable>()->name;
        name = name.substr(0, name.find(".buffer"));
        int32_t d = op->args[1].as<IntImm>()->value;
        value = sym_get(name + "_dim_" + std::to_string(d) + "_min");
    } else if(op->name == Call::buffer_get_extent) {
        auto name = op->args[0].as<Variable>()->name;
        name = name.substr(0, name.find(".buffer"));
        int32_t d = op->args[1].as<IntImm>()->value;
        value = sym_get(name + "_dim_" + std::to_string(d) + "_extent");
    } else if(op->name == Call::buffer_get_stride) {
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

    mlir::Value baseAddr = sym_get(op->name + "_host");
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

}  // namespace Internal
}  // namespace Halide
