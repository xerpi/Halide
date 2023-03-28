#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/Transforms/Passes.h>
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

#include "CodeGen_CIRCT_Dev.h"
#include "Debug.h"
#include "IROperator.h"

namespace Halide {

namespace Internal {

bool CodeGen_CIRCT_Dev::compile(mlir::LocationAttr &loc, mlir::ModuleOp &mlir_module, Stmt stmt, const std::string &name,
                                const std::vector<DeviceArgument> &args, std::string &calyxOutput) {
    debug(1) << "[Compiling kernel '" << name << "']\n";

    mlir::ImplicitLocOpBuilder builder = mlir::ImplicitLocOpBuilder::atBlockEnd(loc, mlir_module.getBody());

    mlir::SmallVector<mlir::Type> inputs;
    std::vector<std::string> inputNames;
    mlir::SmallVector<mlir::Type> results;
    mlir::SmallVector<mlir::NamedAttribute> attrs;
    mlir::SmallVector<mlir::DictionaryAttr> argAttrs;

    for (const auto &arg : args) {
        inputs.push_back(builder.getIntegerType(argGetHWBits(arg)));
        inputNames.push_back(arg.name);
        argAttrs.push_back(builder.getDictionaryAttr(
            builder.getNamedAttr(circt::scfToCalyx::sPortNameAttr, builder.getStringAttr(arg.name))));
    }

    for (const auto &arg : args) {
        if (arg.is_buffer) {
            // Add memref to the arguments. Treat buffers as 1D
            inputs.push_back(mlir::MemRefType::get({0}, builder.getIntegerType(arg.type.bits())));
            inputNames.push_back(arg.name + ".buffer");
            argAttrs.push_back(builder.getDictionaryAttr(
                builder.getNamedAttr(circt::scfToCalyx::sSequentialReads, builder.getBoolAttr(true))));
        }
    }

    mlir::FunctionType functionType = builder.getFunctionType(inputs, results);
    mlir::func::FuncOp functionOp = builder.create<mlir::func::FuncOp>(builder.getStringAttr(name), functionType, attrs, argAttrs);
    builder.setInsertionPointToStart(functionOp.addEntryBlock());

    CodeGen_CIRCT_Dev::Visitor visitor(builder, inputNames);
    stmt.accept(&visitor);

    builder.create<mlir::func::ReturnOp>();

    // Print MLIR before running passes
    debug(1) << "Original MLIR:\n";
    mlir_module.dump();

    // Verify module (before running passes)
    auto moduleVerifyResult = mlir::verify(mlir_module);
    debug(1) << "Module verify (before passess) result: " << moduleVerifyResult.succeeded() << "\n";
    internal_assert(moduleVerifyResult.succeeded());

    // Create and run passes
    debug(1) << "[SCF to Calyx] Start.\n";
    mlir::PassManager pmSCFToCalyx(mlir_module.getContext());
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
    llvm::raw_string_ostream os(calyxOutput);
    debug(1) << "[Exporting Calyx]\n";
    auto exportVerilogResult = circt::calyx::exportCalyx(mlir_module, os);
    debug(1) << "[Export Calyx] Result: " << exportVerilogResult.succeeded() << "\n";

    debug(1) << "[Calyx to FSM] Start.\n";
    mlir::PassManager pmCalyxToFSM(mlir_module.getContext());
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

    debug(1) << "[Compiled kernel '" << name << "']\n";
    return true;
}

CodeGen_CIRCT_Dev::Visitor::Visitor(mlir::ImplicitLocOpBuilder &builder, const std::vector<std::string> &inputNames)
    : builder(builder) {

    mlir::func::FuncOp funcOp = cast<mlir::func::FuncOp>(builder.getBlock()->getParentOp());

    // Add function arguments to the symbol table
    for (unsigned int i = 0; i < funcOp.getFunctionType().getNumInputs(); i++) {
        std::string name = inputNames[i];
        sym_push(name, funcOp.getArgument(i));
    }
}

mlir::Value CodeGen_CIRCT_Dev::Visitor::codegen(const Expr &e) {
    internal_assert(e.defined());
    debug(4) << "Codegen (E): " << e.type() << ", " << e << "\n";
    value = mlir::Value();
    e.accept(this);
    internal_assert(value) << "Codegen of an expr did not produce a MLIR value\n"
                           << e;
    return value;
}

void CodeGen_CIRCT_Dev::Visitor::codegen(const Stmt &s) {
    internal_assert(s.defined());
    debug(4) << "Codegen (S): " << s << "\n";
    value = mlir::Value();
    s.accept(this);
}

void CodeGen_CIRCT_Dev::Visitor::visit(const IntImm *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    mlir::Type type = builder.getIntegerType(op->type.bits());
    value = builder.create<mlir::arith::ConstantOp>(type, builder.getIntegerAttr(type, op->value));
}

void CodeGen_CIRCT_Dev::Visitor::visit(const UIntImm *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    mlir::Type type = builder.getIntegerType(op->type.bits());
    value = builder.create<mlir::arith::ConstantOp>(type, builder.getIntegerAttr(type, op->value));
}

void CodeGen_CIRCT_Dev::Visitor::visit(const FloatImm *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT_Dev::Visitor::visit(const StringImm *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Cast *op) {
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

void CodeGen_CIRCT_Dev::Visitor::visit(const Reinterpret *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Variable *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    debug(1) << "\tname: " << op->name << "\n";
    value = sym_get(op->name, true);
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Add *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<mlir::arith::AddIOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Sub *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<mlir::arith::SubIOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Mul *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<mlir::arith::MulIOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Div *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    if (op->type.is_int()) {
        value = builder.create<mlir::arith::DivSIOp>(codegen(op->a), codegen(op->b));
    } else {
        value = builder.create<mlir::arith::DivUIOp>(codegen(op->a), codegen(op->b));
    }
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Mod *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    if (op->type.is_int()) {
        value = builder.create<mlir::arith::RemSIOp>(codegen(op->a), codegen(op->b));
    } else {
        value = builder.create<mlir::arith::RemUIOp>(codegen(op->a), codegen(op->b));
    }
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Min *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    std::string a_name = unique_name('a');
    std::string b_name = unique_name('b');
    Expr a = Variable::make(op->a.type(), a_name);
    Expr b = Variable::make(op->b.type(), b_name);
    value = codegen(Let::make(a_name, op->a, Let::make(b_name, op->b, select(a < b, a, b))));
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Max *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    std::string a_name = unique_name('a');
    std::string b_name = unique_name('b');
    Expr a = Variable::make(op->a.type(), a_name);
    Expr b = Variable::make(op->b.type(), b_name);
    value = codegen(Let::make(a_name, op->a, Let::make(b_name, op->b, select(a > b, a, b))));
}

void CodeGen_CIRCT_Dev::Visitor::visit(const EQ *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::eq, codegen(op->a), codegen(op->b));
}

void CodeGen_CIRCT_Dev::Visitor::visit(const NE *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::ne, codegen(op->a), codegen(op->b));
}

void CodeGen_CIRCT_Dev::Visitor::visit(const LT *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    mlir::arith::CmpIPredicate predicate = op->type.is_int() ? mlir::arith::CmpIPredicate::slt :
                                                               mlir::arith::CmpIPredicate::ult;
    value = builder.create<mlir::arith::CmpIOp>(predicate, codegen(op->a), codegen(op->b));
}

void CodeGen_CIRCT_Dev::Visitor::visit(const LE *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    mlir::arith::CmpIPredicate predicate = op->type.is_int() ? mlir::arith::CmpIPredicate::sle :
                                                               mlir::arith::CmpIPredicate::ule;
    value = builder.create<mlir::arith::CmpIOp>(predicate, codegen(op->a), codegen(op->b));
}

void CodeGen_CIRCT_Dev::Visitor::visit(const GT *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    mlir::arith::CmpIPredicate predicate = op->type.is_int() ? mlir::arith::CmpIPredicate::sgt :
                                                               mlir::arith::CmpIPredicate::ugt;
    value = builder.create<mlir::arith::CmpIOp>(predicate, codegen(op->a), codegen(op->b));
}

void CodeGen_CIRCT_Dev::Visitor::visit(const GE *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    mlir::arith::CmpIPredicate predicate = op->type.is_int() ? mlir::arith::CmpIPredicate::sge :
                                                               mlir::arith::CmpIPredicate::uge;
    value = builder.create<mlir::arith::CmpIOp>(predicate, codegen(op->a), codegen(op->b));
}

void CodeGen_CIRCT_Dev::Visitor::visit(const And *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    // value = builder.create<mlir::arith::CmpIOp>(predicate, codegen(op->a), codegen(op->b));
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Or *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Not *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    mlir::Value a = codegen(op->a);
    mlir::Value allZeroes = builder.create<mlir::arith::ConstantOp>(a.getType(), builder.getIntegerAttr(a.getType(), 0));
    value = builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::eq, a, allZeroes);
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Select *op) {
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

void CodeGen_CIRCT_Dev::Visitor::visit(const Load *op) {
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

void CodeGen_CIRCT_Dev::Visitor::visit(const Ramp *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Broadcast *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Call *op) {
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

void CodeGen_CIRCT_Dev::Visitor::visit(const Let *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    sym_push(op->name, codegen(op->value));
    value = codegen(op->body);
    sym_pop(op->name);
}

void CodeGen_CIRCT_Dev::Visitor::visit(const LetStmt *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    debug(1) << "Contents:"
             << "\n";
    debug(1) << "\tName: " << op->name << "\n";
    sym_push(op->name, codegen(op->value));
    codegen(op->body);
    sym_pop(op->name);
}

void CodeGen_CIRCT_Dev::Visitor::visit(const AssertStmt *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT_Dev::Visitor::visit(const ProducerConsumer *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    debug(1) << "\tName: " << op->name << "\n";
    debug(1) << "\tIs producer: " << op->is_producer << "\n";
    codegen(op->body);
}

void CodeGen_CIRCT_Dev::Visitor::visit(const For *op) {
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

void CodeGen_CIRCT_Dev::Visitor::visit(const Store *op) {
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

void CodeGen_CIRCT_Dev::Visitor::visit(const Provide *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Allocate *op) {
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

void CodeGen_CIRCT_Dev::Visitor::visit(const Free *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Realize *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Block *op) {
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

void CodeGen_CIRCT_Dev::Visitor::visit(const IfThenElse *op) {
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

void CodeGen_CIRCT_Dev::Visitor::visit(const Evaluate *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    codegen(op->value);

    // Discard result
    value = mlir::Value();
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Shuffle *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT_Dev::Visitor::visit(const VectorReduce *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Prefetch *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Fork *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Acquire *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT_Dev::Visitor::visit(const Atomic *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT_Dev::Visitor::sym_push(const std::string &name, mlir::Value value) {
    // value.getDefiningOp()->setAttr("sv.namehint", builder.getStringAttr(name));
    symbol_table.push(name, value);
}

void CodeGen_CIRCT_Dev::Visitor::sym_pop(const std::string &name) {
    symbol_table.pop(name);
}

mlir::Value CodeGen_CIRCT_Dev::Visitor::sym_get(const std::string &name, bool must_succeed) const {
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
