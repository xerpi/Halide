#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Passes.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "CodeGen_Internal.h"
#include "CodeGen_MLIR.h"
#include "Debug.h"
#include "IROperator.h"

namespace Halide {

namespace Internal {

bool CodeGen_MLIR::compile(mlir::LocationAttr &loc, mlir::ModuleOp &mlir_module, Stmt stmt, const std::string &name,
                           const std::vector<DeviceArgument> &args, int axiDataWidth) {
    mlir::ImplicitLocOpBuilder builder = mlir::ImplicitLocOpBuilder::atBlockEnd(loc, mlir_module.getBody());

    mlir::SmallVector<mlir::Type> inputs;
    std::vector<std::string> inputNames;
    mlir::SmallVector<mlir::Type> results;
    mlir::SmallVector<mlir::NamedAttribute> funcAttrs;
    mlir::SmallVector<mlir::DictionaryAttr> funcArgAttrs;

    for (const auto &arg : args) {
        inputs.push_back(builder.getIntegerType(argGetHWBits(arg)));
        inputNames.push_back(arg.name);
    }

    // Put memrefs at the end. Treat buffers as 1D
    for (const auto &arg : args) {
        if (arg.is_buffer) {
            inputs.push_back(mlir::MemRefType::get({0}, builder.getIntegerType(arg.type.bits())));
            inputNames.push_back(arg.name + ".buffer");
        }
    }

    mlir::FunctionType functionType = builder.getFunctionType(inputs, results);
    mlir::func::FuncOp functionOp = builder.create<mlir::func::FuncOp>(builder.getStringAttr(name), functionType, funcAttrs, funcArgAttrs);
    builder.setInsertionPointToStart(functionOp.addEntryBlock());

    CodeGen_MLIR::Visitor visitor(builder, inputNames);
    stmt.accept(&visitor);
    builder.create<mlir::func::ReturnOp>();

    return mlir::verify(mlir_module).succeeded();
}

CodeGen_MLIR::Visitor::Visitor(mlir::ImplicitLocOpBuilder &builder, const std::vector<std::string> &inputNames)
    : builder(builder) {

    mlir::func::FuncOp funcOp = cast<mlir::func::FuncOp>(builder.getBlock()->getParentOp());

    // Add function arguments to the symbol table
    for (unsigned int i = 0; i < funcOp.getFunctionType().getNumInputs(); i++)
        sym_push(inputNames[i], funcOp.getArgument(i));
}

mlir::Value CodeGen_MLIR::Visitor::codegen(const Expr &e) {
    internal_assert(e.defined());
    debug(4) << "Codegen (E): " << e.type() << ", " << e << "\n";
    value = mlir::Value();
    e.accept(this);
    internal_assert(value) << "Codegen of an expr did not produce a MLIR value\n"
                           << e;
    return value;
}

void CodeGen_MLIR::Visitor::codegen(const Stmt &s) {
    internal_assert(s.defined());
    debug(4) << "Codegen (S): " << s << "\n";
    value = mlir::Value();
    s.accept(this);
}

void CodeGen_MLIR::Visitor::visit(const IntImm *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
    mlir::Type type = builder.getIntegerType(op->type.bits());
    value = builder.create<mlir::arith::ConstantOp>(type, builder.getIntegerAttr(type, op->value));
}

void CodeGen_MLIR::Visitor::visit(const UIntImm *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
    mlir::Type type = builder.getIntegerType(op->type.bits());
    value = builder.create<mlir::arith::ConstantOp>(type, builder.getIntegerAttr(type, op->value));
}

void CodeGen_MLIR::Visitor::visit(const FloatImm *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_MLIR::Visitor::visit(const StringImm *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_MLIR::Visitor::visit(const Cast *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    Halide::Type src = op->value.type();
    Halide::Type dst = op->type;
    mlir::Type dstType = builder.getIntegerType(dst.bits());

    debug(3) << "\tSrc type: " << src << "\n";
    debug(3) << "\tDst type: " << dst << "\n";

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

void CodeGen_MLIR::Visitor::visit(const Reinterpret *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_MLIR::Visitor::visit(const Variable *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
    debug(3) << "\tname: " << op->name << "\n";
    value = sym_get(op->name, true);
}

void CodeGen_MLIR::Visitor::visit(const Add *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<mlir::arith::AddIOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR::Visitor::visit(const Sub *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<mlir::arith::SubIOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR::Visitor::visit(const Mul *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<mlir::arith::MulIOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR::Visitor::visit(const Div *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    if (op->type.is_int()) {
        value = builder.create<mlir::arith::DivSIOp>(codegen(op->a), codegen(op->b));
    } else {
        value = builder.create<mlir::arith::DivUIOp>(codegen(op->a), codegen(op->b));
    }
}

void CodeGen_MLIR::Visitor::visit(const Mod *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    if (op->type.is_int()) {
        value = builder.create<mlir::arith::RemSIOp>(codegen(op->a), codegen(op->b));
    } else {
        value = builder.create<mlir::arith::RemUIOp>(codegen(op->a), codegen(op->b));
    }
}

void CodeGen_MLIR::Visitor::visit(const Min *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    if (op->type.is_int()) {
        value = builder.create<mlir::arith::MinSIOp>(codegen(op->a), codegen(op->b));
    } else {
        value = builder.create<mlir::arith::MinUIOp>(codegen(op->a), codegen(op->b));
    }
}

void CodeGen_MLIR::Visitor::visit(const Max *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    if (op->type.is_int()) {
        value = builder.create<mlir::arith::MaxSIOp>(codegen(op->a), codegen(op->b));
    } else {
        value = builder.create<mlir::arith::MaxUIOp>(codegen(op->a), codegen(op->b));
    }
}

void CodeGen_MLIR::Visitor::visit(const EQ *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::eq, codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR::Visitor::visit(const NE *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::ne, codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR::Visitor::visit(const LT *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    mlir::arith::CmpIPredicate predicate = op->type.is_int() ? mlir::arith::CmpIPredicate::slt :
                                                               mlir::arith::CmpIPredicate::ult;
    value = builder.create<mlir::arith::CmpIOp>(predicate, codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR::Visitor::visit(const LE *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    mlir::arith::CmpIPredicate predicate = op->type.is_int() ? mlir::arith::CmpIPredicate::sle :
                                                               mlir::arith::CmpIPredicate::ule;
    value = builder.create<mlir::arith::CmpIOp>(predicate, codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR::Visitor::visit(const GT *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    mlir::arith::CmpIPredicate predicate = op->type.is_int() ? mlir::arith::CmpIPredicate::sgt :
                                                               mlir::arith::CmpIPredicate::ugt;
    value = builder.create<mlir::arith::CmpIOp>(predicate, codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR::Visitor::visit(const GE *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    mlir::arith::CmpIPredicate predicate = op->type.is_int() ? mlir::arith::CmpIPredicate::sge :
                                                               mlir::arith::CmpIPredicate::uge;
    value = builder.create<mlir::arith::CmpIOp>(predicate, codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR::Visitor::visit(const And *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<mlir::arith::AndIOp>(codegen(NE::make(op->a, 0)), codegen(NE::make(op->b, 0)));
}

void CodeGen_MLIR::Visitor::visit(const Or *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<mlir::arith::OrIOp>(codegen(NE::make(op->a, 0)), codegen(NE::make(op->b, 0)));
}

void CodeGen_MLIR::Visitor::visit(const Not *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    value = codegen(EQ::make(op->a, 0));
}

void CodeGen_MLIR::Visitor::visit(const Select *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

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

void CodeGen_MLIR::Visitor::visit(const Load *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
    debug(3) << "\tName: " << op->name << "\n";

    mlir::Value buffer = sym_get(op->name + ".buffer");
    mlir::Value index;
    if (op->type.is_scalar()) {
        index = codegen(op->index);
    } else if (Expr ramp_base = strided_ramp_base(op->index); ramp_base.defined()) {
        index = codegen(ramp_base);
    } else {
        user_error << "Unsupported load.";
    }

    mlir::Value baseAddr = sym_get(op->name);
    mlir::Value indexI64 = builder.create<mlir::arith::ExtUIOp>(builder.getI64Type(), index);
    mlir::Value elementSize = builder.create<mlir::arith::ConstantOp>(builder.getI64IntegerAttr(op->type.bytes()));
    mlir::Value offset = builder.create<mlir::arith::MulIOp>(indexI64, elementSize);
    mlir::Value loadAddress = builder.create<mlir::arith::AddIOp>(baseAddr, offset);
    mlir::Value loadAddressAsIndex = builder.create<mlir::arith::IndexCastOp>(builder.getIndexType(), loadAddress);

    if (op->type.is_scalar()) {
        value = builder.create<mlir::memref::LoadOp>(buffer, mlir::ValueRange{loadAddressAsIndex});
    } else {
        mlir::Type elementType = builder.getIntegerType(op->type.bits());
        mlir::VectorType vectorType = mlir::VectorType::get(op->type.lanes(), elementType);
        value = builder.create<mlir::vector::LoadOp>(vectorType, buffer, mlir::ValueRange{loadAddressAsIndex});
    }
}

void CodeGen_MLIR::Visitor::visit(const Ramp *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    mlir::Value base = codegen(op->base);
    mlir::Value stride = codegen(op->stride);
    mlir::Type elementType = builder.getIntegerType(op->base.type().bits());
    mlir::VectorType vectorType = mlir::VectorType::get(op->lanes, elementType);

    mlir::SmallVector<mlir::Attribute> indicesAttrs(op->lanes);
    for (int i = 0; i < op->lanes; i++)
        indicesAttrs[i] = mlir::IntegerAttr::get(elementType, i);

    mlir::DenseElementsAttr indicesDenseAttr = mlir::DenseElementsAttr::get(vectorType, indicesAttrs);
    mlir::Value indicesConst = builder.create<mlir::arith::ConstantOp>(indicesDenseAttr);
    mlir::Value splatStride = builder.create<mlir::vector::SplatOp>(vectorType, stride);
    mlir::Value offsets = builder.create<mlir::arith::MulIOp>(splatStride, indicesConst);
    mlir::Value splatBase = builder.create<mlir::vector::SplatOp>(vectorType, base);
    value = builder.create<mlir::arith::AddIOp>(splatBase, offsets);
}

void CodeGen_MLIR::Visitor::visit(const Broadcast *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    mlir::Type elementType = builder.getIntegerType(op->value.type().bits());
    mlir::VectorType vectorType = mlir::VectorType::get(op->lanes, elementType);
    value = builder.create<mlir::vector::SplatOp>(vectorType, codegen(op->value));
}

void CodeGen_MLIR::Visitor::visit(const Call *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
    debug(3) << "\tName: " << op->name << "\n";
    debug(3) << "\tCall type: " << op->call_type << "\n";
    for (const Expr &e : op->args)
        debug(3) << "\tArg: " << e << "\n";

    if (op->is_intrinsic(Call::shift_left)) {
        value = builder.create<mlir::arith::ShLIOp>(codegen(op->args[0]), codegen(op->args[1]));
    } else if (op->is_intrinsic(Call::shift_right)) {
        if (op->type.is_int())
            value = builder.create<mlir::arith::ShRSIOp>(codegen(op->args[0]), codegen(op->args[1]));
        else
            value = builder.create<mlir::arith::ShRUIOp>(codegen(op->args[0]), codegen(op->args[1]));
    } else if (op->is_intrinsic(Call::widen_right_mul)) {
        mlir::Value a = codegen(op->args[0]);
        mlir::Value b = codegen(op->args[1]);
        if (op->type.is_int())
            b = builder.create<mlir::arith::ExtSIOp>(a.getType(), b);
        else
            b = builder.create<mlir::arith::ExtUIOp>(a.getType(), b);
        value = builder.create<mlir::arith::MulIOp>(a, b);
    } else {
        mlir::Type op_type = builder.getIntegerType(op->type.bits());
        // Just return 1 for now
        value = builder.create<mlir::arith::ConstantOp>(op_type, builder.getIntegerAttr(op_type, 1));

        internal_error << "CodeGen_MLIR::Visitor::visit(const Call *op): " << op->name << " not implemented\n";
    }
}

void CodeGen_MLIR::Visitor::visit(const Let *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    sym_push(op->name, codegen(op->value));
    value = codegen(op->body);
    sym_pop(op->name);
}

void CodeGen_MLIR::Visitor::visit(const LetStmt *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
    debug(3) << "Contents:\n";
    debug(3) << "\tName: " << op->name << "\n";
    sym_push(op->name, codegen(op->value));
    codegen(op->body);
    sym_pop(op->name);
}

void CodeGen_MLIR::Visitor::visit(const AssertStmt *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_MLIR::Visitor::visit(const ProducerConsumer *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
    debug(3) << "\tName: " << op->name << "\n";
    debug(3) << "\tIs producer: " << op->is_producer << "\n";
    codegen(op->body);
}

void CodeGen_MLIR::Visitor::visit(const For *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
    debug(3) << "\tName: " << op->name << "\n";
    debug(3) << "\tMin: " << op->min << "\n";
    debug(3) << "\tExtent: " << op->extent << "\n";
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
    debug(3) << "\tForType: " << for_types[unsigned(op->for_type)] << "\n";

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

void CodeGen_MLIR::Visitor::visit(const Store *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
    debug(3) << "\tName: " << op->name << "\n";
    debug(3) << "\tValue lanes: " << op->value.type().lanes() << "\n";

    mlir::Value value = codegen(op->value);
    mlir::Value buffer = sym_get(op->name + ".buffer");
    mlir::Value index;
    if (op->value.type().is_scalar()) {
        index = codegen(op->index);
    } else if (Expr ramp_base = strided_ramp_base(op->index); ramp_base.defined()) {
        index = codegen(ramp_base);
    } else {
        user_error << "Unsupported store.";
    }

    mlir::Value baseAddr = sym_get(op->name);
    mlir::Value indexI64 = builder.create<mlir::arith::ExtUIOp>(builder.getI64Type(), index);
    mlir::Value elementSize = builder.create<mlir::arith::ConstantOp>(builder.getI64IntegerAttr(op->value.type().bytes()));
    mlir::Value offset = builder.create<mlir::arith::MulIOp>(indexI64, elementSize);
    mlir::Value storeAddress = builder.create<mlir::arith::AddIOp>(baseAddr, offset);
    mlir::Value storeAddressAsIndex = builder.create<mlir::arith::IndexCastOp>(builder.getIndexType(), storeAddress);

    if (op->value.type().is_scalar())
        builder.create<mlir::memref::StoreOp>(value, buffer, mlir::ValueRange{storeAddressAsIndex});
    else
        builder.create<mlir::vector::StoreOp>(value, buffer, mlir::ValueRange{storeAddressAsIndex});
}

void CodeGen_MLIR::Visitor::visit(const Provide *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_MLIR::Visitor::visit(const Allocate *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    int32_t size = op->constant_allocation_size();

    debug(3) << "  name: " << op->name << "\n";
    debug(3) << "  type: " << op->type << "\n";
    debug(3) << "  memory_type: " << int(op->memory_type) << "\n";
    debug(3) << "  size: " << size << "\n";

    for (auto &ext : op->extents) {
        debug(3) << "  ext: " << ext << "\n";
    }

    codegen(op->body);
}

void CodeGen_MLIR::Visitor::visit(const Free *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_MLIR::Visitor::visit(const Realize *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_MLIR::Visitor::visit(const Block *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
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

void CodeGen_MLIR::Visitor::visit(const IfThenElse *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
    debug(3) << "Contents:\n";
    debug(3) << "\tcondition: " << op->condition << "\n";
    debug(3) << "\tthen_case: " << op->then_case << "\n";
    debug(3) << "\telse_case: " << op->else_case << "\n";

    codegen(op->condition);
    codegen(op->then_case);

    if (op->else_case.defined()) {
        codegen(op->else_case);
        // halide_buffer_t
    } else {
    }
}

void CodeGen_MLIR::Visitor::visit(const Evaluate *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
    codegen(op->value);

    // Discard result
    value = mlir::Value();
}

void CodeGen_MLIR::Visitor::visit(const Shuffle *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_MLIR::Visitor::visit(const VectorReduce *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_MLIR::Visitor::visit(const Prefetch *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_MLIR::Visitor::visit(const Fork *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_MLIR::Visitor::visit(const Acquire *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_MLIR::Visitor::visit(const Atomic *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_MLIR::Visitor::sym_push(const std::string &name, mlir::Value value) {
    // value.getDefiningOp()->setAttr("sv.namehint", builder.getStringAttr(name));
    symbol_table.push(name, value);
}

void CodeGen_MLIR::Visitor::sym_pop(const std::string &name) {
    symbol_table.pop(name);
}

mlir::Value CodeGen_MLIR::Visitor::sym_get(const std::string &name, bool must_succeed) const {
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
