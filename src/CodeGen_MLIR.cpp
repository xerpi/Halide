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
#include "Simplify.h"

namespace Halide {

namespace Internal {

bool CodeGen_MLIR::compile(mlir::LocationAttr &loc, mlir::ModuleOp &mlir_module, Stmt stmt, const std::string &name,
                           const std::vector<DeviceArgument> &args, int axiDataWidth) {
    mlir::ImplicitLocOpBuilder builder = mlir::ImplicitLocOpBuilder::atBlockEnd(loc, mlir_module.getBody());

    mlir::SmallVector<mlir::Type> inputs;
    mlir::SmallVector<mlir::Type> results;
    mlir::SmallVector<mlir::NamedAttribute> funcAttrs;
    mlir::SmallVector<mlir::DictionaryAttr> funcArgAttrs;

    for (const auto &arg : args)
        inputs.push_back(arg.is_buffer ? builder.getI64Type() : mlir_type_of(builder, arg.type));

    // Put memrefs at the end. Treat buffers as 1D
    for (const auto &arg : args) {
        if (arg.is_buffer)
            inputs.push_back(mlir::MemRefType::get({0}, mlir_type_of(builder, arg.type)));
    }

    mlir::FunctionType functionType = builder.getFunctionType(inputs, results);
    mlir::func::FuncOp functionOp = builder.create<mlir::func::FuncOp>(builder.getStringAttr(name), functionType, funcAttrs, funcArgAttrs);
    builder.setInsertionPointToStart(functionOp.addEntryBlock());

    CodeGen_MLIR::Visitor visitor(builder, args);
    stmt.accept(&visitor);
    builder.create<mlir::func::ReturnOp>();

    return mlir::verify(mlir_module).succeeded();
}

mlir::Type CodeGen_MLIR::mlir_type_of(mlir::ImplicitLocOpBuilder &builder, Halide::Type t) {
    if (t.lanes() == 1) {
        if (t.is_int_or_uint()) {
            return builder.getIntegerType(t.bits());
        } else if (t.is_bfloat()) {
            return builder.getBF16Type();
        } else if (t.is_float()) {
            switch (t.bits()) {
            case 16:
                return builder.getF16Type();
            case 32:
                return builder.getF32Type();
            case 64:
                return builder.getF64Type();
            default:
                internal_error << "There is no MLIR type matching this floating-point bit width: " << t << "\n";
                return nullptr;
            }
        } else {
            internal_error << "Type not supported: " << t << "\n";
        }
    } else {
        return mlir::VectorType::get(t.lanes(), mlir_type_of(builder, t.element_of()));
    }

    return mlir::Type();
}

CodeGen_MLIR::Visitor::Visitor(mlir::ImplicitLocOpBuilder &builder, const std::vector<DeviceArgument> &args)
    : builder(builder) {

    mlir::func::FuncOp funcOp = cast<mlir::func::FuncOp>(builder.getBlock()->getParentOp());

    // Add function arguments to the symbol table
    size_t bufIdx = 0;
    for (auto [index, arg] : llvm::enumerate(args)) {
        mlir::Value value = funcOp.getArgument(index);
        // MemRefs are at the end. Also shift the base address to a base index
        if (arg.is_buffer) {
            int bits = llvm::Log2_32_Ceil(mlir_type_of(arg.type).getIntOrFloatBitWidth() / 8);
            mlir::Value bitsCst = builder.create<mlir::arith::ConstantOp>(builder.getIntegerAttr(builder.getI64Type(), bits));
            value = builder.create<mlir::arith::ShRUIOp>(value, bitsCst);
            sym_push(arg.name + ".buffer", funcOp.getArgument(bufIdx + args.size()));
            bufIdx++;
        }
        sym_push(arg.name, value);
    }
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
    mlir::Type type = mlir_type_of(op->type);
    value = builder.create<mlir::arith::ConstantOp>(type, builder.getIntegerAttr(type, op->value));
}

void CodeGen_MLIR::Visitor::visit(const UIntImm *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
    mlir::Type type = mlir_type_of(op->type);
    value = builder.create<mlir::arith::ConstantOp>(type, builder.getIntegerAttr(type, op->value));
}

void CodeGen_MLIR::Visitor::visit(const FloatImm *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
    mlir::Type type = mlir_type_of(op->type);
    value = builder.create<mlir::arith::ConstantOp>(type, builder.getFloatAttr(type, op->value));
}

void CodeGen_MLIR::Visitor::visit(const StringImm *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    internal_error << "String immediates are not supported.\n";
}

void CodeGen_MLIR::Visitor::visit(const Cast *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    Halide::Type src = op->value.type();
    Halide::Type dst = op->type;
    mlir::Type mlir_type = mlir_type_of(dst);

    debug(3) << "\tsrc type: " << src << "\n";
    debug(3) << "\tdst type: " << dst << "\n";

    value = codegen(op->value);

    if (src.is_int_or_uint() && dst.is_int_or_uint()) {
        if (dst.bits() > src.bits()) {
            if (src.is_int())
                value = builder.create<mlir::arith::ExtSIOp>(mlir_type, value);
            else
                value = builder.create<mlir::arith::ExtUIOp>(mlir_type, value);
        } else {
            value = builder.create<mlir::arith::TruncIOp>(mlir_type, value);
        }
    } else if (src.is_float() && dst.is_int()) {
        value = builder.create<mlir::arith::FPToSIOp>(mlir_type, value);
    } else if (src.is_float() && dst.is_uint()) {
        value = builder.create<mlir::arith::FPToUIOp>(mlir_type, value);
    } else if (src.is_int() && dst.is_float()) {
        value = builder.create<mlir::arith::SIToFPOp>(mlir_type, value);
    } else if (src.is_uint() && dst.is_float()) {
        value = builder.create<mlir::arith::UIToFPOp>(mlir_type, value);
    } else if (src.is_float() && dst.is_float()) {
        if (dst.bits() > src.bits()) {
            value = builder.create<mlir::arith::ExtFOp>(mlir_type, value);
        } else {
            value = builder.create<mlir::arith::TruncFOp>(mlir_type, value);
        }
    } else {
        internal_error << "Cast of " << src << " to " << dst << " is not implemented.\n";
    }
}

void CodeGen_MLIR::Visitor::visit(const Reinterpret *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<mlir::arith::BitcastOp>(mlir_type_of(op->type), codegen(op->value));
}

void CodeGen_MLIR::Visitor::visit(const Variable *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
    debug(3) << "\tname: " << op->name << "\n";
    value = sym_get(op->name, true);
}

void CodeGen_MLIR::Visitor::visit(const Add *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    if (op->type.is_int_or_uint())
        value = builder.create<mlir::arith::AddIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_float())
        value = builder.create<mlir::arith::AddFOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR::Visitor::visit(const Sub *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    if (op->type.is_int_or_uint())
        value = builder.create<mlir::arith::SubIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_float())
        value = builder.create<mlir::arith::SubFOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR::Visitor::visit(const Mul *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    int bits;
    if (is_const_power_of_two_integer(op->b, &bits))
        value = codegen(op->a << make_const(op->a.type(), bits));
    else if (op->type.is_int_or_uint())
        value = builder.create<mlir::arith::MulIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_float())
        value = builder.create<mlir::arith::MulFOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR::Visitor::visit(const Div *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    int bits;
    if (is_const_power_of_two_integer(op->b, &bits))
        value = codegen(op->a >> make_const(op->a.type(), bits));
    else if (op->type.is_int())
        value = builder.create<mlir::arith::DivSIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_uint())
        value = builder.create<mlir::arith::DivUIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_float())
        value = builder.create<mlir::arith::DivFOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR::Visitor::visit(const Mod *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    int bits;
    if (is_const_power_of_two_integer(op->b, &bits))
        value = codegen(op->a & make_const(op->a.type(), (1 << bits) - 1));
    else if (op->type.is_int())
        value = builder.create<mlir::arith::RemSIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_uint())
        value = builder.create<mlir::arith::RemUIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_float())
        value = builder.create<mlir::arith::RemFOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR::Visitor::visit(const Min *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    if (op->type.is_int())
        value = builder.create<mlir::arith::MinSIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_uint())
        value = builder.create<mlir::arith::MinUIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_float())
        value = builder.create<mlir::arith::MinFOp>(codegen(op->a), codegen(op->b));
}

void CodeGen_MLIR::Visitor::visit(const Max *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    if (op->type.is_int())
        value = builder.create<mlir::arith::MaxSIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_uint())
        value = builder.create<mlir::arith::MaxUIOp>(codegen(op->a), codegen(op->b));
    else if (op->type.is_float())
        value = builder.create<mlir::arith::MaxFOp>(codegen(op->a), codegen(op->b));
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

    value = builder.create<mlir::arith::AndIOp>(codegen(NE::make(op->a, make_zero(op->a.type()))),
                                                codegen(NE::make(op->b, make_zero(op->b.type()))));
}

void CodeGen_MLIR::Visitor::visit(const Or *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<mlir::arith::OrIOp>(codegen(NE::make(op->a, make_zero(op->a.type()))),
                                               codegen(NE::make(op->b, make_zero(op->b.type()))));
}

void CodeGen_MLIR::Visitor::visit(const Not *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    value = codegen(EQ::make(op->a, make_zero(op->a.type())));
}

void CodeGen_MLIR::Visitor::visit(const Select *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    value = builder.create<mlir::arith::SelectOp>(codegen(op->condition),
                                                  codegen(op->true_value),
                                                  codegen(op->false_value));
}

void CodeGen_MLIR::Visitor::visit(const Load *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
    debug(3) << "\tName: " << op->name << "\n";

    Expr index;
    if (op->type.is_scalar()) {
        index = op->index;
    } else if (Expr ramp_base = strided_ramp_base(op->index); ramp_base.defined()) {
        index = ramp_base;
    } else {
        user_error << "Unsupported load.";
    }

    mlir::Value baseIndex = sym_get(op->name);
    mlir::Value indexI64 = builder.create<mlir::arith::ExtUIOp>(builder.getI64Type(), codegen(index));
    mlir::Value address = builder.create<mlir::arith::AddIOp>(baseIndex, indexI64);
    mlir::Value addressAsIndex = builder.create<mlir::arith::IndexCastOp>(builder.getIndexType(), address);

    mlir::Value buffer = sym_get(op->name + ".buffer");

    if (op->type.is_scalar()) {
        value = builder.create<mlir::memref::LoadOp>(buffer, mlir::ValueRange{addressAsIndex});
    } else {
        value = builder.create<mlir::vector::LoadOp>(mlir_type_of(op->type), buffer, mlir::ValueRange{addressAsIndex});
    }
}

void CodeGen_MLIR::Visitor::visit(const Ramp *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";

    mlir::Value base = codegen(op->base);
    mlir::Value stride = codegen(op->stride);
    mlir::Type elementType = mlir_type_of(op->base.type());
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

    value = builder.create<mlir::vector::SplatOp>(mlir_type_of(op->type), codegen(op->value));
}

void CodeGen_MLIR::Visitor::visit(const Call *op) {
    debug(2) << __PRETTY_FUNCTION__ << "\n";
    debug(3) << "\tName: " << op->name << "\n";
    debug(3) << "\tCall type: " << op->call_type << "\n";
    for (const Expr &e : op->args)
        debug(3) << "\tArg: " << e << "\n";

    if (op->is_intrinsic(Call::bitwise_and)) {
        value = builder.create<mlir::arith::AndIOp>(codegen(op->args[0]), codegen(op->args[1]));
    } else if (op->is_intrinsic(Call::shift_left)) {
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

    Expr index;
    if (op->value.type().is_scalar()) {
        index = op->index;
    } else if (Expr ramp_base = strided_ramp_base(op->index); ramp_base.defined()) {
        index = ramp_base;
    } else {
        user_error << "Unsupported store.";
    }

    mlir::Value baseIndex = sym_get(op->name);
    mlir::Value indexI64 = builder.create<mlir::arith::ExtUIOp>(builder.getI64Type(), codegen(index));
    mlir::Value address = builder.create<mlir::arith::AddIOp>(baseIndex, indexI64);
    mlir::Value addressAsIndex = builder.create<mlir::arith::IndexCastOp>(builder.getIndexType(), address);

    mlir::Value value = codegen(op->value);
    mlir::Value buffer = sym_get(op->name + ".buffer");

    if (op->value.type().is_scalar())
        builder.create<mlir::memref::StoreOp>(value, buffer, mlir::ValueRange{addressAsIndex});
    else
        builder.create<mlir::vector::StoreOp>(value, buffer, mlir::ValueRange{addressAsIndex});
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
    for (auto &ext : op->extents)
        debug(3) << "  ext: " << ext << "\n";

    internal_assert(op->type.is_scalar()) << "Local memories with vector types not supported.";

    mlir::MemRefType type = mlir::MemRefType::get({size}, mlir_type_of(op->type));
    mlir::memref::AllocOp alloc = builder.create<mlir::memref::AllocOp>(type);
    mlir::Value constantZero = builder.create<mlir::arith::ConstantOp>(builder.getI64IntegerAttr(0));

    sym_push(op->name, constantZero);
    sym_push(op->name + ".buffer", alloc);
    codegen(op->body);
    sym_pop(op->name + ".buffer");
    sym_pop(op->name);
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

    internal_assert(!op->else_case.defined()) << "Else case not supported yet.";

    codegen(For::make("if_then_branch", 0, Cast::make(Int(32), op->condition),
                      ForType::Serial, DeviceAPI::None, op->then_case));
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

mlir::Type CodeGen_MLIR::Visitor::mlir_type_of(Halide::Type t) const {
    return CodeGen_MLIR::mlir_type_of(builder, t);
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
