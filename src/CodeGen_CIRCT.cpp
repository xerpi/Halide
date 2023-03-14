#include <fstream>
#include <vector>

#include <circt/Conversion/SCFToCalyx.h>
#include <circt/Dialect/Calyx/CalyxDialect.h>
#include <circt/Dialect/Calyx/CalyxOps.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include "CodeGen_CIRCT.h"
#include "CodeGen_Internal.h"
#include "Debug.h"
#include "IROperator.h"
#include "Util.h"

namespace Halide {

namespace Internal {


CodeGen_CIRCT::CodeGen_CIRCT() {
    mlir_context.loadDialect<circt::calyx::CalyxDialect>();
    mlir_context.loadDialect<mlir::arith::ArithDialect>();
    mlir_context.loadDialect<mlir::func::FuncDialect>();
}

void CodeGen_CIRCT::compile(const Module &input) {
    debug(1) << "Generating CIRCT MLIR IR...\n";

    mlir::LocationAttr loc = mlir::UnknownLoc::get(&mlir_context);
    mlir::ModuleOp mlir_module = mlir::ModuleOp::create(loc, {});
    mlir::ImplicitLocOpBuilder builder = mlir::ImplicitLocOpBuilder::atBlockEnd(loc, mlir_module.getBody());

    // Translate each function into a Calyx component
    for (const auto &function : input.functions()) {
        std::cout << "Generating CIRCT MLIR IR for function " << function.name << std::endl;

        // Inputs: {enable, finished}
        mlir::SmallVector<mlir::Type> inputs{builder.getI1Type(), builder.getI1Type()};
        // Outputs: {valid, done}
        mlir::SmallVector<mlir::Type> results{};
        mlir::FunctionType functionType = builder.getFunctionType(inputs, results);

        mlir::func::FuncOp functionOp = builder.create<mlir::func::FuncOp>(builder.getStringAttr(function.name), functionType);
        builder.setInsertionPointToStart(functionOp.addEntryBlock());

        CodeGen_CIRCT::Visitor visitor(builder, function);
        function.body.accept(&visitor);

        builder.create<mlir::func::ReturnOp>();
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
    pm.addPass(circt::createSCFToCalyxPass());

    auto pmRunResult = pm.run(mlir_module);
    std::cout << "Run passes result: " << pmRunResult.succeeded() << std::endl;

    // Print MLIR after running passes
    std::cout << "MLIR after running passes" << std::endl;
    mlir_module.dump();

    // Verify module (after running passes)
    moduleVerifyResult = mlir::verify(mlir_module);
    std::cout << "Module verify (after passes) result: " << moduleVerifyResult.succeeded() << std::endl;
    internal_assert(moduleVerifyResult.succeeded());

    std::cout << "Done!" << std::endl;
}

CodeGen_CIRCT::Visitor::Visitor(mlir::ImplicitLocOpBuilder &builder, const Internal::LoweredFunc &function) : builder(builder) {

    // Generate module ports (inputs and outputs)
    mlir::SmallVector<circt::calyx::PortInfo> ports;

    // Clock and reset signals
    ports.push_back({
        builder.getStringAttr("clk"),
        builder.getI1Type(),
        circt::calyx::Direction::Input,
        builder.getDictionaryAttr({builder.getNamedAttr(builder.getStringAttr("clk"),
                                                        builder.getUnitAttr())})
    });
    ports.push_back({
        builder.getStringAttr("done"),
        builder.getI1Type(),
        circt::calyx::Direction::Output,
        builder.getDictionaryAttr({builder.getNamedAttr(builder.getStringAttr("done"),
                                                        builder.getUnitAttr())})
    });
    ports.push_back({
        builder.getStringAttr("go"),
        builder.getI1Type(),
        circt::calyx::Direction::Input,
        builder.getDictionaryAttr({builder.getNamedAttr(builder.getStringAttr("go"),
                                                        builder.getUnitAttr())})
    });
    ports.push_back({
        builder.getStringAttr("reset"),
        builder.getI1Type(),
        circt::calyx::Direction::Input,
        builder.getDictionaryAttr({builder.getNamedAttr(builder.getStringAttr("reset"),
                                                        builder.getUnitAttr())})
    });

    // Convert function arguments to module ports
    std::vector<LoweredArgument> bufferArgs;
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

        switch (arg.kind) {
        case Argument::Kind::InputScalar:
            switch (arg.type.code()) {
            case Type::Int:
            case Type::UInt:
                ports.push_back({
                    builder.getStringAttr(arg.name),
                    builder.getIntegerType(arg.type.bits(), arg.type.is_int()),
                    circt::calyx::Direction::Input,
                    builder.getDictionaryAttr({builder.getNamedAttr(builder.getStringAttr(arg.name),
                                                                    builder.getUnitAttr())})
                });
                break;
            case Type::Float:
            case Type::BFloat:
                assert(0 && "TODO");
                break;
            case Type::Handle:
                // TODO: Ignore for now
                break;
            }
            break;
        case Argument::Kind::InputBuffer:
        case Argument::Kind::OutputBuffer:
            bufferArgs.push_back(arg);
            break;
        }
    }

#if 0
    auto component = builder.create<circt::calyx::ComponentOp>(builder.getStringAttr(function.name), ports);
    builder.setInsertionPointToStart(component.getBodyBlock());

    // Instantiate external memories (buffer arguments). Consider all buffers as flat ([x + y * stride])
    for (const auto &buf: bufferArgs) {
        mlir::SmallVector<int64_t> addrSizes{64};
        mlir::SmallVector<int64_t> sizes{32};
        auto memoryOp = builder.create<circt::calyx::MemoryOp>(buf.name, buf.type.bits(), addrSizes, sizes);
        memoryOp->setAttr("external", builder.getBoolAttr(true));
    }

    auto wires = component.getWiresOp();
    mlir::Block *wiresBody = wires.getBodyBlock();

    auto wiresBuilder = mlir::ImplicitLocOpBuilder::atBlockEnd(wires.getLoc(), wiresBody);

    wiresBuilder.create<circt::calyx::AssignOp>(component.getDonePort(),
                                                wiresBuilder.create<mlir::arith::ConstantOp>(wiresBuilder.getBoolAttr(true)));
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
}

void CodeGen_CIRCT::Visitor::visit(const NE *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const LT *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const LE *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const GT *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const GE *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const And *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
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
}

void CodeGen_CIRCT::Visitor::visit(const Load *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    debug(1) << "\tname: " << op->name << "\n";
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

    value = builder.create<mlir::arith::ConstantOp>(op_type, builder.getIntegerAttr(op_type, 0));

#if 0
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
#endif
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
}

void CodeGen_CIRCT::Visitor::visit(const Store *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    debug(1) << "\tName: " << op->name << "\n";
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
