#include <fstream>
#include <vector>

#include <circt/Conversion/SCFToCalyx.h>
#include <circt/Dialect/Calyx/CalyxDialect.h>
#include <circt/Dialect/Calyx/CalyxEmitter.h>
#include <circt/Dialect/Calyx/CalyxOps.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Passes.h>
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
    mlir_context.loadDialect<mlir::memref::MemRefDialect>();
    mlir_context.loadDialect<mlir::scf::SCFDialect>();
}

void CodeGen_CIRCT::compile(const Module &input) {
    debug(1) << "Generating CIRCT MLIR IR...\n";

    mlir::LocationAttr loc = mlir::UnknownLoc::get(&mlir_context);
    mlir::ModuleOp mlir_module = mlir::ModuleOp::create(loc, {});
    mlir::ImplicitLocOpBuilder builder = mlir::ImplicitLocOpBuilder::atBlockEnd(loc, mlir_module.getBody());

    // Translate each function into a Calyx component
    std::vector<std::string> inputNames;
    mlir::SmallVector<Internal::LoweredFunc> bufferArguments;
    for (const auto &function : input.functions()) {
        std::cout << "Generating CIRCT MLIR IR for function " << function.name << std::endl;

        mlir::SmallVector<mlir::Type> inputs;
        mlir::SmallVector<mlir::Type> results;

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
                    inputs.push_back(builder.getIntegerType(arg.type.bits(), arg.type.is_int()));
                    inputNames.push_back(arg.name);
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
                struct BufferDescriptorEntry {
                    std::string suffix;
                    mlir::Type type;
                };

                std::vector<BufferDescriptorEntry> entries;
                mlir::Type i32 = builder.getI32Type();
                mlir::Type i64 = builder.getI64Type();

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
                    inputNames.push_back(arg.name + "_" + entry.suffix);
                }

                break;
                // Treat buffers as 1D
                //mlir::SmallVector<int64_t> shape(arg.dimensions, mlir::ShapedType::kDynamic);
                /*mlir::SmallVector<int64_t> shape{mlir::ShapedType::kDynamic};
                mlir::Type elementType = builder.getIntegerType(arg.type.bits());
                inputs.push_back(mlir::MemRefType::get(shape, elementType));
                inputNames.push_back(arg.name + ".buffer");*/
                //bufferArguments.push_back(arg);
                //break;
            }
        }

        mlir::FunctionType functionType = builder.getFunctionType(inputs, results);
        mlir::func::FuncOp functionOp = builder.create<mlir::func::FuncOp>(builder.getStringAttr(function.name), functionType);
        builder.setInsertionPointToStart(functionOp.addEntryBlock());

        CodeGen_CIRCT::Visitor visitor(builder, function, inputNames);
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
    pm.addPass(mlir::createForToWhileLoopPass());
    pm.addPass(circt::createSCFToCalyxPass());
    //pm.addPass(circt::createCalyxToHWPass());

    auto pmRunResult = pm.run(mlir_module);
    std::cout << "Run passes result: " << pmRunResult.succeeded() << std::endl;

    // Print MLIR after running passes
    std::cout << "MLIR after running passes" << std::endl;
    mlir_module.dump();

    // Verify module (after running passes)
    moduleVerifyResult = mlir::verify(mlir_module);
    std::cout << "Module verify (after passes) result: " << moduleVerifyResult.succeeded() << std::endl;
    internal_assert(moduleVerifyResult.succeeded());

    // Emit Calyx
    std::string str;
    llvm::raw_string_ostream os(str);
    std::cout << "Exporting Calyx." << std::endl;
    auto exportVerilogResult = circt::calyx::exportCalyx(mlir_module, os);
    std::cout << "Export Calyx result: " << exportVerilogResult.succeeded() << std::endl;
    std::cout << str << std::endl;

    std::cout << "Done!" << std::endl;
}

CodeGen_CIRCT::Visitor::Visitor(mlir::ImplicitLocOpBuilder &builder, const Internal::LoweredFunc &function, const std::vector<std::string> &inputNames)
    : builder(builder) {

    mlir::func::FuncOp funcOp = cast<mlir::func::FuncOp>(builder.getBlock()->getParentOp());

    // Add function arguments to the symbol table
    for (unsigned int i = 0; i < funcOp.getFunctionType().getNumInputs(); i++) {
        std::string name = inputNames[i];
        sym_push(name, funcOp.getArgument(i));
    }

    // Instantiate buffers (external memories by default) and add them to the symbol table
    for (const auto &arg: function.args) {
        if (arg.is_buffer()) {
            mlir::SmallVector<int64_t> shape{mlir::ShapedType::kDynamic};
            mlir::Type elementType = builder.getIntegerType(arg.type.bits());
            mlir::MemRefType memRefType = mlir::MemRefType::get(shape, elementType);
            mlir::Value dynamicSize = builder.create<mlir::arith::ConstantOp>(builder.getI64IntegerAttr(1));
            for (int i = 0; i < arg.dimensions; i++) {
                mlir::Value stride = sym_get(arg.name + "_dim_" + std::to_string(i) + "_stride");
                mlir::Value stride64 = builder.create<mlir::arith::ExtUIOp>(builder.getI64Type(), stride);
                dynamicSize = builder.create<mlir::arith::MulIOp>(dynamicSize, stride64);
            }
            dynamicSize = builder.create<mlir::arith::IndexCastOp>(builder.getIndexType(), dynamicSize);
            sym_push(arg.name + ".buffer", builder.create<mlir::memref::AllocOp>(memRefType, mlir::ValueRange{dynamicSize}));
        }
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

    mlir::Value value = codegen(op->value);
    mlir::Value index = builder.create<mlir::arith::IndexCastOp>(builder.getIndexType(), codegen(op->index));
    mlir::Value buf = sym_get(op->name + ".buffer");

    builder.create<mlir::memref::StoreOp>(value, buf, mlir::ValueRange{index});
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
