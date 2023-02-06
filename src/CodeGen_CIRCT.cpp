#include <vector>

#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/HW/HWOps.h>
#include <circt/Dialect/HWArith/HWArithDialect.h>
#include <circt/Dialect/HWArith/HWArithOps.h>
#include <circt/Dialect/SV/SVDialect.h>

#include <mlir/IR/Verifier.h>

#include "CodeGen_CIRCT.h"
#include "Debug.h"
#include "IROperator.h"
#include "Util.h"

namespace Halide {

namespace Internal {


CodeGen_CIRCT::CodeGen_CIRCT() {
    mlir_context.loadDialect<circt::comb::CombDialect>();
    mlir_context.loadDialect<circt::hw::HWDialect>();
    mlir_context.loadDialect<circt::hwarith::HWArithDialect>();
    mlir_context.loadDialect<circt::sv::SVDialect>();
    create_halide_circt_types();
}

void CodeGen_CIRCT::compile(const Module &input) {
    //init_codegen(input.name(), input.any_strict_float());

    // Generate the code for this module.
    debug(1) << "Generating CIRCT MLIR IR...\n";
    //for (const auto &b : input.buffers()) {
        //compile_buffer(b);
    //}

    //vector<MangledNames> function_names;

    // Declare all functions
#if 0
    for (const auto &f : input.functions()) {
        const auto names = get_mangled_names(f, get_target());
        function_names.push_back(names);

        // Deduce the types of the arguments to our function
        vector<llvm::Type *> arg_types(f.args.size());
        for (size_t i = 0; i < f.args.size(); i++) {
            if (f.args[i].is_buffer()) {
                arg_types[i] = halide_buffer_t_type->getPointerTo();
            } else {
                arg_types[i] = llvm_type_of(upgrade_type_for_argument_passing(f.args[i].type));
            }
        }
        FunctionType *func_t = FunctionType::get(i32_t, arg_types, false);
        function = llvm::Function::Create(func_t, llvm_linkage(f.linkage), names.extern_name, module.get());
        set_function_attributes_from_halide_target_options(*function);

        // Mark the buffer args as no alias and save indication for add_argv_wrapper if needed
        std::vector<bool> buffer_args(f.args.size());
        for (size_t i = 0; i < f.args.size(); i++) {
            bool is_buffer = f.args[i].is_buffer();
            buffer_args[i] = is_buffer;
            if (is_buffer) {
                function->addParamAttr(i, Attribute::NoAlias);
            }
        }

        // sym_push helpfully calls setName, which we don't want
        symbol_table.push("::" + f.name, function);

        // If the Func is externally visible, also create the argv wrapper and metadata.
        // (useful for calling from JIT and other machine interfaces).
        if (f.linkage == LinkageType::ExternalPlusArgv || f.linkage == LinkageType::ExternalPlusMetadata) {
            add_argv_wrapper(function, names.argv_name, false, buffer_args);
            if (f.linkage == LinkageType::ExternalPlusMetadata) {
                embed_metadata_getter(names.metadata_name,
                                      names.simple_name, f.args, input.get_metadata_name_map());
            }
        }
    }
#endif

    // Define all functions
    // int idx = 0;
    for (const auto &f : input.functions()) {
        //const auto names = function_names[idx++];

        /*run_with_large_stack([&]() {
            compile_func(f, names.simple_name, names.extern_name);
        });*/

        // Generate the function body.
        debug(1) << "Generating CIRCT MLIR IR for function " << f.name << "\n";
        debug(1) << "\tArg count: " << f.args.size() << "\n";
        for (const auto &arg: f.args) {
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
        }

        mlir::LocationAttr loc = mlir::UnknownLoc::get(&mlir_context);
        mlir::ModuleOp mlir_module = mlir::ModuleOp::create(loc, {});
        mlir:: ImplicitLocOpBuilder builder = mlir::ImplicitLocOpBuilder::atBlockEnd(loc, mlir_module.getBody());
        CodeGen_CIRCT::Visitor visitor(builder);
        f.body.accept(&visitor);
        // Print MLIR before running passes
        std::cout << "Generated MLIR for function " << f.name << ":" << std::endl;
        mlir_module.dump();

        // Verify module
        auto moduleVerifyResult = mlir::verify(mlir_module);
        std::cout << "Module verify result: " << moduleVerifyResult.succeeded() << std::endl;

    }

    //debug(2) << "llvm::Module pointer: " << module.get() << "\n";
}

void CodeGen_CIRCT::create_halide_circt_types() {
    //if (std::is_same<decltype(halide_buffer_t::)::element_type, int>::value) {

    //}
}

mlir::Value CodeGen_CIRCT::Visitor::codegen(const Expr &e) {
    internal_assert(e.defined());
    debug(4) << "Codegen (E): " << e.type() << ", " << e << "\n";
    value = mlir::Value();
    e.accept(this);
    //internal_assert(value) << "Codegen of an expr did not produce a MLIR value\n"
    //                       << e;
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
    value = builder.create<circt::hwarith::ConstantOp>(builder.getIntegerType(op->type.bits(), true),
                                                       builder.getI64IntegerAttr(op->value));
}

void CodeGen_CIRCT::Visitor::visit(const UIntImm *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    value = builder.create<circt::hwarith::ConstantOp>(builder.getIntegerType(op->type.bits(), false),
                                                       builder.getI64IntegerAttr(op->value));
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

    mlir::Value a = codegen(op->a);
    mlir::Value b = codegen(op->b);

    value = builder.create<circt::hwarith::AddOp>(mlir::ValueRange({a, b}));

    debug(1) << "Add value bits: " << value.getType().getIntOrFloatBitWidth() << ", op bits: " << op->type.bits() << "\n";
    if (value.getType().getIntOrFloatBitWidth() != unsigned(op->type.bits())) {
        mlir::Type new_type = builder.getIntegerType(op->type.bits(), value.getType().isSignedInteger());
        value = builder.create<circt::hwarith::CastOp>(new_type, value);
    }
}

void CodeGen_CIRCT::Visitor::visit(const Sub *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    mlir::Value a = codegen(op->a);
    mlir::Value b = codegen(op->b);

    value = builder.create<circt::hwarith::SubOp>(mlir::ValueRange({a, b}));

    debug(1) << "Sub value bits: " << value.getType().getIntOrFloatBitWidth() << ", op bits: " << op->type.bits() << "\n";
    if (value.getType().getIntOrFloatBitWidth() != unsigned(op->type.bits())) {
        mlir::Type new_type = builder.getIntegerType(op->type.bits(), value.getType().isSignedInteger());
        value = builder.create<circt::hwarith::CastOp>(new_type, value);
    }
}

void CodeGen_CIRCT::Visitor::visit(const Mul *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    mlir::Value a = codegen(op->a);
    mlir::Value b = codegen(op->b);

    value = builder.create<circt::hwarith::MulOp>(mlir::ValueRange({a, b}));

    debug(1) << "Mul value bits: " << value.getType().getIntOrFloatBitWidth() << ", op bits: " << op->type.bits() << "\n";
    if (value.getType().getIntOrFloatBitWidth() != unsigned(op->type.bits())) {
        mlir::Type new_type = builder.getIntegerType(op->type.bits(), value.getType().isSignedInteger());
        value = builder.create<circt::hwarith::CastOp>(new_type, value);
    }
}

void CodeGen_CIRCT::Visitor::visit(const Div *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const Mod *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const Min *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const Max *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
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
    // Implement Logic NOT as XOR with all ones, and then compare with zeroes
    mlir::Value a = codegen(op->a);
    int bits = op->a.type().bits();
    circt::hw::ConstantOp allzeroes = builder.create<circt::hw::ConstantOp>(builder.getIntegerType(bits), 0);
    if (bits == 1) {
        value = builder.create<circt::comb::ICmpOp>(circt::comb::ICmpPredicate::eq, a, allzeroes);
    } else { // bits > 1
        circt::hw::ConstantOp all_minus1_zeroes = builder.create<circt::hw::ConstantOp>(builder.getIntegerType(bits - 1), 0);
        mlir::Value cmpOp = builder.create<circt::comb::ICmpOp>(circt::comb::ICmpPredicate::eq, a, allzeroes);
        value = builder.create<circt::comb::ConcatOp>(all_minus1_zeroes, cmpOp);
    }
}

void CodeGen_CIRCT::Visitor::visit(const Select *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const Load *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
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

    if (op->name == Call::buffer_get_host) {
        auto name = op->args[0].as<Variable>()->name;
        name = name.substr(0, name.find(".buffer"));
        debug(1) << "\t\targ name: " << name << "\n";
        debug(1) << "\t\top bits: " << op->type.bits() << "\n";
        // Return true
        value = builder.create<circt::hw::ConstantOp>(builder.getIntegerType(op->type.bits()), 1);
        //mlir::Location loc = mlir::NameLoc::get(mlir::StringAttr::get(builder.getContext(), "foo"));
        //value.setLoc(loc);
    } else if (op->name == Call::buffer_is_bounds_query) {
        auto name = op->args[0].as<Variable>()->name;
        name = name.substr(0, name.find(".buffer"));
        debug(1) << "\t\targ name: " << name << "\n";
        debug(1) << "\t\top bits: " << op->type.bits() << "\n";
        // Return true
        value = builder.create<circt::hw::ConstantOp>(builder.getIntegerType(op->type.bits()), 1);
    } else if(op->name == Call::buffer_get_min) {
        auto name = op->args[0].as<Variable>()->name;
        name = name.substr(0, name.find(".buffer"));
        debug(1) << "\t\targ name: " << name << "\n";
        debug(1) << "\t\top bits: " << op->type.bits() << "\n";
        // TODO: buf->dim[d].min
        value = builder.create<circt::hwarith::ConstantOp>(builder.getIntegerType(op->type.bits(), true),
                                                           builder.getSI32IntegerAttr(41));
    } else if(op->name == Call::buffer_get_extent) {
        auto name = op->args[0].as<Variable>()->name;
        name = name.substr(0, name.find(".buffer"));
        debug(1) << "\t\targ name: " << name << "\n";
        debug(1) << "\t\top bits: " << op->type.bits() << "\n";
        // TODO: buf->dim[d].extent
        value = builder.create<circt::hwarith::ConstantOp>(builder.getIntegerType(op->type.bits(), true),
                                                           builder.getSI32IntegerAttr(42));
    } else if(op->name == Call::buffer_get_stride) {
        auto name = op->args[0].as<Variable>()->name;
        name = name.substr(0, name.find(".buffer"));
        debug(1) << "\t\targ name: " << name << "\n";
        debug(1) << "\t\top bits: " << op->type.bits() << "\n";
        // TODO: buf->dim[d].stride
        value = builder.create<circt::hwarith::ConstantOp>(builder.getIntegerType(op->type.bits(), true),
                                                           builder.getSI32IntegerAttr(43));
    } else {
        // Just return 1 for now
        value = builder.create<circt::hw::ConstantOp>(builder.getIntegerType(op->type.bits()), 1);
    }
}

void CodeGen_CIRCT::Visitor::visit(const Let *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const LetStmt *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    debug(1) << "Contents:" << "\n";
    debug(1) << "\tName: " << op->name << "\n";
    debug(1) << "\tValue: " << op->value << "\n";
    debug(1) << "\tBody: " << op->body << "\n";
    sym_push(op->name, codegen(op->value));
    codegen(op->body);
    sym_pop(op->name);
    debug(1) << __PRETTY_FUNCTION__ << " finished!\n";
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

    sym_push(op->name, min);
    codegen(op->body);
    sym_pop(op->name);
}

void CodeGen_CIRCT::Visitor::visit(const Store *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    debug(1) << "\tName: " << op->name << "\n";

    mlir::Value predicate = codegen(op->predicate);
    mlir::Value value = codegen(op->value);
    mlir::Value index = codegen(op->index);

    // if (predicate) buffer[op->name].store(index, value);
}

void CodeGen_CIRCT::Visitor::visit(const Provide *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
}

void CodeGen_CIRCT::Visitor::visit(const Allocate *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
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
    //value = nullptr;
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
    //mlir::NameLoc::get(StringAttr::get(unwrap(context), unwrap(name))));
    //value.setLoc(mlir::NameLoc::get(builder.getStringAttr(name)));
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
