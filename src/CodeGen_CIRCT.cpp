#include <vector>

#include <circt/Conversion/ExportVerilog.h>
#include <circt/Conversion/HWArithToHW.h>
#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/HW/HWOps.h>
#include <circt/Dialect/HWArith/HWArithDialect.h>
#include <circt/Dialect/HWArith/HWArithOps.h>
#include <circt/Dialect/Seq/SeqDialect.h>
#include <circt/Dialect/Seq/SeqOps.h>
#include <circt/Dialect/Seq/SeqPasses.h>
#include <circt/Dialect/SV/SVDialect.h>
#include <circt/Dialect/SV/SVOps.h>

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
    mlir_context.loadDialect<circt::hw::HWDialect>();
    mlir_context.loadDialect<circt::hwarith::HWArithDialect>();
    mlir_context.loadDialect<circt::seq::SeqDialect>();
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

    // Translate each function into a CIRCT HWModuleOp
    for (const auto &f : input.functions()) {
        /*run_with_large_stack([&]() {
            compile_func(f, names.simple_name, names.extern_name);
        });*/

        std::cout << "Generating CIRCT MLIR IR for function " << f.name << std::endl;

        mlir::LocationAttr loc = mlir::UnknownLoc::get(&mlir_context);
        mlir::ModuleOp mlir_module = mlir::ModuleOp::create(loc, {});
        mlir::ImplicitLocOpBuilder builder = mlir::ImplicitLocOpBuilder::atBlockEnd(loc, mlir_module.getBody());

        mlir::SmallVector<circt::hw::PortInfo> ports;

        // Convert function arguments to module ports (inputs and outputs)
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

            mlir::Type type;

            switch (arg.kind) {
            case Argument::Kind::InputScalar:
                switch (arg.type.code()) {
                case Type::Int:
                case Type::UInt:
                    type = builder.getIntegerType(arg.type.bits(), arg.type.is_int());
                    break;
                case Type::Float:
                case Type::BFloat:
                    assert(0 && "TODO");
                case Type::Handle:
                    // TODO
                    type = builder.getIntegerType(64, false);
                    break;
                }
                break;
            case Argument::Kind::InputBuffer:
            case Argument::Kind::OutputBuffer:
                // Buffer descriptor
                type = builder.getIntegerType(32, false);
                break;
            }

            ports.push_back(circt::hw::PortInfo{builder.getStringAttr(arg.name), circt::hw::PortDirection::INPUT, type, 0});
        }

        // Create module top
        circt::hw::HWModuleOp top = builder.create<circt::hw::HWModuleOp>(builder.getStringAttr(f.name), ports);
        builder.setInsertionPointToStart(top.getBodyBlock());

        // Generate CIRCT MLIR IR
        CodeGen_CIRCT::Visitor visitor(builder, top);
        f.body.accept(&visitor);

        // Module output
        //circt::hw::ConstantOp c0 = builder.create<circt::hw::ConstantOp>(builder.getIntegerType(32, true), 42);
        //auto outputOp = top.getBodyBlock()->getTerminator();
        //outputOp->setOperands(mlir::ValueRange{c0});

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
        pm.addPass(circt::seq::createSeqLowerToSVPass());
#if 0
        pm.addPass(circt::createSimpleCanonicalizerPass());
        pm.nest<circt::hw::HWModuleOp>().addPass(circt::circt::seq::createLowerSeqHLMemPass());
        pm.nest<circt::hw::HWModuleOp>().addPass(circt::seq::createSeqFIRRTLLowerToSVPass());
        pm.addPass(circt::sv::createHWMemSimImplPass(false, false));
        pm.addPass(circt::seq::createSeqLowerToSVPass());
        pm.nest<circt::hw::HWModuleOp>().addPass(circt::sv::createHWCleanupPass());

        // Legalize unsupported operations within the modules.
        pm.nest<circt::hw::HWModuleOp>().addPass(sv::createHWLegalizeModulesPass());
        pm.addPass(circt::createSimpleCanonicalizerPass());

        // Tidy up the IR to improve verilog emission quality.
        auto &modulePM = pm.nestcirct::<hw::HWModuleOp>();
        modulePM.addPass(circt::sv::createPrettifyVerilogPass());
#endif

        auto pmRunResult = pm.run(mlir_module);

        std::cout << "Run passes result: " << pmRunResult.succeeded() << std::endl;
        std::cout << "Module inputs: " << top.getNumInputs() << ", outputs: " << top.getNumOutputs() << std::endl;

        // Print MLIR after running passes
        std::cout << "MLIR after running passes" << std::endl;
        mlir_module.dump();

        // Verify module (after running passes)
        moduleVerifyResult = mlir::verify(mlir_module);
        std::cout << "Module verify (after passes) result: " << moduleVerifyResult.succeeded() << std::endl;
        internal_assert(moduleVerifyResult.succeeded());

        // Exmit Verilog
        std::string str;
        llvm::raw_string_ostream os(str);
        std::cout << "Exporting Verilog." << std::endl;
        auto exportVerilogResult = circt::exportVerilog(mlir_module, os);
        std::cout << "Export Verilog result: " << exportVerilogResult.succeeded() << std::endl;
        std::cout << str << std::endl;

        std::cout << "Done!" << std::endl;

    }

    //debug(2) << "llvm::Module pointer: " << module.get() << "\n";
}

void CodeGen_CIRCT::create_halide_circt_types() {
    //if (std::is_same<decltype(halide_buffer_t::)::element_type, int>::value) {

    //}
}

CodeGen_CIRCT::Visitor::Visitor(mlir::ImplicitLocOpBuilder &builder, circt::hw::HWModuleOp &top)
    : builder(builder) {

    // Add function arguments to the symbol table
    for (unsigned i = 0; i < top.getNumArguments(); i++) {
        std::string name = top.getArgNames()[i].cast<mlir::StringAttr>().str();
        sym_push(name, top.getArgument(i));
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
    mlir::Type type = builder.getIntegerType(op->type.bits(), op->type.is_int());
    value = builder.create<circt::hwarith::ConstantOp>(type, builder.getIntegerAttr(type, op->value));
}

void CodeGen_CIRCT::Visitor::visit(const UIntImm *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    mlir::Type type = builder.getIntegerType(op->type.bits(), op->type.is_int());
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

    mlir::Value a = codegen(op->a);
    mlir::Value b = codegen(op->b);
    circt::hwarith::ConstantOp allzeroes_a = builder.create<circt::hwarith::ConstantOp>(a.getType(), builder.getIntegerAttr(a.getType(), 0));
    circt::hwarith::ConstantOp allzeroes_b = builder.create<circt::hwarith::ConstantOp>(b.getType(), builder.getIntegerAttr(b.getType(), 0));
    mlir::Value isnotzero_a = builder.create<circt::hwarith::ICmpOp>(circt::hwarith::ICmpPredicate::ne, a, allzeroes_a);
    mlir::Value isnotzero_b = builder.create<circt::hwarith::ICmpOp>(circt::hwarith::ICmpPredicate::ne, b, allzeroes_b);
    value = to_unsigned(builder.create<circt::comb::AndOp>(to_signless(isnotzero_a), to_signless(isnotzero_b)));
}

void CodeGen_CIRCT::Visitor::visit(const Or *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";

    mlir::Value a = codegen(op->a);
    mlir::Value b = codegen(op->b);
    circt::hwarith::ConstantOp allzeroes_a = builder.create<circt::hwarith::ConstantOp>(a.getType(), builder.getIntegerAttr(a.getType(), 0));
    circt::hwarith::ConstantOp allzeroes_b = builder.create<circt::hwarith::ConstantOp>(b.getType(), builder.getIntegerAttr(b.getType(), 0));
    mlir::Value isnotzero_a = builder.create<circt::hwarith::ICmpOp>(circt::hwarith::ICmpPredicate::ne, a, allzeroes_a);
    mlir::Value isnotzero_b = builder.create<circt::hwarith::ICmpOp>(circt::hwarith::ICmpPredicate::ne, b, allzeroes_b);
    value = to_unsigned(builder.create<circt::comb::OrOp>(to_signless(isnotzero_a), to_signless(isnotzero_b)));
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

    mlir::Type type = builder.getIntegerType(op->type.bits(), op->type.is_int());

    if (op->name == Call::buffer_get_host) {
        auto name = op->args[0].as<Variable>()->name;
        name = name.substr(0, name.find(".buffer"));
        debug(1) << "\t\targ name: " << name << "\n";
        debug(1) << "\t\top bits: " << op->type.bits() << "\n";
        // Return true
        value = builder.create<circt::hwarith::ConstantOp>(type, builder.getIntegerAttr(type, 1));
        //mlir::Location loc = mlir::NameLoc::get(mlir::StringAttr::get(builder.getContext(), "foo"));
        //value.setLoc(loc);
    } else if (op->name == Call::buffer_is_bounds_query) {
        auto name = op->args[0].as<Variable>()->name;
        name = name.substr(0, name.find(".buffer"));
        debug(1) << "\t\targ name: " << name << "\n";
        debug(1) << "\t\top bits: " << op->type.bits() << "\n";
        // Return true
        value = builder.create<circt::hwarith::ConstantOp>(type, builder.getIntegerAttr(type, 1));
    } else if(op->name == Call::buffer_get_min) {
        auto name = op->args[0].as<Variable>()->name;
        name = name.substr(0, name.find(".buffer"));
        debug(1) << "\t\targ name: " << name << "\n";
        debug(1) << "\t\top bits: " << op->type.bits() << "\n";
        // TODO: buf->dim[d].min
        value = builder.create<circt::hwarith::ConstantOp>(type, builder.getIntegerAttr(type, 41));
    } else if(op->name == Call::buffer_get_extent) {
        auto name = op->args[0].as<Variable>()->name;
        name = name.substr(0, name.find(".buffer"));
        debug(1) << "\t\targ name: " << name << "\n";
        debug(1) << "\t\top bits: " << op->type.bits() << "\n";
        // TODO: buf->dim[d].extent
        value = builder.create<circt::hwarith::ConstantOp>(type, builder.getIntegerAttr(type, 42));
    } else if(op->name == Call::buffer_get_stride) {
        auto name = op->args[0].as<Variable>()->name;
        name = name.substr(0, name.find(".buffer"));
        debug(1) << "\t\targ name: " << name << "\n";
        debug(1) << "\t\top bits: " << op->type.bits() << "\n";
        // TODO: buf->dim[d].stride
        value = builder.create<circt::hwarith::ConstantOp>(type, builder.getIntegerAttr(type, 43));
    } else {
        // Just return 1 for now
        value = builder.create<circt::hwarith::ConstantOp>(type, builder.getIntegerAttr(type, 1));
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

    mlir::Value clk = builder.create<circt::hw::ConstantOp>(builder.getI1Type(), builder.getBoolAttr(true));
    //mlir::Value ce = builder.create<circt::hw::ConstantOp>(builder.getI1Type(), builder.getBoolAttr(true));
    mlir::Value reset = builder.create<circt::hw::ConstantOp>(builder.getI1Type(), builder.getBoolAttr(false));

    // Execute the 'body' statement for all values of the variable 'name' from 'min' to 'min + extent'
    mlir::Value min = codegen(op->min);
    mlir::Value max = codegen(Add::make(op->min, op->extent));
    mlir::Type type = builder.getIntegerType(max.getType().getIntOrFloatBitWidth());

    mlir::Value iterator_next = builder.create<circt::sv::LogicOp>(type, "iterator_next");
    mlir::Value iterator_next_read = builder.create<circt::sv::ReadInOutOp>(iterator_next);
    mlir::Value iterator = builder.create<circt::seq::CompRegOp>(iterator_next_read, clk, reset, min, op->name);

    mlir::Value const_1 = builder.create<circt::hw::ConstantOp>(type, builder.getIntegerAttr(type, 1));
    mlir::Value iterator_add_1 = builder.create<circt::comb::AddOp>(iterator, const_1);
    builder.create<circt::sv::AssignOp>(iterator_next, iterator_add_1);

    sym_push(op->name, builder.create<circt::hwarith::CastOp>(max.getType(), iterator));
    codegen(op->body);
    sym_pop(op->name);
}

void CodeGen_CIRCT::Visitor::visit(const Store *op) {
    debug(1) << __PRETTY_FUNCTION__ << "\n";
    debug(1) << "\tName: " << op->name << "\n";

    //mlir::Value predicate = codegen(op->predicate);
    //mlir::Value value = codegen(op->value);
    //mlir::Value index = codegen(op->index);

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
