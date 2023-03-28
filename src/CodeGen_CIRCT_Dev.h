#ifndef HALIDE_CODEGEN_CIRCT_DEV_H
#define HALIDE_CODEGEN_CIRCT_DEV_H

/** \file
 * Defines the code-generator for producing CIRCT MLIR code
 */

#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/MLIRContext.h>

#include <memory>

namespace Halide {

struct Target;

namespace Internal {

static inline int argGetHWBits(const DeviceArgument &arg) {
    return arg.is_buffer ? 64 : arg.type.bits();
}

class CodeGen_CIRCT_Dev {
public:
    bool compile(mlir::LocationAttr &loc, mlir::ModuleOp &mlir_module, Stmt stmt,
                 const std::string &name, const std::vector<DeviceArgument> &args,
                 std::string &calyxOutput);

protected:
    class Visitor : public IRVisitor {
    public:
        Visitor(mlir::ImplicitLocOpBuilder &builder, const std::vector<std::string> &inputNames);

    protected:
        mlir::Value codegen(const Expr &);
        void codegen(const Stmt &);

        void visit(const IntImm *) override;
        void visit(const UIntImm *) override;
        void visit(const FloatImm *) override;
        void visit(const StringImm *) override;
        void visit(const Cast *) override;
        void visit(const Reinterpret *) override;
        void visit(const Variable *) override;
        void visit(const Add *) override;
        void visit(const Sub *) override;
        void visit(const Mul *) override;
        void visit(const Div *) override;
        void visit(const Mod *) override;
        void visit(const Min *) override;
        void visit(const Max *) override;
        void visit(const EQ *) override;
        void visit(const NE *) override;
        void visit(const LT *) override;
        void visit(const LE *) override;
        void visit(const GT *) override;
        void visit(const GE *) override;
        void visit(const And *) override;
        void visit(const Or *) override;
        void visit(const Not *) override;
        void visit(const Select *) override;
        void visit(const Load *) override;
        void visit(const Ramp *) override;
        void visit(const Broadcast *) override;
        void visit(const Call *) override;
        void visit(const Let *) override;
        void visit(const LetStmt *) override;
        void visit(const AssertStmt *) override;
        void visit(const ProducerConsumer *) override;
        void visit(const For *) override;
        void visit(const Store *) override;
        void visit(const Provide *) override;
        void visit(const Allocate *) override;
        void visit(const Free *) override;
        void visit(const Realize *) override;
        void visit(const Block *) override;
        void visit(const IfThenElse *) override;
        void visit(const Evaluate *) override;
        void visit(const Shuffle *) override;
        void visit(const VectorReduce *) override;
        void visit(const Prefetch *) override;
        void visit(const Fork *) override;
        void visit(const Acquire *) override;
        void visit(const Atomic *) override;

        void sym_push(const std::string &name, mlir::Value value);
        void sym_pop(const std::string &name);
        mlir::Value sym_get(const std::string &name, bool must_succeed = true) const;

    private:
        mlir::ImplicitLocOpBuilder &builder;
        mlir::Value value;
        Scope<mlir::Value> symbol_table;
    };
};

}  // namespace Internal
}  // namespace Halide

#endif
