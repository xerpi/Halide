#ifndef HALIDE_CODEGEN_CIRCT_H
#define HALIDE_CODEGEN_CIRCT_H

/** \file
 * Defines the code-generator for producing CIRCT MLIR code
 */

#include <string>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/MLIRContext.h>

#include <circt/Dialect/HW/HWOps.h>

#include "IRVisitor.h"
#include "Module.h"
#include "Scope.h"

namespace Halide {

namespace Internal {

class CodeGen_CIRCT {
public:
    CodeGen_CIRCT();

    void compile(const Module &module);

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

private:
    struct KernelArg {
        std::string name;
        Type type;
        bool isPointer = false;

        int getHWBits() const {
            return isPointer ? 64 : type.bits();
        }
    };
    using FlattenedKernelArgs = std::vector<KernelArg>;

    // XRT-Managed Kernels Control Requirements
    // See https://docs.xilinx.com/r/en-US/ug1393-vitis-application-acceleration/Control-Requirements-for-XRT-Managed-Kernels
    static constexpr uint64_t XRT_KERNEL_ARGS_OFFSET = 0x10;
    static constexpr int M_AXI_ADDR_WIDTH = 64;
    static constexpr int M_AXI_DATA_WIDTH = 32;
    static constexpr int S_AXI_ADDR_WIDTH = 32;
    static constexpr int S_AXI_DATA_WIDTH = 32;

    static constexpr char AXI_MANAGER_PREFIX[] = "m_axi_";

    static void flattenKernelArguments(const std::vector<LoweredArgument> &inArgs, FlattenedKernelArgs &args);

    static void generateKernelXml(const std::string &kernelName, const FlattenedKernelArgs &kernelArgs);
    static void generateCalyxExtMemToAxi(mlir::ImplicitLocOpBuilder &builder);
    static void generateControlAxi(mlir::ImplicitLocOpBuilder &builder, const FlattenedKernelArgs &kernelArgs);
    static void generateToplevel(mlir::ImplicitLocOpBuilder &builder, const std::string &kernelName, const FlattenedKernelArgs &kernelArgs);

    static void portsAddAXI4ManagerSignalsPrefix(mlir::ImplicitLocOpBuilder &builder, const std::string &prefix,
                                                 int addrWidth, int dataWidth,
                                                 mlir::SmallVector<circt::hw::PortInfo> &ports);
    static void portsAddAXI4LiteSubordinateSignals(mlir::ImplicitLocOpBuilder &builder, int addrWidth, int dataWidth,
                                                   mlir::SmallVector<circt::hw::PortInfo> &ports);

    static std::string getAxiManagerSignalNamePrefixId(int id) {
        return "m" + std::string(id < 10 ? "0" : "") + std::to_string(id) + "_axi";
    }

    static std::string toFullAxiManagerSignalName(const std::string &name) {
        return std::string(AXI_MANAGER_PREFIX) + name;
    };

    static std::string toFullAxiManagerSignalNameId(int id, const std::string &name) {
        return getAxiManagerSignalNamePrefixId(id) + "_" + name;
    };

    static std::string toFullAxiSubordinateSignalName(const std::string &name) {
        return "s_axi_" + name;
    };

    static std::string fullAxiSignalNameIdGetBasename(const std::string &name) {
        std::string token = "axi_";
        return name.substr(name.find(token) + token.size());
    };

    mlir::MLIRContext mlir_context;
};

}  // namespace Internal
}  // namespace Halide

#endif
