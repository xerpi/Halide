#include "SelectAcceleratorAPI.h"
#include "DeviceInterface.h"
#include "IRMutator.h"

namespace Halide {
namespace Internal {

namespace {

class SelectAcceleratorAPI : public IRMutator {
    using IRMutator::visit;

    DeviceAPI default_api;

    Expr visit(const Call *op) override {
        if (op->name == "halide_default_device_interface") {
            return make_device_interface_call(default_api);
        } else {
            return IRMutator::visit(op);
        }
    }

    Stmt visit(const For *op) override {
        DeviceAPI selected_api = op->device_api;
        if (op->device_api != default_api) {
            selected_api = default_api;
        }

        Stmt stmt = IRMutator::visit(op);

        op = stmt.as<For>();
        internal_assert(op);

        if (op->device_api != selected_api) {
            return For::make(op->name, op->min, op->extent, op->for_type, selected_api, op->body);
        }
        return stmt;
    }

public:
    SelectAcceleratorAPI(const Target &t) {
        default_api = get_default_device_api_for_target(t);
    }
};

}  // namespace

Stmt select_accelerator_api(const Stmt &s, const Target &t) {
    return SelectAcceleratorAPI(t).mutate(s);
}

}  // namespace Internal
}  // namespace Halide
