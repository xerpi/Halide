#ifndef HALIDE_OFFLOAD_LOOPS_TO_ACCELERATOR_H
#define HALIDE_OFFLOAD_LOOPS_TO_ACCELERATOR_H

/** \file
 * Defines a lowering pass to pull loops marked with to a
 * separate module tobe compiled for an accelerator,
 * and call them through the appropriate host runtime module.
 */

#include "Expr.h"

namespace Halide {

struct Target;

namespace Internal {

Stmt inject_accelerator_offload(const Stmt &s, const Target &host_target);

}  // namespace Internal
}  // namespace Halide

#endif
