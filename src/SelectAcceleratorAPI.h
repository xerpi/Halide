#ifndef HALIDE_INTERNAL_SELECT_ACCELERATOR_API_H
#define HALIDE_INTERNAL_SELECT_ACCELERATOR_API_H

#include "Expr.h"

/** \file
 * Defines a lowering pass that selects which accelerator api to use for each
 * for loop
 */

namespace Halide {

struct Target;

namespace Internal {

/** Replace *all* for loops with the default depending on what's enabled in the target */
Stmt select_accelerator_api(const Stmt &s, const Target &t);

}  // namespace Internal
}  // namespace Halide

#endif
