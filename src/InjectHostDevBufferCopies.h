#ifndef HALIDE_HOST_GPU_BUFFER_COPIES_H
#define HALIDE_HOST_GPU_BUFFER_COPIES_H

/** \file
 * Defines the lowering passes that deal with host and device buffer flow.
 */

#include <string>
#include <vector>

#include "Expr.h"

namespace Halide {

struct Target;

namespace Internal {

/** A helper function to call an extern function, and assert that it
 * returns 0. */
Stmt call_extern_and_assert(const std::string &name, const std::vector<Expr> &args);

Expr get_state_var(const std::string &name, bool &state_needed);

Expr make_state_var(const std::string &name);

// Create a Buffer containing the given vector, and return an
// expression for a pointer to the first element.
Expr make_buffer_ptr(const std::vector<char> &data, const std::string &name);

Expr make_string_ptr(const std::string &str, const std::string &name);

/** Inject calls to halide_device_malloc, halide_copy_to_device, and
 * halide_copy_to_host as needed. */
Stmt inject_host_dev_buffer_copies(Stmt s, const Target &t);

}  // namespace Internal
}  // namespace Halide

#endif
