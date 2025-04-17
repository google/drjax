# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A library of MapReduce primitives defined as JAX primitives."""

from collections.abc import Callable, Mapping, MutableMapping
from typing import Protocol, Union

from drjax._src import impls
import jax
from jax import core
from jax import numpy as jnp
from jax.extend import core as extended_core
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir


class BroadcastType(Protocol):

  def __call__(
      self,
      x: jnp.ndarray,
      mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
  ) -> jnp.ndarray:
    ...


AggType = Callable[[jnp.ndarray], jnp.ndarray]


def _define_broadcast_prim(
    broadcast_name,
) -> tuple[extended_core.Primitive, BroadcastType]:
  """Defines and returns broadcast ptimitive and associated binding."""
  broadcast_p = extended_core.Primitive(broadcast_name)  # Create the primitive

  def broadcast_prim_fn(x, *, mesh=None):
    return broadcast_p.bind(x, mesh=mesh)

  return (broadcast_p, broadcast_prim_fn)


def _register_broadcast_impls(
    broadcast_p: extended_core.Primitive,
    broadcast_prim_fn: BroadcastType,
    broadcast_array_eval: BroadcastType,
    sum_prim_fn: AggType,
    n_elements: int,
) -> None:
  """Registers implementations for the broadcast primitive.

  Definitions for Jacobian-vector products and vector-jacobian products are
  derived from forward and reverse differentiation rules presented in
  Federated Automatic Differentiation, https://arxiv.org/pdf/2301.07806v1.pdf.

  Args:
    broadcast_p: `jax.extend.core.Primitive` representing the broadcast
      operation.
    broadcast_prim_fn: A callable which binds its arguments to `broadcast_p`, as
      in
      https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html#defining-new-jax-primitives
    broadcast_array_eval: A callable accepting and returning Jax arrays, which
      defines the array-valued implementation of this broadcast primitive.
    sum_prim_fn: A callable which binds its arguments to the summation primitive
      from the placement inserted by this broadcast. Similar to
      `broadcast_prim_fn`.
    n_elements: The number of elements present at the placement which this
      broadcast targets.
  """

  def broadcast_abstract_eval(xs, *, mesh):
    del mesh
    return core.ShapedArray((n_elements,) + xs.shape, xs.dtype)

  # Abstract eval rule.
  broadcast_p.def_abstract_eval(broadcast_abstract_eval)
  # Concrete eval rule.
  broadcast_p.def_impl(broadcast_array_eval)
  # Lowering rule to MLIR.
  mlir.register_lowering(
      broadcast_p, mlir.lower_fun(broadcast_array_eval, multiple_results=False)
  )

  def broadcast_jvp(primals_in, tangents_in, mesh):
    primals_out = broadcast_prim_fn(*primals_in, mesh=mesh)
    tangents_out = broadcast_prim_fn(*tangents_in, mesh=mesh)
    return primals_out, tangents_out

  # Registering JVP should allow forward AD.
  ad.primitive_jvps[broadcast_p] = broadcast_jvp

  def broadcast_vjp(cotangents_out, primals_in, mesh):
    del mesh
    if isinstance(cotangents_out, jax.interpreters.ad.Zero):
      # We are differerentiating back through a broadcast; the incoming value,
      # therefore, has the right shape and dtype for the Zero we generate.
      return (jax.interpreters.ad.Zero(primals_in.aval),)
    # This implementation *must* use the sum_prim_fn, rather than the array
    # implementation of summation, to result in a reduce_sum in the Jaxpr.
    return (sum_prim_fn(cotangents_out),)

  ad.primitive_transposes[broadcast_p] = broadcast_vjp

  def _batch_broadcast(xs, batched_shape, mesh):
    # We inserted clients dimension in front, so batch dim went down one.
    return broadcast_prim_fn(*xs, mesh=mesh), batched_shape[0] + 1

  # Make sure this can also be batched / mapped. This happens when dispatching
  # forward AD, I think.
  batching.primitive_batchers[broadcast_p] = _batch_broadcast


def _define_single_arg_agg_prim(
    agg_name,
) -> tuple[extended_core.Primitive, AggType]:
  """Defines and returns an aggregation primitive taking a single argument."""
  agg_p = extended_core.Primitive(agg_name)  # Create the primitive

  def agg_prim_fn(x):
    return agg_p.bind(x)

  return agg_p, agg_prim_fn


def _register_single_arg_agg_impls(
    agg_p: extended_core.Primitive,
    agg_prim_fn: AggType,
    agg_array_eval: AggType,
    vjp_impl: BroadcastType,
) -> None:
  """Registers implementations for aggregation primitive taking a single arg.

  Definitions for Jacobian-vector products and vector-jacobian products are
  derived from forward and reverse differentiation rules presented in
  Federated Automatic Differentiation, https://arxiv.org/pdf/2301.07806v1.pdf.

  Args:
    agg_p: `jax.extend.core.Primitive` representing the primitive aggregation
      operation.
    agg_prim_fn: A callable which binds its arguments to `agg_p`, as in
      https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html#defining-new-jax-primitives
    agg_array_eval: A callable accepting and returning Jax arrays, which defines
      the array-valued implementation of this aggregation primitive.
    vjp_impl: An implementation of reverse-mode differentiation through this
      aggregation primitive.
  """

  def agg_abstract_eval(xs):
    return jax.tree_util.tree_map(
        # We slice away the first dimension in doing the reduction; its gone!
        lambda x: core.ShapedArray(x.shape[1:], x.dtype),
        xs,
    )

  # Abstract eval rule
  agg_p.def_abstract_eval(agg_abstract_eval)
  # Concrete eval rule
  agg_p.def_impl(agg_array_eval)
  # Lowering rule to MLIR.
  mlir.register_lowering(
      agg_p,
      mlir.lower_fun(agg_array_eval, multiple_results=False),
  )

  def agg_jvp(primals_in, tangents_in):
    return agg_prim_fn(*primals_in), agg_prim_fn(*tangents_in)

  # Registering JVP should allow forward AD.
  ad.primitive_jvps[agg_p] = agg_jvp

  def agg_vjp(cotangents_out, primals_in):
    if isinstance(cotangents_out, jax.interpreters.ad.Zero):
      # We are differerentiating back through an aggregation; the incoming
      # value, therefore, has the right shape and dtype for the Zero we
      # generate. This is always correct if jax's symbolic Zero is a static
      # concept, depending on data flow in the program (rather than e.g. runtime
      # values).
      return (jax.interpreters.ad.Zero(primals_in.aval),)
    return (vjp_impl(cotangents_out),)

  ad.primitive_transposes[agg_p] = agg_vjp

  def _batch_agg(xs, batched_shape):
    # Certain jax libs can silently insert the 'batching' dim 'all the way at
    # the front'; we are about to destroy the front axis by agging, so move
    # that puppy to the back. Tell the rest of JAX what happened here.
    xs = batching.moveaxis(*xs, *batched_shape, -1)
    return agg_prim_fn(xs), len(xs.shape) - 2

  # Make sure this can also be batched / mapped. This happens when dispatching
  # forward AD, I think.
  batching.primitive_batchers[agg_p] = _batch_agg


def _define_and_register_prims_for_placement(
    primitive_dict: MutableMapping[str, Union[BroadcastType, AggType]],
    primdef_dict: MutableMapping[str, extended_core.Primitive],
    impl_defs: impls.PlacedComputations,
    placement_str: str,
    n_elements: int,
) -> None:
  """Registers primitives for a given placement and cardinality.

  Args:
    primitive_dict: Dictionary in which to place the defined primitive
      _functions_.
    primdef_dict: Dictionary in which to place the defined primitive _objects_.
    impl_defs: Instance of `impls.PlacedComputations` which provides the
      Jax-array accepting and returning implementations for the primitives we
      define here.
    placement_str: String representing the name of the placement to which we may
      broadcast, and from which we may aggregate.
    n_elements: The cardinality of this placement. That is, how many
      participants this placement supports.
  """
  broadcast_name = f'broadcast_{placement_str}'
  sum_name = f'sum_from_{placement_str}'
  mean_name = f'mean_from_{placement_str}'
  # Primitive and prim_fn definitions must come first, since they may need to
  # generate other primitives in the body of their autodiff implementations.
  broadcast_p, broadcast_prim_fn = _define_broadcast_prim(broadcast_name)
  sum_p, sum_prim_fn = _define_single_arg_agg_prim(sum_name)
  mean_p, mean_prim_fn = _define_single_arg_agg_prim(mean_name)

  primitive_dict[broadcast_name] = broadcast_prim_fn
  primitive_dict[sum_name] = sum_prim_fn
  primitive_dict[mean_name] = mean_prim_fn

  primdef_dict[broadcast_name] = broadcast_p
  primdef_dict[sum_name] = sum_p
  primdef_dict[mean_name] = mean_p

  def broadcast_array_eval(x, *, mesh):
    return impl_defs.broadcast_to_placement(x, placement_str, mesh)

  _register_broadcast_impls(
      broadcast_p,
      broadcast_prim_fn,
      broadcast_array_eval,
      sum_prim_fn,
      n_elements,
  )

  _register_single_arg_agg_impls(
      sum_p,
      sum_prim_fn,
      impl_defs.sum_from_placement,
      broadcast_prim_fn,
  )
  _register_single_arg_agg_impls(
      mean_p,
      mean_prim_fn,
      impl_defs.mean_from_placement,
      lambda x: jnp.divide(broadcast_prim_fn(x), n_elements),
  )


def register_primitives(
    placements: Mapping[str, int],
) -> tuple[
    dict[str, Union[BroadcastType, AggType]], dict[str, extended_core.Primitive]
]:
  """Registers broadcast, sum and mean primitives for all placements present."""
  impl_defs = impls.PlacedComputations(placements)
  primitive_dict = {}
  primdef_dict = {}
  for placement_str, n_elements in placements.items():
    _define_and_register_prims_for_placement(
        primitive_dict, primdef_dict, impl_defs, placement_str, n_elements
    )

  return primitive_dict, primdef_dict
