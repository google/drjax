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
"""Implementations of MapReduce primitives in JAX."""

from collections.abc import Mapping
from typing import Any

from absl import logging
import jax
from jax import numpy as jnp
from jax.experimental.shard_alike import shard_alike
from jax.interpreters import pxla
from jax.sharding import PartitionSpec as P


# The type aliases below encode a MapReduce programming model, inpsired by
# the programming model of TensorFlow Federated (TFF). Notable differences
# are that placements are static, and can only appear on arrays. There are no
# nested placements. Placement dimension comes first, so there is no need to
# worry about a total order on placements or figuring out what representation a
# nested-placed thing should have.

UnplacedArray = jnp.ndarray
PlacedArray = jnp.ndarray

PyTreePlaced = Any


def call_jaxpr(fn, arg):
  # Handles multi-element arguments.
  if isinstance(arg, tuple):
    return fn(*arg)
  else:
    return fn(arg)


# TODO(b/366437841): Remove use of pxla.thread_resources.env.physical_mesh,
# which is a JAX internal API.
def _global_mesh() -> jax.sharding.Mesh | None:
  """Returns the JAX global mesh if installed, or `None` otherwise."""
  jax_global_mesh = pxla.thread_resources.env.physical_mesh
  return None if jax_global_mesh.empty else jax_global_mesh


def _placement_axis_in_mesh(
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None,
    placement: str,
) -> bool:
  """Checks if a clients axis is present in the mesh."""
  if mesh is None:
    return False
  placement_is_in_mesh = placement in mesh.axis_names
  if not placement_is_in_mesh:
    logging.log_first_n(
        logging.WARNING,
        'No mesh axis named "%s" found in the current mesh, which had names'
        ' %s. DrJax will not inject sharding constraints.',
        10,
        placement,
        mesh.axis_names,
    )
  return placement_is_in_mesh


def _constrain_if_mesh(
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None,
    x: UnplacedArray,
    pspec: jax.sharding.PartitionSpec,
):
  if mesh is None:
    return x
  return jax.lax.with_sharding_constraint(
      x, jax.sharding.NamedSharding(mesh, pspec)
  )


class PlacedComputations:
  """Concrete implementations of MapReduce primitives in JAX."""

  def __init__(
      self,
      placements_to_n_elements: Mapping[str, int],
  ):
    self._placements_to_n_elements = placements_to_n_elements

  def broadcast_to_placement(
      self,
      arg: UnplacedArray,
      placement: str,
      mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
  ) -> PlacedArray:
    """Broadcasts (tiles) to the specified placement.

    That is, given an `arg` of shape `[a, ... b]`, and a `placement` with `n`
    elements, the result of this function should be an array of shape
    `[n, a, ... b]`, each of whose slices on the zeroth axis is identical to
    `arg`.

    This function shards the resulting array along a mesh axis corresponding to
    'placement', if a mesh is available and contains a 'placement' axis. If
    `mesh` is provided then this defines the mesh to be used. Else JAX's global
    mesh is used, if one is installed. If no mesh is available or if the
    available mesh does not contain a 'placement' axis then the result is
    unconstrained and the GSPMD partitioner may do whatever it wants.

    When a mesh is available and 'placement' is in the mesh, this function also
    directs the GSPMD compiler to shard the zeroth-axis slices of this tiled
    array in a similar manner to the argument.

    Args:
      arg: An array to be broadcast.
      placement: String representing the placement to which to broadcast `arg`.
      mesh: User provided mesh. If `None` then the JAX global mesh is used, if
        one is installed.

    Returns:
      A logically tiled array along the zeroth axis, as described above.
    """
    if mesh is None:
      mesh = _global_mesh()

    arg = jnp.array(arg)
    n_elements = self._placements_to_n_elements[placement]

    # Note that this pspec will only result in a sharding constraint defined if
    # a mesh is installed at tracing time.
    if _placement_axis_in_mesh(mesh, placement):
      pspec = P(placement, *([P.UNCONSTRAINED] * len(arg.shape)))
    else:
      # Without a clients axis in the mesh, we simply explicitly tell the
      # compiler that there are no constraints on this tensor. This will leave
      # the choices in the hands of the compiler.
      pspec = P(*([P.UNCONSTRAINED] * (len(arg.shape) + 1)))

    def single_arg_broadcast(x):
      unconstrained_tensor = jnp.tile(x, reps=[n_elements] + [1] * len(x.shape))
      if mesh is None:
        logging.warning(
            'No mesh found; defaulting to fully unconstrained broadcast and'
            ' *NOT* adding sharding constraints over the requested placement'
            ' axis %s.',
            placement,
        )
        return unconstrained_tensor
      else:

        def _shard_slice_like_arg(s):
          s_sharded, _ = shard_alike(s, x)
          return s_sharded

        original_dims_constrained = jax.vmap(_shard_slice_like_arg, in_axes=0)(
            unconstrained_tensor
        )
        fully_constrained = _constrain_if_mesh(
            mesh, original_dims_constrained, pspec
        )
        return fully_constrained

    return jax.jit(single_arg_broadcast)(arg)

  def normalized_broadcast_to_placement(
      self,
      arg: UnplacedArray,
      placement: str,
  ) -> PlacedArray:
    # This broadcasts arg / placement_size. This is intended only for reverse-
    # mode differentiation of mean-based aggregations.
    n_elements = self._placements_to_n_elements[placement]
    unnormalized_broadcast = self.broadcast_to_placement(arg, placement)
    return jnp.divide(unnormalized_broadcast, n_elements)

  def mean_from_placement(self, arg: PlacedArray) -> UnplacedArray:
    placement_idx = 0
    return jnp.mean(arg, axis=[placement_idx])

  def weighted_mean_from_placement(
      self, arg: PlacedArray, weight: PlacedArray
  ) -> UnplacedArray:
    placement_idx = 0
    return jnp.average(arg, axis=[placement_idx], weights=weight)

  def sum_from_placement(self, arg: PlacedArray) -> UnplacedArray:
    placement_idx = 0
    return jnp.sum(arg, axis=[placement_idx])

  def map_to_placement(
      self,
      fn,
      arg: PyTreePlaced,
      placement: str,
      mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
  ):
    """Maps a function to the specified placement.

    Suppose the user has a mesh with three axes, [placement, 'y', z']. Suppose
    we are asked to map a function f, of signature ([a], [b]) -> [c]. Suppose we
    have an array e, of shape [d, a, b], layed out along the mesh's three axes.

    Now, our mapping implementation is required to be able to: map f across the
    axis of size d, producing an array of shape [d, c], whose d-sized axis is
    layed out along the placement axis of the mesh, with layout of c-sized axis
    inherited from the operation of f. e.g., if f is jit-compiled with no
    sharding specifications, the layout of c will be determined by JAX's
    sharding propagation.

    In the case that the user has a mesh which does _not_ have placement as an
    axis name, the resulting array of shape [d, c], where the axis of size d is
    replicated across devices, and the axis of size c may be similarly sharded
    as above (depending on annotations internal to f, and the way the function
    constructed here is jit-compiled).

    The implementation here is intended to satisfy the sketch above (and its
    generalizations, including to pytrees, etc).

    Args:
      fn: Function to be mapped.
      arg: PyTree of arrays for which to map leading axis. Each array in the
        structure is assumed to have a leading axis of the same size, the number
        of elements at `placement`.
      placement: String representing the placement of input and output arrays to
        this map.
      mesh: User provided mesh. If `None` then the JAX global mesh is used, if
        one is installed.

    Returns:
      The result of mapping `fn` over the leading axis, satisfying the sharding
      requirements specified above.
    """
    if mesh is None:
      mesh = _global_mesh()

    def _constrain_at_placement_with_slices_like(x, y):
      pspec = P(placement, *([P.UNCONSTRAINED] * (len(x.shape) - 1)))
      placement_constrained = _constrain_if_mesh(mesh, x, pspec)

      def _shard_slice(s):
        s_sharded, _ = shard_alike(s, y[0])
        return s_sharded

      return jax.vmap(_shard_slice, in_axes=0)(placement_constrained)

    # `spmd_axis_name`` causes any internal with_sharding_constraints or
    # shard_map calls inside the `vmapped` function to respect the
    # sharding along this axis name. But it doesn't enrich annotations on
    # input / output tensors. Since we have a very limited mapping semantic
    # here, adding these annotations is always safe for us, as long as
    # `placement` is in the mesh.
    if _placement_axis_in_mesh(mesh, placement):
      arg = jax.tree_util.tree_map(
          _constrain_at_placement_with_slices_like, arg, arg
      )
      mapped_fn = jax.vmap(
          # We must not have an `axis_name` argument here in order to work
          # with any potential `shard_map` inside of `fn`.
          fn,
          in_axes=0,
          out_axes=0,
          spmd_axis_name=placement,
      )

      # In some cases, vmap may prevent placement sharding from propagating. We
      # ensure placement sharding on the output just in case.
      result = call_jaxpr(mapped_fn, arg)
      return jax.tree_util.tree_map(
          _constrain_at_placement_with_slices_like, result, result
      )
    else:
      logging.warning(
          'No mesh containing axis name %s found; defaulting to standard vmap.'
          ' Mesh contains names: %s',
          placement,
          mesh.axis_names if mesh is not None else 'None',
      )
      # Users should be free to use whatever mesh their model needs without
      # _necessarily_ registering a mesh-dimension for every placement with
      # which they are programming.
      mapped_fn = jax.vmap(
          fn,
          axis_name=placement,
          in_axes=0,
      )
      return call_jaxpr(mapped_fn, arg)
