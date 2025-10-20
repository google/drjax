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

from collections.abc import Sequence
import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from drjax._src import impls
from drjax._src import primitives
import jax
from jax import numpy as jnp
from jax.sharding import AxisType  # pylint: disable=g-importing-member
import numpy as np


def _jaxpr_has_primitive(jaxpr, prim_name: str):
  """A reimplementation of the fun of the same name in jax._src_dispatch."""
  for eqn in jaxpr.eqns:
    if prim_name in eqn.primitive.name:
      return True
    for subjaxpr in jax.core.subjaxprs(jaxpr):
      if _jaxpr_has_primitive(subjaxpr, prim_name):
        return True
  return False


def create_mesh(
    axis_type: AxisType,
) -> jax.sharding.Mesh:
  return jax.sharding.Mesh(
      np.asarray(jax.devices()).reshape(2, 4),
      axis_names=('clients', 'data'),
      axis_types=(axis_type, axis_type),
  )


def run_in_mesh(mesh_axes_types: Sequence[AxisType]):

  def _decorator(fn):

    @functools.wraps(fn)
    def _wrapped(self, *args, **kwargs):
      with self.subTest('no_mesh'):
        # Run once without a mesh, must not raise error.
        fn(self, *args, **kwargs)
      for mesh_axes_type in mesh_axes_types:
        with self.subTest(f'{mesh_axes_type=}'):
          mesh = create_mesh(mesh_axes_type)
          with jax.set_mesh(mesh), mesh:
            fn(self, *args, **kwargs)

    return _wrapped

  return _decorator


class PrimitivesActingOnArraysTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._n_clients = 100
    self._impl_defs = impls.PlacedComputations(
        {'clients': self._n_clients},
    )
    self._primdefs, _ = primitives.register_primitives(
        {'clients': self._n_clients},
    )

  @run_in_mesh((AxisType.Auto, AxisType.Explicit))
  def test_broadcast_clients_evaluation(self):
    fn = self._primdefs['broadcast_clients']
    # Check that this function is callable.
    chex.assert_trees_all_close(
        fn(jnp.array(1.0)), jnp.ones(shape=[self._n_clients])
    )
    # Check that it's jittable.
    chex.assert_trees_all_close(
        jax.jit(fn)(jnp.array(1.0)), jnp.ones(shape=[self._n_clients])
    )
    # Check that its forward-diffable.
    chex.assert_trees_all_close(
        jax.jacfwd(fn)(jnp.array(1.0)), jnp.ones(shape=[self._n_clients])
    )

    # Also that it's reverse-diffable.
    chex.assert_trees_all_close(
        jax.jacrev(fn)(jnp.array(1.0)), jnp.ones(shape=[self._n_clients])
    )

  @run_in_mesh((AxisType.Auto, AxisType.Explicit))
  def test_broadcast_clients_closure_under_fad(self):
    fn = self._primdefs['broadcast_clients']
    # Check that the forward and reverse-mode derivatives generate the expected
    # primitives.
    fwd_mode_jaxpr = jax.make_jaxpr(jax.jacfwd(fn))(jnp.array(1.0))
    self.assertTrue(_jaxpr_has_primitive(fwd_mode_jaxpr, 'broadcast_clients'))
    rev_mode_jaxpr = jax.make_jaxpr(jax.jacrev(fn))(jnp.array(1.0))
    self.assertTrue(_jaxpr_has_primitive(rev_mode_jaxpr, 'sum_from_clients'))

  @run_in_mesh((AxisType.Auto, AxisType.Explicit))
  def test_sum_from_clients_evaluation(self):
    fn = self._primdefs['sum_from_clients']
    clients_ones = jnp.ones(shape=[self._n_clients, 1])
    # Check that this function is callable.
    chex.assert_trees_all_close(
        fn(clients_ones), jnp.array([1.0 * self._n_clients])
    )
    # Check that it's jittable.
    chex.assert_trees_all_close(
        jax.jit(fn)(clients_ones), jnp.array([1.0 * self._n_clients])
    )
    # Check that its forward-diffable.
    chex.assert_trees_all_close(
        jax.jacfwd(fn)(clients_ones), jnp.ones(shape=[1, 100, 1])
    )
    # Check that its reverse-diffable.
    chex.assert_trees_all_close(
        jax.jacrev(fn)(clients_ones), jnp.ones(shape=[1, self._n_clients, 1])
    )

  @run_in_mesh((AxisType.Auto, AxisType.Explicit))
  def test_broadcast_and_sum_from_clients_eval(self):
    fn = self._primdefs['sum_from_clients']

    def _broadcast_then_sum(x):
      broadcasted_x = self._primdefs['broadcast_clients'](x)
      return fn(broadcasted_x)

    # This thing corresponds to fwd-mode AD in our paper.
    chex.assert_trees_all_close(
        jax.jacfwd(_broadcast_then_sum)(jnp.array([1.0])),
        jnp.array([[1.0 * self._n_clients]]),
    )

    # And here's reverse-ad.
    chex.assert_trees_all_close(
        jax.jacrev(_broadcast_then_sum)(jnp.array([1.0])),
        jnp.array([[1.0 * self._n_clients]]),
    )

  @run_in_mesh((AxisType.Auto, AxisType.Explicit))
  def test_sum_from_clients_closure_under_fad(self):
    # Check that the forward and reverse-mode derivatives generate the expected
    # primitives.
    fn = self._primdefs['sum_from_clients']
    clients_ones = jnp.ones(shape=[self._n_clients, 1])
    fwd_mode_jaxpr = jax.make_jaxpr(jax.jacfwd(fn))(clients_ones)
    self.assertTrue(_jaxpr_has_primitive(fwd_mode_jaxpr, 'sum_from_clients'))
    rev_mode_jaxpr = jax.make_jaxpr(jax.jacrev(fn))(clients_ones)
    self.assertTrue(_jaxpr_has_primitive(rev_mode_jaxpr, 'broadcast_clients'))

  @run_in_mesh((AxisType.Auto, AxisType.Explicit))
  def test_mean_from_clients_eval(self):
    fn = self._primdefs['mean_from_clients']
    clients_ones = jnp.ones(shape=[self._n_clients, 1])
    # Check that this function is callable.
    chex.assert_trees_all_close(fn(clients_ones), jnp.array([1.0]))
    # Check that it's jittable.
    chex.assert_trees_all_close(jax.jit(fn)(clients_ones), jnp.array([1.0]))
    # Check that its forward-diffable.
    chex.assert_trees_all_close(
        jax.jacfwd(fn)(clients_ones),
        1 / self._n_clients * jnp.ones(shape=[1, self._n_clients, 1]),
    )

  @run_in_mesh((AxisType.Auto, AxisType.Explicit))
  def test_broadcast_then_mean_from_clients_eval(self):
    fn = self._primdefs['mean_from_clients']

    def _broadcast_then_sum(x):
      broadcasted_x = self._primdefs['broadcast_clients'](x)
      return fn(broadcasted_x)

    # Again, let's do the forward-mode, reverse-mode checks.
    chex.assert_trees_all_close(
        jax.jacfwd(_broadcast_then_sum)(jnp.array([1.0])),
        jnp.array([[1.0]]),
    )
    chex.assert_trees_all_close(
        jax.jacrev(_broadcast_then_sum)(jnp.array([1.0])),
        jnp.array([[1.0]]),
    )

  @run_in_mesh((AxisType.Auto, AxisType.Explicit))
  def test_mean_from_clients_closure_under_fad(self):
    # Check that the forward and reverse-mode derivatives generate the expected
    # primitives.
    fn = self._primdefs['mean_from_clients']
    clients_ones = jnp.ones(shape=[self._n_clients, 1])
    fwd_mode_jaxpr = jax.make_jaxpr(jax.jacfwd(fn))(clients_ones)
    self.assertTrue(_jaxpr_has_primitive(fwd_mode_jaxpr, 'mean_from_clients'))
    rev_mode_jaxpr = jax.make_jaxpr(jax.jacrev(fn))(clients_ones)
    self.assertTrue(_jaxpr_has_primitive(rev_mode_jaxpr, 'broadcast_clients'))

  @parameterized.named_parameters(
      (
          'broadcast',
          'broadcast_clients',
          lambda _: jnp.array(1.0),
          lambda n: jnp.ones(shape=[n]),
      ),
      (
          'sum',
          'sum_from_clients',
          lambda n: jnp.ones(shape=[n, 1]),
          lambda n: jnp.ones(shape=[1, n, 1]),
      ),
      (
          'mean',
          'mean_from_clients',
          lambda n: jnp.ones(shape=[n, 1]),
          lambda n: 1 / n * jnp.ones(shape=[1, n, 1]),
      ),
  )
  def test_broadcast_clients_reverse_ad_with_symbolic_zero_and_jit(
      self, prim_name, arg_fn, result_fn
  ):
    fn = self._primdefs[prim_name]

    @jax.jit
    def duplicate_prim_result(x):
      return fn(x), fn(x)

    @jax.jit
    def ignore_prim_result(x):
      # Ignoring one result from this tuple-returning function triggers
      # reverse evaluation with a symbolic zero cotangent argument.
      y, _ = duplicate_prim_result(x)
      return y

    jac = jax.jacrev(ignore_prim_result)
    chex.assert_trees_all_close(
        jac(arg_fn(self._n_clients)), result_fn(self._n_clients)
    )


# This allows us to test sharding behavior across multiple devices.
def setUpModule():
  chex.set_n_cpu_devices(8)


if __name__ == '__main__':
  absltest.main()
