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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from drjax._src import api
import jax
from jax import numpy as jnp
import numpy as np


@functools.wraps(api.drjax_program)
def drjax_program(*, placements):
  return api.drjax_program(placements=placements, self_module=api)


@parameterized.named_parameters(
    ("clients_placed", "clients"), ("XY_placed", "XY")
)
class ApiTest(absltest.TestCase):

  def test_sharded_broadcast(self, placement_name):

    @drjax_program(placements={placement_name: 100})
    def broadcast_val(val):
      return api.broadcast(val)

    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("some_axis",))
    arg_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("some_axis")
    )
    with mesh:
      result = broadcast_val(
          jax.device_put(jnp.ones(shape=[8, 8]), arg_sharding)
      )

    chex.assert_trees_all_close(result, jnp.ones(shape=[100, 8, 8]))
    # No clients dimension in the mesh, we don't lay out the clients along that
    # nonexistent dimension, but rather replicate them. Notice that we don't
    # need to specify the sharding to DrJAX; it should be inferred by GSPMD.
    expected_result_pspec = jax.sharding.PartitionSpec(None, "some_axis")
    self.assertEqual(
        result.sharding, jax.sharding.NamedSharding(mesh, expected_result_pspec)
    )

  def test_sharded_broadcast_mesh_arg(self, placement_name):

    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("some_axis",))

    @drjax_program(placements={placement_name: 100})
    def broadcast_val(val):
      return api.broadcast(val, mesh=mesh)

    arg_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("some_axis")
    )
    result = broadcast_val(jax.device_put(jnp.ones(shape=[8, 8]), arg_sharding))

    chex.assert_trees_all_close(result, jnp.ones(shape=[100, 8, 8]))
    # No clients dimension in the mesh, we don't lay out the clients along that
    # nonexistent dimension, but rather replicate them. Notice that we don't
    # need to specify the sharding to DrJAX; it should be inferred by GSPMD.
    expected_result_pspec = jax.sharding.PartitionSpec(None, "some_axis")
    self.assertEqual(
        result.sharding, jax.sharding.NamedSharding(mesh, expected_result_pspec)
    )

  def test_fully_sharded_broadcast_mesh_arg(self, placement_name):

    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape([4, 2]), (placement_name, "some_axis")
    )

    @drjax_program(placements={placement_name: 8})
    def broadcast_val(val):
      return api.broadcast(val, mesh=mesh)

    arg_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("some_axis")
    )

    result = broadcast_val(jax.device_put(jnp.ones(shape=[8, 8]), arg_sharding))

    chex.assert_trees_all_close(result, jnp.ones(shape=[8, 8, 8]))
    # The result should be sharded across the placement_name axis.
    expected_result_pspec = jax.sharding.PartitionSpec(
        placement_name, "some_axis"
    )
    self.assertEqual(
        result.sharding, jax.sharding.NamedSharding(mesh, expected_result_pspec)
    )

  def test_temp_sens_example(self, placement_name):
    def one_if_over(threshold, value):
      return jax.lax.cond(value > threshold, lambda: 1.0, lambda: 0.0)

    placement_dim = 100

    @drjax_program(placements={placement_name: placement_dim})
    def temp_sens_example(threshold, values):
      threshold_at_clients = api.broadcast(threshold)
      values_over = api.map_fn(one_if_over, (threshold_at_clients, values))
      return api.reduce_mean(values_over)

    measurements = jnp.arange(placement_dim)

    self.assertEqual(temp_sens_example(24, measurements), 0.75)

  def test_temp_sens_example_multiple_placement_values(self, placement_name):
    def one_if_over(threshold, value):
      return jax.lax.cond(value > threshold, lambda: 1.0, lambda: 0.0)

    @drjax_program(placements={placement_name: 100})
    def temp_sens_example_100_clients(threshold, values):
      threshold_at_clients = api.broadcast(threshold)
      values_over = api.map_fn(one_if_over, (threshold_at_clients, values))

      return api.reduce_mean(values_over)

    @drjax_program(placements={placement_name: 10})
    def temp_sens_example_10_clients(threshold, values):
      threshold_at_clients = api.broadcast(threshold)
      values_over = api.map_fn(one_if_over, (threshold_at_clients, values))
      return api.reduce_mean(values_over)

    measurements_100 = jnp.arange(100)
    measurements_10 = jnp.arange(10)

    self.assertEqual(temp_sens_example_100_clients(24, measurements_100), 0.75)
    self.assertEqual(
        temp_sens_example_10_clients(3, measurements_10),
        0.6,
    )
    # We should be able to recover the original result flipping back to the
    # original function.
    self.assertEqual(temp_sens_example_100_clients(24, measurements_100), 0.75)

  def test_multiple_placements_raises(self, placement_name):

    with self.assertRaises(ValueError):

      @drjax_program(placements={placement_name: 1, placement_name + "x": 1})
      def _(values):
        return api.reduce_mean(values)

  def test_raises_outside_program_context(self, placement_name):
    with self.assertRaises(api.OperatorUndefinedError):
      api.broadcast(jnp.array(0.5))

    num_clients = 10

    @drjax_program(placements={placement_name: num_clients})
    def test(values):
      return api.reduce_mean(values)

    # Should not raise, inside a drjax context.
    test(jax.random.uniform(jax.random.PRNGKey(42), shape=[num_clients]))

    # Should raise again now that we've left the context.
    with self.assertRaises(api.OperatorUndefinedError):
      api.broadcast(jnp.array(0.5))

  def test_broadcast_raises_type_error_within_program_context(
      self, placement_name
  ):

    @drjax_program(placements={placement_name: 1})
    def test(*args):
      return api.broadcast(*args)

    with self.assertRaisesRegex(
        TypeError, r"broadcast\(\) takes 1 positional argument but 2 were given"
    ):
      test(jnp.array(0.5), jnp.array(0.5))

  def test_map_fn_raises_type_error_within_program_context(
      self, placement_name
  ):

    @drjax_program(placements={placement_name: 1})
    def test(*args):
      return api.map_fn(lambda x: x, *args)

    with self.assertRaisesRegex(
        TypeError, r"map_fn\(\) takes 2 positional arguments but 3 were given"
    ):
      test(jnp.array(0.5), jnp.array(0.5))

  def test_reduce_sum_raises_type_error_within_program_context(
      self, placement_name
  ):
    @drjax_program(placements={placement_name: 1})
    def test(*args):
      return api.reduce_sum(*args)

    with self.assertRaisesRegex(
        TypeError,
        r"reduce_sum\(\) takes 1 positional argument but 2 were given",
    ):
      test(jnp.array(0.5), jnp.array(0.5))

  def test_reduce_mean_raises_type_error_within_program_context(
      self, placement_name
  ):
    @drjax_program(placements={placement_name: 1})
    def test(*args):
      return api.reduce_mean(*args)

    with self.assertRaisesRegex(
        TypeError,
        r"reduce_mean\(\) takes 1 positional argument but 2 were given",
    ):
      test(jnp.array(0.5), jnp.array(0.5))

  def test_map_fn_error_propagates(self, placement_name):
    test_msg = "This is a test value error."
    def foo(_):
      raise ValueError(test_msg)

    @drjax_program(placements={placement_name: 1})
    def trigger_error(x):
      return api.map_fn(foo, x)

    with self.assertRaisesRegex(ValueError, test_msg):
      trigger_error(jnp.asarray([0]))


# This allows us to test sharding behavior across multiple devices.
def setUpModule():
  chex.set_n_cpu_devices(8)


if __name__ == "__main__":
  absltest.main()
