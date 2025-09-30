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
import itertools

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


@parameterized.product(
    placement_name=["clients", "XY"],
    axes_type=[
        jax.sharding.AxisType.Auto,
        jax.sharding.AxisType.Explicit,
    ],
)
class ApiTest(absltest.TestCase):

  def assertShardingEqual(self, arr, sharding):
    canonical_array_sharding = jax.sharding.NamedSharding(
        arr.sharding.mesh,
        # Canonicalize with trailing `None`s to the rank of the input array.
        # This canonicalizes across Auto and Explicit axis types, the former
        # which may not include trailing `None`s.
        jax.sharding.PartitionSpec(*(
            axis
            for axis, _ in itertools.zip_longest(arr.sharding.spec, arr.shape)
        )),
    )
    self.assertEqual(canonical_array_sharding, sharding)

  def test_broadcast_with_placement_in_mesh(self, placement_name, axes_type):

    @drjax_program(placements={placement_name: 100})
    def broadcast_val(val):
      return api.broadcast(val)

    mesh = jax.sharding.Mesh(
        np.array(jax.devices()),
        axis_names=("some_axis",),
        axis_types=(axes_type,),
    )
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
    expected_result_pspec = jax.sharding.PartitionSpec(None, "some_axis", None)
    self.assertShardingEqual(
        result, jax.sharding.NamedSharding(mesh, expected_result_pspec)
    )

  def test_broadcast_mesh_arg_without_placement(
      self, placement_name, axes_type
  ):
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()),
        axis_names=("some_axis",),
        axis_types=(axes_type,),
    )

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
    expected_result_pspec = jax.sharding.PartitionSpec(None, "some_axis", None)
    self.assertShardingEqual(
        result, jax.sharding.NamedSharding(mesh, expected_result_pspec)
    )

  def test_fully_sharded_broadcast_mesh_arg(self, placement_name, axes_type):
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape([4, 2]),
        axis_names=(placement_name, "some_axis"),
        axis_types=(axes_type, axes_type),
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
        placement_name, "some_axis", None
    )
    self.assertShardingEqual(
        result, jax.sharding.NamedSharding(mesh, expected_result_pspec)
    )

  def test_temperature_sensors_example(self, placement_name, axes_type):
    def one_if_over(threshold, value):
      return jax.lax.cond(
          value > threshold,
          lambda: jnp.ones_like(value),
          lambda: jnp.zeros_like(value),
      )

    placement_dim = 100
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape([4, 2]),
        axis_names=(placement_name, "some_axis"),
        axis_types=(axes_type, axes_type),
    )
    jax.set_mesh(mesh)

    @drjax_program(placements={placement_name: placement_dim})
    def temperature_sensors_example(threshold, values):
      threshold_at_clients = api.broadcast(threshold)
      values_over = api.map_fn(one_if_over, (threshold_at_clients, values))
      return api.reduce_mean(values_over)

    measurements = jax.device_put(
        jnp.arange(placement_dim),
        jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(placement_name)
        ),
    )

    self.assertEqual(temperature_sensors_example(24, measurements), 0.75)

  def test_temperature_sensors_example_multiple_placement_values(
      self, placement_name, axes_type
  ):

    def one_if_over(threshold, value):
      return jax.lax.cond(
          value > threshold,
          lambda: jnp.ones_like(value),
          lambda: jnp.zeros_like(value),
      )

    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape([4, 2]),
        axis_names=(placement_name, "some_axis"),
        axis_types=(axes_type, axes_type),
    )
    jax.set_mesh(mesh)

    @drjax_program(placements={placement_name: 100})
    def temperature_sensors_example_100_clients(threshold, values):
      threshold_at_clients = api.broadcast(threshold)
      values_over = api.map_fn(one_if_over, (threshold_at_clients, values))
      return api.reduce_mean(values_over)

    @drjax_program(placements={placement_name: 20})
    def temperature_sensors_example_20_clients(threshold, values):
      threshold_at_clients = api.broadcast(threshold)
      values_over = api.map_fn(one_if_over, (threshold_at_clients, values))
      return api.reduce_mean(values_over)

    placement_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(placement_name)
    )
    measurements_100 = jax.device_put(jnp.arange(100), placement_sharding)
    measurements_20 = jax.device_put(jnp.arange(20), placement_sharding)

    self.assertEqual(
        temperature_sensors_example_100_clients(24, measurements_100), 0.75
    )
    self.assertEqual(
        temperature_sensors_example_20_clients(3, measurements_20),
        0.8,
    )
    # We should be able to recover the original result flipping back to the
    # original function.
    self.assertEqual(
        temperature_sensors_example_100_clients(24, measurements_100), 0.75
    )


class ApiErrorsTest(absltest.TestCase):

  def test_multiple_placements_raises(self):
    placement_name = "XY"

    with self.assertRaises(ValueError):

      @drjax_program(placements={placement_name: 1, placement_name + "x": 1})
      def _(values):
        return api.reduce_mean(values)

  def test_raises_outside_program_context(self):
    with self.assertRaises(api.OperatorUndefinedError):
      api.broadcast(jnp.array(0.5))

    num_clients = 10

    @drjax_program(placements={"xy": num_clients})
    def test(values):
      return api.reduce_mean(values)

    # Should not raise, inside a drjax context.
    test(jax.random.uniform(jax.random.PRNGKey(42), shape=[num_clients]))

    # Should raise again now that we've left the context.
    with self.assertRaises(api.OperatorUndefinedError):
      api.broadcast(jnp.array(0.5))

  def test_broadcast_raises_type_error_within_program_context(self):

    @drjax_program(placements={"xy": 1})
    def test(*args):
      return api.broadcast(*args)

    with self.assertRaisesRegex(
        TypeError, r"broadcast\(\) takes 1 positional argument but 2 were given"
    ):
      test(jnp.array(0.5), jnp.array(0.5))

  def test_map_fn_raises_type_error_within_program_context(self):

    @drjax_program(placements={"xy": 1})
    def test(*args):
      return api.map_fn(lambda x: x, *args)

    with self.assertRaisesRegex(
        TypeError, r"map_fn\(\) takes 2 positional arguments but 3 were given"
    ):
      test(jnp.array(0.5), jnp.array(0.5))

  def test_reduce_sum_raises_type_error_within_program_context(self):

    @drjax_program(placements={"xy": 1})
    def test(*args):
      return api.reduce_sum(*args)

    with self.assertRaisesRegex(
        TypeError,
        r"reduce_sum\(\) takes 1 positional argument but 2 were given",
    ):
      test(jnp.array(0.5), jnp.array(0.5))

  def test_reduce_mean_raises_type_error_within_program_context(self):

    @drjax_program(placements={"xy": 1})
    def test(*args):
      return api.reduce_mean(*args)

    with self.assertRaisesRegex(
        TypeError,
        r"reduce_mean\(\) takes 1 positional argument but 2 were given",
    ):
      test(jnp.array(0.5), jnp.array(0.5))

  def test_map_fn_error_propagates(self):

    test_msg = "This is a test value error."
    def foo(_):
      raise ValueError(test_msg)

    @drjax_program(placements={"clients": 1})
    def trigger_error(x):
      return api.map_fn(foo, x)

    with self.assertRaisesRegex(ValueError, test_msg):
      trigger_error(jnp.asarray([0]))

  def test_apis_with_mixed_mode_mesh_axes_raise_error(self):

    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape([4, 2]),
        axis_names=("xy", "some_axis"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Auto),
    )
    with jax.set_mesh(mesh):
      with self.subTest("map"), self.assertRaisesRegex(
          ValueError, "Mesh axis types must all be either auto or manual"
      ):

        @drjax_program(placements={"xy": 1})
        def test_map(x):
          return api.map_fn(lambda arr: arr, x)

        test_map(jnp.asarray([0]))

      with self.subTest("broadcast"), self.assertRaisesRegex(
          ValueError, "Mesh axis types must all be either auto or manual"
      ):

        @drjax_program(placements={"xy": 1})
        def test_broadcast(x):
          return api.broadcast(x)

        test_broadcast(jnp.asarray([0]))


# This allows us to test sharding behavior across multiple devices.
def setUpModule():
  chex.set_n_cpu_devices(8)


if __name__ == "__main__":
  absltest.main()
