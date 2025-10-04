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

from absl.testing import absltest
from absl.testing import parameterized
import chex
from drjax._src import impls
import jax
from jax import numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec  # pylint: disable=g-multiple-import
import numpy as np


def create_mesh(
    axis_type: AxisType,
) -> jax.sharding.Mesh:
  return jax.sharding.Mesh(
      np.asarray(jax.devices()).reshape(2, 4),
      axis_names=('clients', 'data'),
      axis_types=(axis_type, axis_type),
  )


class ImplsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._placements = {'clients': 100}
    self._sequence_length = 10

  def test_broadcast_on_float(self):
    comp_factory = impls.PlacedComputations(
        placements_to_n_elements=self._placements,
    )
    actual_output = comp_factory.broadcast_to_placement(0.0, 'clients')
    expected_output = jnp.zeros(shape=[100])
    chex.assert_trees_all_equal(actual_output, expected_output)

  def test_runs_temp_sens_example(self):
    comp_factory = impls.PlacedComputations(
        placements_to_n_elements=self._placements,
    )
    def _one_if_over(x, y):
      return jax.lax.cond(x > y, lambda: 1.0, lambda: 0.0)

    def temp_sens_example(m, t):
      t_at_c = comp_factory.broadcast_to_placement(t, 'clients')
      total_over = comp_factory.map_to_placement(
          _one_if_over, (m, t_at_c), 'clients'
      )
      return comp_factory.mean_from_placement(total_over)

    measurements = jnp.arange(self._placements['clients'])

    self.assertEqual(
        temp_sens_example(measurements, jnp.median(measurements)), 0.5
    )

  @parameterized.product(
      mesh_axes_type=[AxisType.Auto, AxisType.Explicit],
  )
  def test_runs_grad_training(self, mesh_axes_type):
    mesh = create_mesh(mesh_axes_type)
    with jax.set_mesh(mesh):
      comp_factory = impls.PlacedComputations(
          placements_to_n_elements=self._placements,
      )

      def update(model, x):
        return jax.value_and_grad(lambda m, x: jnp.sum(m * jnp.square(x)))(
            model, x
        )

      def test_training(model, data):
        model_at_clients = comp_factory.broadcast_to_placement(model, 'clients')
        grads, _ = comp_factory.map_to_placement(
            update, (model_at_clients, data), 'clients'
        )
        return comp_factory.mean_from_placement(grads)

      clients_data = jax.device_put(
          jnp.ones(shape=(self._placements['clients'],), dtype=jnp.float32),
          device=NamedSharding(mesh, PartitionSpec('clients')),
      )
      model = jax.device_put([0.0], device=NamedSharding(mesh, PartitionSpec()))
      self.assertEqual(jax.jit(test_training)(model, clients_data), 0.0)


# This allows us to test sharding behavior across multiple devices.
def setUpModule():
  chex.set_n_cpu_devices(8)


if __name__ == '__main__':
  absltest.main()
