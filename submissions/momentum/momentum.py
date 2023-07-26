"""Submission file for a SGD with HeavyBall momentum optimizer in Jax."""

from typing import Callable

from flax import jax_utils
import jax
import jax.numpy as jnp
import optax

from algorithmic_efficiency import spec
from reference_algorithms.target_setting_algorithms.data_selection import \
    data_selection  # pylint: disable=unused-import
from reference_algorithms.target_setting_algorithms.get_batch_size import \
    get_batch_size  # pylint: disable=unused-import
from reference_algorithms.target_setting_algorithms.jax_submission_base import \
    update_params  # pylint: disable=unused-import


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates a Nesterov optimizer and a learning rate schedule."""
  del model_params
  del model_state
  del rng

  # Create learning rate schedule.
  if hyperparameters.schedule == 1:
    lr_schedule_fn = create_lr_schedule_fn(workload.step_hint,
                                          hyperparameters)
  elif hyperparameters.schedule == 0:
    lr_schedule_fn = hyperparameters.learning_rate
  else:
    raise Exception(f"Unknown value {hyperparameters.schedule} for schedule hyperparameters")

  # Create optimizer.
  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)
  opt_init_fn, opt_update_fn = sgd(
      learning_rate=lr_schedule_fn,
      weight_decay=hyperparameters.weight_decay,
      momentum=hyperparameters.beta1,
      nesterov=False)
  optimizer_state = opt_init_fn(params_zeros_like)

  return jax_utils.replicate(optimizer_state), opt_update_fn


def create_lr_schedule_fn(
    step_hint: int,
    hyperparameters: spec.Hyperparameters) -> Callable[[int], float]:
  warmup_steps = step_hint * hyperparameters.warmup_steps
  warmup_fn = optax.linear_schedule(
      init_value=0.,
      end_value=hyperparameters.learning_rate,
      transition_steps=warmup_steps)
  decay_steps = step_hint - warmup_steps
  polynomial_schedule_fn = optax.polynomial_schedule(
      init_value=hyperparameters.learning_rate,
      end_value=hyperparameters.end_value,
      power=1,
      transition_steps=decay_steps)
  lr_schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, polynomial_schedule_fn],
      boundaries=[warmup_steps])
  return lr_schedule_fn


# Forked from github.com/google/init2winit/blob/master/init2winit/ (cont. below)
# optimizer_lib/optimizers.py.
def sgd(learning_rate, weight_decay, momentum=None, nesterov=False):
  r"""A customizable gradient descent optimizer.

  NOTE: We apply weight decay **before** computing the momentum update.
  This is equivalent to applying WD after for heavy-ball momentum,
  but slightly different when using Nesterov acceleration. This is the same as
  how the Flax optimizers handle weight decay
  https://flax.readthedocs.io/en/latest/_modules/flax/optim/momentum.html.

  Args:
    learning_rate: The learning rate. Expected as the positive learning rate,
      for example `\alpha` in `w -= \alpha * u` (as opposed to `\alpha`).
    weight_decay: The weight decay hyperparameter.
    momentum: The momentum hyperparameter.
    nesterov: Whether or not to use Nesterov momentum.
  Returns:
    An optax gradient transformation that applies weight decay and then one of a
    {SGD, Momentum, Nesterov} update.
  """
  return optax.chain(
      optax.add_decayed_weights(weight_decay),
      optax.sgd(
          learning_rate=learning_rate, momentum=momentum, nesterov=nesterov))
