"""Submission file for a NAdamW optimizer with warmup+cosine LR in Jax."""

from typing import Any, Callable, NamedTuple, Optional, Union

import chex
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

def jax_cosine_warmup(step_hint: int, hyperparameters):
  # Create learning rate schedule.
  warmup_steps = step_hint * hyperparameters.warmup_steps
  warmup_fn = optax.linear_schedule(
      init_value=0.,
      end_value=hyperparameters.learning_rate,
      transition_steps=warmup_steps)
  cosine_steps = max(step_hint - warmup_steps, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=hyperparameters.learning_rate, decay_steps=cosine_steps, alpha=hyperparameters.alpha)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[warmup_steps])
  return schedule_fn

def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates an AdamW optimizer and a learning rate schedule."""
  del model_params
  del model_state
  del rng

  target_setting_step_hint = int(0.75 * workload.step_hint)
  lr_schedule_fn = jax_cosine_warmup(target_setting_step_hint,
                                                   hyperparameters)

  # Create optimizer.
  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)
  epsilon = (
      hyperparameters.epsilon if hasattr(hyperparameters, 'epsilon') else 1e-8)
  opt_init_fn, opt_update_fn = optax.adam(
      learning_rate=lr_schedule_fn,
      b1=hyperparameters.beta1,
      b2=hyperparameters.beta2,
      eps=epsilon)
  optimizer_state = opt_init_fn(params_zeros_like)

  return jax_utils.replicate(optimizer_state), opt_update_fn