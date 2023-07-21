"""Template submission module.

See https://github.com/mlcommons/algorithmic-efficiency/blob/main/RULES.md#allowed-submissions 
and https://github.com/mlcommons/algorithmic-efficiency/blob/main/RULES.md#disallowed-submissions
for guidelines. 
"""
import functools
import dataclasses
from typing import List, Dict, Any, Mapping, Optional, Tuple

import jax
import chex
from jax import lax
import jax.numpy as jnp
import optax
from flax import jax_utils

from algorithmic_efficiency import spec
from learned_optimization import tree_utils
from learned_optimization.research.general_lopt import prefab
from learned_optimization.research.general_lopt.prefab import LearnedOptimizer
from learned_optimization.research.general_lopt import pretrained_optimizers

from reference_algorithms.target_setting_algorithms.data_selection import \
    data_selection  # pylint: disable=unused-import

_GRAD_CLIP_EPS = 1e-6
TOTAL_STEPS = 0

@functools.partial(
    jax.pmap,
    axis_name='batch',
    in_axes=(None, None, 0, 0, 0, 0, 0, None, None),
    static_broadcasted_argnums=(0, 1),
    donate_argnums=(2, 3, 4))
def pmapped_train_step(workload,
                       opt_update_fn,
                       model_state,
                       optimizer_state,
                       current_param_container,
                       batch,
                       rng,
                       grad_clip,
                       label_smoothing):

  def _loss_fn(params):
    """Loss function used for training."""
    logits, new_model_state = workload.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.TRAIN,
        rng,
        update_batch_norm=True)
    loss_dict = workload.loss_fn(
        label_batch=batch['targets'],
        logits_batch=logits,
        mask_batch=batch.get('weights'),
        label_smoothing=label_smoothing)
    summed_loss = loss_dict['summed']
    n_valid_examples = loss_dict['n_valid_examples']
    return summed_loss, (n_valid_examples, new_model_state)

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (summed_loss, (n_valid_examples, new_model_state)), grad = grad_fn(
      current_param_container)
  # Get correct global mean loss and grad.
  (summed_loss, n_valid_examples, grad) = lax.psum(
      (summed_loss, n_valid_examples, grad), axis_name='batch')
  loss = summed_loss / n_valid_examples
  grad = jax.tree_map(lambda x: x / n_valid_examples, grad)

  grad_norm = jnp.sqrt(
      sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))

  if grad_clip is not None:
    grad_scaling_factor = grad_clip / (grad_norm + _GRAD_CLIP_EPS)
    grad_scaling_factor = jax.lax.clamp(min=0.0, x=grad_scaling_factor, max=1.0)
    grad = jax.tree_map(lambda x: x * grad_scaling_factor, grad)

  updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container, extra_args={"loss": loss})
  updated_params = optax.apply_updates(current_param_container, updates)
  return new_optimizer_state, updated_params, new_model_state, loss, grad_norm

def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates a VeLO optimizer state.
    Returns:
     optimizer state
     optimizer_update_fn
    """
  tx = prefab.optax_lopt(TOTAL_STEPS, max_training_steps=200_000)
  opt_state = tx.init(jax_utils.unreplicate(model_params))
  return jax_utils.replicate(opt_state), tx.update

def update_params(workload: spec.Workload,
                  current_param_container: spec.ParameterContainer,
                  current_params_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState,
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del eval_results

  optimizer_state, opt_update_fn = optimizer_state
  per_device_rngs = jax.random.split(rng, jax.local_device_count())
  if hasattr(hyperparameters, 'label_smoothing'):
    label_smoothing = hyperparameters.label_smoothing
  else:
    label_smoothing = 0.0
  if hasattr(hyperparameters, 'grad_clip'):
    grad_clip = hyperparameters.grad_clip
  else:
    grad_clip = None
  new_optimizer_state, new_params, new_model_state, loss, grad_norm = pmapped_train_step( # pylint: disable=line-too-long
      workload, opt_update_fn, model_state, optimizer_state,
      current_param_container, batch, per_device_rngs, grad_clip,
      label_smoothing)

  # Log loss, grad_norm.
  if global_step % 100 == 0 and workload.metrics_logger is not None:
    workload.metrics_logger.append_scalar_metrics(
        {
            'loss': loss[0],
            'grad_norm': grad_norm[0],
        }, global_step)
  return (new_optimizer_state, opt_update_fn), new_params, new_model_state

def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb':
    TOTAL_STEPS = 7672
    return 262_144
  elif workload_name == 'fastmri':
    TOTAL_STEPS = 38021
    return 32
  elif workload_name == 'imagenet_resnet':
    TOTAL_STEPS = 158845
    return 1024
  elif workload_name == 'ogbg':
    TOTAL_STEPS = 79460
    return 512
  else:
    raise ValueError(f'Unsupported workload name: {workload_name}.')