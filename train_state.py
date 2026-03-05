"""Custom TrainState with EMA parameters and RNG management."""

from __future__ import annotations

import functools
from typing import Any, Callable

import flax.struct
import jax
import jax.numpy as jnp
import optax

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class TrainState(flax.struct.PyTreeNode):
    """Training state holding params, EMA params, optimizer, and RNG."""

    step: int
    apply_fn: Callable = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    ema_params: Any
    tx: Any = nonpytree_field()
    opt_state: Any
    rng: jax.Array

    @classmethod
    def create(cls, *, model_def, params, tx, rng):
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            ema_params=jax.tree.map(jnp.copy, params),
            tx=tx,
            opt_state=opt_state,
            rng=rng,
        )

    def apply_gradients(self, grads):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        )

    def update_ema(self, decay: float = 0.999):
        new_ema = jax.tree.map(
            lambda e, p: decay * e + (1.0 - decay) * p,
            self.ema_params,
            self.params,
        )
        return self.replace(ema_params=new_ema)

    def next_rng(self):
        """Advance RNG and return (new_state, step_rng)."""
        rng, step_rng = jax.random.split(self.rng)
        return self.replace(rng=rng), step_rng
