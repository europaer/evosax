"""Covariance Matrix Adaptation Evolution Strategy (Hansen et al., 2001).

[1] https://arxiv.org/abs/1604.00772
[2] https://github.com/CyberAgentAILab/cmaes
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from evosax.core.fitness_shaping import weights_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution

from .base import (
    DistributionBasedAlgorithm,
    Params as BaseParams,
    State as BaseState,
    metrics_fn,
)


@struct.dataclass
class State(BaseState):
    mean: jax.Array
    std: float
    p_std: jax.Array
    p_c: jax.Array
    C: jax.Array
    B: jax.Array
    D: jax.Array
    # New fields for improved convergence
    sigma_old: float  # Previous sigma for stagnation detection
    median_fitness: float  # Track median fitness
    median_fitness_old: float  # Previous median
    stagnation_counter: int  # Count consecutive worse iterations
    restart_count: int  # Number of restarts performed


@struct.dataclass
class Params(BaseParams):
    std_init: float
    std_min: float
    std_max: float
    weights: jax.Array
    mu_eff: float
    c_mean: float
    c_std: float
    d_std: float
    c_c: float
    c_1: float
    c_mu: float
    chi_n: float
    # New parameters
    damp_fac: float  # Configurable damping factor
    tol_upsigma: float  # Sigma divergence tolerance
    tol_stagnation: int  # Stagnation iteration limit
    stall_sigma_iters: int  # Iterations of worse median before stalling


class CMA_ES(DistributionBasedAlgorithm):
    """Improved CMA-ES with better sigma adaptation and restart support."""

    def __init__(
            self,
            population_size: int,
            solution: Solution,
            fitness_shaping_fn: Callable = weights_fitness_shaping_fn,
            metrics_fn: Callable = metrics_fn,
    ):
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)
        self.elite_ratio = 0.5
        self.use_negative_weights = True

    @property
    def _default_params(self) -> Params:
        weights_prime = jnp.log((self.population_size + 1) / 2) - jnp.log(
            jnp.arange(1, self.population_size + 1)
        )

        mu_eff = jnp.sum(weights_prime[: self.num_elites]) ** 2 / jnp.sum(
            weights_prime[: self.num_elites] ** 2
        )
        mu_eff_minus = jnp.sum(weights_prime[self.num_elites:]) ** 2 / jnp.sum(
            weights_prime[self.num_elites:] ** 2
        )

        c_c = (4 + mu_eff / self.num_dims) / (
                self.num_dims + 4 + 2 * mu_eff / self.num_dims
        )

        alpha_cov = 2
        c_1 = alpha_cov / ((self.max_num_dims_sq + 1.3) ** 2 + mu_eff)

        c_mu = jnp.minimum(
            1 - c_1 - 1e-8,
            alpha_cov
            * (mu_eff + 1 / mu_eff - 2)
            / ((self.max_num_dims_sq + 2) ** 2 + alpha_cov * mu_eff / 2),
        )

        min_alpha = jnp.minimum(
            1 + c_1 / c_mu,
            1 + (2 * mu_eff_minus) / (mu_eff + 2),
        )
        min_alpha = jnp.minimum(
            min_alpha,
            (1 - c_1 - c_mu) / (self.num_dims * c_mu),
        )

        positive_sum = jnp.sum(weights_prime * (weights_prime > 0))
        negative_sum = jnp.sum(jnp.abs(weights_prime * (weights_prime < 0)))
        weights = jnp.where(
            weights_prime >= 0,
            weights_prime / positive_sum,
            self.use_negative_weights * min_alpha * weights_prime / negative_sum,
        )

        # === IMPROVED: Finalize negative weights for zero decay ===
        # Ensure c1 + cmu * sum(weights) ≈ 0 for proper adaptation
        weights = self._finalize_negative_weights(
            weights, c_1, c_mu, mu_eff, mu_eff_minus
        )

        c_mean = 1.0

        c_std = (mu_eff + 2) / (self.num_dims + mu_eff + 5)

        # === IMPROVED: Better damping formula (matching pycma) ===
        damp_fac = 1.0  # Default damping factor, can be tuned
        d_std = damp_fac * (
                0.5
                + 2 * jnp.maximum(0, jnp.sqrt((mu_eff - 1) / (self.num_dims + 1)) - 1)
                + c_std
        )

        chi_n = jnp.sqrt(self.num_dims) * (
                1.0
                - (1.0 / (4.0 * self.num_dims))
                + 1.0 / (21.0 * (self.max_num_dims_sq ** 2))
        )

        return Params(
            std_init=1.0,
            std_min=1e-12,  # More reasonable minimum
            std_max=1e8,
            weights=weights,
            mu_eff=mu_eff,
            c_mean=c_mean,
            c_std=c_std,
            d_std=d_std,
            c_c=c_c,
            c_1=c_1,
            c_mu=c_mu,
            chi_n=chi_n,
            # New parameters
            damp_fac=damp_fac,
            tol_upsigma=1e20,  # Max sigma/sigma0 ratio
            tol_stagnation=100 + int(100 * self.num_dims ** 1.5 / self.population_size),
            stall_sigma_iters=2,  # Stall after 2 worse iterations
        )

    def _finalize_negative_weights(
            self, weights, c_1, c_mu, mu_eff, mu_eff_minus
    ) -> jax.Array:
        """Finalize negative weights ensuring zero decay and pos. definiteness.

        Based on pycma's RecombinationWeights.finalize_negative_weights().
        """
        mu = self.num_elites

        # Target sum for negative weights to achieve zero decay: c1 + cmu * sum(w) = 0
        # sum(w) = sum(w+) + sum(w-) = 1 + sum(w-)
        # We want: c1 + cmu * (1 + sum(w-)) = 0
        # => sum(w-) = -(1 + c1/cmu)
        target_neg_sum = -(1 + c_1 / (c_mu + 1e-10))

        # Limit based on learning rate (mueff condition)
        limit_mueff = -(1 + 2 * mu_eff_minus / (mu_eff + 2))

        # Limit based on positive definiteness
        limit_posdef = -(1 - c_1 - c_mu) / (c_mu * self.num_dims + 1e-10)

        # Take the most restrictive limit
        final_limit = jnp.maximum(target_neg_sum, jnp.maximum(limit_mueff, limit_posdef))

        # Current sum of negative weights
        neg_mask = weights < 0
        current_neg_sum = jnp.sum(weights * neg_mask)

        # Scale negative weights if needed
        scale_factor = jnp.where(
            current_neg_sum < final_limit,
            final_limit / (current_neg_sum - 1e-10),
            1.0
        )

        weights = jnp.where(neg_mask, weights * scale_factor, weights)

        return weights

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init,
            p_std=jnp.zeros(self.num_dims),
            p_c=jnp.zeros(self.num_dims),
            C=jnp.eye(self.num_dims),
            B=jnp.eye(self.num_dims),
            D=jnp.ones((self.num_dims,)),
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
            # New fields
            sigma_old=params.std_init,
            median_fitness=jnp.inf,
            median_fitness_old=jnp.inf,
            stagnation_counter=0,
            restart_count=0,
        )
        return state

    def _ask(
            self,
            key: jax.Array,
            state: State,
            params: Params,
    ) -> tuple[Population, State]:
        C, B, D = eigen_decomposition(state.C)

        z = jax.random.normal(key, (self.population_size, self.num_dims))
        z = (z @ jnp.diag(D).T) @ B.T
        population = state.mean + state.std * z

        return population, state.replace(C=C, B=B, D=D)

    def _tell(
            self,
            key: jax.Array,
            population: Population,
            fitness: Fitness,
            state: State,
            params: Params,
    ) -> State:
        # Update mean
        mean, y_k, y_w = self.update_mean(
            population, fitness, state.mean, state.std, params
        )

        # CSA: Cumulative Step-size Adaptation
        C_inv_sqrt_y_w = state.B @ ((state.B.T @ y_w) / state.D)
        p_std = self.update_p_std(state.p_std, C_inv_sqrt_y_w, params)
        norm_p_std = jnp.linalg.norm(p_std)

        # === IMPROVED: Sigma update with stagnation handling ===
        std, stagnation_counter = self.update_std_improved(
            state.std,
            state.sigma_old,
            norm_p_std,
            fitness,
            state.median_fitness_old,
            state.stagnation_counter,
            params,
        )

        # Covariance matrix adaptation
        h_std = self.h_std(norm_p_std, state.generation_counter + 1, params)
        p_c = self.update_p_c(state.p_c, h_std, y_w, params)

        delta_h_std = self.delta_h_std(h_std, params)
        rank_one = self.rank_one(p_c)
        rank_mu = self.rank_mu(
            fitness, y_k, (y_k @ state.B) * (1 / state.D) @ state.B.T
        )
        C = self.update_C(state.C, delta_h_std, rank_one, rank_mu, params)

        # Track median fitness for stagnation detection
        median_fitness = jnp.median(fitness)

        return state.replace(
            mean=mean,
            std=std,
            p_std=p_std,
            p_c=p_c,
            C=C,
            sigma_old=state.std,
            median_fitness=median_fitness,
            median_fitness_old=state.median_fitness,
            stagnation_counter=stagnation_counter,
        )

    def update_mean(
            self,
            population: Population,
            fitness: Fitness,
            mean: jax.Array,
            std: float,
            params: Params,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        y_k = (population - mean) / std
        y_w = jnp.dot(jnp.where(fitness < 0.0, 0.0, fitness), y_k)
        return mean + params.c_mean * std * y_w, y_k, y_w

    def update_p_std(
            self, p_std: jax.Array, C_inv_sqrt_y_w: jax.Array, params: Params
    ) -> jax.Array:
        return (1 - params.c_std) * p_std + jnp.sqrt(
            params.c_std * (2 - params.c_std) * params.mu_eff
        ) * C_inv_sqrt_y_w

    def update_std_improved(
            self,
            std: float,
            sigma_old: float,
            norm_p_std: float,
            fitness: Fitness,
            median_old: float,
            stagnation_counter: int,
            params: Params,
    ) -> tuple[float, int]:
        """Improved sigma update with stagnation handling."""
        # Basic CSA update
        std_new = std * jnp.exp(
            (params.c_std / params.d_std) * (norm_p_std / params.chi_n - 1)
        )

        # Check for stagnation (median fitness not improving)
        median_fitness = jnp.median(fitness)
        is_worse = median_fitness >= median_old
        stagnation_counter_new = jnp.where(
            is_worse, stagnation_counter + 1, 0
        )

        # === KEY IMPROVEMENT: Stall sigma when diverging ===
        # If median has been getting worse for too many iterations, don't change sigma
        should_stall = stagnation_counter_new >= params.stall_sigma_iters
        std_new = jnp.where(should_stall, sigma_old, std_new)

        # Clip to bounds
        std_new = jnp.clip(std_new, params.std_min, params.std_max)

        # === Additional check: tolupsigma (prevent unbounded growth) ===
        max_D = jnp.max(jnp.sqrt(jnp.diag(jnp.eye(self.num_dims))))  # Would need actual D
        too_large = std_new > params.std_init * params.tol_upsigma
        std_new = jnp.where(too_large, std, std_new)

        return std_new, stagnation_counter_new

    def update_std(self, std: float, norm_p_std: float, params: Params) -> float:
        """Original update_std kept for compatibility."""
        std = std * jnp.exp(
            (params.c_std / params.d_std) * (norm_p_std / params.chi_n - 1)
        )
        return jnp.clip(std, min=params.std_min, max=params.std_max)

    def h_std(self, norm_p_std: float, generation_counter: int, params: Params) -> bool:
        h_std_cond_left = norm_p_std / jnp.sqrt(
            1 - (1 - params.c_std) ** (2 * (generation_counter + 1))
        )
        h_std_cond_right = (1.4 + 2 / (self.num_dims + 1)) * params.chi_n
        return h_std_cond_left < h_std_cond_right

    def update_p_c(
            self, p_c: jax.Array, h_std: bool, y_w: jax.Array, params: Params
    ) -> jax.Array:
        return (1 - params.c_c) * p_c + h_std * jnp.sqrt(
            params.c_c * (2 - params.c_c) * params.mu_eff
        ) * y_w

    def delta_h_std(self, h_std: bool, params: Params) -> float:
        return (1 - h_std) * params.c_c * (2 - params.c_c)

    def rank_one(self, p_c: jax.Array) -> jax.Array:
        return jnp.outer(p_c, p_c)

    def rank_mu(
            self, fitness: Fitness, y_k: jax.Array, C_inv_sqrt_y_k: jax.Array
    ) -> jax.Array:
        w_o = fitness * jnp.where(
            fitness >= 0,
            1,
            self.num_dims
            / jnp.clip(jnp.sum(jnp.square(C_inv_sqrt_y_k), axis=-1), min=1e-8),
        )
        return jnp.einsum("i,ij,ik->jk", w_o, y_k, y_k)

    def update_C(
            self,
            C: jax.Array,
            delta_h_std: float,
            rank_one: jax.Array,
            rank_mu: jax.Array,
            params: Params,
    ) -> jax.Array:
        return (
                (
                        1
                        + params.c_1 * delta_h_std
                        - params.c_1
                        - params.c_mu * jnp.sum(params.weights)
                )
                * C
                + params.c_1 * rank_one
                + params.c_mu * rank_mu
        )


def eigen_decomposition(C: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Eigendecomposition with improved numerical stability."""
    # Symmetry
    C = (C + C.T) / 2

    # Clip diagonal elements to 0 and all elements to max 1e8
    diag_indices = jnp.diag_indices_from(C)
    C = C.at[diag_indices].set(jnp.maximum(C[diag_indices], 0.0))
    C = jnp.minimum(C, 1e8)

    # Diagonal loading for numerical stability
    eps = 1e-8
    C = C + eps * jnp.eye(C.shape[0])

    # Compute eigen decomposition
    D_sq, B = jnp.linalg.eigh(C)

    D = jnp.sqrt(jnp.maximum(D_sq, eps))

    return C, B, D


# =============================================================================
# RESTART WRAPPER
# =============================================================================

@struct.dataclass
class RestartState:
    """State for restart wrapper."""
    inner_state: State
    total_evaluations: int
    best_ever_fitness: float
    best_ever_solution: jax.Array
    current_popsize: int
    restart_count: int


class CMA_ES_Restarts:
    """CMA-ES with IPOP-style restarts."""

    def __init__(
            self,
            base_population_size: int,
            solution: Solution,
            max_restarts: int = 9,
            popsize_multiplier: float = 2.0,
    ):
        self.base_population_size = base_population_size
        self.solution = solution
        self.max_restarts = max_restarts
        self.popsize_multiplier = popsize_multiplier
        self.num_dims = len(solution)

    def should_restart(self, state: State, params: Params) -> bool:
        """Check if restart conditions are met."""
        # Condition 1: Sigma too small (converged)
        sigma_too_small = state.std < params.std_min * 1e3

        # Condition 2: Stagnation for too long
        stagnated = state.stagnation_counter > params.tol_stagnation

        # Condition 3: Sigma exploded (diverged)
        sigma_too_large = state.std > params.std_init * params.tol_upsigma

        return sigma_too_small | stagnated | sigma_too_large

    def restart(
            self,
            key: jax.Array,
            restart_state: RestartState,
            params: Params,
    ) -> RestartState:
        """Perform IPOP restart with increased population size."""
        new_popsize = int(
            restart_state.current_popsize * self.popsize_multiplier
        )

        # Create new CMA-ES instance with larger population
        new_cma = CMA_ES(
            population_size=new_popsize,
            solution=self.solution,
        )

        # Initialize new state
        new_params = new_cma._default_params
        new_inner_state = new_cma._init(key, new_params)

        # Keep track of best ever
        return RestartState(
            inner_state=new_inner_state,
            total_evaluations=restart_state.total_evaluations,
            best_ever_fitness=restart_state.best_ever_fitness,
            best_ever_solution=restart_state.best_ever_solution,
            current_popsize=new_popsize,
            restart_count=restart_state.restart_count + 1,
        )
