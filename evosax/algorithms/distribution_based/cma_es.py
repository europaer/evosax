"""Covariance Matrix Adaptation Evolution Strategy (Hansen et al., 2001).

Enhanced version with pycma-compliant stability features:
- Proper negative weights finalization (sequential clamping)
- Condition number limiting
- Trace normalization
- CSA path clipping
- Asymmetric sigma damping

[1] https://arxiv.org/abs/1604.00772
[2] https://github.com/CyberAgentAILab/cmaes
"""

from collections.abc import Callable
from typing import Literal

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
    # Track eigendecomposition for lazy updates
    eigenupdated_iter: int


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
    # Sigma adaptation parameters
    max_delta_log_sigma: float
    dampdown: float
    # Stability parameters
    condition_limit: float
    # CSA path clipping: [lower_factor, upper_factor] relative to sqrt(N)
    csa_clip_length_lower: float
    csa_clip_length_upper: float
    # Trace normalization: 'none', 'arithmetic', 'geometric'
    trace_normalization: int  # 0=none, 1=arithmetic, 2=geometric


class CMA_ES(DistributionBasedAlgorithm):
    """CMA-ES with robust sigma adaptation matching pycma behavior.
    
    Enhanced stability features:
    - Sequential negative weights finalization (pycma-compliant)
    - Condition number limiting to prevent numerical instability
    - Optional trace normalization to bound covariance growth
    - CSA path length clipping for controlled sigma updates
    - Asymmetric damping for sigma decrease vs increase
    """

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        fitness_shaping_fn: Callable = weights_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
        # Configuration options
        use_negative_weights: bool = True,
        condition_limit: float = 1e14,
        trace_normalization: Literal['none', 'arithmetic', 'geometric'] = 'none',
        csa_clip_length: tuple[float, float] | None = None,
    ):
        """Initialize CMA-ES.
        
        Args:
            population_size: Number of samples per generation
            solution: Solution specification
            fitness_shaping_fn: Function for fitness shaping
            metrics_fn: Function for computing metrics
            use_negative_weights: Whether to use active CMA (negative weights)
            condition_limit: Maximum condition number for C (default 1e14)
            trace_normalization: How to normalize trace ('none', 'arithmetic', 'geometric')
            csa_clip_length: Optional (lower, upper) factors for path length clipping
                            relative to sqrt(N). E.g., (-1, 1) clips to [sqrt(N)-N/(N+2), sqrt(N)+N/(N+2)]
        """
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        self.elite_ratio = 0.5
        self.use_negative_weights = use_negative_weights
        self.condition_limit = condition_limit
        
        # Convert trace_normalization to int for JAX compatibility
        self._trace_norm_map = {'none': 0, 'arithmetic': 1, 'geometric': 2}
        self.trace_normalization = self._trace_norm_map.get(trace_normalization, 0)
        
        # CSA path clipping
        self.csa_clip_length = csa_clip_length

    @property
    def _default_params(self) -> Params:
        # === WEIGHTS CALCULATION (Eq. 48) ===
        weights_prime = jnp.log((self.population_size + 1) / 2) - jnp.log(
            jnp.arange(1, self.population_size + 1)
        )

        # Variance effective selection mass (Eq. 8)
        mu_eff = jnp.sum(weights_prime[: self.num_elites]) ** 2 / jnp.sum(
            weights_prime[: self.num_elites] ** 2
        )
        mu_eff_minus = jnp.sum(weights_prime[self.num_elites:]) ** 2 / (
            jnp.sum(weights_prime[self.num_elites:] ** 2) + 1e-10
        )

        # === LEARNING RATES ===
        # Cumulation for C (Eq. 56)
        c_c = (4 + mu_eff / self.num_dims) / (
            self.num_dims + 4 + 2 * mu_eff / self.num_dims
        )

        # Rank-one update learning rate (Eq. 57)
        alpha_cov = 2
        c_1 = alpha_cov / ((self.max_num_dims_sq + 1.3) ** 2 + mu_eff)

        # Rank-mu update learning rate (Eq. 58)
        c_mu = jnp.minimum(
            1 - c_1 - 1e-8,
            alpha_cov
            * (mu_eff + 1 / mu_eff - 2)
            / ((self.max_num_dims_sq + 2) ** 2 + alpha_cov * mu_eff / 2),
        )

        # === NEGATIVE WEIGHTS SETUP ===
        # Normalize positive and negative weights separately first
        positive_sum = jnp.sum(weights_prime * (weights_prime > 0))
        negative_sum = jnp.sum(jnp.abs(weights_prime * (weights_prime < 0)))
        
        weights = jnp.where(
            weights_prime >= 0,
            weights_prime / (positive_sum + 1e-10),
            weights_prime / (negative_sum + 1e-10),  # Negative weights sum to -1
        )

        # Apply pycma-style sequential finalization
        if self.use_negative_weights:
            weights = self._finalize_negative_weights_sequential(
                weights, c_1, c_mu, mu_eff, mu_eff_minus
            )
        else:
            # Zero out negative weights
            weights = jnp.where(weights < 0, 0.0, weights)

        # Mean learning rate
        c_mean = 1.0

        # === CSA PARAMETERS (Eq. 55) ===
        c_std = (mu_eff + 2) / (self.num_dims + mu_eff + 5)

        # Damping for sigma (pycma formula)
        d_std = (
            0.5
            + 2 * jnp.maximum(0, jnp.sqrt((mu_eff - 1) / (self.num_dims + 1)) - 1)
            + c_std
        )

        # Expected value of ||N(0,I)|| (Page 28)
        chi_n = jnp.sqrt(self.num_dims) * (
            1.0
            - (1.0 / (4.0 * self.num_dims))
            + 1.0 / (21.0 * (self.max_num_dims_sq ** 2))
        )

        # CSA clipping bounds
        if self.csa_clip_length is not None:
            clip_lower, clip_upper = self.csa_clip_length
        else:
            clip_lower, clip_upper = -jnp.inf, jnp.inf

        return Params(
            std_init=1.0,
            std_min=1e-16,
            std_max=1e8,  # pycma default
            weights=weights,
            mu_eff=mu_eff,
            c_mean=c_mean,
            c_std=c_std,
            d_std=d_std,
            c_c=c_c,
            c_1=c_1,
            c_mu=c_mu,
            chi_n=chi_n,
            # Sigma control
            max_delta_log_sigma=1.0,
            dampdown=1.0,  # Set to 1.0 for pycma default; increase for more stability
            # Stability
            condition_limit=self.condition_limit,
            csa_clip_length_lower=clip_lower,
            csa_clip_length_upper=clip_upper,
            trace_normalization=self.trace_normalization,
        )

    def _finalize_negative_weights_sequential(
        self, weights: jax.Array, c_1: float, c_mu: float, 
        mu_eff: float, mu_eff_minus: float
    ) -> jax.Array:
        """
        Finalize negative weights using pycma's sequential clamping approach.
        
        This implements the three constraints in order:
        1. Set negative sum to achieve zero decay: sum(w) = -(1 + c1/cmu)
        2. Limit by mueff ratio: |sum(w-)| <= 1 + 2*mueff_minus/(mueff+2)
        3. Limit for positive definiteness: |sum(w-)| <= (1-c1-cmu)/(cmu*N)
        """
        neg_mask = weights < 0
        pos_mask = weights >= 0
        
        current_neg_sum = jnp.sum(jnp.where(neg_mask, weights, 0.0))
        
        # Step 1: Set sum to achieve zero decay (c1 + cmu * sum(w) = 0)
        # => sum(w) = -c1/cmu, but we want 1 + c1/cmu for the negative part magnitude
        target_neg_sum = -(1 + c_1 / (c_mu + 1e-10))
        
        scale1 = jnp.where(
            jnp.abs(current_neg_sum) > 1e-10,
            target_neg_sum / current_neg_sum,
            1.0
        )
        weights = jnp.where(neg_mask, weights * scale1, weights)
        current_neg_sum = current_neg_sum * scale1
        
        # Step 2: Limit by mueff ratio (only reduce, never increase magnitude)
        limit_mueff = -(1 + 2 * mu_eff_minus / (mu_eff + 2))
        scale2 = jnp.where(
            (current_neg_sum < limit_mueff) & (jnp.abs(current_neg_sum) > 1e-10),
            limit_mueff / current_neg_sum,
            1.0
        )
        scale2 = jnp.minimum(scale2, 1.0)  # Only reduce magnitude
        weights = jnp.where(neg_mask, weights * scale2, weights)
        current_neg_sum = current_neg_sum * scale2
        
        # Step 3: Limit for positive definiteness
        limit_posdef = -(1 - c_1 - c_mu) / (c_mu * self.num_dims + 1e-10)
        scale3 = jnp.where(
            (current_neg_sum < limit_posdef) & (jnp.abs(current_neg_sum) > 1e-10),
            limit_posdef / current_neg_sum,
            1.0
        )
        scale3 = jnp.minimum(scale3, 1.0)  # Only reduce magnitude
        weights = jnp.where(neg_mask, weights * scale3, weights)
        
        return weights

    def _init(self, key: jax.Array, params: Params) -> State:
        return State(
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
            eigenupdated_iter=0,
        )

    def _ask(
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> tuple[Population, State]:
        # Eigendecomposition with stability features
        C, B, D = eigen_decomposition(
            state.C, 
            condition_limit=params.condition_limit,
            trace_normalization=params.trace_normalization,
        )

        # Sample from N(0, I)
        z = jax.random.normal(key, (self.population_size, self.num_dims))
        # Transform: z -> B * D * z (equivalent to sampling from N(0, C))
        z = (z @ jnp.diag(D).T) @ B.T
        # Scale and shift
        population = state.mean + state.std * z

        return population, state.replace(
            C=C, B=B, D=D, 
            eigenupdated_iter=state.generation_counter
        )

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # === MEAN UPDATE ===
        mean, y_k, y_w = self.update_mean(
            population, fitness, state.mean, state.std, params
        )

        # === CSA (Cumulative Step-size Adaptation) ===
        # Compute C^(-1/2) * y_w
        C_inv_sqrt_y_w = state.B @ ((state.B.T @ y_w) / state.D)
        
        # Update evolution path for sigma
        p_std = self.update_p_std(state.p_std, C_inv_sqrt_y_w, params)
        
        # Optionally clip path length (pycma CSA_clip_length_value)
        p_std_clipped = self.clip_path_length(p_std, params)
        
        norm_p_std = jnp.linalg.norm(p_std_clipped)

        # === SIGMA UPDATE ===
        std = self.update_std_robust(
            state.std, norm_p_std, state.generation_counter + 1, params
        )

        # === COVARIANCE MATRIX ADAPTATION ===
        # Compute hsig (stall indicator)
        h_std = self.h_std(norm_p_std, state.generation_counter + 1, params)
        
        # Update evolution path for C
        p_c = self.update_p_c(state.p_c, h_std, y_w, params)

        # Compute update terms
        delta_h_std = self.delta_h_std(h_std, params)
        rank_one = self.rank_one(p_c)
        rank_mu = self.rank_mu(
            fitness, y_k, (y_k @ state.B) * (1 / state.D) @ state.B.T, params
        )
        
        # Update C
        C = self.update_C(state.C, delta_h_std, rank_one, rank_mu, params)

        return state.replace(
            mean=mean,
            std=std,
            p_std=p_std,  # Store unclipped for continuity
            p_c=p_c,
            C=C,
        )

    def update_mean(
        self,
        population: Population,
        fitness: Fitness,
        mean: jax.Array,
        std: float,
        params: Params,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Update distribution mean using weighted recombination."""
        y_k = (population - mean) / std  # Normalized steps
        # Weighted recombination (only positive weights for mean)
        weights_for_mean = jnp.where(fitness < 0.0, 0.0, fitness)
        y_w = jnp.dot(weights_for_mean, y_k)
        new_mean = mean + params.c_mean * std * y_w
        return new_mean, y_k, y_w

    def update_p_std(
        self, p_std: jax.Array, C_inv_sqrt_y_w: jax.Array, params: Params
    ) -> jax.Array:
        """Update evolution path for step-size control (Eq. 43)."""
        return (1 - params.c_std) * p_std + jnp.sqrt(
            params.c_std * (2 - params.c_std) * params.mu_eff
        ) * C_inv_sqrt_y_w

    def clip_path_length(
        self, p_std: jax.Array, params: Params
    ) -> jax.Array:
        """
        Clip evolution path length (pycma CSA_clip_length_value).
        
        Clipping bounds are relative to sqrt(N):
            min_len = sqrt(N) + lower_factor * N / (N + 2)
            max_len = sqrt(N) + upper_factor * N / (N + 2)
        """
        lower = params.csa_clip_length_lower
        upper = params.csa_clip_length_upper
        
        # If no clipping configured, return unchanged
        no_clip = (lower == -jnp.inf) & (upper == jnp.inf)
        
        sqrt_n = jnp.sqrt(self.num_dims)
        adjustment = self.num_dims / (self.num_dims + 2)
        
        min_len = sqrt_n + lower * adjustment
        max_len = sqrt_n + upper * adjustment
        
        current_len = jnp.linalg.norm(p_std)
        
        # Compute clipped length
        new_len = jnp.clip(current_len, min_len, max_len)
        
        # Scale path to new length (avoid division by zero)
        scale = jnp.where(
            current_len > 1e-10,
            new_len / current_len,
            1.0
        )
        
        return jnp.where(no_clip, p_std, p_std * scale)

    def update_std_robust(
        self,
        std: float,
        norm_p_std: float,
        generation: int,
        params: Params,
    ) -> float:
        """
        Robust sigma update with pycma features:
        1. Core CSA formula
        2. Asymmetric damping (optional)
        3. Log-sigma clipping
        4. Hard bounds
        """
        # Core CSA update (Eq. 44)
        log_sigma_change = (params.c_std / params.d_std) * (
            norm_p_std / params.chi_n - 1
        )
        
        # Asymmetric damping: slower decrease than increase
        # pycma has csa_dampdown_fac (default 1, but can be set higher)
        log_sigma_change = jnp.where(
            log_sigma_change < 0,
            log_sigma_change / params.dampdown,
            log_sigma_change
        )
        
        # Clip log-sigma change to prevent extreme updates
        log_sigma_change = jnp.clip(
            log_sigma_change,
            -params.max_delta_log_sigma,
            params.max_delta_log_sigma
        )
        
        # Apply update
        std_new = std * jnp.exp(log_sigma_change)
        
        # Hard bounds
        std_new = jnp.clip(std_new, params.std_min, params.std_max)
        
        return std_new

    def h_std(
        self, norm_p_std: float, generation: int, params: Params
    ) -> jax.Array:
        """
        Compute stall indicator for rank-one update (Page 28).
        
        Returns True if evolution path length is not too large,
        indicating the step-size is appropriate.
        """
        # Correct for initial bias in path
        correction = jnp.sqrt(1 - (1 - params.c_std) ** (2 * generation))
        h_std_cond_left = norm_p_std / (correction + 1e-10)
        h_std_cond_right = (1.4 + 2 / (self.num_dims + 1)) * params.chi_n
        return (h_std_cond_left < h_std_cond_right).astype(jnp.float32)

    def update_p_c(
        self, p_c: jax.Array, h_std: jax.Array, y_w: jax.Array, params: Params
    ) -> jax.Array:
        """Update evolution path for covariance matrix (Eq. 45)."""
        return (1 - params.c_c) * p_c + h_std * jnp.sqrt(
            params.c_c * (2 - params.c_c) * params.mu_eff
        ) * y_w

    def delta_h_std(self, h_std: jax.Array, params: Params) -> float:
        """Compute correction term when hsig stalls (Page 28)."""
        return (1 - h_std) * params.c_c * (2 - params.c_c)

    def rank_one(self, p_c: jax.Array) -> jax.Array:
        """Compute rank-one update term."""
        return jnp.outer(p_c, p_c)

    def rank_mu(
        self, 
        fitness: Fitness, 
        y_k: jax.Array, 
        C_inv_sqrt_y_k: jax.Array,
        params: Params,
    ) -> jax.Array:
        """
        Compute rank-mu update term (Eq. 46-47).
        
        For negative weights, vectors are normalized by their Mahalanobis norm
        to ensure positive definiteness.
        """
        # Compute Mahalanobis norms squared for each vector
        mahal_sq = jnp.sum(jnp.square(C_inv_sqrt_y_k), axis=-1)
        
        # Weight modification for negative weights (Eq. 46)
        # This ensures ||w_i * y_i||_C^2 <= N for negative weights
        w_o = fitness * jnp.where(
            fitness >= 0,
            1.0,
            self.num_dims / jnp.clip(mahal_sq, min=1e-8),
        )
        
        # Weighted outer product sum
        return jnp.einsum("i,ij,ik->jk", w_o, y_k, y_k)

    def update_C(
        self,
        C: jax.Array,
        delta_h_std: float,
        rank_one: jax.Array,
        rank_mu: jax.Array,
        params: Params,
    ) -> jax.Array:
        """Update covariance matrix (Eq. 47)."""
        # Compute decay factor
        c_decay = (
            1
            + params.c_1 * delta_h_std
            - params.c_1
            - params.c_mu * jnp.sum(params.weights)
        )
        
        # Ensure decay doesn't go negative (numerical safety)
        c_decay = jnp.maximum(c_decay, 0.0)
        
        # Update
        C_new = c_decay * C + params.c_1 * rank_one + params.c_mu * rank_mu
        
        return C_new


def eigen_decomposition(
    C: jax.Array,
    condition_limit: float = 1e14,
    trace_normalization: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Eigendecomposition with pycma-style stability features.
    
    Args:
        C: Covariance matrix
        condition_limit: Maximum condition number (eigenvalue ratio)
        trace_normalization: 0=none, 1=arithmetic, 2=geometric
    
    Returns:
        C: Processed covariance matrix
        B: Eigenvector matrix (columns are eigenvectors)
        D: Square roots of eigenvalues (standard deviations along principal axes)
    """
    N = C.shape[0]
    
    # === SYMMETRIZE ===
    C = (C + C.T) / 2
    
    # === ENSURE NON-NEGATIVE DIAGONAL ===
    diag_indices = jnp.diag_indices_from(C)
    C = C.at[diag_indices].set(jnp.maximum(C[diag_indices], 0.0))
    
    # === CLIP EXTREME VALUES ===
    C = jnp.clip(C, -1e14, 1e14)
    
    # === DIAGONAL LOADING FOR NUMERICAL STABILITY ===
    eps = 1e-14
    C = C + eps * jnp.eye(N)
    
    # === EIGENDECOMPOSITION ===
    D_sq, B = jnp.linalg.eigh(C)
    
    # Ensure positive eigenvalues
    D_sq = jnp.maximum(D_sq, eps)
    
    # === CONDITION NUMBER LIMITING ===
    # Add eps to small eigenvalues to limit condition number
    # Derived from: limit = (max_eig + eps) / (min_eig + eps)
    # => eps = (max_eig - limit * min_eig) / (limit - 1)
    max_eig = D_sq[-1]  # Eigenvalues are sorted ascending
    min_eig = D_sq[0]
    current_cond = max_eig / (min_eig + 1e-30)
    
    eps_cond = jnp.where(
        current_cond > condition_limit,
        (max_eig - condition_limit * min_eig) / (condition_limit - 1),
        0.0
    )
    eps_cond = jnp.maximum(eps_cond, 0.0)
    
    D_sq = D_sq + eps_cond
    
    # Update C to reflect condition limiting
    C = C + eps_cond * jnp.eye(N)
    
    # === TRACE NORMALIZATION ===
    # 0 = none, 1 = arithmetic (trace/N), 2 = geometric (det^(1/N))
    trace_norm_factor = jnp.where(
        trace_normalization == 1,
        N / jnp.sum(D_sq),  # Arithmetic: trace = N
        jnp.where(
            trace_normalization == 2,
            jnp.exp(-jnp.mean(jnp.log(D_sq))),  # Geometric: det = 1
            1.0  # None
        )
    )
    
    D_sq = D_sq * trace_norm_factor
    C = C * trace_norm_factor
    
    # === COMPUTE STANDARD DEVIATIONS ===
    D = jnp.sqrt(D_sq)
    
    return C, B, D


# === UTILITY FUNCTIONS ===

def get_cma_es_with_defaults(
    population_size: int,
    solution: Solution,
    stability_level: str = 'standard',
) -> CMA_ES:
    """
    Factory function to create CMA-ES with preset stability configurations.
    
    Args:
        population_size: Number of samples per generation
        solution: Solution specification
        stability_level: One of 'minimal', 'standard', 'robust', 'maximum'
    
    Returns:
        Configured CMA_ES instance
    """
    configs = {
        'minimal': {
            'condition_limit': 1e14,
            'trace_normalization': 'none',
            'csa_clip_length': None,
        },
        'standard': {
            'condition_limit': 1e14,
            'trace_normalization': 'none',
            'csa_clip_length': None,
        },
        'robust': {
            'condition_limit': 1e12,
            'trace_normalization': 'arithmetic',
            'csa_clip_length': (-2, 2),
        },
        'maximum': {
            'condition_limit': 1e10,
            'trace_normalization': 'arithmetic',
            'csa_clip_length': (-1, 1),
        },
    }
    
    config = configs.get(stability_level, configs['standard'])
    
    return CMA_ES(
        population_size=population_size,
        solution=solution,
        **config,
    )
