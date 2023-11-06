"""Soft indicator function evaluating whether a number is within bounds."""

import warnings
import torch


@torch.jit.script
def _sigmoids(x: torch.Tensor, value_at_margin: float, sigmoid: str) -> torch.Tensor:
    """Returns 1 when `x` == 0, between 0 and 1 otherwise.
    Args:
      x: A scalar or numpy array.
      value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
      sigmoid: String, choice of sigmoid type.
    Returns:
      A numpy array with values between 0.0 and 1.0.
    Raises:
      ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
        `quadratic` sigmoids which allow `value_at_1` == 0.
      ValueError: If `sigmoid` is of an unknown type.
    """
    if sigmoid in ('cosine', 'linear', 'quadratic'):
        if not 0 <= value_at_margin < 1:
            raise ValueError('`value_at_1` must be nonnegative and smaller than 1, '
                             'got {}.'.format(value_at_margin))
    else:
        if not 0 < value_at_margin < 1:
            raise ValueError('`value_at_1` must be strictly between 0 and 1, '
                             'got {}.'.format(value_at_margin))

    value_at_1 =  value_at_margin * torch.ones_like(x)

    if sigmoid == 'gaussian':
        scale = torch.sqrt(-2 * torch.log(value_at_1))
        return torch.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == 'hyperbolic':
        scale = torch.arccosh(1 / value_at_1)
        return 1 / torch.cosh(x * scale)

    elif sigmoid == 'long_tail':
        scale = torch.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == 'reciprocal':
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    # elif sigmoid == 'cosine':
    #     scale = torch.arccos(2 * value_at_1 - 1) / torch.pi
    #     scaled_x = x * scale
    #     with warnings.catch_warnings():
    #         warnings.filterwarnings(
    #             action='ignore', message='invalid value encountered in cos')
    #         cos_pi_scaled_x = torch.cos(torch.pi * scaled_x)
    #     return torch.where(abs(scaled_x) < 1, (1 + cos_pi_scaled_x) / 2, 0.0)

    elif sigmoid == 'linear':
        scale = 1 - value_at_1
        scaled_x = x * scale
        return torch.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0*scaled_x)

    elif sigmoid == 'quadratic':
        scale = torch.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return torch.where(abs(scaled_x) < 1, 1 - scaled_x ** 2, 0.0*scaled_x)

    elif sigmoid == 'tanh_squared':
        scale = torch.arctanh(torch.sqrt(1 - value_at_1))
        return 1 - torch.tanh(x * scale) ** 2

    else:
        raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))


@torch.jit.script
def tolerance(x: torch.Tensor, bounds_lower: float, bounds_upper: float, margin: float, sigmoid: str,
              value_at_margin: float) -> torch.Tensor:
    """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.
    Args:
      x: A scalar or numpy array.
      bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
      margin: Float. Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
          outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
          increasing distance from the nearest bound.
      sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
         'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
      value_at_margin: A float between 0 and 1 specifying the output value when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.
    Returns:
      A float or numpy array with values between 0.0 and 1.0.
    Raises:
      ValueError: If `bounds[0] > bounds[1]`.
      ValueError: If `margin` is negative.
    """
    lower = bounds_lower
    upper = bounds_upper
    if lower > upper:
        raise ValueError('Lower bound must be <= upper bound.')
    if margin < 0:
        raise ValueError('`margin` must be non-negative.')

    in_bounds = torch.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = torch.where(in_bounds, 1.0, 0.0)
    else:
        d = torch.where(x < lower, lower - x, x - upper) / margin
        reward_tensor=_sigmoids(d, value_at_margin, sigmoid)
        value = torch.where(in_bounds, torch.ones_like(reward_tensor), reward_tensor)
    return value
