import numpy as np

def add_laplace_noise(data, epsilon, lower_bound, upper_bound):
    """
    Add Laplace noise to each element in the list for differential privacy.

    Args:
    - data: list of values.
    - epsilon: privacy budget for each element.
    - lower_bound: lower bound of data.
    - upper_bound: upper bound of data.

    Returns:
    - A new list with Laplace noise added to each element.

    在实践中，ϵ 的值通常在 0.01 到 1 的范围内，尽管在某些情况下也可能使用更高或更低的值。较小的值（如 0.01、0.1）提供较强的隐私保护，但可能会显著降低数据的实用性。较高的值（如 0.7、1）提供较弱的隐私保护，但保留了更多的数据实用性
    """
    scale = 1.0 / epsilon
    noisy_data = []

    for value in data:
        # Ensure the value is within bounds
        bounded_value = max(min(value, upper_bound), lower_bound)
        # Add Laplace noise
        noise = np.random.laplace(0, scale)
        noisy_value = bounded_value + noise
        noisy_data.append(noisy_value)

    return noisy_data