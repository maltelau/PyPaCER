"""Benchmarking utilities for comparing interpolation performance."""

import time
from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import LinearNDInterpolator


def generate_test_data(
    n_points: int = 10000, dim: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random test data for interpolation."""
    np.random.seed(42)
    points = np.random.randn(n_points, dim).astype(np.float32)
    values = np.random.randn(n_points).astype(np.float32)
    return points, values


def benchmark_scipy_interpolation(
    points: np.ndarray, values: np.ndarray, query_points: np.ndarray
) -> Dict[str, float]:
    """Benchmark scipy LinearNDInterpolator."""
    # Setup time
    start_setup = time.time()
    interpolator = LinearNDInterpolator(points, values)
    setup_time = time.time() - start_setup

    # Interpolation time
    start_interp = time.time()
    results = interpolator(query_points)
    interp_time = time.time() - start_interp

    return {
        "setup_time": setup_time,
        "interp_time": interp_time,
        "total_time": setup_time + interp_time,
        "points_per_sec": len(query_points) / interp_time,
    }


def benchmark_pytorch_interpolation(
    points: np.ndarray, values: np.ndarray, query_points: np.ndarray, k: int = 8
) -> Dict[str, float]:
    """Benchmark PyTorch KNN interpolator."""
    try:
        import torch

        from .pytorch_interpolation import PyTorchKNNInterpolator
    except ImportError:
        return {"error": "PyTorch not available"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup time
    start_setup = time.time()
    interpolator = PyTorchKNNInterpolator(points, values, k=k, device=device)
    setup_time = time.time() - start_setup

    # Interpolation time
    start_interp = time.time()
    results = interpolator(query_points)
    interp_time = time.time() - start_interp

    # Include GPU sync time if using CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()

    return {
        "setup_time": setup_time,
        "interp_time": interp_time,
        "total_time": setup_time + interp_time,
        "points_per_sec": len(query_points) / interp_time,
        "device": str(device),
    }


def benchmark_oor_scenario(
    n_source_points: int = 50000, n_trajectory_points: int = 3000, grid_size: int = 10
) -> Dict[str, Dict]:
    """
    Benchmark the specific OOR scenario from PyPaCER.

    This simulates interpolating ~3000 orthogonal grids of ~100 points each
    from a point cloud of ~50000 points.
    """
    print("\nBenchmarking OOR scenario:")
    print(f"  Source points: {n_source_points}")
    print(f"  Trajectory points: {n_trajectory_points}")
    print(f"  Grid size: {grid_size}x{grid_size}")
    print(f"  Total query points: {n_trajectory_points * grid_size * grid_size}")

    # Generate test data
    points, values = generate_test_data(n_source_points, dim=3)

    # Generate query points (simulating orthogonal grids)
    query_points = []
    for i in range(n_trajectory_points):
        # Simulate an orthogonal grid around a trajectory point
        center = np.random.randn(3)
        u = np.linspace(-1, 1, grid_size)
        v = np.linspace(-1, 1, grid_size)
        uu, vv = np.meshgrid(u, v)
        grid = np.stack(
            [
                center[0] + uu.ravel(),
                center[1] + vv.ravel(),
                center[2] + np.zeros_like(uu.ravel()),
            ],
            axis=1,
        )
        query_points.append(grid)

    query_points = np.vstack(query_points).astype(np.float32)

    results = {}

    # Benchmark scipy
    print("\n  Testing scipy LinearNDInterpolator...")
    results["scipy"] = benchmark_scipy_interpolation(points, values, query_points)
    print(f"    Total time: {results['scipy']['total_time']:.2f}s")
    print(f"    Points/sec: {results['scipy']['points_per_sec']:.0f}")

    # Benchmark PyTorch
    print("\n  Testing PyTorch KNN interpolator...")
    results["pytorch"] = benchmark_pytorch_interpolation(points, values, query_points)
    if "error" not in results["pytorch"]:
        print(f"    Device: {results['pytorch']['device']}")
        print(f"    Total time: {results['pytorch']['total_time']:.2f}s")
        print(f"    Points/sec: {results['pytorch']['points_per_sec']:.0f}")

        # Calculate speedup
        speedup = results["scipy"]["total_time"] / results["pytorch"]["total_time"]
        results["speedup"] = speedup
        print(f"\n  Speedup: {speedup:.1f}x")

    return results


def run_full_benchmark():
    """Run comprehensive interpolation benchmarks."""
    print("PyPaCER Interpolation Performance Benchmark")
    print("=" * 50)

    # Test different scenarios
    scenarios = [
        {"n_source": 10000, "n_traj": 1000, "grid": 10},
        {"n_source": 50000, "n_traj": 3000, "grid": 10},
        {"n_source": 100000, "n_traj": 3000, "grid": 15},
    ]

    all_results = []
    for scenario in scenarios:
        results = benchmark_oor_scenario(
            n_source_points=scenario["n_source"],
            n_trajectory_points=scenario["n_traj"],
            grid_size=scenario["grid"],
        )
        all_results.append(results)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    for i, (scenario, results) in enumerate(zip(scenarios, all_results)):
        print(f"\nScenario {i+1}:")
        print(f"  Source points: {scenario['n_source']}")
        print(f"  Query points: {scenario['n_traj'] * scenario['grid']**2}")

        if "pytorch" in results and "error" not in results["pytorch"]:
            print(f"  Speedup: {results.get('speedup', 0):.1f}x")
            print(
                f"  Time saved: {results['scipy']['total_time'] - results['pytorch']['total_time']:.1f}s"
            )


if __name__ == "__main__":
    run_full_benchmark()
