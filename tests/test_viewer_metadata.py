"""Explicit provenance in exported visualization frames; no browser required."""
from conductor_sim import NestedShellBackend
from conductor_sim.visual_adapters import context_for_solution
from conductor_sim.visualization import sample_frame
from test_visualization import Uniform, problem, plane


def test_custom_backend_is_not_mislabeled_as_analytical():
    context = context_for_solution(problem(), Uniform())
    frame = sample_frame(context, plane(), seeds=[], profile_samples=9)
    assert 'Supplied solver' in frame['conductor_results_method']
    assert 'Analytical' not in frame['conductor_results_method']
    assert frame['profiles'] == []


def test_analytical_conductor_result_provenance_is_explicit():
    p = problem()
    context = context_for_solution(p, NestedShellBackend().solve(p))
    frame = sample_frame(context, plane(), seeds=[], profile_samples=9)
    assert frame['conductor_results_method'] == 'Analytical reference conductor properties.'
