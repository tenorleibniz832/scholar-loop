"""Domain engines. Each subpackage is a (train_entrypoint, fixed_module) pair that a
domain profile points at. Engines are deliberately self-contained so the orchestrator
can run any of them as a subprocess under a wall-clock budget.
"""
