"""Financial evaluation helper namespace.

The selected checkpoint evaluator does not require these downstream routines.
Modules in this package lazily expose the migrated legacy helpers to avoid
importing optional finance dependencies unless a caller explicitly uses them.
"""
