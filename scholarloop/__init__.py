"""scholarloop: an auto-research framework that replicates a PhD experiment workflow.

This package currently ships the three foundational data contracts (the schemas the
rest of the system is built on top of):

  - scholarloop.profile  : Domain Profile loader + validator (makes domains pluggable)
  - scholarloop.ledger   : Experiment Ledger writer/reader (the cross-run memory)
  - scholarloop.registry : VerifiedRegistry (captures measured numbers; anti-hallucination)

Import the contracts directly, e.g.:

    from scholarloop.ledger import Ledger, LedgerEntry
    from scholarloop.profile import load_profile
    from scholarloop.registry import VerifiedRegistry

Submodules are intentionally not re-exported here so that `python -m scholarloop.<mod>`
CLIs run without an import-ordering warning. Each module's docstring documents its role.
"""

__version__ = "0.0.1"
