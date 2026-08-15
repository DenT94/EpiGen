"""Literature/annotation lookups via Paperclip (UniProt features, PDB cross-refs).

Informs edit-window choice (pipeline step 3, CLAUDE.md): which residues are
known-functional or part of a known secondary-structure element, so the app
can steer the insertion/edit window away from them.
"""

from __future__ import annotations

from epigen.pipeline.literature.annotations import AnnotationRange, get_annotations

__all__ = ["AnnotationRange", "get_annotations"]
