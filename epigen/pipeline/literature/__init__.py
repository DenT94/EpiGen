"""Literature/annotation lookups via Paperclip (UniProt features, PDB cross-refs).

Informs edit-window choice (pipeline step 3, CLAUDE.md): which residues are
known-functional or part of a known secondary-structure element, so the app
can steer the insertion/edit window away from them.
"""

from __future__ import annotations

from epigen.pipeline.literature.annotations import AnnotationRange, flag_positions, get_annotations
from epigen.pipeline.literature.plot import plot_annotation_map

__all__ = ["AnnotationRange", "flag_positions", "get_annotations", "plot_annotation_map"]
