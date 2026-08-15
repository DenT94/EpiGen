# TODO

Running list of "how would this work for insertions?" questions raised while
building the substitution-only MVP. Insertions stay conceptual for now per
mypipelinethoughts.md — nothing here should block substitution-path work.

- [ ] **PDB-ID structure short-circuit.** Skipping the initial ESMFold2 call
      when a PDB ID is given only works for substitution edits (same-length,
      same-geometry backbone). Insertions always need ESMFold2 to generate a
      new backbone for the edit-only state first, even with a solved WT
      structure available -- there's no existing structure for "WT + inserted
      loop, no compensation yet."
