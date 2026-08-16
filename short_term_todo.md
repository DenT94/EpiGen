1. [x] Split design page into Landing (where the starting box is) + results (after Structure viewer)
2. [x] In the results page, we want to have a comparison of WT score, starting points of the chains and endpoints of the chain. Possibly histograms of the two classes (can use the precomputed window substitutions), with the WT being a dashed vertical line
3. [x] Microenvironment and SAE diff are not very informative on their own: they should probably be on the back end, with only the agent explanation being shown (also as usual, we cache everything that is computationally expensive)
4. [x] We don't need the ! warning for everything inside the edit window. Did we make sure that our original edit is preserved by the sampling?
5. [x] Now in the structural viewer page we have two conflicting Color by labeled delta delta sae in conflict, doesn't make much sense.
6. [x] Other MCMC candidate does not show WT by default nor the top candidate, and sees which of the elements dropdown have a structure in cache and shows the top scoring of those
7. something broke in the structure viewer. All tabs inside structure viewer should have the same view. Also please color the edited region eg WHSPRAL red, and SAE activations in a different color, e.g. orange
8. also past runs should be labeled with PST? 

 