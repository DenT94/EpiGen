1. Split design page into Landing (where the starting box is) + results (after Structure viewer)
2. In the results page, we want to have a comparison of WT score, starting points of the chains and endpoints of the chain. Possibly histograms of the two classes (can use the precomputed window substitutions), with the WT being a dashed vertical line
3. Microenvironment and SAE diff are not very informative on their own: they should probably be on the back end, with only the agent explanation being shown (also as usual, we cache everything that is computationally expensive)
4. We don't need the ! warning for everything inside the edit window. Did we make sure that our original edit is preserved by the sampling?
5. Now in the structural viewer page we have two conflicting Color by labeled delta delta sae in conflict, doesn't make much sense.
6. Other MCMC candidate does not show WT by default nor the top candidate, and sees which of the elements dropdown have a structure in cache and shows the top scoring of those
 