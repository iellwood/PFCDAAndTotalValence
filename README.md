### Prefrontal Cortex Dopamine Responds to the Total Valence of Stimuli
#### Ian T. Ellwood, Department of Neurobiology and Behavior, Cornell University

This repository contains the code and data required to reproduce the figures in the paper "Prefrontal Cortex Dopamine Responds to the Total Valence of Stimuli" by Yating Yang, Willem Parent, Hunter Rice, Risa Mark, Matthew Desimone, Monzilur Rahman and Ian T. Ellwood. Any questions about the repository should be sent to Ian Ellwood, the corresponding author. 

#### Running the scripts

All of the scripts were written in python 3.9.13 in the Pycharm Community Edition IDE and reqire `numpy`, `matplotlib` and `scipy`. In order to produce an analysis pipeline which runs efficiently, the original datafiles were preprocessed by filtering and then downsampling the data to 100 Hz from the original 1000Hz photometry recordings. Additionally, the isosbestic substraction was performed, but the original unsubtracted signals are also found in the datafiles. The preprocessed data, as well as the code that was used to produce them has been made available in the repository, but the original 1000 Hz recordings can be made available via a dropbox link on request.

