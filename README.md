### Prefrontal Cortex Dopamine Responds to the Total Valence of Stimuli
#### Ian T. Ellwood, Department of Neurobiology and Behavior, Cornell University

This repository contains the code and data required to reproduce the figures in the paper "Prefrontal Cortex Dopamine Responds to the Total Valence of Stimuli" by Yating Yang, Willem Parent, Hunter Rice, Risa Mark, Matthew Desimone, Monzilur Rahman and Ian T. Ellwood. Any questions about the repository should be sent to the corresponding author (I.T.E.). 

#### Running the scripts

All of the scripts were written in python 3.9.13 in the Pycharm Community Edition IDE and require `numpy`, `matplotlib` and `scipy`.

#### Preprocessed data vs. the original recordings

In order to produce an analysis pipeline which runs efficiently and to save costs for online storage, the original datafiles were preprocessed by filtering and then downsampling the data to 100 Hz from the original 1000Hz photometry recordings.

Additionally, the isosbestic substraction was performed, but the original unsubtracted signals are also found in the datafiles. The preprocessed data, as well as the code that was used to produce them has been made available in the repository, but the original 1000 Hz recordings can be made available via a dropbox link on request. Note that none of the files with names like `PreprocessData.py` or `PreprocessDrawerData.py` will run without the `OriginalData/` folder.

The preprocessed data is found in the folder `PreprocessedData/`. If you plan to use  this data for analysis in your own research, please contact the corresponding author and include a citation of our work.

#### Additional Notes

Some of the files require data that is saved when another script is run. For example,  `PlotHeatMapsFromSavedData` and `PlotLickRates.py`. The saved files have been included in the repository, but if changes are made to the analysis scripts, they must be recomputed.



