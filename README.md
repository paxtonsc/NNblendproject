Summer research promect. The goal is to develop a recurrent neural network (RNN) to emulate the Large Synoptic Survey Telescope (LSST) deblending pipeline in order to better understand systemic bias in weak lensing probes as a result of unrecognized galaxy-galaxy blends.


Workflow:

(1) run input_maker.py to generated randomized parameters in input folder
(2) run generate_blends.py to draw images using gal sim
(3) run sep_identify.py to run source extractor on gal sim blends. Resulting data is stored in the params folder.
Also, can optially draw figures to visualize the gal sim images and source extractor identifications in 'figures'.
(4) train neural net. 
