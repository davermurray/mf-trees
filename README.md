# mf-trees

Repository for work using feature importance in Tree-based ML algorithms on finding potential monitoring locations in a groundwater basin. The code is based on an idealized basin. 

We use the USGS MODFLOW-2005 model to generate synthetic hydraulic head a nd streamflow values for testing. We use <i>flopy</i> to generate inputs and run the MODFLOW model. The code for this is in the mf_notebooks folder. There are 3 scenarios: a simple steady state run, a transient scenario with constant pumping, and a transient scenario with seasonal pumping.

The notebooks in this directory house the Tree-based ML algorithm iimplementation (using <i>sklean</i>) for training and testing code blocks for each of the 3 scenarios. 

treefuncs.py is a utility file that has useful functions for running and plotting results.
