# mf-trees

Repository for working with Tree-based ML algorithms on finding potential monitoring locations in a groundwater basin using the inherent tree feature importance.

We use the USGS MODFLOW-2005 model to generate synthetic hydraulic head a nd streamflow values for testing. We use <i>flopy</i> to generate inputs and run the MODFLOW model. The code for this is in the mf_notebooks folder. There are 3 scenarios: a simple steady state run, a transient scenario with constant pumping, and a transient scenario with seasonal pumping.

The notebooks in this directory house the Tree-based ML algorithm iimplementation (using <i>sklean</i>) for training and testing code blocks for each of the 3 scenarios. 

treefuncs.py is a utility file that has useful functions for running and plotting results.

11/7/24 - Requirements.txt added for conda enviornment 
Primary package versions used:
flopy=3.3.4
numpy=1.24.3
pandas=1.3.4
scikit-learn=1.2.2

