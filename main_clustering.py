from input_scripts.Inputs_Clustering_data import setup_clustering_data_params
from input_scripts.Inputs_Clustering import setup_clustering_params
from functions.Run_Clustering import run_clustering


###############################################################################
# DESCRIPTION
# This is the main script used to run the third step, clustering, of the
# innovation-driven co-activation patterns
#
# REFERENCES
# For theoretical references about those techniques and applications to
# functional brain imaging, see the following:
#
# - A Signal Processing Approach to Generalized 1-D Total Variation
#   (Karahanoglu et al., IEEE Transactions on Signal Processing, vol. 59,
#   no. 11, November 2011) for details on the linear operator used in total
#   activation (deconvolution coupled to derivation) and its discrete
#   implementation
#
# - Total activation: fMRI deconvolution through spatio-temporal
#   regularization (Karahanoglu et al., Neuroimage, vol. 73, pp. 121-134,
#   2013) for an overview of total activation in its first version (still
#   used as such for temporal regularization; see Appendix A. for an
#   outline of the algorithm)
#
# - Transient brain activity disentangles fMRI resting-state dynamics in
#   terms of spatially and temporally overlapping networks (Karahanoglu
#   et al., Nature communications, DOI: 10.1038/ncomms8751, 2015) for an
#   application of total activation to resting-state brain data, and the
#   introduction of the thresholding and iCAP steps
#
# - Regularized spatiotemporal deconvolution of fMRI data using
#   gray-matter constrained total variation (Farouj et al., ISBI abstract,
#   2016) for an overview of the presently used version of spatial
#   regularization
#
# ACKNOWLEDGMENTS
# The present codes are adaptations from the initial version developed
# by Dr Isik Karahanoglu, former PhD student at MIP:Lab, and Dr Younes
# Farouj, former visiting PhD student and now post-doc at MIP:Lab
#
# UPDATES ON UTILITIES
# V1.0, December 9th 2016, Thomas Bolton: simplified total activation
#       scripts running for one subject (not yet functional)
# V1.1, December 23rd 2016, Thomas Bolton: finished modifying total
#       activation routines for one subject
# V1.2, December 28th 2016, Thomas Bolton: everything written down as
#       functions for total activation, and tested on .img/.hdr functional
#       and segmentation files
# V2.0, May 2018, Daniela Zoeller: including of updates of all lab members:
#       - changed structure of results saving
#       - separated TA, thresholding, clustering and regression in different
#         functions
#       - included checking whether TA has already been done
#
#       Jun 2018, Younes Farouj:
#       - check writing permission
#       - check post-interpolation data size

# Mar 2026, Scott Ensel
# - Converted entire library to Python iCAP(y)
###############################################################################


def main():
    # Data-related params (path, subjects, TR, etc.)
    param = setup_clustering_data_params()

    # Clustering-specific params
    param.update(setup_clustering_params())

    # Run clustering (Aggregation + k-means + consensus, etc.)
    run_clustering(param)


if __name__ == "__main__":
    main()
