# Takes RNA reference and alternative read counts as input
# Input format should be comparable to the example data files reference_reads.csv and alternative_reads.csv

# The algorithm (more details -> scores.py/calculate_pmat) transfering these read counts to probabilities of mutations depends on several parameters.
# Two overdispersion terms (overdispersion_mut for the mutated and overdispersion_wt for the non-mutated case) describing the shape of the distributions, 
# an allelic dropout term (dropout) and the prior probability of mutation (prior_p_mutation) can be optimized with the Metropolis-Hastings algorithm.
# Additionally the same algorithm tries to derive the phylogenetic relation inbetween single cells to find the tree and parameters, which best 
# explain the observed read counts. This approach is based on SCIPHI "https://www.nature.com/articles/s41467-018-07627-7".

# It can either be used to compute posterior distributions of the overdispersion terms as well as dropout and mutation probabilities
# or for the inference of the cell lineage

import numpy as np
import pandas as pd
import math
from numpy import savetxt
import random
from random import gauss
from .metropolis_hastings import runMCMCoodp
from .output import getAttachmentPoints, oneZeroMut, graphviz
from .scores import calculate_pmat, log_scoretree2


# Options

moveProbsParams = [0.25, 0.4, 0.35, 0.05]           # Probability of different moves: 
                                                    # 1.  Range (0,1): Determines the probability that in one round of the Metropolis-Hastings algorithm
                                                    #     the parameters are updated, One minus this probability is the probability that the trees are updated
                                                    # 2.  prune&re-attach: Prune a subtree and re-attach it to the main tree
                                                    # 3.  swap node labels: Two nodes are randomly chosen and their labels exchanged
                                                    # 4.  swap subtrees: Swap subtrees only if nodes in different lineages else prune&re-attach
                                                    # 2,3 and 4 are weights -> they don't have to sum up to 1
oodp = [100, 1, 0.2, 0.1]                           # overdispersion_wt, overdispersion_mut, dropout, prior_p_mutation
            
# For the two overdispersion, dropout and mutation 
# parameters, a prior beta distribution is specified.
priorAlphaBetaoodp = [2, 10, 2, 2, 1.5, 3, 2, 18]   # alpha overdispersion_wt, beta overdispersion_wt, alpha overdispersion_mut, beta overdispersion_mut, 
                                                    # alpha dropout, beta dropout, alpha prior_p_mutation, beta prior_p_mutation
frequency_of_nucleotide = 0.5                       # expected allele frequency
sequencing_error_rate = 0.01                        # If small, it has little effect on the mutation probability

# The covariance matrix is learned adaptively.
# New parameters are drawn from a multivariate
# normal distribution.
covDiagonal = [1, 0.001, 0.0002, 0.00001]           # Initial covariance Matrix is all zeros expcept these values in the diagonal from upper left to lower right
maxValues = [1000, 2, 1, 1]                         # The maximum values for the parameters overdispersion_wt, overdispersion_mut, dropout, prior_p_mutation
                                                    # Shouldn't be smaller than 1
minValues = [0,0,0,0]                               # The minimal values for the parameters overdispersion_wt, overdispersion_mut, dropout, prior_p_mutation
outFile = "tree"                                    # The name of the output files
rep = 1                                             # number of repetitions of the MCMC
loops = 100000                                      # number of loops within a MCMC
gamma = 1                                           # if = 1 -> no influence
initialPeriod = 10000                               # number of iterations before the initial covariance matrix is adapted                               
sampleStep = 1                                      # stepsize between sampling of parameters and trees/ 1 -> sampled every round, 2 -> sampled every second round,...
burnInPhase = 0.25                                  # burnIn / total number of loops
decVar = 0.1                                        # The covariance matrix is multiplied with this factor, to increase or decrease it 0.1 -> 10 times smaller
                                                    # increases or decreases the acceptance rate
adaptAcceptanceRate = True                          # if true starts with given decVar, but adapts it every 1000 steps, if the acceptance rate lies outside 1/4 to 1/2
factorParamsLogScore = 10                           # Is multiplied with the parameter log score to increase or decrease its influence compared to the tree log score
                                                    # -> Helps to prevent empty tree solution (all cells attached to the root).
factor_owt = 2                                      # Is additionally multiplied with the overdisperison_wt log-score, because this is the main parameter 
                                                    # responsible for the empty tree solution.
marginalization = True                              # If false the program maximizes the placement of the cells, if true the program tries to find the marginal 
                                                    # distribution of the parameters.

optTreeOutputNum = 3                                # Determines the maximal number of optimal trees for the output, if output_gv = True and / or output_mut_csv = True
                                                    # if = -1 -> no restrictions on the output size
output_mut_csv = True                               # if true outputs 1,0 mutation matrix of optimal trees / marginalization should be False
output_gv = True                                    # if true outputs graphviz file of optimal trees / marginalization should be False
output_samples = True                               # if true outputs all samples after burn-in of current tree log-score,
                                                    # current params and curent parent vector as numpy array
output_ProbabilityMatrix = True                     # if true outputs probability matrix of best parameters as csv file

path = ".Data/"                                     # Specify the main path to the files
alt_file = "alternative_reads.csv"                  # Name of alternative read file
ref_file = "reference_reads.csv"                    # Name of reference read file


# Load files
pd_ref = pd.read_csv(path + ref_file, sep = ",")
pd_alt = pd.read_csv(path + alt_file, sep = ",")
gene_names = list(pd_ref.iloc[:,0])
cell_names = list(pd_ref.columns[1:])

# replace NANs with 0
ref = np.array(pd_ref.fillna(0)).tolist()
alt = np.array(pd_alt.fillna(0)).tolist()

# Determine number of cells and mutation sites
rows_ref = len(ref)
num_mut = rows_ref # number of mutations is equal to the number of rows, because the cellnames are removed

columns_ref = len(ref[1])
num_cells = columns_ref - 1

print("Number of mutation sites:", num_mut, " Number of cells:", num_cells)

rows_alt = len(alt)
columns_alt = len(alt[1])

if rows_ref != rows_alt:
    print("The number of mutation sites is not the same in files", ref_file, "and", alt_file)

if columns_ref != columns_alt:
    print("The number of cells s is not the same in files", ref_file, "and", alt_file)
    

# run Markov chain Monte Carlo / Metropolis Hastings algorithm
samples, sampleParams, optimal, bestParams = runMCMCoodp(rep, loops, oodp, priorAlphaBetaoodp, gamma, moveProbsParams, sampleStep, initialPeriod, \
                                                         covDiagonal, maxValues, minValues, burnInPhase, decVar)


# create all desired output files

# Uses the best parameters to calculate the probability of mutation using the RNA read counts
if output_ProbabilityMatrix == True:           
    pmat = calculate_pmat(bestParams[0], bestParams[1], bestParams[2], bestParams[3], frequency_of_nucleotide, sequencing_error_rate)
    savetxt(outFile + "_pmat.csv", pmat, delimiter = ",")

    
# if true outputs 1,0 mutation matrix
if output_mut_csv == True:                               
    if optTreeOutputNum == -1:
        optTreeOutputNum = len(optimal)
        
    for o, opt in enumerate(optimal):
        if o >= optTreeOutputNum:
            break
        mut = oneZeroMut(opt[1], opt[0], num_mut, num_cells, frequency_of_nucleotide, sequencing_error_rate)
        savetxt(outFile + "_mut_" + str(o) + ".csv", mut, delimiter = ',')
        

# if true outputs graphviz file of optimal trees
if output_gv == True:
    if optTreeOutputNum == -1:
        optTreeOutputNum = len(optimal)
        
    for o, opt in enumerate(optimal):
        if o >= optTreeOutputNum:
            break
        gv = graphviz(opt[1], opt[0], num_mut, num_cells, frequency_of_nucleotide, sequencing_error_rate)
        with open( outFile + "_" + str(o) + ".gv", "w") as text_file:
            text_file.write(gv)

            
# outputs all samples as np.array
if output_samples == True:
    samples = np.array(samples)
    np.save(outFile, samples)
