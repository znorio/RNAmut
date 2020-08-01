# Takes RNA reference and alternative read counts as input
# Can either be used to compute posterior distributions of overdispersion, dropout and mutation probabilities
# or for inference of the cell lineage

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
moveProbsParams = [0.25, 0.4, 0.35, 0.05]           # moves: change Params (0,1) / prune&re-attach / swap node labels / swap subtrees -> weights -> don't have to sum up to 1
                                                    # swap subtrees only if nodes in different lineages else prune&re-attach
priorAlphaBetaoodp = [2, 10, 2, 2, 1.5, 3, 2, 18]   # alpha overdispersion_wt, beta overdispersion_wt, alpha overdispersion_mut, beta overdispersion_mut, 
                                                    # alpha dropout, beta dropout, alpha prior_p_mutation, beta prior_p_mutation
oodp = [100, 1, 0.2, 0.1]                           # overdispersion_wt, overdispersion_mut, dropout, prior_p_mutation
covDiagonal = [1, 0.001, 0.0002, 0.00001]           # Initial covariance Matrix is all zeros expcept these values in the diagonal from upper left to lower right
maxValues = [1000, 2, 1, 1]                         # The maximum values for the parameters overdispersion_wt, overdispersion_mut, dropout, prior_p_mutation
                                                    # Shouldn't be smaller than 1
minValues = [0,0,0,0]                               # The minimal values for the parameters overdispersion_wt, overdispersion_mut, dropout, prior_p_mutation
outFile = "tree"                                    # The name of the output files
rep = 1                                             # number of repetitions of the MCMC
loops = 100000                                      # number of loops within a MCMC
gamma = 1                                           # if = 1 no influence
initialPeriod = 10000                               # number of iterations before the initial covariance matrix is adapted                               
sampleStep = 1                                      # stepsize between sampling of parameters and trees/ 1 -> sampled every round, 2 -> sampled every second round,...
burnInPhase = 0.25                                  # burnIn / total number of loops
decVar = 0.1                                        # The covariance matrix is multiplied with this factor, to increase or decrease it 0.1 -> 10 times smaller
                                                    # increases or decreases the acceptance rate
adaptAcceptanceRate = True                          # if true starts with given decVar, but adapts it every 1000 steps if the acceptance rate lies outside 1/4 to 1/2
factorParamsLogScore = 10                           # Is multiplied with the parameter log score to increase or decrease its influence compared to the tree log score
                                                    # -> Helps to prevent empty tree solution
factor_owt = 2                                      # Is additionally multiplied with the overdisperison_wt log-score, because this is the main parameter 
                                                    # responsible for the empty tree solution
marginalization = True                              # If false the program maximizes the placement of the cells, if true the program tries to find the marginal distribution 
                                                    # of the parameters

optTreeOutputNum = 3                                # Determines the maximal number of optimal trees for the output, if output_gv = True and / or output_mut_csv = True
                                                    # if = -1 no restrictions on the output size
output_mut_csv = True                               # if true outputs 1,0 mutation matrix of optimal trees / marginalization should be False
output_gv = True                                    # if true outputs graphviz file of optimal trees / marginalization should be False
output_samples = True                               # if true outputs all samples after burn-in of current tree log-score, current params and curent parent vector as numpy array
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
num_mut = rows_ref

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

# probability matrix of mutation using best parameters
if output_ProbabilityMatrix == True:           
    pmat = calculate_pmat(bestParams[0], bestParams[1], bestParams[2], bestParams[3])
    savetxt(outFile + "_pmat.csv", pmat, delimiter = ",")

    
# if true outputs 1,0 mutation matrix
if output_mut_csv == True:                               
    if optTreeOutputNum == -1:
        optTreeOutputNum = len(optimal)
        
    for o, opt in enumerate(optimal):
        if o >= optTreeOutputNum:
            break
        mut = oneZeroMut(opt[1], opt[0], num_mut, num_cells)
        savetxt(outFile + "_mut_" + str(o) + ".csv", mut, delimiter = ',')
        

# if true outputs graphviz file of tree
if output_gv == True:
    if optTreeOutputNum == -1:
        optTreeOutputNum = len(optimal)
        
    for o, opt in enumerate(optimal):
        if o >= optTreeOutputNum:
            break
        gv = graphviz(opt[1], opt[0], num_mut, num_cells)
        with open( outFile + "_" + str(o) + ".gv", "w") as text_file:
            text_file.write(gv)

            
# outputs all samples as np.array
if output_samples == True:
    samples = np.array(samples)
    np.save(outFile, samples)
