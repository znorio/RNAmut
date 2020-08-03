"""
RNAmut takes RNA reference and alternative nucleotide read counts as input.
The input format should be equivalent to the example data files reference_reads.csv and alternative_reads.csv (apart from the size).

An algorithm (scores.py/calculate_pmat) transfers the read counts to probabilities of mutation. To model the nucleotide read counts 
in relation to the total coverage, the algorithm uses a beta-binomial distribution.
The algorithms parameters include two overdispersion terms (overdispersion_mut for the mutated and overdispersion_wt for the non-mutated case)
describing the shape of the beta distributions (overdispersion = alpha + beta). Furthermore the parameters include an allelic dropout term (dropout)
and the prior probability of mutation (prior_p_mutation). RNAmut can be used to optimize these parameters using a Metropolis-Hastings algorithm and 
to sample from the posterior probability distribution.
Additionally, this program tries to derive the phylogenetic relation inbetween single cells to find the cell lineage trees and parameters, which best 
explain the observed read counts. The tree consists of the possible mutation sites, with the cells attached to them.
Cells attached to the root have no mutations in any of the mutation sites. A cell attached to another part of the tree has the mutation 
it is attached to and all mutations of its ancestors. This approach is based on SCIPHI "https://www.nature.com/articles/s41467-018-07627-7".

As optional outputs, we have on the one hand the calculated probabilities of mutation, making use of the best parameters. 
On the other hand the mutations derived from the best cell lineage trees. In addition, a graphviz file to picture the tree and all samples 
collected after the burn-in phase can be produced as well. To visualize some of the results, it might be helpful to take a look at the 
following notebook "https://github.com/znorio/RNAmut/edit/master/Notebooks/"
"""

import numpy as np
import pandas as pd
from numpy import savetxt
from .metropolis_hastings import runMCMCoodp
from .output import oneZeroMut, graphviz
from .scores import calculate_pmat


# Options

moveProbsParams = [0.25, 0.4, 0.35, 0.05]           # probabilities of different moves: 
                                                    # 1.  range (0,1): determines the probability that in one round of the Metropolis-Hastings algorithm
                                                    #     the parameters are updated, one minus this probability is the probability that the trees are updated
                                                    # 2.  prune&re-attach: prune a subtree and re-attach it to the main tree
                                                    # 3.  swap node labels: two nodes are randomly chosen and their labels exchanged
                                                    # 4.  swap subtrees: swap subtrees only if nodes in different lineages, else prune&re-attach
                                                    # 2,3 and 4 are weights -> they don't have to sum up to 1
            
oodp = [100, 1, 0.2, 0.1]                           # initial values for overdispersion_wt, overdispersion_mut, dropout, prior_p_mutation

priorAlphaBetaoodp = [2, 10, 2, 2, 1.5, 3, 2, 18]   # for the two overdispersion, dropout and mutation parameters, a prior beta distribution is specified:
                                                    # alpha overdispersion_wt, beta overdispersion_wt, alpha overdispersion_mut, beta overdispersion_mut, 
                                                    # alpha dropout, beta dropout, alpha prior_p_mutation, beta prior_p_mutation
 
covDiagonal = [1, 0.001, 0.0002, 0.00001]           # initial covariance Matrix is all zeros expcept these values in the diagonal from upper left to lower right
                                                    # the covariance matrix is learned adaptively and it's used for drawing from a multivariate normal distribution
                                                          
maxValues = [1000, 2, 1, 1]                         # the maximum values for the parameters overdispersion_wt, overdispersion_mut, dropout, prior_p_mutation
                                                    # values larger than (maximum_value - 0.00001) are not considered
  
minValues = [0,0,0,0]                               # the minimal values for the parameters overdispersion_wt, overdispersion_mut, dropout, prior_p_mutatio
                                                    # values smaller than (minimum_value + 0.00001) are not considered
  
outFile = "tree"                                    # the name of the output files
frequency_of_nucleotide = 0.5                       # expected allele frequency
sequencing_error_rate = 0.01                        # if small, it has little effect on the mutation probability
reps = 1                                            # number of repetitions of the MCMC
loops = 100000                                      # number of loops within a MCMC
initialPeriod = 10000                               # number of iterations before the initial covariance matrix is adapted 
sampleStep = 1                                      # stepsize between sampling of parameters and trees/ 1 -> sampled every round, 2 -> sampled every second round,...
burnInPhase = 0.25                                  # burn-in loops / total number of loops

decVar = 0.1                                        # the covariance matrix is multiplied with this factor, to increase or decrease it 0.1 -> 10 times smaller
                                                    # increases or decreases the acceptance rate
  
adaptAcceptanceRate = True                          # if true starts with given decVar, but adapts it every 1000 loops, if the acceptance rate lies outside 1/4 to 1/2

factorParamsLogScore = 10                           # is multiplied with the parameter log score to increase or decrease its influence compared to the tree log score
                                                    # -> Helps to prevent empty tree solution (all cells attached to the root).
  
factor_owt = 2                                      # is additionally multiplied with the overdisperison_wt log-score, because this is the main parameter 
                                                    # responsible for the empty tree solution.
  
marginalization = True                              # if false the program maximizes the placement of the cells, if true the program tries to find the marginal 
                                                    # distribution of the parameters.

optTreeOutputNum = 3                                # determines the maximal number of optimal trees for the output, if output_gv = True and / or output_mut_csv = True
                                                    # if = -1 -> no restrictions on the output size
  
output_mut_csv = True                               # if true outputs 1,0 mutation matrix of optimal trees / marginalization should be False
output_gv = True                                    # if true outputs graphviz file of optimal trees / marginalization should be False
output_samples = False                              # if true outputs all samples after burn-in of current log-score, current tree log-score,
                                                    # current parameters and curent parent vector as numpy array
output_sampleParams = True                          # if true outputs all samples after burn-in of current parameters and current log-score as numpy array
output_ProbabilityMatrix = True                     # if true outputs probability matrix using the best parameters as csv file

path = ".Data/"                                     # specify the main path to the files
alt_file = "alternative_reads.csv"                  # name of alternative read file
ref_file = "reference_reads.csv"                    # name of reference read file


# Load files
pd_ref = pd.read_csv(path + ref_file, sep = ",")
pd_alt = pd.read_csv(path + alt_file, sep = ",")
gene_names = list(pd_ref.iloc[:,0])
cell_names = list(pd_ref.columns[1:])

# Replace NANs with 0
ref = np.array(pd_ref.fillna(0))[:,1:].tolist()
alt = np.array(pd_alt.fillna(0))[:,1:].tolist()

# Determine number of cells and mutation sites
num_mut = len(ref) # number of mutations is equal to the number of rows

num_cells = len(ref[1]) # number of cells is equal to the number of columns

print("Number of mutation sites:", num_mut, " Number of cells:", num_cells)

rows_alt = len(alt)
columns_alt = len(alt[1])

if num_mut != rows_alt:
    print("The number of mutation sites is not the same in files", ref_file, "and", alt_file)

if num_cells != columns_alt:
    print("The number of cells s is not the same in files", ref_file, "and", alt_file)
    

# Run Markov chain Monte Carlo / Metropolis Hastings algorithm
samples, sampleParams, optimal, bestParams = runMCMCoodp(reps, loops, oodp, priorAlphaBetaoodp, moveProbsParams, sampleStep, initialPeriod, adaptAcceptanceRate, \
                                                         covDiagonal, maxValues, minValues, burnInPhase, decVar, factor_owt, factorParamsLogScore, marginalization, \
                                                         frequency_of_nucleotide, sequencing_error_rate, num_mut, num_cells, alt, ref)


# Create all desired output files

# Uses the best parameters to calculate the probability of mutation using the RNA read counts
if output_ProbabilityMatrix == True:           
    pmat = calculate_pmat(bestParams[0], bestParams[1], bestParams[2], bestParams[3], frequency_of_nucleotide, sequencing_error_rate, num_mut, num_cells, alt, ref)
    savetxt(outFile + "_pmat.csv", pmat, delimiter = ",")

    
# If true outputs 1,0 mutation matrix
if output_mut_csv == True:                               
    if optTreeOutputNum == -1:
        optTreeOutputNum = len(optimal)
        
    for o, opt in enumerate(optimal):
        if o >= optTreeOutputNum:
            break
        mut = oneZeroMut(opt[1], opt[0], num_mut, num_cells, frequency_of_nucleotide, sequencing_error_rate)
        savetxt(outFile + "_mut_" + str(o) + ".csv", mut, delimiter = ',')
        

# If true outputs graphviz file of optimal trees
if output_gv == True:
    if optTreeOutputNum == -1:
        optTreeOutputNum = len(optimal)
        
    for o, opt in enumerate(optimal):
        if o >= optTreeOutputNum:
            break
        gv = graphviz(opt[1], opt[0], num_mut, num_cells, frequency_of_nucleotide, sequencing_error_rate, gene_names)
        with open( outFile + "_" + str(o) + ".gv", "w") as text_file:
            text_file.write(gv)

            
# Outputs all samples as np.array
if output_samples == True:
    samples = np.array(samples)
    np.save(outFile, samples)

# Outputs all parameter samples as np.array
if output_sampleParams == True:
    sampleParams = np.array(sampleParams)
    np.save(outFile, sampleParams)
