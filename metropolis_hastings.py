# Functions for running the parameter and tree optimization using a metropolis-hastings algorithm

import numpy as np
import math
import random
from random import gauss
from .trees import getRandParentVec, proposeNewTree, parentVector2ancMatrix
from .scores import calculate_pmat, log_scoretree, log_scoreparams


# If the new score is better, the move is accepted
# If it is worse, the probability of acceptance depends on how much worse it is
def acceptance(x_logs, x_new_logs):
    """
    Args:
        x_logs      - previous log score (float)
        x_new_logs  - new log score (float)
        
    Returns:
        accepted or not (bool)
    """
    if x_new_logs > x_logs:
        return True
    
    else:
        accept = np.random.uniform(0,1)
        return (accept < (math.e**(x_new_logs - x_logs)))
      

# Used to draw samples from a multivariate normal distribution
def sample_multivariate_normal(x, cov):
    """
    Args:
        x     - previous parameters (list)
        cov   - covariance matrix (numpy array)
        
    Returns:
        x_new - new parameters (list)
    """
    n = len(x)
    x_new = [0,0,0,0]
    chol = np.linalg.cholesky(cov)
    un = [gauss(0,1),gauss(0,1),gauss(0,1),gauss(0,1)]
    for i in range(n):
        for j in range(n):
            x_new[i] += float(chol[i][j]) * un[j]
        x_new[i] += x[i]
    return x_new
  
    
# Runs the Markov chain Monte Carlo (MCMC)/ Metropolis Hastings algorithm for learning the tree and parameters.
# It either samples from the posterior paramter distributions / optimizes the parameters, muatation tree and the attachment of cells
def runMCMCoodp(reps, loops, oodp, priorAlphaBetaoodp, moveProbsParams, sampleStep, initialPeriod, adaptAcceptanceRate, \
                covDiagonal, maxValues, minValues, burnInPhase, decVar, factor_owt, factorParamsLogScore, marginalization, \
                frequency_of_nucleotide, sequencing_error_rate, num_mut, num_cells, alt, ref):
    """
    Args:
        reps                    - number of repetitions of the MCMC (int)
        loops                   - number of loops within a MCMC (int)
        oodp                    - initial values for overdispersion_wt, overdispersion_mut, dropout, prior_p_mutation (list)
        priorAlphaBetaoodp      - alphas and betas of prior parameter distributions (list)
        moveProbsParams         - probabilities of different moves [parameters updated, prune&re-attach, swap node labels, swap subtrees] (list)
        sampleStep              - stepsize between sampling of parameters and trees (int)
        initialPeriod           - number of iterations before the initial covariance matrix is adapted (int)
        adaptAcceptanceRate     - if true starts with given decVar, but adapts it every 1000 loops, if the acceptance rate lies outside 1/4 to 1/2
        covDiagonal             - initial values of the covariance matrix in the diagonal from upper left to lower right (list)
        maxValues               - the maximum values for the parameters overdispersion_wt, overdispersion_mut, dropout, prior_p_mutation (list)
        minValues               - the minimal values for the parameters overdispersion_wt, overdispersion_mut, dropout, prior_p_mutation (list)
        burnInPhase             - burn-in loops / total number of loops (float)
        decVar                  - the covariance matrix is multiplied with this factor (float)
        factor_owt              - is multiplied with the overdisperison_wt log-score (float/int)
        factorParamsLogScore    - is multiplied with the parameter log score to increase or decrease its influence compared to the tree log score (float/int)
        marginalization         - false -> optimizes the tree, the placement of cells, true -> marginal distribution of the parameters (bool)
        frequency_of_nucleotide - expected allele frequency (float)
        sequencing_error_rate   - sequencing error rate (float)
        num_mut                 - number of mutation sites (int)
        num_cells               - number of cells (int)
        alt                     - alternative read counts (list)
        ref                     - wildtype/reference read counts (list)
        
    Returns:
        sample                  - all samples after burn-in of current tree log-score, current params and curent parent vector (list)
        sampleParams            - all samples after burn-in of current parameters and current log-score (list)
        optimalTreelist         - all optimal trees that are not equivalent and current parameters (list)
        bestParams              - optimal parameters (list)
    """
    optStatesAfterBurnIn = 0
    n = len(oodp)  # number of parameters
    burnIn = loops * burnInPhase
    parentVectorSize = num_mut
    eps = 0.00000000001
    optimalTreelist = []
    sample = []
    sampleParams = []
    newtree = True
    bestScore = bestTreeLogScore = -1000000
    
    for r in range(reps):       # starts over, but keeps sampling, bestScore, bestTreeLogScore

        av_params_t = oodp
        currTreeParentVec = getRandParentVec(parentVectorSize)     # start MCMC with random tree
        currTreeAncMatrix =  parentVector2ancMatrix(currTreeParentVec, parentVectorSize)
        currParams = oodp
        pmat = calculate_pmat(currParams[0], currParams[1], currParams[2], currParams[3], frequency_of_nucleotide, sequencing_error_rate, num_mut, num_cells, alt, ref)
        currpmat = pmat
        currTreeLogScore = log_scoretree(pmat, currTreeParentVec, marginalization, num_mut, num_cells)
        currParamsLogScore = log_scoreparams(currParams, maxValues, priorAlphaBetaoodp, factor_owt, factorParamsLogScore)
        currScore = currTreeLogScore + currParamsLogScore
        
        if currScore > bestScore:
            bestScore = currScore
            bestTreeLogScore = currTreeLogScore
            bestParams = currParams

        moveAcceptedParams = 0
        totalMovesParams = lastmoveAcceptedParams = 0
        moveAcceptedTrees = 0
        totalMovesTrees = lasttotalMoveParams = 0
        
        t = 1

        cov_mat = np.zeros((n,n))
        for z in range(n):
            cov_mat[z, z] = covDiagonal[z] * decVar
            
        for l in range(loops):
            
            if(l % 10000 == 0):
                print("At mcmc repetition " , r + 1 , "/" , rep , ", step " , l , " best tree score: " , bestTreeLogScore \
                , " and best overdispersion_wt: " , bestParams[0] , ", best overdispersion_mut: " , bestParams[1] , ", best dropout: " \
                , bestParams[2] , ", best prior_p_mutation: " , bestParams[3] , " and best overall score: " , bestScore , "\n", sep = "")

            if (adaptAcceptanceRate == True) and (l % 1000 == 999):
                currAcceptedMoveParams  = moveAcceptedParams - lastmoveAcceptedParams
                currtotalMoveParams = totalMovesParams - lasttotalMoveParams
                
                if currAcceptedMoveParams / currtotalMoveParams > 0.5:
                    decVar *= 1.1
                    
                if currAcceptedMoveParams / currtotalMoveParams < 0.25:
                    decVar *= 0.9
                    
                lasttotalMoveParams = totalMovesParams
                lastmoveAcceptedParams = moveAcceptedParams
                
            rand = np.random.uniform(0,1)
            if rand < moveProbsParams[0]:         # true if this move changes parameters, not the tree
                
                totalMovesParams += 1
                propParams = sample_multivariate_normal(currParams, cov_mat)

                #if the proposed parameters are out of range (or close to), they are not considered
                if (propParams[0] < (minValues[0] + 0.00001)) or (propParams[0] > (maxValues[0] - 0.00001)) or (propParams[1] < (minValues[1] + 0.00001)) \
                   or (propParams[1] > (maxValues[1] - 0.00001)) or (propParams[2] < (minValues[2] + 0.00001)) or (propParams[2] > (maxValues[2] - 0.00001)) \
                   or (propParams[3] < (minValues[3] + 0.00001)) or (propParams[3] > (maxValues[3] - 0.00001)):
                    continue 
                
                propParamsLogScore = log_scoreparams(propParams, maxValues, priorAlphaBetaoodp, factor_owt, factorParamsLogScore)
                pmat = calculate_pmat(propParams[0], propParams[1], propParams[2], propParams[3], frequency_of_nucleotide, sequencing_error_rate)
                propTreeLogScore = log_scoretree(pmat, currTreeParentVec, marginalization)
                propScore = propTreeLogScore + propParamsLogScore

                if acceptance(currScore, propScore, gamma):  # the proposed move is accepted
                    moveAcceptedParams += 1
                    currTreeLogScore  = propTreeLogScore
                    currParams = propParams
                    currParamsLogScore = propParamsLogScore
                    currScore = propScore
                    currpmat = pmat
                    
                if (l > 0.9 * initialPeriod):
                    t += 1
                    av_params_t += (currParams - av_params_t) / t
                    
                if (l > initialPeriod):
                    cov_mat = (t - 1)/t * (cov_mat  +  1/t * np.dot(np.transpose([currParams - av_params_t]), [currParams - av_params_t]) * decVar) + eps * np.identity(4)

        
            else:                 # if the move changes the tree not the parameter
                totalMovesTrees += 1
                propTreeParentVec = proposeNewTree(moveProbsParams, currTreeAncMatrix[:], currTreeParentVec[:], num_mut)
                propTreeLogScore = log_scoretree(currpmat, propTreeParentVec, marginalization)
                
                if acceptance(currTreeLogScore, propTreeLogScore, gamma):                   # the proposed tree is accepted
                    moveAcceptedTrees += 1
                    currTreeAncMatrix = parentVector2ancMatrix(propTreeParentVec, parentVectorSize)
                    currTreeParentVec = propTreeParentVec                                
                    currTreeLogScore  = propTreeLogScore                     
                    currScore = currTreeLogScore + currParamsLogScore
                    
            if currScore > bestScore:       # create a list with optimal trees and parameters
                optimalTreelist = []
                optimalTreelist.append([currTreeParentVec, currParams])

            if (currScore >= bestScore - eps) and (currScore <= bestScore + eps): # might not be necessary to use eps
                for o in optimalTreelist:
                    newtree = False
                    for u in range(num_mut):
                        if o[0][u] != currTreeParentVec[u]:
                            newtree = True
                            break
                            
                    if newtree == False:
                        break
                
                if (newtree == True):
                    optimalTreelist.append([currTreeParentVec, currParams])
                    
                if (l >= burnIn):
                    optStatesAfterBurnIn += 1
                    

            if(l >= burnIn and (l % sampleStep == 0) and (rand < moveProbsParams[0])):
                sampleParams.append([currParams, currScore])
                
            if(l >= burnIn and l % sampleStep == 0):
                sample.append([currTreeLogScore, currParams, currTreeParentVec])
            
            if(currScore > bestScore + eps):
                optStatesAfterBurnIn = 1          
                bestTreeLogScore = currTreeLogScore
                bestScore = currScore
                bestParams = currParams

    noStepsAfterBurnin = rep * (loops - burnIn)

    print( "best log score for tree: " , bestTreeLogScore)
    print( "optimal steps after burn-in: " , optStatesAfterBurnIn)
    print( "total #steps after burn-in: ", noStepsAfterBurnin)
    print( "percentage of optimal steps after burn-in: " , optStatesAfterBurnIn / noStepsAfterBurnin)
    print( "percentage of new Parameters accepted:", (moveAcceptedParams / totalMovesParams) * 100, "%")
    print( "percentage of Tree moves accepted:", (moveAcceptedTrees / totalMovesTrees) * 100, "%")
    
    if(moveProbsParams[0] != 0):
        print( "best value for overdispersion_wt: " , bestParams[0])
        print( "best value for overdispersion_mut: " , bestParams[1])
        print( "best value for  dropout: " , bestParams[2])
        print( "best value for  prior_p_mutation: " , bestParams[3])
        print( "best log score for (Tree, Params): " , bestScore)

    return sample, sampleParams, optimalTreelist, bestParams
