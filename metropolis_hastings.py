import numpy as np
import math
import random
from random import gauss


def acceptance(x_logs, x_new_logs, gamma):
    if x_new_logs > x_logs:
        return True
    else:
        accept = np.random.uniform(0,1)
        return (accept < (math.e**((x_new_logs - x_logs) * gamma)))    # as long as gamma = 1 it has no influence
      
    
def sample_multivariate_normal(x, cov):
    
    n = len(x)
    x_new = [0,0,0,0]
    chol = np.linalg.cholesky(cov)
    un = [gauss(0,1),gauss(0,1),gauss(0,1),gauss(0,1)]
    for i in range(n):
        for j in range(n):
            x_new[i] += float(chol[i][j]) * un[j]
        x_new[i] += x[i]
    return x_new
  

def runMCMCoodp(rep, loops, oodp, priorAlphaBetaoodp, gamma, moveProbsParams, sampleStep, initialPeriod, covDiagonal, maxValues, minValues, burnInPhase, decVar):
    
    optStatesAfterBurnIn = 0
    n = len(oodp)  # number of parameters
    burnInPhase = 0.25
    burnIn = loops * burnInPhase
    parentVectorSize = num_mut
    eps = 0.00000000001
    optimalTreelist = []
    sample = []
    sampleParams = []
    newtree = True
    bestScore = bestTreeLogScore = -1000000
    
    for r in range(rep):       # starts over, but keeps sampling, bestScore, bestTreeLogScore
        
        sum_parameters = np.array(oodp)
        av_params_t = oodp
        
        currTreeParentVec = getRandParentVec(parentVectorSize)     # start MCMC with random tree
        currTreeAncMatrix =  parentVector2ancMatrix(currTreeParentVec, parentVectorSize)
        currParams = oodp
        pmat = calculate_pmat(currParams[0], currParams[1], currParams[2], currParams[3])
        currpmat = pmat
        currTreeLogScore = log_scoretree2(pmat, currTreeParentVec)
        currParamsLogScore = log_scoreparams(currParams, maxValues, priorAlphaBetaoodp)
        currScore = currTreeLogScore + currParamsLogScore
        if currScore > bestScore:
            bestScore = currScore
            bestTreeLogScore = currTreeLogScore
            bestParams = currParams

        moveAcceptedParams = 0
        totalMovesParams = lastmoveAcceptedParams = 0
        moveAcceptedTrees = 0
        totalMovesTrees = lasttotalMoveParams = 0
        
        
        t = 0

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

                if (propParams[0] < (minValues[0] + 0.0001)) or (propParams[0] > (maxValues[0] - 0.0001)) or (propParams[1] < (minValues[1] + 0.0001)) or (propParams[1] > (maxValues[1] - 0.0001)) or \
                        (propParams[2] < (minValues[2] + 0.0001)) or (propParams[2] > (maxValues[2] - 0.0001)) or (propParams[3] < (minValues[3] + 0.0001)) or (propParams[3] > (maxValues[3] - 0.0001)):
                    continue #if the proposed parameters are out of range, they are not considered
                
                propParamsLogScore = log_scoreparams(propParams, maxValues, priorAlphaBetaoodp)
                
                pmat = calculate_pmat(propParams[0], propParams[1], propParams[2], propParams[3])

                propTreeLogScore = log_scoretree2(pmat, currTreeParentVec)
                
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
                    sum_parameters += currParams
                    av_params_t = sum_parameters / t
                    
                if (l > initialPeriod):

                    cov_mat = (t - 1)/t * (cov_mat  +  1/t * np.dot(np.transpose([currParams - av_params_t]), [currParams - av_params_t]) * decVar) + eps * np.identity(4)

        
            else:                 # if the move changes the tree not the parameters

                totalMovesTrees += 1
                
                propTreeParentVec = proposeNewTree(moveProbsParams, currTreeAncMatrix[:], currTreeParentVec[:]) 
                
                propTreeLogScore = log_scoretree2(currpmat, propTreeParentVec)
                
                if acceptance(currTreeLogScore, propTreeLogScore, gamma):                   # the proposed tree is accepted
                    
                    moveAcceptedTrees += 1
          
                    currTreeAncMatrix = parentVector2ancMatrix(propTreeParentVec, parentVectorSize)
        
                    currTreeParentVec = propTreeParentVec                                
                    currTreeLogScore  = propTreeLogScore                     
                    currScore = currTreeLogScore + currParamsLogScore
                    
            if currScore > bestScore:       # create a list with optimal trees and parameters
                optimalTreelist = []
                optimalTreelist.append([currTreeParentVec, currParams])

            if (currScore >= bestScore - eps) and (currScore <= bestScore + eps): # don't know if truly necessary, but == usually gave optStatesAfterBurnIn = 1
                
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
