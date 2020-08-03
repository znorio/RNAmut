# Functions to convert read counts to mutation probabilities and to calculate the paramter and tree log-scores

import math
import numpy as np


# Convert read counts to mutation probabilities
def calculate_pmat(overdispersion_wt, overdispersion_mut, dropout, prior_p_mutation, frequency_of_nucleotide, sequencing_error_rate, num_mut, num_cells, alt, ref):
    """
    Args:
        overdispersion_wt       - overdispersion wildtype (non-mutated case) (float)
        overdispersion_mut      - overdispersion mutated case (float)
        dropout                 - dropout rate (float)
        prior_p_mutation        - prior probability of a mutation occurring (float)
        frequency_of_nucleotide - expected allele frequency (float)
        sequencing_error_rate   - sequencing error rate (float)
        
    Returns:
        pmat                    - mutation probabilities (numpy array)
    """
    alpha_wt = overdispersion_wt * sequencing_error_rate
    beta_wt = overdispersion_wt * (1 - sequencing_error_rate)

    alpha_mut = overdispersion_mut * (frequency_of_nucleotide - 1/3 * sequencing_error_rate) 
    beta_mut = overdispersion_mut * (1 - (frequency_of_nucleotide - 1/3 * sequencing_error_rate))

    pmat = np.zeros((num_mut, num_cells))
    
    # alpha_wt + beta_wt = overdispersion_wt
    # alpha_mut + beta_mut = overdispersion_mut
    gamma_const_nor = math.lgamma(overdispersion_wt) - math.lgamma(alpha_wt) - math.lgamma(beta_wt)
    gamma_const_mut = math.lgamma(overdispersion_mut) - math.lgamma(alpha_mut) - math.lgamma(beta_mut)
    
    # sc.gammaln(c + 1) and - sc.gammaln(c - s + 1) and - sc.gammaln(s + 1) are the same for all terms 
    # -> they cancel out and it is not necessary to calculate them
    for i in range(num_mut):
        for j in range(num_cells):    

            c = ref[i][j] + alt[i][j]  # total_coverage
            s = alt[i][j]  # nucleotide_counts 
            
            if (c == 0):
                pmat[i][j] = prior_p_mutation
                continue
            
            gamma_core =  gamma_const_nor - math.lgamma(c + overdispersion_wt)
            
            p_nor = math.lgamma(s + alpha_wt) + math.lgamma(c - s + beta_wt) + gamma_core
            
            p_nor = (math.e**(p_nor))
            
            p_mut = (dropout/2) * (p_nor) \
                    + (dropout/2) * math.e**(math.lgamma(s + beta_wt) + math.lgamma(c - s + alpha_wt) + gamma_core) \
                    + (1 - dropout) * math.e**(math.lgamma(s + alpha_mut) + math.lgamma(c - s + beta_mut) \
                    + gamma_const_mut - math.lgamma(c + overdispersion_mut))

            pmat[i][j] = (p_mut * prior_p_mutation)/((p_mut * prior_p_mutation) + (p_nor * (1 - prior_p_mutation)))
            
    pmat[pmat > 0.9999] = 0.9999 # highest allowed probability of mutation
    pmat[pmat < 0.0001] = 0.0001 # lowest
    return pmat


# Logarithm of probability density function (pdf) of the beta distribution
def log_pdf(a,b,x):
    """
    Args:
        a - alpha (float)
        b - beta (float)
        x - range: (0,1) / probability density is determined at point x (float)
        
    Returns:
        Logarithm of pdf (float)
    """
    return math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + (a - 1) * math.log(x) + (b - 1) * math.log(1 - x)


# Logarithmic tree score
# Uses the logarithmic scoring rule to determine the difference between the calculated probabilities of mutation 
# and the one zero mutation probabilities derived from the mutation tree
# Marginalization of the attachment points is possible (to get posterior distributions of the learnable parameters)
def log_scoretree(pmat, parVec, marginalization):
     """
    Args:
        pmat            - calculated probabilities of mutation (numpy array)
        parVec          - parent vector of tree (list)
        marginalization - if true the attachment points are marginalized (bool)
        
    Returns:
        log_score       - logarithmic tree score (float)
    """
    log_pmat_m = np.log(pmat)
    log_pmat_r = np.log(1 - pmat)
    
    log_score = 0

    children = [[] for v in range(num_mut + 1)]
    for q in range(num_mut):
        children[parVec[q]].append(q) 

    bf = [0] * (num_mut + 1)
    bf[0] = num_mut
    z = 0

    for w in range(num_mut + 1):
        for t in range(len(children[bf[w]])):
            z += 1
            bf[z] = children[bf[w]][t]           # determine where the mutation_sites are located in the tree 
                                                 # to later be able to add the log score from root to bottom of the tree
    
    for i in range(num_cells):
        
        score = [0] * (num_mut + 1)
        
        for j in range(num_mut):
            score[num_mut] += log_pmat_r[j,i] 
        
        for k in range(1, num_mut + 1):
                       
            node = bf[k]

            score[node] = score[parVec[node]]
            score[node] += log_pmat_m[node,i]   # step by step the mutation log score is added 
            score[node] -= log_pmat_r[node,i]   # because the mutation replaces a reference, the reference log score is substracted

        if marginalization == False:
            log_score += max(score)
            
        if marginalization == True:
            log_score += max(score) + math.log(sum(math.e ** (score - max(score))))

    return log_score


# Logarithmic parameter score
# The factorParamsLogScore determines how strongly the prior parameter distributions affect the combined tree and parameters log score
# Because a small overdispersion_wt parameter might lead to an empty tree solution, meaning none of the cells are predicted to have a mutation, 
# it is possible to strengthen the prior for this parameter with the factor_owt.
def log_scoreparams(params, maxValues, priorAlphaBetaoodp, factor_owt, factorParamsLogScore):
    """
    Args:
        params               - [overdispersion_wt, overdispersion_mut, dropout, prior_p_mutation] (list)
        maxValues            - maximum values of the parameters (list)
        priorAlphaBetaoodp   - prior alphas and betas of the parameter beta distributions (list)
        factor_owt           - strengthens/(weakens) prior of overdispersion_wt (int/float)
        factorParamsLogScore - strengthens/weakens priors (int/float)
        
    Returns:
        Logarithmic parameters score (float)
    """
    log_score = 0            
    log_score += log_pdf(priorAlphaBetaoodp[0], priorAlphaBetaoodp[1], params[0] / maxValues[0]) * factor_owt
    log_score += log_pdf(priorAlphaBetaoodp[2], priorAlphaBetaoodp[3], params[1] / maxValues[1])
    log_score += log_pdf(priorAlphaBetaoodp[4], priorAlphaBetaoodp[5], params[2] / maxValues[2])
    log_score += log_pdf(priorAlphaBetaoodp[6], priorAlphaBetaoodp[7], params[3] / maxValues[3])
    
    return log_score * factorParamsLogScore
