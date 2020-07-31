# alpha_wt + beta_wt = overdispersion_wt
# alpha_mut + beta_mut = overdispersion_mut
# sc.gammaln(c + 1) and - sc.gammaln(c - s + 1) and - sc.gammaln(s + 1) are the same for all terms so they cancel out and it is 
# not necessary to calculate them
    
def calculate_pmat(overdispersion_wt, overdispersion_mut, dropout, prior_p_mutation):
    
    frequency_of_nucleotide = 0.5 # expected allele frequency
    sequencing_error_rate = 0.01

    alpha_wt = overdispersion_wt * sequencing_error_rate
    beta_wt = overdispersion_wt * (1 - sequencing_error_rate)

    alpha_mut = overdispersion_mut * (frequency_of_nucleotide - 1/3 * sequencing_error_rate) 
    beta_mut = overdispersion_mut * (1 - (frequency_of_nucleotide - 1/3 * sequencing_error_rate))

    pmat = np.zeros((rows_ref,columns_ref - 1))
    
    gamma_const_nor = math.lgamma(overdispersion_wt) - math.lgamma(alpha_wt) - math.lgamma(beta_wt)
    gamma_const_mut = math.lgamma(overdispersion_mut) - math.lgamma(alpha_mut) - math.lgamma(beta_mut)
    
    for i in range(rows_ref):
        for j in range(1,columns_ref):    

            c = ref[i][j] + alt[i][j]  #total_coverage
            s = alt[i][j]  #nucleotide_counts 
            
            if (c == 0):
                pmat[i][j-1] = prior_p_mutation
                continue
            
            gamma_core =  gamma_const_nor - math.lgamma(c + overdispersion_wt)
            
            p_nor = math.lgamma(s + alpha_wt) + math.lgamma(c - s + beta_wt) + gamma_core
            
            p_nor = (math.e**(p_nor))
            
            p_mut = (dropout/2) * (p_nor) \
                    + (dropout/2) * math.e**(math.lgamma(s + beta_wt) + math.lgamma(c - s + alpha_wt) + gamma_core) \
                    + (1 - dropout) * math.e**(math.lgamma(s + alpha_mut) + math.lgamma(c - s + beta_mut) \
                    + gamma_const_mut - math.lgamma(c + overdispersion_mut))

            pmat[i][j-1] = (p_mut * prior_p_mutation)/((p_mut * prior_p_mutation) + (p_nor * (1 - prior_p_mutation)))
            
    pmat[pmat > 0.9999] = 0.9999
    pmat[pmat < 0.0001] = 0.0001
    return pmat
  
  
#calculate log-score
def pdf(a,b,x):
    return math.e**(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)) * x**(a-1) * (1-x)**(b-1)

def log_pdf(a,b,x):
    return math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + (a - 1) * math.log(x) + (b - 1) * math.log(1 - x)


def log_scoretree2(pmat, parVec):
    
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
            bf[z] = children[bf[w]][t]           # determine where the mutations are in the tree to later be able to add the log score from root to bottom of the tree
    
    for i in range(num_cells):
        
        score = [0] * (num_mut + 1)
        
        for j in range(num_mut):
            score[num_mut] += log_pmat_r[j,i] 
        
        for k in range(1, num_mut + 1):
                       
            node = bf[k]

            score[node] = score[parVec[node]]
            score[node] -= log_pmat_r[node,i]     # step by step the mutation log score is added and because the mutation replaces a reference, the reference log score is substracted
            score[node] += log_pmat_m[node,i]

        log_score += max(score)

    return log_score

if marginalization == True:
    
    def log_scoretree2(pmat, parVec):

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
                bf[z] = children[bf[w]][t]           # determine where the mutations are in the tree to later be able to add the log score from root to bottom of the tree

        for i in range(num_cells):

            score = [0] * (num_mut + 1)

            for j in range(num_mut):
                score[num_mut] += log_pmat_r[j,i] 

            for k in range(1, num_mut + 1):

                node = bf[k]

                score[node] = score[parVec[node]]
                score[node] -= log_pmat_r[node,i]     # step by step the mutation log score is added and because the mutation replaces a reference, the reference log score is substracted
                score[node] += log_pmat_m[node,i]

            log_score += max(score) + math.log(sum(math.e ** (score - max(score))))

        return log_score

def log_scoreparams(Params, maxValues, ab):

    log_score = 0            
    log_score += log_pdf(ab[0], ab[1], Params[0] / maxValues[0]) * factor_owt
    log_score += log_pdf(ab[2], ab[3], Params[1] / maxValues[1])
    log_score += log_pdf(ab[4], ab[5], Params[2] / maxValues[2])
    log_score += log_pdf(ab[6], ab[7], Params[3] / maxValues[3])
    
    return log_score * factorParamsLogScore
