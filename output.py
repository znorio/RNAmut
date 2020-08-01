# Functions used to create the desired output files

import numpy as np
from .scores import calculate_pmat
from .trees import parentVector2ancMatrix

# Determines the optimal attachment points of the cells to the mutation tree
def getAttachmentPoints(parVec, Params, num_mut, num_cells, frequency_of_nucleotide, sequencing_error_rate):
    """
    Args:
        parVec                  - Parent vector (list)
        Params                  - [overdispersion_wt, overdispersion_mut, dropout, prior_p_mutation] (list)
        num_mut                 - Number of mutation sites (int)
        num_cells               - Number of cells (int)
        frequency_of_nucleotide - Expected allele frequency (float)
        sequencing_error_rate   - Sequencing error rate (float)
        
    Returns:
        Optimal attachment points (list)
    """    
    pmat = calculate_pmat(Params[0], Params[1], Params[2], Params[3], frequency_of_nucleotide, sequencing_error_rate )
    
    log_pmat_m = np.log(pmat)
    log_pmat_r = np.log(1 - pmat)
    
    attachmentPoints = []
    
    children = [[] for v in range(num_mut + 1)]
    for q in range(num_mut):
        children[parVec[q]].append(q) 

    bf = [0] * (num_mut + 1)
    bf[0] = num_mut
    z = 0

    for w in range(num_mut + 1):
        for t in range(len(children[bf[w]])):
            z += 1

            bf[z] = children[bf[w]][t]           # determine where the mutations are located in the tree to later be able to 
                                                 # add the log score from root to bottom of the tree
    
    for i in range(m):
        
        score = [0] * (num_mut + 1)
        
        for j in range(num_mut):
            score[num_mut] += log_pmat_r[j,i] 
        
        for k in range(1, num_mut + 1):
                       
            node = bf[k]
            
            # Step by step the mutation log score is added and because the mutation replaces a reference, 
            # the reference log score is substracted.
            score[node] = score[parVec[node]]
            score[node] -= log_pmat_r[node,i] 
            score[node] += log_pmat_m[node,i] 

        attachmentPoints.append(score.index(max(score)))
    return attachmentPoints
  
    
# Calculates a one zero mutation matrix
# All cells are optimally attached to the muatation tree
def oneZeroMut(Params, parVec, num_mut, num_cells, frequency_of_nucleotide, sequencing_error_rate):
    """
    Args:
        Params                  - [overdispersion_wt, overdispersion_mut, dropout, prior_p_mutation] (list)
        parVec                  - Parent vector of tree (list)
        num_mut                 - Number of mutation sites (int)
        num_cells               - Number of cells (int)
        frequency_of_nucleotide - Expected allele frequency (float)
        sequencing_error_rate   - Sequencing error rate (float)
        
    Returns:
        One zero matrix (numpy array)
    """    
    ancMatrix = parentVector2ancMatrix(parVec, num_mut)
    
    attachmentPoints = getAttachmentPoints(parVec, Params, num_mut, num_cells, frequency_of_nucleotide, sequencing_error_rate)

    mat = np.zeros((num_mut, num_cells))

    for m, j in enumerate(attachmentPoints):
        if(j < num_mut):
            for k in range(num_mut):
                if (ancMatrix[k][j] == 1):
                    mat[k][m] = 1

    return mat
  

# Creates a graphviz file
def graphviz(Params, parVec, num_mut, num_cells, frequency_of_nucleotide, sequencing_error_rate):
    """
    Args:
        Params                  - [overdispersion_wt, overdispersion_mut, dropout, prior_p_mutation] (list)
        parVec                  - Parent vector of tree (list)
        num_mut                 - Number of mutation sites (int)
        num_cells               - Number of cells (int)
        frequency_of_nucleotide - Expected allele frequency (float)
        sequencing_error_rate   - Sequencing error rate (float)
        
    Returns:
        Graphviz file (string)
    """    
    gene_names.append("Root")
    
    gv = "digraph G {\n"
    gv += "node [color=deeppink4, style=filled, fontcolor=white];\n"

    for i in range(num_mut):
        # The parantheses around the gene names help displaying them if they contain special characters
        gv += ("\"" + gene_names[parVec[i]] + "\"" + " -> "  + "\"" + gene_names[i]  + "\"" + ";\n" ) 

    gv += "node [color=lightgrey, style=filled, fontcolor=black];\n"
                            
    attachmentPoints = getAttachmentPoints(parVec, Params, num_mut, num_cells, frequency_of_nucleotide, sequencing_error_rate)
    
    for y, a in enumerate(attachmentPoints):
        gv += "\"" + gene_names[a] + "\"" + " -> s"  + str(y + 1) + ";\n"

    gv += "}\n"
    return gv
