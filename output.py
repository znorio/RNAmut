import numpy as np

def getAttachmentPoints(parVec, Params, n, m):
        
    pmat = calculate_pmat(Params[0], Params[1], Params[2], Params[3])
    
    log_pmat_m = np.log(pmat)
    log_pmat_r = np.log(1 - pmat)
    
    attachmentPoints = []
    
    children = [[] for v in range(n + 1)]
    for q in range(n):
        children[parVec[q]].append(q) 

    bf = [0] * (n + 1)
    bf[0] = n
    z = 0

    for w in range(n + 1):
        for t in range(len(children[bf[w]])):
            z += 1

            bf[z] = children[bf[w]][t]           # determine where the mutations are in the tree to later be able to add the log score from root to bottom of the tree
    
    for i in range(m):
        
        score = [0] * (n + 1)
        
        for j in range(num_mut):
            score[n] += log_pmat_r[j,i] 
        
        for k in range(1, num_mut + 1):
                       
            node = bf[k]

            score[node] = score[parVec[node]]
            score[node] -= log_pmat_r[node,i]     # step by step the mutation log score is added and because the mutation replaces a reference, the reference log score is substracted
            score[node] += log_pmat_m[node,i]

        attachmentPoints.append(score.index(max(score)))
    return attachmentPoints
  
    
def oneZeroMut(Params, parVec, n, m):
    
    ancMatrix = parentVector2ancMatrix(parVec, n)
    
    attachmentPoints = getAttachmentPoints(parVec, Params, n, m)

    mat = np.zeros((n, m))

    for m, j in enumerate(attachmentPoints):
        if(j < n):
            for k in range(n):
                if (ancMatrix[k][j] == 1):
                    mat[k][m] = 1

    return mat
  

def graphviz(Params, parentVector, n, m):

    gene_names.append("Root")
    
    gv = "digraph G {\n"
    gv += "node [color=deeppink4, style=filled, fontcolor=white];\n"

    for i in range(n):
        # The parantheses around the gene names help displaying them if they contain special characters
        gv += ("\"" + gene_names[parentVector[i]] + "\"" + " -> "  + "\"" + gene_names[i]  + "\"" + ";\n" ) 

    gv += "node [color=lightgrey, style=filled, fontcolor=black];\n"
                            
    attachmentPoints = getAttachmentPoints(parentVector, Params, n, m)
    
    for y, a in enumerate(attachmentPoints):
        gv += "\"" + gene_names[a] + "\"" + " -> s"  + str(y + 1) + ";\n"

    gv += "}\n"
    return gv
