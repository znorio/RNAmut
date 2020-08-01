import numpy as np
import random


def prüfer_to_parent(code, codelen):
    """
    converts Prüfer code to parent vector
    
    Args:
        code    - Prüfer code (list)
        codelen - length of Prüfer code (int)
        
    Returns:
        - Parent vector (list)
    """
    root = codelen + 2        # same as node count
    par_vec = [0] * (codelen + 1)
    baum = []

    for s in range(codelen):
        comb = code + baum
        for c in range(codelen + 1):
            si = 0
            for l in comb:
                if (c == l):
                    si = 1
                    break
            if (si == 0):
                break

        baum.append(c)
        par_vec[c] = code.pop(0)

    # the last two remaining nodes seperately
    last = []

    for l in range(root):
        si = 0
        for b in baum:
            if (l == b):
                si = 1
                break
        if si == 0:
            last.append(l)

    par_vec[last[0]] = last[1]
    return par_vec

                
# Starts tree optimization with a random tree
def getRandParentVec(n):
    """
    Create random parent vector
    
    Args:
        n - length of parent vector (int)
        
    Returns:
        Parent vector (list)
    """
    randCode = []
    codelen = n - 1             
    for i in range(codelen):                         # length of Prüfer code
        randCode.append(random.randint(0, n))               # random Prüfer code with n+1 nodes
        
    return prüfer_to_parent(randCode, codelen)


def parentVector2ancMatrix(parVec, n):
    """
    Determines ancestor matrix from parent vector
        
    Args:
        parVec - parent vector (list)
        n      - length of parent vector (int)
        
    Returns:
        Ancestor matrix (numpy array)
    """
    ancMatrix = np.zeros((n,n))
    for j in range(n):
        ancMatrix[j][j] = 1     # mutation counted as it's own ancestor
     
    for i in range(n):
        a = i
        while a < n:
            if parVec[a] < n:
                ancMatrix[parVec[a]][i] = 1
            a = parVec[a]
        
    return ancMatrix

def proposeNewTree(moveProbsParams, AncMatrix, currTreeParentVec):
    
    moveType = random.choices([1,2,3], weights = (moveProbsParams[1], moveProbsParams[2], moveProbsParams[3]), k = 1)[0]

    currTreeAncMatrix = parentVector2ancMatrix(currTreeParentVec, 26)
    
    
    if (moveType == 3):  # swap two subtrees in different lineages
        
        swapNodes = np.random.choice(num_mut, 2, replace=False)
        if (AncMatrix[swapNodes[1]][swapNodes[0]] == 0) and (AncMatrix[swapNodes[0]][swapNodes[1]] == 0):

            propTreeParentVec =  currTreeParentVec
            propTreeParentVec[swapNodes[1]] =  currTreeParentVec[swapNodes[0]]
            propTreeParentVec[swapNodes[0]] =  currTreeParentVec[swapNodes[1]]
            
        else:
            moveType = 1
            
    if(moveType == 1):     # prune and re-attach 
        nodeToMove = random.randrange(num_mut)   # pick a node to move with its subtree
        possibleParents = []

        for i in range(num_mut):
            if AncMatrix[nodeToMove][i] == 0:
                possibleParents.append(i)                             # possible attachment points
                
        newParent = random.choice(possibleParents + [num_mut])                  # randomly pick a new parent among available nodes, root (num_mut + 1) is also possible parent
                                           
        propTreeParentVec =  currTreeParentVec
        propTreeParentVec[nodeToMove] = newParent
        
        
        
    if (moveType == 2):         # swap two nodes
        switchNodes = np.random.choice(num_mut, 2, replace=False)
        propTreeParentVec =  currTreeParentVec[:]
        
        for j in range(num_mut):
            
            if((currTreeParentVec[j] == switchNodes[0]) and (j != switchNodes[1])):   # change the parent of the children
                propTreeParentVec[j] = switchNodes[1]
                
            if((currTreeParentVec[j] == switchNodes[1]) and (j != switchNodes[0])):
                propTreeParentVec[j] = switchNodes[0]
                        
        propTreeParentVec[switchNodes[1]] = currTreeParentVec[switchNodes[0]]   # switch the nodes
        propTreeParentVec[switchNodes[0]] = currTreeParentVec[switchNodes[1]]
        
        if(propTreeParentVec[switchNodes[1]] == switchNodes[1]):     # if one is parent of the other
            propTreeParentVec[switchNodes[1]] = switchNodes[0]
        
        if(propTreeParentVec[switchNodes[0]] == switchNodes[0]):
            propTreeParentVec[switchNodes[0]] = switchNodes[1]

            
    return propTreeParentVec
