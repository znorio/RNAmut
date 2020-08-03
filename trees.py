# Functions used for initializing and optimizing cell lineage trees.
# The tree consists of the possible mutation sites. The cells are attached to the tree.
# A cell attached to the root has no mutation in any of the mutation sites.
# A cell attached to another part of the tree has the mutation it is attached to and all
# mutations of the respective parents.

import numpy as np
import random


# Converts Prüfer code to parent vector
def prüfer_to_parent(code, codelen):
    """
    Args:
        code    - prüfer code (list)
        codelen - length of Prüfer code (int)
        
    Returns:
        par_vec - parent vector (list)
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

    # the last two remaining nodes treated seperately
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

                
# Creates a random parent vector -> This is used to start the tree optimization with a random tree
def getRandParentVec(n):
    """
    Arg:
        n - length of parent vector (int)
        
    Returns:
        Parent vector (list)
    """
    randCode = []
    codelen = n - 1
    
    for i in range(codelen):                         # length of Prüfer code
        randCode.append(random.randint(0, n))               # random Prüfer code with n+1 nodes
        
    return prüfer_to_parent(randCode, codelen)


# Determines ancestor matrix from parent vector
def parentVector2ancMatrix(parVec, n):
    """
    Args:
        parVec    - parent vector (list)
        n         - length of parent vector (int)
        
    Returns:
        ancMatrix - ancestor matrix (numpy array)
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


# Is used in the Metropolis-Hastings algorithm to propose new cell lineage trees similar to the current tree
def proposeNewTree(moveProbsParams, ancMatrix, currTreeParentVec, num_mut):
    """
    Args:
        moveProbsParams   - determines the weights of the three move types (prune&re-attach, swap node labels, swap subtrees) (list)
        ancMatrix         - ancestor matrix of current parent vector (numpy array)
        currTreeParentVec - parent vector of current tree (list)
        num_mut           - number of mutation sites
        
    Returns:
        propTreeParentVec - parent vector of proposal tree (list)
    """
    moveType = random.choices([1,2,3], weights = (moveProbsParams[1], moveProbsParams[2], moveProbsParams[3]), k = 1)[0]
    
    if (moveType == 3):  # swap two subtrees in different lineages
        
        swapNodes = np.random.choice(num_mut, 2, replace=False)
        if (ancMatrix[swapNodes[1]][swapNodes[0]] == 0) and (ancMatrix[swapNodes[0]][swapNodes[1]] == 0):

            propTreeParentVec =  currTreeParentVec
            propTreeParentVec[swapNodes[1]] =  currTreeParentVec[swapNodes[0]]
            propTreeParentVec[swapNodes[0]] =  currTreeParentVec[swapNodes[1]]
            
        else:
            moveType = 1
            
    if (moveType == 1):     # prune and re-attach 
        nodeToMove = random.randrange(num_mut)   # pick a node to move with its subtree
        possibleParents = []

        for i in range(num_mut):
            if ancMatrix[nodeToMove][i] == 0:
                possibleParents.append(i)                        # possible attachment points
                
        newParent = random.choice(possibleParents + [num_mut])   # randomly pick a new parent among available nodes, root (num_mut + 1) is also possible parent
                                           
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
        
        if(propTreeParentVec[switchNodes[1]] == switchNodes[1]):     # if one is the parent of the other
            propTreeParentVec[switchNodes[1]] = switchNodes[0]
        
        if(propTreeParentVec[switchNodes[0]] == switchNodes[0]):
            propTreeParentVec[switchNodes[0]] = switchNodes[1]

    return propTreeParentVec
