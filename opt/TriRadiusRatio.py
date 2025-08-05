import numpy as np
import time
from scipy.sparse import csc_matrix, csr_matrix, spdiags, triu, tril, find, hstack, eye
from scipy.sparse.linalg import cg, inv, dsolve
from scipy.linalg import norm

class TriRadiusRatio():
    def __init__(self, mesh):
        self.mesh = mesh

    def get_free_node_info(self):
        NN = self.mesh.number_of_nodes()
        isBdNode = self.mesh.ds.boundary_node_flag()
        isFreeNode = np.ones((NN, ), dtype=np.bool)
        isFreeNode[isBdNode] = False
        return isFreeNode

    def get_quality(self):
        node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        NC = self.mesh.number_of_cells() 
        localEdge = self.mesh.ds.local_edge()
        v = [node[cell[:,j],:] - node[cell[:,i],:] for i,j in localEdge]
        l2 = np.zeros((NC, 3))
        for i in range(3):
            l2[:, i] = np.sum(v[i]**2, axis=1)
        l = np.sqrt(l2)
        p = l.sum(axis=1)
        q = l.prod(axis=1)
        area = np.cross(v[2], -v[1])/2
        quality = p*q/(16*area**2)
        return quality

    def get_iterate_matrix(self):
        node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')
        NN = self.mesh.number_of_nodes()
        NC = self.mesh.number_of_cells() 
        localEdge = self.mesh.ds.local_edge()
        v = [node[cell[:,j],:] - node[cell[:,i],:] for i,j in localEdge]
        l2 = np.zeros((NC, 3))
        for i in range(3):
            l2[:, i] = np.sum(v[i]**2, axis=1)
        l = np.sqrt(l2)
        p = l.sum(axis=1)
        q = l.prod(axis=1)
        area = np.cross(v[2], -v[1])/2
        mu = p*q/(16*area**2)

        c = mu[:, None]*(1/(p[:, None]*l) + 1/l2)
        val = np.concatenate((
            c[:, [1, 2]].sum(axis=1), -c[:, 2], -c[:, 1],
            -c[:, 2], c[:, [0, 2]].sum(axis=1), -c[:, 0],
            -c[:, 1], -c[:, 0], c[:, [0, 1]].sum(axis=1)))
        I = np.einsum('ij, k->ijk', cell, np.ones(3))
        J = I.swapaxes(-1, -2)
        A = csr_matrix((val, (I.flat, J.flat)), shape=(NN, NN))

        cn = mu/area;
        val = np.concatenate((-cn, cn, cn, -cn, -cn, cn))
        I = np.concatenate((
            cell[:, 0], cell[:, 0], 
            cell[:, 1], cell[:, 1],
            cell[:, 2], cell[:, 2]))
        J = np.concatenate((
            cell[:, 1], cell[:, 2], 
            cell[:, 0], cell[:, 2],
            cell[:, 0], cell[:, 1]))
        B = csr_matrix((val, (I, J)), shape=(NN, NN))
        return  (A, B)

    def grad(self,functype=1):
        NC = self.mesh.number_of_cells()
        NN = self.mesh.number_of_nodes()
        node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')


        idxi = cell[:, 0]
        idxj = cell[:, 1] 
        idxk = cell[:, 2] 

        v0 = node[idxk] - node[idxj]
        v1 = node[idxi] - node[idxk]
        v2 = node[idxj] - node[idxi]

        area = 0.5*(-v2[:, [0]]*v1[:, [1]] + v2[:, [1]]*v1[:, [0]])
        l2 = np.zeros((NC, 3), dtype=np.float64)
        l2[:, 0] = np.sum(v0**2, axis=1)
        l2[:, 1] = np.sum(v1**2, axis=1)
        l2[:, 2] = np.sum(v2**2, axis=1)
        l = np.sqrt(l2)
        p = l.sum(axis=1, keepdims=True)
        q = l.prod(axis=1, keepdims=True)
        mu = p*q/(16*area**2)
        if functype==1:
            c = mu*(1/(p*l) + 1/l2)
        if functype==2:
            c = 2*mu**2*(1/(p*l) + 1/l2)
        val = np.concatenate((
            c[:, [1, 2]].sum(axis=1), -c[:, 2], -c[:, 1],
            -c[:, 2], c[:, [0, 2]].sum(axis=1), -c[:, 0],
            -c[:, 1], -c[:, 0], c[:, [0, 1]].sum(axis=1)))
        I = np.concatenate((
            idxi, idxi, idxi,
            idxj, idxj, idxj,
            idxk, idxk, idxk))
        J = np.concatenate((idxi, idxj, idxk))
        J = np.concatenate((J, J, J))
        A = csr_matrix((val, (I, J)), shape=(NN, NN))

        if functype==1:
            cn = mu/area
        if functype==2:
            cn = 2*mu**2/area
        cn.shape = (cn.shape[0],)
        val = np.concatenate((-cn, cn, cn, -cn, -cn, cn))
        I = np.concatenate((idxi, idxi, idxj, idxj, idxk, idxk))
        J = np.concatenate((idxj, idxk, idxi, idxk, idxi, idxj))
        B = csr_matrix((val, (I, J)), shape=(NN, NN))

        return (A/NC, B/NC)

    def iterate_solver(self,method='jacobi',functype=1):
        NC = self.mesh.number_of_cells()
        node = self.mesh.entity('node')
        isFreeNode = self.get_free_node_info()        
        q = np.zeros((2, NC),dtype=np.float64)
        q[0] = self.get_quality()
        if method=='jacobi':
            for i in range(0,500):
                A, B = self.grad(functype=functype)
                node=self.Jacobi(node, A, B, isFreeNode)		
                            ## count quality
                q[1] = self.get_quality()
                minq = np.min(q[1])
                avgq = np.mean(q[1])
                 
                if np.max(np.abs(q[1]-q[0]))<1e-4:
                    print("The number of Jacobi iterations is %d"%(i+1))
                    break
                q[0] = q[1]
            print("The number of Jacobi iterations is 500")
            
        if method=='Bjacobi':
            for i in range(0,500):
                A, B = self.grad(functype=functype)
                node = self.BlockJacobi(node, A, B, isFreeNode)
                            ## count quality
                q[1] = self.get_quality()
                minq = np.min(q)
                avgq = np.mean(q)
                maxq = np.max(q)
                print('minq=',minq,'avgq=',avgq, 'maxq=',maxq)
                
                if np.max(np.abs(q[1]-q[0]))<1e-8:
                    print("The number of Bjacobi iterations is %d"%(i+1))
                    break
                q[0] = q[1]
        return node

    def Jacobi(self, node, A, B, isFreeNode):
        NN = self.mesh.number_of_nodes() 
        D = spdiags(1.0/A.diagonal(), 0, NN, NN)
        C = -(triu(A, 1) + tril(A, -1))
        X = D*(C*node[:, 0] - B*node[:, 1])
        Y = D*(B*node[:, 0] + C*node[:, 1])
        p = np.zeros((NN, 2)) 
        p[isFreeNode, 0] = X[isFreeNode] - node[isFreeNode, 0]
        p[isFreeNode, 1] = Y[isFreeNode] - node[isFreeNode, 1]
        node +=100*p/NN
        return node


    def BlockJacobi(self, node, A, B, isFreeNode):
        NN = self.mesh.number_of_nodes() 
        isBdNode = np.logical_not(isFreeNode)
        newNode = np.zeros((NN, 2), dtype=np.float64)
        newNode[isBdNode, :] = node[isBdNode, :]        
        b = -B*node[:, 1] - A*newNode[:, 0]
        newNode[isFreeNode, 0], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
                b[isFreeNode], x0=node[isFreeNode, 0], tol=1e-6)
        b = B*node[:, 0] - A*newNode[:, 1]
        newNode[isFreeNode, 1], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
                b[isFreeNode], x0=node[isFreeNode, 1], tol=1e-6)
        node[isFreeNode, :] = newNode[isFreeNode, :]
        return node


