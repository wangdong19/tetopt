import numpy as np
from scipy.spatial import Delaunay
from mesh.tetrahedron_mesh import TetrahedronMesh

class ODT:
    def __init__(self,mesh):
        self.mesh=mesh
    
    def odt_iterate(self,model:int=0):
        node = self.mesh.node.copy()
        cell = self.mesh.entity("cell")
        alpha = 1

        NN = self.mesh.number_of_nodes()
        NC = self.mesh.number_of_cells()

        isBdNode = self.mesh.ds.boundary_node_flag()
        isBdCell = self.mesh.ds.boundary_cell_flag()
        isFreeNode = ~isBdNode

        cm = self.mesh.entity_measure("cell") # 单元体积
        cb = self.mesh.entity_barycenter('cell') # 单元重心
        cc = self.mesh.circumcenter() # 单元外心
        cc[isBdCell] = cb[isBdCell]
        cc = cm[...,None]*cc

        newNode = np.zeros((NN,3),dtype=np.float64)
        patch_area = np.zeros(NN,dtype=np.float64)

        np.add.at(newNode,cell,np.broadcast_to(cc[:,None],(NC,4,3)))
        np.add.at(patch_area,cell,np.broadcast_to(cm[:,None],(NC,4)))

        newNode[isBdNode] = node[isBdNode]
        if np.sum(patch_area==0)>0:
            isFreeNode[patch_area==0]=False
        newNode[isFreeNode] = newNode[isFreeNode]/patch_area[...,None][isFreeNode]
        self.mesh.node = newNode
        cm = self.mesh.entity_measure("cell")

        while np.sum(cm<=0)>0:
            alpha = alpha/2
            self.mesh.node[isFreeNode] = (1-alpha)*node[isFreeNode]+alpha*newNode[isFreeNode]
            cm = self.mesh.entity_measure("cell")
        if model==0:
            self.mesh = self.edge_swap(self.mesh)
        elif model==1:
            self.mesh = self.edge_swap_Lshape(self.mesh)
        elif model==2:
            self.mesh = self.edge_swap_is(self.mesh)

        cell = self.mesh.entity('cell')
        cm = self.mesh.entity_measure("cell")
        col1 = cell[cm<0,2].copy()
        cell[cm<0,2] = cell[cm<0,3]
        cell[cm<0,3] = col1


    def edge_swap(self,mesh):
        node = self.mesh.node
        tet = Delaunay(node)
        mesh = TetrahedronMesh(node,tet.simplices)
        return mesh

    def edge_swap_is(self,mesh):
        node = self.mesh.node
        tet = Delaunay(node)
        mesh = TetrahedronMesh(node,tet.simplices)
        center = np.array([
            [1.0,0.0,0.0],[-1.0,0.0,0.0],[0.5,0.866025403784439,0.0],
            [-0.5,0.866025403784439,0.0],[0.5,-0.866025403784439,0.0],
            [-0.5,-0.866025403784439,0.0],[2.0, 0.0,0.0],[-2.0,0.0,0.0],
            [1.0,1.73205080756888,0.0],[-1.0,1.73205080756888,0.0],
            [-1.0,-1.73205080756888,0.0],[1.0,-1.73205080756888,0.0]],dtype=np.float64)
        cell = mesh.entity('cell')
        barycenter = mesh.entity_barycenter('cell')
        remove_flag = np.ones(len(cell),dtype=np.bool_)
        for i in range(12):
            remove_flag[np.sum((barycenter-center[i])**2,axis=-1)<0.49]=False
        cell = cell[~remove_flag]
        mesh = TetrahedronMesh(node,cell)
        return mesh

    def edge_swap_Lshape(self,mesh):
        node = self.mesh.node
        tet = Delaunay(node)
        mesh = TetrahedronMesh(node,tet.simplices)
        cell = mesh.entity('cell')
        barycenter = mesh.entity_barycenter('cell')
        center = np.array([[0.75,0.75,0.75]],dtype=np.float64)
        remove_flag1 = np.sum((barycenter-center)**2,axis=-1)<=0.0081
        remove_flag2 = (barycenter[:,0]<=0.5)&(barycenter[:,1]<=0.5)&(barycenter[:,2]<=0.5)
        cell = cell[~remove_flag1 & ~remove_flag2]
        #cell = cell[~remove_flag2]
        mesh = TetrahedronMesh(node,cell)
        return mesh
