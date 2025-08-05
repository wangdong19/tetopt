import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator

import time
import MeshModel
from opt.TetMeshProblem import TetMeshProblem
from opt.TriRadiusRatio import TriRadiusRatio
from mesh.tetrahedron_mesh import TetrahedronMesh
from opt.odt import ODT
from opt.PLBFGSAlg import PLBFGS
from opt.PNLCGAlg import PNLCG

parser = argparse.ArgumentParser(description=
        '''
        Radius Ratio Optimization Example
        ''')
parser.add_argument('--exam',
        default='sp', type=str,
        help='''
        The default example is a sphere,
        sp for sphere, ls for L-shaped domain, 
        tsp for intersecting spheres,
        tri for triangle domain, 
        sq for square hole, 
        triodt for triangle domain with ODT,
        sqodt for square hole with ODT. Default is sp.
        ''')

parser.add_argument('--nodt',
        default= 0, type=int,
        help='odt iteration times, not optimized by default')

parser.add_argument('--optmethod',
        default= 'LBFGS', type=str,
        help='Optimization algorithm, default LBFGS, optional NLCG')

parser.add_argument('--p', 
            default=0, type=int,
            help='''Preprocessor type, 0 means do not use the preprocessor, 
                  1 means use the preprocessor''')

args = parser.parse_args()

def test_unit_sphere(h=0.1, nodt=0, optmethod='LBFGS', P=0):
    mesh = MeshModel.unit_sphere(h)

    q = mesh.cell_quality()
    NC = mesh.number_of_cells()
    ylim = int(0.1*NC)
    
    q = mesh.cell_quality()
    show_mesh_quality(q,ylim=2500,title='initial mesh quality') 
    opt = ODT(mesh)
    t1 = time.time() 
    if nodt != 0:
        print("ODT optimization in progress...")
    for i in range(nodt):
        q0 = opt.mesh.cell_quality()
        opt.odt_iterate(model=0)
        q1 = opt.mesh.cell_quality()
        if np.abs(np.max(q1)-np.max(q0))<1e-8:
            print("The number of ODT iterations is %d"%(i+1))
            break
    t2 = time.time()
    mesh = opt.mesh
    if nodt !=0:
        print("The number of ODT iterations is %d"%(nodt))
        print("ODT iteration time:",t2-t1)    
        q = mesh.cell_quality()
        show_mesh_quality(q,ylim=2500,title='odt optimization') 
    
    problem = TetMeshProblem(mesh)

    NDof = len(problem.x0)
    if P==1:
        problem.Preconditioner = problem.preconditioner()
    elif P==0:
        problem.Preconditioner = None
    problem.StepLength = 1.0
    problem.FunValDiff = 1e-4
    problem.MaxIters = 100
    problem.Print = True
    
    t1 = time.time()
    if optmethod == 'LBFGS':
        opt = PLBFGS(problem)
        x, f, g,flag = opt.run()
    if optmethod == 'NLCG':
        opt = PNLCG(problem)
        x, f, g = opt.run()
    t2 = time.time()
    print('optimize time:',t2-t1)
    #If you need to output a visualization file, use to_vtk method
    #mesh.to_vtk(fname='opt_sphere.vtu') 
    node = mesh.entity('node')
    isFreeNode = ~mesh.ds.boundary_node_flag()
    n = len(x)//3 
    node[isFreeNode,0] = x[:n]
    node[isFreeNode,1] = x[n:2*n]
    node[isFreeNode,2] = x[2*n:]

    q = mesh.cell_quality()
    show_mesh_quality(q,ylim=2500,title='radius ratio energy optimization')
    return mesh

def test_LShape(h=0.05, nodt=0, optmethod='LBFGS', P=0):
    from opt.TetMeshProblem import TetMeshProblemls
    mesh = MeshModel.LShape(h)

    q = mesh.cell_quality()
    NC = mesh.number_of_cells()
    q = mesh.cell_quality()
    show_mesh_quality(q,ylim=4000,title='initial mesh quality') 

    opt = ODT(mesh)
    t1 = time.time() 
    if nodt != 0:
        print("ODT optimization in progress...")
    for i in range(nodt):
        q0 = opt.mesh.cell_quality()
        opt.odt_iterate(model=1)
        q1 = opt.mesh.cell_quality()
        if np.abs(np.max(q1)-np.max(q0))<1e-8:
            print("The number of ODT iterations is %d"%(i+1))
            break
    t2 = time.time()
    mesh = opt.mesh     
    if nodt !=0:
        print("The number of ODT iterations is %d"%(nodt))
        print("ODT iteration time:",t2-t1)
        q = mesh.cell_quality()
        show_mesh_quality(q,ylim=4000,title='odt optimization') 

    problem = TetMeshProblemls(mesh)

    NDof = len(problem.x0)
    if P==1:
        problem.Preconditioner = problem.preconditioner()
    elif P==0:
        problem.Preconditioner = None
    problem.StepLength = 1.0
    problem.FunValDiff = 1e-4
    
    t1 = time.time()
    if optmethod == 'LBFGS':
        opt = PLBFGS(problem)
        x, f, g,flag = opt.run()
    if optmethod == 'NLCG':
        opt = PNLCG(problem)
        x, f, g = opt.run()
    t2 = time.time()
    print('optimize time:',t2-t1)
    #If you need to output a visualization file, use to_vtk method
    #mesh.to_vtk(fname='opt_lshape.vtu') 
    node = mesh.entity('node')
    isFreeNode = problem.isFreeNode
    freenodetag = problem.freenodetag
    NI = np.sum(isFreeNode)
    xx = problem.xx.copy()
    xx[np.array(freenodetag.T.flat)] = x
    xnode = xx.reshape(3,NI).T
    node[isFreeNode] = xnode

    q = mesh.cell_quality()
    show_mesh_quality(q,ylim=4000,title='radius ratio energy optimization')
    return mesh

def test_intersectsphere(h=0.1, nodt=0, optmethod='LBFGS', P=0):
    mesh = MeshModel.intersect_spheres(h)

    q = mesh.cell_quality()
    NC = mesh.number_of_cells()
    ylim = int(0.1*NC)
    volume = mesh.entity_measure('cell')

    q = mesh.cell_quality()
    show_mesh_quality(q,ylim=9000,title='initial mesh quality') 
    opt = ODT(mesh)
    t1 = time.time() 
    if nodt != 0:
        print("ODT optimization in progress...")
    for i in range(nodt):
        q0 = opt.mesh.cell_quality()
        opt.odt_iterate(model=2)
        q1 = opt.mesh.cell_quality()
        if np.abs(np.max(q1)-np.max(q0))<1e-8:
            print("The number of ODT iterations is %d"%(i+1))
            break
    t2 = time.time()
    mesh = opt.mesh    
    if nodt !=0:
        print("The number of ODT iterations is %d"%(nodt))
        print("ODT iteration time:",t2-t1)
        q = mesh.cell_quality()
        show_mesh_quality(q,ylim=9000,title='odt optimization') 

    problem = TetMeshProblem(mesh)

    NDof = len(problem.x0)
    if P==1:
        problem.Preconditioner = problem.preconditioner()
    elif P==0:
        problem.Preconditioner = None
    problem.StepLength = 1.0
    problem.FunValDiff = 1e-4
    problem.Print = True

    t1 = time.time()
    if optmethod == 'LBFGS':
        opt = PLBFGS(problem)
        x, f, g,flag = opt.run()
    if optmethod == 'NLCG':
        opt = PNLCG(problem)
        x, f, g = opt.run()
    t2 = time.time()
    print('time:',t2-t1)
    #If you need to output a visualization file, use to_vtk method
    #mesh.to_vtk(fname='opt_tsp.vtu') 

    node = mesh.entity('node')
    isFreeNode = ~mesh.ds.boundary_node_flag()
    n = len(x)//3 
    node[isFreeNode,0] = x[:n]
    node[isFreeNode,1] = x[n:2*n]
    node[isFreeNode,2] = x[2*n:]

    q = mesh.cell_quality()
    show_mesh_quality(q,ylim=9000,title='radius ratio energy optimization')
    return mesh

def test_square_hole():
    mesh = MeshModel.square_hole(h=0.05)
    isBdNode = mesh.ds.boundary_node_flag()
    node = mesh.entity('node')
    np.random.seed(0)
    node[~isBdNode] += 0.01*np.random.rand(node[~isBdNode].shape[0],node[~isBdNode].shape[1])
    mesh.uniform_refine(2)
    opt = TriRadiusRatio(mesh)
    q = opt.get_quality()
    print("initial mesh quality: minq:",np.min(q),"meanq:",np.mean(q),"maxq",np.max(q))
    opt.iterate_solver(method='Bjacobi')

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    plt.show()

    q = opt.get_quality()
    print("Optimizing mesh quality: minq:",np.min(q),"meanq:",np.mean(q),"maxq",np.max(q))
    show_mesh_quality(q,ylim=14000)
    return mesh

def test_triangle_domain():
    mesh = MeshModel.triangle_domain()
    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    plt.show()

    opt = TriRadiusRatio(mesh)
    opt.iterate_solver(method='Bjacobi')
    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    plt.show()
    return mesh

def test_triangle_domain_odt():
    mesh = MeshModel.triangle_domain()
    
    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    plt.show()
    q = mesh.cell_quality()

    for i in range(100):
        mesh.odt_iterate()

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    plt.show()

    q = mesh.cell_quality()
    show_mesh_quality(q)
    return mesh 

def test_square_hole_odt():
    mesh = MeshModel.square_hole(h=0.05)
    node = mesh.entity('node')
    isBdNode = mesh.ds.boundary_node_flag()
    node = mesh.entity('node')
    np.random.seed(0)
    node[~isBdNode] += 0.01*np.random.rand(node[~isBdNode].shape[0],node[~isBdNode].shape[1])
    
    mesh.uniform_refine(2)
    q = mesh.cell_quality()
    show_mesh_quality(q,14000)
    for i in range(100):
        mesh.odt_iterate()
    q = mesh.cell_quality()
    show_mesh_quality(q,14000)
    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    plt.show()
    return mesh

def show_mesh_quality(q1,ylim=8000,title=None):
    fig,axes= plt.subplots()
    q1 = 1/q1
    minq1 = np.min(q1)
    maxq1 = np.max(q1)
    meanq1 = np.mean(q1)
    rmsq1 = np.sqrt(np.mean(q1**2))
    stdq1 = np.std(q1)
    NC = len(q1)
    SNC = np.sum((q1<0.3))
    hist, bins = np.histogram(q1, bins=50, range=(0, 1))
    center = (bins[:-1] + bins[1:]) / 2
    axes.bar(center, hist, align='center', width=0.02)
    axes.set_xlim(0, 1)
    axes.set_ylim(0,ylim)

    if title is not None:
        axes.set_title(title, fontsize=16, pad=20)

    #TODO: fix the textcoords warning
    axes.annotate('Min quality: {:.6}'.format(minq1), xy=(0, 0),
            xytext=(0.15, 0.85),
            textcoords="figure fraction",
            horizontalalignment='left', verticalalignment='top', fontsize=15)
    axes.annotate('Max quality: {:.6}'.format(maxq1), xy=(0, 0),
            xytext=(0.15, 0.8),
            textcoords="figure fraction",
            horizontalalignment='left', verticalalignment='top', fontsize=15)
    axes.annotate('Average quality: {:.6}'.format(meanq1), xy=(0, 0),
            xytext=(0.15, 0.75),
            textcoords="figure fraction",
            horizontalalignment='left', verticalalignment='top', fontsize=15)
    axes.annotate('RMS: {:.6}'.format(rmsq1), xy=(0, 0),
            xytext=(0.15, 0.7),
            textcoords="figure fraction",
            horizontalalignment='left', verticalalignment='top', fontsize=15)
    axes.annotate('STD: {:.6}'.format(stdq1), xy=(0, 0),
            xytext=(0.15, 0.65),
            textcoords="figure fraction",
            horizontalalignment='left', verticalalignment='top', fontsize=15)
    axes.annotate('radius radio less than 0.3:{:.0f}/{:.0f}'.format(SNC,NC), xy=(0, 0),
            xytext=(0.15, 0.6),
            textcoords="figure fraction",
            horizontalalignment='left', verticalalignment='top', fontsize=15)
    plt.tight_layout() 
    plt.show()
    return 0
# main function to run the tests
if args.exam == 'sp':
    mesh = test_unit_sphere(nodt=args.nodt, optmethod=args.optmethod, P=args.p)
if args.exam == 'ls':
    mesh = test_LShape(nodt=args.nodt, optmethod=args.optmethod, P=args.p)
if args.exam == 'tsp':
    mesh = test_intersectsphere(nodt=args.nodt, optmethod=args.optmethod, P=args.p)
if args.exam == 'tri':
    mesh = test_triangle_domain()
if args.exam == 'sq':
    mesh = test_square_hole()
if args.exam == 'triodt':
    mesh = test_triangle_domain_odt() 
if args.exam =='sqodt':
    mesh = test_square_hole_odt()

