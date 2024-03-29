import firedrake as fd
import math
import time as time_mod
import argparse
from pyop2.profiling import timed_stage
from firedrake.assemble import OneFormAssembler
from functools import partial
from pyop2 import datatypes as dtypes


class FastParloop:
    """
    Faster Parloop execution via a reference to the C function

    The function reference includes the arguments, assumed to be fixed.
    """
    def __init__(self, parloop, no_inc_zeroing=False):
        self.parloop = parloop
        self.no_inc_zeroing = no_inc_zeroing

        kernel = self.parloop.global_kernel
        kernel_cache_key = id(self.parloop.comm)
        c_func = kernel._func_cache[kernel_cache_key]

        start_core = self.parloop.iterset.core_part.offset
        end_core = start_core + self.parloop.iterset.core_part.size

        start_own = self.parloop.iterset.owned_part.offset
        end_own = start_own + self.parloop.iterset.owned_part.size

        start_full = start_core
        end_full = end_own

        self.c_func_core = partial(c_func.__call__, start_core, end_core,
                                   *self.parloop.arglist)
        self.c_func_owned = partial(c_func.__call__, start_own, end_own,
                                    *self.parloop.arglist)
        self.c_func_full = partial(c_func.__call__, start_full, end_full,
                                   *self.parloop.arglist)


class FastAssembler:
    """
    Speed up OneFormAssembler by precomputing as much as possible.
    """
    def __init__(self, form, tensor, bcs=(), form_compiler_parameters=None,
                 needs_zeroing=True):
        self.assembler = OneFormAssembler(form, tensor,
                                          needs_zeroing=needs_zeroing)
        self.assembler.assemble()  # warm cache
        self._needs_zeroing = needs_zeroing
        self._tensor = tensor
        self.parloops = []
        for p in self.assembler.parloops:
            # assert we can skip some operations in __call__()
            assert not p._has_mats  # skip replace_lgmaps()
            assert len(p._reduction_idxs) == 0  # skip reduction
            assert len(p.reduced_globals) == 0  # skip global increments
            f = FastParloop(p, no_inc_zeroing=needs_zeroing)
            self.parloops.append(f)

        # get global2local (dat, access_mode) pairs
        arg_pairs = []
        for p in self.parloops:
            for idx in p.parloop._g2l_idxs:
                dat = p.parloop.arguments[idx].data
                access_mode = p.parloop.accesses[idx]
                e = (dat, access_mode)
                if e not in arg_pairs:
                    arg_pairs.append(e)
        g2l_entries = arg_pairs
        self.g2l_entries = g2l_entries

        # get local2global (dat, access_mode) pairs
        arg_pairs = []
        for p in self.parloops:
            for idx in p.parloop._l2g_idxs:
                dat = p.parloop.arguments[idx].data
                access_mode = p.parloop.accesses[idx]
                e = (dat, access_mode)
                if e not in arg_pairs:
                    arg_pairs.append(e)
        l2g_entries = arg_pairs
        self.l2g_entries = l2g_entries

    def assemble(self):
        """Perform the assembly.

        :returns: The assembled object.
        """
        if self._needs_zeroing:
            self._tensor.dat.zero()
        for dat, access in self.g2l_entries:
            dat.global_to_local_begin(access)
        for p in self.parloops:
            p.c_func_core()
        for dat, access in self.g2l_entries:
            dat.global_to_local_end(access)
        for p in self.parloops:
            p.c_func_owned()
        for dat, access in self.l2g_entries:
            dat.local_to_global_begin(access)
        for dat, access in self.l2g_entries:
            dat.local_to_global_end(access)

        return self._tensor


def run_problem(refine, no_exports=True, nsteps=None):
    nx = 40*refine
    mesh = fd.UnitSquareMesh(nx, nx, quadrilateral=True)
    comm = mesh.comm

    compute_final_error = nsteps is None

    T = 2*math.pi
    revolutions = 1
    steps_per_revolution = 600*refine
    dt = T/steps_per_revolution
    if nsteps is None:
        nsteps = revolutions * steps_per_revolution

    V = fd.FunctionSpace(mesh, "DQ", 1)
    W = fd.VectorFunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)

    nprocs = comm.size
    nvertex = W.dim()
    nelem2d = int(V.dim()/V.ufl_cell().num_vertices())
    dofs = V.dim()
    dofs_per_core = int(dofs/nprocs)

    if comm.rank == 0:
        print('Running DG advection test')
        print(f'Refinement: {refine}')
        print(f'nsteps: {nsteps}')
        print(f'Number of vertices: {nvertex} vertices')
        print(f'Number of elements: {nelem2d} elements')
        print(f'Mesh cell type: {mesh.ufl_cell()}')
        print(f'Number of DOFs: {dofs}')
        print(f'Number of processes: {nprocs}')
        print(f'DOFs per core: {dofs_per_core}')
        print(f'Time step: {dt:.3e}')

    velocity = fd.as_vector((0.5 - y, x - 0.5))
    u = fd.Function(W, name='u').interpolate(velocity)

    # cosine-bell--cone--slotted-cylinder initial coniditon
    bell_r0 = 0.15
    bell_x0 = 0.25
    bell_y0 = 0.5
    cone_x0 = 0.5
    cone_y0 = 0.25
    cyl_r0 = 0.15
    cyl_x0 = 0.5
    cyl_y0 = 0.75
    slot_left = 0.475
    slot_right = 0.525
    slot_top = 0.85

    bell = 0.25*(1+fd.cos(math.pi*fd.min_value(fd.sqrt(pow(x-bell_x0, 2) +
                 pow(y-bell_y0, 2))/bell_r0, 1.0)))
    cone = 1.0 - fd.min_value(fd.sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2)) /
                              cyl_r0, 1.0)
    slot_cyl = fd.conditional(
        fd.sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
        fd.conditional(fd.And(fd.And(x > slot_left, x < slot_right),
                              y < slot_top),
                       0.0, 1.0),
        0.0)

    q = fd.Function(V, name='solution')
    q_analytical = 1.0 + bell + cone + slot_cyl
    q.project(q_analytical)
    q_init = fd.Function(V, name='q_init').assign(q)

    dtc = fd.Constant(dt)
    q_in = fd.Constant(1.0)

    dq_trial = fd.TrialFunction(V)
    phi = fd.TestFunction(V)
    a = phi*dq_trial*fd.dx

    n = fd.FacetNormal(mesh)
    un = 0.5*(fd.dot(u, n) + abs(fd.dot(u, n)))
    L1 = dtc*(
        q*fd.div(phi*u)*fd.dx
        - fd.conditional(fd.dot(u, n) < 0, phi*fd.dot(u, n)*q_in, 0.0)*fd.ds
        - fd.conditional(fd.dot(u, n) > 0, phi*fd.dot(u, n)*q, 0.0)*fd.ds
        - (phi('+') - phi('-'))*(un('+')*q('+') - un('-')*q('-'))*fd.dS
    )

    q1 = fd.Function(V, name='q1')
    q2 = fd.Function(V, name='q2')
    L2 = fd.replace(L1, {q: q1})
    L3 = fd.replace(L1, {q: q2})

    dq = fd.Function(V)

    params = {
        'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'
    }
    mass_matrix = fd.assemble(a)
    lin_solver = fd.LinearSolver(mass_matrix, solver_parameters=params)

    # inverse mass matrix
    inv_M = fd.assemble(fd.Tensor(fd.inner(dq_trial, phi)*fd.dx).inv)

    # These functions do not change in the time loop, skip halo exchange
    L1_fassembler = FastAssembler(L1, q1, needs_zeroing=True)
    L2_fassembler = FastAssembler(L2, q2, needs_zeroing=True)
    L3_fassembler = FastAssembler(L3, q1, needs_zeroing=True)

    if not no_exports:
        output_freq = 20 * refine
        outfile = fd.File('output.pvd')

    t = 0.0
    step = 0
    # take first step outside the timed loop
    L1_fassembler.assemble()
    with q1.dat.vec_ro as src, dq.dat.vec_wo as res:
        inv_M.petscmat.mult(src, res)
    q1.dat.data[:] = q.dat.data_ro[:] + dq.dat.data_ro[:]
    L2_fassembler.assemble()
    with q2.dat.vec_ro as src, dq.dat.vec_wo as res:
        inv_M.petscmat.mult(src, res)
    q2.dat.data[:] = 0.75*q.dat.data_ro[:] + \
        0.25*(q1.dat.data_ro[:] + dq.dat.data_ro[:])
    L3_fassembler.assemble()
    with q1.dat.vec_ro as src, dq.dat.vec_wo as res:
        inv_M.petscmat.mult(src, res)
    q.dat.data[:] = (1.0/3.0)*q.dat.data_ro[:] + \
        (2.0/3.0)*(q2.dat.data_ro[:] + dq.dat.data_ro[:])
    step += 1
    t += dt
    tic = time_mod.perf_counter()
    one_third = 1.0/3.0
    two_third = 2.0/3.0
    with timed_stage('Time loop'):
        for i in range(nsteps - 1):
            L1_fassembler.assemble()
            with q1.dat.vec_ro as src, dq.dat.vec_wo as res:
                inv_M.petscmat.mult(src, res)

            q1.dat.zero()
            with q1.dat.vec_wo as q1_w, q.dat.vec_ro as q_r, dq.dat.vec_ro as dq_r:
                q1_w.maxpy([1, 1], [q_r, dq_r])
            L2_fassembler.assemble()
            with q2.dat.vec_ro as src, dq.dat.vec_wo as res:
                inv_M.petscmat.mult(src, res)
            q2.dat.zero()
            with q2.dat.vec_wo as q2_w, q.dat.vec_ro as q_r, q1.dat.vec_ro as q1_r, dq.dat.vec_ro as dq_r:
                q2_w.maxpy([0.75, 0.25, 0.25], [q_r, q1_r, dq_r])
            L3_fassembler.assemble()
            with q1.dat.vec_ro as src, dq.dat.vec_wo as res:
                inv_M.petscmat.mult(src, res)
            with q.dat.vec_wo as q_w, q2.dat.vec_ro as q2_r, dq.dat.vec_ro as dq_r:
                q_w.scale(one_third)
                q_w.maxpy([two_third, two_third], [q2_r, dq_r])
            step += 1
            t += dt
            if not no_exports and step % output_freq == 0:
                outfile.write(q)
                print("t=", t)
    toc = time_mod.perf_counter()
    if comm.rank == 0:
        print(f'CPU time in time loop: {toc-tic} s')

    L2_init = fd.errornorm(q_analytical, q_init)
    if comm.rank == 0:
        print(f'Initial L2 error: {L2_init}')
    if compute_final_error:
        L2_final = fd.errornorm(q_analytical, q)
        if comm.rank == 0:
            print(f'Final L2 error: {L2_final}')


def process_args():
    parser = argparse.ArgumentParser(
        description='Run 2D advection test',
        # includes default values in help entries
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-r', '--refine', type=int, default=1,
                        help='mesh refinement factor')
    parser.add_argument('-o', '--store-output', action='store_true',
                        help='store VTK output to disk')
    parser.add_argument('--nsteps', type=int,
                        help='Run only given number of steps')
    args, unknown_args = parser.parse_known_args()

    run_problem(args.refine, no_exports=not args.store_output,
                nsteps=args.nsteps)


if __name__ == '__main__':
    process_args()
