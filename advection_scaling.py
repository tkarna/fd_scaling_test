import firedrake as fd
import math
import time as time_mod
import argparse
from pyop2.profiling import timed_stage
from firedrake.assemble import OneFormAssembler
from functools import partial
from firedrake.petsc import PETSc


class FastAssembler:
    def __init__(self, form, tensor, bcs=(), form_compiler_parameters=None, needs_zeroing=True):
        self.assembler = OneFormAssembler(form, tensor, needs_zeroing=needs_zeroing)
        self.assembler.assemble()  # warm cache
        self._needs_zeroing = needs_zeroing
        self._tensor = tensor
        self.parloops = []
        for p in self.assembler.parloops:
            f = FastParloop(p)
            self.parloops.append(f)

    def assemble(self):
        """Perform the assembly.

        :returns: The assembled object.
        """
        if self._needs_zeroing:
            self._tensor.dat.zero()

        for parloop in self.parloops:
            parloop()

        # for bc in self._bcs:
        #     if isinstance(bc, EquationBC):  # can this be lifted?
        #         bc = bc.extract_form("F")
        #     self._apply_bc(bc)

        return self._tensor


class FastParloop:
    def __init__(self, parloop):
        self.parloop = parloop

        assert not self.parloop._has_mats  # skip replace_lgmaps()
        assert len(self.parloop._reduction_idxs) == 0  # skip reduction
        assert len(self.parloop.reduced_globals) == 0  # skip global increments

        kernel = self.parloop.global_kernel
        kernel_cache_key = id(self.parloop.comm)
        c_func = kernel._func_cache[kernel_cache_key]

        start_core = self.parloop.iterset.core_part.offset
        end_core = start_core + self.parloop.iterset.core_part.size

        start_own = self.parloop.iterset.owned_part.offset
        end_own = start_own + self.parloop.iterset.owned_part.size

        self.c_func_core = partial(c_func.__call__, start_core, end_core, *self.parloop.arglist)
        self.c_func_owned = partial(c_func.__call__, start_own, end_own, *self.parloop.arglist)

    # @mpi.collective
    # @PETSc.Log.EventDecorator("ParLoopExecute")
    def __call__(self):
        """Execute the kernel over all members of the iteration space."""
        # self.parloop.zero_global_increments()
        # orig_lgmaps = self.parloop.replace_lgmaps()
        self.parloop.global_to_local_begin()
        # self.parloop._compute(self.parloop.iterset.core_part)
        self.c_func_core()
        self.parloop.global_to_local_end()
        # self.parloop._compute(self.parloop.iterset.owned_part)
        self.c_func_owned()
        # requests = self.parloop.reduction_begin()
        self.parloop.local_to_global_begin()
        self.parloop.update_arg_data_state()
        # self.parloop.restore_lgmaps(orig_lgmaps)
        # self.parloop.reduction_end(requests)
        # self.parloop.finalize_global_increments()
        self.parloop.local_to_global_end()


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
    u = fd.Function(W).interpolate(velocity)

    # cosine-bell--cone--slotted-cylinder initial coniditon
    bell_r0 = 0.15
    bell_x0 = 0.25
    bell_y0 = 0.5
    cone_r0 = 0.15
    cone_x0 = 0.5
    cone_y0 = 0.25
    cyl_r0 = 0.15
    cyl_x0 = 0.5
    cyl_y0 = 0.75
    slot_left = 0.475
    slot_right = 0.525
    slot_top = 0.85

    bell = 0.25*(1+fd.cos(math.pi*fd.min_value(fd.sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
    cone = 1.0 - fd.min_value(fd.sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cyl_r0, 1.0)
    slot_cyl = fd.conditional(fd.sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
                fd.conditional(fd.And(fd.And(x > slot_left, x < slot_right), y < slot_top),
                0.0, 1.0), 0.0)

    q = fd.Function(V, name='solution')
    q_analytical = 1.0 + bell + cone + slot_cyl
    q.project(q_analytical)
    q_init = fd.Function(V).assign(q)

    dtc = fd.Constant(dt)
    q_in = fd.Constant(1.0)

    dq_trial = fd.TrialFunction(V)
    phi = fd.TestFunction(V)
    a = phi*dq_trial*fd.dx

    n = fd.FacetNormal(mesh)
    un = 0.5*(fd.dot(u, n) + abs(fd.dot(u, n)))
    L1 = dtc*(q*fd.div(phi*u)*fd.dx
            - fd.conditional(fd.dot(u, n) < 0, phi*fd.dot(u, n)*q_in, 0.0)*fd.ds
            - fd.conditional(fd.dot(u, n) > 0, phi*fd.dot(u, n)*q, 0.0)*fd.ds
            - (phi('+') - phi('-'))*(un('+')*q('+') - un('-')*q('-'))*fd.dS)

    q1 = fd.Function(V)
    q2 = fd.Function(V)
    L2 = fd.replace(L1, {q: q1})
    L3 = fd.replace(L1, {q: q2})

    dq = fd.Function(V)

    params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
    mass_matrix = fd.assemble(a)
    lin_solver = fd.LinearSolver(mass_matrix, solver_parameters=params)

    L1_assembler = OneFormAssembler(L1, q1, needs_zeroing=True)
    L2_assembler = OneFormAssembler(L2, q2, needs_zeroing=True)
    L3_assembler = OneFormAssembler(L3, q1, needs_zeroing=True)

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
    lin_solver.solve(dq, q1)
    q1.dat.data[:] = q.dat.data_ro[:] + dq.dat.data_ro[:]
    L2_fassembler.assemble()
    lin_solver.solve(dq, q2)
    q2.dat.data[:] = 0.75*q.dat.data_ro[:] + \
        0.25*(q1.dat.data_ro[:] + dq.dat.data_ro[:])
    L3_fassembler.assemble()
    lin_solver.solve(dq, q1)
    q.dat.data[:] = (1.0/3.0)*q.dat.data_ro[:] + \
        (2.0/3.0)*(q2.dat.data_ro[:] + dq.dat.data_ro[:])
    step += 1
    t += dt
    tic = time_mod.perf_counter()
    with timed_stage('Time loop'):
        for i in range(nsteps - 1):
            L1_fassembler.assemble()
            lin_solver.solve(dq, q1)
            q1.dat.data[:] = q.dat.data_ro[:] + dq.dat.data_ro[:]
            L2_fassembler.assemble()
            lin_solver.solve(dq, q2)
            q2.dat.data[:] = 0.75*q.dat.data_ro[:] + \
                0.25*(q1.dat.data_ro[:] + dq.dat.data_ro[:])
            L3_fassembler.assemble()
            lin_solver.solve(dq, q1)
            q.dat.data[:] = (1.0/3.0)*q.dat.data_ro[:] + \
                (2.0/3.0)*(q2.dat.data_ro[:] + dq.dat.data_ro[:])
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

    run_problem(args.refine, no_exports=not args.store_output, nsteps=args.nsteps)


if __name__ == '__main__':
    process_args()