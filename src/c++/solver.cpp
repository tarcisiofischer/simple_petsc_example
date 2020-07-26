#include <solver.hpp>
#include <petsc.h>

#define PETSC_CHECK(X) \
    { auto _err = X; if (_err) throw std::runtime_error("PETSc error at line " + std::to_string(__LINE__)); }

PetscErrorCode residual_function(SNES snes, Vec x, Vec f, void *ctx)
{
    const PetscScalar *xx;
    PETSC_CHECK(VecGetArrayRead(x, &xx));
    
    PetscScalar *ff;
    PETSC_CHECK(VecGetArray(f, &ff));
    
    ff[0] = xx[0] * xx[0] + xx[1] * xx[1] - 20.;
    ff[1] = xx[0] - xx[1] + 2;
    
    PETSC_CHECK(VecRestoreArrayRead(x, &xx));
    PETSC_CHECK(VecRestoreArray(f, &ff));
    
    return 0;
}

PetscErrorCode jacobian_function(SNES snes, Vec x, Mat jac, Mat B, void *dummy)
{
    PetscInt idx[2] = {0, 1};

    const PetscScalar *xx;
    PETSC_CHECK(VecGetArrayRead(x, &xx));

    PetscScalar A[4] = {
        2. * xx[0], 2. * xx[1],
        1.0, -1.0
    };

    PETSC_CHECK(VecRestoreArrayRead(x,&xx));
    PETSC_CHECK(MatSetValues(B, 2, idx, 2, idx, A, INSERT_VALUES));
    
    PETSC_CHECK(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PETSC_CHECK(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
    return 0;
}

int monitor(SNES snes, int its, PetscReal norm, void *ctx)
{
    auto &steps_information = *reinterpret_cast<std::vector<std::pair<double, double>>*>(ctx);

    Vec x;
    SNESGetSolution(snes, &x);

    const PetscScalar *xx;
    VecGetArrayRead(x, &xx);

    steps_information.push_back({xx[0], xx[1]});

    VecRestoreArrayRead(x, &xx);
    return 0;
}

void init_petsc()
{
    PETSC_CHECK(PetscInitializeNoArguments());    
}

void finalize_petsc()
{
    PETSC_CHECK(PetscFinalize());
}

std::vector<std::pair<double, double>> run_solver(double x_guess, double y_guess)
{
    SNES snes;
    PETSC_CHECK(SNESCreate(PETSC_COMM_WORLD, &snes));
    PETSC_CHECK(SNESSetType(snes, SNESNEWTONLS));

    SNESLineSearch linesearch;
    PETSC_CHECK(SNESGetLineSearch(snes, &linesearch));
    PETSC_CHECK(SNESLineSearchSetType(linesearch, SNESLINESEARCHBASIC));

    Mat J;
    PETSC_CHECK(MatCreate(PETSC_COMM_WORLD, &J));
    PETSC_CHECK(MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 2, 2));
    PETSC_CHECK(MatSetFromOptions(J));
    PETSC_CHECK(MatSetUp(J));
    PETSC_CHECK(SNESSetJacobian(snes, J, J, jacobian_function, NULL));

    Vec r;
    PETSC_CHECK(VecCreate(PETSC_COMM_WORLD, &r));
    PETSC_CHECK(VecSetSizes(r, PETSC_DECIDE, 2));
    PETSC_CHECK(VecSetFromOptions(r));
    PETSC_CHECK(SNESSetFunction(snes, r, residual_function, NULL));

    auto steps_information = std::vector<std::pair<double, double>>();
    PETSC_CHECK(SNESMonitorSet(snes, monitor, &steps_information, nullptr));

    KSP ksp;
    PETSC_CHECK(SNESGetKSP(snes,&ksp));
    PC pc;
    PETSC_CHECK(KSPGetPC(ksp,&pc));
    PETSC_CHECK(KSPSetType(ksp, KSPPREONLY));
    PETSC_CHECK(PCSetType(pc, PCLU));
    PETSC_CHECK(SNESSetFromOptions(snes));

    Vec x;
    PETSC_CHECK(VecCreate(PETSC_COMM_WORLD, &x));
    PETSC_CHECK(VecSetSizes(x, PETSC_DECIDE, 2));
    PETSC_CHECK(VecSetFromOptions(x));

    {
        PetscScalar *xx;
        VecGetArray(x, &xx);
        xx[0] = x_guess;
        xx[1] = y_guess;
        VecRestoreArray(x, &xx);
    }
    PETSC_CHECK(SNESSolve(snes, NULL, x));    
    {
        const PetscScalar *xx;
        VecGetArrayRead(x, &xx);
        steps_information.push_back({xx[0], xx[1]});
        VecRestoreArrayRead(x, &xx);
    }

    PETSC_CHECK(VecDestroy(&x));
    PETSC_CHECK(VecDestroy(&r));
    PETSC_CHECK(MatDestroy(&J));
    PETSC_CHECK(SNESDestroy(&snes));

    return steps_information;
}
