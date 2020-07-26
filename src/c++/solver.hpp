#ifndef _SOLVER_HPP
#define _SOLVER_HPP

#include <vector>

void init_petsc();
void finalize_petsc();
std::vector<std::pair<double, double>> run_solver(double x_guess, double y_guess);

#endif
