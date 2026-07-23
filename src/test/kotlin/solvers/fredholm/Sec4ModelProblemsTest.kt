package solvers.fredholm

import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.DeBoorFixFunctionals
import numerics.functionals.errorEh
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §4 model problems (production): M2 (Fredholm Gaussian-blur, u*=cosh(1.5t)) and the
 * "constant beats order" collapse on the matched tuned generating system hyperbolic(1.5).
 * Also a span-reproduction health-check for a tuned system (xi reproduces span{1,rho,sigma}).
 */
class Sec4ModelProblemsTest {
    private fun solver(sys: GeneratingSystem, fam: Int, grid: Grid): SecondKindSolver {
        val basis = MinimalSplineBasis(sys, grid)
        val funcs = DeBoorFixFunctionals(basis, fam)
        val op = FredholmOperator(ModelProblem.M2.kernel, grid, numerics.GaussLegendre(8))
        return SecondKindSolver(basis, funcs, op, 1.0,
            { t -> ModelProblem.M2.rhsExact(t, op) },
            { t -> ModelProblem.M2.rhsExactDeriv(t, op) },
            { t -> ModelProblem.M2.rhsExactDeriv2(t, op) })
    }

    /**
     * M2 on the generic polynomial basis B (u* NOT exactly representable): error
     * genuinely decreases under refinement (Sloan scheme for a clean rate).
     */
    @Test fun m2_converges_on_generic_basis() {
        val B = GeneratingSystem.B
        val e8 = errorEh({ t -> ModelProblem.M2.exact(t) }, solver(B, 1, Grid.uniform(8)).sloan().eval, Grid.uniform(8))
        val e16 = errorEh({ t -> ModelProblem.M2.exact(t) }, solver(B, 1, Grid.uniform(16)).sloan().eval, Grid.uniform(16))
        assertTrue(e16 < e8, "M2 xi1(B) sloan should converge: e8=$e8 e16=$e16")
    }

    /**
     * Constant collapse: u*=cosh(1.5t) is EXACTLY in span{1,sinh(1.5t),cosh(1.5t)} of
     * hyperbolic(1.5), so even the low-order base scheme error hits the FP floor at n=8.
     * No hardcoded magic constant — just a loose machine-precision-ish bound.
     */
    @Test fun m2_constant_collapse_on_tuned_basis() {
        val H15 = GeneratingSystem.hyperbolic(1.5)
        val g = Grid.uniform(8)
        val e8 = errorEh({ t -> ModelProblem.M2.exact(t) }, solver(H15, 1, g).base().eval, g)
        assertTrue(e8 < 1e-9, "M2 xi1(H1.5) base error at n=8 not collapsed: e8=$e8")
    }

    /** Health-check: on tuned hyperbolic(1.5), P_xi reproduces span{1,rho,sigma} to ~0. */
    @Test fun xi_reproduces_span_on_tuned_basis() {
        val sys = GeneratingSystem.hyperbolic(1.5)
        val g = Grid.uniform(16)
        val basis = MinimalSplineBasis(sys, g)
        val funcs = DeBoorFixFunctionals(basis, 1)
        val members = listOf<Triple<(Double) -> Double, (Double) -> Double, (Double) -> Double>>(
            Triple({ _ -> 1.0 }, { _ -> 0.0 }, { _ -> 0.0 }),
            Triple({ t -> sys.rho(t) }, { t -> sys.rhoD(t) }, { t -> sys.rhoDD(t) }),
            Triple({ t -> sys.sigma(t) }, { t -> sys.sigmaD(t) }, { t -> sys.sigmaDD(t) }),
        )
        for ((f, fD, fDD) in members) {
            val coeffs = funcs.projectorCoeffs(f, fD, fDD)
            var maxErr = 0.0
            for (i in 0..200) {
                val t = g.a + (g.b - g.a) * i / 200.0
                maxErr = maxOf(maxErr, kotlin.math.abs(basis.evalSpline(coeffs, t) - f(t)))
            }
            assertTrue(maxErr < 1e-9, "xi(H1.5) span reproduction error=$maxErr")
        }
    }
}
