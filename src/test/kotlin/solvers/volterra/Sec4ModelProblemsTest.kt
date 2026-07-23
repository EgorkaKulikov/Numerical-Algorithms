package solvers.volterra

import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.DeBoorFixFunctionals
import numerics.functionals.errorEh
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * §4 headline model problem (production): M3 (Volterra exp-memory kernel, u*=cos(3t)).
 * Convergence on the matched tuned basis trig(3.0) and the "constant beats order"
 * collapse (u* exactly in span{1,sin(3t),cos(3t)}). Plus a span-reproduction health-check.
 */
class Sec4ModelProblemsTest {
    private fun solver(sys: GeneratingSystem, fam: Int, grid: Grid): SecondKindSolver {
        val basis = MinimalSplineBasis(sys, grid)
        val funcs = DeBoorFixFunctionals(basis, fam)
        val op = VolterraOperator(ModelProblem.M3.kernel, grid, numerics.GaussLegendre(8))
        return SecondKindSolver(basis, funcs, op, 1.0,
            { t -> ModelProblem.M3.rhsExact(t, op) },
            { t -> ModelProblem.M3.rhsExactDeriv(t, op) },
            { t -> ModelProblem.M3.rhsExactDeriv2(t, op) })
    }

    /**
     * M3 on the generic polynomial basis B (u* NOT exactly representable): error
     * genuinely decreases under refinement (Sloan scheme for a clean rate).
     */
    @Test fun m3_converges_on_generic_basis() {
        val B = GeneratingSystem.B
        val e8 = errorEh({ t -> ModelProblem.M3.exact(t) }, solver(B, 1, Grid.uniform(8)).sloan().eval, Grid.uniform(8))
        val e16 = errorEh({ t -> ModelProblem.M3.exact(t) }, solver(B, 1, Grid.uniform(16)).sloan().eval, Grid.uniform(16))
        assertTrue(e16 < e8, "M3 xi1(B) sloan should converge: e8=$e8 e16=$e16")
    }

    /**
     * Constant collapse: u*=cos(3t) is EXACTLY in span{1,sin(3t),cos(3t)} of trig(3.0),
     * so the base-scheme error hits the FP floor at n=8. Loose machine-precision-ish bound,
     * no hardcoded magic constant.
     */
    @Test fun m3_constant_collapse_on_tuned_basis() {
        val T3 = GeneratingSystem.trig(3.0)
        val g = Grid.uniform(8)
        val e8 = errorEh({ t -> ModelProblem.M3.exact(t) }, solver(T3, 1, g).base().eval, g)
        assertTrue(e8 < 1e-9, "M3 xi1(T3) base error at n=8 not collapsed: e8=$e8")
    }

    /** Health-check: on tuned trig(3.0), P_xi reproduces span{1,rho,sigma} to ~0. */
    @Test fun xi_reproduces_span_on_tuned_basis() {
        val sys = GeneratingSystem.trig(3.0)
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
            assertTrue(maxErr < 1e-9, "xi(T3) span reproduction error=$maxErr")
        }
    }
}
