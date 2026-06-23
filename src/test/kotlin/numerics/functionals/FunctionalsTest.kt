package numerics.functionals

import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import kotlin.test.Test
import kotlin.test.assertEquals

class FunctionalsTest {
    private val grid = Grid.uniform(8)
    private val basis = MinimalSplineBasis(GeneratingSystem.B, grid)

    @Test fun thetaBiorthogonalToOmega() {
        val theta = ProjFunctionals(basis)
        val n = basis.n
        for (i in -2 until n) {
            for (j in -2 until n) {
                val v = theta.chi(i).apply({ t -> basis.omega(j, t) })
                val expected = if (i == j) 1.0 else 0.0
                assertEquals(expected, v, 1e-7, "theta_$i(omega_$j) wrong")
            }
        }
    }

    @Test fun xiBiorthogonalWithDerivative() {
        val xi = DeBoorFixFunctionals(basis)
        val n = basis.n
        for (i in -2 until n) {
            for (j in -2 until n) {
                val v = xi.chi(i).apply(
                    { t -> basis.omega(j, t) },
                    { t -> basis.omegaDeriv(j, t) },
                )
                val expected = if (i == j) 1.0 else 0.0
                assertEquals(expected, v, 1e-7, "xi_$i(omega_$j) wrong")
            }
        }
    }

    @Test fun projectorExactOnSpan() {
        // A projector reproduces functions in span{omega_j}. Use a known spline:
        // c_j chosen, build f = sum c_j omega_j, then theta_i(f) must equal c_i.
        val theta = ProjFunctionals(basis)
        val n = basis.n
        val c = DoubleArray(n + 2) { (it + 1).toDouble() }
        val f: (Double) -> Double = { t ->
            var s = 0.0
            for (j in -2 until n) s += c[j + 2] * basis.omega(j, t)
            s
        }
        for (i in -2 until n) {
            assertEquals(c[i + 2], theta.chi(i).apply(f), 1e-6, "projector not exact at i=$i")
        }
    }
}
