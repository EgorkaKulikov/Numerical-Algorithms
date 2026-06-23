package numerics.functionals

import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Тесты квазипроекционных семейств (mu, lambda) и общей инфраструктуры
 * FunctionalFamily/ValueFunctional/DerivFunctional. Главный инвариант
 * квазиинтерполянтов — точность на span{1,rho,sigma}: P_chi g = g.
 */
class FunctionalsExtraTest {
    private val grid = Grid.uniform(8)
    private val basis = MinimalSplineBasis(GeneratingSystem.B, grid)

    /** Воспроизведение g сплайном P_chi g = sum chi_j(g) omega_j на наборе точек. */
    private fun assertReproduces(fam: FunctionalFamily, g: (Double) -> Double, tol: Double) {
        val c = fam.projectorCoeffs(g)
        for (t in listOf(0.07, 0.23, 0.5, 0.71, 0.93)) {
            assertEquals(g(t), basis.evalSpline(c, t), tol, "${fam.name}: P g != g at t=$t")
        }
    }

    /** mu (усредняющие): точны на 1, rho=t, sigma=t^2 для phi^B (квазиинтерполянт span). */
    @Test fun averagingExactOnSpanB() {
        val mu = AveragingFunctionals(basis)
        assertTrue(!mu.isProjector && !mu.usesDerivative)
        assertReproduces(mu, { 1.0 }, 1e-9)
        assertReproduces(mu, { t -> t }, 1e-8)
        assertReproduces(mu, { t -> t * t }, 1e-8)
    }

    /** mu на гиперболической системе H: точность на span{1,sinh,cosh}. */
    @Test fun averagingExactOnSpanH() {
        val basisH = MinimalSplineBasis(GeneratingSystem.H, grid)
        val mu = AveragingFunctionals(basisH)
        val c = mu.projectorCoeffs({ t -> Math.sinh(t) })
        for (t in listOf(0.2, 0.6, 0.85)) {
            assertEquals(Math.sinh(t), basisH.evalSpline(c, t), 1e-7, "mu H sinh at $t")
        }
    }

    /** lambda (трёхточечные): точны на span{1,rho,sigma} для phi^B. */
    @Test fun threePointExactOnSpanB() {
        val lam = ThreePointFunctionals(basis)
        assertTrue(!lam.isProjector && !lam.usesDerivative)
        assertReproduces(lam, { 1.0 }, 1e-9)
        assertReproduces(lam, { t -> t }, 1e-8)
        assertReproduces(lam, { t -> t * t }, 1e-8)
    }

    /** Краевые mu_{-2}, mu_{n-1} — чистые значения u(x_0), u(x_n). */
    @Test fun averagingBoundaryAreEndpointValues() {
        val mu = AveragingFunctionals(basis)
        assertEquals(grid.x(0) * 3.0, mu.chi(-2).apply { t -> 3.0 * t }, 1e-12)
        assertEquals(grid.x(grid.n) * 3.0, mu.chi(grid.n - 1).apply { t -> 3.0 * t }, 1e-12)
        assertEquals(1.0, mu.chi(-2).absSum(), 1e-12)
    }

    /** Краевые lambda_{-2}, lambda_{n-1} — чистые значения на концах. */
    @Test fun threePointBoundaryAreEndpointValues() {
        val lam = ThreePointFunctionals(basis)
        assertEquals(5.0, lam.chi(-2).apply { _ -> 5.0 }, 1e-12)
        assertEquals(5.0, lam.chi(grid.n - 1).apply { _ -> 5.0 }, 1e-12)
    }

    /** cChi() = max_j sum|coeff| — положительная конечная константа устойчивости. */
    @Test fun cChiPositiveFinite() {
        for (fam in listOf(AveragingFunctionals(basis), ThreePointFunctionals(basis), ProjFunctionals(basis))) {
            val c = fam.cChi()
            assertTrue(c >= 1.0 && c.isFinite(), "${fam.name}: cChi=$c")
        }
    }

    /** projectorCoeffs возвращает вектор длины n+2. */
    @Test fun projectorCoeffsLength() {
        val mu = AveragingFunctionals(basis)
        assertEquals(grid.n + 2, mu.projectorCoeffs({ t -> t }).size)
    }

    /** DerivFunctional: apply = f(node)+cD*f'(node); absSum = 1+|cD|. */
    @Test fun derivFunctionalApplyAndAbsSum() {
        val df = DerivFunctional(0.5, -2.0)
        // f(t)=t^2 -> f(0.5)=0.25, f'(0.5)=1.0 -> 0.25 + (-2)*1 = -1.75
        assertEquals(-1.75, df.apply({ t -> t * t }, { t -> 2 * t }), 1e-12)
        assertEquals(3.0, df.absSum(), 1e-12) // 1+|-2|
    }

    /** ValueFunctional.absSum = сумма модулей коэффициентов. */
    @Test fun valueFunctionalAbsSum() {
        val vf = ValueFunctional(doubleArrayOf(0.0, 1.0), doubleArrayOf(-3.0, 4.0))
        assertEquals(7.0, vf.absSum(), 1e-12)
    }

    /** xi-семейство (де Бур–Фикс): apply без производной игнорирует производный член у краёв. */
    @Test fun deBoorFixUsesDerivativeFlag() {
        val xi = DeBoorFixFunctionals(basis)
        assertTrue(xi.isProjector && xi.usesDerivative)
        // краевой xi_{-2} = u(x_0): чистое значение, производная не нужна
        assertEquals(2.0 * grid.x(0), xi.chi(-2).apply { t -> 2.0 * t }, 1e-12)
    }

    /** closedFormInternal согласуется с биортогональной theta на omega_i (health-check). */
    @Test fun closedFormMatchesBuiltTheta() {
        val theta = ProjFunctionals(basis)
        val j = 2
        val closed = theta.closedFormInternal(j)
        // theta_j(omega_i)=delta_ij через closed-form тоже
        for (i in (j - 2)..(j + 2)) {
            val v = closed.apply { t -> basis.omega(i, t) }
            assertEquals(if (i == j) 1.0 else 0.0, v, 1e-6, "closedForm theta_$j(omega_$i)")
        }
    }
}
