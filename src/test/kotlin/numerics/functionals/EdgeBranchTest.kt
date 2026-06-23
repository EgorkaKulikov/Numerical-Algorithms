package numerics.functionals

import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Точечные тесты на краевые ветви: контракт ValueFunctional, обе ветви maxOf в
 * errorEh (пик ошибки в середине), значения по умолчанию фабрик Grid.
 */
class EdgeBranchTest {
    /** ValueFunctional требует совпадения длин nodes и coeffs (ветвь require=false). */
    @Test fun valueFunctionalRejectsSizeMismatch() {
        assertFailsWith<IllegalArgumentException> {
            ValueFunctional(doubleArrayOf(0.0, 1.0), doubleArrayOf(1.0))
        }
    }

    /** errorEh с пиком разности в середине отрезка задействует обе ветви обновления максимума. */
    @Test fun errorEhPeakInMiddle() {
        val g = Grid.uniform(4, 0.0, 1.0)
        // |diff| = "шляпа": растёт до t=0.5, затем убывает -> maxOf обновляется и не обновляется
        val e = errorEh({ _ -> 0.0 }, { t -> 0.5 - Math.abs(t - 0.5) }, g)
        assertEquals(0.5, e, 1e-12) // максимум в t=0.5
    }

    /** Grid.uniform со значениями по умолчанию (a=0,b=1) — покрывает синтетический $default. */
    @Test fun gridUniformDefaults() {
        val g = Grid.uniform(4)
        assertEquals(0.0, g.a, 1e-15)
        assertEquals(1.0, g.b, 1e-15)
    }

    /** Grid.quasiUniform со значениями по умолчанию — покрывает синтетический $default. */
    @Test fun gridQuasiUniformDefaults() {
        val g = Grid.quasiUniform(6)
        assertEquals(0.0, g.a, 1e-15)
        assertEquals(1.0, g.b, 1e-15)
        for (i in 0 until g.n) assertTrue(g.x(i + 1) > g.x(i))
    }

    /** projectorCoeffs с явной производной gD задействует ветвь необязательного параметра. */
    @Test fun projectorCoeffsWithExplicitDerivative() {
        val basis = MinimalSplineBasis(GeneratingSystem.B, Grid.uniform(8))
        val xi = DeBoorFixFunctionals(basis)
        val c = xi.projectorCoeffs({ t -> t }, { _ -> 1.0 })
        // xi воспроизводит линейную f(t)=t точно
        for (t in listOf(0.2, 0.6)) assertEquals(t, basis.evalSpline(c, t), 1e-8)
    }
}
