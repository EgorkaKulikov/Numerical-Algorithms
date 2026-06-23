package numerics

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Дополнительные тесты сетки и эталонных формул: quasiUniform, проверка require,
 * вырожденность узлов и все ветви ReferenceSplines (B, B', H) внутри и вне носителя.
 */
class GridExtraTest {
    private val tol = 1e-12

    /** quasiUniform: монотонная сетка с концами a,b и фиксированной амплитудой. */
    @Test fun quasiUniformMonotoneEnds() {
        val g = Grid.quasiUniform(10, 0.0, 1.0, amp = 0.04)
        assertEquals(0.0, g.a, tol)
        assertEquals(1.0, g.b, tol)
        // строгая монотонность внутренних узлов
        for (i in 0 until g.n) assertTrue(g.x(i + 1) > g.x(i), "node $i not increasing")
        // граница тройная
        assertEquals(g.a, g.x(0), tol); assertEquals(g.b, g.x(g.n), tol)
    }

    /** Конструктор Grid требует interior размера n+1. */
    @Test fun gridRequiresCorrectSize() {
        assertFailsWith<IllegalArgumentException> { Grid(4, doubleArrayOf(0.0, 1.0)) }
    }

    /** nonDegenerate=false при слиянии узлов на тройном крае (j=-2). */
    @Test fun nonDegenerateFalseAtTripleKnot() {
        val g = Grid.uniform(8)
        // x(-2)=x(-1)=x(0)=a -> вырождено
        assertTrue(!nonDegenerate(g, -2))
    }

    /** omegaB равен нулю вне носителя [x_j, x_{j+3}] (обе ветви t<xj и t>xj3). */
    @Test fun omegaBZeroOutsideSupport() {
        val g = Grid.uniform(8)
        val j = 2
        assertEquals(0.0, ReferenceSplines.omegaB(g, j, g.x(j) - 0.01), tol)
        assertEquals(0.0, ReferenceSplines.omegaB(g, j, g.x(j + 3) + 0.01), tol)
        assertEquals(0.0, ReferenceSplines.omegaBDeriv(g, j, g.x(j) - 0.01), tol)
        assertEquals(0.0, ReferenceSplines.omegaBDeriv(g, j, g.x(j + 3) + 0.01), tol)
    }

    /** omegaB неотрицателен и попадает во все три куска (left, middle, right). */
    @Test fun omegaBThreePieces() {
        val g = Grid.uniform(8)
        val j = 2
        val tLeft = 0.5 * (g.x(j) + g.x(j + 1))
        val tMid = 0.5 * (g.x(j + 1) + g.x(j + 2))
        val tRight = 0.5 * (g.x(j + 2) + g.x(j + 3))
        for (t in listOf(tLeft, tMid, tRight)) {
            assertTrue(ReferenceSplines.omegaB(g, j, t) > 0.0, "omegaB<=0 at $t")
        }
    }

    /** omegaBDeriv — численная производная согласуется с аналитической в каждом куске. */
    @Test fun omegaBDerivMatchesNumeric() {
        val g = Grid.uniform(8)
        val j = 2
        val eps = 1e-6
        for (t in listOf(g.x(j) + 0.02, 0.5 * (g.x(j + 1) + g.x(j + 2)), g.x(j + 3) - 0.02)) {
            val num = (ReferenceSplines.omegaB(g, j, t + eps) - ReferenceSplines.omegaB(g, j, t - eps)) / (2 * eps)
            assertEquals(num, ReferenceSplines.omegaBDeriv(g, j, t), 1e-4, "deriv mismatch at $t")
        }
    }

    /** omegaH: ноль вне носителя и положителен во всех трёх кусках. */
    @Test fun omegaHPiecesAndSupport() {
        val g = Grid.uniform(8)
        val j = 2
        assertEquals(0.0, ReferenceSplines.omegaH(g, j, g.x(j) - 0.01), tol)
        assertEquals(0.0, ReferenceSplines.omegaH(g, j, g.x(j + 3) + 0.01), tol)
        val tLeft = 0.5 * (g.x(j) + g.x(j + 1))
        val tMid = 0.5 * (g.x(j + 1) + g.x(j + 2))
        val tRight = 0.5 * (g.x(j + 2) + g.x(j + 3))
        for (t in listOf(tLeft, tMid, tRight)) {
            assertTrue(ReferenceSplines.omegaH(g, j, t) > 0.0, "omegaH<=0 at $t")
        }
    }
}
