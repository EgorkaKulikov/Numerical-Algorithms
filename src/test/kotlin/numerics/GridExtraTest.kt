package numerics

import org.junit.jupiter.api.Tag
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Дополнительные тесты сетки и эталонных формул: quasiUniform, проверка require,
 * вырожденность узлов и все ветви ReferenceSplines (B, B', H) внутри и вне носителя.
 */
@Tag("fast")
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

    /**
     * Немонотонные узлы отбраковываются: без этого h = max_j(x_{j+1}-x_j) и бинарный
     * поиск интервала тихо давали бы мусор вместо ошибки.
     */
    @Test fun gridRejectsNonMonotoneNodes() {
        val e = assertFailsWith<IllegalArgumentException> {
            Grid(3, doubleArrayOf(0.0, 0.7, 0.3, 1.0)) // узлы 1 и 2 переставлены
        }
        assertTrue(e.message!!.contains("i=1"), "сообщение должно указывать индекс: ${e.message}")
    }

    /** Дубликат внутреннего узла (нулевой шаг) — тоже нарушение СТРОГОГО возрастания. */
    @Test fun gridRejectsDuplicatedNode() {
        val e = assertFailsWith<IllegalArgumentException> {
            Grid(3, doubleArrayOf(0.0, 0.5, 0.5, 1.0))
        }
        assertTrue(e.message!!.contains("i=1"), "сообщение должно указывать индекс: ${e.message}")
    }

    /** n = 0: интервалов нет, раньше падало NoSuchElementException на max_j по пустому диапазону. */
    @Test fun gridRejectsZeroIntervals() {
        val e = assertFailsWith<IllegalArgumentException> { Grid(0, doubleArrayOf(0.0)) }
        assertTrue(e.message!!.contains("n=0"), "сообщение должно содержать n: ${e.message}")
    }

    /**
     * Все четыре фабрики строят СТРОГО возрастающие узлы на всех используемых n и
     * параметрах — т.е. новая проверка монотонности не срабатывает на штатных путях.
     */
    @Test fun allFactoriesProduceStrictlyIncreasingNodes() {
        val ns = listOf(2, 3, 4, 5, 8, 16, 32, 64, 128)
        for (n in ns) {
            val grids = buildList {
                add("uniform(n=$n)" to Grid.uniform(n, 0.0, 1.0))
                add("uniform(n=$n,[-3,7])" to Grid.uniform(n, -3.0, 7.0))
                for (amp in listOf(-0.04, 0.0, 0.04, 0.15)) {
                    add("quasiUniform(n=$n,amp=$amp)" to Grid.quasiUniform(n, 0.0, 1.0, amp))
                }
                for (r in listOf(0.1, 0.5, 1.0, 1.5, 2.0, 10.0, 1e6)) {
                    add("geometric(n=$n,R=$r)" to Grid.geometric(n, 0.0, 1.0, r))
                    add("graded(n=$n,ratio=$r)" to Grid.graded(n, 0.0, 1.0, r))
                }
            }
            for ((name, g) in grids) {
                for (i in 0 until g.n) {
                    assertTrue(g.x(i + 1) > g.x(i), "$name: узлы не возрастают на i=$i")
                }
                assertTrue(g.h > 0.0, "$name: h <= 0")
            }
        }
    }

    /**
     * quasiUniform — единственная фабрика, способная породить немонотонные узлы:
     * Psi'(u) = 1 + 2*pi*amp*cos(2*pi*u) меняет знак при |amp| > 1/(2*pi) ≈ 0.15915.
     *
     * Отбраковка идёт по ФАКТИЧЕСКИМ узлам (инвариант [Grid]), а НЕ по самому amp:
     * немонотонность Psi как функции не равносильна немонотонности конечного набора
     * узлов. Закрепляем обе стороны: реально немонотонные входы падают, а легитимные
     * (amp = 1/(2*pi) при любом n; amp = 0.16 при n = 8) — строятся.
     */
    @Test fun quasiUniformRejectsOnlyActuallyNonMonotoneNodes() {
        val e = assertFailsWith<IllegalArgumentException> { Grid.quasiUniform(19, 0.0, 1.0, amp = 0.16) }
        assertTrue(
            e.message!!.contains("возрастать"),
            "сообщение должно указывать на немонотонные узлы: ${e.message}",
        )
        assertFailsWith<IllegalArgumentException> { Grid.quasiUniform(8, 0.0, 1.0, amp = -0.5) }
        // ТОТ ЖЕ amp = 0.16 при n = 8 даёт строго возрастающие узлы и законен.
        val coarse = Grid.quasiUniform(8, 0.0, 1.0, amp = 0.16)
        for (i in 0 until coarse.n) assertTrue(coarse.x(i + 1) > coarse.x(i), "amp=0.16,n=8: узел $i")
        // Граница ровно 1/(2*pi): Psi' обращается в ноль в ОДНОЙ точке, но узлы
        // строго возрастают при любом конечном n — отбраковывать его нечего.
        for (n in intArrayOf(8, 64, 1024)) {
            val g = Grid.quasiUniform(n, 0.0, 1.0, amp = 1.0 / (2.0 * Math.PI))
            for (i in 0 until g.n) assertTrue(g.x(i + 1) > g.x(i), "amp=1/(2pi),n=$n: узел $i")
        }
        // чуть ниже границы — всё ещё строится
        val g = Grid.quasiUniform(64, 0.0, 1.0, amp = 0.158)
        for (i in 0 until g.n) assertTrue(g.x(i + 1) > g.x(i), "amp=0.158: узел $i")
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
