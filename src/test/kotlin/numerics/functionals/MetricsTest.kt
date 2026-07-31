package numerics.functionals

import numerics.Grid
import org.junit.jupiter.api.Tag
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Тесты метрик ошибки: errorEh (равномерная норма на мелкой сетке),
 * orders (порядок сходимости log2), constCh (константа E_h/h^p).
 */
@Tag("fast")
class MetricsTest {
    /** errorEh: для exact-eval с известной разностью даёт точный максимум модуля. */
    @Test fun errorEhCapturesMaxDifference() {
        val g = Grid.uniform(4, 0.0, 1.0)
        // eval = exact + bump; максимум |bump| = 0.5 в t=1 (линейный сдвиг 0.5*t)
        val e = errorEh({ t -> t }, { t -> t + 0.5 * t }, g)
        assertEquals(0.5, e, 1e-12)
    }

    /** errorEh = 0 при совпадающих функциях. */
    @Test fun errorEhZeroWhenEqual() {
        val g = Grid.uniform(4)
        assertEquals(0.0, errorEh({ t -> t * t }, { t -> t * t }, g), 1e-15)
    }

    /** orders: для геометрически убывающих ошибок (деление на 2) порядок = 1, последний = NaN. */
    @Test fun ordersGeometricHalving() {
        val errs = listOf(1.0, 0.5, 0.25)
        val p = orders(errs)
        assertEquals(3, p.size)
        assertEquals(1.0, p[0], 1e-12)
        assertEquals(1.0, p[1], 1e-12)
        assertTrue(p[2].isNaN()) // последний элемент без следующего -> NaN
    }

    /** orders: убывание в 4 раза даёт порядок 2. */
    @Test fun ordersQuarteringIsOrderTwo() {
        val p = orders(listOf(1.0, 0.25))
        assertEquals(2.0, p[0], 1e-12)
    }

    /**
     * orders: нулевая погрешность (совпадение с точным решением) — порядок НЕ ОПРЕДЕЛЁН,
     * а не «бесконечен»: раньше log2(E/0) давал +Inf, log2(0/E) — -Inf.
     */
    @Test fun ordersUndefinedOnZeroError() {
        assertTrue(orders(listOf(1.0, 0.0))[0].isNaN(), "E_{h/2}=0 -> порядок не определён")
        assertTrue(orders(listOf(0.0, 1.0))[0].isNaN(), "E_h=0 -> порядок не определён")
        assertTrue(orders(listOf(0.0, 0.0))[0].isNaN(), "обе нулевые -> порядок не определён")
        // отрицательная «погрешность» бессмысленна и тоже даёт NaN, а не log от отрицательного
        assertTrue(orders(listOf(-1.0, 1.0))[0].isNaN(), "отрицательная погрешность -> NaN")
        // положительные значения рядом с нулём считаются по-прежнему
        assertEquals(1.0, orders(listOf(1.0, 0.5, 0.0))[0], 1e-12)
        assertTrue(orders(listOf(1.0, 0.5, 0.0))[1].isNaN())
    }

    /**
     * errorEh явно требует n >= 1: при n = 0 делитель m = 100n обратился бы в ноль.
     * Конструировать Grid(0, ...) теперь нельзя, поэтому проверяем оба звена контракта:
     * сетка с n = 0 недостижима, а сама метрика считается на минимальной допустимой n = 1.
     */
    @Test fun errorEhRequiresPositiveN() {
        assertFailsWith<IllegalArgumentException> { Grid(0, doubleArrayOf(0.0)) }
        val e = errorEh({ t -> t }, { t -> t + 0.25 }, Grid.uniform(1, 0.0, 1.0))
        assertEquals(0.25, e, 1e-12)
    }

    /** constCh: C = E_h / h^p — обратный к определению. */
    @Test fun constChDefinition() {
        val eh = 0.08; val h = 0.25; val pp = 2.0
        val c = constCh(eh, h, pp)
        assertEquals(eh / (h * h), c, 1e-12)
        // обратная сверка: c*h^p = eh
        assertTrue(abs(c * h * h - eh) < 1e-12)
    }
}
