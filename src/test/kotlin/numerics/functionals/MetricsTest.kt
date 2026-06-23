package numerics.functionals

import numerics.Grid
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Тесты метрик ошибки: errorEh (равномерная норма на мелкой сетке),
 * orders (порядок сходимости log2), constCh (константа E_h/h^p).
 */
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

    /** constCh: C = E_h / h^p — обратный к определению. */
    @Test fun constChDefinition() {
        val eh = 0.08; val h = 0.25; val pp = 2.0
        val c = constCh(eh, h, pp)
        assertEquals(eh / (h * h), c, 1e-12)
        // обратная сверка: c*h^p = eh
        assertTrue(abs(c * h * h - eh) < 1e-12)
    }
}
