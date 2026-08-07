package numerics.functionals

import numerics.Grid
import org.junit.jupiter.api.Tag
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Тесты параметра контрольной сетки в [errorEh]: обратная совместимость значения
 * по умолчанию и контроль равномерной нормы на измельчённой сетке.
 */
@Tag("fast")
class MetricsControlGridTest {

    /** Значение по умолчанию — ровно прежние 100n+1 точек: существующие вызовы не меняют результат. */
    @Test fun defaultRefinementReproducesPreviousBehaviour() {
        val g = Grid.uniform(8, 0.0, 1.0)
        val exact = { t: Double -> sin(6.0 * t) }
        val eval = { t: Double -> sin(6.0 * t) + 1e-3 * sin(40.0 * t) }
        assertEquals(DEFAULT_CONTROL_REFINEMENT, 100)
        assertEquals(errorEh(exact, eval, g), errorEh(exact, eval, g, refinement = 100), 0.0)
    }

    /** Измельчение контрольной сетки может только увеличить максимум по точкам, но не уменьшить. */
    @Test fun finerControlGridNeverDecreasesMaximum() {
        val g = Grid.uniform(16, 0.0, 1.0)
        val exact = { t: Double -> t * t }
        val eval = { t: Double -> t * t + 0.01 * sin(50.0 * t) }
        val coarse = errorEh(exact, eval, g)
        val fine = errorEh(exact, eval, g, refinement = 1000)
        assertTrue(fine >= coarse - 1e-15, "coarse=$coarse, fine=$fine")
    }

    /**
     * Контрольные данные Round 21: для гладкой разности переход со 100n+1 на 1000n+1
     * точку сдвигает величину не более чем на 4.071e-4 в относительной мере, то есть
     * сетка по умолчанию максимум ошибки НЕ занижает.
     */
    @Test fun defaultControlGridDoesNotUnderestimateSmoothError() {
        val g = Grid.uniform(32, 0.0, 1.0)
        val exact = { t: Double -> exp(-t) }
        val eval = { t: Double -> exp(-t) + 1e-4 * sin(9.0 * t) * (1.0 - t) }
        val coarse = errorEh(exact, eval, g)
        val fine = errorEh(exact, eval, g, refinement = 1000)
        val shift = abs(fine - coarse) / fine
        assertTrue(shift <= 4.071e-4, "относительный сдвиг $shift превысил эталонные 4.071e-4")
    }

    /**
     * Зачем параметр вообще нужен: на разности с узким пиком между узлами контрольной
     * сетки грубая сетка максимум ЗАНИЖАЕТ, и без измельчения это не обнаружить.
     */
    @Test fun coarseControlGridMissesNarrowSpike() {
        val g = Grid.uniform(4, 0.0, 1.0)
        val spikeCentre = 0.5 + 1.0 / (2.0 * 100 * 4) // ровно между точками сетки 100n+1
        val exact = { _: Double -> 0.0 }
        val eval = { t: Double -> exp(-4e7 * (t - spikeCentre) * (t - spikeCentre)) }
        val coarse = errorEh(exact, eval, g)
        val fine = errorEh(exact, eval, g, refinement = 100_000)
        assertTrue(coarse < 0.9, "грубая контрольная сетка обязана пропустить пик, получено $coarse")
        assertTrue(fine > 0.99, "измельчённая контрольная сетка обязана поймать пик, получено $fine")
    }

    /** refinement < 1 дал бы m = 0 и деление 0/0: это ошибка контракта, а не тихий NaN. */
    @Test fun refinementBelowOneIsRejected() {
        val g = Grid.uniform(4)
        assertFailsWith<IllegalArgumentException> { errorEh({ 0.0 }, { 0.0 }, g, refinement = 0) }
        assertFailsWith<IllegalArgumentException> { errorEh({ 0.0 }, { 0.0 }, g, refinement = -1) }
    }

    /** refinement = 1 законен: контрольная сетка совпадает с узлами равномерной основной. */
    @Test fun refinementOneEvaluatesAtGridNodes() {
        val g = Grid.uniform(4, 0.0, 1.0)
        val e = errorEh({ _ -> 0.0 }, { t -> t }, g, refinement = 1)
        assertEquals(1.0, e, 1e-12)
    }
}
