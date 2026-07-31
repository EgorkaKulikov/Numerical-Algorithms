package numerics.functionals

import kotlin.math.abs

import numerics.Grid

// ============================================================================
// 11. МЕТРИКИ
// ============================================================================

/**
 * E_h = max|u*(t) - u_h(t)| на 100n+1 точках равномерной подсетки [a, b].
 *
 * @throws IllegalArgumentException если `grid.n < 1`. Проверка дублирует контракт [Grid]
 *   (сетку с n = 0 построить нельзя), но контракт ИМЕННО этого метода — деление на
 *   m = 100n — фиксируем явно: при m = 0 шаг i/m дал бы 0/0 = NaN и метрика тихо вернула
 *   бы NaN вместо ошибки.
 */
fun errorEh(exact: (Double) -> Double, eval: (Double) -> Double, grid: Grid): Double {
    require(grid.n >= 1) { "errorEh: требуется grid.n >= 1 (иначе m = 100n = 0), получено n=${grid.n}" }
    val m = 100 * grid.n
    var e = 0.0
    for (i in 0..m) {
        val t = grid.a + (grid.b - grid.a) * i / m
        e = maxOf(e, abs(exact(t) - eval(t)))
    }
    return e
}

/**
 * p_h = log2(E_h / E_{h/2}) по соседним строкам таблицы погрешностей.
 *
 * Возвращает [Double.NaN] в двух случаях: (1) для последней строки — следующей просто нет;
 * (2) если любая из двух соседних погрешностей неположительна — порядок НЕ ОПРЕДЕЛЁН на
 * машинной точности. Второй случай важен практически: при E = 0 (совпадение с точным
 * решением до последнего бита) отношение даёт 0 или бесконечность, и log2 вернул бы ±Inf,
 * что в таблице выглядело бы как «бесконечный порядок» — численно бессмысленное
 * утверждение о сходимости, тогда как NaN честно печатается как «---».
 */
fun orders(errs: List<Double>): List<Double> =
    errs.indices.map { i ->
        when {
            i + 1 >= errs.size -> Double.NaN
            errs[i] <= 0.0 || errs[i + 1] <= 0.0 -> Double.NaN
            else -> Math.log(errs[i] / errs[i + 1]) / Math.log(2.0)
        }
    }

/** C_h = E_h / h^p. */
fun constCh(eh: Double, h: Double, p: Double): Double = eh / Math.pow(h, p)
