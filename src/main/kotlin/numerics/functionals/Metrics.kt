package numerics.functionals

import kotlin.math.abs

import numerics.Grid

// ============================================================================
// 11. МЕТРИКИ
// ============================================================================

/** E_h = max|u*(t) - u_h(t)| на 100n+1 точках. */
fun errorEh(exact: (Double) -> Double, eval: (Double) -> Double, grid: Grid): Double {
    val m = 100 * grid.n
    var e = 0.0
    for (i in 0..m) {
        val t = grid.a + (grid.b - grid.a) * i / m
        e = maxOf(e, abs(exact(t) - eval(t)))
    }
    return e
}

/** p_h = log2(E_h / E_{h/2}) по соседним строкам. */
fun orders(errs: List<Double>): List<Double> =
    errs.indices.map { i -> if (i + 1 < errs.size) Math.log(errs[i] / errs[i + 1]) / Math.log(2.0) else Double.NaN }

/** C_h = E_h / h^p. */
fun constCh(eh: Double, h: Double, p: Double): Double = eh / Math.pow(h, p)
