package numerics

// ============================================================================
// 5. БАЗИС МИНИМАЛЬНЫХ СПЛАЙНОВ (устойчивое построение M_k^{-1} phi(t))
// ============================================================================

/**
 * Базис квадратичных минимальных B_phi-сплайнов {omega_j}_{j=-2}^{n-1} на сетке
 * (net) для порождающей phi. На интервале (x_k,x_{k+1}) три активных сплайна
 * omega_{k-2},omega_{k-1},omega_k удовлетворяют соотношениям воспроизведения
 *   a_{k-2} omega_{k-2}(t) + a_{k-1} omega_{k-1}(t) + a_k omega_k(t) = phi(t),
 * откуда (omega) = M_k^{-1} phi(t), M_k = (a_{k-2}|a_{k-1}|a_k). Устойчиво при
 * тройных краевых узлах (где явная формула через d_j = phi_j x phi'_j вырождается).
 */
class MinimalSplineBasis(val sys: GeneratingSystem, val grid: Grid) {
    val n = grid.n

    // a_j: на (x_{j+1},x_{j+2}) предел при тройном узле a_j = phi(x_{j+1}).
    private val aMin = -2
    private val aMax = n - 1
    private val aVec: Array<DoubleArray> = Array(aMax - aMin + 1) { k -> computeA(k + aMin) }

    private val invM: Array<Array<DoubleArray>> = Array(n) { k -> invert3(a(k - 2), a(k - 1), a(k)) }

    private fun a(j: Int): DoubleArray = aVec[j - aMin]

    private fun computeA(j: Int): DoubleArray {
        val xj1 = grid.x(j + 1)
        val xj2 = grid.x(j + 2)
        val phiJ1 = sys.phi(xj1)
        if (xj1 == xj2) return phiJ1 // тройной узел на краю
        val phiDJ1 = sys.phiD(xj1)
        val dJ2 = cross3(sys.phi(xj2), sys.phiD(xj2))
        val denom = dot3(dJ2, phiDJ1)
        require(kotlin.math.abs(denom) >= 1e-14) {
            "computeA(j=$j): degenerate approximation relation, dot3(dJ2, phiDJ1)=$denom (near-zero denominator)"
        }
        val coef = dot3(dJ2, phiJ1) / denom
        return doubleArrayOf(
            phiJ1[0] - coef * phiDJ1[0],
            phiJ1[1] - coef * phiDJ1[1],
            phiJ1[2] - coef * phiDJ1[2],
        )
    }

    /** Индекс сеточного интервала k с x_k <= t < x_{k+1} (для t=b возвращает n-1). */
    private fun intervalOf(t: Double): Int {
        // Бинарный поиск (breakpoints x_0..x_n возрастают): наибольший k in [0,n-1]
        // с x_k <= t, с клампами на концах. Семантика идентична линейному поиску:
        // t < x_1 -> 0; t >= x_{n-1} -> n-1; иначе x_lo <= t < x_{lo+1}.
        if (t < grid.x(1)) return 0
        if (t >= grid.x(n - 1)) return n - 1
        var lo = 1
        var hi = n - 1
        while (hi - lo > 1) {
            val mid = (lo + hi) / 2
            if (grid.x(mid) <= t) lo = mid else hi = mid
        }
        return lo
    }

    /** Индекс сеточного интервала, содержащего t (публичный доступ). */
    fun interval(t: Double): Int = intervalOf(t)

    /** Три активных значения omega_{k-2},omega_{k-1},omega_k в точке t (одно M_k^{-1} phi(t)). */
    fun activeOmega(k: Int, t: Double): DoubleArray {
        val inv = invM[k]
        val p = sys.phi(t)
        return doubleArrayOf(
            inv[0][0] * p[0] + inv[0][1] * p[1] + inv[0][2] * p[2],
            inv[1][0] * p[0] + inv[1][1] * p[1] + inv[1][2] * p[2],
            inv[2][0] * p[0] + inv[2][1] * p[1] + inv[2][2] * p[2],
        )
    }

    /** Значение omega_j(t), j из [-2, n-1]. Носитель [x_j, x_{j+3}]; вне него 0. */
    fun omega(j: Int, t: Double): Double {
        if (t < grid.x(j) || t > grid.x(j + 3)) return 0.0
        val k = intervalOf(t)
        val slot = j - (k - 2)
        if (slot < 0 || slot > 2) return 0.0
        val inv = invM[k]
        val phiT = sys.phi(t)
        return inv[slot][0] * phiT[0] + inv[slot][1] * phiT[1] + inv[slot][2] * phiT[2]
    }

    /** Производная omega_j'(t) (phi заменяется на phi'). Нужна для xi-функционалов. */
    fun omegaDeriv(j: Int, t: Double): Double {
        if (t < grid.x(j) || t > grid.x(j + 3)) return 0.0
        val k = intervalOf(t)
        val slot = j - (k - 2)
        if (slot < 0 || slot > 2) return 0.0
        val inv = invM[k]
        val phiDT = sys.phiD(t)
        return inv[slot][0] * phiDT[0] + inv[slot][1] * phiDT[1] + inv[slot][2] * phiDT[2]
    }

    /** Значение сплайна u_h(t) = sum_j c_j omega_j(t), c размера n+2. */
    fun evalSpline(c: DoubleArray, t: Double): Double {
        val k = intervalOf(t)
        val w = activeOmega(k, t)
        return c[k] * w[0] + c[k + 1] * w[1] + c[k + 2] * w[2] // индексы k-2,k-1,k -> +2
    }

    /** Значение производной сплайна u_h'(t). */
    fun evalSplineDeriv(c: DoubleArray, t: Double): Double {
        val k = intervalOf(t)
        val inv = invM[k]
        val p = sys.phiD(t)
        val w0 = inv[0][0] * p[0] + inv[0][1] * p[1] + inv[0][2] * p[2]
        val w1 = inv[1][0] * p[0] + inv[1][1] * p[1] + inv[1][2] * p[2]
        val w2 = inv[2][0] * p[0] + inv[2][1] * p[1] + inv[2][2] * p[2]
        return c[k] * w0 + c[k + 1] * w1 + c[k + 2] * w2
    }
}

// ----------------------------------------------------------------------------
// 5.1 ЭТАЛОННЫЕ ФОРМУЛЫ базисов (для health-checks)
// ----------------------------------------------------------------------------

/** Эталонные явные формулы для сверки с общим базисом (health-check). */
object ReferenceSplines {
    /** Классический квадратичный B-сплайн omega^B_j(t). */
    fun omegaB(grid: Grid, j: Int, t: Double): Double {
        val xj = grid.x(j); val xj1 = grid.x(j + 1); val xj2 = grid.x(j + 2); val xj3 = grid.x(j + 3)
        if (t < xj || t > xj3) return 0.0
        return when {
            t < xj1 -> (t - xj) * (t - xj) / ((xj1 - xj) * (xj2 - xj))
            t < xj2 -> (1.0 / (xj1 - xj)) * (
                (t - xj) * (t - xj) / (xj2 - xj)
                    - (t - xj1) * (t - xj1) * (xj3 - xj) / ((xj2 - xj1) * (xj3 - xj1))
                )
            else -> (t - xj3) * (t - xj3) / ((xj3 - xj1) * (xj3 - xj2))
        }
    }

    /** Производная эталонного B-сплайна. */
    fun omegaBDeriv(grid: Grid, j: Int, t: Double): Double {
        val xj = grid.x(j); val xj1 = grid.x(j + 1); val xj2 = grid.x(j + 2); val xj3 = grid.x(j + 3)
        if (t < xj || t > xj3) return 0.0
        return when {
            t < xj1 -> 2.0 * (t - xj) / ((xj1 - xj) * (xj2 - xj))
            t < xj2 -> (1.0 / (xj1 - xj)) * (
                2.0 * (t - xj) / (xj2 - xj)
                    - 2.0 * (t - xj1) * (xj3 - xj) / ((xj2 - xj1) * (xj3 - xj1))
                )
            else -> 2.0 * (t - xj3) / ((xj3 - xj1) * (xj3 - xj2))
        }
    }

    /** Гиперболический минимальный сплайн omega^H_j(t). */
    fun omegaH(grid: Grid, j: Int, t: Double): Double {
        val xj = grid.x(j); val xj1 = grid.x(j + 1); val xj2 = grid.x(j + 2); val xj3 = grid.x(j + 3)
        if (t < xj || t > xj3) return 0.0
        fun sh(x: Double) = Math.sinh(x)
        fun ch(x: Double) = Math.cosh(x)
        val cosCenter = ch((xj2 - xj1) / 2.0)
        return when {
            t < xj1 -> cosCenter * sh((t - xj) / 2.0) * sh((t - xj) / 2.0) /
                (sh((xj1 - xj) / 2.0) * sh((xj2 - xj) / 2.0))
            t < xj2 -> (cosCenter / sh((xj1 - xj) / 2.0)) * (
                sh((t - xj) / 2.0) * sh((t - xj) / 2.0) / sh((xj2 - xj) / 2.0)
                    - sh((xj3 - xj) / 2.0) * sh((t - xj1) / 2.0) * sh((t - xj1) / 2.0) /
                    (sh((xj3 - xj1) / 2.0) * sh((xj2 - xj1) / 2.0))
                )
            else -> cosCenter * sh((xj3 - t) / 2.0) * sh((xj3 - t) / 2.0) /
                (sh((xj3 - xj1) / 2.0) * sh((xj3 - xj2) / 2.0))
        }
    }
}

/** Узлы x_j..x_{j+3} различны (нет слияния кратных узлов). */
fun nonDegenerate(grid: Grid, j: Int): Boolean =
    grid.x(j) < grid.x(j + 1) && grid.x(j + 1) < grid.x(j + 2) && grid.x(j + 2) < grid.x(j + 3)
