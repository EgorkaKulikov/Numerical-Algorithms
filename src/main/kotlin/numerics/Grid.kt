package numerics

// ============================================================================
// 4. СЕТКА (net): тройные узлы на концах
// ============================================================================

/**
 * Сетка X: a=x_{-2}=x_{-1}=x_0 < x_1 < ... < x_{n-1} < x_n=x_{n+1}=x_{n+2}=b
 * с узлами кратности 3 на концах. Хранит x_j для j = -2 .. n+2; idx(j)=j+2.
 *
 * @param n число внутренних интервалов.
 * @param interior внутренние узлы x_0..x_n (размер n+1), включая концы a и b.
 */
class Grid(val n: Int, interior: DoubleArray) {
    init {
        require(interior.size == n + 1) { "interior должен иметь размер n+1" }
    }

    val a: Double = interior.first()
    val b: Double = interior.last()

    /** Узлы x_{-2..n+2}, длина n+5; idx(j) = j+2. */
    val knots: DoubleArray = DoubleArray(n + 5).also { arr ->
        arr[idx(-2)] = interior[0]; arr[idx(-1)] = interior[0]
        for (j in 0..n) arr[idx(j)] = interior[j]
        arr[idx(n + 1)] = interior[n]; arr[idx(n + 2)] = interior[n]
    }

    /** Шаг h = max_j (x_{j+1} - x_j) по внутренним интервалам. */
    val h: Double = (0 until n).maxOf { interior[it + 1] - interior[it] }

    /** Отображение математического индекса j в индекс массива. */
    fun idx(j: Int): Int = j + 2

    /** Узел x_j по математическому индексу j (-2..n+2). */
    fun x(j: Int): Double = knots[idx(j)]

    /** Внутренние узлы x_0..x_n (размер n+1) — точки разрыва для квадратуры. */
    val breakpoints: DoubleArray = DoubleArray(n + 1) { x(it) }

    companion object {
        /** Равномерная сетка: x_j = a + (b-a) j/n. */
        fun uniform(n: Int, a: Double = 0.0, b: Double = 1.0): Grid =
            Grid(n, DoubleArray(n + 1) { a + (b - a) * it / n })

        /**
         * Квазиравномерная сетка X^q: x_j = a + (b-a) Psi(j/n),
         * Psi(u) = u + amp*sin(2*pi*u) — гладкая монотонная с фиксированным (не
         * зависящим от n) параметром локальной квазиравномерности.
         */
        fun quasiUniform(n: Int, a: Double = 0.0, b: Double = 1.0, amp: Double = 0.04): Grid =
            Grid(n, DoubleArray(n + 1) { i ->
                val u = i.toDouble() / n
                a + (b - a) * (u + amp * Math.sin(2.0 * Math.PI * u))
            })
    }
}
