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

        /**
         * Геометрическая (неравномерная) сетка семейства sn-article:
         *   x_j = a + (b-a) (q^j - 1)/(q^n - 1),  q = R^{1/(n-1)},  j = 0..n.
         *
         * Фиксированное отношение крайних шагов h_{n-1}/h_0 = R; шаги растут
         * геометрически (h_j = (b-a) q^j (q-1)/(q^n-1)), так что локальный параметр
         * квазиравномерности постоянен: h_j/h_{j-1} = q =: mu_n, mu_n -> 1 при n->inf.
         * Сетка существенно неравномерна, но локально квазиравномерна.
         *
         * @param n число внутренних интервалов (>= 2).
         * @param R отношение крайних шагов (> 0). R=1 вырождается в равномерную.
         */
        fun geometric(n: Int, a: Double = 0.0, b: Double = 1.0, R: Double = 2.0): Grid {
            require(n >= 2) { "geometric: требуется n >= 2, получено n=$n" }
            require(R > 0.0) { "geometric: требуется R > 0, получено R=$R" }
            val q = Math.pow(R, 1.0 / (n - 1))
            if (kotlin.math.abs(q - 1.0) < 1e-15) {
                // Вырожденный случай R=1: равномерная сетка (q^n - 1 -> 0).
                return Grid(n, DoubleArray(n + 1) { a + (b - a) * it / n })
            }
            val denom = Math.pow(q, n.toDouble()) - 1.0
            return Grid(n, DoubleArray(n + 1) { j ->
                when (j) {
                    0 -> a
                    n -> b
                    else -> a + (b - a) * (Math.pow(q, j.toDouble()) - 1.0) / denom
                }
            })
        }

        /**
         * Градуированная двухмасштабная сетка с ФИКСИРОВАННЫМ (не зависящим от n)
         * параметром локальной квазиравномерности. Шаги чередуются между двумя
         * значениями s и ratio*s по шаблону s*(1, ratio, 1, ratio, ...), так что
         * отношение соседних шагов h_j/h_{j-1} равно ratio или 1/ratio при ЛЮБОМ n.
         * Нормировка: сумма шагов = b-a (интервал покрыт точно). При нечётном n
         * последний шаг остаётся непарным (одиночное значение шаблона).
         *
         * В отличие от geometric (mu_n = q -> 1 при n->inf), здесь mu = ratio
         * фиксировано: сетка остаётся существенно локально неравномерной при
         * измельчении (h->0), что и требуется для устойчиво большой константы
         * Лебега. Сетка локально квазиравномерна (соседние шаги ограничены), поэтому
         * минимальные сплайны на ней существуют.
         *
         * @param n число внутренних интервалов (>= 2).
         * @param ratio отношение соседних шагов (> 0). ratio=1 вырождается в равномерную.
         */
        fun graded(n: Int, a: Double = 0.0, b: Double = 1.0, ratio: Double = 2.0): Grid {
            require(n >= 2) { "graded: требуется n >= 2, получено n=$n" }
            require(ratio > 0.0) { "graded: требуется ratio > 0, получено ratio=$ratio" }
            // Множители шаблона: чётный индекс -> 1, нечётный -> ratio (пары s, ratio*s).
            val mult = DoubleArray(n) { if (it % 2 == 0) 1.0 else ratio }
            val sum = mult.sum()
            val s = (b - a) / sum // базовый шаг: нормировка на длину интервала
            val interior = DoubleArray(n + 1)
            interior[0] = a
            var acc = a
            for (i in 0 until n) { acc += s * mult[i]; interior[i + 1] = acc }
            interior[n] = b // защита от накопленной ошибки округления на правом конце
            return Grid(n, interior)
        }
    }
}
