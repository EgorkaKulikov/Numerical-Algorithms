/*
 * VolterraSolver.kt
 * ============================================================================
 * ЧИСЛЕННЫЙ ЭКСПЕРИМЕНТ К СТАТЬЕ new-01 (раздел \label{sec:numerical})
 * Линейные интегральные уравнения ВОЛЬТЕРРА (оператор \mathcal V,
 * \int_a^t) в базисе квадратичных минимальных B_phi-сплайнов макс. гладкости.
 * ============================================================================
 *
 * СТАТУС. Файл — артефакт «Агента реализации численных методов (11)». Код НЕ
 * объявляется проверенным (это прерогатива агента 12). Все отклонения от
 * `papers/new-01/algorithms/numerical-scheme.md` и нехватка формул зафиксированы
 * в `papers/new-01/src/implementation-notes.md`.
 *
 * ЧТО РЕАЛИЗОВАНО
 * -----------------
 * Модифицированный метод сплайн-коллокации на основе метода Кулкарни для
 * ЛИНЕЙНЫХ уравнений Вольтерра (\mathcal V u(t)=\int_a^t K(t,s)u(s)ds):
 *   - II рода (V2):  u - \mathcal V u = f;
 *   - I рода  (V1):   \mathcal V u = f — КОРРЕКТНА (r3), сводится к V2
 *                     ДИФФЕРЕНЦИРОВАНИЕМ (m раз; здесь m=1, т.к. K(t,t)≠0),
 *                     БЕЗ регуляризации (в отличие от Фредгольма I рода).
 *
 * КЛЮЧЕВОЕ ОТЛИЧИЕ ОТ FredholmSolver.kt: переменный верхний предел [a,t]
 * (нет фиксированных глобальных узлов; L применяется замыканиями) и ГРАНИЧНЫЙ
 * член K(t,t)u(t) в производной (правило Лейбница).
 *
 * В отличие от эталона `UrysonSolver.kt` (НЕЛИНЕЙНЫЙ Урысон, Ньютон/квази-Ньютон)
 * здесь коллокационные системы ЛИНЕЙНЫ (прямой LU).
 *
 * ЧЕТЫРЕ СЕМЕЙСТВА (КВАЗИ)ПРОЕКЦИОННЫХ ФУНКЦИОНАЛОВ
 * -------------------------------------------------
 * Источники формул: `papers/new-01/prfunc-formulas.md` (theta) и
 * `papers/new-01/references/monograph.tex` (xi/mu/lambda):
 *   theta  — проекционные/prfunc (значения f в узлах и серединах); ПРОЕКТОР,
 *            биортогональны theta_i(omega_j)=delta_ij; редукция Кулкарни (L7) есть.
 *            Сверка с (pr_func_b): {1/14,-2/7,10/7,-2/7,1/14} на равномерной B-сетке.
 *   xi     — де Бура--Фикса \xi_j^{<1>} (monograph, \subsection «Функционалы типа
 *            де Бура-Фикса», \xi_j^{<1>}(u)=u(x_{j+1})+C_j u'(x_{j+1})):
 *            ИСПОЛЬЗУЮТ ЗНАЧЕНИЕ И ПРОИЗВОДНУЮ, работа в C^1; ПРОЕКТОР (biorth
 *            (bior_m=2)); редукция Кулкарни (L7) есть.
 *   mu     — усредняющие \mu_j (monograph, (mu_j(f)) на вспом. сетке Y,
 *            y_j=x_{j+1}+theta(x_{j+2}-x_{j+1}), theta=1/2). КВАЗИИНТЕРПОЛЯНТ.
 *   lambda — трёхточечные \lambda_j (monograph, (lambda_j(f)) по точкам
 *            x_{j+1}, x_{j+3/2}=x_{j+1}+theta^(x_{j+2}-x_{j+1}), x_{j+2}; theta^=1/2;
 *            для B: -1/2(f(x_{j+1})-4 f(x_{j+3/2})+f(x_{j+2}))). КВАЗИИНТЕРПОЛЯНТ.
 *
 * Для theta,lambda,mu используется только значение f; для xi — значение И
 * производная (поэтому оператор предоставляет d/dt от \mathcal K omega_i и \mathcal K f).
 *
 * СХЕМЫ (L = \mathcal K либо \mathcal K_eff; numerical-scheme.md §5, discrete-problem.md)
 * --------------------------------------------------------------------------------------
 *   M_{j,i}   = chi_j(L omega_i),   M2_{j,i} = chi_j(L(L omega_i)),
 *   g_j       = chi_j(f),           d_j      = chi_j(L f).
 *   base      : (I - M) c = g,  u_h = sum c_i omega_i.
 *   Sloan     : ~u_h(t) = f(t) + (L u_h)(t).
 *   Kulkarni  (theta,xi): (I - M - M2 + M^2) c = (I - M) g + d;
 *               u_h^K = y_h + (I - P_chi)[f + L y_h].
 *   iter.Kulk : ^u_h^K = f + L u_h^K.
 *   Kulkarni  (mu,lambda): без редукции L7 — прямая итерация конечноранговым
 *               оператором U^K_h = P_chi L + L P_chi - P_chi L P_chi  [численное наблюдение].
 *
 * МЕТРИКИ: E_h = max|u*-u_h| на 100n+1 точках; p_h=log2(E_h/E_{h/2}); C_h=E_h/h^p.
 *
 * СТРУКТУРА ФАЙЛА
 * --------------
 * 1.LinearAlgebra 2.GaussLegendre 3.GeneratingSystem(+cross/dot/det/invert3)
 * 4.Grid 5.MinimalSplineBasis(+ReferenceSplines) 6.Функционалы (theta/xi/mu/lambda)
 * 7.KernelF + VolterraOperator 8.ModelProblem 9.SecondKindSolver 10.FirstKindSolver
 * 11.Метрики 12.HealthChecks 13.Fmt+Tables+main.
 *
 * ВОСПРОИЗВЕДЕНИЕ: JDK 11+, kotlinc; внешних зависимостей нет.
 *   kotlinc FredholmSolver.kt -include-runtime -d fred.jar && java -jar fred.jar
 */

import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.system.exitProcess

// ============================================================================
// 1. УТИЛИТЫ ЛИНЕЙНОЙ АЛГЕБРЫ
// ============================================================================

/**
 * Минимальная плотная линейная алгебра на Double: матрично-векторные/матричные
 * операции и решение СЛАУ методом LU с частичным выбором ведущего элемента.
 */
object LinearAlgebra {

    /** Нулевая матрица rows x cols. */
    fun zeros(rows: Int, cols: Int): Array<DoubleArray> = Array(rows) { DoubleArray(cols) }

    /** Единичная матрица n x n. */
    fun identity(n: Int): Array<DoubleArray> = Array(n) { i -> DoubleArray(n) { j -> if (i == j) 1.0 else 0.0 } }

    /** Произведение матрицы A (m x k) на вектор x (k) -> вектор (m). */
    fun matVec(a: Array<DoubleArray>, x: DoubleArray): DoubleArray {
        val m = a.size
        val out = DoubleArray(m)
        for (i in 0 until m) {
            var s = 0.0
            val row = a[i]
            for (j in x.indices) s += row[j] * x[j]
            out[i] = s
        }
        return out
    }

    /** Транспонированное произведение A^T y, A: m x n, y: m -> вектор n. */
    fun matTransVec(a: Array<DoubleArray>, y: DoubleArray): DoubleArray {
        val m = a.size
        val n = a[0].size
        val out = DoubleArray(n)
        for (i in 0 until m) {
            val row = a[i]
            val yi = y[i]
            for (j in 0 until n) out[j] += row[j] * yi
        }
        return out
    }

    /** Произведение матриц A (m x k) на B (k x p) -> (m x p). */
    fun matMat(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> {
        val m = a.size
        val k = b.size
        val p = b[0].size
        val out = zeros(m, p)
        for (i in 0 until m) {
            val ai = a[i]
            val oi = out[i]
            for (l in 0 until k) {
                val ail = ai[l]
                if (ail == 0.0) continue
                val bl = b[l]
                for (j in 0 until p) oi[j] += ail * bl[j]
            }
        }
        return out
    }

    /** Поэлементная сумма матриц A + s*B (одинаковые размеры). */
    fun addScaled(a: Array<DoubleArray>, b: Array<DoubleArray>, s: Double): Array<DoubleArray> {
        val out = Array(a.size) { a[it].copyOf() }
        for (i in a.indices) for (j in a[i].indices) out[i][j] += s * b[i][j]
        return out
    }

    /** Евклидова норма вектора. */
    fun norm2(x: DoubleArray): Double = sqrt(x.fold(0.0) { acc, v -> acc + v * v })

    /** Бесконечная (равномерная) норма вектора. */
    fun normInf(x: DoubleArray): Double = x.fold(0.0) { acc, v -> maxOf(acc, abs(v)) }

    /**
     * Решение плотной СЛАУ A x = b методом LU с частичным выбором ведущего
     * элемента. Матрица A и вектор b не изменяются (работаем на копиях).
     */
    fun solve(a: Array<DoubleArray>, b: DoubleArray): DoubleArray {
        val n = a.size
        val lu = Array(n) { a[it].copyOf() }
        val x = b.copyOf()
        for (col in 0 until n) {
            var pivRow = col
            var pivVal = abs(lu[col][col])
            for (r in col + 1 until n) {
                val v = abs(lu[r][col])
                if (v > pivVal) { pivVal = v; pivRow = r }
            }
            if (pivVal < 1e-300) error("LU: матрица вырождена (col=$col)")
            if (pivRow != col) {
                val t = lu[col]; lu[col] = lu[pivRow]; lu[pivRow] = t
                val tx = x[col]; x[col] = x[pivRow]; x[pivRow] = tx
            }
            val pivotR = lu[col]
            val pivot = pivotR[col]
            for (r in col + 1 until n) {
                val factor = lu[r][col] / pivot
                lu[r][col] = factor
                val rowR = lu[r]
                for (c in col + 1 until n) rowR[c] -= factor * pivotR[c]
                x[r] -= factor * x[col]
            }
        }
        for (i in n - 1 downTo 0) {
            var s = x[i]
            val row = lu[i]
            for (j in i + 1 until n) s -= row[j] * x[j]
            x[i] = s / row[i]
        }
        return x
    }

    /** Симметрия: max|A - A^T| (диагностика). */
    fun maxAsymmetry(a: Array<DoubleArray>): Double {
        var m = 0.0
        for (i in a.indices) for (j in a.indices) m = maxOf(m, abs(a[i][j] - a[j][i]))
        return m
    }
}

// ============================================================================
// 2. КВАДРАТУРА ГАУССА--ЛЕЖАНДРА
// ============================================================================

/**
 * Составная квадратура Гаусса--Лежандра по подынтервалам. Базовый инструмент для
 * интегралов \mathcal K omega_i, M, M^{(2)}, правых частей и L2-величин.
 * Порядок выбирается выше порядка аппроксимации проектора (чтобы погрешность
 * квадратур не маскировала различие Слоан/Кулкарни; risk R3 numerical-scheme.md).
 */
class GaussLegendre(val nodesPerSub: Int = 8) {

    private val refNodes: DoubleArray
    private val refWeights: DoubleArray

    init {
        require(nodesPerSub >= 1)
        val (nodes, weights) = gaussLegendreReference(nodesPerSub)
        refNodes = nodes
        refWeights = weights
    }

    /** Интеграл f по составному разбиению breakpoints (возрастающие, с концами). */
    fun integrate(breakpoints: DoubleArray, f: (Double) -> Double): Double {
        var sum = 0.0
        for (k in 0 until breakpoints.size - 1) {
            val lo = breakpoints[k]
            val hi = breakpoints[k + 1]
            if (hi <= lo) continue
            val half = 0.5 * (hi - lo)
            val mid = 0.5 * (hi + lo)
            for (q in refNodes.indices) {
                val t = mid + half * refNodes[q]
                sum += half * refWeights[q] * f(t)
            }
        }
        return sum
    }

    /** Интеграл по одному отрезку [lo, hi]. */
    fun integrateInterval(lo: Double, hi: Double, f: (Double) -> Double): Double =
        integrate(doubleArrayOf(lo, hi), f)

    /** Эталонные узлы и веса на [-1,1] (для ручного обхода подынтервалов). */
    fun refNodesWeights(): Pair<DoubleArray, DoubleArray> = refNodes to refWeights

    companion object {
        /**
         * Узлы и веса Гаусса--Лежандра на [-1,1] для m точек (метод Ньютона по
         * нулям многочлена Лежандра P_m). Точна для многочленов степени <= 2m-1.
         */
        fun gaussLegendreReference(m: Int): Pair<DoubleArray, DoubleArray> {
            val nodes = DoubleArray(m)
            val weights = DoubleArray(m)
            for (i in 0 until (m + 1) / 2) {
                var x = Math.cos(Math.PI * (i + 0.75) / (m + 0.5))
                var dp = 0.0
                repeat(100) {
                    var p0 = 1.0
                    var p1 = x
                    for (k in 2..m) {
                        val p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                        p0 = p1; p1 = p2
                    }
                    dp = m * (x * p1 - p0) / (x * x - 1.0)
                    val dx = p1 / dp
                    x -= dx
                    if (abs(dx) < 1e-15) return@repeat
                }
                var p0 = 1.0
                var p1 = x
                for (k in 2..m) {
                    val p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                    p0 = p1; p1 = p2
                }
                dp = m * (x * p1 - p0) / (x * x - 1.0)
                val w = 2.0 / ((1.0 - x * x) * dp * dp)
                nodes[i] = -x
                nodes[m - 1 - i] = x
                weights[i] = w
                weights[m - 1 - i] = w
            }
            return nodes to weights
        }
    }
}

// ============================================================================
// 3. ПОРОЖДАЮЩАЯ ВЕКТОР-ФУНКЦИЯ phi(t) = (1, rho(t), sigma(t))^T
// ============================================================================

/**
 * Порождающая вектор-функция phi(t) = (1, rho(t), sigma(t))^T и её производные
 * до второго порядка. phi_0 == 1 обеспечивает разбиение единицы.
 */
class GeneratingSystem(
    val name: String,
    val rho: (Double) -> Double,
    val sigma: (Double) -> Double,
    val rhoD: (Double) -> Double,
    val sigmaD: (Double) -> Double,
    val rhoDD: (Double) -> Double,
    val sigmaDD: (Double) -> Double,
) {
    /** phi(t) = (1, rho(t), sigma(t)). */
    fun phi(t: Double): DoubleArray = doubleArrayOf(1.0, rho(t), sigma(t))

    /** phi'(t) = (0, rho'(t), sigma'(t)). */
    fun phiD(t: Double): DoubleArray = doubleArrayOf(0.0, rhoD(t), sigmaD(t))

    /** phi''(t) = (0, rho''(t), sigma''(t)). */
    fun phiDD(t: Double): DoubleArray = doubleArrayOf(0.0, rhoDD(t), sigmaDD(t))

    /** Вронскиан det(phi, phi', phi'') — проверка невырожденности. */
    fun wronskian(t: Double): Double = det3(phi(t), phiD(t), phiDD(t))

    companion object {
        /** Полиномиальная phi^B(t) = (1, t, t^2)^T. */
        val B = GeneratingSystem(
            name = "B",
            rho = { t -> t }, sigma = { t -> t * t },
            rhoD = { 1.0 }, sigmaD = { t -> 2.0 * t },
            rhoDD = { 0.0 }, sigmaDD = { 2.0 },
        )

        /** Гиперболическая phi^H(t) = (1, sinh t, cosh t)^T. */
        val H = GeneratingSystem(
            name = "H",
            rho = { t -> Math.sinh(t) }, sigma = { t -> Math.cosh(t) },
            rhoD = { t -> Math.cosh(t) }, sigmaD = { t -> Math.sinh(t) },
            rhoDD = { t -> Math.sinh(t) }, sigmaDD = { t -> Math.cosh(t) },
        )

        /** Тригонометрическая phi^T(t) = (1, sin t, cos t)^T. */
        val T = GeneratingSystem(
            name = "T",
            rho = { t -> Math.sin(t) }, sigma = { t -> Math.cos(t) },
            rhoD = { t -> Math.cos(t) }, sigmaD = { t -> -Math.sin(t) },
            rhoDD = { t -> -Math.sin(t) }, sigmaDD = { t -> -Math.cos(t) },
        )
    }
}

/** Векторное произведение u x v в R^3. */
fun cross3(u: DoubleArray, v: DoubleArray): DoubleArray = doubleArrayOf(
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0],
)

/** Скалярное произведение в R^3. */
fun dot3(u: DoubleArray, v: DoubleArray): Double = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

/** Определитель 3x3 из столбцов-векторов (a, b, c). */
fun det3(a: DoubleArray, b: DoubleArray, c: DoubleArray): Double = dot3(a, cross3(b, c))

/**
 * Обращение матрицы 3x3 со столбцами c0, c1, c2 (строки M^{-1} = (c1 x c2,
 * c2 x c0, c0 x c1)/det). Используется для M_k^{-1} при устойчивом построении omega_j.
 */
fun invert3(c0: DoubleArray, c1: DoubleArray, c2: DoubleArray): Array<DoubleArray> {
    val det = det3(c0, c1, c2)
    val r0 = cross3(c1, c2)
    val r1 = cross3(c2, c0)
    val r2 = cross3(c0, c1)
    return arrayOf(
        doubleArrayOf(r0[0] / det, r0[1] / det, r0[2] / det),
        doubleArrayOf(r1[0] / det, r1[1] / det, r1[2] / det),
        doubleArrayOf(r2[0] / det, r2[1] / det, r2[2] / det),
    )
}
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
        val coef = dot3(dJ2, phiJ1) / dot3(dJ2, phiDJ1)
        return doubleArrayOf(
            phiJ1[0] - coef * phiDJ1[0],
            phiJ1[1] - coef * phiDJ1[1],
            phiJ1[2] - coef * phiDJ1[2],
        )
    }

    /** Индекс сеточного интервала k с x_k <= t < x_{k+1} (для t=b возвращает n-1). */
    private fun intervalOf(t: Double): Int {
        var k = 0
        while (k < n - 1 && t >= grid.x(k + 1)) k++
        return k
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
// ============================================================================
// 6. ЧЕТЫРЕ СЕМЕЙСТВА (КВАЗИ)ПРОЕКЦИОННЫХ ФУНКЦИОНАЛОВ
// theta (prfunc), xi (де Бура--Фикса, значение+производная), mu (усредняющие),
// lambda (трёхточечные). См. шапку файла об источниках формул.
// ============================================================================

/**
 * Общий интерфейс аппроксимационного функционала chi_j. Для theta,mu,lambda нужны
 * только значения f; для xi нужна и производная f'. Поэтому интерфейс принимает
 * функцию и её производную; семейства без производных её просто игнорируют.
 */
interface ApproxFunctional {
    /** chi_j(f) по функции f и её производной fD. */
    fun apply(f: (Double) -> Double, fD: (Double) -> Double): Double

    /** Сумма |коэффициентов| — для оценки нормы C_chi. */
    fun absSum(): Double
}

/** Удобная обёртка: chi_j(f) без явной производной (производная = 0). */
fun ApproxFunctional.apply(f: (Double) -> Double): Double = apply(f) { 0.0 }

/**
 * Семейство (квази)проекционных функционалов {chi_j}_{j=-2}^{n-1} и (квази)проектор
 * P_chi g = sum_j chi_j(g) omega_j. Общий интерфейс для theta/xi/mu/lambda.
 *
 * @property isProjector true для проекторов (theta, xi): P^2=P, биортогональность,
 *           редукция Кулкарни (L7) применима. false для квазиинтерполянтов (mu, lambda).
 * @property usesDerivative true для xi (работа в C^1).
 */
abstract class FunctionalFamily(val basis: MinimalSplineBasis, val name: String) {
    val grid = basis.grid
    val n = grid.n
    abstract val isProjector: Boolean
    abstract val usesDerivative: Boolean

    /** Функционал chi_j, j = -2..n-1. */
    abstract fun chi(j: Int): ApproxFunctional

    /** Коэффициенты проекции P_chi g = sum chi_j(g) omega_j: вектор (chi_j(g)) размера n+2. */
    fun projectorCoeffs(g: (Double) -> Double, gD: (Double) -> Double = { 0.0 }): DoubleArray =
        DoubleArray(n + 2) { chi(it - 2).apply(g, gD) }

    /** Максимум sum_k |коэффициенты| по всем j — константа C_chi. */
    fun cChi(): Double = (-2..n - 1).maxOf { chi(it).absSum() }
}

// ----------------------------------------------------------------------------
// 6.1 theta — ПРОЕКЦИОННЫЕ ФУНКЦИОНАЛЫ (prfunc-formulas.md)
// ----------------------------------------------------------------------------

/** Линейная комбинация значений f в точках nodes с коэффициентами coeffs (только значения). */
class ValueFunctional(val nodes: DoubleArray, val coeffs: DoubleArray) : ApproxFunctional {
    init { require(nodes.size == coeffs.size) }
    override fun apply(f: (Double) -> Double, fD: (Double) -> Double): Double {
        var s = 0.0
        for (k in nodes.indices) s += coeffs[k] * f(nodes[k])
        return s
    }
    override fun absSum(): Double = coeffs.fold(0.0) { acc, v -> acc + abs(v) }
}

/**
 * Семейство проекционных theta_j (prfunc). Внутренние/краевые строятся локальной
 * биортогонализацией (решение theta_j(omega_i)=delta_ij по узлам и серединам) —
 * устойчивое эквивалентное представление, сверяемое с (pr_func_b) в health-check.
 */
class ProjFunctionals(basis: MinimalSplineBasis) : FunctionalFamily(basis, "theta") {
    override val isProjector = true
    override val usesDerivative = false
    private val funcs: Array<ApproxFunctional> = Array(n + 2) { buildTheta(it - 2) }
    override fun chi(j: Int): ApproxFunctional = funcs[j + 2]

    private fun mid(p: Double, q: Double) = 0.5 * (p + q)

    private fun buildTheta(j: Int): ApproxFunctional {
        val x0 = grid.x(0); val x1 = grid.x(1)
        val xnm1 = grid.x(n - 1); val xn = grid.x(n)
        return when (j) {
            -2 -> ValueFunctional(doubleArrayOf(x0), doubleArrayOf(1.0))
            -1 -> localFunctional(j = -1, points = doubleArrayOf(x0, mid(x0, x1), x1), indices = intArrayOf(-2, -1, 0))
            n - 1 -> ValueFunctional(doubleArrayOf(xn), doubleArrayOf(1.0))
            n - 2 -> localFunctional(j = n - 2, points = doubleArrayOf(xnm1, mid(xnm1, xn), xn), indices = intArrayOf(n - 3, n - 2, n - 1))
            else -> {
                val xj = grid.x(j); val xj1 = grid.x(j + 1); val xj2 = grid.x(j + 2); val xj3 = grid.x(j + 3)
                localFunctional(
                    j = j,
                    points = doubleArrayOf(xj, mid(xj, xj1), mid(xj1, xj2), mid(xj2, xj3), xj3),
                    indices = intArrayOf(j - 2, j - 1, j, j + 1, j + 2),
                )
            }
        }
    }

    /** Локальная биортогонализация: coeff так, что sum_p coeff_p omega_i(points_p)=delta_ij. */
    private fun localFunctional(j: Int, points: DoubleArray, indices: IntArray): ValueFunctional {
        val m = points.size
        val matrix = Array(m) { r -> DoubleArray(m) { c -> basis.omega(indices[r], points[c]) } }
        val rhs = DoubleArray(m) { if (indices[it] == j) 1.0 else 0.0 }
        val coeff = LinearAlgebra.solve(matrix, rhs)
        return ValueFunctional(points, coeff)
    }

    /** Закрытая формула (pr_func) для внутреннего j (только для health-check). */
    fun closedFormInternal(j: Int): ValueFunctional {
        val xj = grid.x(j); val xj1 = grid.x(j + 1); val xj2 = grid.x(j + 2); val xj3 = grid.x(j + 3)
        val a = basis.omega(j, mid(xj, xj1))
        val c = basis.omega(j, mid(xj1, xj2))
        val e = basis.omega(j, mid(xj2, xj3))
        val b = basis.omega(j, xj1)
        val d = basis.omega(j, xj2)
        val k1 = c * c * d - b * c * e - a * d * e - d * e * e
        return ValueFunctional(
            doubleArrayOf(xj, mid(xj, xj1), mid(xj1, xj2), mid(xj2, xj3), xj3),
            doubleArrayOf(e * e / k1, -d * e / k1, (c * d - b * e) / k1, -d * e / k1, e * e / k1),
        )
    }
}
// ----------------------------------------------------------------------------
// 6.2 xi — ФУНКЦИОНАЛЫ ДЕ БУРА--ФИКСА (monograph, значение + производная, C^1)
// ----------------------------------------------------------------------------

/**
 * Функционал вида xi(u) = u(node) + cD * u'(node) (де Бура--Фикса r=1).
 * При cD=0 и chаистом значении — краевой u(x_0)/u(x_n).
 */
class DerivFunctional(val node: Double, val cD: Double) : ApproxFunctional {
    override fun apply(f: (Double) -> Double, fD: (Double) -> Double): Double = f(node) + cD * fD(node)
    override fun absSum(): Double = 1.0 + abs(cD)
}

/**
 * Семейство xi_j^{<1>}(u) = u(x_{j+1}) + C_j u'(x_{j+1}) (monograph, формула xi^{<1>}):
 * C_j = ((sigma_{j+2}-sigma_{j+1})rho'_{j+2} - (rho_{j+2}-rho_{j+1})sigma'_{j+2})
 *       / (rho'_{j+2}sigma'_{j+1} - rho'_{j+1}sigma'_{j+2}).
 * ПРОЕКТОР (биортогональность (bior_m=2)); работает в C^1.
 * Краевые j=-2,n-1: чистые значения u(x_0), u(x_n).
 */
class DeBoorFixFunctionals(basis: MinimalSplineBasis) : FunctionalFamily(basis, "xi") {
    override val isProjector = true
    override val usesDerivative = true
    private val sys = basis.sys
    private val funcs: Array<ApproxFunctional> = Array(n + 2) { buildXi(it - 2) }
    override fun chi(j: Int): ApproxFunctional = funcs[j + 2]

    private fun buildXi(j: Int): ApproxFunctional {
        if (j == -2) return DerivFunctional(grid.x(0), 0.0)
        if (j == n - 1) return DerivFunctional(grid.x(n), 0.0)
        val x1 = grid.x(j + 1); val x2 = grid.x(j + 2)
        val rho1 = sys.rho(x1); val rho2 = sys.rho(x2)
        val sig1 = sys.sigma(x1); val sig2 = sys.sigma(x2)
        val rhoD1 = sys.rhoD(x1); val rhoD2 = sys.rhoD(x2)
        val sigD1 = sys.sigmaD(x1); val sigD2 = sys.sigmaD(x2)
        val denom = rhoD2 * sigD1 - rhoD1 * sigD2
        val cD = ((sig2 - sig1) * rhoD2 - (rho2 - rho1) * sigD2) / denom
        return DerivFunctional(x1, cD)
    }
}

// ----------------------------------------------------------------------------
// 6.3 mu — УСРЕДНЯЮЩИЕ ФУНКЦИОНАЛЫ (monograph, (mu_j(f)), сетка Y, theta=1/2)
// ----------------------------------------------------------------------------

/**
 * Усредняющие mu_j(f) = a_j f(y_{j-1}) + b_j f(y_j) + c_j f(y_{j+1}),
 * y_j = x_{j+1} + theta(x_{j+2}-x_{j+1}), theta=1/2 (monograph (net_Y)). Коэффициенты
 * из условия точности на span{1,rho,sigma}: система
 *   [1,1,1; rho(y_{j-1}),rho(y_j),rho(y_{j+1}); sigma(...)] (a,b,c)^T = a^N_j,
 * где a^N_j = (1, S_{j+1}(rho,sigma,rho'), S_{j+1}(rho,sigma,sigma')) — вектор
 * аппроксимационного соотношения (тот же, что строит базис). КВАЗИИНТЕРПОЛЯНТ.
 * Для B при theta=1/2 на равномерной сетке: -1/8(f(y_{j-1})-10 f(y_j)+f(y_{j+1})).
 */
class AveragingFunctionals(basis: MinimalSplineBasis, val theta: Double = 0.5) :
    FunctionalFamily(basis, "mu") {
    override val isProjector = false
    override val usesDerivative = false
    private val sys = basis.sys
    private val funcs: Array<ApproxFunctional> = Array(n + 2) { buildMu(it - 2) }
    override fun chi(j: Int): ApproxFunctional = funcs[j + 2]

    /** y_j по (net_Y): краевые y_{-2}=x_0, y_{n-1}=x_n; внутренние — x_{j+1}+theta(x_{j+2}-x_{j+1}). */
    private fun yNode(j: Int): Double = when (j) {
        -2 -> grid.x(0)
        n - 1 -> grid.x(n)
        else -> grid.x(j + 1) + theta * (grid.x(j + 2) - grid.x(j + 1))
    }

    /** Вектор a^N_j (аппроксимационного соотношения) — как в MinimalSplineBasis.computeA. */
    private fun aN(j: Int): DoubleArray {
        val xj1 = grid.x(j + 1); val xj2 = grid.x(j + 2)
        val phiJ1 = sys.phi(xj1)
        if (xj1 == xj2) return phiJ1
        val phiDJ1 = sys.phiD(xj1)
        val dJ2 = cross3(sys.phi(xj2), sys.phiD(xj2))
        val coef = dot3(dJ2, phiJ1) / dot3(dJ2, phiDJ1)
        return doubleArrayOf(phiJ1[0] - coef * phiDJ1[0], phiJ1[1] - coef * phiDJ1[1], phiJ1[2] - coef * phiDJ1[2])
    }

    private fun buildMu(j: Int): ApproxFunctional {
        if (j == -2) return ValueFunctional(doubleArrayOf(grid.x(0)), doubleArrayOf(1.0))
        if (j == n - 1) return ValueFunctional(doubleArrayOf(grid.x(n)), doubleArrayOf(1.0))
        val ym = yNode(j - 1); val y0 = yNode(j); val yp = yNode(j + 1)
        val matrix = arrayOf(
            doubleArrayOf(1.0, 1.0, 1.0),
            doubleArrayOf(sys.rho(ym), sys.rho(y0), sys.rho(yp)),
            doubleArrayOf(sys.sigma(ym), sys.sigma(y0), sys.sigma(yp)),
        )
        val coeff = LinearAlgebra.solve(matrix, aN(j))
        return ValueFunctional(doubleArrayOf(ym, y0, yp), coeff)
    }
}

// ----------------------------------------------------------------------------
// 6.4 lambda — ТРЁХТОЧЕЧНЫЕ ФУНКЦИОНАЛЫ (monograph, (lambda_j(f)), theta^=1/2)
// ----------------------------------------------------------------------------

/**
 * Трёхточечные lambda_j(f) по точкам x_{j+1}, x_{j+3/2}=x_{j+1}+theta^(x_{j+2}-x_{j+1}),
 * x_{j+2}; theta^=1/2 (выбор, prfunc-formulas.md / monograph (net) remark). Реализованы
 * через локальную аппроксимацию P^I на I=[x_{j+1},x_{j+2}] (monograph remark):
 * на (x_{j+1},x_{j+2}) активны omega_{j-1},omega_j,omega_{j+1}; решаем M c = fvals в трёх
 * точках, lambda_j(f) = коэффициент при omega_j = (M^{-1})[1][.] · fvals. КВАЗИИНТЕРПОЛЯНТ.
 * Для B при theta^=1/2: -1/2(f(x_{j+1})-4 f(x_{j+3/2})+f(x_{j+2})).
 */
class ThreePointFunctionals(basis: MinimalSplineBasis, val thetaHat: Double = 0.5) :
    FunctionalFamily(basis, "lambda") {
    override val isProjector = false
    override val usesDerivative = false
    private val funcs: Array<ApproxFunctional> = Array(n + 2) { buildLambda(it - 2) }
    override fun chi(j: Int): ApproxFunctional = funcs[j + 2]

    private fun buildLambda(j: Int): ApproxFunctional {
        if (j == -2) return ValueFunctional(doubleArrayOf(grid.x(0)), doubleArrayOf(1.0))
        if (j == n - 1) return ValueFunctional(doubleArrayOf(grid.x(n)), doubleArrayOf(1.0))
        val x1 = grid.x(j + 1); val x2 = grid.x(j + 2)
        val xMid = x1 + thetaHat * (x2 - x1)
        val points = doubleArrayOf(x1, xMid, x2)
        val active = intArrayOf(j - 1, j, j + 1) // активные на (x_{j+1},x_{j+2})
        // M[p][slot] = omega_active[slot](point_p); решаем M c = fvals -> lambda_j = c[slot==j].
        val mt = Array(3) { p -> DoubleArray(3) { s -> basis.omega(active[s], points[p]) } }
        // coeffs_p = (M^{-1})[1][p]: решаем M^T r = e_1 (строка 1 обратной).
        val mTrans = Array(3) { i -> DoubleArray(3) { p -> mt[p][i] } }
        val e1 = doubleArrayOf(0.0, 1.0, 0.0)
        val coeff = LinearAlgebra.solve(mTrans, e1)
        return ValueFunctional(points, coeff)
    }
}
// ============================================================================
// 7. ЯДРО И ОПЕРАТОР ВОЛЬТЕРРА \mathcal V u(t) = \int_a^t K(t,s) u(s) ds
// ============================================================================

/** Ядро K(t,s) линейного уравнения Вольтерра и (при необходимости для xi) dK/dt. */
class KernelV(val k: (Double, Double) -> Double, val kT: (Double, Double) -> Double = { _, _ -> 0.0 })

/**
 * Оператор Вольтерра: (\mathcal V u)(t) = \int_a^t K(t,s) u(s) ds, квадратура по [a,t].
 *
 * Из-за ПЕРЕМЕННОГО верхнего предела (в отличие от Фредгольма) фиксированный набор
 * глобальных гауссовых узлов непригоден: область интегрирования зависит от t.
 * Поэтому интеграл считается напрямую (замыканиями) по составному разбиению [a,t]
 * (узлы сетки, попавшие в (a,t), плюс концы a и t).
 *
 * Производная по правилу Лейбница (для xi-функционалов и метрики):
 *   d/dt (\mathcal V u)(t) = K(t,t) u(t) + \int_a^t dK/dt(t,s) u(s) ds.
 * ГРАНИЧНЫЙ член K(t,t) u(t) — специфика Вольтерра (у Фредгольма его нет).
 */
class VolterraOperator(val kernel: KernelV, val grid: Grid, val quad: GaussLegendre) {
    val a = grid.a
    val b = grid.b

    /** Составное разбиение [a, t]: внутренние узлы сетки < t, затем сам t. */
    private fun subBreakpoints(t: Double): DoubleArray {
        val bp = grid.breakpoints
        val list = ArrayList<Double>()
        for (x in bp) { if (x < t - 1e-15) list.add(x) else break }
        if (list.isEmpty()) list.add(a)
        list.add(t)
        return list.toDoubleArray()
    }

    /** (\mathcal V u)(t) = \int_a^t K(t,s) u(s) ds для произвольной u(s). */
    fun apply(t: Double, u: (Double) -> Double): Double {
        if (t <= a) return 0.0
        return quad.integrate(subBreakpoints(t)) { s -> kernel.k(t, s) * u(s) }
    }

    /**
     * d/dt (\mathcal V u)(t) = K(t,t) u(t) + \int_a^t dK/dt(t,s) u(s) ds (Лейбниц).
     * ВАЖНО: граничный член K(t,t)u(t) остаётся и при t=a (интеграл по [a,a] нулевой).
     * Этот член критичен для сведения V1->V2 на левом конце (g(a)=f'(a)/K(a,a)=u*(a)).
     */
    fun applyDeriv(t: Double, u: (Double) -> Double): Double {
        if (t < a) return 0.0
        val boundary = kernel.k(t, t) * u(t)
        val integral = if (t <= a) 0.0 else quad.integrate(subBreakpoints(t)) { s -> kernel.kT(t, s) * u(s) }
        return boundary + integral
    }
}

// ============================================================================
// 8. МОДЕЛЬНЫЕ ЗАДАЧИ
// ============================================================================

/**
 * Модельная задача: ядро, точное решение, род. Правая часть строится из точного
 * решения численно (квадратурой), без зашивания.
 * II рода: f = u* - \mathcal K u*;  I рода: f = \mathcal K u*.
 */
class ModelProblem(
    val name: String,
    val kernel: KernelV,
    val exact: (Double) -> Double,
    val exactDeriv: (Double) -> Double,
    val secondKind: Boolean,
    val a: Double = 0.0,
    val b: Double = 1.0,
) {
    /** Точная правая часть f(t). */
    fun rhsExact(t: Double, op: VolterraOperator): Double {
        val integral = op.apply(t) { s -> exact(s) }
        return if (secondKind) exact(t) - integral else integral
    }

    /** d/dt f(t) (для xi-функционалов): II рода u*' - d/dt \mathcal K u*. */
    fun rhsExactDeriv(t: Double, op: VolterraOperator): Double {
        val integralD = op.applyDeriv(t) { s -> exact(s) }
        return if (secondKind) exactDeriv(t) - integralD else integralD
    }

    companion object {
        /** V2span: K=e^{t-2s}, u*=t^2 ∈ span{1,t,t^2}=phi^B (машинная точность на B). */
        val V2span = ModelProblem(
            name = "V2span",
            kernel = KernelV({ t, s -> Math.exp(t - 2.0 * s) }, { t, s -> Math.exp(t - 2.0 * s) }),
            exact = { t -> t * t }, exactDeriv = { t -> 2.0 * t },
            secondKind = true,
        )

        /** V2: K=1/(1+t+s), u*=1/(t+1) (проверка порядка на полиномиальном базисе). */
        val V2 = ModelProblem(
            name = "V2",
            kernel = KernelV({ t, s -> 1.0 / (1.0 + t + s) }, { t, s -> -1.0 / ((1.0 + t + s) * (1.0 + t + s)) }),
            exact = { t -> 1.0 / (t + 1.0) }, exactDeriv = { t -> -1.0 / ((t + 1.0) * (t + 1.0)) },
            secondKind = true,
        )

        /** V2exp: K=e^{-(t-s)^2}, u*=e^t (согласован с phi^H). */
        val V2exp = ModelProblem(
            name = "V2exp",
            kernel = KernelV({ t, s -> Math.exp(-(t - s) * (t - s)) },
                { t, s -> -2.0 * (t - s) * Math.exp(-(t - s) * (t - s)) }),
            exact = { t -> Math.exp(t) }, exactDeriv = { t -> Math.exp(t) },
            secondKind = true,
        )

        /**
         * V2win: K=t-s (K(t,t)=0 — слабое/сглаживающее ядро), u*=cos t.
         * Разведочный пример (tests/VolterraKernelSearch.kt), где ОДНОШАГОВЫЙ Кулкарни
         * СТРОГО точнее Слоана (на ~2 порядка при n=32): p_h база≈3, Слоан≈4,
         * Кулкарни≈5, итер.Кулкарни≈6. Причина: K(t,t)=0 усиливает сглаживание \mathcal V,
         * и член суперсходимости (I-P)\mathcal V(I-P) работает в полную силу.
         */
        val V2win = ModelProblem(
            name = "V2win",
            kernel = KernelV({ t, s -> t - s }, { _, _ -> 1.0 }),
            exact = { t -> Math.cos(t) }, exactDeriv = { t -> -Math.sin(t) },
            secondKind = true,
        )

        /**
         * V1: K=1+t-s (K(t,t)=1≠0), u*=cos t, уравнение I рода \mathcal V u = f.
         * Корректна (r3): сводится к V2 однократным дифференцированием (m=1, т.к. K(t,t)≠0).
         */
        val V1 = ModelProblem(
            name = "V1",
            kernel = KernelV({ t, s -> 1.0 + t - s }, { _, _ -> 1.0 }),
            exact = { t -> Math.cos(t) }, exactDeriv = { t -> -Math.sin(t) },
            secondKind = false,
        )
    }
}
// ============================================================================
// 9. ЯДРО КОЛЛОКАЦИИ + РЕШАТЕЛЬ УРАВНЕНИЯ II РОДА (ЛИНЕЙНО)
// ============================================================================

/** Результат решения: вычислитель u_h(t). */
class SolutionFunc(val eval: (Double) -> Double)

/**
 * Линейный решатель уравнения II рода u - L u = f, L = c_L * \mathcal K
 * (c_L = 1 для F2; c_L = -1/alpha для F1 Wazwaz, \mathcal K_eff = -(1/alpha)\mathcal K).
 * Правая часть f и её производная задаются явно (fEff/fEffDeriv) для переиспользования в F1.
 *
 * Матрицы (discrete-problem.md):
 *   M_{j,i}  = chi_j(L omega_i),  M2_{j,i} = chi_j(L(L omega_i)),
 *   g_j      = chi_j(f),          d_j      = chi_j(L f).
 */
class SecondKindSolver(
    val basis: MinimalSplineBasis,
    val funcs: FunctionalFamily,
    val op: VolterraOperator,
    val cL: Double,
    val fEff: (Double) -> Double,
    val fEffDeriv: (Double) -> Double,
) {
    val grid = basis.grid
    val n = grid.n
    val dim = n + 2

    // ВНИМАНИЕ (отличие от Фредгольма): у оператора Вольтерра область интегрирования
    // [a,t] зависит от t, поэтому предвычисление на фиксированных узлах невозможно.
    // Все применения L = c_L \mathcal V выражаются через замыкания op.apply / op.applyDeriv.

    /** L g(t) = c_L (\mathcal V g)(t) и её производная (Лейбниц). */
    private fun applyL(g: (Double) -> Double): (Double) -> Double = { t -> cL * op.apply(t, g) }
    private fun applyLDeriv(g: (Double) -> Double): (Double) -> Double = { t -> cL * op.applyDeriv(t, g) }

    /** chi_j(g) по значениям g и g' (обёртка). */
    private fun chiOf(g: (Double) -> Double, gD: (Double) -> Double): DoubleArray =
        DoubleArray(dim) { funcs.chi(it - 2).apply(g, gD) }

    /** Матрица M_{j,i} = chi_j(L omega_i). Для xi учитывается производная (L omega_i)'. */
    fun matrixM(): Array<DoubleArray> {
        val m = LinearAlgebra.zeros(dim, dim)
        for (i in 0 until dim) {
            val idx = i - 2
            val omega = { s: Double -> basis.omega(idx, s) }
            val Lom = applyL(omega)
            val LomD = applyLDeriv(omega)
            for (j in 0 until dim) m[j][i] = funcs.chi(j - 2).apply(Lom, LomD)
        }
        return m
    }

    /** Матрица M2_{j,i} = chi_j(L(L omega_i)) (двойное применение L). */
    fun matrixM2(): Array<DoubleArray> {
        val m = LinearAlgebra.zeros(dim, dim)
        for (i in 0 until dim) {
            val idx = i - 2
            val Lom = applyL { s -> basis.omega(idx, s) }
            val LLom = applyL(Lom)
            val LLomD = applyLDeriv(Lom)
            for (j in 0 until dim) m[j][i] = funcs.chi(j - 2).apply(LLom, LLomD)
        }
        return m
    }

    /** g_j = chi_j(f). */
    fun vectorG(): DoubleArray = chiOf(fEff, fEffDeriv)

    /** d_j = chi_j(L f). */
    fun vectorD(): DoubleArray {
        val Lf = { t: Double -> cL * op.apply(t) { s -> fEff(s) } }
        val LfD = { t: Double -> cL * op.applyDeriv(t) { s -> fEff(s) } }
        return chiOf(Lf, LfD)
    }

    /** Базовая схема: (I - M) c = g. */
    fun solveBaseCoeffs(): DoubleArray {
        val m = matrixM()
        val a = LinearAlgebra.zeros(dim, dim)
        for (r in 0 until dim) { for (c in 0 until dim) a[r][c] = -m[r][c]; a[r][r] += 1.0 }
        return LinearAlgebra.solve(a, vectorG())
    }

    fun base(): SolutionFunc {
        val c = solveBaseCoeffs()
        return SolutionFunc { t -> basis.evalSpline(c, t) }
    }

    /** Слоан: ~u_h(t) = f(t) + (L u_h)(t). u_h — сплайн, L применяется замыканием. */
    fun sloan(): SolutionFunc {
        val c = solveBaseCoeffs()
        val Luh = applyL { s -> basis.evalSpline(c, s) }
        return SolutionFunc { t -> fEff(t) + Luh(t) }
    }

    /**
     * Кулкарни для проекторов (theta, xi): (I - M - M2 + M^2) c = (I - M) g + d;
     * u_h^K = y_h + (I - P_chi)[f + L y_h]. Для квазиинтерполянтов (mu, lambda) без
     * редукции — прямая итерация конечноранговым U^K_h [численное наблюдение].
     */
    fun kulkarni(): SolutionFunc {
        return if (funcs.isProjector) kulkarniProjector() else kulkarniQuasi()
    }

    private fun kulkarniProjector(): SolutionFunc {
        val m = matrixM(); val m2 = matrixM2(); val g = vectorG(); val d = vectorD()
        val mm = LinearAlgebra.matMat(m, m)
        // A = I - M - M2 + M^2
        val a = LinearAlgebra.zeros(dim, dim)
        for (r in 0 until dim) {
            for (c in 0 until dim) a[r][c] = -m[r][c] - m2[r][c] + mm[r][c]
            a[r][r] += 1.0
        }
        // rhs = (I - M) g + d
        val mg = LinearAlgebra.matVec(m, g)
        val rhs = DoubleArray(dim) { g[it] - mg[it] + d[it] }
        val c = LinearAlgebra.solve(a, rhs) // коэффициенты y_h
        // u_h^K = y_h + (I - P_chi)[f + L y_h]; (I - P_chi)w = w - P_chi w.
        val yh = { s: Double -> basis.evalSpline(c, s) }
        val Lyh = applyL(yh)
        val LyhD = applyLDeriv(yh)
        val wFun = { t: Double -> fEff(t) + Lyh(t) }
        val wDFun = { t: Double -> fEffDeriv(t) + LyhD(t) }
        val pwCoeffs = funcs.projectorCoeffs(wFun, wDFun)
        return SolutionFunc { t -> basis.evalSpline(c, t) + (wFun(t) - basis.evalSpline(pwCoeffs, t)) }
    }

    /**
     * Кулкарни для mu, lambda: итерация u^{(m+1)} = f + U^K_h u^{(m)},
     * U^K_h u = P_chi(L u) + L(P_chi u) - P_chi(L(P_chi u)). Работа в узлах квадратуры.
     * [численное наблюдение]: разрешимость/сходимость не гарантированы (нет P^2=P).
     */
    private val sampleXs: DoubleArray = DoubleArray(20 * n + 1) { grid.a + (grid.b - grid.a) * it / (20 * n) }

    private fun kulkarniQuasi(): SolutionFunc {
        var uVals = DoubleArray(sampleXs.size) { fEff(sampleXs[it]) }
        repeat(200) {
            val cur = uVals
            val uFun = { t: Double -> evalNodalLinear(t, cur) }
            val pc = funcs.projectorCoeffs(uFun)
            val pcFun = { s: Double -> basis.evalSpline(pc, s) }
            val LuFun = applyL(uFun)                       // L u
            val pLu = funcs.projectorCoeffs(LuFun)        // P_chi(L u)
            val LPu = applyL(pcFun)                        // L(P_chi u)
            val pLPu = funcs.projectorCoeffs(LPu)         // P_chi(L(P_chi u))
            val next = DoubleArray(sampleXs.size) { k ->
                val t = sampleXs[k]
                fEff(t) + basis.evalSpline(pLu, t) + LPu(t) - basis.evalSpline(pLPu, t)
            }
            var diff = 0.0
            for (k in next.indices) diff = maxOf(diff, abs(next[k] - cur[k]))
            uVals = next
            if (diff < 1e-12) return@repeat
        }
        val finalVals = uVals
        return SolutionFunc { t -> evalNodalLinear(t, finalVals) }
    }

    /** Кусочно-линейное восстановление по значениям на sampleXs (для mu/lambda). */
    private fun evalNodalLinear(t: Double, nodes: DoubleArray): Double {
        val xs = sampleXs
        if (t <= xs[0]) return nodes[0]
        if (t >= xs[xs.size - 1]) return nodes[xs.size - 1]
        var lo = 0; var hi = xs.size - 1
        while (hi - lo > 1) { val mid = (lo + hi) / 2; if (xs[mid] <= t) lo = mid else hi = mid }
        val w = (t - xs[lo]) / (xs[hi] - xs[lo])
        return nodes[lo] * (1 - w) + nodes[hi] * w
    }

    /** Итерированный Кулкарни: ^u_h^K = f + L u_h^K. */
    fun iteratedKulkarni(): SolutionFunc {
        val uK = kulkarni()
        val Luk = applyL { s -> uK.eval(s) }
        return SolutionFunc { t -> fEff(t) + Luk(t) }
    }
}
// ============================================================================
// 10. РЕШАТЕЛЬ УРАВНЕНИЯ I РОДА — СВЕДЕНИЕ ДИФФЕРЕНЦИРОВАНИЕМ (r3)
// ============================================================================

/**
 * V1 — уравнение Вольтерра I рода: (\mathcal V u)(t) = \int_a^t K(t,s)u(s)ds = f(t).
 * В отличие от Фредгольма I рода (некорректного, требует Wazwaz), здесь
 * задача КОРРЕКТНА (r3) и сводится к II роду дифференцированием.
 *
 * При K(t,t) ≠ 0 (m=1) дифференцируем \mathcal V u = f по t (Лейбниц):
 *   K(t,t) u(t) + \int_a^t K_t(t,s) u(s) ds = f'(t).
 * Деля на K(t,t), получаем V2:
 *   u(t) - \mathcal W u(t) = g(t),   \mathcal W u = \int_a^t [-K_t(t,s)/K(t,t)] u(s) ds,
 *   g(t) = f'(t)/K(t,t).
 * Далее — та же схема II рода (база/Слоан/Кулкарни) с c_L = 1.
 * Производная g'(t) (для xi) и редуцированное K_t — конечные разности.
 */
class FirstKindSolver(
    val problem: ModelProblem,
    val basis: MinimalSplineBasis,
    val funcs: FunctionalFamily,
    op: VolterraOperator,
) {
    private val grid = basis.grid
    private val quad = GaussLegendre(8)
    private val Ktt = { t: Double -> problem.kernel.k(t, t) }
    // Редуцированное ядро W: K_W(t,s) = -K_t(t,s)/K(t,t); K_W_t — центральной разностью.
    private val hFD = 1e-6
    private val kernelW = KernelV(
        k = { t, s -> -problem.kernel.kT(t, s) / Ktt(t) },
        kT = { t, s ->
            val kp = -problem.kernel.kT(t + hFD, s) / Ktt(t + hFD)
            val km = -problem.kernel.kT(t - hFD, s) / Ktt(t - hFD)
            (kp - km) / (2 * hFD)
        },
    )
    private val opW = VolterraOperator(kernelW, grid, quad)
    // g(t) = f'(t)/K(t,t), f — правая часть I рода; f'(t) = d/dt \mathcal V u* (Лейбниц).
    private val gEff = { t: Double -> problem.rhsExactDeriv(t, op) / Ktt(t) }
    private val gEffDeriv = { t: Double -> (gEff(t + hFD) - gEff(t - hFD)) / (2 * hFD) }
    private val inner = SecondKindSolver(basis, funcs, opW, cL = 1.0, fEff = gEff, fEffDeriv = gEffDeriv)

    fun base(): SolutionFunc = inner.base()
    fun sloan(): SolutionFunc = inner.sloan()
    fun kulkarni(): SolutionFunc = inner.kulkarni()
    fun iteratedKulkarni(): SolutionFunc = inner.iteratedKulkarni()
}

/** Фабрика решателя II рода для модельной задачи (c_L = 1, f = u* - \mathcal V u*). */
fun secondKindSolver(problem: ModelProblem, basis: MinimalSplineBasis, funcs: FunctionalFamily,
                     op: VolterraOperator): SecondKindSolver =
    SecondKindSolver(basis, funcs, op, cL = 1.0,
        fEff = { t -> problem.rhsExact(t, op) },
        fEffDeriv = { t -> problem.rhsExactDeriv(t, op) })

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

// ============================================================================
// 12. HEALTH-CHECKS
// ============================================================================

/** Результат одной самопроверки. */
class CheckResult(val name: String, val measured: Double, val threshold: Double, val critical: Boolean) {
    val ok: Boolean get() = !measured.isNaN() && measured <= threshold
}

/**
 * Набор health-checks (миррор UrysonSolver). Критический провал -> exit(1).
 */
object HealthChecks {
    private val quad = GaussLegendre(8)
    private val gridsU = Grid.uniform(8)
    private val gridsQ = Grid.quasiUniform(8)
    private val sampleTs = (0..200).map { it / 200.0 }
    private val allSys = listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)

    fun runAll(): List<CheckResult> = listOf(
        checkSplineVsB(), checkSplineVsH(), checkPartitionOfUnity(),
        checkBiorthTheta(), checkBiorthXi(), checkProjectorIdempotent(),
        checkExactOnSpan(), checkThetaReduction(), checkQuadrature(),
        checkRhsConsistency(), checkBaseSanity(),
    )

    private fun forEachGrid(action: (Grid) -> Double): Double = maxOf(action(gridsU), action(gridsQ))

    /** 1. omega == omegaB (+производные). */
    private fun checkSplineVsB(): CheckResult {
        val err = forEachGrid { grid ->
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            var m = 0.0
            for (ti in sampleTs) {
                val t = grid.a + (grid.b - grid.a) * ti
                for (j in -2..grid.n - 1) if (nonDegenerate(grid, j)) {
                    m = maxOf(m, abs(basis.omega(j, t) - ReferenceSplines.omegaB(grid, j, t)))
                    m = maxOf(m, abs(basis.omegaDeriv(j, t) - ReferenceSplines.omegaBDeriv(grid, j, t)))
                }
            }
            m
        }
        return CheckResult("1. omega == omegaB (и произв.)", err, 1e-10, true)
    }

    /** 2. omega == omegaH. */
    private fun checkSplineVsH(): CheckResult {
        val err = forEachGrid { grid ->
            val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
            var m = 0.0
            for (ti in sampleTs) {
                val t = grid.a + (grid.b - grid.a) * ti
                for (j in -2..grid.n - 1) if (nonDegenerate(grid, j))
                    m = maxOf(m, abs(basis.omega(j, t) - ReferenceSplines.omegaH(grid, j, t)))
            }
            m
        }
        return CheckResult("2. omega == omegaH", err, 1e-10, true)
    }

    /** 3. Разбиение единицы. */
    private fun checkPartitionOfUnity(): CheckResult {
        val err = forEachGrid { grid ->
            var m = 0.0
            for (sys in allSys) {
                val basis = MinimalSplineBasis(sys, grid)
                for (ti in sampleTs) {
                    val t = grid.a + (grid.b - grid.a) * ti
                    var sum = 0.0
                    for (j in -2..grid.n - 1) sum += basis.omega(j, t)
                    m = maxOf(m, abs(sum - 1.0))
                }
            }
            m
        }
        return CheckResult("3. Разбиение единицы", err, 1e-10, true)
    }

    /** 4. Биортогональность theta_i(omega_j)=delta. */
    private fun checkBiorthTheta(): CheckResult {
        val err = forEachGrid { grid ->
            var m = 0.0
            for (sys in allSys) {
                val basis = MinimalSplineBasis(sys, grid)
                val funcs = ProjFunctionals(basis)
                for (i in -2..grid.n - 1) for (j in -2..grid.n - 1) {
                    val v = funcs.chi(i).apply({ t -> basis.omega(j, t) }, { t -> basis.omegaDeriv(j, t) })
                    m = maxOf(m, abs(v - if (i == j) 1.0 else 0.0))
                }
            }
            m
        }
        return CheckResult("4. Биортогональность theta", err, 1e-9, true)
    }

    /** 5. Биортогональность xi_i(omega_j)=delta (с производной). */
    private fun checkBiorthXi(): CheckResult {
        val err = forEachGrid { grid ->
            var m = 0.0
            for (sys in allSys) {
                val basis = MinimalSplineBasis(sys, grid)
                val funcs = DeBoorFixFunctionals(basis)
                for (i in -2..grid.n - 1) for (j in -2..grid.n - 1) {
                    val v = funcs.chi(i).apply({ t -> basis.omega(j, t) }, { t -> basis.omegaDeriv(j, t) })
                    m = maxOf(m, abs(v - if (i == j) 1.0 else 0.0))
                }
            }
            m
        }
        return CheckResult("5. Биортогональность xi", err, 1e-9, true)
    }

    /** 6. Идемпотентность P_chi u_h = u_h для theta, xi (mu,lambda не обязаны). */
    private fun checkProjectorIdempotent(): CheckResult {
        val rnd = kotlin.random.Random(777)
        val err = forEachGrid { grid ->
            var m = 0.0
            for (sys in allSys) {
                val basis = MinimalSplineBasis(sys, grid)
                for (funcs in listOf(ProjFunctionals(basis), DeBoorFixFunctionals(basis))) {
                    val c = DoubleArray(grid.n + 2) { rnd.nextDouble(-1.0, 1.0) }
                    val uh = { t: Double -> basis.evalSpline(c, t) }
                    val uhD = { t: Double -> basis.evalSplineDeriv(c, t) }
                    val pc = funcs.projectorCoeffs(uh, uhD)
                    for (i in c.indices) m = maxOf(m, abs(pc[i] - c[i]))
                }
            }
            m
        }
        return CheckResult("6. Идемпотентность theta,xi", err, 1e-8, true)
    }

    /** 7. Точность на span{1,rho,sigma}: P_chi g = g для g=rho (все семейства). */
    private fun checkExactOnSpan(): CheckResult {
        val err = forEachGrid { grid ->
            var m = 0.0
            for (sys in allSys) {
                val basis = MinimalSplineBasis(sys, grid)
                val fams: List<FunctionalFamily> = listOf(
                    ProjFunctionals(basis), DeBoorFixFunctionals(basis),
                    AveragingFunctionals(basis), ThreePointFunctionals(basis),
                )
                for (funcs in fams) {
                    val pc = funcs.projectorCoeffs({ t -> sys.rho(t) }, { t -> sys.rhoD(t) })
                    for (ti in sampleTs) {
                        val t = grid.a + (grid.b - grid.a) * ti
                        m = maxOf(m, abs(sys.rho(t) - basis.evalSpline(pc, t)))
                    }
                }
            }
            m
        }
        return CheckResult("7. Точность на span (все chi)", err, 1e-8, true)
    }

    /** 8. Редукция theta к (pr_func_b) {1/14,-2/7,10/7,-2/7,1/14}. */
    private fun checkThetaReduction(): CheckResult {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val expected = doubleArrayOf(1.0 / 14.0, -2.0 / 7.0, 10.0 / 7.0, -2.0 / 7.0, 1.0 / 14.0)
        var m = 0.0
        for (j in 0..grid.n - 3) {
            val cf = funcs.closedFormInternal(j).coeffs
            for (k in cf.indices) m = maxOf(m, abs(cf[k] - expected[k]))
        }
        return CheckResult("8. Редукция theta к (pr_func_b)", m, 1e-10, true)
    }

    /** 9. Точность квадратуры: t^k до 15, e^t, 1/(t+1). */
    private fun checkQuadrature(): CheckResult {
        val bp = doubleArrayOf(0.0, 1.0)
        var m = 0.0
        for (k in 0..15) m = maxOf(m, abs(quad.integrate(bp) { t -> Math.pow(t, k.toDouble()) } - 1.0 / (k + 1)))
        m = maxOf(m, abs(quad.integrate(bp) { t -> Math.exp(t) } - (Math.E - 1.0)))
        m = maxOf(m, abs(quad.integrate(bp) { t -> 1.0 / (t + 1.0) } - Math.log(2.0)))
        return CheckResult("9. Точность квадратуры", m, 1e-10, true)
    }

    /** 10. Согласованность правой части: на span-задаче F2span база(B) даёт машинную точность. */
    private fun checkRhsConsistency(): CheckResult {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = VolterraOperator(ModelProblem.V2span.kernel, grid, quad)
        val solver = secondKindSolver(ModelProblem.V2span, basis, funcs, op)
        val eh = errorEh({ t -> ModelProblem.V2span.exact(t) }, solver.base().eval, grid)
        return CheckResult("10. Точность на span-задаче (база B)", eh, 1e-8, true)
    }

    /** 11. Sanity: базовая схема II рода сходится (E_8 > E_16 для F2/H). */
    private fun checkBaseSanity(): CheckResult {
        fun ehAt(nn: Int): Double {
            val grid = Grid.uniform(nn)
            val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
            val funcs = ProjFunctionals(basis)
            val op = VolterraOperator(ModelProblem.V2.kernel, grid, quad)
            val solver = secondKindSolver(ModelProblem.V2, basis, funcs, op)
            return errorEh({ t -> ModelProblem.V2.exact(t) }, solver.base().eval, grid)
        }
        val e8 = ehAt(8); val e16 = ehAt(16)
        val measured = if (e16 <= e8 && e8 < 1e-1) e16 else 1.0
        return CheckResult("11. Sanity базовой схемы (сходимость)", measured, 1e-1, true)
    }
}
// ============================================================================
// 13. ФОРМАТИРОВАНИЕ + ТАБЛИЦЫ + MAIN
// ============================================================================

object Fmt {
    fun e(x: Double): String = if (x.isNaN()) "---" else "%.3e".format(x)
    fun p(x: Double): String = if (x.isNaN()) "---" else "%.2f".format(x)
    fun h(x: Double): String = "%.4f".format(x)
    fun tex(x: Double): String {
        if (x.isNaN()) return "---"
        if (x == 0.0) return "0"
        val exp = Math.floor(Math.log10(abs(x))).toInt()
        val mant = x / Math.pow(10.0, exp.toDouble())
        return "%.3f".format(mant).replace(".", "{,}") + "\\!\\cdot\\!10^{" + exp + "}"
    }
}

object Tables {
    // Для Вольтерра matrixM2 = O(dim^2 * Q^2) (двойной \int_a^t), поэтому сетка ограничена n<=64.
    private val NS = listOf(8, 16, 32, 64)
    private val quad = GaussLegendre(8)

    private fun makeSolver(p: ModelProblem, sys: GeneratingSystem, fam: String, nn: Int): Pair<SecondKindSolver, Grid> {
        val grid = Grid.uniform(nn)
        val basis = MinimalSplineBasis(sys, grid)
        val funcs = family(fam, basis)
        val op = VolterraOperator(p.kernel, grid, quad)
        return SecondKindSolver(basis, funcs, op, 1.0,
            { t -> p.rhsExact(t, op) }, { t -> p.rhsExactDeriv(t, op) }) to grid
    }

    private fun family(name: String, basis: MinimalSplineBasis): FunctionalFamily = when (name) {
        "theta" -> ProjFunctionals(basis)
        "xi" -> DeBoorFixFunctionals(basis)
        "mu" -> AveragingFunctionals(basis)
        else -> ThreePointFunctionals(basis)
    }

    /** T1: V2exp, базисы B/H/T, базовая схема theta: E_h, p_h, C_h. */
    fun tablePhi(p: ModelProblem) {
        println("\n--- T1[${p.name}]: theta, базисы B/H/T (E_h,p_h,C_h) ---")
        for (sys in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
            val errs = ArrayList<Double>(); val hs = ArrayList<Double>()
            for (nn in NS) { val (s, grid) = makeSolver(p, sys, "theta", nn)
                hs.add(grid.h); errs.add(errorEh({ t -> p.exact(t) }, s.base().eval, grid)) }
            val ps = orders(errs)
            println("  базис ${sys.name}:")
            for (i in NS.indices) println("   n=%4d h=%s E_h=%s p_h=%s C_h=%s".format(
                NS[i], Fmt.h(hs[i]), Fmt.e(errs[i]), Fmt.p(ps[i]), Fmt.e(errs[i] / Math.pow(hs[i], 3.0))))
        }
    }

    /** T2[p]: сравнение база/Слоан/Кулкарни/итер.Кулкарни (theta). */
    fun tableMethods(p: ModelProblem, sys: GeneratingSystem) {
        println("\n--- T2[${p.name}]: базис ${sys.name}, theta: база/Слоан/Кулкарни/итер.Кулкарни (E_h,p_h) ---")
        val names = listOf("база", "Слоан", "Кулк", "ит.Кулк")
        val errs = names.map { ArrayList<Double>() }
        for (nn in NS) {
            val (s, grid) = makeSolver(p, sys, "theta", nn)
            val ex = { t: Double -> p.exact(t) }
            errs[0].add(errorEh(ex, s.base().eval, grid))
            errs[1].add(errorEh(ex, s.sloan().eval, grid))
            errs[2].add(errorEh(ex, s.kulkarni().eval, grid))
            errs[3].add(errorEh(ex, s.iteratedKulkarni().eval, grid))
        }
        val ps = names.indices.map { orders(errs[it]) }
        for (i in NS.indices) println("   n=%4d | ".format(NS[i]) +
            names.indices.joinToString(" | ") { mi -> "%s:%s(%s)".format(names[mi], Fmt.e(errs[mi][i]), Fmt.p(ps[mi][i])) })
    }

    /** Сравнение семейств theta/xi/mu/lambda (базовая схема). */
    fun tableFamilies(p: ModelProblem, sys: GeneratingSystem) {
        println("\n--- Семейства[${p.name}]: базис ${sys.name}, базовая схема (E_h,p_h) ---")
        for (fam in listOf("theta", "xi", "mu", "lambda")) {
            val errs = ArrayList<Double>()
            for (nn in NS) { val (s, grid) = makeSolver(p, sys, fam, nn)
                errs.add(errorEh({ t -> p.exact(t) }, s.base().eval, grid)) }
            val ps = orders(errs)
            println("  %-7s: ".format(fam) + NS.indices.joinToString(" ") { i -> "%s(%s)".format(Fmt.e(errs[i]), Fmt.p(ps[i])) })
        }
    }

    /** V1 (сведение дифференцированием, m=1): база/Слоан/Кулкарни, базис B. */
    fun tableF1() {
        println("\n--- V1 (I рода, сведение дифференцированием m=1), базис B, theta: база/Слоан/Кулк ---")
        val p = ModelProblem.V1
        for (nn in listOf(8, 16, 32, 64)) {
            val grid = Grid.uniform(nn)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val op = VolterraOperator(p.kernel, grid, quad)
            val solver = FirstKindSolver(p, basis, ProjFunctionals(basis), op)
            val ehB = errorEh({ t -> p.exact(t) }, solver.base().eval, grid)
            val ehS = errorEh({ t -> p.exact(t) }, solver.sloan().eval, grid)
            val ehK = errorEh({ t -> p.exact(t) }, solver.kulkarni().eval, grid)
            println("   n=%4d E_h(база)=%s E_h(Слоан)=%s E_h(Кулк)=%s".format(nn, Fmt.e(ehB), Fmt.e(ehS), Fmt.e(ehK)))
        }
        println("   [V1 корректна (r3); K(t,t)=1≠0 ⇒ m=1; g=f'/K(t,t), ядро W=-K_t/K(t,t)]")
    }
}

fun main() {
    println("=".repeat(72))
    println("VolterraSolver — линейные уравнения Вольтерра (new-01)")
    println("=".repeat(72))
    println("HEALTH-CHECKS:")
    var anyFail = false
    for (r in HealthChecks.runAll()) {
        val verdict = if (r.ok) "OK  " else "FAIL"
        if (!r.ok && r.critical) anyFail = true
        println("  [$verdict] ${r.name.padEnd(38)} measured=${"%.3e".format(r.measured)} thr=${"%.0e".format(r.threshold)}")
    }
    if (anyFail) { println("\nКРИТИЧЕСКИЙ ПРОВАЛ health-check. Расчёт остановлен."); exitProcess(1) }
    println("\nВсе критические health-checks пройдены.\n")

    println("=".repeat(72)); println("ТАБЛИЦЫ"); println("=".repeat(72))
    // Численные примеры II рода (каждый — полный набор сравнений):
    //   V2   : K=1/(1+t+s), u*=1/(t+1) (рациональный, K(t,t)!=0);
    //   V2exp: K=e^{-(t-s)^2}, u*=e^t (K(t,t)!=0);
    //   V2win: K=t-s, u*=cos t (K(t,t)=0) — одношаговый Кулкарни СТРОГО точнее Слоана.
    val secondKindExamples = listOf(
        ModelProblem.V2 to GeneratingSystem.B,
        ModelProblem.V2exp to GeneratingSystem.B,
        ModelProblem.V2win to GeneratingSystem.B,
    )
    for ((p, sys) in secondKindExamples) {
        Tables.tablePhi(p)
        Tables.tableMethods(p, sys)
        Tables.tableFamilies(p, sys)
    }
    // Один пример I рода (сведение дифференцированием).
    Tables.tableF1()
    println("\nРасчёт завершён.")
}

