package numerics.functionals

import kotlin.math.abs

import numerics.*

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
    /**
     * chi_j(f) по функции f, её первой производной fD и второй производной fDD.
     * Семейства без производных (theta,mu,lambda) игнорируют fD/fDD; xi^<1>,xi^<2>
     * используют только fD; xi^<0> использует и fDD. fDD по умолчанию нулевая, чтобы
     * не ломать существующие двухаргументные вызовы (theta/mu/lambda и xi^<1>).
     */
    fun apply(f: (Double) -> Double, fD: (Double) -> Double, fDD: (Double) -> Double = { 0.0 }): Double

    /** Сумма |коэффициентов| — для оценки нормы C_chi. */
    fun absSum(): Double
}

/** Удобная обёртка: chi_j(f) без явных производных (производные = 0). */
fun ApproxFunctional.apply(f: (Double) -> Double): Double = apply(f, { 0.0 }, { 0.0 })

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

    /** true для семейств, требующих ВТОРУЮ производную образа (xi^<0>). */
    open val usesSecondDerivative: Boolean = false

    /** Функционал chi_j, j = -2..n-1. */
    abstract fun chi(j: Int): ApproxFunctional

    /** Коэффициенты проекции P_chi g = sum chi_j(g) omega_j: вектор (chi_j(g)) размера n+2. */
    fun projectorCoeffs(
        g: (Double) -> Double,
        gD: (Double) -> Double = { 0.0 },
        gDD: (Double) -> Double = { 0.0 },
    ): DoubleArray = DoubleArray(n + 2) { chi(it - 2).apply(g, gD, gDD) }

    /** Максимум sum_k |коэффициенты| по всем j — константа C_chi. */
    fun cChi(): Double = (-2..n - 1).maxOf { chi(it).absSum() }
}

// ----------------------------------------------------------------------------
// 6.1 theta — ПРОЕКЦИОННЫЕ ФУНКЦИОНАЛЫ (prfunc-formulas.md)
// ----------------------------------------------------------------------------

/** Линейная комбинация значений f в точках nodes с коэффициентами coeffs (только значения). */
class ValueFunctional(val nodes: DoubleArray, val coeffs: DoubleArray) : ApproxFunctional {
    init { require(nodes.size == coeffs.size) }
    override fun apply(f: (Double) -> Double, fD: (Double) -> Double, fDD: (Double) -> Double): Double {
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
    override fun apply(f: (Double) -> Double, fD: (Double) -> Double, fDD: (Double) -> Double): Double =
        f(node) + cD * fD(node)
    override fun absSum(): Double = 1.0 + abs(cD)
}

/**
 * Функционал вида xi^<0>(u) = u(node) + c1*u'(node) + c2*u''(node) (де Бура--Фикса r=0):
 * использует значение, ПЕРВУЮ и ВТОРУЮ производные в одном узле.
 */
class SecondDerivFunctional(val node: Double, val c1: Double, val c2: Double) : ApproxFunctional {
    override fun apply(f: (Double) -> Double, fD: (Double) -> Double, fDD: (Double) -> Double): Double =
        f(node) + c1 * fD(node) + c2 * fDD(node)
    override fun absSum(): Double = 1.0 + abs(c1) + abs(c2)
}

/**
 * Семейство функционалов де Бура--Фикса xi_j^{<r>}, r in {0,1,2}
 * (deboorfix-spec.md, метки eq:xi0, eq:xi0-coef, eq:xi). Выбор r — параметр
 * конструктора (r=1 по умолчанию — обратная совместимость). Все три — ПРОЕКТОРЫ
 * (биортогональность xi_i(omega_j)=delta_ij).
 *
 *  - xi^<1>(u) = u(x_{j+1}) + C1_j u'(x_{j+1}),
 *      C1_j = ((sigma_{j+2}-sigma_{j+1})rho'_{j+2} - (rho_{j+2}-rho_{j+1})sigma'_{j+2}) / W_j;
 *  - xi^<2>(u) = u(x_{j+2}) + C2_j u'(x_{j+2}),
 *      C2_j = ((sigma_{j+2}-sigma_{j+1})rho'_{j+1} - (rho_{j+2}-rho_{j+1})sigma'_{j+1}) / W_j;
 *      W_j = rho'_{j+2}sigma'_{j+1} - rho'_{j+1}sigma'_{j+2};
 *  - xi^<0>(u) = u(x_j) + (N1_j/Delta_j) u'(x_j) + (N2_j/Delta_j) u''(x_j)
 *      (работает в C^2), коэффициенты — eq:xi0-coef.
 *
 * Краевые j=-2, n-1: чистые значения u(x_0), u(x_n) (ДОПУЩЕНИЕ по аналогии с theta,
 * deboorfix-spec.md §1.3, R1 — подтверждается тестом биортогональности).
 */
class DeBoorFixFunctionals(basis: MinimalSplineBasis, val r: Int = 1) :
    FunctionalFamily(basis, if (r == 1) "xi" else "xi<$r>") {
    init { require(r in 0..2) { "DeBoorFix: r must be in {0,1,2}, got $r" } }
    override val isProjector = true
    override val usesDerivative = true
    override val usesSecondDerivative = (r == 0)
    private val sys = basis.sys
    private val funcs: Array<ApproxFunctional> = Array(n + 2) { buildXi(it - 2) }
    override fun chi(j: Int): ApproxFunctional = funcs[j + 2]

    private fun buildXi(j: Int): ApproxFunctional {
        if (j == -2) return DerivFunctional(grid.x(0), 0.0)
        if (j == n - 1) return DerivFunctional(grid.x(n), 0.0)
        return when (r) {
            0 -> buildXi0(j)
            2 -> buildXi2(j)
            else -> buildXi1(j)
        }
    }

    /** xi^<1>_j: узел x_{j+1}, коэффициент C1_j. */
    private fun buildXi1(j: Int): ApproxFunctional {
        val x1 = grid.x(j + 1); val x2 = grid.x(j + 2)
        val rho1 = sys.rho(x1); val rho2 = sys.rho(x2)
        val sig1 = sys.sigma(x1); val sig2 = sys.sigma(x2)
        val rhoD1 = sys.rhoD(x1); val rhoD2 = sys.rhoD(x2)
        val sigD1 = sys.sigmaD(x1); val sigD2 = sys.sigmaD(x2)
        val denom = rhoD2 * sigD1 - rhoD1 * sigD2
        require(kotlin.math.abs(denom) >= 1e-14) {
            "buildXi1(j=$j): degenerate Wronskian rhoD2*sigD1 - rhoD1*sigD2=$denom (near-zero denominator)"
        }
        val cD = ((sig2 - sig1) * rhoD2 - (rho2 - rho1) * sigD2) / denom
        return DerivFunctional(x1, cD)
    }

    /** xi^<2>_j: узел x_{j+2}, коэффициент C2_j (тот же знаменатель W_j, штрихи в x_{j+1}). */
    private fun buildXi2(j: Int): ApproxFunctional {
        val x1 = grid.x(j + 1); val x2 = grid.x(j + 2)
        val rho1 = sys.rho(x1); val rho2 = sys.rho(x2)
        val sig1 = sys.sigma(x1); val sig2 = sys.sigma(x2)
        val rhoD1 = sys.rhoD(x1); val rhoD2 = sys.rhoD(x2)
        val sigD1 = sys.sigmaD(x1); val sigD2 = sys.sigmaD(x2)
        val denom = rhoD2 * sigD1 - rhoD1 * sigD2
        require(kotlin.math.abs(denom) >= 1e-14) {
            "buildXi2(j=$j): degenerate Wronskian rhoD2*sigD1 - rhoD1*sigD2=$denom (near-zero denominator)"
        }
        val cD = ((sig2 - sig1) * rhoD1 - (rho2 - rho1) * sigD1) / denom
        return DerivFunctional(x2, cD)
    }

    /**
     * xi^<0>_j: узел x_j, коэффициенты N1_j/Delta_j (при u') и N2_j/Delta_j (при u'')
     * по eq:xi0-coef (перенесено дословно).
     */
    private fun buildXi0(j: Int): ApproxFunctional {
        val xj = grid.x(j); val xj1 = grid.x(j + 1); val xj2 = grid.x(j + 2)
        // Значения rho, sigma и производные в трёх узлах.
        val rj = sys.rho(xj); val sj = sys.sigma(xj)
        val rDj = sys.rhoD(xj); val sDj = sys.sigmaD(xj)
        val rDDj = sys.rhoDD(xj); val sDDj = sys.sigmaDD(xj)
        val rj1 = sys.rho(xj1); val sj1 = sys.sigma(xj1)
        val rDj1 = sys.rhoD(xj1); val sDj1 = sys.sigmaD(xj1)
        val rj2 = sys.rho(xj2); val sj2 = sys.sigma(xj2)
        val rDj2 = sys.rhoD(xj2); val sDj2 = sys.sigmaD(xj2)

        val delta = (rDj1 * sDj2 - rDj2 * sDj1) * (rDj * sDDj - rDDj * sDj)
        require(kotlin.math.abs(delta) >= 1e-14) {
            "buildXi0(j=$j): degenerate denominator Delta_j=$delta (near-zero)"
        }
        val n1 = (rj1 * sDj1 - rDj1 * sj1) * (rDDj * sDj2 - rDj2 * sDDj) +
            (rDj1 * sDj2 - rDj2 * sDj1) * (rDDj * sj - rj * sDDj) +
            (rj2 * sDj2 - rDj2 * sj2) * (rDj1 * sDDj - rDDj * sDj1)
        val n2 = (rj1 * sDj1 - rDj1 * sj1) * (rDj2 * sDj - rDj * sDj2) +
            (rDj1 * sDj2 - rDj2 * sDj1) * (rj * sDj - rDj * sj) +
            (rj2 * sDj2 - rDj2 * sj2) * (rDj * sDj1 - rDj1 * sDj)
        return SecondDerivFunctional(xj, n1 / delta, n2 / delta)
    }
}

// ----------------------------------------------------------------------------
// 6.2b xitilde — ДИСКРЕТИЗОВАННЫЕ (differentiation-free) ФУНКЦИОНАЛЫ де Бура--Фикса
// ----------------------------------------------------------------------------

/**
 * Дискретизованные (differentiation-free) функционалы де Бура--Фикса
 * xitilde^{<r>}_j, r in {1,2} (main.tex §3.4, eq:fd и две формулы \tilde\xi;
 * дизайн `.tasks/nystrom-diff-free/design-spec.md`). Производная f'(x_k) в исходных
 * xi^{<1>},xi^{<2>} заменена ЦЕНТРАЛЬНОЙ разделённой разностью узловых значений
 *   f'(x_k) ~ (f(x_{k+1}) - f(x_{k-1})) / (x_{k+1} - x_{k-1}),
 * которая корректна и на неравномерных сетках. Итог:
 *   xitilde^{<1>}_j(f) = f(x_{j+1}) + w1_j (f(x_{j+2}) - f(x_j)) / (x_{j+2} - x_j),
 *   xitilde^{<2>}_j(f) = f(x_{j+2}) + w2_j (f(x_{j+3}) - f(x_{j+1})) / (x_{j+3} - x_{j+1}),
 * где множители w1_j = A^{<1>}_j, w2_j = A^{<2>}_j — В ТОЧНОСТИ коэффициенты при
 * производной из DeBoorFixFunctionals (buildXi1/buildXi2). Сплайновая алгебра НЕ
 * пересчитывается: множитель извлекается из DerivFunctional.cD исходного семейства.
 *
 * Функционалы VALUE-ONLY (usesDerivative=false) и представлены как ValueFunctional,
 * поэтому совместимы с узловой квадратурой Nyström (eq:nyval) без изменений в солверах.
 * Оператор P_xitilde — КВАЗИИНТЕРПОЛЯНТ, не проектор (isProjector=false): точная
 * биортогональность при FD-замене в общем случае теряется (design-spec §2, §7).
 * Краевые j=-2, n-1 — чистые значения f(x_0), f(x_n) (как в theta/xi).
 *
 * xitilde^{<0>} НЕ реализован: он требует f'' в узле, где сплайн вырождается
 * (design-spec §5.2, remark после eq:fd) — исключён по построению.
 */
class DiscreteDeBoorFixFunctionals(basis: MinimalSplineBasis, val r: Int = 1) :
    FunctionalFamily(basis, if (r == 1) "xitilde" else "xitilde<$r>") {
    init { require(r in 1..2) { "DiscreteDeBoorFix: r must be in {1,2}, got $r" } }
    override val isProjector = false
    override val usesDerivative = false
    private val raw = DeBoorFixFunctionals(basis, r)
    private val funcs: Array<ApproxFunctional> = Array(n + 2) { buildXiTilde(it - 2) }
    override fun chi(j: Int): ApproxFunctional = funcs[j + 2]

    private fun buildXiTilde(j: Int): ApproxFunctional {
        if (j == -2) return ValueFunctional(doubleArrayOf(grid.x(0)), doubleArrayOf(1.0))
        if (j == n - 1) return ValueFunctional(doubleArrayOf(grid.x(n)), doubleArrayOf(1.0))
        // Множитель при производной — тот же A^{<r>}_j, что и в исходном xi.
        val w = (raw.chi(j) as DerivFunctional).cD
        // Узел опоры и соседи для центральной разности вокруг него.
        val (node, left, right) = when (r) {
            2 -> Triple(grid.x(j + 2), grid.x(j + 1), grid.x(j + 3))
            else -> Triple(grid.x(j + 1), grid.x(j), grid.x(j + 2))
        }
        val denom = right - left
        require(kotlin.math.abs(denom) >= 1e-14) {
            "buildXiTilde(j=$j,r=$r): degenerate divided-difference span x=$right - x=$left = $denom"
        }
        // f(node) + w (f(right) - f(left))/denom как комбинация значений.
        return ValueFunctional(
            doubleArrayOf(left, node, right),
            doubleArrayOf(-w / denom, 1.0, w / denom),
        )
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
        val denom = dot3(dJ2, phiDJ1)
        require(kotlin.math.abs(denom) >= 1e-14) {
            "aN(j=$j): degenerate approximation relation, dot3(dJ2, phiDJ1)=$denom (near-zero denominator)"
        }
        val coef = dot3(dJ2, phiJ1) / denom
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
