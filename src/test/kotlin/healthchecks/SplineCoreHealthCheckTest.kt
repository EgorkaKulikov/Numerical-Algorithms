package healthchecks

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.ReferenceSplines
import numerics.functionals.AveragingFunctionals
import numerics.functionals.DeBoorFixFunctionals
import numerics.functionals.FunctionalFamily
import numerics.functionals.ProjFunctionals
import numerics.functionals.ThreePointFunctionals
import numerics.nonDegenerate
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * ОБЩИЕ health-checks вычислительного ядра: минимальные сплайны, порождающие
 * системы, аппроксимационные функционалы и квадратура.
 *
 * Эти проверки НЕ зависят от конкретного интегрального оператора (Фредгольма,
 * Вольтерры или Урысона) — они относятся к пакету `numerics`. Поэтому они собраны
 * здесь в одном месте: ранее тот же набор был продублирован в решателях Фредгольма
 * и Вольтерры почти дословно.
 *
 * Каждая проверка выполняется на двух сетках — равномерной и квазиравномерной, —
 * чтобы отловить ошибки, проявляющиеся только при неравных шагах.
 */
class SplineCoreHealthCheckTest {

    private companion object {
        /** Порог сравнения для тождеств, выполняющихся точно (с точностью до округления). */
        const val EXACT_IDENTITY_TOLERANCE = 1e-10

        /** Порог для величин, накапливающих ошибку решения малых линейных систем. */
        const val LINEAR_SOLVE_TOLERANCE = 1e-9

        /** Порог для величин, накапливающих ошибку проектирования и вычисления сплайна. */
        const val PROJECTION_TOLERANCE = 1e-8

        /** Число точек выборки внутри отрезка при поточечных сравнениях. */
        const val SAMPLE_COUNT = 200
    }

    private val quad = GaussLegendre(8)
    private val uniformGrid = Grid.uniform(8)
    private val quasiUniformGrid = Grid.quasiUniform(8)
    private val sampleFractions = (0..SAMPLE_COUNT).map { it.toDouble() / SAMPLE_COUNT }
    private val allSystems = listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)

    /** Возвращает наибольшее отклонение по обеим тестовым сеткам. */
    private fun worstOverGrids(action: (Grid) -> Double): Double =
        maxOf(action(uniformGrid), action(quasiUniformGrid))

    /** Точки выборки, равномерно покрывающие отрезок сетки. */
    private fun samplePoints(grid: Grid): List<Double> =
        sampleFractions.map { grid.a + (grid.b - grid.a) * it }

    /**
     * Общий базис минимальных сплайнов на полиномиальной порождающей системе должен
     * совпадать с классической явной формулой квадратичного B-сплайна — как по
     * значению, так и по первой производной.
     *
     * Это ключевая проверка: общая конструкция через обращение матрицы `M_k`
     * сверяется с независимо выписанной замкнутой формулой.
     */
    @Test
    fun polynomialSplineMatchesClosedFormB() {
        val deviation = worstOverGrids { grid ->
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            var worst = 0.0
            for (t in samplePoints(grid)) {
                for (j in -2..grid.n - 1) if (nonDegenerate(grid, j)) {
                    worst = maxOf(worst, abs(basis.omega(j, t) - ReferenceSplines.omegaB(grid, j, t)))
                    worst = maxOf(
                        worst,
                        abs(basis.omegaDeriv(j, t) - ReferenceSplines.omegaBDeriv(grid, j, t)),
                    )
                }
            }
            worst
        }
        assertTrue(
            deviation < EXACT_IDENTITY_TOLERANCE,
            "Базис B должен совпадать с явной формулой квадратичного B-сплайна, " +
                "наибольшее отклонение = $deviation",
        )
    }

    /**
     * То же для гиперболической порождающей системы: общий базис должен совпадать
     * с явной формулой гиперболического минимального сплайна.
     */
    @Test
    fun hyperbolicSplineMatchesClosedFormH() {
        val deviation = worstOverGrids { grid ->
            val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
            var worst = 0.0
            for (t in samplePoints(grid)) {
                for (j in -2..grid.n - 1) if (nonDegenerate(grid, j)) {
                    worst = maxOf(worst, abs(basis.omega(j, t) - ReferenceSplines.omegaH(grid, j, t)))
                }
            }
            worst
        }
        assertTrue(
            deviation < EXACT_IDENTITY_TOLERANCE,
            "Базис H должен совпадать с явной формулой гиперболического сплайна, " +
                "наибольшее отклонение = $deviation",
        )
    }

    /**
     * Разбиение единицы: сумма всех базисных сплайнов тождественно равна единице
     * в любой точке отрезка. Свойство обеспечивается тем, что первая компонента
     * порождающей вектор-функции равна константе.
     */
    @Test
    fun basisFormsPartitionOfUnity() {
        val deviation = worstOverGrids { grid ->
            var worst = 0.0
            for (system in allSystems) {
                val basis = MinimalSplineBasis(system, grid)
                for (t in samplePoints(grid)) {
                    var sum = 0.0
                    for (j in -2..grid.n - 1) sum += basis.omega(j, t)
                    worst = maxOf(worst, abs(sum - 1.0))
                }
            }
            worst
        }
        assertTrue(
            deviation < EXACT_IDENTITY_TOLERANCE,
            "Сумма базисных сплайнов должна равняться единице, наибольшее отклонение = $deviation",
        )
    }

    /**
     * Биортогональность проекционных функционалов: `theta_i(omega_j) = delta_ij`.
     * Именно это свойство делает оператор `P_theta` проектором.
     */
    @Test
    fun projectionFunctionalsAreBiorthogonal() {
        val deviation = worstOverGrids { grid ->
            var worst = 0.0
            for (system in allSystems) {
                val basis = MinimalSplineBasis(system, grid)
                val funcs = ProjFunctionals(basis)
                for (i in -2..grid.n - 1) for (j in -2..grid.n - 1) {
                    val value = funcs.chi(i).apply(
                        { t -> basis.omega(j, t) },
                        { t -> basis.omegaDeriv(j, t) },
                    )
                    worst = maxOf(worst, abs(value - if (i == j) 1.0 else 0.0))
                }
            }
            worst
        }
        assertTrue(
            deviation < LINEAR_SOLVE_TOLERANCE,
            "Функционалы theta должны быть биортогональны базису, наибольшее отклонение = $deviation",
        )
    }

    /**
     * Биортогональность функционалов де Бура–Фикса для всех трёх вариантов
     * `r = 0, 1, 2`, включая краевые индексы.
     *
     * Для краевых функционалов (`j = -2` и `j = n-1`) реализация использует чистые
     * значения в концах отрезка — это допущение, не выписанное в первоисточнике явно,
     * и настоящая проверка служит его обоснованием.
     */
    @Test
    fun deBoorFixFunctionalsAreBiorthogonalForAllOrders() {
        val deviation = worstOverGrids { grid ->
            var worst = 0.0
            for (system in allSystems) {
                val basis = MinimalSplineBasis(system, grid)
                for (r in 0..2) {
                    val funcs = DeBoorFixFunctionals(basis, r)
                    for (i in -2..grid.n - 1) for (j in -2..grid.n - 1) {
                        val value = funcs.chi(i).apply(
                            { t -> basis.omega(j, t) },
                            { t -> basis.omegaDeriv(j, t) },
                            { t -> basis.omegaDeriv2(j, t) },
                        )
                        worst = maxOf(worst, abs(value - if (i == j) 1.0 else 0.0))
                    }
                }
            }
            worst
        }
        assertTrue(
            deviation < LINEAR_SOLVE_TOLERANCE,
            "Функционалы xi<0>, xi<1>, xi<2> должны быть биортогональны базису, " +
                "наибольшее отклонение = $deviation",
        )
    }

    /**
     * Идемпотентность проекторов: если функция уже является сплайном, то её проекция
     * должна возвращать в точности те же коэффициенты (`P^2 = P`).
     *
     * Проверяется только для проекторов (theta и все варианты xi). Семейства
     * `mu` и `lambda` — квазиинтерполянты, идемпотентностью они не обязаны обладать.
     *
     * Коэффициенты берутся псевдослучайными с ФИКСИРОВАННЫМ зерном: тест обязан быть
     * воспроизводимым.
     */
    @Test
    fun projectorsAreIdempotentOnSplines() {
        val random = kotlin.random.Random(seed = 777)
        val deviation = worstOverGrids { grid ->
            var worst = 0.0
            for (system in allSystems) {
                val basis = MinimalSplineBasis(system, grid)
                val projectorFamilies = listOf(
                    ProjFunctionals(basis),
                    DeBoorFixFunctionals(basis, 0),
                    DeBoorFixFunctionals(basis, 1),
                    DeBoorFixFunctionals(basis, 2),
                )
                for (funcs in projectorFamilies) {
                    val coeffs = DoubleArray(grid.n + 2) { random.nextDouble(-1.0, 1.0) }
                    val spline = { t: Double -> basis.evalSpline(coeffs, t) }
                    val splineDeriv = { t: Double -> basis.evalSplineDeriv(coeffs, t) }
                    val splineDeriv2 = { t: Double -> basis.evalSplineDeriv2(coeffs, t) }
                    val projected = funcs.projectorCoeffs(spline, splineDeriv, splineDeriv2)
                    for (i in coeffs.indices) worst = maxOf(worst, abs(projected[i] - coeffs[i]))
                }
            }
            worst
        }
        assertTrue(
            deviation < PROJECTION_TOLERANCE,
            "Проекторы theta и xi должны быть идемпотентны на сплайнах, " +
                "наибольшее отклонение = $deviation",
        )
    }

    /**
     * Точность на порождающем пространстве: любое из четырёх семейств функционалов
     * должно воспроизводить функции из `span{1, rho, sigma}` без погрешности.
     *
     * Это минимальное требование к аппроксимационному оператору; его нарушение
     * означает ошибку в построении коэффициентов функционалов.
     */
    @Test
    fun allFamiliesAreExactOnGeneratingSpan() {
        val deviation = worstOverGrids { grid ->
            var worst = 0.0
            for (system in allSystems) {
                val basis = MinimalSplineBasis(system, grid)
                val families: List<FunctionalFamily> = listOf(
                    ProjFunctionals(basis),
                    DeBoorFixFunctionals(basis),
                    AveragingFunctionals(basis),
                    ThreePointFunctionals(basis),
                )
                for (funcs in families) {
                    val projected = funcs.projectorCoeffs({ t -> system.rho(t) }, { t -> system.rhoD(t) })
                    for (t in samplePoints(grid)) {
                        worst = maxOf(worst, abs(system.rho(t) - basis.evalSpline(projected, t)))
                    }
                }
            }
            worst
        }
        assertTrue(
            deviation < PROJECTION_TOLERANCE,
            "Все семейства функционалов должны быть точны на span{1, rho, sigma}, " +
                "наибольшее отклонение = $deviation",
        )
    }

    /**
     * Проверка замкнутой формулы проекционных функционалов: на равномерной сетке с
     * полиномиальной порождающей системой коэффициенты внутренних функционалов
     * должны равняться `{1/14, -2/7, 10/7, -2/7, 1/14}`.
     *
     * Числа взяты из первоисточника (см. `docs/REFERENCES.md`, раздел
     * «Аппроксимационные функционалы»), поэтому проверка сверяет реализацию
     * с опубликованной формулой, а не с самой собой.
     */
    @Test
    fun closedFormCoefficientsMatchPublishedValues() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val expected = doubleArrayOf(1.0 / 14.0, -2.0 / 7.0, 10.0 / 7.0, -2.0 / 7.0, 1.0 / 14.0)
        var deviation = 0.0
        for (j in 0..grid.n - 3) {
            val actual = funcs.closedFormInternal(j).coeffs
            for (k in actual.indices) deviation = maxOf(deviation, abs(actual[k] - expected[k]))
        }
        assertTrue(
            deviation < EXACT_IDENTITY_TOLERANCE,
            "Коэффициенты замкнутой формулы theta должны совпадать с опубликованными " +
                "{1/14, -2/7, 10/7, -2/7, 1/14}, наибольшее отклонение = $deviation",
        )
    }

    /**
     * Точность квадратуры Гаусса–Лежандра: формула с восемью узлами точна для
     * многочленов степени до `2*8 - 1 = 15` включительно, а на гладких неполиномиальных
     * функциях даёт погрешность на уровне машинной точности.
     */
    @Test
    fun gaussLegendreQuadratureIsAccurate() {
        val interval = doubleArrayOf(0.0, 1.0)
        var deviation = 0.0
        // Многочлены t^k, k = 0..15: точное значение интеграла равно 1/(k+1).
        for (k in 0..15) {
            val computed = quad.integrate(interval) { t -> Math.pow(t, k.toDouble()) }
            deviation = maxOf(deviation, abs(computed - 1.0 / (k + 1)))
        }
        deviation = maxOf(deviation, abs(quad.integrate(interval) { t -> Math.exp(t) } - (Math.E - 1.0)))
        deviation = maxOf(deviation, abs(quad.integrate(interval) { t -> 1.0 / (t + 1.0) } - Math.log(2.0)))
        assertTrue(
            deviation < EXACT_IDENTITY_TOLERANCE,
            "Квадратура должна быть точна на многочленах степени <= 15 и точна для гладких функций, " +
                "наибольшее отклонение = $deviation",
        )
    }
}
