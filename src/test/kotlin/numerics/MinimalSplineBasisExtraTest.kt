package numerics

import org.junit.jupiter.api.Tag
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * Дополнительные тесты базиса минимальных сплайнов: interval/activeOmega,
 * вычисление сплайна и его производной, согласие omegaDeriv с численной,
 * а также вырожденность узлов на правом тройном крае.
 */
@Tag("fast")
class MinimalSplineBasisExtraTest {
    private val grid = Grid.uniform(8)
    private val b = MinimalSplineBasis(GeneratingSystem.B, grid)

    /** interval(t): возвращает индекс k с x_k<=t<x_{k+1}, для t=b — n-1. */
    @Test fun intervalIndexing() {
        assertEquals(0, b.interval(grid.x(0) + 1e-6))
        assertEquals(grid.n - 1, b.interval(grid.b))
        val k = b.interval(0.5 * (grid.x(3) + grid.x(4)))
        assertEquals(3, k)
    }

    /** activeOmega: три активных сплайна суммируются в 1 (разбиение единицы для B). */
    @Test fun activeOmegaPartitionOfUnity() {
        val t = 0.43
        val k = b.interval(t)
        val w = b.activeOmega(k, t)
        assertEquals(1.0, w[0] + w[1] + w[2], 1e-10)
    }

    /** evalSpline воспроизводит постоянную 1 (sum c_j omega_j с c_j=1 = разбиение единицы). */
    @Test fun evalSplineReproducesConstant() {
        val c = DoubleArray(grid.n + 2) { 1.0 }
        for (t in listOf(0.1, 0.37, 0.62, 0.88)) {
            assertEquals(1.0, b.evalSpline(c, t), 1e-9, "const not reproduced at $t")
        }
    }

    /** evalSpline воспроизводит rho(t)=t для B: коэффициенты = значения rho в узлах де Бура. */
    @Test fun evalSplineReproducesLinear() {
        // Для phi^B=(1,t,t^2) сплайн воспроизводит линейную функцию точно.
        // Возьмём проектор theta, спроецируем f(t)=t и проверим воспроизведение.
        val theta = numerics.functionals.ProjFunctionals(b)
        val c = theta.projectorCoeffs({ t -> t })
        for (t in listOf(0.15, 0.5, 0.81)) {
            assertEquals(t, b.evalSpline(c, t), 1e-8, "linear not reproduced at $t")
        }
    }

    /** evalSplineDeriv: производная линейной функции t равна 1. */
    @Test fun evalSplineDerivOfLinear() {
        val theta = numerics.functionals.ProjFunctionals(b)
        val c = theta.projectorCoeffs({ t -> t })
        for (t in listOf(0.2, 0.55, 0.77)) {
            assertEquals(1.0, b.evalSplineDeriv(c, t), 1e-7, "deriv != 1 at $t")
        }
    }

    /** omegaDeriv: согласуется с центральной численной производной omega. */
    @Test fun omegaDerivMatchesNumeric() {
        val j = 2
        val eps = 1e-6
        for (t in listOf(grid.x(j) + 0.03, 0.5 * (grid.x(j + 1) + grid.x(j + 2)), grid.x(j + 3) - 0.03)) {
            val num = (b.omega(j, t + eps) - b.omega(j, t - eps)) / (2 * eps)
            assertEquals(num, b.omegaDeriv(j, t), 1e-4, "omegaDeriv mismatch at $t")
        }
    }

    /** omega и omegaDeriv равны нулю вне носителя [x_j, x_{j+3}]. */
    @Test fun omegaZeroOutsideSupport() {
        val j = 3
        assertEquals(0.0, b.omega(j, grid.x(j) - 0.01), 1e-12)
        assertEquals(0.0, b.omega(j, grid.x(j + 3) + 0.01), 1e-12)
        assertEquals(0.0, b.omegaDeriv(j, grid.x(j) - 0.01), 1e-12)
        assertEquals(0.0, b.omegaDeriv(j, grid.x(j + 3) + 0.01), 1e-12)
    }

    /** omega возвращает 0 при slot вне [0,2] (точка в носителе, но иной активный интервал). */
    @Test fun omegaZeroWhenSlotOutOfRange() {
        // omega_{-2} имеет носитель [x_{-2},x_1]=[a, x_1]; в интервале k=0 slot=j-(k-2)= -2+2=0 ок,
        // но для j с носителем правее текущего интервала slot<0 -> 0.
        val t = 0.5 * (grid.x(0) + grid.x(1)) // интервал k=0, активны j=-2,-1,0
        assertEquals(0.0, b.omega(1, t), 1e-12) // j=1 в носителе? нет: x_1>t -> уже 0 по носителю
        // явная проверка slot>2: возьмём j значительно левее
        val t2 = 0.5 * (grid.x(5) + grid.x(6)) // интервал k=5, активны 3,4,5
        assertTrue(abs(b.omega(2, t2)) < 1e-12) // slot=2-(5-2)=-1 <0
    }

    /** nonDegenerate=false на правом тройном крае: первая часть true, вторая false. */
    @Test fun nonDegenerateFalseRightEdge() {
        // j=n-1: x(n-1)<x(n) истинно, но x(n)==x(n+1) -> второе сравнение ложно
        assertTrue(!nonDegenerate(grid, grid.n - 1))
    }

    /**
     * Регресс #4: бинарный поиск интервала даёт тот же результат, что наивный
     * линейный, во ВСЕХ ТОЧКАХ ОТРЕЗКА (внутри интервалов, на внутренних узлах,
     * левый/правый концы).
     *
     * Точки ВНЕ отрезка из этого набора УБРАНЫ сознательно: эталонный линейный
     * поиск воспроизводит СТАРОЕ поведение — кламп — а кламп больше НЕ является
     * контрактом. Новый контракт на тех же точках проверяет [outsideSegmentIsRejected]:
     * покрытие не потеряно, перенесено туда, где утверждение верно.
     */
    @Test fun intervalBinarySearchMatchesLinear() {
        val g = Grid.uniform(5)
        val basis = MinimalSplineBasis(GeneratingSystem.B, g)
        // Эталонный линейный поиск (совпадает с прежней реализацией intervalOf).
        fun linear(t: Double): Int {
            var k = 0
            while (k < g.n - 1 && t >= g.x(k + 1)) k++
            return k
        }
        val pts = mutableListOf<Double>()
        pts.add(g.a)                     // левый конец
        pts.add(g.b)                     // правый конец
        for (i in 0..g.n) {
            pts.add(g.x(i))              // ровно на узле (в т.ч. внутренние)
            if (i < g.n) pts.add(0.5 * (g.x(i) + g.x(i + 1))) // середина интервала
        }
        for (t in pts) {
            assertEquals(linear(t), basis.interval(t), "interval mismatch at t=$t")
        }
    }

    /**
     * Точка ВНЕ отрезка ОТВЕРГАЕТСЯ, а не клампуется молча (пункт 8.3).
     *
     * До правки `intervalOf` возвращал первый интервал при `t < a` и последний при
     * `t > b`, а [MinimalSplineBasis.evalSpline] честно считал многочлен крайнего слоя в
     * точке, где сплайн не определён, и возвращал правдоподобное число — то есть
     * молча ЭКСТРАПОЛИРОВАЛ. Проверяются ВСЕ четыре публичные точки входа, а не
     * одна: проверка стоит в общем `intervalOf`, и тест фиксирует, что через него
     * действительно проходят все четыре (иначе одна из них могла бы тихо уйти
     * в обход при будущей правке).
     */
    @Test fun outsideSegmentIsRejected() {
        val g = Grid.uniform(5)
        val basis = MinimalSplineBasis(GeneratingSystem.B, g)
        val c = DoubleArray(g.n + 2) { 1.0 }
        val outside = listOf(
            g.a - 0.1 to "левее a на 0.1",
            g.b + 0.1 to "правее b на 0.1",
            Math.nextAfter(g.a, Double.NEGATIVE_INFINITY) to "nextDown(a) — один ulp левее a",
            Math.nextAfter(g.b, Double.POSITIVE_INFINITY) to "nextUp(b) — один ulp правее b",
        )
        val entryPoints: List<Pair<String, (Double) -> Any>> = listOf(
            "interval" to { t -> basis.interval(t) },
            "evalSpline" to { t -> basis.evalSpline(c, t) },
            "evalSplineDeriv" to { t -> basis.evalSplineDeriv(c, t) },
            "evalSplineDeriv2" to { t -> basis.evalSplineDeriv2(c, t) },
        )
        for ((t, where) in outside) {
            for ((name, call) in entryPoints) {
                val error = try {
                    val value = call(t)
                    fail(
                        "$name(t=$t) ($where, отрезок [${g.a}, ${g.b}]) обязан был бросить " +
                            "IllegalArgumentException, но тихо вернул $value — то есть молчаливая " +
                            "экстраполяция вернулась",
                    )
                } catch (e: IllegalArgumentException) {
                    e
                }
                val message = error.message ?: ""
                // Сообщение обязано быть ДИАГНОСТИЧНЫМ: называть точку, обе границы
                // и объяснять, что вне отрезка сплайн не определён, — иначе исключение
                // лишь меняет один непонятный исход на другой.
                for (fragment in listOf("MinimalSplineBasis", "$t", "${g.a}", "${g.b}", "экстрапол")) {
                    assertTrue(
                        message.contains(fragment),
                        "Сообщение об ошибке $name невнятное: нет фрагмента \"$fragment\". " +
                            "Получено: \"$message\"",
                    )
                }
            }
        }
    }

    /**
     * `t = NaN` НЕ бросает: проверка принадлежности записана отрицаниями.
     *
     * Это КРИТЕРИЙ ПРАВИЛЬНОСТИ ФОРМУЛИРОВКИ условия, а не прихоть. Проект
     * сознательно пропускает NaN насквозь (см. `VolterraOperator.apply` и
     * `VolterraIntegrandCacheEquivalenceTest.nanArgument_bothPathsAgree`): плохой вход
     * обязан проявиться как NaN, а не как правдоподобное число. Запись проверки
     * как `require(t in a..b)` превратила бы NaN в исключение и сломала это соглашение.
     */
    @Test fun nanArgumentPassesThroughWithoutThrowing() {
        val g = Grid.uniform(5)
        val c = DoubleArray(g.n + 2) { 1.0 }
        for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
            val basis = MinimalSplineBasis(system, g)
            // ГЛАВНОЕ УТВЕРЖДЕНИЕ: ни одна из точек входа НЕ бросает на NaN.
            // Именно это сломала бы запись `require(t in a..b)`.
            val k = basis.interval(Double.NaN)
            assertTrue(k in 0 until g.n, "${system.name}: interval(NaN) вернул $k вне [0, ${g.n - 1}]")
            val value = basis.evalSpline(c, Double.NaN)
            val deriv = basis.evalSplineDeriv(c, Double.NaN)
            val deriv2 = basis.evalSplineDeriv2(c, Double.NaN)

            // ГДЕ phi ЗАВИСИТ ОТ t, NaN обязан РАСПРОСТРАНИТЬСЯ в результат.
            // Значение и первая производная таковы у всех трёх систем.
            assertTrue(value.isNaN(), "${system.name}: evalSpline(NaN) обязан дать NaN, получено $value")
            assertTrue(deriv.isNaN(), "${system.name}: evalSplineDeriv(NaN) обязан дать NaN, получено $deriv")

            // Вторая производная — ОСОБЫЙ СЛУЧАЙ и НЕ дефект. Для полиномиальной
            // phi^B = (1, t, t^2) вторая производная phi'' = (0, 0, 2) КОНСТАНТНА: аргумент
            // в вычисление не входит вовсе, поэтому результат конечен при ЛЮБОМ t,
            // включая NaN. У H (sinh/cosh) и T (sin/cos) phi'' от t зависит, и NaN проходит.
            if (system.name == "B") {
                assertTrue(
                    deriv2.isFinite(),
                    "B: phi''=(0,0,2) константна, поэтому evalSplineDeriv2(NaN) обязан быть " +
                        "конечным (аргумент не участвует в вычислении), получено $deriv2",
                )
            } else {
                assertTrue(
                    deriv2.isNaN(),
                    "${system.name}: phi'' зависит от t, поэтому evalSplineDeriv2(NaN) обязан дать " +
                        "NaN, получено $deriv2",
                )
            }
        }
    }

    /**
     * Семейство `omega*` на точках вне отрезка НЕ бросает, а возвращает 0.
     *
     * Обоснование размещения проверки в `intervalOf`: семейство `omega*` туда с внешней
     * точкой НЕ ПОПАДАЕТ — его предваряет отсечка по носителю, а носитель любого
     * базисного сплайна лежит внутри отрезка. Проверяются именно КРАЙНИЕ j, где
     * носитель упирается в границу (j = -2 в `a`, j = n-1 в `b`), — если бы рассуждение
     * было неверно, сломалось бы именно здесь. Это также защищает штатный путь
     * `VolterraSolver.matrixM`, где `omega_j` интегрируется квадратурой.
     */
    @Test fun omegaFamilyOutsideSegmentReturnsZeroWithoutThrowing() {
        val g = Grid.uniform(5)
        val basis = MinimalSplineBasis(GeneratingSystem.B, g)
        val outside = listOf(g.a - 0.1, g.b + 0.1, g.a - 1e-12, g.b + 1e-12)
        for (j in -2 until g.n) {
            for (t in outside) {
                assertEquals(0.0, basis.omega(j, t), 0.0, "omega($j, $t) вне отрезка обязана быть 0")
                assertEquals(0.0, basis.omegaDeriv(j, t), 0.0, "omegaDeriv($j, $t) вне отрезка обязана быть 0")
                assertEquals(0.0, basis.omegaDeriv2(j, t), 0.0, "omegaDeriv2($j, $t) вне отрезка обязана быть 0")
            }
        }
    }
}
