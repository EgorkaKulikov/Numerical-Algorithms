package numerics

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Дополнительные тесты базиса минимальных сплайнов: interval/activeOmega,
 * вычисление сплайна и его производной, согласие omegaDeriv с численной,
 * а также вырожденность узлов на правом тройном крае.
 */
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
}
