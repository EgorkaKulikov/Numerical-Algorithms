package numerics

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Тесты геометрической (неравномерной) сетки Grid.geometric (семейство sn-article):
 * концы a,b точны и узлы на концах тройные; строгая монотонность; h = max шаг;
 * отношение крайних шагов h_{n-1}/h_0 ~ R; локальная квазиравномерность
 * h_j/h_{j-1} в [mu^{-1}, mu] с mu=q (для этой сетки ratio постоянно и равно q).
 */
class GridGeometricTest {
    private val tol = 1e-12

    @Test fun endpointsExactAndTripleKnots() {
        val g = Grid.geometric(16, 0.0, 1.0, R = 2.0)
        assertEquals(0.0, g.a, tol)
        assertEquals(1.0, g.b, tol)
        assertEquals(0.0, g.x(0), tol)
        assertEquals(1.0, g.x(g.n), tol)
        assertEquals(g.a, g.x(-2), tol); assertEquals(g.a, g.x(-1), tol)
        assertEquals(g.b, g.x(g.n + 1), tol); assertEquals(g.b, g.x(g.n + 2), tol)
    }

    @Test fun monotoneIncreasing() {
        val g = Grid.geometric(20, 2.0, 5.0, R = 3.0)
        for (i in 0 until g.n) assertTrue(g.x(i + 1) > g.x(i), "node $i not increasing")
    }

    @Test fun hIsMaxStep() {
        val n = 16
        val g = Grid.geometric(n, 0.0, 1.0, R = 2.0)
        var maxStep = 0.0
        for (i in 0 until n) maxStep = maxOf(maxStep, g.x(i + 1) - g.x(i))
        assertEquals(maxStep, g.h, tol)
        // при q>1 максимальный шаг — последний.
        assertEquals(g.x(n) - g.x(n - 1), g.h, tol)
    }

    @Test fun extremeStepRatioApproxR() {
        val n = 24; val R = 2.0
        val g = Grid.geometric(n, 0.0, 1.0, R)
        val h0 = g.x(1) - g.x(0)
        val hLast = g.x(n) - g.x(n - 1)
        assertEquals(R, hLast / h0, 1e-9)
    }

    @Test fun localQuasiUniformityRatioEqualsQ() {
        val n = 24; val R = 2.0
        val g = Grid.geometric(n, 0.0, 1.0, R)
        val mu = Math.pow(R, 1.0 / (n - 1)) // = q
        for (i in 1 until n) {
            val ratio = (g.x(i + 1) - g.x(i)) / (g.x(i) - g.x(i - 1))
            assertTrue(ratio >= 1.0 / mu - 1e-9 && ratio <= mu + 1e-9,
                "ratio $ratio out of [mu^-1, mu] at $i")
            assertEquals(mu, ratio, 1e-9)
        }
    }

    @Test fun requiresNAtLeastTwo() {
        assertFailsWith<IllegalArgumentException> { Grid.geometric(1) }
    }
}
