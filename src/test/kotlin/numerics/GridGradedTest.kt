package numerics

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Тесты градуированной двухмасштабной сетки Grid.graded: концы a,b точны и тройные;
 * строгая монотонность; отношение соседних шагов h_j/h_{j-1} in {ratio, 1/ratio} при
 * ЛЮБОМ n (локальный параметр квазиравномерности = ratio, НЕ стремится к 1 при росте n,
 * в отличие от geometric); h = max шаг. Проверяются чётное и нечётное n.
 */
class GridGradedTest {
    private val tol = 1e-12

    @Test fun endpointsExactAndTripleKnots() {
        val g = Grid.graded(16, 0.0, 1.0, ratio = 2.0)
        assertEquals(0.0, g.a, tol)
        assertEquals(1.0, g.b, tol)
        assertEquals(0.0, g.x(0), tol)
        assertEquals(1.0, g.x(g.n), tol)
        assertEquals(g.a, g.x(-2), tol); assertEquals(g.a, g.x(-1), tol)
        assertEquals(g.b, g.x(g.n + 1), tol); assertEquals(g.b, g.x(g.n + 2), tol)
    }

    @Test fun monotoneIncreasing() {
        for (n in listOf(8, 9, 16, 17)) {
            val g = Grid.graded(n, 2.0, 5.0, ratio = 3.0)
            for (i in 0 until g.n) assertTrue(g.x(i + 1) > g.x(i), "node $i (n=$n) not increasing")
        }
    }

    @Test fun neighborRatioIsRatioOrInverse() {
        val ratio = 2.0
        for (n in listOf(8, 9, 16, 17, 64)) {
            val g = Grid.graded(n, 0.0, 1.0, ratio)
            for (i in 1 until g.n) {
                val r = (g.x(i + 1) - g.x(i)) / (g.x(i) - g.x(i - 1))
                val isRatio = kotlin.math.abs(r - ratio) < 1e-9
                val isInv = kotlin.math.abs(r - 1.0 / ratio) < 1e-9
                assertTrue(isRatio || isInv, "ratio $r at i=$i (n=$n) not in {ratio, 1/ratio}")
            }
        }
    }

    /** Локальный параметр квазиравномерности = ratio, НЕ зависит от n (не стремится к 1). */
    @Test fun quasiUniformityParamFixedIndependentOfN() {
        val ratio = 2.0
        fun maxNeighborRatio(n: Int): Double {
            val g = Grid.graded(n, 0.0, 1.0, ratio)
            var m = 0.0
            for (i in 1 until g.n) {
                val r = (g.x(i + 1) - g.x(i)) / (g.x(i) - g.x(i - 1))
                m = maxOf(m, r, 1.0 / r)
            }
            return m
        }
        val m8 = maxNeighborRatio(8)
        val m64 = maxNeighborRatio(64)
        assertEquals(ratio, m8, 1e-9)
        assertEquals(ratio, m64, 1e-9)
        // Явно: параметр НЕ падает к 1 при росте n (контраст с geometric mu_n=R^{1/(n-1)}).
        assertTrue(kotlin.math.abs(m64 - m8) < 1e-9, "quasi-uniformity param drifted with n")
    }

    @Test fun hIsMaxStep() {
        val ratio = 2.0
        for (n in listOf(8, 9, 16)) {
            val g = Grid.graded(n, 0.0, 1.0, ratio)
            var maxStep = 0.0
            for (i in 0 until n) maxStep = maxOf(maxStep, g.x(i + 1) - g.x(i))
            assertEquals(maxStep, g.h, tol)
        }
    }

    @Test fun requiresNAtLeastTwoAndPositiveRatio() {
        assertFailsWith<IllegalArgumentException> { Grid.graded(1) }
        assertFailsWith<IllegalArgumentException> { Grid.graded(8, ratio = 0.0) }
    }
}
