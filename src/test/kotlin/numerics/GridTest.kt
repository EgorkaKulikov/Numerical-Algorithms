package numerics

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class GridTest {
    @Test fun tripleKnotsAtEnds() {
        val g = Grid.uniform(8, 0.0, 1.0)
        assertEquals(g.a, g.x(-2), 1e-15)
        assertEquals(g.a, g.x(-1), 1e-15)
        assertEquals(g.a, g.x(0), 1e-15)
        assertEquals(g.b, g.x(g.n), 1e-15)
        assertEquals(g.b, g.x(g.n + 1), 1e-15)
        assertEquals(g.b, g.x(g.n + 2), 1e-15)
    }

    @Test fun uniformStep() {
        val n = 8
        val g = Grid.uniform(n, 0.0, 1.0)
        assertEquals(1.0 / n, g.h, 1e-15)
        for (i in 0 until n) assertEquals(1.0 / n, g.x(i + 1) - g.x(i), 1e-12)
    }

    @Test fun nonDegenerateInterior() {
        val g = Grid.uniform(8)
        assertTrue(nonDegenerate(g, 0))
        assertTrue(nonDegenerate(g, g.n - 3))
    }
}
