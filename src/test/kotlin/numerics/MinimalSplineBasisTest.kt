package numerics

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MinimalSplineBasisTest {
    private fun basis(sys: GeneratingSystem, n: Int = 8) =
        MinimalSplineBasis(sys, Grid.uniform(n))

    @Test fun partitionOfUnity() {
        val b = basis(GeneratingSystem.B)
        val n = b.n
        for (t in listOf(0.05, 0.2, 0.5, 0.73, 0.95)) {
            var s = 0.0
            for (j in -2 until n) s += b.omega(j, t)
            assertEquals(1.0, s, 1e-9, "sum omega != 1 at t=$t")
        }
    }

    @Test fun matchesReferenceB() {
        val grid = Grid.uniform(8)
        val b = MinimalSplineBasis(GeneratingSystem.B, grid)
        for (j in 0 until b.n - 2) {
            for (t in listOf(grid.x(j) + 1e-3, (grid.x(j) + grid.x(j + 3)) / 2.0, grid.x(j + 3) - 1e-3)) {
                assertEquals(ReferenceSplines.omegaB(grid, j, t), b.omega(j, t), 1e-9,
                    "omegaB mismatch j=$j t=$t")
            }
        }
    }

    @Test fun matchesReferenceH() {
        val grid = Grid.uniform(8)
        val b = MinimalSplineBasis(GeneratingSystem.H, grid)
        for (j in 0 until b.n - 2) {
            val tm = (grid.x(j) + grid.x(j + 3)) / 2.0
            assertEquals(ReferenceSplines.omegaH(grid, j, tm), b.omega(j, tm), 1e-8,
                "omegaH mismatch j=$j")
        }
    }

    @Test fun zeroSupportOutside() {
        val grid = Grid.uniform(8)
        val b = MinimalSplineBasis(GeneratingSystem.B, grid)
        val j = 2
        // support is [x_j, x_{j+3}]; sample a point well left of x_j
        val left = grid.x(j) - 0.5 * (grid.x(j + 1) - grid.x(j))
        if (left > grid.a) {
            assertTrue(abs(b.omega(j, left)) < 1e-9)
        }
        val right = grid.x(j + 3) + 0.5 * grid.h
        if (right < grid.b) {
            assertTrue(abs(b.omega(j, right)) < 1e-9)
        }
    }
}
