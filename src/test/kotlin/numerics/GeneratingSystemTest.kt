package numerics

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class GeneratingSystemTest {
    @Test fun wronskianNonZero() {
        for (sys in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
            for (t in listOf(0.1, 0.3, 0.5, 0.7, 0.9)) {
                assertTrue(abs(sys.wronskian(t)) > 1e-9, "wronskian zero for ${sys.name} at $t")
            }
        }
    }

    @Test fun cross3DotDetConsistency() {
        val u = doubleArrayOf(1.0, 0.0, 0.0)
        val v = doubleArrayOf(0.0, 1.0, 0.0)
        val c = cross3(u, v)
        assertTrue(abs(c[2] - 1.0) < 1e-15)
        assertTrue(abs(det3(u, v, doubleArrayOf(0.0, 0.0, 1.0)) - 1.0) < 1e-15)
        assertTrue(abs(dot3(u, v)) < 1e-15)
    }

    @Test fun invert3IsInverse() {
        val c0 = doubleArrayOf(2.0, 0.0, 1.0)
        val c1 = doubleArrayOf(0.0, 3.0, 0.0)
        val c2 = doubleArrayOf(1.0, 0.0, 2.0)
        val inv = invert3(c0, c1, c2)
        // inv * [c0|c1|c2] = I ; check inv applied to c0 gives e0
        fun mv(m: Array<DoubleArray>, x: DoubleArray) = DoubleArray(3) { i -> m[i][0]*x[0]+m[i][1]*x[1]+m[i][2]*x[2] }
        val e0 = mv(inv, c0)
        assertTrue(abs(e0[0]-1.0)<1e-12 && abs(e0[1])<1e-12 && abs(e0[2])<1e-12)
    }

    /** Регресс #5: вырожденная 3x3 (две одинаковые строки/столбца, det=0) -> исключение. */
    @Test fun invert3SingularThrows() {
        val c0 = doubleArrayOf(1.0, 2.0, 3.0)
        val c1 = doubleArrayOf(1.0, 2.0, 3.0) // столбец совпадает с c0 -> det=0
        val c2 = doubleArrayOf(4.0, 5.0, 6.0)
        assertFailsWith<IllegalArgumentException> { invert3(c0, c1, c2) }
    }
}
