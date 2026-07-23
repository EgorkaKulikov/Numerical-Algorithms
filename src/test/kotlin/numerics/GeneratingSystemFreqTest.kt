package numerics

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Тесты частотно-параметризованных порождающих систем hyperbolic(omega)/trig(omega):
 * невырожденность вронскиана на [0,1]; hyperbolic(1.0)/trig(1.0) воспроизводят H/T;
 * корректность производных против конечных разностей rho,sigma; для trig(omega)
 * ограничение неотрицательности сплайна h*omega<pi на используемых сетках.
 */
class GeneratingSystemFreqTest {
    private val ts = (0..100).map { it / 100.0 }

    @Test fun wronskianNonzeroHyperbolic() {
        for (w in listOf(0.5, 1.0, 1.5, 3.0)) {
            val sys = GeneratingSystem.hyperbolic(w)
            for (t in ts) assertTrue(kotlin.math.abs(sys.wronskian(t)) > 1e-12,
                "hyperbolic($w) Wronskian ~0 at t=$t")
        }
    }

    @Test fun wronskianNonzeroTrig() {
        for (w in listOf(0.5, 1.0, 3.0)) {
            val sys = GeneratingSystem.trig(w)
            for (t in ts) assertTrue(kotlin.math.abs(sys.wronskian(t)) > 1e-12,
                "trig($w) Wronskian ~0 at t=$t")
        }
    }

    @Test fun hyperbolicUnitReproducesH() {
        val h = GeneratingSystem.H; val hf = GeneratingSystem.hyperbolic(1.0)
        for (t in ts) {
            assertEquals(h.rho(t), hf.rho(t), 1e-14)
            assertEquals(h.sigma(t), hf.sigma(t), 1e-14)
            assertEquals(h.rhoD(t), hf.rhoD(t), 1e-14)
            assertEquals(h.sigmaD(t), hf.sigmaD(t), 1e-14)
            assertEquals(h.rhoDD(t), hf.rhoDD(t), 1e-14)
            assertEquals(h.sigmaDD(t), hf.sigmaDD(t), 1e-14)
        }
    }

    @Test fun trigUnitReproducesT() {
        val tt = GeneratingSystem.T; val tf = GeneratingSystem.trig(1.0)
        for (t in ts) {
            assertEquals(tt.rho(t), tf.rho(t), 1e-14)
            assertEquals(tt.sigma(t), tf.sigma(t), 1e-14)
            assertEquals(tt.rhoD(t), tf.rhoD(t), 1e-14)
            assertEquals(tt.sigmaD(t), tf.sigmaD(t), 1e-14)
            assertEquals(tt.rhoDD(t), tf.rhoDD(t), 1e-14)
            assertEquals(tt.sigmaDD(t), tf.sigmaDD(t), 1e-14)
        }
    }

    @Test fun derivativesMatchFiniteDifference() {
        val eps = 1e-6
        for (sys in listOf(GeneratingSystem.hyperbolic(1.5), GeneratingSystem.trig(3.0))) {
            for (t in listOf(0.2, 0.5, 0.8)) {
                val rhoDNum = (sys.rho(t + eps) - sys.rho(t - eps)) / (2 * eps)
                val sigDNum = (sys.sigma(t + eps) - sys.sigma(t - eps)) / (2 * eps)
                val rhoDDNum = (sys.rho(t + eps) - 2 * sys.rho(t) + sys.rho(t - eps)) / (eps * eps)
                val sigDDNum = (sys.sigma(t + eps) - 2 * sys.sigma(t) + sys.sigma(t - eps)) / (eps * eps)
                assertEquals(rhoDNum, sys.rhoD(t), 1e-4, "${sys.name} rhoD at $t")
                assertEquals(sigDNum, sys.sigmaD(t), 1e-4, "${sys.name} sigmaD at $t")
                assertEquals(rhoDDNum, sys.rhoDD(t), 1e-3, "${sys.name} rhoDD at $t")
                assertEquals(sigDDNum, sys.sigmaDD(t), 1e-3, "${sys.name} sigmaDD at $t")
            }
        }
    }

    /** trig(omega): использованные сетки n=8..64 на [0,1] удовлетворяют h*omega<pi (omega=3). */
    @Test fun trigGridSatisfiesNonNegativityCondition() {
        val omega = 3.0
        for (n in listOf(8, 16, 32, 64)) {
            val h = Grid.uniform(n).h
            assertTrue(h * omega < Math.PI, "h*omega=${h * omega} !< pi at n=$n")
        }
    }
}
