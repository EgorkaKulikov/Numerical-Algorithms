package numerics.functionals

import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Тесты булевых флагов семейств: theta — проектор без производной, xi — проектор
 * с производной, mu/lambda — квазиинтерполянты. Покрывают геттеры свойств.
 */
class FlagsTest {
    private val basis = MinimalSplineBasis(GeneratingSystem.B, Grid.uniform(8))

    /** theta (ProjFunctionals): isProjector=true, usesDerivative=false. */
    @Test fun thetaFlags() {
        val theta = ProjFunctionals(basis)
        assertTrue(theta.isProjector)
        assertFalse(theta.usesDerivative)
        assertEquals("theta", theta.name)
    }

    /** Имена семейств соответствуют греческим обозначениям. */
    @Test fun familyNames() {
        assertEquals("xi", DeBoorFixFunctionals(basis).name)
        assertEquals("mu", AveragingFunctionals(basis).name)
        assertEquals("lambda", ThreePointFunctionals(basis).name)
    }
}
