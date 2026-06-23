package numerics

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Тесты CheckResult.ok: ok=true когда measured<=threshold и не NaN;
 * false при превышении порога и при NaN (две ветви предиката).
 */
class CheckResultTest {
    /** В пределах порога — ok=true. */
    @Test fun okWhenBelowThreshold() {
        assertTrue(CheckResult("a", 0.5, 1.0, true).ok)
        assertTrue(CheckResult("eq", 1.0, 1.0, false).ok) // ровно на пороге
    }

    /** Превышение порога — ok=false. */
    @Test fun notOkWhenAboveThreshold() {
        assertFalse(CheckResult("b", 2.0, 1.0, true).ok)
    }

    /** NaN-измерение всегда не ok (короткое замыкание по isNaN). */
    @Test fun notOkWhenNaN() {
        assertFalse(CheckResult("c", Double.NaN, 1.0, false).ok)
    }
}
