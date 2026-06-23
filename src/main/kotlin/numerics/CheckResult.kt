package numerics

/** Результат одной самопроверки. */
class CheckResult(val name: String, val measured: Double, val threshold: Double, val critical: Boolean) {
    val ok: Boolean get() = !measured.isNaN() && measured <= threshold
}
