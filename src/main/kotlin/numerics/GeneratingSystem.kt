package numerics

// ============================================================================
// 3. ПОРОЖДАЮЩАЯ ВЕКТОР-ФУНКЦИЯ phi(t) = (1, rho(t), sigma(t))^T
// ============================================================================

/**
 * Порождающая вектор-функция phi(t) = (1, rho(t), sigma(t))^T и её производные
 * до второго порядка. phi_0 == 1 обеспечивает разбиение единицы.
 */
class GeneratingSystem(
    val name: String,
    val rho: (Double) -> Double,
    val sigma: (Double) -> Double,
    val rhoD: (Double) -> Double,
    val sigmaD: (Double) -> Double,
    val rhoDD: (Double) -> Double,
    val sigmaDD: (Double) -> Double,
) {
    /** phi(t) = (1, rho(t), sigma(t)). */
    fun phi(t: Double): DoubleArray = doubleArrayOf(1.0, rho(t), sigma(t))

    /** phi'(t) = (0, rho'(t), sigma'(t)). */
    fun phiD(t: Double): DoubleArray = doubleArrayOf(0.0, rhoD(t), sigmaD(t))

    /** phi''(t) = (0, rho''(t), sigma''(t)). */
    fun phiDD(t: Double): DoubleArray = doubleArrayOf(0.0, rhoDD(t), sigmaDD(t))

    /** Вронскиан det(phi, phi', phi'') — проверка невырожденности. */
    fun wronskian(t: Double): Double = det3(phi(t), phiD(t), phiDD(t))

    companion object {
        /** Полиномиальная phi^B(t) = (1, t, t^2)^T. */
        val B = GeneratingSystem(
            name = "B",
            rho = { t -> t }, sigma = { t -> t * t },
            rhoD = { 1.0 }, sigmaD = { t -> 2.0 * t },
            rhoDD = { 0.0 }, sigmaDD = { 2.0 },
        )

        /** Гиперболическая phi^H(t) = (1, sinh t, cosh t)^T. */
        val H = GeneratingSystem(
            name = "H",
            rho = { t -> Math.sinh(t) }, sigma = { t -> Math.cosh(t) },
            rhoD = { t -> Math.cosh(t) }, sigmaD = { t -> Math.sinh(t) },
            rhoDD = { t -> Math.sinh(t) }, sigmaDD = { t -> Math.cosh(t) },
        )

        /** Тригонометрическая phi^T(t) = (1, sin t, cos t)^T. */
        val T = GeneratingSystem(
            name = "T",
            rho = { t -> Math.sin(t) }, sigma = { t -> Math.cos(t) },
            rhoD = { t -> Math.cos(t) }, sigmaD = { t -> -Math.sin(t) },
            rhoDD = { t -> -Math.sin(t) }, sigmaDD = { t -> -Math.cos(t) },
        )
    }
}

/** Векторное произведение u x v в R^3. */
fun cross3(u: DoubleArray, v: DoubleArray): DoubleArray = doubleArrayOf(
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0],
)

/** Скалярное произведение в R^3. */
fun dot3(u: DoubleArray, v: DoubleArray): Double = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

/** Определитель 3x3 из столбцов-векторов (a, b, c). */
fun det3(a: DoubleArray, b: DoubleArray, c: DoubleArray): Double = dot3(a, cross3(b, c))

/**
 * Обращение матрицы 3x3 со столбцами c0, c1, c2 (строки M^{-1} = (c1 x c2,
 * c2 x c0, c0 x c1)/det). Используется для M_k^{-1} при устойчивом построении omega_j.
 */
fun invert3(c0: DoubleArray, c1: DoubleArray, c2: DoubleArray): Array<DoubleArray> {
    val det = det3(c0, c1, c2)
    val r0 = cross3(c1, c2)
    val r1 = cross3(c2, c0)
    val r2 = cross3(c0, c1)
    return arrayOf(
        doubleArrayOf(r0[0] / det, r0[1] / det, r0[2] / det),
        doubleArrayOf(r1[0] / det, r1[1] / det, r1[2] / det),
        doubleArrayOf(r2[0] / det, r2[1] / det, r2[2] / det),
    )
}
