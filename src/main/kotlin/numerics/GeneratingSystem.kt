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

        /** Гиперболическая phi^H(t) = (1, sinh t, cosh t)^T (единичная частота). */
        val H = hyperbolic(1.0, name = "H")

        /** Тригонометрическая phi^T(t) = (1, sin t, cos t)^T (единичная частота). */
        val T = trig(1.0, name = "T")

        /**
         * Гиперболическая система с частотой omega:
         *   phi(t) = (1, sinh(omega t), cosh(omega t))^T,
         *   rho'=omega cosh(omega t), sigma'=omega sinh(omega t),
         *   rho''=omega^2 sinh(omega t), sigma''=omega^2 cosh(omega t).
         * Span точно представляет sinh(omega t), cosh(omega t) (частотная настройка).
         *
         * @param omega частота (> 0 для невырожденной системы).
         * @param name имя системы (по умолчанию "H(omega)").
         */
        fun hyperbolic(omega: Double, name: String = "H($omega)"): GeneratingSystem =
            GeneratingSystem(
                name = name,
                rho = { t -> Math.sinh(omega * t) }, sigma = { t -> Math.cosh(omega * t) },
                rhoD = { t -> omega * Math.cosh(omega * t) }, sigmaD = { t -> omega * Math.sinh(omega * t) },
                rhoDD = { t -> omega * omega * Math.sinh(omega * t) },
                sigmaDD = { t -> omega * omega * Math.cosh(omega * t) },
            )

        /**
         * Тригонометрическая система с частотой omega:
         *   phi(t) = (1, sin(omega t), cos(omega t))^T,
         *   rho'=omega cos(omega t), sigma'=-omega sin(omega t),
         *   rho''=-omega^2 sin(omega t), sigma''=-omega^2 cos(omega t).
         * Span точно представляет sin(omega t), cos(omega t) (частотная настройка).
         * ЗАМЕЧАНИЕ: неотрицательность минимальных сплайнов требует h*omega < pi
         * (шаг сетки достаточно мелкий относительно частоты).
         *
         * @param omega частота (> 0 для невырожденной системы).
         * @param name имя системы (по умолчанию "T(omega)").
         */
        fun trig(omega: Double, name: String = "T($omega)"): GeneratingSystem =
            GeneratingSystem(
                name = name,
                rho = { t -> Math.sin(omega * t) }, sigma = { t -> Math.cos(omega * t) },
                rhoD = { t -> omega * Math.cos(omega * t) }, sigmaD = { t -> -omega * Math.sin(omega * t) },
                rhoDD = { t -> -omega * omega * Math.sin(omega * t) },
                sigmaDD = { t -> -omega * omega * Math.cos(omega * t) },
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
    require(kotlin.math.abs(det) >= 1e-14) {
        "invert3: matrix is singular or near-singular, det=$det"
    }
    val r0 = cross3(c1, c2)
    val r1 = cross3(c2, c0)
    val r2 = cross3(c0, c1)
    return arrayOf(
        doubleArrayOf(r0[0] / det, r0[1] / det, r0[2] / det),
        doubleArrayOf(r1[0] / det, r1[1] / det, r1[2] / det),
        doubleArrayOf(r2[0] / det, r2[1] / det, r2[2] / det),
    )
}
