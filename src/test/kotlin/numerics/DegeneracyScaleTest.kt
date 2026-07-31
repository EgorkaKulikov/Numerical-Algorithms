package numerics

import org.junit.jupiter.api.Tag
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Масштабная инвариантность порогов вырожденности (пункт 2.6 спека).
 *
 * До правки все проверки использовали АБСОЛЮТНЫЙ порог `1e-14`, что давало две
 * симметричные ошибки:
 *  - ЛОЖНОЕ «singular» на мелком масштабе: `det(M_k) ~ h^3`, поэтому на отрезке
 *    `[0, 1e-6]` совершенно корректная сетка отбраковывалась;
 *  - ПРОПУСК вырожденности на крупном масштабе: на `[0, 1e6]` определитель почти
 *    зависимых столбцов — чистый шум округления, но по модулю велик.
 *
 * Оба сценария закреплены здесь фактически. См. KDoc [DEGENERACY_RELATIVE_EPS].
 */
@Tag("fast")
class DegeneracyScaleTest {

    /**
     * МЕЛКИЙ МАСШТАБ, invert3: `diag(s,s,s)` при `s = 1e-6` идеально обусловлена
     * (число обусловленности = 1), но `det = 1e-18 < 1e-14` — старый абсолютный
     * порог давал ложное «singular».
     */
    @Test
    fun invert3AcceptsWellConditionedMatrixAtSmallScale() {
        val s = 1e-6
        val inv = invert3(
            doubleArrayOf(s, 0.0, 0.0),
            doubleArrayOf(0.0, s, 0.0),
            doubleArrayOf(0.0, 0.0, s),
        )
        // M = s*I, значит M^{-1} = (1/s)*I.
        for (i in 0..2) for (j in 0..2) {
            val expected = if (i == j) 1.0 / s else 0.0
            assertTrue(
                kotlin.math.abs(inv[i][j] - expected) <= 1e-6 * kotlin.math.abs(expected) + 1e-9,
                "invert3(diag($s))^{-1}[$i][$j] = ${inv[i][j]}, ожидалось $expected"
            )
        }
    }

    /**
     * КРУПНЫЙ МАСШТАБ, invert3: третий столбец почти равен сумме двух первых
     * (относительное отклонение 1e-13). При `s = 1e6` определитель по модулю ~1e5,
     * то есть старый порог `1e-14` пропускал эту матрицу и обращение давало мусор.
     * Относительный критерий её отбраковывает.
     */
    @Test
    fun invert3RejectsNearlyDependentColumnsAtLargeScale() {
        val s = 1e6
        val c0 = doubleArrayOf(s, s, s)
        val c1 = doubleArrayOf(s, 2 * s, 4 * s)
        // c2 = c0 + c1 + (0, 0, s*1e-13): линейная зависимость нарушена лишь шумом.
        val c2 = doubleArrayOf(2 * s, 3 * s, 5 * s + s * 1e-13)

        val det = det3(c0, c1, c2)
        val scale = det3Scale(c0, c1, c2)
        // Диагностика для протокола: |det| велик по модулю, но ничтожен на своём масштабе.
        assertTrue(kotlin.math.abs(det) > 1e-14, "старый абсолютный порог такую матрицу пропускал: det=$det")
        assertTrue(
            kotlin.math.abs(det) <= DEGENERACY_RELATIVE_EPS * scale,
            "относительный критерий обязан её поймать: det=$det, scale=$scale"
        )
        assertFailsWith<IllegalArgumentException> { invert3(c0, c1, c2) }
    }

    /**
     * МЕЛКИЙ МАСШТАБ, реальный путь построения: базис минимальных сплайнов на
     * отрезке `[0, 1e-6]` для полиномиальной системы B.
     *
     * Здесь `det(M_k) ~ h^3 ~ 2e-21` при собственном масштабе определителя `~1e-19`
     * (компоненты столбцов — `1, t, t^2`), то есть отношение `det/scale ~ 1e-2` и
     * определитель вычислен полностью достоверно. Тем не менее ДО правки
     * построение падало с ложным «invert3: matrix is singular» из-за абсолютного
     * порога `1e-14`. Это и есть ключевое доказательство ценности правки.
     *
     * Проверяется не только отсутствие исключения, но и содержательное свойство —
     * разбиение единицы `sum_j omega_j(t) = 1` (первая компонента phi равна 1).
     */
    @Test
    fun splineBasisBuildsOnTinyInterval() {
        val grid = Grid.uniform(8, 0.0, 1e-6)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        for (k in 0 until grid.n) {
            val t = 0.5 * (grid.x(k) + grid.x(k + 1))
            var sum = 0.0
            for (j in -2..grid.n - 1) sum += basis.omega(j, t)
            assertEquals(1.0, sum, 1e-9, "разбиение единицы в t=$t")
        }
    }

    /**
     * ГРАНИЦА ПРИМЕНИМОСТИ (найдено фактически): для НЕПОЛИНОМИАЛЬНЫХ
     * порождающих систем (H, T) отрезок длины `1e-6` отбраковывается и ПОСЛЕ
     * правки — и это ПРАВИЛЬНО, а не недоработка порога.
     *
     * Причина: у таких систем третья компонента phi НЕ мала на мелком отрезке
     * (`cos t -> 1`, `cosh t -> 1`), поэтому собственный масштаб определителя — `~t`,
     * а сам определитель — `~h^3`: отношение `det/scale ~ 2e-15`, то есть в ответе
     * осталось около ОДНОЙ верной цифры (eps = 2.2e-16). Обращать такую матрицу
     * бессмысленно; отказ — честное поведение. Старый абсолютный порог тоже
     * отказывал, но «случайно правильно» и без диагностики; теперь в сообщении
     * видно отношение `det/scale`.
     */
    @Test
    fun nonPolynomialSystemOnTinyIntervalIsHonestlyRejected() {
        for (sys in listOf(GeneratingSystem.H, GeneratingSystem.T)) {
            val grid = Grid.uniform(8, 0.0, 1e-6)
            val ex = assertFailsWith<IllegalArgumentException>("система ${sys.name}") {
                MinimalSplineBasis(sys, grid)
            }
            assertTrue(
                ex.message!!.contains("det/scale"),
                "диагностика обязана показывать потерю значимости: ${ex.message}"
            )
        }
    }

    /**
     * КРУПНЫЙ МАСШТАБ, реальный путь: на `[0, 1e6]` построение тоже обязано
     * работать — сетка невырождена, просто велика. Проверка симметрична предыдущей
     * и доказывает, что относительный критерий не «съехал» в другую сторону.
     *
     * Используются только полиномиальная и тригонометрическая системы: у
     * гиперболической `cosh(1e6)` — переполнение double, и это ограничение самой
     * порождающей системы, а не порогов вырожденности.
     */
    @Test
    fun splineBasisBuildsOnHugeInterval() {
        for (sys in listOf(GeneratingSystem.B, GeneratingSystem.T)) {
            val grid = Grid.uniform(8, 0.0, 1e6)
            val basis = MinimalSplineBasis(sys, grid)
            for (k in 0 until grid.n) {
                val t = 0.5 * (grid.x(k) + grid.x(k + 1))
                var sum = 0.0
                for (j in -2..grid.n - 1) sum += basis.omega(j, t)
                assertEquals(1.0, sum, 1e-6, "разбиение единицы для ${sys.name} в t=$t")
            }
        }
    }

    /** Нулевая матрица вырождена на любом масштабе: `scale == 0` даёт false по построению. */
    @Test
    fun invert3RejectsZeroMatrix() {
        assertFailsWith<IllegalArgumentException> {
            invert3(DoubleArray(3), DoubleArray(3), DoubleArray(3))
        }
    }
}
