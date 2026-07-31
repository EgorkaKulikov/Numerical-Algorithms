package numerics

import org.junit.jupiter.api.Tag
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * ДОКАЗАТЕЛЬСТВО НЕЙТРАЛЬНОСТИ замены критерия кратного узла.
 *
 * Ранее `MinimalSplineBasis.computeA` (и её дословная копия `AveragingFunctionals.aN`)
 * определяли тройной краевой узел сравнением значений: `x(j+1) == x(j+2)`. Теперь
 * используется индексный предикат [Grid.isCoincident]. Тест проверяет, что новый
 * критерий совпадает со старым ПОБИТОВО — не «почти всегда», а на всём допустимом
 * диапазоне индексов `j = -2..n+1`, на ВСЕХ ЧЕТЫРЁХ фабриках сеток и при нескольких n.
 *
 * Почему этого не покрывает характеризационный гейт: эталон `baseline-eh.tsv` снят
 * только на `Grid.uniform` при n из {8,16} на отрезке [0,1]. Неравномерные фабрики
 * (`quasiUniform`, `geometric`, `graded`) им не защищены вовсе — именно поэтому
 * эквивалентность доказывается здесь отдельно и исчерпывающе.
 */
@Tag("fast")
class GridCoincidenceTest {

    /** Отрезки, на которых строятся сетки (проверяем независимость от масштаба и знака). */
    private val segments = listOf(0.0 to 1.0, -2.0 to 3.5)

    /**
     * Все сетки для данного n: uniform и quasiUniform определены при n >= 1,
     * geometric и graded требуют n >= 2.
     */
    private fun gridsFor(n: Int, a: Double, b: Double): List<Pair<String, Grid>> = buildList {
        add("uniform" to Grid.uniform(n, a, b))
        add("quasiUniform" to Grid.quasiUniform(n, a, b))
        if (n >= 2) {
            add("geometric" to Grid.geometric(n, a, b))
            add("graded" to Grid.graded(n, a, b))
        }
    }

    /**
     * ГЛАВНАЯ ПРОВЕРКА: `isCoincident(j)` == (`x(j) == x(j+1)`) для всех допустимых j.
     *
     * Диапазон j = -2..n+1 — максимальный, при котором ОБА узла пары лежат в хранимом
     * диапазоне -2..n+2. Обе границы диапазона включены.
     */
    @Test fun predicateMatchesValueComparisonEverywhere() {
        var checks = 0
        var gridsChecked = 0
        var coincidentSeen = 0
        for (n in listOf(1, 2, 8, 16)) {
            for ((a, b) in segments) {
                for ((name, g) in gridsFor(n, a, b)) {
                    gridsChecked++
                    for (j in -2..n + 1) {
                        val byValue = g.x(j) == g.x(j + 1)
                        val byIndex = g.isCoincident(j)
                        assertEquals(
                            byValue, byIndex,
                            "$name(n=$n, [$a,$b]): j=$j -> x($j)=${g.x(j)}, x(${j + 1})=${g.x(j + 1)}; " +
                                "сравнение значений даёт $byValue, индексный предикат — $byIndex",
                        )
                        if (byIndex) coincidentSeen++
                        checks++
                    }
                }
            }
        }
        // Контроль самого теста: обе ветви предиката реально встречались.
        assertEquals(28, gridsChecked, "ожидалось 28 сеток (2+4+4+4 фабрик на 2 отрезках)")
        assertEquals(324, checks, "ожидалось 324 проверки (по n+4 индекса на сетку)")
        assertEquals(4 * gridsChecked, coincidentSeen, "на каждой сетке ровно 4 кратные пары: j=-2,-1,n,n+1")
    }

    /**
     * Сдвиг индекса в `computeA` проверяется НА САМОМ `computeA`, а не повторением
     * того же тождества про сетку.
     *
     * `computeA(j)` спрашивает про пару (x_{j+1}, x_{j+2}), то есть обязана вызывать
     * `isCoincident(j + 1)`. Наблюдаемое следствие ветвления: при кратном узле
     * возвращается РОВНО `phi(x_{j+1})` (ранний возврат, без вычитания поправки),
     * а на общей ветви результат ему НЕ равен (вычитается `coef * phiD`, где `coef != 0`).
     * Этого достаточно, чтобы различить ветви извне, не заглядывая в приватные детали.
     *
     * Прежняя версия этого теста к `computeA` не обращалась вовсе и прошла бы при
     * ЛЮБОМ сдвиге внутри `computeA` — то есть не проверяла заявленного.
     */
    @Test fun computeAUsesCoincidenceAtShiftedIndex() {
        var coincidentSeen = 0
        var regularSeen = 0
        for (n in listOf(2, 8, 16)) {
            for ((a, b) in segments) {
                for ((name, g) in gridsFor(n, a, b)) {
                    val basis = MinimalSplineBasis(GeneratingSystem.B, g)
                    for (j in -2..n - 1) {
                        val actual = basis.computeA(j)
                        val phiAtLeft = basis.sys.phi(g.x(j + 1))
                        val isTripleKnot = g.isCoincident(j + 1)
                        // Контроль самого критерия: предикат обязан совпадать со старым сравнением.
                        assertEquals(
                            g.x(j + 1) == g.x(j + 2), isTripleKnot,
                            "$name(n=$n, [$a,$b]): предикат разошёлся со старым критерием при j=$j",
                        )
                        if (isTripleKnot) {
                            coincidentSeen++
                            assertTrue(
                                actual.contentEquals(phiAtLeft),
                                "$name(n=$n, [$a,$b]): при кратном узле (j=$j) computeA обязана вернуть " +
                                    "ровно phi(x_${j + 1})=${phiAtLeft.toList()}, получено ${actual.toList()}",
                            )
                        } else {
                            regularSeen++
                            assertTrue(
                                !actual.contentEquals(phiAtLeft),
                                "$name(n=$n, [$a,$b]): при НЕкратном узле (j=$j) computeA вернула ровно " +
                                    "phi(x_${j + 1}) — значит ошибочно сработала ветвь тройного узла " +
                                    "(вероятная причина: неверный сдвиг индекса в isCoincident)",
                            )
                        }
                    }
                }
            }
        }
        // Защита от вырождения: обе ветви computeA действительно были пройдены.
        assertTrue(coincidentSeen > 0, "Ветвь тройного узла ни разу не сработала")
        assertTrue(regularSeen > 0, "Общая ветвь computeA ни разу не сработала")
    }

    /**
     * Сдвиг в терминах самой сетки: для рабочего диапазона `computeA` (j = -2..n-1)
     * вызов `isCoincident(j + 1)` всегда лежит в допустимом диапазоне -2..n+1
     * и совпадает со старым сравнением значений (то есть исключение недостижимо).
     */
    @Test fun shiftedIndexStaysInAllowedRange() {
        for (n in listOf(1, 2, 8, 16)) {
            for ((a, b) in segments) {
                for ((name, g) in gridsFor(n, a, b)) {
                    for (j in -2..n - 1) {
                        assertTrue(
                            (j + 1) in -2..g.n + 1,
                            "$name(n=$n): сдвинутый индекс ${j + 1} вышел за допустимый диапазон",
                        )
                        assertEquals(
                            g.x(j + 1) == g.x(j + 2), g.isCoincident(j + 1),
                            "$name(n=$n, [$a,$b]): сдвиг неверен при j=$j",
                        )
                    }
                }
            }
        }
    }

    /** Кратны ровно краевые пары: j = -2, -1 (левый тройной узел) и j = n, n+1 (правый). */
    @Test fun onlyBoundaryPairsAreCoincident() {
        val g = Grid.graded(8, 0.0, 1.0)
        for (j in listOf(-2, -1, g.n, g.n + 1)) assertTrue(g.isCoincident(j), "j=$j обязан быть кратным")
        for (j in 0 until g.n) assertTrue(!g.isCoincident(j), "внутренний j=$j не может быть кратным")
    }

    /** Вне диапазона -2..n+1 второй узел пары не существует — предикат обязан отвергнуть вход. */
    @Test fun rejectsOutOfRangeIndex() {
        val g = Grid.uniform(4, 0.0, 1.0)
        assertFailsWith<IllegalArgumentException> { g.isCoincident(-3) }
        assertFailsWith<IllegalArgumentException> { g.isCoincident(g.n + 2) }
    }
}
