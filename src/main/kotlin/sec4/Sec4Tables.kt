package sec4

import kotlin.math.ln
import numerics.Fmt
import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.DeBoorFixFunctionals
import numerics.functionals.DiscreteDeBoorFixFunctionals
import numerics.functionals.FunctionalFamily
import numerics.functionals.ProjFunctionals
import numerics.functionals.errorEh
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FirstKindSolver as FFirst
import solvers.fredholm.ModelProblem as FProblem
import solvers.fredholm.SecondKindSolver as FSolver
import solvers.volterra.VolterraOperator
import solvers.volterra.FirstKindSolver as VFirst
import solvers.volterra.ModelProblem as VProblem
import solvers.volterra.SecondKindSolver as VSolver

// ============================================================================
// PRODUCTION §4 DRIVER (role 11). Reproducible, verbatim-parse-friendly tables
// for section 4 of papers/new-01. Logic mirrors the validated V&V harness
// (vv/Sec4VV.kt): E_h = sup over 100n+1 control grid via errorEh; p_h from the
// GENERAL max-step h (non-uniform safe). NO order is claimed as proven here —
// labels only. Numbers are whatever the code computes; never hand-edited.
// ============================================================================

private val QUAD = GaussLegendre(8)

private fun ordersGen(errs: List<Double>, hs: List<Double>): List<Double> =
    errs.indices.map { i ->
        if (i + 1 < errs.size) ln(errs[i] / errs[i + 1]) / ln(hs[i] / hs[i + 1]) else Double.NaN
    }

private fun makeFuncs(fam: String, basis: MinimalSplineBasis): FunctionalFamily = when (fam) {
    "theta" -> ProjFunctionals(basis)
    "xi1" -> DeBoorFixFunctionals(basis, 1)
    "xit1" -> DiscreteDeBoorFixFunctionals(basis, 1)
    "xit2" -> DiscreteDeBoorFixFunctionals(basis, 2)
    else -> error("unknown family $fam")
}

private fun gridMaker(kind: String): (Int) -> Grid = when (kind) {
    "uniform" -> { n -> Grid.uniform(n) }
    "geometric" -> { n -> Grid.geometric(n, R = 2.0) }
    else -> error("unknown grid $kind")
}

// --- Fredholm / Volterra plumbing -------------------------------------------
private fun fSolver(p: FProblem, sys: GeneratingSystem, fam: String, grid: Grid): FSolver {
    val basis = MinimalSplineBasis(sys, grid)
    val op = FredholmOperator(p.kernel, grid, QUAD)
    return FSolver(
        basis, makeFuncs(fam, basis), op, 1.0,
        { t -> p.rhsExact(t, op) }, { t -> p.rhsExactDeriv(t, op) }, { t -> p.rhsExactDeriv2(t, op) },
    )
}

private fun fEval(s: FSolver, scheme: String): (Double) -> Double = when (scheme) {
    "base" -> s.base().eval
    "sloan" -> s.sloan().eval
    "kulk" -> s.kulkarni().eval
    "itkulk" -> s.iteratedKulkarni().eval
    "nyst" -> s.nystrom().eval
    "itnyst" -> s.iteratedNystrom().eval
    else -> error("unknown scheme $scheme")
}

private fun vSolver(p: VProblem, sys: GeneratingSystem, fam: String, grid: Grid): VSolver {
    val basis = MinimalSplineBasis(sys, grid)
    val op = VolterraOperator(p.kernel, grid, QUAD)
    return VSolver(
        basis, makeFuncs(fam, basis), op, 1.0,
        { t -> p.rhsExact(t, op) }, { t -> p.rhsExactDeriv(t, op) }, { t -> p.rhsExactDeriv2(t, op) },
    )
}

private fun vEval(s: VSolver, scheme: String): (Double) -> Double = when (scheme) {
    "base" -> s.base().eval
    "sloan" -> s.sloan().eval
    "kulk" -> s.kulkarni().eval
    "itkulk" -> s.iteratedKulkarni().eval
    "nyst" -> s.nystrom().eval
    "itnyst" -> s.iteratedNystrom().eval
    else -> error("unknown scheme $scheme")
}

private fun fEh(p: FProblem, sys: GeneratingSystem, fam: String, scheme: String, grid: Grid): Double =
    errorEh({ t -> p.exact(t) }, fEval(fSolver(p, sys, fam, grid), scheme), grid)

private fun vEh(p: VProblem, sys: GeneratingSystem, fam: String, scheme: String, grid: Grid): Double =
    errorEh({ t -> p.exact(t) }, vEval(vSolver(p, sys, fam, grid), scheme), grid)

private fun sep(title: String) {
    println("\n" + "=".repeat(78)); println(title); println("=".repeat(78))
}

// --- per-(problem,family,basis) block: one row per scheme, columns E_h & p_h --
private fun rowFredholm(tag: String, p: FProblem, sys: GeneratingSystem, fam: String, schemes: List<String>, gk: String, ns: List<Int>) {
    val gm = gridMaker(gk); val hs = ns.map { gm(it).h }
    println("[$tag] problem=${p.name} basis=${sys.name} func=$fam grid=$gk  (h: ${hs.joinToString(", ") { Fmt.h(it) }})")
    for (sc in schemes) {
        val errs = ns.map { fEh(p, sys, fam, sc, gm(it)) }
        val ps = ordersGen(errs, hs)
        println("   %-7s ".format(sc) + ns.indices.joinToString("  ") { "n=%3d E=%s p=%s".format(ns[it], Fmt.e(errs[it]), Fmt.p(ps[it])) })
    }
}

private fun rowVolterra(tag: String, p: VProblem, sys: GeneratingSystem, fam: String, schemes: List<String>, gk: String, ns: List<Int>) {
    val gm = gridMaker(gk); val hs = ns.map { gm(it).h }
    println("[$tag] problem=${p.name} basis=${sys.name} func=$fam grid=$gk  (h: ${hs.joinToString(", ") { Fmt.h(it) }})")
    for (sc in schemes) {
        val errs = ns.map { vEh(p, sys, fam, sc, gm(it)) }
        val ps = ordersGen(errs, hs)
        println("   %-7s ".format(sc) + ns.indices.joinToString("  ") { "n=%3d E=%s p=%s".format(ns[it], Fmt.e(errs[it]), Fmt.p(ps[it])) })
    }
}

// --- multi-column table: one column per (label,sys,fam,scheme), rows over n ---
private data class Col(val label: String, val sys: GeneratingSystem, val fam: String, val scheme: String)

private fun colFredholm(tag: String, p: FProblem, cols: List<Col>, gk: String, ns: List<Int>) {
    val gm = gridMaker(gk); val hs = ns.map { gm(it).h }
    println("[$tag] problem=${p.name} grid=$gk")
    println("   columns: " + cols.joinToString("  ") { it.label })
    val errsBy = cols.map { c -> ns.map { fEh(p, c.sys, c.fam, c.scheme, gm(it)) } }
    for (i in ns.indices) println("   n=%3d ".format(ns[i]) + cols.indices.joinToString("  ") { Fmt.e(errsBy[it][i]) })
    for (ci in cols.indices) println("   p_h[${cols[ci].label}] = ${ordersGen(errsBy[ci], hs).map { Fmt.p(it) }}")
}

private fun colVolterra(tag: String, p: VProblem, cols: List<Col>, gk: String, ns: List<Int>) {
    val gm = gridMaker(gk); val hs = ns.map { gm(it).h }
    println("[$tag] problem=${p.name} grid=$gk")
    println("   columns: " + cols.joinToString("  ") { it.label })
    val errsBy = cols.map { c -> ns.map { vEh(p, c.sys, c.fam, c.scheme, gm(it)) } }
    for (i in ns.indices) println("   n=%3d ".format(ns[i]) + cols.indices.joinToString("  ") { Fmt.e(errsBy[it][i]) })
    for (ci in cols.indices) println("   p_h[${cols[ci].label}] = ${ordersGen(errsBy[ci], hs).map { Fmt.p(it) }}")
}

// ============================================================================
// BLOCKS
// ============================================================================

private val NS = listOf(8, 16, 32, 64)
private val NS_V_NYST = listOf(8, 16, 32) // Volterra Nyström is O(n) per eval -> cap

private fun blockH1H2() {
    val schemes = listOf("base", "sloan", "kulk", "itkulk")
    sep("BLOCK H1/H2 (uniform): theta(B) baseline, xi1(B), xi1(H); base/Sloan/Kulk/itKulk")
    for (p in listOf(FProblem.M1, FProblem.M2)) {
        rowFredholm("H1H2-F", p, GeneratingSystem.B, "theta", schemes, "uniform", NS)
        rowFredholm("H1H2-F", p, GeneratingSystem.B, "xi1", schemes, "uniform", NS)
        rowFredholm("H1H2-F", p, GeneratingSystem.H, "xi1", schemes, "uniform", NS)
    }
}

private fun blockH3() {
    sep("BLOCK H3 (uniform): constant-beats-order. M2: theta(B) vs xi1(H_1.5); M3: theta(B) vs xi1(T_3)")
    val H15 = GeneratingSystem.hyperbolic(1.5)
    val m2cols = listOf(
        Col("theta(B)-base", GeneratingSystem.B, "theta", "base"),
        Col("theta(B)-kulk", GeneratingSystem.B, "theta", "kulk"),
        Col("xi1(H1.5)-base", H15, "xi1", "base"),
        Col("xi1(H1.5)-kulk", H15, "xi1", "kulk"),
    )
    colFredholm("H3-M2", FProblem.M2, m2cols, "uniform", NS)
    val T3 = GeneratingSystem.trig(3.0)
    val m3cols = listOf(
        Col("theta(B)-base", GeneratingSystem.B, "theta", "base"),
        Col("theta(B)-kulk", GeneratingSystem.B, "theta", "kulk"),
        Col("xi1(T3)-base", T3, "xi1", "base"),
        Col("xi1(T3)-kulk", T3, "xi1", "kulk"),
    )
    colVolterra("H3-M3", VProblem.M3, m3cols, "uniform", NS)
}

private fun blockH3Grid() {
    sep("BLOCK H3-grid (Grid.geometric R=2): M3 headline theta(B) vs xi1(T_3) [works analogously]")
    val T3 = GeneratingSystem.trig(3.0)
    val m3cols = listOf(
        Col("theta(B)-base", GeneratingSystem.B, "theta", "base"),
        Col("theta(B)-kulk", GeneratingSystem.B, "theta", "kulk"),
        Col("xi1(T3)-base", T3, "xi1", "base"),
        Col("xi1(T3)-kulk", T3, "xi1", "kulk"),
    )
    colVolterra("H3grid-M3", VProblem.M3, m3cols, "geometric", NS)
}

private fun blockH4Nystrom() {
    sep("BLOCK H4 Nyström (uniform): theta-Nyström(B) vs xitilde-Nyström (r=1) on B and matched tuned basis")
    println("   [honest: xitilde-Nyström order ~3, BELOW O(h^8) orient; error tracks the integrand K*u]")
    // M1 (control), matched basis = B.
    val m1cols = listOf(
        Col("theta(B)-kulk[ref]", GeneratingSystem.B, "theta", "kulk"),
        Col("theta(B)-nyst", GeneratingSystem.B, "theta", "nyst"),
        Col("theta(B)-itnyst", GeneratingSystem.B, "theta", "itnyst"),
        Col("xit1(B)-nyst", GeneratingSystem.B, "xit1", "nyst"),
        Col("xit1(B)-itnyst", GeneratingSystem.B, "xit1", "itnyst"),
    )
    colFredholm("H4-M1", FProblem.M1, m1cols, "uniform", NS)
    // M3 headline, matched tuned basis = trig(3.0).
    val T3 = GeneratingSystem.trig(3.0)
    val m3cols = listOf(
        Col("theta(B)-kulk[ref]", GeneratingSystem.B, "theta", "kulk"),
        Col("theta(B)-nyst", GeneratingSystem.B, "theta", "nyst"),
        Col("theta(B)-itnyst", GeneratingSystem.B, "theta", "itnyst"),
        Col("xit1(B)-nyst", GeneratingSystem.B, "xit1", "nyst"),
        Col("xit1(T3)-nyst", T3, "xit1", "nyst"),
        Col("xit1(T3)-itnyst", T3, "xit1", "itnyst"),
    )
    colVolterra("H4-M3", VProblem.M3, m3cols, "uniform", NS_V_NYST)
}

private fun blockFirstKind() {
    val nsF = listOf(8, 16, 32)
    val nsV = listOf(8, 16, 32, 64)
    sep("BLOCK F1/V1: F1 (Wazwaz alpha=1e-10) base/Sloan; V1 (reduction) base/Sloan/Kulk")
    val fp = FProblem.F1
    for (sys in listOf(GeneratingSystem.B, GeneratingSystem.H)) {
        val hs = nsF.map { Grid.uniform(it).h }
        val eB = ArrayList<Double>(); val eS = ArrayList<Double>()
        for (n in nsF) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(sys, grid)
            val op = FredholmOperator(fp.kernel, grid, QUAD)
            val solver = FFirst(fp, basis, makeFuncs("xi1", basis), op, alpha = 1e-10)
            eB.add(errorEh({ t -> fp.exact(t) }, solver.base().eval, grid))
            eS.add(errorEh({ t -> fp.exact(t) }, solver.sloan().eval, grid))
        }
        val pB = ordersGen(eB, hs); val pS = ordersGen(eS, hs)
        println("[F1] basis=${sys.name} func=xi1 :")
        for (i in nsF.indices) println("   n=%3d base:E=%s p=%s | sloan:E=%s p=%s".format(nsF[i], Fmt.e(eB[i]), Fmt.p(pB[i]), Fmt.e(eS[i]), Fmt.p(pS[i])))
    }
    val vp = VProblem.V1
    for (sys in listOf(GeneratingSystem.B, GeneratingSystem.H)) {
        val hs = nsV.map { Grid.uniform(it).h }
        val eB = ArrayList<Double>(); val eS = ArrayList<Double>(); val eK = ArrayList<Double>()
        for (n in nsV) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(sys, grid)
            val op = VolterraOperator(vp.kernel, grid, QUAD)
            val solver = VFirst(vp, basis, makeFuncs("xi1", basis), op)
            eB.add(errorEh({ t -> vp.exact(t) }, solver.base().eval, grid))
            eS.add(errorEh({ t -> vp.exact(t) }, solver.sloan().eval, grid))
            eK.add(errorEh({ t -> vp.exact(t) }, solver.kulkarni().eval, grid))
        }
        val pB = ordersGen(eB, hs); val pS = ordersGen(eS, hs); val pK = ordersGen(eK, hs)
        println("[V1] basis=${sys.name} func=xi1 :")
        for (i in nsV.indices) println("   n=%3d base:E=%s p=%s | sloan:E=%s p=%s | kulk:E=%s p=%s".format(
            nsV[i], Fmt.e(eB[i]), Fmt.p(pB[i]), Fmt.e(eS[i]), Fmt.p(pS[i]), Fmt.e(eK[i]), Fmt.p(pK[i])))
    }
}

fun main() {
    println("### Sec4Tables production driver | papers/new-01 §4")
    println("### E_h = sup over 100n+1 control grid (errorEh); p_h from general max-step h.")
    println("### Model problems: M1=F2(1/(1+t+s),1/(t+1)), M2=Gaussian-blur/cosh(1.5t), M3=exp-memory/cos(3t).")
    blockH1H2()
    blockH3()
    blockH3Grid()
    blockH4Nystrom()
    blockFirstKind()
    println("\n### Sec4Tables done.")
}
