plugins {
    kotlin("jvm") version "2.0.0"
    application
    // Coverage measurement (JetBrains Kover, Kotlin-native).
    id("org.jetbrains.kotlinx.kover") version "0.8.3"
}

repositories {
    mavenCentral()
}

dependencies {
    // Linear-algebra backend: multik with native OpenBLAS (multik-default).
    implementation("org.jetbrains.kotlinx:multik-core:0.2.3")
    implementation("org.jetbrains.kotlinx:multik-default:0.2.3")

    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

kotlin {
    jvmToolchain(21)
}

application {
    // Default run target; per-solver tasks are defined below.
    mainClass.set("solvers.fredholm.FredholmSolverKt")
}

tasks.test {
    useJUnitPlatform()
}

// --- Coverage configuration -------------------------------------------------
// Pragmatic target: ~100% on the computational core and solver logic.
// main() entry points, console/table printing and number formatting are
// reporting concerns, not algorithmic logic, so they are excluded from the
// coverage goal (covered indirectly by a single smoke run, if any).
kover {
    reports {
        filters {
            excludes {
                // main() entry points generated as <File>Kt classes.
                classes("solvers.fredholm.FredholmSolverKt")
                classes("solvers.volterra.VolterraSolverKt")
                classes("solvers.uryson.UrysonSolverKt")
                // Section-4 drivers/harness (reporting only, exercised by runSec4* tasks).
                classes("sec4.Sec4TablesKt")
                classes("vv.Sec4VVKt")
                // Pure formatting helpers.
                classes("numerics.Fmt", "numerics.FmtKt")
                classes("solvers.uryson.Fmt", "solvers.uryson.FmtKt")
                // Console table builders (reporting only).
                classes("*.Tables", "*.TablesKt")
                classes("*.HealthChecks", "*.HealthChecksKt")
            }
        }
    }
}

// Per-solver run tasks. Each solver keeps its own fun main().
tasks.register<JavaExec>("runFredholm") {
    group = "application"
    description = "Run the Fredholm solver"
    mainClass.set("solvers.fredholm.FredholmSolverKt")
    classpath = sourceSets["main"].runtimeClasspath
    maxHeapSize = "4g"
}

tasks.register<JavaExec>("runVolterra") {
    group = "application"
    description = "Run the Volterra solver"
    mainClass.set("solvers.volterra.VolterraSolverKt")
    classpath = sourceSets["main"].runtimeClasspath
    maxHeapSize = "4g"
}

tasks.register<JavaExec>("runUryson") {
    group = "application"
    description = "Run the Uryson solver"
    mainClass.set("solvers.uryson.UrysonSolverKt")
    classpath = sourceSets["main"].runtimeClasspath
}

tasks.register<JavaExec>("runBenchmark") {
    group = "application"
    description = "Run the HPC benchmark harness (time vs N; sequential vs parallel assembly speedup)"
    mainClass.set("bench.BenchmarkKt")
    classpath = sourceSets["main"].runtimeClasspath
}

tasks.register<JavaExec>("runSec4VV") {
    group = "application"
    description = "Independent V&V harness for section 4 (H1-H4 experiment matrix)"
    mainClass.set("vv.Sec4VVKt")
    classpath = sourceSets["main"].runtimeClasspath
    maxHeapSize = "4g"
}

tasks.register<JavaExec>("runSec4Tables") {
    group = "application"
    description = "Reproducible production driver for section 4 tables (H1/H2/H3/H4, F1/V1)"
    mainClass.set("sec4.Sec4TablesKt")
    classpath = sourceSets["main"].runtimeClasspath
    maxHeapSize = "4g"
}
