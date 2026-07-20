# Run all Lua tests in tests/luajit3_test through luajit-rs
# Usage: .\run_tests.ps1 [--release]

param([switch]$Release)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TestDir = Join-Path $ScriptDir "tests\luajit3_test"
$Bin = if ($Release) { "target\release\luajit-rs.exe" } else { "target\debug\luajit-rs.exe" }

if (-not (Test-Path $Bin)) {
    Write-Host "Building luajit-rs..."
    if ($Release) { cargo build --release } else { cargo build }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# Tests that are expected to fail (unsupported features)
$ExpectedFail = @{
    "ffi_expr_bit.lua"        = "LL/ULL cdata literal syntax not supported"
    "ffi_expr_bit_bit64.lua"  = "LL/ULL cdata literal syntax not supported"
    "ffi_expr_bit_bitop.lua"  = "LL/ULL cdata literal syntax not supported"
}

$Passed = 0
$Failed = 0
$Skipped = @()
$SkippedReason = @{}

$TestFiles = Get-ChildItem -Path $TestDir -Filter "*.lua" | Sort-Object Name

Write-Host "Running $($TestFiles.Count) tests from luajit3_test/"
Write-Host ("=" * 60)

foreach ($File in $TestFiles) {
    $Name = $File.Name
    $BinPath = Join-Path $ScriptDir $Bin
    $TestPath = $File.FullName

    if ($ExpectedFail.ContainsKey($Name)) {
        Write-Host "  SKIP  $Name  -- $($ExpectedFail[$Name])"
        $Skipped += $Name
        continue
    }

    $result = & $BinPath $TestPath 2>&1
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        Write-Host "  PASS  $Name"
        $Passed++
    } else {
        $msg = if ($result) { $result -join " " } else { "exit code $exitCode" }
        # Truncate long messages
        if ($msg.Length -gt 120) { $msg = $msg.Substring(0, 120) + "..." }
        Write-Host "  FAIL  $Name  -- $msg"
        $Failed++
    }
}

Write-Host ("=" * 60)
Write-Host "Passed: $Passed | Failed: $Failed | Skipped: $($Skipped.Count)"

if ($Skipped.Count -gt 0) {
    Write-Host "Skipped:"
    foreach ($s in $Skipped) {
        Write-Host "  $s -- $($ExpectedFail[$s])"
    }
}

exit $Failed
