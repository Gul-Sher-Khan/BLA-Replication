param(
  [string]$ProjectsRoot = "$PSScriptRoot\..\projects",
  [string]$AnonRoot = "$PSScriptRoot\..\anon_results",
  [string]$ResultsRoot  = "$PSScriptRoot\..\results",
  [int]$Bins = 20,
  [int]$MaxParallelProjects = 0,
  [switch]$SkipExisting = $true,
  [bool]$IncludeRQ5 = $true,
  [string[]]$RQ5Projects = @("cassandra", "flink", "groovy", "ignite", "openstack", "qt"),
  [string[]]$RQ5Methods = @("gmm_joint_2_global", "cla_gmm_2_no_cluster", "gmm_joint_2_cluster", "cla_gmm_2"),
  [int[]]$RQ5QValues = @(5, 10, 20, 50),
  [int[]]$RQ5Components = @(1, 2, 3),
  [int]$RQ5RfRuns = 30,
  [int]$RQ5MaxWorkers = 0,
  [bool]$SkipRQ5Existing = $true
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..").Path
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Missing venv python: $Py" }

function Resolve-RepoRelativePath([string]$PathValue, [string]$FallbackRelative) {
  if ([string]::IsNullOrWhiteSpace($PathValue)) {
    return [System.IO.Path]::GetFullPath((Join-Path $Root $FallbackRelative))
  }

  if ([System.IO.Path]::IsPathRooted($PathValue)) {
    return [System.IO.Path]::GetFullPath($PathValue)
  }

  $fromCwd = [System.IO.Path]::GetFullPath($PathValue)
  if (Test-Path $fromCwd) {
    return $fromCwd
  }

  return [System.IO.Path]::GetFullPath((Join-Path $Root $PathValue))
}

$ProjectsRoot = Resolve-RepoRelativePath $ProjectsRoot "projects"
$AnonRoot = Resolve-RepoRelativePath $AnonRoot "anon_results"
$ResultsRoot = Resolve-RepoRelativePath $ResultsRoot "results"

if (-not (Test-Path $ProjectsRoot)) {
  throw "ProjectsRoot not found: $ProjectsRoot"
}
if (-not (Test-Path $AnonRoot)) {
  throw "AnonRoot not found: $AnonRoot"
}

# Limit BLAS threads
$env:OMP_NUM_THREADS       = "1"
$env:MKL_NUM_THREADS       = "1"
$env:NUMEXPR_NUM_THREADS   = "1"
$env:OPENBLAS_NUM_THREADS  = "1"

$AnonScript = Join-Path $Root "scripts-anon\anonymizer-stats.py"
$JointAnonScript = Join-Path $Root "scripts-anon\anonymizer_gmm_joint.py"
$Compare = Join-Path $Root "scripts-results\compare_privacy_and_utility.py"
$Stats = Join-Path $Root "scripts-results\stats_tests.py"
$Viz = Join-Path $Root "scripts-results\visualize_results.py"
$RQ5 = Join-Path $Root "scripts-results\run_rq5_sensitivity.py"
foreach ($p in @($AnonScript, $JointAnonScript, $Compare, $Stats, $Viz, $RQ5)) {
  if (-not (Test-Path $p)) { throw "Missing: $p" }
}

New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null

$skipExistingEnabled = [bool]$SkipExisting
$projectWorker = {
  param(
    [string]$ProjDir,
    [string]$ProjName,
    [string]$Py,
    [string]$AnonScript,
    [string]$JointAnonScript,
    [string]$Compare,
    [string]$AnonRoot,
    [string]$ResultsRoot,
    [int]$Bins,
    [bool]$SkipExistingEnabled
  )

  $ErrorActionPreference = "Stop"

  # Keep per-job numeric libraries single-threaded to avoid oversubscription in parallel mode.
  $env:OMP_NUM_THREADS = "1"
  $env:MKL_NUM_THREADS = "1"
  $env:NUMEXPR_NUM_THREADS = "1"
  $env:OPENBLAS_NUM_THREADS = "1"

  function Map-ProjectToAnonDir([string]$name) {
    switch ($name) {
      "cassandra" { "data_split_apache_cassandra_train" }
      "flink"     { "data_split_apache_flink_train" }
      "groovy"    { "data_split_apache_groovy_train" }
      "ignite"    { "data_split_apache_ignite_train" }
      "openstack" { "data_split_openstack_train" }
      "qt"        { "data_split_qt_train" }
      default { $null }
    }
  }

  function Get-LatestAnonCsv([string]$dir) {
    if (-not (Test-Path $dir)) { return $null }
    $items = Get-ChildItem -File $dir -Filter *.csv |
      Where-Object { $_.Name -notmatch "graph_csv_dump" -and $_.Name -notmatch "non_anon" } |
      Sort-Object -Property LastWriteTime -Descending
    return $items | Select-Object -First 1
  }

  function Invoke-PythonChecked([string]$stepName, [string]$exe, [string[]]$cmdArgs) {
    & $exe @cmdArgs
    if ($LASTEXITCODE -ne 0) {
      throw "${stepName} failed for ${ProjName} (exit code $LASTEXITCODE)"
    }
  }

  function Invoke-GenerateCsvChecked(
    [string]$stepName,
    [string]$exe,
    [string[]]$cmdArgs,
    [string]$outCsv,
    [bool]$forceRegenerate
  ) {
    if ($forceRegenerate -and (Test-Path $outCsv)) {
      Remove-Item -Path $outCsv -Force
    }

    Invoke-PythonChecked $stepName $exe $cmdArgs
    if (Test-Path $outCsv) {
      return
    }

    Write-Output "  Retry ${stepName}: output was not created on first attempt"
    Invoke-PythonChecked "${stepName} (retry)" $exe $cmdArgs
    if (-not (Test-Path $outCsv)) {
      throw "${stepName} finished but output is missing: $outCsv"
    }
  }

  function Get-GeneratedAnonCsv([string]$methodName, [string]$projectName) {
    $dir = Join-Path (Join-Path $AnonRoot $methodName) $projectName
    return (Join-Path $dir ("{0}.csv" -f $methodName))
  }

  $generatedMethods = @(
    @{ Name = "cla_gmm_2"; Script = $AnonScript; Method = "gmm_2" },
    @{ Name = "cla_gmm_2_no_cluster"; Script = $AnonScript; Method = "gmm_2_no_cluster" },
    @{ Name = "gmm_joint_2_global"; Script = $JointAnonScript; Method = "gmm_joint_2_global" },
    @{ Name = "gmm_joint_2_cluster"; Script = $JointAnonScript; Method = "gmm_joint_2_cluster" }
  )

  $graphMethods = @("gen", "k_da_anon", "random_add_delete", "random_switch")

  $inputCsv = Join-Path $ProjDir "input.csv"
  if (-not (Test-Path $inputCsv)) {
    return "Skip ${ProjName}: input.csv missing"
  }

  Write-Output "==> Anonymizing $ProjName"
  foreach ($m in $generatedMethods) {
    $outCsv = Get-GeneratedAnonCsv $m.Name $ProjName
    $outDir = Split-Path -Parent $outCsv
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    if ($SkipExistingEnabled -and (Test-Path $outCsv)) {
      Write-Output "  Skip $($m.Name): exists"
      continue
    }

    $cmdArgs = @(
      $m.Script,
      "--input-csv", "$inputCsv",
      "--output-csv", "$outCsv",
      "--method", $m.Method
    )
    if ($m.Script -eq $AnonScript) {
      $cmdArgs += @("--bins", "$Bins")
    }
    Invoke-GenerateCsvChecked "Anonymizer $($m.Name)" $Py $cmdArgs $outCsv (-not $SkipExistingEnabled)
  }

  $anonDirName = Map-ProjectToAnonDir $ProjName
  if (-not $anonDirName) {
    return "Skip ${ProjName}: no anon_results mapping"
  }

  $variants = New-Object System.Collections.Generic.List[string]
  foreach ($m in $generatedMethods) {
    $outCsv = Get-GeneratedAnonCsv $m.Name $ProjName
    if (-not (Test-Path $outCsv)) { throw "Missing generated output: $outCsv" }
    $variants.Add("$($m.Name)=$outCsv")
  }

  foreach ($gm in $graphMethods) {
    $methodDir = Join-Path (Join-Path $AnonRoot $anonDirName) $gm
    $latest = Get-LatestAnonCsv $methodDir
    if (-not $latest) { throw "Missing graph method CSV for $ProjName/$gm" }
    $variants.Add("$gm=$($latest.FullName)")
  }

  $variantArgs = @()
  foreach ($v in $variants) {
    $variantArgs += "--variant"
    $variantArgs += $v
  }

  $outDir = Join-Path $ResultsRoot $ProjName
  $outJson = Join-Path $outDir "results.json"
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null

  Write-Output "==> Comparing $ProjName"
  $compareArgs = @(
    $Compare,
    "--baseline", "$inputCsv",
    "--rf-runs", "30",
    "--out-dir", "$outDir",
    "--out-json", "$outJson",
    "--project-name", "$ProjName",
    "--metrics-set", "all"
  ) + $variantArgs
  Invoke-PythonChecked "Comparison" $Py $compareArgs

  return "Done ${ProjName}"
}

$projects = Get-ChildItem -Directory $ProjectsRoot
if (-not $projects -or $projects.Count -eq 0) {
  throw "No project directories found in $ProjectsRoot"
}

$parallelLimit = $MaxParallelProjects
if ($parallelLimit -le 0) {
  $parallelLimit = [Math]::Min([Environment]::ProcessorCount, 4)
}
if ($parallelLimit -lt 1) {
  $parallelLimit = 1
}

if ($parallelLimit -le 1 -or $projects.Count -le 1) {
  Write-Host "==> Processing projects sequentially" -ForegroundColor Cyan
  foreach ($proj in $projects) {
    $messages = & $projectWorker $proj.FullName $proj.Name $Py $AnonScript $JointAnonScript $Compare $AnonRoot $ResultsRoot $Bins $skipExistingEnabled
    foreach ($line in $messages) {
      Write-Host $line
    }
  }
}
else {
  Write-Host "==> Processing projects in parallel (max $parallelLimit)" -ForegroundColor Cyan
  $activeJobs = @()

  function Drain-JobOutput([object[]]$jobs) {
    foreach ($j in $jobs) {
      $lines = Receive-Job -Job $j -ErrorAction SilentlyContinue
      foreach ($line in $lines) {
        Write-Host "[$($j.Name)] $line"
      }
    }
  }

  function Throw-JobFailure([object]$job) {
    $jobErrors = @($job.ChildJobs | ForEach-Object { $_.Error | ForEach-Object { $_.ToString() } })
    $reason = ($job.ChildJobs | ForEach-Object { $_.JobStateInfo.Reason } | Where-Object { $_ } | ForEach-Object { $_.ToString() }) -join " | "
    if ($jobErrors.Count -gt 0) {
      throw "Project job failed: $($job.Name). $reason`n$($jobErrors -join [Environment]::NewLine)"
    }
    throw "Project job failed: $($job.Name). $reason"
  }

  foreach ($proj in $projects) {
    while ($activeJobs.Count -ge $parallelLimit) {
      $done = Wait-Job -Job $activeJobs -Any -Timeout 10
      Drain-JobOutput $activeJobs

      if (-not $done) {
        $running = ($activeJobs | ForEach-Object { "$($_.Name):$($_.State)" }) -join ", "
        Write-Host "  still running: $running" -ForegroundColor DarkGray
        continue
      }

      if ($done.State -ne "Completed") {
        Remove-Job -Job $done -Force
        Throw-JobFailure $done
      }
      Remove-Job -Job $done -Force
      $activeJobs = @($activeJobs | Where-Object { $_.Id -ne $done.Id })
    }

    $job = Start-Job -Name ("project_{0}" -f $proj.Name) -ScriptBlock $projectWorker -ArgumentList @(
      $proj.FullName,
      $proj.Name,
      $Py,
      $AnonScript,
      $JointAnonScript,
      $Compare,
      $AnonRoot,
      $ResultsRoot,
      $Bins,
      $skipExistingEnabled
    )
    Write-Host "  started: $($job.Name)" -ForegroundColor DarkGray
    $activeJobs += $job
  }

  while ($activeJobs.Count -gt 0) {
    $done = Wait-Job -Job $activeJobs -Any -Timeout 10
    Drain-JobOutput $activeJobs

    if (-not $done) {
      $running = ($activeJobs | ForEach-Object { "$($_.Name):$($_.State)" }) -join ", "
      Write-Host "  still running: $running" -ForegroundColor DarkGray
      continue
    }

    if ($done.State -ne "Completed") {
      Remove-Job -Job $done -Force
      Throw-JobFailure $done
    }
    Remove-Job -Job $done -Force
    $activeJobs = @($activeJobs | Where-Object { $_.Id -ne $done.Id })
  }
}

Write-Host "==> Running stats tests" -ForegroundColor Cyan
& $Py $Stats --results-root "$ResultsRoot"
if ($LASTEXITCODE -ne 0) { throw "Stats tests failed (exit code $LASTEXITCODE)" }
Write-Host "==> Creating visualizations" -ForegroundColor Cyan
& $Py $Viz --results-root "$ResultsRoot"
if ($LASTEXITCODE -ne 0) { throw "Visualization generation failed (exit code $LASTEXITCODE)" }
if ($IncludeRQ5) {
  Write-Host "==> Running RQ5 sensitivity" -ForegroundColor Cyan
  $rq5Args = @(
    $RQ5,
    "--projects-root", "$ProjectsRoot",
    "--anon-root", "$AnonRoot",
    "--results-root", "$ResultsRoot",
    "--rf-runs", "$RQ5RfRuns",
    "--max-workers", "$RQ5MaxWorkers",
    "--methods"
  ) + @($RQ5Methods | ForEach-Object { "$_" }) + @(
    "--projects"
  ) + @($RQ5Projects | ForEach-Object { "$_" }) + @(
    "--q-values"
  ) + @($RQ5QValues | ForEach-Object { "$_" }) + @(
    "--gmm-components"
  ) + @($RQ5Components | ForEach-Object { "$_" })

  if ($SkipRQ5Existing) {
    $rq5Args += "--skip-existing-eval"
  }
  if (-not $skipExistingEnabled) {
    $rq5Args += "--force-anon"
  }

  & $Py @rq5Args
  if ($LASTEXITCODE -ne 0) { throw "RQ5 sensitivity failed (exit code $LASTEXITCODE)" }
}
Write-Host "All done. Results in $ResultsRoot" -ForegroundColor Green
