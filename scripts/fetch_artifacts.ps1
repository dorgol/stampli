# scripts/fetch_artifacts.ps1
$ErrorActionPreference = "Stop"

$indexUrl = "https://github.com/dorgol/stampli/releases/download/v0.1/chroma_db.zip"

# Uncomment if needed:
# $parquetUrl = "https://github.com/dorgol/stampli/releases/download/v0.1/reviews_with_clusters_labeled.parquet"

New-Item -ItemType Directory -Force -Path "index" | Out-Null
New-Item -ItemType Directory -Force -Path "clustering_out" | Out-Null

Invoke-WebRequest -Uri $indexUrl -OutFile "index/chroma_db.zip"
Expand-Archive -Path "index/chroma_db.zip" -DestinationPath "index" -Force
Remove-Item "index/chroma_db.zip" -Force

# Optional parquet download:
# Invoke-WebRequest -Uri $parquetUrl -OutFile "clustering_out/reviews_with_clusters_labeled.parquet"

Write-Host "`n✅ Artifacts are ready!
- index/chroma_db/ (unzipped)
- clustering_out/reviews_with_clusters_labeled.parquet (if needed)`n"
