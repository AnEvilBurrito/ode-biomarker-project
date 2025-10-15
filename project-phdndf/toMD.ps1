function Convert-NotebooksToMarkdown {
    param(
        [string]$InputPath = ".",
        [string]$OutputPath = ".\markdown_output"
    )
    
    # Create output directory
    if (!(Test-Path $OutputPath)) {
        New-Item -ItemType Directory -Path $OutputPath -Force
    }
    
    # Get all .ipynb files
    $notebooks = Get-ChildItem -Path $InputPath -Filter "*.ipynb"
    
    if ($notebooks.Count -eq 0) {
        Write-Host "No .ipynb files found in $InputPath" -ForegroundColor Yellow
        return
    }
    
    foreach ($notebook in $notebooks) {
        try {
            $outputFile = Join-Path $OutputPath ($notebook.BaseName + ".md")
            
            # Try multiple methods to convert
            $success = $false
            
            # Method 1: Try jupyter command
            try {
                jupyter nbconvert --to markdown $notebook.FullName --output $outputFile
                if ($LASTEXITCODE -eq 0) { $success = $true }
            } catch {
                # Method 2: Try python module
                try {
                    python -m nbconvert --to markdown $notebook.FullName --output $outputFile
                    if ($LASTEXITCODE -eq 0) { $success = $true }
                } catch {
                    # Method 3: Try with full python path
                    try {
                        $pythonPath = Get-Command python | Select-Object -ExpandProperty Source
                        & $pythonPath -m nbconvert --to markdown $notebook.FullName --output $outputFile
                        if ($LASTEXITCODE -eq 0) { $success = $true }
                    } catch {
                        $success = $false
                    }
                }
            }
            
            if ($success) {
                Write-Host "✓ Converted: $($notebook.Name)" -ForegroundColor Green
            } else {
                Write-Host "✗ Failed to convert: $($notebook.Name)" -ForegroundColor Red
            }
        } catch {
            Write-Host "✗ Error converting $($notebook.Name): $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

# Run the function
Convert-NotebooksToMarkdown