# Script para ativar venv com encoding UTF-8 correto (PowerShell)

Write-Host " Configurando encoding Python para UTF-8..." -ForegroundColor Cyan
$env:PYTHONIOENCODING = "utf-8"

Write-Host " Ativando venv..." -ForegroundColor Cyan
& ".\venv\Scripts\Activate.ps1"

Write-Host ""
Write-Host " Virtual environment ativada!" -ForegroundColor Green
Write-Host "Python version:" -ForegroundColor Yellow
python --version

Write-Host ""
Write-Host "Próximos comandos:" -ForegroundColor Yellow
Write-Host "  ✓ Testar tudo:        python teste_sistema_completo.py" -ForegroundColor Gray
Write-Host "  ✓ Processar dados:    python src/preprocessamento/carregar_paysim.py" -ForegroundColor Gray
Write-Host "  ✓ Executar sistema:   python run_simulation.py" -ForegroundColor Gray
Write-Host "  ✓ Dashboard:          streamlit run src/dashboard/app.py" -ForegroundColor Gray
Write-Host ""
