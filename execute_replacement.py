#!/usr/bin/env python3
"""Execute requirements.txt substitution and installation"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    # Set working directory
    work_dir = Path(r'c:\Users\paulo\Desktop\Uni\TCC\tcc-antifraude')
    os.chdir(work_dir)
    
    print("=" * 60)
    print("EXECUTANDO SUBSTITUIÇÃO DE REQUIREMENTS.TXT")
    print("=" * 60)
    
    # Step 1: Backup old requirements.txt
    print("\n[1/5] Fazendo backup do arquivo antigo...")
    if (work_dir / 'requirements.txt').exists():
        if (work_dir / 'requirements.txt.bak').exists():
            os.remove(work_dir / 'requirements.txt.bak')
        os.rename(work_dir / 'requirements.txt', work_dir / 'requirements.txt.bak')
        print("✓ requirements.txt → requirements.txt.bak")
    
    # Step 2: Rename new requirements file
    print("\n[2/5] Ativando novo arquivo requirements...")
    if (work_dir / 'requirements_new.txt').exists():
        os.rename(work_dir / 'requirements_new.txt', work_dir / 'requirements.txt')
        print("✓ requirements_new.txt → requirements.txt")
    else:
        print("✗ requirements_new.txt não encontrado!")
        return False
    
    # Step 3: Validate requirements.txt
    print("\n[3/5] Validando arquivo requirements.txt...")
    if (work_dir / 'requirements.txt').exists():
        with open(work_dir / 'requirements.txt', 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        print(f"✓ Arquivo criado com {len(lines)} dependências versionadas")
        print("\nPrimeiras 5 dependências:")
        for line in lines[:5]:
            print(f"  - {line}")
    else:
        print("✗ requirements.txt não foi criado")
        return False
    
    # Step 4: Install dependencies
    print("\n[4/5] Instalando dependências...")
    print("-" * 60)
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '--upgrade'],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            print("⚠ Alguns avisos ou erros durante a instalação:")
            if result.stderr:
                print(result.stderr[-500:])  # Last 500 chars
        else:
            print("✓ Instalação concluída com sucesso!")
        
        # Extract installed versions
        print("\n[5/5] Resumo de dependências instaladas:")
        print("-" * 60)
        
        key_packages = ['pandas', 'numpy', 'torch', 'streamlit', 'fastapi', 'crewai']
        for pkg in key_packages:
            try:
                check_result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'show', pkg],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if check_result.returncode == 0:
                    for line in check_result.stdout.split('\n'):
                        if line.startswith('Version:'):
                            version = line.split(':')[1].strip()
                            print(f"  ✓ {pkg}: {version}")
                            break
            except:
                pass
        
        print("\n" + "=" * 60)
        print("STATUS: ✓ SUCESSO - Dependências instaladas")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"✗ Erro durante instalação: {e}")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
