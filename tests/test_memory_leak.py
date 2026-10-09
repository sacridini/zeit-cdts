import gc
import numpy as np
import tracemalloc
import pytest
import zeit

def test_landtrendr_memory_leak():
    print("\n=== Iniciando Teste de Memory Leak (Python + C++) ===")
    
    # 1. Preparar dados simulados (40 anos, pequena grade 50x50)
    years = np.arange(1985, 2025)
    data = (np.random.rand(40, 50, 50) * 10000).astype(np.float32)
    
    # 2. Iniciar o rastreador de memória do Python
    tracemalloc.start()
    snapshot_inicial = tracemalloc.take_snapshot()
    
    # 3. Rodar a função intensiva do C++ em loop
    iteracoes = 10
    print(f"Rodando LandTrendr C++ por {iteracoes} iterações...")
    for i in range(iteracoes):
        _ = zeit.raster.run_landtrendr_array(years, data, max_segments=6)
        gc.collect()  # Forçar o Garbage Collector do Python a rodar
        
    # 4. Tirar uma nova "foto" da memória e comparar
    snapshot_final = tracemalloc.take_snapshot()
    top_stats = snapshot_final.compare_to(snapshot_inicial, 'lineno')
    
    print("\n=== Top 5 Alocações de Memória do Lado Python ===")
    for stat in top_stats[:5]:
        print(stat)
        
    # 5. Calcular diferença total
    total_size_ini = sum(stat.size for stat in snapshot_inicial.statistics('lineno'))
    total_size_fin = sum(stat.size for stat in snapshot_final.statistics('lineno'))
    
    diff_mb = (total_size_fin - total_size_ini) / (1024 * 1024)
    print(f"\nDiferença total de memória no Python: {diff_mb:.4f} MB")
    
    print("\n[!] Se houvesse um memory leak puro no C++, o AddressSanitizer (ASan) teria abortado o processo durante o teste exibindo a linha problemática no C++.")
    
    # Assert para o Pytest (tolerância de até 5MB para overhead do Python)
    assert diff_mb < 5.0, f"Possível leak de memória no Python detectado: {diff_mb:.2f} MB"
