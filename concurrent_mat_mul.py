import concurrent.futures
import time
import random
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from codecarbon import OfflineEmissionsTracker

# --- Fonctions de calcul ---

def _mul_row(args):
    i, A, B, n = args
    p = len(B[0])
    row = [sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)]
    return i, row

def _transpose_row(args):
    j, A, m = args
    col = [A[i][j] for i in range(m)]
    return j, col

def matrix_multiply_mt(A, B, max_workers=None):
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    C = [[0] * p for _ in range(m)]
    # On utilise le nombre de threads de ton PC (4)
    workers = max_workers or os.cpu_count() 
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_mul_row, (i, A, B, n)) for i in range(m)]
        for future in concurrent.futures.as_completed(futures):
            idx, row = future.result()
            C[idx] = row
    return C

def matrix_transpose_mt(A, max_workers=None):
    m, n = len(A), len(A[0])
    B = [[0] * m for _ in range(n)]
    workers = max_workers or os.cpu_count()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_transpose_row, (j, A, m)) for j in range(n)]
        for future in concurrent.futures.as_completed(futures):
            idx, col = future.result()
            B[idx] = col
    return B

# --- Script principal ---

if __name__ == "__main__":
    # Listes pour stocker les données du graphique
    list_tailles = []
    list_temps = []
    list_co2 = []

    print(f"Détection CPU : {os.cpu_count()} threads disponibles.")
    
    # On initialise le tracker
    tracker = OfflineEmissionsTracker(
        project_name="3d_matrix_benchmark",
        country_iso_code="CAN",
        measure_power_secs=1, # Mesure fréquente pour plus de précision
        save_to_file=False
    )

    print("\n🚀 Démarrage des 15 itérations de benchmark...")
    
    try:
        for i in range(1, 16):
            # On augmente la taille à chaque itération : itération 1 = 100x100, ..., 15 = 240x240
            taille = 100 + (i * 10) 
            print(f"Itération {i}/15 - Taille {taille}x{taille}...", end=" ", flush=True)
            
            # Génération de matrice aléatoire
            M = [[random.uniform(0, 1) for _ in range(taille)] for _ in range(taille)]
            
            tracker.start() # On lance le tracker pour cette itération précise
            start_it = time.perf_counter()
            
            # Calculs
            Mt = matrix_transpose_mt(M)
            M_res = matrix_multiply_mt(M, Mt)
            
            duree = time.perf_counter() - start_it
            emissions = tracker.stop() # Retourne les émissions itératives
            
            # Stockage des résultats
            list_tailles.append(taille)
            list_temps.append(duree)
            list_co2.append(emissions)
            
            print(f"Terminé en {duree:.2f}s")

    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur.")

    # --- Génération du graphique 3D ---
    print("\n📊 Génération du graphique 3D...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Création de la courbe 3D
    # X = Taille de la matrice
    # Y = Temps d'exécution
    # Z = Émissions de CO2
    ax.plot(list_tailles, list_temps, list_co2, marker='o', color='green', linewidth=2, label='Consommation énergétique')

    # Étiquettes des axes
    ax.set_xlabel('Taille de la Matrice (n x n)')
    ax.set_ylabel('Temps d\'exécution (secondes)')
    ax.set_zlabel('Émissions GES (kg de CO2eq)')
    ax.set_title('Impact Environnemental du Calcul Matriciel')
    
    plt.legend()
    plt.show()