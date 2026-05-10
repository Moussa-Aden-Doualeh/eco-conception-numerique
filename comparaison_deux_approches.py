import torch
import torch.nn.functional as F
import concurrent.futures
import time
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from codecarbon import OfflineEmissionsTracker

# --- 1. Implémentations ---
def _mul_row(args):
    i, A, B, n = args
    p = len(B[0])
    row = [sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)]
    return i, row

def matrix_multiply_mt(A, B):
    m, n = len(A), len(A[0])
    p = len(B[0])
    C = [[0] * p for _ in range(m)]
    workers = os.cpu_count() or 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_mul_row, (i, A, B, n)) for i in range(m)]
        for future in concurrent.futures.as_completed(futures):
            idx, row = future.result()
            C[idx] = row
    return C

def pytorch_scaled_dot_product(q, k, v):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / (d_k ** 0.5)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values

# --- 2. Benchmark ---
if __name__ == "__main__":
    tailles_a_tester = [100, 130, 160, 190, 220, 250] 
    iterations = 15 

    data_mt = {s: {"time": [], "co2": []} for s in tailles_a_tester}
    data_pt = {s: {"time": [], "co2": []} for s in tailles_a_tester}

    tracker = OfflineEmissionsTracker(country_iso_code="CAN", measure_power_secs=1, save_to_file=False)

    print(f"Démarrage du benchmark... CPU: {os.cpu_count()} threads")

    for size in tailles_a_tester:
        print(f"\nTest Taille {size}x{size}\n")
        for rep in range(iterations):
            # Python Multi-Threades
            M1 = [[random.uniform(0, 1) for _ in range(size)] for _ in range(size)]
            M2 = [[random.uniform(0, 1) for _ in range(size)] for _ in range(size)]
            tracker.start()
            t0 = time.perf_counter()
            matrix_multiply_mt(M1, M2)
            data_mt[size]["time"].append(time.perf_counter() - t0)
            data_mt[size]["co2"].append(tracker.stop())

            # PyTorch
            q, k, v = torch.randn(size, size), torch.randn(size, size), torch.randn(size, size)
            tracker.start()
            t0 = time.perf_counter()
            n_repeat = 30 
            for _ in range(n_repeat): pytorch_scaled_dot_product(q, k, v)
            data_pt[size]["time"].append((time.perf_counter() - t0) / n_repeat)
            data_pt[size]["co2"].append(tracker.stop() / n_repeat)
            print(f"  Répétition {rep+1}/{iterations} terminée", end="\r")

    # --- 3. Statistiques ---
    def get_stats(data_dict):
        sizes = sorted(data_dict.keys())
        m_time = [np.mean(data_dict[s]["time"]) for s in sizes]
        s_time = [np.std(data_dict[s]["time"]) for s in sizes]
        m_co2 = [np.mean(data_dict[s]["co2"]) for s in sizes]
        s_co2 = [np.std(data_dict[s]["co2"]) for s in sizes]
        return np.array(sizes), np.array(m_time), np.array(s_time), np.array(m_co2), np.array(s_co2)

    sz_mt, mt_t_m, mt_t_s, mt_c_m, mt_c_s = get_stats(data_mt)
    sz_pt, pt_t_m, pt_t_s, pt_c_m, pt_c_s = get_stats(data_pt)

    # --- 4. GRAPHIQUE 3D---
    fig_3d = plt.figure(figsize=(12, 8))
    ax3d = fig_3d.add_subplot(111, projection='3d')

    # Dessin des lignes moyennes
    ax3d.plot(sz_mt, mt_t_m, mt_c_m, label='Python Multi-threadé (Moyenne)', color='red', marker='o', linewidth=2)
    ax3d.plot(sz_pt, pt_t_m, pt_c_m, label='PyTorch (Moyenne)', color='blue', marker='^', linewidth=2)

    # Barres d'erreur 3D (Segments)
    for i in range(len(sz_mt)):
        ax3d.plot([sz_mt[i], sz_mt[i]], [mt_t_m[i]-mt_t_s[i], mt_t_m[i]+mt_t_s[i]], [mt_c_m[i], mt_c_m[i]], color='red', alpha=0.3)
        ax3d.plot([sz_mt[i], sz_mt[i]], [mt_t_m[i], mt_t_m[i]], [mt_c_m[i]-mt_c_s[i], mt_c_m[i]+mt_c_s[i]], color='red', alpha=0.3)
        ax3d.plot([sz_pt[i], sz_pt[i]], [pt_t_m[i]-pt_t_s[i], pt_t_m[i]+pt_t_s[i]], [pt_c_m[i], pt_c_m[i]], color='blue', alpha=0.3)
        ax3d.plot([sz_pt[i], sz_pt[i]], [pt_t_m[i], pt_t_m[i]], [pt_c_m[i]-pt_c_s[i], pt_c_m[i]+pt_c_s[i]], color='blue', alpha=0.3)

    # Amélioration lisibilité 3D
    ax3d.set_xlabel('Taille des matrices (N)')
    ax3d.set_ylabel('Temps (s)')
    ax3d.set_zlabel('Émissions GES (kg CO2eq)')
    ax3d.set_title('Impact environnemental 3D : Python MT vs PyTorch')
    ax3d.view_init(elev=20, azim=-45) # Meilleur angle
    ax3d.grid(True)
    ax3d.legend()

    plt.savefig("ges_3d_final.png")
    print("\nGraphique 3D sauvegardé.")

    # --- 5. GRAPHIQUES 2D ---
    fig_2d, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Graphique Temps
    ax1.errorbar(sz_mt, mt_t_m, yerr=mt_t_s, label='Python MT', color='red', fmt='-o', capsize=5)
    ax1.errorbar(sz_pt, pt_t_m, yerr=pt_t_s, label='PyTorch', color='blue', fmt='-^', capsize=5)
    ax1.set_xlabel('Taille (N)')
    ax1.set_ylabel('Temps moyen (s)')
    ax1.set_title('Efficacité Temporelle')
    ax1.grid(True, linestyle='--')
    ax1.legend()

    # Graphique CO2
    ax2.errorbar(sz_mt, mt_c_m, yerr=mt_c_s, label='Python MT', color='red', fmt='-o', capsize=5)
    ax2.errorbar(sz_pt, pt_c_m, yerr=pt_c_s, label='PyTorch', color='blue', fmt='-^', capsize=5)
    ax2.set_xlabel('Taille (N)')
    ax2.set_ylabel('Émissions moyennes (kg CO2eq)')
    ax2.set_title('Impact Carbone (GES)')
    ax2.grid(True, linestyle='--')
    ax2.legend()

    plt.tight_layout()
    plt.savefig("comparaison_2d_precision.png")
    print("Graphiques 2D sauvegardés.")

    # --- 6. AFFICHAGE FINAL ---
    plt.show() # Affiche toutes les fenêtres proprement à la fin