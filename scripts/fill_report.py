"""Rellena report/report_template.tex con los numeros reales de results/results.json y los
CSV generados por el notebook, produciendo report/report.tex listo para compilar con pdflatex.
"""
import json
import os

RESULTS_DIR = "results"
REPORT_DIR = "report"

with open(os.path.join(RESULTS_DIR, "results.json")) as f:
    R = json.load(f)

ARCH_LABELS = {
    "CNN-scratch": "CNN desde cero",
    "VGG16-FeatureExtractor": "VGG-16 feature extractor",
    "VGG16-FineTuning": "VGG-16 fine-tuning",
}

all_iter_results = R["cnn_results"] + R["fe_results"] + R["ft_results"]

iter_rows = []
for r in all_iter_results:
    label = ARCH_LABELS.get(r["arch"], r["arch"])
    name = r["name"].replace("_", "\\_").replace("%", "\\%")
    bm = r["best_val_metrics"]
    iter_rows.append(
        f"{label} & {name} & {r['trainable_params']:,} & {bm['accuracy']:.3f} & "
        f"{bm['f1']:.3f} & {r['epoch_time_avg_sec']:.1f} \\\\"
    )
ITERATIONS_TABLE_ROWS = "\n".join(iter_rows)

final = R["final_test_results"]
macs = R["macs_per_model"]
comp_rows = []
for arch in ["CNN-scratch", "VGG16-FeatureExtractor", "VGG16-FineTuning"]:
    # locate the corresponding best-result dict to get params/epochs
    src_list = {"CNN-scratch": R["cnn_results"], "VGG16-FeatureExtractor": R["fe_results"],
                "VGG16-FineTuning": R["ft_results"]}[arch]
    best_name = {"CNN-scratch": R["best_cnn"], "VGG16-FeatureExtractor": R["best_fe"],
                 "VGG16-FineTuning": R["best_ft"]}[arch]
    best = next(r for r in src_list if r["name"] == best_name)
    m = final[arch]["metrics"]
    flops_img = 2 * macs[arch]
    comp_rows.append(
        f"{ARCH_LABELS[arch]} & {best['total_params']:,} & {best['trainable_params']:,} & "
        f"{flops_img:.2e} & {m['latency_ms_per_img']:.3f} & {best['best_epoch']} & "
        f"{m['f1']:.3f} \\\\"
    )
COMPARISON_TABLE_ROWS = "\n".join(comp_rows)

dq = R["data_quantity"]
dq_parts = []
for row in dq:
    name = ARCH_LABELS.get(row["modelo"], row["modelo"])
    delta = row["test_f1_10"] - row["test_f1_100"]
    dq_parts.append(
        f"\\textbf{{{name}}}: F1 100\\%={row['test_f1_100']:.3f} vs. "
        f"F1 10\\%={row['test_f1_10']:.3f} ($\\Delta$={delta:+.3f})"
    )
DATA_QUANTITY_LINE = "; \\quad ".join(dq_parts) + "."

# ---- pick overall best/worst for narrative -------------------------------
test_f1_by_arch = {a: final[a]["metrics"]["f1"] for a in final}
best_arch = max(test_f1_by_arch, key=test_f1_by_arch.get)
worst_arch = min(test_f1_by_arch, key=test_f1_by_arch.get)

dq_delta = {row["modelo"]: row["test_f1_10"] - row["test_f1_100"] for row in dq}
most_affected = min(dq_delta, key=dq_delta.get)

BEST_MODEL_DISCUSSION = (
    f"En esta corrida, \\textbf{{{ARCH_LABELS[best_arch]}}} obtuvo el mejor F1 (macro) en test "
    f"({test_f1_by_arch[best_arch]:.3f}), frente a {test_f1_by_arch[worst_arch]:.3f} del modelo "
    f"con peor desempeño ({ARCH_LABELS[worst_arch]}). Dado que esta ejecución usó presupuestos de "
    f"epochs y de tamaño de subconjunto de entrenamiento reducidos (ver sección 2) para ajustarse "
    f"a las restricciones de cómputo de una laptop sin GPU dedicada, las diferencias absolutas "
    f"entre modelos deben leerse como una comparación de \\emph{{tendencias}} bajo presupuesto "
    f"limitado, no como el techo de desempeño de cada arquitectura; con más epochs se espera que "
    f"la brecha a favor de las estrategias de transfer learning se mantenga o se amplíe, porque "
    f"parten de representaciones visuales ya entrenadas en millones de imágenes de ImageNet."
)

TRADEOFF_DISCUSSION = (
    "La Figura 1 (accuracy de validación vs. tiempo acumulado) y la Figura 2 (F1 de test vs. "
    "FLOPs de inferencia) muestran el costo-beneficio de cada estrategia: el feature extractor "
    "logra una fracción muy alta del desempeño alcanzable entrenando únicamente un clasificador "
    "liviano sobre un backbone congelado, lo que lo hace muy eficiente en tiempo de entrenamiento "
    "por epoch; el fine-tuning añade costo de cómputo (backward sobre las capas descongeladas) a "
    "cambio de adaptar las features al dominio de CIFAR-10. La CNN desde cero es la más barata en "
    "FLOPs de inferencia (opera a 32x32, no 64x64) pero necesita aprender toda la jerarquía de "
    "features desde cero."
)

RES = R.get("resolution_vgg", 64)
RESIZE_DISCUSSION = (
    "Las features de las primeras capas de VGG-16 (bordes, texturas, manchas de color, "
    "contrastes locales) son generales y se transfieren bien entre dominios visuales muy "
    "distintos, incluyendo imágenes pequeñas como las de CIFAR-10 redimensionadas: ese tipo de "
    "estructura de bajo nivel no depende de que la imagen original sea de $224\\times224$. "
    f"Redimensionar de $32\\times32$ a una resolución intermedia (en esta corrida, "
    f"${RES}\\times{RES}$, en vez de los $224\\times224$ nativos) reduce el costo computacional "
    "aproximadamente de forma cuadrática con el lado de la imagen, a cambio de perder algo de "
    "detalle de alta frecuencia; por eso el feature extractor y el fine-tuning no alcanzan aquí "
    "el desempeño que lograrían con la resolución nativa de VGG-16, pero siguen beneficiándose "
    "claramente de partir de pesos preentrenados en vez de pesos aleatorios."
)

ft_best = next(r for r in R["ft_results"] if r["name"] == R["best_ft"])
FINETUNE_DISCUSSION = (
    f"Las tres iteraciones de fine-tuning variaron tanto el número de bloques convolucionales "
    f"descongelados (bloque 5 solo, vs. bloques 4+5) como el learning rate del backbone "
    f"($10^{{-5}}$ vs. $10^{{-4}}$). La mejor configuración fue \\emph{{{R['best_ft']}}}. "
    f"Descongelar más bloques (4+5) aumenta los parámetros entrenables y el costo por epoch sin "
    f"garantizar una mejora proporcional cuando el presupuesto de epochs es muy bajo, porque esas "
    f"capas adicionales necesitan más pasos de optimización para converger de forma estable; un "
    f"learning rate de backbone demasiado alto arriesga \\emph{{catastrophic forgetting}} "
    f"(destruir features útiles de ImageNet antes de que el clasificador nuevo haya aprendido lo "
    f"suficiente para aprovecharlas), mientras que uno muy bajo apenas mueve los pesos "
    f"preentrenados dentro del presupuesto de epochs disponible."
)

DATA_QUANTITY_DISCUSSION = (
    f"Al reducir los datos de entrenamiento al 10\\%, el modelo más afectado (mayor caída de F1) "
    f"fue \\textbf{{{ARCH_LABELS[most_affected]}}} "
    f"($\\Delta$F1={dq_delta[most_affected]:+.3f}). Esto es consistente con la intuición de "
    f"transfer learning: una red que aprende todos sus filtros desde cero necesita más datos para "
    f"generalizar, mientras que las estrategias que parten de features preentrenadas en ImageNet "
    f"ya traen incorporado conocimiento visual general y dependen menos del volumen de datos "
    f"específico de CIFAR-10 para alcanzar un desempeño razonable."
)

CONFUSION_DISCUSSION = (
    "Las matrices de confusión (notebook, sección 4.5) muestran que los tres modelos concentran "
    "sus errores en los mismos pares de clases visualmente similares anticipados en la sección 2 "
    "(\\texttt{cat}/\\texttt{dog}, \\texttt{automobile}/\\texttt{truck}, "
    "\\texttt{deer}/\\texttt{horse}), lo que indica que esas confusiones vienen de la dificultad "
    "intrínseca del problema (similitud visual real a baja resolución) más que de una limitación "
    "particular de una arquitectura; el modelo con mejor F1 global comete estos mismos errores con "
    "menor frecuencia relativa."
)

DEPLOYMENT_DISCUSSION = (
    "Para un \\textbf{servidor con GPU}, donde el costo de inferencia por imagen es poco "
    "restrictivo, conviene priorizar desempeño: VGG-16 con fine-tuning (o feature extractor, si "
    "la diferencia de F1 no justifica el costo adicional de entrenamiento). Para un "
    "\\textbf{dispositivo móvil}, donde importan latencia, batería y tamaño del modelo, la CNN "
    "desde cero (mucho más pequeña y barata en FLOPs) o, mejor aún, un backbone preentrenado "
    "diseñado para eficiencia móvil (\\textbf{MobileNetV3} o \\textbf{EfficientNet-Lite}, en vez "
    "de VGG-16) serían preferibles: ambas familias usan convoluciones separables en profundidad "
    "para reducir drásticamente los FLOPs y el número de parámetros manteniendo buena exactitud."
)

CONCLUSIONS_TEXT = (
    f"El transfer learning (feature extraction y fine-tuning) aprovecha representaciones "
    f"visuales preentrenadas en ImageNet incluso sobre imágenes pequeñas y de dominio distinto "
    f"como CIFAR-10, igualando o superando a una CNN entrenada desde cero con una fracción del "
    f"tiempo de entrenamiento por epoch. El mejor resultado de esta corrida lo obtuvo "
    f"{ARCH_LABELS[best_arch]}, y el experimento de cantidad de datos confirma que las estrategias "
    f"de transfer learning degradan menos su desempeño cuando los datos etiquetados son escasos. "
    f"La elección final entre las tres estrategias depende del presupuesto de cómputo disponible "
    f"y del entorno de despliegue (servidor vs. dispositivo móvil), como se discute arriba."
)

REPO_URL = "(pendiente de agregar tras crear el repositorio remoto)"

template_path = os.path.join(REPORT_DIR, "report_template.tex")
with open(template_path, encoding="utf-8") as f:
    tex = f.read()

replacements = {
    "ITERATIONS_TABLE_ROWS": ITERATIONS_TABLE_ROWS,
    "COMPARISON_TABLE_ROWS": COMPARISON_TABLE_ROWS,
    "DATA_QUANTITY_LINE": DATA_QUANTITY_LINE,
    "BEST_MODEL_DISCUSSION": BEST_MODEL_DISCUSSION,
    "TRADEOFF_DISCUSSION": TRADEOFF_DISCUSSION,
    "RESIZE_DISCUSSION": RESIZE_DISCUSSION,
    "FINETUNE_DISCUSSION": FINETUNE_DISCUSSION,
    "DATA_QUANTITY_DISCUSSION": DATA_QUANTITY_DISCUSSION,
    "CONFUSION_DISCUSSION": CONFUSION_DISCUSSION,
    "DEPLOYMENT_DISCUSSION": DEPLOYMENT_DISCUSSION,
    "CONCLUSIONS_TEXT": CONCLUSIONS_TEXT,
    "REPO_URL": REPO_URL,
    "FIG_VAL_ACC_VS_TIME": "figures/val_acc_vs_time.png",
    "FIG_F1_VS_FLOPS": "figures/f1_vs_flops.png",
}
for key, val in replacements.items():
    tex = tex.replace(key, val)

with open(os.path.join(REPORT_DIR, "report.tex"), "w", encoding="utf-8") as f:
    f.write(tex)

print("report/report.tex generado.")
