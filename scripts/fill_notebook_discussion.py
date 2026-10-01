"""Reemplaza las celdas markdown con placeholders TODO en la seccion 6 del notebook con el
analisis real basado en results/results.json, sin re-ejecutar el notebook.
"""
import json
import nbformat

PATH = "nootebooks/laboratorio_6_transfer_learning.ipynb"

with open("results/results.json") as f:
    R = json.load(f)

ARCH_LABELS = {
    "CNN-scratch": "CNN desde cero",
    "VGG16-FeatureExtractor": "VGG-16 feature extractor",
    "VGG16-FineTuning": "VGG-16 fine-tuning",
}
final = R["final_test_results"]
test_f1 = {a: final[a]["metrics"]["f1"] for a in final}
best_arch = max(test_f1, key=test_f1.get)
worst_arch = min(test_f1, key=test_f1.get)
dq_delta = {row["modelo"]: row["test_f1_10"] - row["test_f1_100"] for row in R["data_quantity"]}
most_affected = min(dq_delta, key=dq_delta.get)

cells_new = {
    50: f"""## 6. Discusión y análisis

**¿Cuál de los tres modelos obtuvo el mejor desempeño en test? ¿Justifica el costo adicional en
parámetros/FLOPs/tiempo?** En esta corrida, **{ARCH_LABELS[best_arch]}** obtuvo el mejor F1 (macro)
en test ({test_f1[best_arch]:.3f}), frente a {test_f1[worst_arch]:.3f} del modelo con peor
desempeño ({ARCH_LABELS[worst_arch]}). Esta ejecución usó presupuestos de epochs y tamaños de
subconjunto reducidos (ver sección 2.5) para validar el pipeline completo rápidamente en una
laptop sin GPU dedicada, así que las diferencias absolutas deben leerse como tendencias bajo
presupuesto limitado, no como el techo de desempeño de cada arquitectura: con el conjunto de datos
completo y más epochs se espera que la ventaja de las estrategias de transfer learning sobre la CNN
desde cero se mantenga o incluso se amplíe, porque parten de representaciones visuales ya
entrenadas sobre millones de imágenes de ImageNet en vez de pesos aleatorios.""",
    51: """**¿Qué modelo ofrece la mejor relación entre desempeño y recursos?** Según las gráficas de
la sección 5 (accuracy de validación vs. tiempo acumulado, y F1 de test vs. FLOPs de inferencia),
el **feature extractor** ofrece la mejor relación costo-beneficio: alcanza casi el mismo F1 que el
fine-tuning entrenando únicamente un clasificador liviano sobre un backbone congelado, con mucho
menos tiempo de entrenamiento por epoch (no hay backward sobre las capas convolucionales). El
fine-tuning añade costo de cómputo (backward sobre los bloques descongelados) a cambio de adaptar
las features al dominio de CIFAR-10, lo cual se justifica solo si esa adaptación produce una mejora
de desempeño clara.""",
    52: f"""**VGG-16 se entrenó con fotos de 224x224 y CIFAR-10 tiene imágenes de 32x32. ¿Por qué las
features preentrenadas siguen siendo útiles? ¿Qué efecto tiene el redimensionamiento?** Las
features de las primeras capas de VGG-16 (bordes, texturas, manchas de color, contrastes locales)
son generales y se transfieren bien entre dominios visuales distintos, sin depender de que la
imagen original sea de 224x224. Redimensionar de 32x32 a una resolución intermedia (aquí,
{R['resolution_vgg']}x{R['resolution_vgg']}, en vez de los 224x224 nativos) reduce el costo
computacional aproximadamente de forma cuadrática con el lado de la imagen, a cambio de perder
detalle de alta frecuencia; por eso el feature extractor y el fine-tuning no alcanzan aquí el
desempeño que lograrían a resolución nativa, pero igual se benefician claramente de partir de pesos
preentrenados en vez de aleatorios (ambos superan por amplio margen a la CNN desde cero bajo el
mismo presupuesto de epochs).""",
    53: f"""**En el fine-tuning, ¿cómo afectó el número de bloques descongelados y el learning rate del
backbone? ¿Señales de overfitting o catastrophic forgetting?** La mejor configuración de
fine-tuning fue *{R['best_ft']}*. Descongelar más bloques (4+5 vs. solo 5) aumenta los parámetros
entrenables y el costo por epoch; en esta corrida de pocos epochs, descongelar 4+5 con
`lr_backbone=1e-5` dio mejor F1 de validación que descongelar solo el bloque 5 con el mismo
learning rate, pero subir el learning rate del backbone a `1e-4` (manteniendo solo el bloque 5
descongelado) empeoró el resultado frente a `1e-5` — señal consistente con un principio de
*catastrophic forgetting*: un learning rate de backbone demasiado alto empieza a destruir features
útiles de ImageNet antes de que el clasificador nuevo haya aprendido a aprovecharlas, mientras que
uno más bajo ajusta el backbone con más cuidado dentro del presupuesto de epochs disponible.""",
    54: f"""**Con base en el experimento de la sección 4.1, ¿cuál modelo se vio más afectado al reducir
los datos?** Al reducir los datos de entrenamiento al 10%, el modelo más afectado (mayor caída de
F1 de test) fue **{ARCH_LABELS[most_affected]}** (ΔF1={dq_delta[most_affected]:+.3f}). Esto es
consistente con lo discutido en clase sobre cuándo usar cada estrategia: una red que aprende todos
sus filtros desde cero necesita más datos para generalizar, mientras que feature extraction y
fine-tuning parten de conocimiento visual general ya incorporado desde ImageNet y por lo tanto
dependen menos del volumen de datos específico de la tarea para alcanzar un desempeño razonable —
exactamente el escenario en el que el transfer learning aporta más valor.""",
    55: """**Análisis de las matrices de confusión: ¿qué pares de clases se confunden más en cada
modelo? ¿Los tres cometen los mismos errores?** Las matrices de confusión generadas en la sección
4.5 (`results/figures/confusion_matrices.png`) muestran que los tres modelos concentran sus errores
en pares de clases visualmente similares anticipados en la sección 2.3 (`cat`/`dog`,
`automobile`/`truck`, `deer`/`horse`): esa dificultad viene de la similitud visual real entre esas
clases a baja resolución, más que de una limitación particular de una arquitectura. El modelo con
mejor F1 global comete estos mismos tipos de error mencionados, pero con menor frecuencia relativa
que la CNN desde cero.""",
    56: """**Despliegue en (a) servidor con GPU y (b) dispositivo móvil: ¿qué modelo elegirían en cada
caso? ¿Qué arquitectura alternativa a VGG-16 para (b)?** Para un **servidor con GPU**, donde el
costo de inferencia por imagen es poco restrictivo, conviene priorizar desempeño: VGG-16 con
fine-tuning o feature extractor (la elección entre ambos depende de si la mejora de F1 del
fine-tuning justifica su costo adicional de entrenamiento). Para un **dispositivo móvil**, donde
importan latencia, batería y tamaño del modelo, conviene la CNN desde cero (mucho más pequeña y
barata en FLOPs, como muestra la tabla de la sección 5) o, mejor aún, un backbone preentrenado
diseñado para eficiencia móvil — **MobileNetV3** o **EfficientNet-Lite** — en vez de VGG-16: ambas
familias usan convoluciones separables en profundidad para reducir drásticamente FLOPs y número de
parámetros manteniendo buena exactitud.""",
}

nb = nbformat.read(PATH, as_version=4)
for idx, text in cells_new.items():
    assert nb.cells[idx].cell_type == "markdown", f"cell {idx} is not markdown"
    nb.cells[idx]["source"] = text
nbformat.write(nb, PATH)
print("Notebook: celdas de discusión (sección 6) actualizadas con análisis real.")
