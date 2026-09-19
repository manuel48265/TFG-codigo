# Criterios de cribado (inclusión / exclusión / borderline)

Pool de partida: `data/merged/all_papers_deduped.csv` (121 candidatos, 4 queries
—`q1_falsification`, `q2_membership`, `q3_dagitty`, `q4_dag_inference`—, 2021-2026,
Scopus+OpenAlex deduplicados). Ver `src/tfg_codigo/Research/queries_research.py` para las
queries exactas.

## Tarea que define el tema

El tema no es "grafos causales" en general, es: **dado un grafo (DAG/CPDAG/PAG) candidato y
datos, decidir si el grafo puede rechazarse / validarse / testarse frente a esos datos**
(falsificación, test de pertenencia a una clase de equivalencia de Markov, test de
implicaciones testables tipo d-separación, etc.).

La generalidad del método (cuántos supuestos necesita) **no es parte de la definición de la
tarea**. Un método que resuelve el mismo problema bajo hipótesis extra sigue siendo la misma
tarea con más restricciones — eso se etiqueta como faceta (qué supuestos añade), no se usa como
motivo de exclusión. Este punto es el que resuelve el caso Orchard/SUGAR más abajo.

## Criterios de inclusión (IC) — basta con cumplir uno

- **IC1**: propone o evalúa un método que, dado un grafo causal candidato y datos, produce un
  test, score o procedimiento de decisión para aceptarlo/rechazarlo/falsificarlo.
- **IC2**: aborda test de pertenencia a una clase de equivalencia de Markov, o test de
  implicaciones testables (d-separación, independencias condicionales) de un grafo dado.
- **IC3**: el test/score viene acompañado de alguna noción de control de error (falsos
  positivos, potencia, garantías tipo "si el grafo es correcto, no se rechaza") — no es solo
  una discusión cualitativa de si un grafo "parece" razonable.

## Criterios de exclusión (EC)

EC = "no cuenta como estudio primario de la revisión (no aporta método nuevo de test/
falsificación)". **EC no significa descartar el paper por completo** — ver la etiqueta secundaria
de utilidad de datos justo debajo, que es independiente de esta decisión.

- **EC1**: causal discovery / structure learning puro (aprender el DAG desde cero), o construcción
  de un DAG por conocimiento de dominio (ej. para selección de confusores), sin un paso explícito
  de validar/falsificar un grafo ya dado con datos. (Si el paper aprende Y valida, y la parte de
  validación es un aporte propio, no se excluye por esto — pasa a IC1.)
- **EC2**: estimación de efecto causal asumiendo el grafo correcto, sin cuestionar el grafo.
- **EC3**: paper de aplicación pura (usa un método de validación/construcción ya existente en un
  caso de dominio) sin aporte metodológico nuevo al test en sí.
- **EC4**: survey/tutorial sin método nuevo de test — se anota aparte para contexto de la
  taxonomía, no cuenta como estudio primario.
- **EC5**: falso positivo léxico. "DAG" y "hypothesis testing" son términos muy sobrecargados
  (DAGs de scheduling/compiladores/workflows, tests de hipótesis no relacionados con grafos
  causales). Se espera que sea la categoría de exclusión más numerosa dado el diseño de las
  queries — no indica un problema con las queries, es el ruido esperado de q3/q4.

## Etiqueta secundaria: utilidad de datos (eje independiente de include/exclude)

Relevancia-core (¿aporta método nuevo de test/falsificación?) y utilidad-de-datos (¿aporta un
dataset, grafo real, o ground truth usable para validar métodos — tuyos o de la literatura?) son
dos preguntas distintas. Un paper EC1/EC3 para la revisión puede seguir siendo valioso por esta
segunda vía — es el caso del DAG de reumatología (21 nodos construidos con juicio experto, sobre
datos administrativos reales): EC1/EC3 para el cribado, pero **dato_util: sí** si en algún momento
sirve como ejemplo/dataset para el diseño experimental. Orchard es el caso inverso-completo:
entra limpio como core (IC1+IC3) *y además* aporta datasets reales (PetShop, Causal Chambers) —
las dos etiquetas en positivo a la vez, por eso "encaja más limpio" que un EC1/EC3 con solo dato.

**Cómo registrar**: en `etiquetas.tsv`/`clasification.md`, añadir una columna aparte
`dato_util (sí/no/quizás)` junto a la decisión de include/exclude/borderline — no colapsar las
dos preguntas en una sola etiqueta. Un "EC3, dato_util: sí" no se descarta del todo: se archiva
para la fase de diseño experimental aunque no cuente como estudio primario del estado del arte.

## Borderline / wildcard (Wohlin 2014/2022)

**Definición precisa (afinada tras el caso Orchard/SUGAR): el eje que define borderline es el
régimen de D, no cualquier hipótesis extra.** Un paper es **borderline** cuando cumple el
mecanismo central de IC1/IC2 (usa el grafo + datos para producir una decisión sobre el grafo)
pero el supuesto adicional que añade recae sobre **la naturaleza de los datos D**: exige que D no
sea puramente observacional i.i.d., sino intervencional, heterogéneo, mixto, o con alguna otra
estructura adicional sobre cómo se generaron los datos (ej. Orchard: necesita un evento outlier
bajo soft-intervention, no solo observacional).

**Segundo eje añadido 2026-09-19: formalismo/notación.** Un paper puede tener el régimen de D más
limpio posible (i.i.d. observacional, sin exigir no-confusión) y aun así ser borderline si **no usa
el lenguaje DAG/SCM del resto del pool** — p.ej. Cornish et al. (Digital Twins) formaliza en
potential outcomes (Rubin/Robins), no en grafos, aunque el propio paper señale que *"could have
used the alternative framework of directed acyclic graphs and structural causal models (Pearl,
2009)"*. Encajarlo junto a Eulig/Kook/SUGAR exige trabajo de traducción de notación — eso basta
para borderline aunque el eje D sea el caso base. Corrige la entrada de Digital Twins más abajo,
que decía "NO borderline" atendiendo solo al eje D.

Un supuesto extra que recae sobre **otra cosa que no es el régimen de D ni el formalismo** — forma
funcional, modelo de ruido, dominio de aplicación, tipo de variable — **no activa borderline por sí
solo**: se anota igual como faceta de la matriz de métodos, pero el paper entra limpio por IC1/IC2 (ej.
SUGAR: exige datos i.i.d. entre sujetos igual que el resto del pool; su supuesto extra es el
modelo de ruido — ANM — no el régimen de D, así que no es borderline, es un include directo con
una faceta de forma funcional).

**Regla de decisión**: si el mecanismo central coincide con IC1/IC2 → **incluir** siempre,
sea borderline o no. Lo único que cambia con el eje de D es el ETIQUETADO: "borderline (régimen
de D no observacional)" vs. "include directo (facetas sobre forma funcional/ruido/dominio)". Solo
se excluye un borderline cuando, quitando la envoltura del caso particular, no queda ninguna
tarea de test/falsificación de grafo (ahí sí es EC1/EC2 disfrazado). Ante duda genuina tras leer
el abstract completo, se prefiere incluir y resolver en la pasada de texto completo — sesgo a
favor de recall en esta fase, igual que en el resto del proceso de búsqueda (ver metodología de
búsqueda acotada, no exhaustiva).

**Para el resto del pool, este es el nivel de escrutinio a aplicar por defecto** (abstract +
esta regla de dos ejes), no la lectura completa a fondo que se hizo con Orchard — esa profundidad
se reserva para papers que acaben siendo centrales en la propuesta/experimental propios, no para
el cribado general.

## Nota: adaptar i.i.d. → heterogéneo para el diseño experimental (abierto, no de cribado)

Distinto del cribado de arriba: esto es sobre si Orchard es *usable* en un entorno experimental
propio, no sobre si entra en la revisión. Búsqueda (no en el pool de las 4 queries, tema
distinto) para localizar el paper que convierte datos i.i.d. en heterogéneos vía clustering:

- **Mejor encaje: Heterogeneous Risk Minimization (HRM)** — Liu, Hu, Cui, Li, Shen, ICML 2021,
  arXiv:2105.03818 (extensión "Kernelized HRM", arXiv:2110.12425). Texto completo confirma
  clustering explícito: "we propose a novel clustering method for heterogeneity identification
  from heterogeneous data [sin labels de entorno]... we cluster the data points according to the
  relationship between Ψ and Y, for which we use P(Y|Ψ) as the cluster centre" → toma un dataset
  i.i.d. sin partición de entorno y produce una partición multi-entorno (pseudo-entornos).
- Candidato relacionado con encaje temático pero mecanismo distinto: **EIIL** (Creager, Jacobsen,
  Zemel, ICML 2021, arXiv:2010.07249) — infiere particiones de entorno, pero vía optimización
  adversarial/bi-nivel, no clustering explícito tipo k-means/GMM.
- Búsqueda en Budhathoki/Ikram (root-cause analysis, la línea que cita Orchard) no encontró nada
  de clustering para derivar la muestra outlier desde i.i.d. — esa literatura asume el evento
  anómalo ya dado, igual que Orchard.

**Brecha sin cerrar**: HRM produce pseudo-*entornos* (particiones para invariant learning bajo
distribution shift), no un evento *outlier* puntual bajo soft-intervention sobre un R desconocido
como pide el formalismo de Orchard (Ec. 2). Que el clustering de HRM sirva para generar el tipo
de muestra que Orchard necesita es una adaptación no trivial y no verificada — no asumir que
transfiere. Si se persigue esto para el diseño experimental, es una pieza de trabajo propia (o
una pregunta a registrar en REGISTRO_NOVEDAD si aplica), no una cita que resuelve el problema tal
cual.

**Actualización (2026-09-19), motivada por Karlsson & Krijthe (arriba)**: buscados candidatos más
recientes de clustering i.i.d.→multi-entorno, con la pregunta explícita de si sirven para producir
los "entornos heterogéneos" que el método de Karlsson necesita como input.

- **CCSL** — Chen, Wu, Cai, Chen, Hao, "CCSL: A Causal Structure Learning Method from Multiple
  Unknown Environments" (IEEE TNNLS, arXiv:2111.09666). Verificado abstract/intro completos.
  Cluster de muestras por similitud de mecanismo causal vía Chinese Restaurant Process
  (no-paramétrico, sin fijar nº de clusters), simultáneo con aprendizaje de estructura causal por
  grupo. Identificabilidad bajo supuesto linear no-Gaussiano.
- **HCL** — Li, Zhang, Wang (Tsinghua), "Interpretable Clustering with Adaptive Heterogeneous
  Causal Structure Learning in Mixed Observational Data" (arXiv:2509.04415, sept. 2025, envío a
  IEEE TNNLS). Verificado abstract/intro completos. Infiere clusters latentes + su estructura
  causal asociada de forma no supervisada, **sin requerir labels de entorno, orden temporal,
  intervenciones ni conocimiento previo** — el más directo a la pregunta original. Validado en
  datos de perturbación single-cell (biología), no en el dominio de este TFG.

**Advertencia conceptual (la razón por la que esto sigue sin cerrar la brecha, no solo
actualizarla)**: tanto CCSL como HCL —igual que HRM— **recuperan heterogeneidad latente que ya
existe pero no está etiquetada** en los datos (los datos vienen de una mezcla real de mecanismos
distintos, p.ej. sujetos/condiciones distintas, solo que sin la etiqueta de a qué entorno
pertenece cada fila). Esto es un problema distinto de **fabricar** entornos a partir de datos
genuinamente i.i.d. (generados por un único mecanismo invariante): si el dato de partida es
realmente i.i.d., no hay heterogeneidad real que recuperar, y clusterizarlo (k-means, GMM, CRP,
o cualquiera de estos métodos) produce particiones arbitrarias sin significado causal — aplicar
después un test tipo Karlsson (que depende de que los cambios entre entornos reflejen cambios
*reales* e independientes en los mecanismos, principio ICM) sobre esas pseudo-particiones no
tiene garantías, porque la premisa del test (entornos = mecanismos que de verdad varían de forma
independiente) no se cumpliría. **Conclusión: sigue sin haber literatura que resuelva "convertir
i.i.d. genuino en multi-entorno"; lo que existe (HRM, EIIL, CCSL, HCL) resuelve el problema vecino
de "recuperar entornos ya latentes en una mezcla sin etiquetar".** Si los datos de tu diseño
experimental son de verdad i.i.d. de un único mecanismo, ninguno de estos métodos aplica
legítimamente; si sospechas que hay heterogeneidad real pero no etiquetada (distintos sujetos,
distintas condiciones de recogida), HCL es el candidato más directo a probar.

**HRM/EIIL/CCSL/HCL no entran en el pool de la revisión** (son invariant learning / environment
inference / causal discovery por cluster — EC1: no validan/falsifican un grafo dado, aprenden o
particionan para otra tarea). Son referencia de apoyo para la sección de diseño experimental, no
estudios primarios del cribado.

**Confirmado leyendo Orchard a fondo (Ap. F.1, §4.6, §5, conclusión) — no hay forma actual de
usarlo fuera del régimen "outlier ya dado":**

- En los dos experimentos con datos reales (PetShop: 18 eventos anómalos; Causal Chambers: 58
  escenarios intervencionales) el outlier **no se detecta**, viene como etiqueta ground-truth
  del propio dataset (Hardt et al. 2024 / Gamella et al. 2025). El paper nunca corre su propio
  detector de anomalías — por eso tampoco discute sensibilidad a una detección propia.
- **El paper no tiene ningún resultado sobre qué pasa si la muestra dada no es un outlier
  genuino** (R vacío, o es solo ruido normal mal etiquetado). La Ec. 2 (P̃ corrupta) se toma como
  premisa del problema, no como algo que el test verifica. Esto confirma tu sospecha: es un gap
  real y no un detalle que se te esté escapando.
- Sí hay un resultado, pero para un problema más estrecho: identificar mal **qué variables** son
  la raíz cuando el outlier es real pero R se infiere en vez de asumirse conocido. §4.6, cita
  literal: "one cannot simply infer the root cause(s) with the outlier sample..., using a root
  cause analysis algorithm..., and then apply one of our tests with the inferred root cause(s).
  This introduces selection bias, so... our tests no longer valid." Su arreglo (Proposición 24,
  "leave-k-out": tomar el máximo p-valor sobre todos los subconjuntos de tamaño ≤k) solo exige
  conocer una cota k≥|R|, pero sigue asumiendo que el outlier existe de verdad.
- Búsqueda de trabajo posterior (mismo grupo — Janzing tiene "Formalizing and Falsifying Causal
  Pathways of Rare Events", arXiv 2605.31254, mayo 2026 — y terceros): nada que adapte el
  framework a i.i.d. sin outlier dado, ni que analice la sensibilidad a una detección propia.
  **Actualización 2026-09-19: el paper de Haghighat & Janzing (2605.31254) ya se localizó y
  clasificó (ver "Lote GS" más abajo) — sigue sin cerrar esta brecha concreta (i.i.d.→heterogéneo),
  pero ya no está pendiente de verificar si existe.**

**Conclusión práctica**: hoy no hay ninguna adaptación publicada de Orchard a un entorno
experimental estándar (i.i.d., sin evento anómalo etiquetado externamente). Para la revisión
sigue incluido (documenta el estado del arte, con esta limitación como parte de su ficha). Para
el diseño experimental propio, la opción honesta es: (i) usar datasets con outliers ya
etiquetados por terceros (como hace el propio paper) en vez de generarlos, o (ii) si se quiere
generar un evento outlier sintético a partir de datos i.i.d. para poder comparar contra ground
truth conocido, es trabajo propio no verificado por nadie — y habría que caracterizar primero qué
pasa con el FPR si ese outlier sintético resulta "mal elegido", precisamente el resultado que el
paper no da.

**Gap confirmado con cita exacta (no es interpretación forzada):** el propio paper reconoce el
riesgo pero no lo analiza. §3, nota al pie 1, justo tras definir H0^R: *"Note that this assumes
that the outlier sample is not selected because it is anomalous and then also used for testing,
as this would introduce selection bias."* Mecanismo de por qué se rompe: el Lema 1 (base de las
Proposiciones 2-5, control de FPR) exige que la muestra outlier esté genuinamente "distributed
according to P"; si en vez de eso se toma el máximo/cola de un pool i.i.d. (procedimiento
mecánico de detección sobre los mismos datos, sin fuente externa), el punto resultante sigue la
ley de un estadístico de orden, no la de P marginal — rompe la premisa Exp(1) de la que dependen
todas las garantías. La Proposición 24 (leave-k-out, §4.6) NO cubre esto: corrige el problema
distinto de no saber qué variables son R cuando el outlier SÍ es genuino, pero sigue asumiendo la
premisa de la nota 1. **No hay teorema, cota ni experimento en el paper que caracterice
FPR/potencia bajo este escenario — es una omisión real, confirmada, no solo sospechada.** Esto es
justo lo que hace inviable usar Orchard sobre datos i.i.d. "puros" sin etiqueta externa: cualquier
forma mecánica de "encontrar" el outlier (umbral, score de anomalía, orden estadístico) cae
exactamente en el caso que la nota 1 excluye sin resolver. Candidato razonable a hueco de
novedad propio si se quiere perseguir (caracterizar el comportamiento del test bajo outlier
autoseleccionado), no algo que se pueda resolver citando otro paper.

**Valoración crítica del alcance real (tras revisar Introducción/Related Work, §1-§2):** el
paper nunca articula un mecanismo de detección de anomalías independiente del propio estadístico
de test S(Xi|PAi^Ĝ) — en sus tres ejemplos reales (PetShop, Causal Chambers, y las menciones de
cloud computing/manufactura/medicina personalizada en la motivación) la etiqueta "esto es
anómalo" viene siempre heredada de una fuente externa (benchmark pre-etiquetado o intervención de
diseño experimental conocida), nunca de aplicar el propio método sobre un pool i.i.d. genérico.
Cero menciones de confounders no observados en todo el texto — no es un objetivo del paper. **El
nicho real de aplicabilidad es "sistemas con anomalías ya señalizadas externamente"
(monitorización de producción con benchmark etiquetado, intervenciones conocidas), no un test
general de consistencia grafo-datos observacionales i.i.d.**, pese a que el título/abstract se
presentan con esa generalidad. Dentro de ese nicho el resultado es real (primer test frecuentista
con garantías de FPR para el caso), pero la faceta a anotar en la matriz de métodos debe reflejar
esta condición explícitamente: *requiere anomalía pre-señalizada por fuente externa al test; no
extiende a detección de outliers sobre i.i.d. genérico sin romper las garantías (nota 1, §3)*.

**Cierre: no hay criterio reproducible para tagging de outliers (verificado con lectura
cuidadosa, no solo búsqueda de términos).** El único elemento previo al grafo es el score
marginal (tomado de Budhathoki et al. 2022, §3) que usa una función de anomalía τ arbitraria —
el propio paper dice "in practice one should choose a feature map suited to their particular
application, as it must be large when the corresponding variable should be considered 'extreme'
in that application": es un molde abierto que cada practicante rellena a su criterio, no un
algoritmo cerrado. En el Apéndice F.1, la selección de la muestra para los 18 eventos de PetShop
y los 58 escenarios de Causal Chambers se describe solo como "we take a single sample" — sin
especificar timestamp/criterio dentro de la ventana del evento. Tampoco Budhathoki et al. 2022
(la fuente del score) resuelve el paso de detección: también asume la fila anómala como dado de
entrada. **Conclusión: el método resuelve "dado un evento anómalo ya identificado por otro medio,
¿es consistente con este grafo candidato?", pero no da ni cita un criterio reproducible para el
paso previo de decidir qué cuenta como evento anómalo.** Esto es una limitación de utilizabilidad
real (dos aplicaciones independientes del método podrían no coincidir en qué muestra es "el
outlier"), distinta de (pero relacionada con) el gap de selection bias ya documentado arriba —
juntas hacen que el método, aunque formalmente correcto en su nicho, no sea aplicable "de fábrica"
por un tercero sin decisiones no especificadas por los autores.

## Casos aplicados

- **Eulig, Mastakouri, Blöbaum, Hardt & Janzing — "Toward Falsifying Causal Graphs Using a
  Permutation-Based Test" (AAAI 2025, DOI 10.1609/aaai.v39i25.34881; preprint arXiv:2305.09565,
  2023)**: **ya estaba en el pool** (`q1_falsification`, Scopus + OpenAlex, coincide con la frase
  exacta "falsifying causal graphs" — no hacía falta snowballing para este). Mismo grupo que
  Orchard (Janzing común a ambos, Amazon Research Tübingen) — la nota de "búsqueda de trabajo
  posterior del mismo grupo" en la ficha de Orchard debería haber encontrado este predecesor.
  Verificado texto completo (AAAI, 21 págs.).

  **Es la respuesta más directa del pool a "test conservador, integrable en pipeline
  automático"**: parte de que las métricas de consistencia grafo-datos existentes (conteo de
  violaciones de independencias condicionales implicadas por el grafo, vía Local Markov
  Condition) dan un número **absoluto** sin base de comparación — el practicante no sabe si "3
  violaciones de 20" es aceptable. Su aporte: construir una **baseline por permutación de nodos**
  del propio grafo candidato, y testar si el nº de violaciones observado es significativamente
  menor que el de la baseline aleatoria (test de permutación con nivel de significancia α
  explícito, p.ej. 1%/5%). **Confirma empíricamente el problema de calibración que ya conocías
  por tu propio validador BIC**: su Fig. 2 muestra que el type-I error real de los tests de
  independencia condicional estándar (KCI, GCM, correlación parcial) se dispara muy por encima
  del α nominal a medida que crece el tamaño del conjunto de condicionamiento D — y diseñan el
  test de permutación precisamente para ser robusto a esa descalibración (al ser una comparación
  *relativa* grafo-vs-baseline, no un p-valor crudo de un solo CI test).

  Mecanismo: **IC1 + IC2 + IC3 directos** — dado un grafo candidato y datos observacionales
  i.i.d. (condición de Markov causal, sin exigir fidelidad/faithfulness), produce un test con
  nivel de significancia fijo que rechaza/no rechaza. Régimen de datos: **i.i.d. observacional
  estándar, NO borderline** (mismo eje que SUGAR, distinto de Orchard/Karlsson que exigen datos
  heterogéneos/outlier). Validado en datos reales de monitorización de microservicios en la nube
  (latencias, N≈432, aprox. i.i.d.) y biología (Sachs et al.), además de sintéticos.
  **Decisión: incluir — no borderline.** Candidato fuerte a paper central: es el único del pool
  visto hasta ahora que además *demuestra* el problema de calibración de los CI tests subyacentes
  y da una solución (comparación relativa vía permutación) en vez de solo asumir que el α nominal
  se sostiene.

- **Karlsson, R.K.A. & Krijthe, J.H. — "Putting Causal Identification to the Test: Falsification
  using Multi-Environment Data" (NeurIPS 2023 Workshop on Causal Representation Learning,
  OpenReview FNlfCBZ1tD) y su extensión "Falsification of Unconfoundedness by Testing Independence
  of Causal Mechanisms" (ICML 2025, PMLR 267, arXiv:2502.06231)**: encontrado *fuera* del pool de
  las 4 queries — confirma en vivo el argumento de vocabulario ya documentado en
  `informacion_relevante.md` (el título usa "falsification"/"multi-environment", no encaja con las
  frases exactas de `q1_falsification`). Verificado texto completo de la versión ICML 2025
  (arXiv). Mecanismo: dado un conjunto de datos observacionales de múltiples entornos heterogéneos,
  proponen un algoritmo de dos etapas que **falsifica la ausencia de confusión no medida**
  (unconfoundedness) explotando que la confusión no medida hace que los mecanismos causales
  observados aparenten estar correlacionados entre sí — bajo el supuesto de independencia de
  mecanismos causales (ICM, Janzing et al. 2012). El paper original (NeurIPS 2023 ws, no
  verificado texto completo — bloqueo de bot en OpenReview) cubre además back-door, front-door e
  IV, no solo unconfoundedness.
  **IC1 + IC3 directos**: "detects these dependencies with high statistical power while
  controlling false positives" — control de FPR y potencia explícito, exactamente el tipo de
  garantía que pide `IC3`. **Decisión: incluir — BORDERLINE (eje D: requiere datos de múltiples
  entornos heterogéneos, no i.i.d. de una sola fuente)**, mismo eje que Orchard. Sin parametric
  assumptions (no asume forma funcional) — faceta distinta a SUGAR (que sí asume ANM). Candidato
  fuerte a paper central de la revisión: es justo la intersección de "falsificar una condición de
  identificación dada" + garantías estadísticas formales que el resto del pool (`q3`/`q4`) rara
  vez tiene junto con `EC1` descartado limpiamente (aquí no hay discovery, la estructura causal se
  da, se falsifica un supuesto sobre ella).

- **López de Prado — "Causal Factor Investing: Can Factor Investing Become Scientific?" (SSRN
  4205613, 2022/2023, Cambridge Elements) / "Causality and Factor Investing: A Primer" (López de
  Prado & Zoonekynd, CFA Institute Research Foundation, 2025)**: fuera del pool de las 4 queries
  (encontrado por fuera, no por Scopus/OpenAlex). **Evaluación en dos capas:**

  - **El "Primer" (2025)** sí verificado completo (PDF). Mecanismo: algoritmo PC (causal
    discovery) sobre retornos de 85 modelos de riesgo Barra para *aprender* un grafo causal entre
    factores, usado para elegir controles correctos (evitar colliders) — **EC1 directo**: aprende
    el grafo desde cero, no valida/falsifica un grafo ya dado. Estructura de
    "primer"/best-practices para profesionales, cercano a EC4. `dato_util: sí` (ejemplo real de
    colliders fuera de dominio sanitario, útil para la introducción).

  - **La monografía SSRN/Cambridge Elements** (no verificada completa — SSRN bloquea fetch
    automatizado; solo se ha visto §3.3 "The Falsification Step" pegado por el usuario). Ese
    fragmento describe un procedimiento — dado un mecanismo causal teorizado, (i) deducir
    implicaciones falsables por modus tollens, (ii) testear cada implicación con una de tres
    estrategias (`P[Y=y|do(X=x)]=P[Y=y]`; `E[Y|do(X=x)]≠y`; o deducir asociaciones del mecanismo y
    usar ML para mostrar que no existen) — pero el procedimiento en sí reutiliza herramientas ya
    existentes (do-calculus de Pearl, mediation analysis estándar, ML para test de asociación) sin
    aportar un test/score nuevo con garantías propias. **Decisión: EC4** — el paper argumenta que
    la industria de factor investing *debería* adoptar causalidad/falsificación como práctica, no
    aporta un método de test nuevo (criterio del usuario tras ver más contexto del documento que el
    fragmento aislado). Igual que el Primer, entra solo como `dato_util: sí` /
    referencia de motivación (ilustra que el problema de falsación de grafos se reconoce como gap
    incluso fuera de epidemiología), no como estudio primario. **Excluir del pool core.**

- **Orchard — "Falsifying Causal Graphs With Outlier Events" (arXiv 2607.12145, 2026,
  q1_falsification, ya en el pool)**: leído el texto completo. Mecanismo central = test
  estadístico de si un grafo candidato Ĝ es consistente con la propagación de un evento outlier,
  con control de falsos positivos y garantías de potencia → IC1 + IC3 directos.

  Hipótesis extra confirmada (Sec. 3, Ec. 2): el método necesita datos **heterogéneos**, no
  i.i.d. — un dataset "no anómalo" (usado como referencia para estimar las distribuciones
  marginales/condicionales, Ap. F.1) más al menos una muestra outlier genuina generada bajo una
  soft-intervention desconocida sobre un conjunto de variables "root cause" R no vacío
  (distribución "corrupta" P̃ = ∏_{j∈R} P̃(Xj|PAj) · ∏_{i∉R} P(Xi|PAi)). El paper funciona con una
  sola muestra outlier, pero esa muestra tiene que ser un evento anómalo real, no derivado.

  **Sobre tu pregunta de si la estrategia de adaptar datos i.i.d. a heterogéneos vía clustering
  se podría aplicar aquí: el paper de Orchard no la menciona.** Se buscó explícitamente
  (clustering, inyección sintética de outliers, perturbación) y no hay ninguna mención — los
  outliers de los experimentos se toman tal cual de datasets ya anómalos (los 18 eventos
  etiquetados de PetShop, los 58 escenarios intervencionales de Causal Chambers), nunca
  derivados de un pool i.i.d. por los autores. El paper trata la necesidad de un evento outlier
  real como una restricción de disponibilidad de datos, no como algo a sintetizar ("puede ser
  prohibitivamente caro obtener múltiples muestras de un evento anómalo, y en medicina
  personalizada puede ser imposible"). Es decir: si esa técnica de clustering existe, es de otro
  paper (probablemente de la literatura de root-cause analysis que Orchard cita como origen del
  principio), y habría que verificar por separado si es válida para pasar del régimen i.i.d. al
  régimen que Orchard necesita — no se puede asumir que transfiere sin comprobarlo.

  Restricción secundaria (droppable per Sec. 4 mediante variantes de test, pero requerida para
  los tests principales): el conjunto de root causes R tiene que conocerse o al menos acotarse
  en tamaño.

  **Decisión: incluir — BORDERLINE (eje D: régimen no observacional).** El supuesto extra recae
  sobre D mismo (necesita un evento outlier genuino bajo soft-intervention, no i.i.d.), no sobre
  forma funcional o dominio → activa la etiqueta borderline por definición. Faceta: requiere
  evento heterogéneo real, R conocido/acotado, sin criterio reproducible de tagging (ver
  discusión completa arriba). Restricción más fuerte que la de SUGAR precisamente porque recae
  en el eje de D, no solo en la forma funcional.
- **SUGAR — "Testing Directed Acyclic Graph via Structural, Supervised and Generative
  Adversarial Learning" (Shi, Zhou, Li, JASA 2024, q4_dag_inference, ya en el pool)**: leído el
  texto completo (via LSE Research Online, eprints.lse.ac.uk/119446). Mecanismo = test de
  hipótesis de si un DAG candidato es consistente con los datos → IC1 + IC3 directos. La
  hipótesis extra confirmada, tal como la habías identificado: el modelo es un **Additive Noise
  Model** — Sec. 2.1, Ec. (1): "X_j = f_j(X_PA_j) + ε_j", con {f_j} funciones continuas
  arbitrarias (no lineales) pero {ε_j} ruido independiente y aditivo. El abstract vende la
  generalidad de f_j, pero eso no incluye ruido multiplicativo/heterocedástico ni ruido
  dependiente de los padres — la identificabilidad se apoya en la teoría ANM de Peters et al.
  2014. Segunda restricción: dependencia temporal solo *estacionaria* dentro de cada sujeto (i.i.d.
  entre sujetos), grafos no estacionarios o edges pasado→futuro quedan fuera ("future research",
  Apéndice A.5). **Decisión: incluir — NO borderline** (opera sobre D observacional i.i.d. entre
  sujetos, igual que el resto del pool; el supuesto extra recae en forma funcional/ruido, no en
  el régimen de D). Faceta: asume ANM (ruido aditivo independiente) + estructura temporal
  estacionaria intra-sujeto. Aparece también en `title_only_merges_review.csv` (fusión
  scopus/openalex solo por título, no por DOI) — revisar aparte que sea una fusión correcta, es un
  chequeo de deduplicación, no de cribado temático.

## Lote GS (2026-09-19) — clasificación de `GS/*.pdf`

18 PDFs descargados manualmente vía búsqueda semántica en Google Scholar. Cribado a nivel
abstract+introducción (pdftotext, 15-30 págs.), profundidad Orchard/SUGAR solo donde se indica.
Cuatro (`06136-EuligE`, `2502.06231v2`, `2602.08340v1`, `causal-factor-investing`) ya estaban
clasificados en entradas anteriores de este documento — no se repiten aquí salvo nota de
actualización.

### Include — core

- **`30_Putting_Causal_Identificati.pdf`** = Karlsson, Creastă & Krijthe, NeurIPS 2023 ws (versión
  original, ya referenciada arriba pero no verificada completa hasta ahora). Confirma que cubre
  **back-door, front-door e IV** (la extensión ICML 2025 solo cubre unconfoundedness) — actualizar
  la entrada de arriba: el paper original es más amplio que su extensión, citar ambos por
  separado, no como si uno reemplazara al otro. Autor añadido: Ștefan Creastă (co-autor
  estudiante, TU Delft).

- **`Causal_falsification_of_digital_twins.pdf`** — Cornish, Taufiq, Doucet & Holmes, **"Causal
  Falsification of Digital Twins"**, JMLR 27 (2026), 1-52 (aceptado, versión final, no preprint).
  **Leído a fondo (texto completo, 52 págs.: intro, formalización, sección 3 completa, sección 5
  completa, resultados 6.2-6.4, discusión 7)** — mismo nivel de profundidad que Orchard/SUGAR.

  **Mecanismo, con la lógica formal completa**:
  1. Definen "twin interventionally correct" (Def. 1/Prop. 2): la distribución de las trayectorias
     simuladas por el twin bajo una secuencia de acciones coincide con la distribución real de las
     trayectorias del proceso bajo esas mismas acciones — el twin es, en esencia, un modelo causal
     candidato (SCM), no necesariamente restringido a un grafo discreto.
  2. **Teorema 3 (el "problema fundamental de la inferencia causal", Holland 1986, formalizado en
     su notación)**: prueban que **certificar** que el twin es correcto (mostrar que una hipótesis
     H tal que "H verdadera ⟹ twin correcto" es cierta) es imposible desde datos observacionales
     sin supuestos fuertes (típicamente no-confusión). Es la versión formal, con teorema y prueba,
     de algo que el resto del pool asume implícitamente.
  3. **Pivote a falsificación (§3.3, inspirado explícitamente en Popper)**: en vez de H con
     "H⟹correcto", usan la hipótesis dual "twin correcto ⟹ H", e intentan mostrar que H es falsa.
     Rechazar H falsifica el twin sin necesitar el supuesto de no-confusión — **funciona incluso
     con confusión no medida arbitraria**, algo que ningún otro paper del pool logra (todos los
     demás, incluido Karlsson, necesitan algún supuesto sobre el mecanismo de confusión o sobre el
     régimen de datos).
  4. **Método concreto (Teorema 6, §5)**: cotas causales longitudinales (generalizan las cotas
     clásicas de Manski 1990 al caso condicional/longitudinal, sin supuestos extra) que acotan una
     cantidad Q (definida por los datos observacionales) en [Q_lo, Q_up]; se compara contra la
     cantidad análoga Q̂ que predice el twin. Si Q̂ cae fuera del intervalo, H se rechaza y el twin
     queda falsificado — con un modo de fallo concreto y accionable (sobreestima/subestima).
  5. **Garantía estadística — la más fuerte del pool completo**: control de error tipo I **exacto,
     de muestra finita**, vía desigualdad de Hoeffding, **sin aproximaciones asintóticas** —
     explícitamente más fuerte que el resto (Eulig, Kook, Karlsson dependen de tests asintóticos o
     de nivel nominal aproximado). Corrección por comparaciones múltiples con el método de Holm.

  **Cita textual que responde directamente a tu pregunta sobre "test conservador"** (§1, casi
  palabra por palabra lo que describiste): *"our goal in this paper is to obtain a methodology
  that is always sound, even possibly at the expense of being conservative: we prefer not to draw
  any conclusion about the accuracy of the twin at all than to draw some conclusion that is
  potentially misleading."*

  **Validación empírica real y de gran escala**: dataset MIMIC-III (UCI, pacientes sépticos reales)
  contra el Pulse Physiology Engine (simulador fisiológico open-source). Rechazan 10 magnitudes
  fisiológicas (cloro, sodio, potasio, temperatura, calcio, glucosa, paCO2, HCO3, presión sistólica)
  con dirección del error identificada (ej. Pulse subestima cloro y temperatura, sobreestima sodio
  y glucosa) — salida accionable, no solo un veredicto binario. Además muestran (§6.4) un caso
  concreto donde comparar ingenuamente twin-vs-datos observacionales da una conclusión **opuesta e
  incorrecta** a la del test causal — ejemplo listo para citar como motivación de por qué la
  corrección causal importa y no basta con comparar salidas directamente.

  **Límites que ellos mismos documentan (§7)**: asume que no hay *distribution shift* entre el
  momento del test y el despliegue (si el proceso real cambia, el conjunto de twins "correctos"
  cambia con él); en su caso de estudio, la elección de subgrupos B0:t fue *ad hoc*, no
  automatizada; las cotas son conservadoras por construcción — se podrían estrechar con supuestos
  adicionales (ej. sensibilidad a confusión) a costa de robustez, mismo trade-off documentado en
  todo el pool.

  **Decisión: incluir — NO borderline, candidato a paper central de la revisión, probablemente el
  encaje más limpio y mejor fundamentado de todo el pool** (IC1+IC3 directos, garantía de muestra
  finita en vez de asintótica, funciona bajo confusión arbitraria, i.i.d. observacional estándar).
  Única salvedad a documentar como faceta: el objeto candidato es un modelo/SCM completo
  (trayectorias bajo intervención), no un grafo-esqueleto discreto — mismo nivel de generalidad que
  Orchard/SUGAR, no una limitación nueva.

- **`faller24a.pdf`** — Faller, Vankadara, Locatello, Mastakouri & Janzing, **"Self-Compatibility:
  Evaluating Causal Discovery without Ground Truth"** (AISTATS 2024, PMLR 238). Mismo clúster de
  autores que Orchard/Eulig (Faller, Mastakouri, Janzing — Amazon Research Tübingen). Verificado
  más profundo. Mecanismo: dado un grafo (candidato, típicamente salida de un algoritmo de
  discovery), testear su **auto-compatibilidad**: aprender el grafo marginal sobre distintos
  subconjuntos de variables y comprobar si las implicaciones interventionales que se derivan
  (p.ej. p(y|do(x))=p(y|x) vs. p(y|do(x))=p(y)) son mutuamente consistentes; una incompatibilidad
  **falsifica** el grafo. Cita y construye explícitamente sobre **Eulig et al. (2023)** ("Eulig et
  al. propose to reject causal graphs...") y sobre **Su & Henckel (2022)** (test estadístico para
  rechazar H0 = "todos los subconjuntos son compatibles", con p-valores específicos por arista) —
  **backward snowballing confirmado en vivo**: este paper es una arista real de la red de citas
  desde el seed set de `q1_falsification`. Los propios autores son honestos sobre el alcance:
  pasar el test es **condición necesaria, no suficiente** ("passing such compatibility tests is
  only a necessary criterion for good performance... we do not think that such [sufficiency]
  guarantees can [be given]") — más débil que Orchard/Eulig en garantías, pero sigue siendo
  **IC1** (produce una decisión de rechazo basada en datos) vía el test heredado de Su & Henckel
  (p-valores por arista, umbral α). **Decisión: incluir — NO borderline** (opera sobre grafos ya
  discovered, no exige datos especiales). **Nuevo candidato a revisar por separado: Su & Henckel
  (2022)**, citado como la fuente del test formal subyacente.

- **`kook-iwsm-2025.pdf`** — Lucas Kook (WU Vienna), **"Falsifying causal models via nonparametric
  conditional independence testing"**, IWSM 2025 (workshop con actas, peer-reviewed pero no
  top-tier). Coincide casi palabra por palabra con la definición de tarea del TFG: *"Given a
  putative causal model and an observational dataset, we first extract testable conditional
  independence relations from the causal model. We then nonparametrically test those relations,
  potentially falsifying the causal model, while controlling Type I error."* Aplicado a datos
  reales de citometría de flujo de proteínas (Sachs et al. — el mismo dataset que usa Eulig et
  al., posible punto de comparación cruzada). **IC1 + IC2 + IC3 directos, NO borderline**. Uno de
  los encajes más limpios y directos del lote entero.

- **`2605.31254v1.pdf`** — Haghighat & Janzing, **"Formalizing and Falsifying Causal Pathways of
  Rare Events"** (arXiv 2605.31254, mayo 2026). Es el paper que la ficha de Orchard ya anticipaba
  ("es plausible que aún no exista" — ya existe). Mismo grupo/autor que Orchard (Janzing).
  Formaliza una noción de "pathway causal" (cadena root-cause→target, no el grafo completo) para
  eventos raros/outliers, define sus implicaciones testables, y estudia bajo qué condiciones esas
  implicaciones dependen solo de la abstracción del pathway y no del grafo completo subyacente.
  **IC1/IC2-adjacent, mismo eje borderline que Orchard** (eventos raros/outliers, no i.i.d. puro).
  Unidad de análisis: pathway/cadena, no grafo completo — más cercano a López de Prado §3.3 en
  granularidad pero con aparato estadístico formal (no solo argumentativo). **Decisión: incluir —
  BORDERLINE**, y además releer para ver si responde algo de la brecha i.i.d.→heterogéneo (poco
  probable dado que sigue asumiendo eventos raros ya identificables, pero no descartado sin leer).

### Excluir del pool core

- **`19725_CausalFusion_Integrating.pdf`** — "CausalFusion: Integrating LLMs and Graph
  Falsification for Causal Discovery" (ICLR 2026, **envío anónimo bajo revisión, no aceptado
  todavía**). Usa "graph falsification tests" como módulo interno de refinamiento dentro de un
  bucle de *discovery* LLM+estadístico, evaluado por SHD contra ground truth — **EC1 directo**, y
  no define un test nuevo (usa alguno existente como caja negra, no se especifica cuál en la
  introducción). `dato_util`: no.

- **`NeurIPS-2022-falsification-before-extrapolation...pdf`** — Hussain, Oberst, Shih & Sontag,
  "Falsification before Extrapolation in Causal Effect Estimation" (NeurIPS 2022). Rechaza
  **estimadores de efecto** (no grafos) comparando contra efectos de validación de un RCT, antes
  de extrapolar a subpoblaciones no observadas — con garantías de cobertura formales (más fuerte
  técnicamente que Dang et al., pero mismo objeto equivocado: efecto, no grafo). **EC2** —
  mismo patrón que Dang et al., aunque con más rigor estadístico. `dato_util`: sí, buen contraste
  metodológico para la introducción ("falsificar antes de confiar" aplicado a efectos vs. a
  grafos — mismo principio, objeto distinto).

- **`Causal_AI_for_XRPL_GossipSub_network_configuration.pdf`** — aplicación de causal discovery a
  configuración de red blockchain (XRPL/GossipSub). **EC1 + EC3** (discovery + aplicación pura de
  dominio, sin aporte metodológico al test). `dato_util`: no (dominio muy nicho, poco ilustrativo
  frente al ejemplo de López de Prado).

- **`gold_standard.pdf`** — Kubinec, "Holistic Causal Learning with Causal Graphs... Study Design
  and Preregistration in the Social Sciences" (2024). Sobre preregistro + entropía de Jaynes para
  puntuar el diseño de un estudio — no testea un grafo candidato frente a datos. **EC4** (marco
  metodológico distinto: preregistro, no falsificación post-hoc). `dato_util`: no.

- **`s11423-023-10241-0.pdf`** — Weidlich, Hicks & Drachsler, "Causal reasoning with causal graphs
  in educational technology research" (2024). Paper teórico/tutorial que introduce DAGs a
  investigadores de tecnología educativa, sin método nuevo. **EC4**, mismo patrón que el Primer de
  López de Prado pero en otro dominio. `dato_util`: no.

- **`721797.pdf`** — Buchholz & Raidl, "A Falsificationist Account of Artificial Neural Networks".
  Filosofía de la ciencia sobre el entrenamiento de redes neuronales como proceso popperiano —
  **no trata grafos causales en absoluto**. Falso positivo léxico de la búsqueda semántica
  (coincide en "falsification"+"machine learning", tema distinto). **EC5.** `dato_util`: no.

- **`Resonance_Density_as_a_Non_External_Dema.pdf`** — filosofía de la ciencia sobre el problema
  de demarcación (Popper/Kuhn/Lakatos), usa "graph-theoretic" en sentido metafórico (red de
  estructuras epistémicas), nada que ver con DAGs causales ni con datos. Autor independiente,
  citando como "compañeros" papers sobre "Symbolic Monism and the Horizon of Consciousness" —
  señal de calidad baja / fuera de literatura académica convencional. **EC5, excluir sin más
  revisión.** `dato_util`: no.

- **`preprints202607.1126.v1.pdf`** — Shukla, Atul, Mishra & Das, "Automated Scientific Discovery
  via Causal Falsification and Formal Multi-Agent Verification" (Preprints.org, **no peer-
  reviewed**, institución poco conocida). El componente de "causal falsification" mapea
  afirmaciones a un SCM y testea identificabilidad — no se ve un test estadístico nuevo con
  control de error propio, es un envoltorio multi-agente LLM alrededor de maquinaria de
  identificabilidad ya existente. Señales de calidad débiles (no revisado por pares, acumula
  buzzwords: multi-agente, verificación formal, demostración automática de teoremas). **Tentativo:
  excluir por calidad/no aporta método propio (EC4-ish)** — marcar como revisado con baja
  confianza, no cerrado con la misma seguridad que el resto del lote.

- **`stephen-savas-nu-cs-2025-13.pdf`** — Savas (Northwestern, MS thesis 2025), **"Online Causal
  Graph Falsification with CC-RuLSIF"**. **Leído a fondo** (intro+background completos, enfoque
  técnico Cap. 2, resultados Cap. 3, límites §2.7).

  **Tarea (§1.1, cita textual)**: *"Assume that a domain expert has created a causal graph (user
  supplied graph) G0 for a multivariate stream {Xt}... is G0 still compatible with the most recent
  data, or has at least one edge X → Y become invalid?"* — misma pregunta que el resto del pool,
  versión streaming: no valida si el grafo fue correcto alguna vez, **monitoriza si sigue
  siéndolo** con el tiempo.

  **Mecanismo**: para cada arista `(U→X)` define `H0^X: p_t(x|u) = q(x|u)` (mecanismo condicional
  actual = el de referencia), estima la divergencia vía razón de densidades α-relativa (RuLSIF)
  entre una ventana de referencia fija y una ventana de test deslizante, y lanza alarma si supera
  un umbral `δ_X` **bootstrapped para controlar el error Tipo I** (Apéndice A.4), con objetivo
  explícito de FPR < 10%.

  **Régimen de datos — tercer eje, distinto de i.i.d. estático y de heterogéneo/intervencional**:
  stream "potentially non-stationary"; solo la ventana de referencia inicial se asume estacionaria
  (§2.7, Asunción 4). Responde directamente a la pregunta de "verdad terreno dinámica" que no
  cubría el resto del pool.

  **Comparación empírica real** (Cap. 3) contra dos baselines: el test de permutación de **Eulig et
  al. vía DoWhy** (ya en el pool — backward confirmado otra vez) y un test de Chow de Causal
  Change-Point Detection. CC-RuLSIF detecta en <5 muestras de media frente a ~40 de DoWhy en casos
  multi-arista — DoWhy testea el grafo completo por lotes, no localiza qué arista cambió.

  **Límites documentados por el propio autor (§2.7)**: asume suficiencia causal y grafo base
  estructuralmente invariante durante la ventana de referencia; se degrada con dimensionalidad de
  padres >10 (maldición de la dimensionalidad de métodos kernel); no compensa propagación indirecta
  de un cambio en un padre a sus hijos; garantía Tipo I exacta requiere muestras bootstrap i.i.d.
  (aunque el método funciona igual si el umbral se fija a mano).

  **Calidad de fuente**: tesis de máster, no revisada por pares fuera del comité — grey literature,
  aceptable per Garousi et al. (2018) cuando aporta algo que el resto del pool no cubre (aquí:
  todo el eje de monitorización online/streaming).

  **Decisión: incluir, categoría propia** ("monitorización online" vs. "validación puntual por
  lotes" del resto del pool) — no es borderline en el eje D ya usado (i.i.d./heterogéneo), es un
  tercer régimen (streaming no-estacionario) que merece su propia fila en la matriz de métodos.
  Related work (§1.2) da 4 candidatos backward nuevos — ver `informacion_relevante.md`.
