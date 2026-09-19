Gusenbauer & Haddaway (2019), "Which academic search systems are suitable for systematic reviews or meta-analyses? Evaluating retrieval qualities of Google Scholar, PubMed, and 26 other resources"

Literatura gris / búsqueda complementaria, con cribado acotado y documentado — Haddaway et al. (2015), "The Role of Google Scholar in Evidence Reviews and Its Applicability to Grey Literature Searching"

Específico a tu campo (ingeniería del software): Yasin, Fatima, Wen, Afzal, Azhar & Torkar (2020), "On Using Grey Literature and Google Scholar in Systematic Literature Reviews in Software Engineering"

Wohlin, C. (2014), "Guidelines for snowballing in systematic literature studies and a replication in software engineering" (EASE '14) — el start set del snowballing NO tiene que venir de una búsqueda sistemática reproducible; puede ser ad hoc. El rigor está en el procedimiento de snowballing posterior (criterios de inclusión/exclusión por iteración + criterio de parada por saturación), no en cómo se encontró la semilla.

Booth, A. (2008), "Unpacking your literature search toolbox: on search styles and tactics", Health Information & Libraries Journal 25(4) — describe el "vocabulary problem": cuando un concepto (DAGs) se usa con terminología distinta en disciplinas distintas (aquí: inferencia causal epidemiológica vs. ingeniería/ML), el keyword/boolean search sobre-representa sistemáticamente la disciplina que domina el vocabulario común. Cataloga citation searching / "pearl growing" (semilla + expansión por citas) como táctica legítima, no como parche de segunda categoría — es la explicación formal de por qué las queries te devuelven solo papers sanitarios.

Greenhalgh, T. & Peacock, R. (2005), "Effectiveness and efficiency of search methods in systematic reviews of complex evidence: audit of primary sources", BMJ 331(7524):1064 — auditoría empírica que muestra que en revisiones de evidencia compleja/interdisciplinar, la mayoría de los estudios incluidos NO salen de la búsqueda protocolizada en bases de datos, sino de snowballing y seguimiento de referencias. Es la cita que blinda relegar el keyword search sistemático a diagnóstico, y usar snowballing como método primario.

---

## Justificación metodológica (borrador para la sección de Metodología)

**Limitación reconocida.** Google Scholar no soporta consultas booleanas reproducibles ni exportación masiva, y su ranking depende de usuario/geolocalización/fecha (Gusenbauer & Haddaway, 2019). Por tanto no se usa como fuente primaria de una búsqueda sistemática con query fija y reproducible.

**Diagnóstico del fallo de las queries.** Las queries booleanas ensayadas devolvieron predominantemente literatura sanitaria/epidemiológica. Esto es un caso del "vocabulary problem" descrito por Booth (2008): DAGs es terminología compartida por la inferencia causal epidemiológica (dominante en volumen) y por la ingeniería/ML (el campo de este TFG), y el keyword search sobre-representa la disciplina con más literatura indexada bajo esos términos. Este fallo se documenta como evidencia empírica propia, no se descarta silenciosamente.

**Decisión metodológica.** Se releva el keyword search sistemático a herramienta diagnóstica y se adopta snowballing como método primario, respaldado por Greenhalgh & Peacock (2005): en revisiones de evidencia compleja/interdisciplinar la mayoría de los estudios finalmente incluidos se encuentran por snowballing y seguimiento de referencias, no por búsqueda protocolizada en bases de datos. La semilla (start set) se obtiene por búsqueda semántica acotada en Google Scholar — su falta de reproducibilidad no compromete el rigor del método porque, siguiendo Wohlin (2014), el start set no necesita ser reproducible; el rigor reside en el procedimiento de snowballing sistemático y documentado que se aplica a partir de él. Como complemento, se sigue la recomendación de Haddaway et al. (2015) de cribar solo los primeros 200-300 resultados de cada búsqueda en GS ordenados por relevancia, documentando query literal, fecha y nº de resultados examinados. El paralelo directo en ingeniería del software (Yasin et al., 2020) confirma que esta combinación (GS + literatura gris + snowballing) es una práctica aceptada en SLRs del campo.

---

## Plan de ejecución

**Fase 0 — Diagnóstico (evidencia del vocabulary problem).**
Registrar las queries booleanas ya probadas, fecha, nº de resultados y % de resultados sanitarios vs. relevantes. Esto se cita como el hallazgo empírico propio que motiva el pivote (respaldado por Booth, 2008).

**Fase 1 — Semilla (start set) por búsqueda semántica.**
Query(s) semántica(s) final(es) en GS, fecha de búsqueda, cribado de los primeros 200-300 resultados ordenados por relevancia (Haddaway et al., 2015). Criterio de inclusión: relevancia directa al método/algoritmo del TFG (no solo mención de DAGs en contexto sanitario). Resultado esperado: S0, ~10-20 papers semilla.

**Fase 2 — Backward snowballing, iteración 1.**
De S0, extraer referencias citadas. Definir y fijar el criterio de "autor/paper común" (p.ej. citado por ≥2 papers de S0, o entre los top-K en frecuencia de aparición en las bibliografías de S0) — este es el criterio reproducible que sustituye a la búsqueda inicial no reproducible.

**Fase 3 — Cribado de candidatos backward.**
Aplicar criterios de inclusión/exclusión (idioma, disponibilidad, relevancia al método) → S1.

**Fase 4 — Forward snowballing.**
Desde S0 ∪ S1, usar "cited by" de Google Scholar para cada paper; mismo criterio de relevancia → candidatos forward → S2.

**Fase 5 — Iterar hasta saturación.**
Repetir backward/forward sobre el conjunto acumulado. Criterio de parada: una iteración completa sin papers nuevos que cumplan el criterio de inclusión (Wohlin, 2014).

**Fase 6 — Trazabilidad.**
Tabla por paper: cómo se encontró (semilla / backward it.N / forward it.N), fecha, motivo de inclusión. Es lo que permite auditar el proceso aunque la búsqueda semántica inicial no sea reproducible bit a bit.

**Fase 7 — Redacción.**
Sección de Metodología con la justificación de arriba + diagrama de flujo (adaptación tipo PRISMA-S del proceso de snowballing) + tabla resumen (nº papers por fase, % filtrado en cada cribado).

---

## Revisión (2026-09-19): justificación con los números reales del pool, ancla = "falsify"

Dato empírico ya disponible en `data/scopus/` y `data/openalex/` (ver `queries_research.py` y
`CRITERIOS_CRIBADO.md`): de las 4 queries ya ejecutadas, las dos más *específicas* del núcleo
temático (falsificación / pertenencia a clase de equivalencia) son las de rendimiento más bajo:

- `q1_falsification` ("falsifying causal graph[s]", "falsifying causal model[s]", "falsify causal
  graph", "graph falsification"): **1 resultado en Scopus + 12 en OpenAlex = 13 brutos**.
- `q2_membership` ("membership testing"/"testing membership" AND "Markov equivalence class"):
  **1 + 1 = 2 brutos**.
- `q3_dagitty` (dagitty AND DAG/testable implications/d-separation): 45 + 36 = 81 brutos.
- `q4_dag_inference` (DAG AND hypothesis testing/DAG inference): 29 + 41 = 70 brutos.

Esto cambia la justificación: no hace falta apelar a contaminación de dominio (sanitario), hay un
hecho más simple y más fuerte — **la query léxicamente más precisa sobre el núcleo exacto del
tema ("falsify"/"falsification" de un grafo) devuelve casi nada** (13 brutos, probablemente menos
de 10 tras dedup y cribado). Una búsqueda booleana estrecha, por definición de estrecha, no puede
sostener por sí sola una revisión — no es un fallo de diseño de la query, es la consecuencia
esperable de acotar tanto los términos en un campo todavía pequeño/emergente.

**Justificación revisada, encadenando las citas ya reunidas:**

1. Se ejecuta primero la query más estrecha y léxicamente más fiel al tema (`q1_falsification`).
   Su bajo rendimiento (n=13 brutos) se documenta como hecho, no se interpreta como fallo de la
   query — es la motivación explícita para no depender solo de búsqueda booleana en bases de
   datos, siguiendo el patrón de Gusenbauer & Haddaway (2019): incluso las bases más completas
   fallan en recall cuando el vocabulario del campo aún no está estandarizado.
2. Los 13 candidatos de `q1` (o el subconjunto que sobrevive al cribado de `CRITERIOS_CRIBADO.md`)
   se toman como **seed set anclado** para el snowballing — deliberadamente se excluyen del ancla
   los papers que no comparten el término "falsify"/"falsification" explícito, para que la semilla
   sea de máxima precisión (mínimo ruido léxico) aunque sacrifique recall; el recall se recupera
   después vía snowballing, no ampliando la query.
3. Sobre ese seed set se añade búsqueda semántica en Google Scholar (no booleana, sin garantía de
   reproducibilidad exacta) para localizar literatura adicional que describe la misma tarea sin
   usar el término literal "falsify" — justificado por Haddaway et al. (2015) para cribado acotado
   (primeros 200-300 resultados) y documentado con query/fecha/nº de resultados examinados.
4. Backward + forward snowballing (Wohlin, 2014) desde seed set ∪ hallazgos de GS. El vocabulario
   que aparece de forma recurrente en las referencias del seed set — "membership testing" de clase
   de equivalencia de Markov, "testable implications"/d-separación, dagitty como herramienta de
   referencia, "DAG inference"/hypothesis testing — **no se asume a priori: se descubre durante el
   snowballing**, y es exactamente el vocabulario que ya se formalizó como `q2`, `q3`, `q4`. Esto
   convierte esas tres queries en **queries de expansión derivadas del snowballing**, no en
   conjeturas iniciales — patrón estándar de refinamiento iterativo de cadena de búsqueda a partir
   de estudios semilla ya conocidos: Brereton, Kitchenham, Budgen, Turner & Khalil (2007),
   "Lessons from applying the systematic literature review process within the software
   engineering domain", *Journal of Systems and Software* 80(4):571-583; y Kitchenham & Charters
   (2007), "Guidelines for performing systematic literature reviews in software engineering"
   (Technical Report EBSE-2007-01, Keele University/Univ. of Durham).
5. Las queries de expansión (`q2`-`q4`) se re-ejecutan de forma reproducible en Scopus+OpenAlex
   para dar cobertura sistemática al vocabulario ya descubierto — es lo que ya produjo el pool de
   121 candidatos tras dedup. El ruido resultante (dagitty/hypothesis testing son términos
   sobrecargados léxicamente) ya está anticipado y gestionado como EC5 en `CRITERIOS_CRIBADO.md`.
6. Como complemento de literatura gris/preprints muy recientes (el campo se mueve rápido: Orchard
   es de julio 2026) se añade Garousi, Felderer & Mäntylä (2018), "Guidelines for including grey
   literature and conducting multivocal literature reviews in software engineering", *Information
   and Software Technology* — guía estándar de MLR en ingeniería del software que respalda
   combinar bases formales + literatura gris vía GS cuando el tema es reciente y aún no está bien
   indexado con keywords estables.

**Nota de trazabilidad**: como `q1`/`q2` ya se ejecutaron y `q3`/`q4` ya existen como queries
formales, la narrativa de "vocabulario descubierto por snowballing → formalizado en queries" debe
poder sostenerse con lo que de verdad ocurrió (¿se derivaron `q2`-`q4` de leer las referencias de
los 13 papers de `q1`, o se diseñaron en paralelo por conocimiento previo del tema?). Si fue lo
segundo, la sección de metodología debe decirlo así — "diseño informado por conocimiento previo
del dominio, confirmado post-hoc por snowballing" es igual de defendible y más honesto que forzar
una secuencia estrictamente cronológica que no ocurrió.

---

## Revisión (2026-09-19, cont.): q3/q4 son mayoritariamente causal discovery, no falsación

Matiz adicional sobre el rendimiento alto de `q3_dagitty` (81 brutos) y `q4_dag_inference` (70
brutos): una lectura del pool sugiere que **la mayoría de esos resultados evalúan/aprenden
causal discovery (aprender el grafo desde cero) y no el problema de este TFG** (dado un grafo,
decidir si se rechaza/valida/testea frente a datos) — es exactamente `EC1` en
`CRITERIOS_CRIBADO.md` ("causal discovery / structure learning puro... sin paso explícito de
validar/falsificar un grafo ya dado"). Esto es un motivo *distinto* del ruido léxico de `EC5`: no
es que el término esté sobrecargado con otro campo (scheduling, compiladores), es que el término
sí es del campo correcto pero la tarea que resuelve el paper es la tarea vecina y más popular
(construir el grafo) en vez de la tarea de este TFG (testear un grafo ya dado). Ambos motivos de
exclusión conviven en `q3`/`q4` y se documentan por separado en el cribado.

Esto refuerza aún más el argumento: ni la query estrecha (`q1`, bajo n) ni las queries amplias
(`q3`/`q4`, n alto pero mayoritariamente fuera de tarea hacia causal discovery) dan un corpus
limpio por sí solas — es la combinación semilla-precisa + búsqueda semántica dirigida por
sinónimos + snowballing la que puede aislar el subconjunto que sí resuelve la tarea de
validación/falsación.

### Paso 1 revisado: Google Scholar con sinónimos, acotado

En vez de una única query semántica, se lanza una **serie de queries por sinónimo** (cada una
apuntando al concepto de testear/validar/falsificar un DAG dado, no de aprenderlo), por ejemplo:
"testing a given DAG", "DAG validation test", "causal graph consistency test", "graph adequacy
test", junto con las variantes ya usadas en `q1`/`q2`. Por cada una:

- **Acotar por año**: Google Scholar sí soporta rango de fechas — en la interfaz, panel lateral
  "Intervalo personalizado"; por URL, parámetros `as_ylo=2021&as_yhi=2026` (mismo rango que
  `FROM_DATE`/`TO_DATE` de `queries_research.py`, para consistencia con las queries formales).
- **Cribar los primeros ~30 resultados** ordenados por relevancia (más estrecho que los 200-300 de
  Haddaway et al. 2015, justificado porque aquí son varias queries por sinónimo, no una sola
  búsqueda amplia — la cobertura agregada across-sinónimos compensa el cap más bajo por consulta,
  y la precisión ya cae rápido pasado el top-30 en un nicho tan específico).
- Documentar por cada query: texto literal, fecha de ejecución, nº de resultados totales que
  reporta GS, nº cribado (30), nº que pasa a candidato tras aplicar `CRITERIOS_CRIBADO.md`.

### Paso 2 revisado: snowballing reproducible vía API de OpenAlex (no GS manual)

No hace falta usar la interfaz de Google Scholar para el snowballing en sí — **OpenAlex expone
tanto las referencias como las citas de forma reproducible y ya está integrado en
`queries_research.py`**:

- **Backward** (bibliografía de un paper semilla): cada `work` de OpenAlex trae el campo
  `referenced_works` (lista de IDs OpenAlex) — basta con resolverlos vía
  `GET /works?filter=openalex_id:ID1|ID2|...`.
- **Forward** (quién cita al paper semilla — el equivalente reproducible al "cited by" de GS):
  filtro `GET /works?filter=cites:{WORK_ID}`.

Con esto, la única parte no reproducible del proceso queda acotada a la semilla inicial (`q1` +
las queries semánticas por sinónimo en GS); el snowballing que construye el resto del corpus sí
es reproducible bit a bit — documentado como funciones nuevas en `queries_research.py`
(`forward_citations`, `backward_references`). El cribado de inclusión/exclusión sobre esos
candidatos sigue siendo manual (`CRITERIOS_CRIBADO.md`), como en cualquier snowballing (Wohlin,
2014) — la API solo reemplaza el "encontrar candidatos", no el "decidir si entran".

---

## Revisión (2026-09-19, cont. 2): ejecución real — lote GS clasificado y vocabulario confirmado

**Lo que ya se ejecutó** (no solo planificado): descarga manual de 18 PDFs vía búsqueda semántica
en GS a `GS/`, clasificados uno a uno en `CRITERIOS_CRIBADO.md` ("Lote GS"). Resultado: 5 includes
nuevos (Cornish et al. "Causal Falsification of Digital Twins" JMLR — el mejor encaje del lote;
Faller et al. "Self-Compatibility" AISTATS 2024; Kook "Falsifying causal models via nonparametric
CI testing" IWSM 2025; Karlsson et al. NeurIPS 2023 ws original, más amplio que su extensión ICML
2025 ya incluida; Haghighat & Janzing "Formalizing and Falsifying Causal Pathways of Rare Events"
2026, borderline), y 8 exclusiones documentadas (2 falsos positivos léxicos claros — filosofía de
la ciencia sin relación con DAGs causales — y 6 EC1/EC2/EC4 de patrones ya conocidos).

**Backward snowballing confirmado en vivo, no solo teórico**: Faller et al. cita explícitamente a
Eulig et al. (2023) — ya en el pool desde `q1_falsification` — y a Su & Henckel (2022), nueva
referencia backward. Esto es la primera arista de citación real capturada del proceso descrito en
la Fase 4 más arriba.

**Nueva referencia backward a verificar/incluir**: Su, Z. & Henckel, L. (2022), "A robustness test
for estimating total effects with covariate adjustment", *Uncertainty in Artificial Intelligence
(UAI)*, pp. 1886-1895. Test paramétrico (asume modelo causal lineal) para falsificar afirmaciones
interventionales de un modelo — facet distinta a Eulig/Kook (no-paramétricos). Pendiente de
cribado propio.

**Backward confirmado en vivo, segunda vez**: Savas (CC-RuLSIF, MS thesis Northwestern 2025,
`CRITERIOS_CRIBADO.md`) cita a Eulig et al. como baseline de comparación — segunda arista real de
la red de citas desde el seed set, esta vez desde un ángulo distinto (monitorización online). Su
§1.2 (Related Work) da candidatos backward nuevos y concretos para la falsificación online/
detección de cambio causal, distinta de la validación puntual por lotes del resto del pool:

- Huang, S., Peters, J. & Pfister, N. (2024), **"Causal change point detection and localization"**
  — el baseline "CausalCP"/test de Chow con el que comparan; detecta el instante en que
  `P(Y_t | Pa(Y_t))` cambia.
- Kocacoban, D. & Cussens, J., **"Online causal structure learning in the presence of latent
  [confounders]"** — fuente de OFCI/FOFCI (Online/Fast Online Causal Inference).
- Gao, S., Addanki, R., Yu, T., Rossi, R.A. & Kocaoglu, M., **"Causal discovery-driven change
  [point detection]"** — combina PCMCI + RuLSIF para localizar cambios de mecanismo.
- Kummerfeld, E. & Danks, D. (2012), **"Online learning of time-varying causal structures"** —
  trabajo fundacional (DOCL) del que derivan OFCI/FOFCI.

Pendientes de cribado propio (¿son EC1 — aprenden/actualizan el grafo — o encajan en IC1 al testear
si el grafo dado sigue siendo válido, como CC-RuLSIF? A revisar caso a caso).

### Vocabulario nuevo confirmado para la segunda ronda de búsqueda (semántica GS + entrenamiento ASReview)

Extraído de los papers ya incluidos en esta ronda, no inventado a priori — cada término viene de
un paper real ya verificado:

1. **"self-compatibility"** causal discovery — Faller et al. 2024
2. **"robustness test"** total effect covariate adjustment — Su & Henckel 2022
3. **"generalized covariance measure"** / **"covariance measure test"** (GCM) causal — Shah &
   Peters 2020, reaparece en Eulig et al. y Faller et al. como test subyacente compartido
4. **"permutation-based test"** causal graph — Eulig et al. 2023/2025
5. **"nonparametric conditional independence testing"** + falsify causal model — Kook 2025
6. **"causal pathway"** + "rare event" / outlier + testable implications — Haghighat & Janzing 2026
7. **"digital twin"** + causal + falsif* — Cornish et al. 2026 (⚠️ combinar siempre con un segundo
   término causal/DAG — "digital twin" solo satura de ingeniería/simulación no relacionada)
8. **"model criticism"** / **"model checking"** structural causal model — término clásico de
   estadística (Gelman-style posterior predictive checks) que el pool actual no ha probado
   todavía, no confirmado aún en ningún paper del pool — candidato exploratorio, no verificado
9. **"identification strategy"** + falsify, **"backdoor criterion"** + falsify — Karlsson et al.
10. **"causal change point detection"** / **"causal change-point"** — Huang, Peters & Pfister 2024;
    Savas (CC-RuLSIF thesis, `CRITERIOS_CRIBADO.md`) — régimen distinto (streaming/no-estacionario,
    no i.i.d.), cubre la pregunta de "grafo dinámico" que no cubría el resto del pool.
11. **"online causal structure learning"**, **"time-varying causal structures"** — Kocacoban &
    Cussens; Kummerfeld & Danks 2012 (DOCL) — ⚠️ alto riesgo de EC1 (muchos de estos aprenden/
    actualizan el grafo en vez de testear si el dado sigue siendo válido; cribar caso a caso).

**Términos ya cubiertos por `q1`-`q4` y por tanto NO añadir de nuevo**: "falsifying causal
graph(s)", "membership testing"+"Markov equivalence", dagitty, "hypothesis testing"+DAG.

**Cómo usarlo para ASReview**: los ~121 candidatos ya cribados (`CRITERIOS_CRIBADO.md`, con
decisión IC/EC explícita por cada uno) + los 18 del lote GS ya clasificados sirven de **conjunto
de entrenamiento inicial etiquetado** (relevante/irrelevante) para el modelo activo de ASReview.
La segunda tanda de queries semánticas (términos 1-7 arriba, cribando primeros ~30 por consulta
como se definió en la Fase 1 de más arriba) alimenta el pool que ASReview prioriza a partir de ese
entrenamiento — así el ranking de relevancia no parte de cero.

### Índice de lo que queda por hacer

1. **Backward restante**: resolver la referencia Su & Henckel (2022) y cribarla; revisar el resto
   de la bibliografía de Cornish et al. (JMLR, 52 págs., no leída a fondo aún) y de Kook 2025 por
   si citan más candidatos directos.
2. **Segunda ronda de búsqueda semántica en GS** con el vocabulario de la lista de arriba (7
   términos confirmados + 1 exploratorio), acotada por año 2021-2026, cribando primeros ~30 por
   consulta, documentando query/fecha/nº resultados como en la Fase 1.
3. **Entrenar ASReview** con el conjunto ya etiquetado (121 + 18) sobre el pool ampliado de (1)+(2).
4. **Forward desde ASReview**: dejar que el modelo activo priorice, cribar hasta saturación (una
   pasada sin candidatos nuevos relevantes) — **criterio de parada explícito del usuario**: dado
   que el campo es pequeño (pool ya en el orden de 20-30 papers núcleo), no perseguir exhaustividad
   más allá de esta pasada; cerrar la búsqueda ahí y pasar a redactar el estado del arte.
5. **Cierre**: con (1)-(4) completos, la sección de Metodología del TFG queda con: semilla
   (booleana + semántica) → diagnóstico de vocabulario → snowballing backward (manual +
   reproducible vía OpenAlex) → snowballing forward (ASReview activo) → criterio de saturación
   explícito — trazable de principio a fin con las citas metodológicas ya reunidas (Wohlin 2014,
   Booth 2008, Greenhalgh & Peacock 2005, Gusenbauer & Haddaway 2019, Haddaway et al. 2015,
   Brereton et al. 2007, Kitchenham & Charters 2007).

