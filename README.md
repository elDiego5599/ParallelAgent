# ParallelAgent

[![CI](https://github.com/elDiego5599/ParallelAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/elDiego5599/ParallelAgent/actions/workflows/ci.yml)

Sistema local de deliberación multi-modelo para tareas de programación. Varios modelos de lenguaje discuten una misma tarea sobre un repositorio real hasta alcanzar una solución común, que se entrega en una rama de Git separada para revisión.

No es un pipeline por etapas. Es una mesa de trabajo compartida: los modelos conocen a los demás participantes, leen el historial completo de la discusión y se corrigen entre sí antes de emitir código.

El framework soporta dos topologías complementarias. En `peer` los modelos deliberan en igualdad de condiciones, sin jerarquías. En `lead` un modelo lidera y redacta, y el resto audita. En ambos casos el único moderador es el orquestador, que reparte turnos, mantiene la transcripción y aplica la regla de parada. No valida contenido técnico.

La herramienta no impone salvaguardas sobre la configuración. El usuario es responsable de los modelos que elige, de las API keys que usa y de los costos asociados.

## Arquitectura

El funcionamiento se divide en dos fases: deliberación y emisión. La topología se infiere del CLI: `--models` activa `peer`, `--lead` con `--advisors` activa `lead`.

**1. Deliberación.**
Los modelos reciben la tarea y el contexto del repositorio. Cada intervención se agrega a una transcripción compartida que todos leen en el turno siguiente. En esta fase está prohibido reescribir archivos completos; solo se discute diseño, APIs, casos borde y riesgos (condiciones de carrera, gestión de memoria, interoperabilidad JNI / MethodChannel, tipado entre capas).

Estados en `peer`:

```
ESTADO: DEBATIENDO
ESTADO: CONSENSO_ALCANZADO
ESTADO: PREGUNTA_AL_USUARIO
```

Ante `PREGUNTA_AL_USUARIO` el orquestador pausa la sala, pide la aclaración por terminal y la inyecta al transcript como mensaje prioritario. Solo se autoriza ante bloqueos arquitectónicos o de negocio; prohibido preguntar por estilo o convenciones. Con `--non-interactive` no se pregunta: se asume la vía conservadora y continúa. Límite de 3 preguntas por sesión.

Estados en `lead` (líder propone y redacta, asesores auditan sin escribir código):

```
ESTADO: CONFORME
ESTADO: DEBATIENDO
ESTADO: OBJECION_BLOQUEANTE
ESTADO: VETO_ARQUITECTONICO
```

`OBJECION_BLOQUEANTE` devuelve el turno prioritario al líder para responder o corregir. `VETO_ARQUITECTONICO` unánime de los asesores obliga al líder a replantear la propuesta desde cero.

**2. Emisión.**
Cuando se detecta quórum o se agota `--max-rounds`, se cierra el debate. En `peer`, el modelo indicado con `--writer` transcribe el resultado acordado; si no se define, redacta el último en hablar. En `lead`, redacta siempre el líder. En modo `build`, el resultado es un parche en formato diff unificado que se aplica sobre una rama nueva del tipo `consensus/<fecha>-<tarea-corta>`. Nunca se escribe directo sobre `main`.

El orquestador no opina sobre el código. Solo reparte turnos, mantiene la transcripción y aplica la regla de parada.

## Modos de ejecución

- `build` (por defecto): deliberación y emisión del diff directamente en una rama de Git.
- `plan`: deliberación y devolución de un plan técnico en Markdown. No modifica código ni crea ramas.
- `ask`: consulta técnica o auditoría. Los modelos debaten entre sí y devuelven la respuesta. No emite parches.

## Componentes previstos

- `cli.py`: punto de entrada. Infiere la topología (`--models` para `peer`, `--lead` con `--advisors` para `lead`) y parámetros de tarea, ruta, límites y modo.
- `orchestrator.py`: `PeerEngine`, estados, HITL y salida a Git compartida (`finish_build_output`).
- `lead_engine.py`: `LeadEngine` (líder propone y redacta, asesores auditan con `CONFORME` / `OBJECION_BLOQUEANTE` / `VETO_ARQUITECTONICO`).
- `providers.py`: adaptadores para proveedores con tier gratuito (Pollinations, OpenRouter, Groq). Cada adaptador expone la misma interfaz de chat.
- `context.py`: construcción del mapa de contexto del repositorio. Recorte por relevancia para no saturar la ventana de contexto.
- `git_bridge.py`: creación de rama, aplicación del diff y reporte de cambios para revisión manual. Solo se usa en modo `build`.

## Instalación

Requisitos: Python 3.10 o superior, Git.

```
pip install -r requirements.txt
```

La única dependencia inicial es `requests` para los proveedores HTTP. Sin claves ni servicios adicionales para el prototipo con tier gratuito.

## Configuración

Los proveedores leen las claves de las variables de entorno (`POLLINATIONS_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`). `.env.example` documenta el formato; expórtalas en tu shell (`export GROQ_API_KEY=...`) o con tu gestor habitual. El `.env` real nunca se commitea.

Pollinations requiere clave gratuita (https://enter.pollinations.ai/keys). El antiguo endpoint anónimo ya no responde.

## Pruebas

```
pip install pytest
pytest tests/
```

44 pruebas con `MockProvider`, sin red ni claves. Cubren CLI, quórum peer, fast-path y vetos lead, HITL, contexto y Git.

## Uso

Ejemplo base en `peer`:

```
python cli.py --task "Corregir fuga de memoria en el puente Flutter/C++" --path ./mi-proyecto --models llama-3.1-70b mistral-large deepseek-r1 --max-rounds 4
```

Ejemplo base en `lead`:

```
python cli.py --task "Agregar logs y try/catch a cinco funciones" --path ./mi-proyecto --lead claude-3-7-sonnet --advisors gpt-4o deepseek-r1 --mode build
```

Parámetros:

- `--task`: descripción técnica de la tarea. Obligatorio.
- `--path`: ruta al repositorio a analizar. Obligatorio.
- `--models`: lista de identificadores de modelo que forman la mesa peer. Mínimo 2. Incompatible con `--lead` / `--advisors`.
- `--lead`: identificador del modelo líder. Requiere `--advisors`.
- `--advisors`: lista de identificadores de modelos asesores. Mínimo 1.
- `--writer` (solo `peer`): identificador del modelo que transcribe el resultado final. Debe pertenecer a `--models`. Opcional. Por defecto, el último en hablar.
- `--mode`: `build`, `plan` o `ask`. Por defecto `build`. En `build`, el diff emitido se aplica vía `git_bridge.py` sobre la rama efímera.
- `--max-rounds`: límite de rondas de deliberación antes de forzar votación final. Por defecto 4.
- `--quorum` (solo `peer`): `unanime` o `mayoria`. Por defecto `unanime`.
- `--non-interactive`: no pedir aclaraciones por terminal (CI/CD). Por defecto se pregunta.
- `--context-budget`: caracteres máximos del mapa de contexto. Por defecto 12000 (~3k tokens, seguro bajo TPM de 8000 de tiers gratuitos).

Comités de ejemplo:

```
# Peer, paranoia máxima contra alucinaciones
python cli.py --task "..." --path ./app --models claude-3-7-sonnet gpt-4o deepseek-r1 --mode build --writer claude-3-7-sonnet

# Lead, tarea quirúrgica y rápida
python cli.py --task "..." --path ./app --lead claude-3-7-sonnet --advisors gpt-4o deepseek-r1 --mode build

# Peer, solo plan sin tocar código
python cli.py --task "..." --path ./app --models claude-3-7-sonnet gpt-4o --mode plan

# Peer, consulta técnica
python cli.py --task "..." --path ./app --models gpt-4o deepseek-r1 --mode ask
```

## Regla de parada

En `peer`, el debate termina cuando se cumple alguna de estas condiciones, en este orden:

1. Todos los modelos (o N-1 si `--quorum mayoria`) cierran con `ESTADO: CONSENSO_ALCANZADO` en la misma ronda.
2. Se alcanza `--max-rounds`. En ese caso se pide una votación final y se emite el resultado según el modo (`build`: parche mayoritario; `plan`: plan; `ask`: respuesta).
3. Fallo de proveedor o tiempo de espera. La ronda se registra como incompleta y se continúa con los modelos disponibles.

En `lead`, termina cuando todos los asesores cierran con `ESTADO: CONFORME` o se alcanza `--max-rounds`. Una `OBJECION_BLOQUEANTE` devuelve el turno al líder. Un `VETO_ARQUITECTONICO` unánime obliga al replanteo total.

El marcador de estado se extrae por expresión regular en cualquier posición del mensaje (tolera envoltorios Markdown como `**ESTADO: ...**`). Si falta o viene malformado, el turno se cuenta como `DEBATIENDO`.

Ante 429 (límite de tokens por minuto) el proveedor reintenta hasta 2 veces respetando `Retry-After`.

## Control de contexto y costo

- En deliberación solo se envían fragmentos relevantes del repositorio, no archivos completos.
- Ventana deslizante: cada turno incluye el contexto, la ronda 1 y desde la ronda anterior; la charla intermedia se omite para acotar el payload.
- El historial se trunca por rondas antiguas cuando supera el límite configurado; se conserva siempre el pitch inicial y la última ronda completa.
- La emisión de código se hace una sola vez, por un solo modelo, para evitar duplicar tokens.

## Estado actual

`PeerEngine` y `LeadEngine` implementados con contexto de repositorio, interrupción al usuario y salida a Git en modo `build`. Verificados con proveedores simulados.
