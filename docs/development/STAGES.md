# Xandra Motion Studio — execution plan

Дата: 9 октября 2026 года. Статус: Stages 0–4 выполнены как `baseline_preview` checkpoint; остановка для review перед Stage 5. Результаты и ограничения: [STAGES_0_4_HANDOFF.md](STAGES_0_4_HANDOFF.md). Содержание этапов ниже сохраняет утверждённые требования.
Основание: `ARCHITECTURE.md`, `IMPLEMENTATION_PLAN.md`, `../../examples/jobs/xandra_preview.job.json` и исследования архивных проектов в `../research/`.

## Как выполнять этот план

22 последовательных stage: **0–21**. Первый сквозной MP4 Xandra из новых наблюдений — **Stage 4**. Первый новый temporal solver — **Stage 8**. Stage 1 делает AnimationBundle исполняемым в Python; Stage 2 подтверждает его исполнение реальным ригом Blender. До Stage 4 нужны только пакет, данные, вычисления и последовательный вызов функций: без SQLite, DAG scheduler, общего cache, UI и installer.

Новый код размещать непосредственно в этом repository: Python package — в `src/xms/`, Blender entry points — в `blender/`, profiles/schemas/tests — в одноимённых каталогах. Архивные `clip02_blender_kit`, `clip_full` и `dance` остаются внешними research/reference sources. Никаких runtime imports из них, редактирования оригинальной модели или обязательного промежуточного GLB. Большие assets и weights регистрируются через конфигурацию и hashes, не копируются в Git или на каждом шаге. Малые численные fixtures можно хранить в tests; реальные видео, кадры и результаты — в исключённых из Git `runs/` и asset store.

Существующие venv и GLB не читать, не запускать и не изменять. Для нового pipeline использовать отдельное environment; его создание требуется только при отсутствии совместимого runtime. Optional Stage 18 работает с отдельно предоставленным для экспорта asset после явного разрешения на его использование и пишет новый результат, не затрагивая защищённые GLB.

Каждый stage выполнять по порядку его Implementation. После одного сфокусированного verification pass зафиксировать результат и остановить изменения этого stage. Успешные проверки не повторять без изменения их зависимостей. После двух неудач одного подхода сохранить evidence и сообщить о неудаче до смены подхода. Если объём существенно расширился, сначала сообщить состояние. Плановые команды ниже — интерфейсы для реализации, сейчас они не существуют.

Для каждого stage сохранять в `runs/<id>/` либо `benchmarks/results/<id>/`: входные hashes, параметры, версии, exit status, измерения и пути результатов. Ранний manifest — обычный JSON, не job database. До общего QA явно маркировать результаты `baseline_preview`/`development_candidate`; создание видео само по себе не означает производственную приёмку.

## Границы, которые нельзя нарушать

- **Ingest → observations:** исходные PTS, координаты и crop transforms, named landmarks/channels, subject identity, masks/confidence. Наблюдения не содержат костей Xandra и не считаются анимацией. 3D tracker estimates не объявляются метрической ground truth.
- **Observations → solver:** solver читает наблюдения, timeline и профиль; не импортирует `bpy`, не читает старые actions/GLB с движением. Calibration/camera estimates — отдельные versioned параметры. Недостающие данные сохраняют явную неопределённость.
- **Solver → AnimationBundle:** один writer/validator, immutable опубликованный bundle. Все ветки тела/лица/кистей собираются на общей временной сетке; ownership головы/глаз/челюсти разрешается до публикации. Baseline и новый solver имеют один output interface.
- **AnimationBundle → Blender/render:** адаптер знает риг и сцену, но не делает tracking, IK-коррекцию или face fusion. Он только вычисляет утверждённую анимацию в выходные моменты, переводит координаты и создаёт actions. Render и optional export используют один bundle hash. Коррекция движения выпускает новый bundle и заново запускает соответствующий QA.
- Формат с Stage 1: `metadata.json` + NPZ, `allow_pickle=False`, метры, правая система, Y-up/+Z forward, quaternion XYZW, `R_local = R_rest_local × R_delta`. Rest TRS и armature transforms учитываются целиком; root применяется ровно один раз. Неиспользуемые каналы имеют neutral значения и `unobserved`, а не фиктивную высокую confidence. Contacts/events вначале пустые, но уже имеют определённую структуру.
- Baseline — поддерживаемый backend для сравнения, а не второй pipeline. Старые world-delta matrices и WXYZ кватернионы допустимы только внутри явно названного conversion helper; bundle всегда local-rest delta/XYZW. Schema меняется только с версией и явной миграцией/отказом.

## Stage 0 — только блокирующий smoke experiment

**Goal:** установить, можно ли начать исполнение контракта на имеющемся Mac, не открывая новое исследование архитектуры.

**Inputs:** внешние read-only assets из архивного kit: `Xandra_Animations.blend`, `clip02_scene.blend`, `sing.mp4`; локальные Blender/FFmpeg/FFprobe и pose task. Их пути передаются аргументами или локальной незакоммиченной конфигурацией; совместимость ещё не измерена.

**Implementation:**
1. Создать только `experiments/runtime_smoke.py` и `experiments/README.md`; принять пути аргументами, проверить версии и импорт NumPy/MediaPipe в совместимом runtime вне существующих venv. Если такого runtime нет, зафиксировать минимальный набор зависимостей для нового изолированного environment; не исследовать чужие venv и не устанавливать весь конечный стек.
2. Через FFprobe прочитать видео/аудио и декодировать один кадр; через pose backend получить результат этого кадра. Проверить доступ к локальной pose model, без скачивания полного набора моделей.
3. Blender в отдельном процессе читает Xandra без сохранения, выводит armature/object transforms, bone inheritance, shape-key имена, версию action API. Создать диагностический action с двумя ключами одной кости, проверить evaluated transform, не рендерить клип.
4. Записать `docs/development/STAGE0_RESULT.md`: выбранные binaries/environment/model/asset hashes, результат и единственный необходимый fix при блокировке. Если всё работает, не исследовать альтернативные backends.

**Output / observable result:** короткий smoke log и установленный минимальный runtime для Stage 1–4.

**Verification:** subprocess exit codes, pose output корректной формы, action меняет evaluated pose; исходные asset hashes не меняются.

**Exit criteria:** доступны pose inference, чтение рига/actions и decode/encode prerequisites. Если отсутствует необходимая зависимость, сообщить конкретный blocker и минимальный объём установки; большие downloads не начинать молча.

**Dependencies:** нет. **Next dependency:** Stage 1 использует проверенный runtime. Никаких экспериментов с Demucs, temporal quality или surface collisions здесь: они не блокируют первый slice.

## Stage 1 — исполняемый AnimationBundle и численный FK

**Goal:** получить реальные сериализуемые и вычисляемые данные, на которых будут работать все backends.

**Inputs:** Stage 0; контракт §5 архитектуры; маленький synthetic rig.

**Implementation:**
1. Создать `pyproject.toml`, `src/xms/animation/{bundle,io,validate,fk,sample}.py`, минимальный `cli.py`; NumPy — необходимая зависимость, без job framework.
2. Реализовать metadata/NPZ writer, reader и validator: shapes/dtypes, unique names, parent hierarchy, finite values, unit quaternions, times, profile hash, ranges, ownership, masks/provenance и source-time map. Определить empty contacts/events и unsupported-version errors.
3. Создать FK/rest composition, quaternion sign continuity/SLERP и sampling по секундам; определить root carrier в profile, включая pelvis mapping.
4. Создать `schemas/animation_bundle.v1.json`, `tests/test_bundle_contract.py`, `tests/test_fk.py`, малый fixture rest + root shift + 90° rotation + facial channel. Добавить `xms bundle validate PATH`.

**Output:** валидный synthetic bundle и численные world transforms без Blender.

**Verification:** round-trip не меняет arrays; analytical FK совпадает с fixture; malformed hierarchy, NaN, nonmonotonic times, conflicting owners и profile mismatch отклоняются. Norm error ≤ 1e-4.

**Exit criteria:** bundle можно записать, независимо прочитать, интерполировать и выполнить; schema не единственное доказательство.

**Dependencies:** 0. **Next dependency:** Stage 2 исполняет тот же контракт на Xandra; Stage 4 не создаёт собственный формат.

## Stage 2 — профиль Xandra и bundle → реальный Blender rig

**Goal:** доказать семантику transforms/morphs до подключения tracking.

**Inputs:** Stage 1; исходные Xandra/bedroom assets; старые `scripts/xcore.py`, `b_common.py`, `b_render.py` только как reference.

**Implementation:**
1. Создать `blender/register_character.py`, `src/xms/profiles/{character,scene}.py`, `profiles/characters/xandra/v1/{character.json,rig.npz,face_map.json}`, `profiles/scenes/bedroom/v1/scene.json`. Извлечь rest TRS, роли/axes, armature transforms, root carrier, morph aliases/ranges и ownership; не переносить приблизительные contact points без проверки.
2. Создать `render/blender_adapter.py` и `blender/apply_bundle.py`: Y-up → Z-up, bone conversion с учётом inheritance, поддерживаемые actions, очищение активного legacy action/NLA в рабочей сессии.
3. Создать Xandra fixture с rest, root shift, одним плечом, запястьем, головой, jaw/blink/smile по отдельности. Снять evaluated matrices/shape weights и диагностические stills. Добавить `tests/integration/test_blender_bundle.py`.
4. Материальные corrections из kit оформить явным versioned look profile; применять идемпотентно к рабочей копии/сессии. Зубной вариант описать отдельно, не изменять canonical asset.

**Output:** Xandra принимает bundle; видны отдельные диагностические позы/морфы. Это ещё не видео из MP4.

**Verification:** canonical Python FK против обратно преобразованного Blender evaluation: position error ≤ 1 мм, angular error ≤ 0.1°, weights ≤ 1e-5 на fixture. Проверить nonidentity armature transforms численным fixture и реальный rig; root не удваивается, jaw/head не применяются дважды; asset hashes неизменны.

**Exit criteria:** numerical round-trip и визуальная проверка направлений проходят; несовместимый профиль отклоняется до render.

**Dependencies:** 1. **Next dependency:** Stage 4 использует готовый consumer bundle, а не старый `anim_v4.npz`.

## Stage 3 — MP4 → timeline → body observations

**Goal:** получить новые наблюдения непосредственно из видео с корректным временем.

**Inputs:** Stage 0 runtime, Stage 1 data conventions, короткий 2–3-секундный интервал `sing.mp4` с видимым движением; полный исходник не обрезается физически.

**Implementation:**
1. Создать `ingest/{probe,timeline,decode}.py`: rational FPS, `[start,end)`, rotation/pixel aspect, исходные PTS и audio offset. `N = ceil(duration × fps)`; time map хранится явно.
2. Создать `observations/{contract,io,pose_mediapipe}.py`, `schemas/observations.v1.json`. Сохранять image/world landmarks, visibility, frame validity, зеркальность и model hash; raw и cleaned arrays различать.
3. Добавить `xms observe INPUT --start ... --end ... --out ...`, `tests/test_timeline.py` и `tests/integration/test_pose_observations.py`; один последовательный pose worker.

**Output:** `timeline.json` и body observations NPZ/metadata, overlay на нескольких кадрах; нет костей/морфов Xandra.

**Verification:** выбранные PTS и time map совпадают с decode; пустая детекция помечена invalid; overlay показывает правильную ориентацию/стороны. Unit fixtures покрывают 24/30 и 30000/1001, ненулевой video/audio start.

**Exit criteria:** реальные новые observations читаются без MediaPipe runtime через общий reader, время не выводится из номера кадра.

**Dependencies:** 0, 1; по последовательности после 2. **Next dependency:** Stage 4 получает observations без повторного decode.

## Stage 4 — первый end-to-end vertical slice и MP4 Xandra

**Goal:** одна команда проходит **input MP4 → observations → baseline solver → AnimationBundle → Xandra → rendered MP4**.

**Inputs:** Stage 2 профиль/адаптер, Stage 3 observations/timeline и выбранный короткий интервал.

**Implementation:**
1. Создать `solve/{interface,baseline_body}.py`: направления сегментов → fixed-length target rig, root estimate, ограниченные rotations и head body prior. Адаптировать полезную математику reference scripts в пакет с явной конверсией world/local; не импортировать scripts соседнего проекта, не использовать готовое движение `data/anim_v4.npz`.
2. Baseline пишет Stage 1 bundle. Face/finger channels пока neutral с unobserved masks; contacts пустые; ограничения явно в manifest. Никакой A-позы первого кадра и поправок по секундам `sing`.
3. Создать `blender/render_frames.py`, `render/process.py`, `assembly/encode.py`, `pipeline.py`, минимальный `config.py` и baseline JobSpec fixture. В последовательном вызове: observe → solve → validate → apply → preview frames → FFmpeg → проверить MP4. JSON manifest с hashes и логами; уникальный каталог run, без database/cache/resume.
4. Добавить `xms run INPUT --character xandra --scene bedroom --quality preview --solver baseline --start ... --end ...`; render 540×960 на EEVEE, fixed camera из проверенного scene profile. Перед кадрами сделать маленький encode preflight с тем же codec/размером. Исходный звук mux по общей timeline, если доступен.
5. Создать `tests/integration/test_vertical_slice.py` и краткую инструкцию запуска. Все paths передавать конфигурацией; каждый модуль можно вызвать отдельно для диагностики.

**Output:** первое `runs/<id>/outputs/video.mp4`, новый bundle, observations и manifest. Xandra движется по этому MP4; качество пока baseline.

**Verification:** один реальный короткий run; MP4 полностью декодируется, содержит N кадров/правильный размер, duration error ≤ 1 выходного кадра, аудио присутствует только при source audio и имеет правильный offset. Bundle hash в render manifest совпадает; независимый запуск render-only из сохранённого bundle не требует исходного MP4 или tracker. Проверить видимое движение на видео, а не только existence файла.

**Exit criteria:** slice проходит без ручных стадий и старой анимации; недостатки лица/кистей зарегистрированы, preview не объявлен final-quality.

**Dependencies:** 1–3. **Next dependency:** Stage 5 добавляет нужные для сравнения артефакты, не меняя контракт.

## Stage 5 — сравнение, базовый QA и время на сложных media

**Goal:** сделать работающий slice наблюдаемым и сохранить исходную временную семантику.

**Inputs:** Stage 4 video/bundle; controlled silent/VFR/rotated/nonzero-PTS fixtures из имеющегося источника.

**Implementation:**
1. Создать `qa/{contract,timing,coverage}.py`, `assembly/compare.py`, `report/html.py`; `quality.json`, `compare.mp4`, `report.html` связать с input/profile/bundle/scene/runtime hashes.
2. Compare синхронизировать через source_time_map, нормализовать размеры/SAR до stack; не растягивать source ради совпадения.
3. Создать `benchmarks/{manifest.json,make_media_fixtures.py}`; silent, VFR и short/invalid media. Метрики без данных возвращают unavailable с причиной; missing tracking не pass.
4. Добавить статусы результатов/exit codes для failed/needs_input/needs_review/warnings/success с development quality marker. Схема input handling пока не обещает надёжную multi-person selection.

**Output:** baseline video + compare + quality/report; измеренные wall time и peak memory основных процессов.

**Verification:** focused media matrix: silent, VFR, rational FPS, rotation и offset; output/source duration и A/V skew ≤ 1 кадра, invalid input заканчивается отчётом до рендера. Проверить, что intentionally unavailable face/contact metrics не зелёные.

**Exit criteria:** временная ошибка не маскируется FFmpeg defaults, любой результат имеет provenance и читаемое объяснение.

**Dependencies:** 4. **Next dependency:** Stage 6/7 используют report для сравнения улучшений.

## Stage 6 — baseline лица, головы и кистей через тот же bundle

**Goal:** довести baseline до всех обязательных групп движения без новых качественных solver решений.

**Inputs:** Stage 5; локальные face/hand tasks из `clip_full/tools/`; профиль Xandra.

**Implementation:**
1. Создать `observations/{face_mediapipe,hands_mediapipe}.py`: named face scores/matrix, handedness, crop transforms, masks. Tasks запускать последовательно по бюджету памяти.
2. Создать `solve/{baseline_face,baseline_hands,head}.py`: mapping по именам/aliases, простые temporal filters, wrist относительно solved forearm, пальцы из rest geometry, face rotation как head input и body fallback.
3. Общий compositor в `animation/compose.py` разрешает ownership и публикует bundle; confidence/fallback provenance сохраняются. Убрать head управление baseline body там, где владеет face branch.
4. Face confidence не брать из blendshape intensity; на baseline использовать availability/ROI size и обозначить ограниченность. Long gaps — neutral fallback, не длинная интерполяция.

**Output:** новый MP4 и close-up диагностические отрезки с лицом/головой/пальцами; тело продолжает baseline solve.

**Verification:** один короткий baseline run и synthetic mapping checks: `_neutral`, duplicate aliases, L/R mirror, forearm rotation, jaw/eye/head ownership. Ни одной зависимости от предварительно экспортированного animated GLB; видимые потери указаны в отчёте.

**Exit criteria:** все группы представлены исполнением и masks; новые backends смогут заменить только соответствующий solve.

**Dependencies:** 2, 3, 5. **Next dependency:** Stage 7 фиксирует этот baseline для количественного сравнения.

## Stage 7 — benchmark и калибровка без neutral первого кадра

**Goal:** подготовить измеримые inputs нового solve и честный baseline, а не большой исследовательский набор.

**Inputs:** Stage 6; `sing`, существующий dance source (точный path/hash определить из archive manifest), отдельный holdout без neutral pose. При отсутствии holdout это явный blocker обобщающей приёмки, а не Stage 4.

**Implementation:**
1. Создать `benchmarks/{annotations,configs,baseline}/`, `qa/{pose,kinematics}.py`: несколько stance/flight intervals, уверенные joints и closure/open событий; разделить tuning/holdout до настройки thresholds. Для речи добавить отдельный source в Stage 12.
2. Создать `solve/calibration.py`: выбрать несколько надёжных участков, оценить global orientation/camera priors/scale при фиксированном росте Xandra; использовать target rest regularization при отсутствии neutral. Сохранить calibration confidence и reference camera.
3. Измерять normalized reprojection, limits, angular velocity/acceleration, root trajectory, foot drift и coverage на одинаковых samples. Source landmarks — proxy; не называть их ground truth глубины.
4. Зафиксировать `benchmarks/acceptance.v1.json`: исходные цели архитектуры, tolerances для regression/continuity и performance budget по Stage 5; thresholds фиксировать до temporal comparisons.

**Output:** baseline measurements и calibration artifact; runnable baseline использует новый calibration interface.

**Verification:** start в non-neutral pose не требует ручной A-позы; missing neutral даёт lower confidence; repeat solve фиксированных inputs даёт одинаковые данные в заявленной tolerance. Tuning/holdout не совпадают по source hash.

**Exit criteria:** можно сравнивать solver на фиксированных данных/camera; thresholds и unavailable policy записаны до эксперимента.

**Dependencies:** 6. **Next dependency:** Stage 8 использует calibration и baseline objective measurements.

## Stage 8 — первый новый temporal body solver на одном окне

**Goal:** получить реальный оптимизатор по времени, отдельно проверив математику до stitching/contacts/collision.

**Inputs:** Stage 7 frozen observations/profile/calibration; 2–4 секунды; baseline как initial state.

**Implementation:**
1. Создать `solve/{temporal_body,residuals,parameterization}.py`; подключить SciPy least_squares. Optimize root и joint rotation increments; reprojection + soft 3D prior + velocity/acceleration + joint limits, confidence weights, robust loss. Camera фиксирована или строго bounded calibration refinement с фиксированным gauge.
2. Сохранить solver diagnostics, termination reason, objective terms до/после, runtime/peak memory. Ограничить iterations/evaluations; solver не превращает exhaustion в success.
3. Добавить `--solver temporal`, сохранив baseline backend и тот же bundle writer/consumer. Контакты/collision hooks предусмотрены, но weights отключены до Stages 10/14 и так отмечены.
4. Создать `tests/test_temporal_body.py`: synthetic known motion + noise/outlier/missing observations; реальные tuning samples сравнивать через Stage 7 harness.

**Output:** первый новый temporal bundle и видео из него без изменения Blender adapter.

**Verification:** synthetic motion recover error уменьшается относительно initializer, temporal jitter уменьшается; на реальном отрезке зафиксировать reprojection/limits/jitter/runtime против baseline в той же camera. Не разрешать улучшение jitter ценой сглаживания размеченного быстрого жеста сверх frozen regression tolerance.

**Exit criteria:** оптимизатор исполняется и производит валидный bundle в бюджете; synthetic test и frozen regression gates проходят. Если реальный quality gain не доказан, новый backend остаётся experimental, baseline — default; сохраняется результат и конкретная следующая гипотеза, без объявления улучшения.

**Dependencies:** 7, 1. **Next dependency:** Stage 9 переносит работающий single-window solve на весь клип.

## Stage 9 — окна, границы и непрерывность temporal solver

**Goal:** обработать полный клип с ограниченной памятью без швов и дрейфа.

**Inputs:** Stage 8 solver; полный `sing`/dance observations и frozen budgets.

**Implementation:**
1. Создать `solve/windows.py`: окна 2–4 секунды, overlap, boundary constraints/warm start, sign-consistent quaternion stitching; terminal interval без добавления времени.
2. Общая проверка после stitch: velocity, acceleration, root/rotations на границах; optional chunked NPZ reader/writer в `animation/io.py` только если измеренная память требует chunks.
3. Добавить `qa/continuity.py`, `tests/test_windows.py`; логировать память и стоимость по окнам.

**Output:** temporal full-length bundle и compare; запись не растёт как плотный solve всего фильма.

**Verification:** known synthetic trajectory с границей в быстром жесте; error/velocity у границ не превышают frozen tolerances. Полный реальный run проходит time/contract QA и performance budget; нет повторённых/потерянных samples.

**Exit criteria:** полный клип runnable, stitching metrics проверены, ухудшение не скрывается smoothing.

**Dependencies:** 8. **Next dependency:** Stage 10 добавляет физические contact terms в эти же окна.

## Stage 10 — опоры, прыжки и совместные constraints ног

**Goal:** уменьшить foot skating, сохранив flight и исходные пропорции.

**Inputs:** Stage 9; floor/profile contact points, размеченные stance/flight intervals Stage 7.

**Implementation:**
1. Создать `solve/contacts.py`: height/velocity/visibility+hysteresis detector, anchors в world space, flight events/confidence.
2. Добавить contact residuals к temporal solve для ног/root/pelvis; double stance решать вместе. Ground clamp только при contact/penetration, не минимум стоп каждого кадра.
3. Заполнить bundle contacts/events; создать `qa/contacts.py`, tests для single/double support и jump.

**Output:** bundle с воспроизводимыми anchors/events, видео и drift/jump metrics.

**Verification:** на high-confidence stance median максимального drift по сегментам ≤ 2 см, p95 ≤ 4 см; отсутствие stance = unavailable. Jump fixture сохраняет flight/height в frozen tolerance; reprojection/limits regression проверяются одновременно.

**Exit criteria:** численные contact gates проходят на tuning; результаты holdout опубликованы без подбора по нему. Если глубина неидентифицируема, явный review status, не искусственное прижатие.

**Dependencies:** 7, 9. **Next dependency:** Stage 11 использует стабильные body/head channels, Stage 13 — contacts API.

## Stage 11 — видеолицо и независимая обработка головы/глаз

**Goal:** заменить baseline фильтры на confidence-aware video solve без аудио зависимости.

**Inputs:** Stage 6 face observations, Stage 10 body, face/profile mapping и Stage 7 annotations.

**Implementation:**
1. Создать `observations/face_quality.py`: ROI size, blur, geometry/pose consistency; один дополнительный ROI pass только на слабых intervals, с обратным crop mapping.
2. Создать `solve/face.py`: neutral offsets по reliable intervals, отдельные filters blink/brows/mouth, bounded short gaps и lower-confidence fallback длинных gaps.
3. Доработать `solve/head.py`: remove matrix scale, neck/head limits/distribution, head-local gaze; owner rules применяет compositor. Добавить `qa/face.py`, mapping/event fixtures.

**Output:** video-only face/head bundle, temporal close-up segments и coverage report.

**Verification:** blink/closure timing на fixtures не выходит за frozen event tolerance; neutral offsets не требуют первого neutral кадра; head и eye rotation не удваиваются. Надёжный видеосигнал ≥ 90% видимых face frames основного benchmark, прочие gaps перечислены.

**Exit criteria:** video-only runnable; neutral/blink/head checks проходят, непокрытые участки имеют причины и статус.

**Dependencies:** 6, 7, 10. **Next dependency:** Stage 12 добавляет только доступные audio cues, не заменяет видео эмоции.

## Stage 12 — audio adapters и confidence-aware lip fusion

**Goal:** улучшить артикуляцию речи/пения при слабом лице, не блокируя silent/video-only jobs.

**Inputs:** Stage 11, исходное аудио/PTS, song и отдельный speech source с размеченными lip events; optional lyrics.

**Implementation:**
1. Создать `observations/audio.py`, `audio/{envelope,rhubarb,demucs}.py`, отдельный environment для separation. Минимальный compatibility/speed experiment Demucs на коротком mixed audio выполнить здесь; lock после smoke, не в Stage 0.
2. Skip separation для чистого speech; silent — без вызова backends. Сохранять envelope/voicing/cues и offset/confidence; отказ backend возвращает video-only с warning. Original mix остаётся soundtrack.
3. Создать `solve/lips.py`: нормированное channel-dependent video/audio fusion, closure и взаимные ограничения pucker/stretch/jaw, profile teeth exposure; audio не генерирует brows/emotions.
4. Дополнить `qa/lips.py`; записывать явный alignment correction в manifest, без таймкодов песни.

**Output:** fused bundle и singing/speech close-ups; video-only fallback работает тем же pipeline.

**Verification:** closure/open error median ≤ 80 мс, p95 ≤ 160 мс по annotations; отдельно video-only vs fusion на frozen intervals. Silence, failed separation и good-video dominance fixtures; A/V timing gates сохраняются.

**Exit criteria:** измеренный выигрыш/регрессия опубликованы, lip gates проходят в принятой области; unsupported singing не считается точным по RMS correlation. Backend failures не уничтожают body/video face result.

**Dependencies:** 5, 7, 11. **Next dependency:** Stage 13 дополняет движение кистей; общий bundle не меняет семантику.

## Stage 13 — temporal кисти, ROI и совместные ладони

**Goal:** заменить baseline hands устойчивым target-rig решением.

**Inputs:** Stage 12 bundle, raw hands/body observations, Xandra wrist/finger axes/limits и palm geometry.

**Implementation:**
1. Создать `observations/hand_identity.py`: body wrist + temporal continuity + explicit mirroring; full frame/ROI, один дополнительный pass на слабых intervals.
2. Создать `solve/hands.py`: wrist в solved forearm frame, finger directions к rest lengths, metacarpal/twist rules, constraints и confidence gap policy.
3. Добавить bilateral palm-contact solve, использующий тело/forearms как переменные в пределах profile correction budget; contact не выводить только из 2D близости. Выпустить новый bundle, не менять Blender actions после публикации.
4. Создать `qa/hands.py`, tests на L/R crossing, occlusion, rotating forearm и prayer pose.

**Output:** новые hand channels/contact events и сравнение проблемных жестов.

**Verification:** нет необъяснимых L/R swaps, wrist/finger limits соблюдены, lengths неизменны; joint palm solution улучшает target residual и не превышает correction budget. Occlusion выходит с masks; after-pass velocity/contact gates проверяются.

**Exit criteria:** hand tests и frozen gesture gates проходят; нет зависимости от animated GLB и ручных frame windows.

**Dependencies:** 6, 10, 12. **Next dependency:** Stage 14 добавляет collision costs в совместное решение.

## Stage 14 — collision proxies внутри temporal solve

**Goal:** ограничить проникновения до дорогой проверки mesh, сохраняя жест.

**Inputs:** Stage 13, versioned body/cloth/palm proxies и margins, floor/obstacles scene profile.

**Implementation:**
1. Создать `geometry/proxies.py`; добавить проверенные proxy definitions в character/scene profile с новым hash/version.
2. Добавить `E_collision` в общие residuals Stage 8–13; arms/hands/body исправлять совместно в ограниченном angular/translation budget.
3. Добавить `qa/collision_proxy.py`, диагностические contact samples; невозможное исправление — needs_review.

**Output:** proxy-constrained bundle и метрики penetration/correction относительно предыдущего bundle.

**Verification:** capsule/plane analytical fixtures, palm-chest/waist/prayer poses; correction limit, temporal/pose/contact regression gates. Proxy depth явно маркируется proxy, не surface acceptance.

**Exit criteria:** bounded correction runnable и сохраняет constraints; невозможно молча переставить плечо на десятки градусов.

**Dependencies:** 9, 10, 13. **Next dependency:** Stage 15 проверяет реальные deform surfaces отдельно.

## Stage 15 — skinned surface QA и один bounded refinement

**Goal:** проверить геометрию, которую действительно увидит render.

**Inputs:** Stage 14 bundle/profile; список контролируемых skin/cloth/palm surfaces и margins.

**Implementation:**
1. Создать `blender/evaluate_surfaces.py`, `geometry/{surface_distance,libigl_adapter}.py`: evaluated deformed meshes по bundle samples; сначала необходимый backend smoke на малом mesh, измерить память.
2. Создать `qa/surface_collision.py`; сначала diagnostic selected segments, затем coverage по всем требуемым samples для окончательного surface pass. Отдельно указать исключённые hair/finger–finger surfaces.
3. Residual evidence вернуть solver, разрешить максимум один дополнительный refinement с фиксированным бюджетом. Новый bundle hash → повторный surface/velocity/contact QA только изменённого результата.

**Output:** surface depths/coverage и либо approved candidate bundle, либо needs_review с offending intervals.

**Verification:** known signed-distance/intersection fixtures и реальные контрольные жесты; high-confidence проверяемые surfaces p95 depth ≤ 2 мм, max ≤ 5 мм; нет данных = unavailable. После refinement проходят temporal/contact/correction gates.

**Exit criteria:** surface gate доказан на заявленном наборе; невозможность исправления завершает job объяснимо, без бесконечных retries и ложной физической гарантии.

**Dependencies:** 14, 2. **Next dependency:** Stage 16 рендерит только bundle с известным QA статусом.

## Stage 16 — постановка, preview/final и художественная приёмка

**Goal:** довести вид до согласованного clip02 ориентира без подмены motion качеством камеры.

**Inputs:** Stage 15 candidate bundle, bedroom preset/look profile и baseline compare.

**Implementation:**
1. Создать `render/camera.py`: presentation framing по траектории/bounds, smooth follow, stable FOV; reference camera остаётся Stage 7 calibration, отдельный render view.
2. Довести versioned bedroom lighting/material/exposure/hair profile; face-closeup использует тот же bundle/materials. Preview 540×960, final 1080×1920; 16/64 EEVEE samples — candidates, окончательно выбрать по измерению flicker/time.
3. Создать `qa/framing.py`, final policy/preflight: whole-clip preview и проблемные face/hand segments → automatic gates → final. Добавить `--face-closeup`; errors optional outputs отдельны.

**Output:** final `video.mp4`, compare, optional close-up, quality/report с source/style references.

**Verification:** 100% planned frames проходят preset framing (intentional crops явно разрешены), codec/timeline gates; визуально проверить короткие temporal segments для лица, жестов, теней, волос/flicker и узнаваемости рядом с clip02. Объективные motion метрики сравнить в reference camera/world space.

**Exit criteria:** measurable gates и художественная приёмка достигнуты в заявленном input class; при неудовлетворительном виде сохранить конкретный defect/evidence, а не бесконечно менять samples/gains.

**Dependencies:** 5, 11–15. **Next dependency:** Stage 17 делает дорогие работающие jobs восстанавливаемыми.

## Stage 17 — Job API, SQLite, cache и recovery по реальной потребности

**Goal:** не терять дорогой render/solve при сбое и безопасно переиспользовать стадии при изменении параметров.

**Inputs:** Stage 16 working pipeline, реальные artifacts/стоимости Stage 5–16. JSON manifest остаётся частью результата.

**Implementation:**
1. Создать `jobs.py`, `execution/{dag,state,keys,atomic,workers}.py`; обернуть существующие чистые stage functions. SQLite transactions/leases, queued/running и terminal statuses; UI API пока без UI.
2. Stage key = input hashes + relevant params + schema/algorithm/model/runtime versions. Tracking не зависит от scene; face params не инвалидируют separation. Публиковать только validated artifacts через temp+atomic rename; bundle immutable.
3. Добавить `xms status/resume/rerender`, frame manifest с hashes/decode/size checks; recovery пересоздаёт missing/corrupt frames, не смешивает версии. Rerender читает approved bundle.
4. Cancellation/heartbeat/timeouts/memory limits, один GPU worker; transient retry максимум один, deterministic mismatch без retry, ROI/refinement бюджеты сохраняются при resume. Free-disk оценка по измеренному frame size с запасом.
5. Создать `tests/integration/test_recovery.py`, `tests/test_invalidation.py`; миграция ранних JSON runs только если поддержка нужна, иначе явный unsupported legacy-run error.

**Output:** resumable jobs на тех же pipeline contracts, selective rerender без tracking.

**Verification:** прервать короткий render, испортить PNG, сменить exposure/scene/face params, симулировать worker death; проверить executed stage set, stale lease, bundle/frame hashes и atomic publication. Cancel не показывает success; второй job не перезаписывает первый.

**Exit criteria:** recovery/cache matrix проходит; результаты и QA согласованы с одной версией движения/сцены.

**Dependencies:** 16. **Next dependency:** Stages 18–20 используют готовый Job API вместо новой оркестрации.

## Stage 18 — optional GLB и производный .blend

**Goal:** экспортировать утверждённую анимацию без изменения главного render path.

**Inputs:** Stage 17 approved bundle, отдельно зарегистрированный compatible export GLB, profile mapping.

**Implementation:**
1. Создать `export/{gltf_adapter,validate}.py`: копия asset, named clip с local TRS и face weights каждого нужного mesh, neutral rest weights, preservation старых animations/resources. Export mismatch отклонять до записи.
2. Создать `blender/save_derived.py` для optional `.blend` только в job directory; embed bundle provenance, не сохранять canonical asset.
3. Добавить `--export-glb`, `--assembled-blend`, tests reload/FK/morph/resource preservation; skeleton-only export явно bones-only, не полный результат.

**Output:** optional animated GLB/derived blend, отдельные export statuses.

**Verification:** glTF validator 0 errors, reload selected bone/morph samples соответствует bundle в Stage 2 tolerances, resource hashes/старые clips сохранены; отказ optional export не удаляет main MP4.

**Exit criteria:** export round-trip доказан для зарегистрированной Xandra; render не зависит от него.

**Dependencies:** 2, 17. **Next dependency:** Stage 19 расширяет preflight/acceptance, не экспорт ради экспорта.

## Stage 19 — область входов, негативные cases и release gates

**Goal:** отличать автоматическое качественное завершение от ложного успеха на неподдерживаемом видео.

**Inputs:** Stage 18, frozen tuning/holdout и benchmark policy.

**Implementation:**
1. Доработать `ingest/subject.py`, `ingest/shots.py`: identity, ambiguity нескольких людей, cuts/strong camera changes; unsupported режим → needs_input, не фиктивная root motion. Не строить движущуюся camera reconstruction.
2. Дополнить `qa/policy.py` правилами tracking coverage, partial loss, неуверенной calibration, over-budget collision; full reports даже при раннем отказе.
3. Создать `tests/integration/test_acceptance.py` и benchmark command: positive song/dance/speech/non-neutral holdout, negative face/hand loss, multi-person, cuts, silent/VFR. Не ретюнить на holdout.
4. Свести измерения runtime/memory/storage на Mac, quality improvements vs baseline и unresolved limitations в `benchmarks/release_report.md`.

**Output:** release candidate CLI с final/compare/report на positive inputs и предсказуемыми negative statuses.

**Verification:** focused acceptance matrix по §13 архитектуры, включая обязательные outputs/hashes, timeline, limits, stance, face/lips/hands/surfaces/framing. Качественные thresholds не ослаблять автоматически; отсутствие независимого holdout не считается pass.

**Exit criteria:** agreed input class проходит gates, unsupported cases объяснимы; ограничения явно задокументированы, не объявлены исправленными по существованию MP4.

**Dependencies:** 7–18. **Next dependency:** Stage 20 предоставляет thin UI над принятым API.

## Stage 20 — минимальный локальный UI над Job API

**Goal:** дать пользовательский MP4 → результат flow без второй реализации pipeline.

**Inputs:** Stage 19 CLI/Job API/status/report; измеренные stage costs.

**Implementation:**
1. Создать `ui/` и локальный service adapter к Job API: выбрать файл/профиль, запустить/cancel/resume, показать statuses/preview/results.
2. ETA показывать как estimate на основе measured costs, не константное обещание; errors/review intervals переводить в понятные причины и ссылки на отчёт.
3. Параметры/валидация берутся из того же JobSpec; UI не делает собственные solves и не редактирует bundle. Один local user, без auth/cloud subsystem.

**Output:** runnable local UI; CLI остаётся полноценным входом.

**Verification:** один короткий job из UI и API fixture для failure/cancel/resume; JobSpec/outputs совпадают с CLI, состояние переживает закрытие UI. Не повторять весь benchmark при изменении только UI.

**Exit criteria:** upload/select/run/result flow работает через общий API. Это этап полного приложения; отдельная CLI поставка может закончиться без UI только по явно согласованному scope, не молчаливым исключением.

**Dependencies:** 17, 19. **Next dependency:** Stage 21 упаковывает работающие CLI/UI/environments.

## Stage 21 — packaging, doctor и окончательная поставка

**Goal:** воспроизводимо запустить принятую систему на целевом Mac из чистого каталога.

**Inputs:** Stage 20, измеренные environment/backend builds, asset/model manifests и release report.

**Implementation:**
1. Создать `environments/` с проверенными platform/backend locks, `doctor.py`, install/launch scripts и native distribution; тяжёлые assets/weights — явно управляемый store, не скрытый download при run.
2. Doctor проверяет binaries/models/codecs/profile compatibility/free disk и минимальный encode/render smoke; фиксирует unsupported builds. Разделить tracking/separation/Blender environments.
3. Оформить `README.md`, configuration/examples, instructions register/run/resume/rerender/optional exports, limits и troubleshooting; package без импортов из archive/чужого venv и абсолютных путей автора.
4. Выполнить install/start/smoke в другом рабочем каталоге. Основные benchmark результаты использовать из Stage 19, повторять затронутые gates только если packaging изменил runtime/asset versions.

**Output:** installable Xandra Motion Studio, CLI/UI, doctor и воспроизводимые profiles/environments.

**Verification:** clean-directory короткий input → required outputs, doctor ошибку missing asset/build сообщает до expensive work; canonical assets unchanged; package не читает archive scripts. Если installer smoke меняет runtime, соответствующая acceptance regression обязательна.

**Exit criteria:** продукт устанавливается/запускается по инструкции, quality/recovery gates подтверждены, все unresolved issues отражены в release report. После этого остановить реализацию, не расширять input class без отдельного запроса.

**Dependencies:** 18–20. **Next dependency:** нет; завершение согласованной поставки.

## Где измеряется переход от baseline к конечной архитектуре

| Замена / возможность | Первый stage | Evidence перехода |
| --- | --- | --- |
| Исполняемый bundle вместо старых NPZ/actions | 1 → 2 | Python FK + Blender numerical round-trip |
| Новый MP4 целиком из input observations | 4 | real end-to-end run + independent bundle render |
| Полный baseline body/head/face/hands | 6 | named mapping/ownership + temporal video |
| Новый temporal solve | 8 → 9 | synthetic recovery + frozen baseline comparison + stitching |
| Temporal foot contacts/flight | 10 | stance drift/jump/pose regression |
| Confidence face и audio articulation | 11 → 12 | coverage + annotated event timing |
| Temporal hands и совместные palm contacts | 13 | swaps/limits/gesture correction metrics |
| Proxy и real-surface collision | 14 → 15 | bounded corrections + surface penetration/coverage |
| clip02 style и final 1080×1920 | 16 | framing + temporal visual acceptance |
| Job DAG/cache/SQLite/resume | 17 | interrupted/corrupt/stale-key matrix |
| Optional exports | 18 | GLB/Blender reload из того же bundle |
| Generalization/input policy | 19 | независимый holdout и negative cases |
| Local UI и reproducible delivery | 20 → 21 | общий Job API + clean-directory installation |

В Stage 8 целевая функция намеренно вводится частями, а не упрощается навсегда: contacts входят в Stage 10, bilateral hands — в Stage 13, collision — в Stage 14; Stage 15 проверяет реальную поверхность и возвращает bounded correction в тот же solver. На каждом переходе baseline остаётся frozen comparator, новый output проходит прежний consumer bundle.

## Три наиболее рискованные технические неизвестные

1. **Точный перенос rest/local transforms Xandra в Blender и export asset.** Legacy `xcore.py` хранит world-delta matrices и WXYZ, новый bundle — local-rest delta и XYZW; bone inheritance, scale и pelvis/root могут дать скрыто неверный FK. Риск снимается Stage 2 numerical round-trip до tracking; GLB compatibility проверяется отдельно в Stage 18.
2. **Идентифицируемость глубины/root/contact и стоимость temporal оптимизации на Mac.** Монокулярные observations неоднозначны; более гладкое движение не обязательно точнее. Stage 7 фиксирует gauge и benchmark, Stage 8 измеряет objective/runtime, Stage 9 — память/швы, Stage 10 — stance/flight. Не маскировать плохую глубину подбором presentation camera.
3. **Качество и видимость face/hand observations именно для пения и перекрытых жестов Xandra.** Mapping, зубная экспозиция, нейтральные offsets и bilateral contacts могут быть ограничены входным сигналом; audio cues для пения тоже ненадёжны. Stage 6 даёт ранний evidence, Stages 11–15 измеряют events/coverage/geometry с bounded fallback. Повышение gain не считается восстановлением невидимой информации.

## Первый шаг после подтверждения

Начать **Stage 0, Implementation 1–3**: создать минимальный `experiments/runtime_smoke.py`, проверить имеющийся Python/pose task на одном кадре внешнего `sing.mp4`, затем в несохраняемой Blender-сессии извлечь rig metadata и выполнить action с двумя ключами одной кости. Записать `docs/development/STAGE0_RESULT.md`; если smoke успешен, сразу перейти к writer/reader/FK в Stage 1. До отдельного подтверждения production implementation не начинается.
