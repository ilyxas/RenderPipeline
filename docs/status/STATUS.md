# Текущий статус (на 2026-10-17)

## Коротко

- Пайплайн работает от видео до MP4 и анимированного GLB: CLI (`./xms run`), GUI (`./xms gui`), отчёты, сравнение, очистка.
- **Качество движения не принято ни для одного солвера.**
- Для просмотра результатов сейчас используется **`--solver viggle --channels full`** (внешний мокап Viggle + наша мимика/губы по звуку). На всех 8 диагностических клипах он заметно лучше `video` по глубине и касаниям; вердикт Ilya: «земля и небо».
- `video` остаётся солвером по умолчанию, но у него системные проблемы с глубиной и калибровкой (см. TODO).
- **Коллизии (Stage-15, эксперимент):** уточнение по уликам кожи (тело, рука–рука, касание) реализовано и функционально проверено на `066332e1`, но **отклоняется гейтами** — источник сохраняется, статус `needs_review`. Лучшее: тело p95 25.1→10.4 мм (slice 1) или рука–рука 25.5→18.6 мм (slice 2), но не одновременно при бюджете 5°. Подробно: [sessions/2026-10-12.md](sessions/2026-10-12.md).

## Солверы

| Солвер | Статус | Когда использовать |
|---|---|---|
| `viggle` | экспериментальный, лучший результат | смотреть, как выглядит движение; платно ~$0.05/с при первой отправке видео, потом кэш |
| `video` | по умолчанию, не принят | без сети и денег; глубина и скручивание рук ненадёжны |
| `temporal`, `baseline` | исторические, не приняты | только для сравнения |

Подробное сравнение: [docs/research/VIGGLE_COMPARISON.md](../research/VIGGLE_COMPARISON.md).

- Ретаргет Viggle доказан до видео: Viggle в rest Xandra даёт нулевые дельты, один повёрнутый плечевой сустав двигает только эту руку ровно на этот угол (`tests/test_viggle_retarget.py`).
- Stages 9–16 построены поверх `temporal` и только функционально проверены, не приняты.

## Персонаж и ассеты

- **С 2026-10-17 основной конфиг: `xandra/v10` (корсетное платье, `blender/Xandra_Wardrobe_v10.blend` = v09 с упакованными текстурами) + `loft/v4` (`blender/Loft_Studio_v10.blend`, передняя стена не видна камере; экспозиция по движку: EEVEE 0.0, Cycles 0.45).** Оба файла в Git LFS (вместе с `Xandra_Animations.blend`); v09 только локально. Подробно: [sessions/2026-10-17.md](sessions/2026-10-17.md).
- **Какие профиль и файл используются — смотреть и менять только через реестр:** Студия → «00 Персонаж и сцена» или `./xms assets status|use NAME|register FILE --base NAME`. Реестр ставит в конфиг только пару «профиль + файл с его хешем» (бэкап в `.xms/backups/`), изменённый `.blend` регистрирует как новую версию профиля после проверки в Blender. Ручная правка `.xms/config.json` 2026-10-17 и дала ошибку «asset hash mismatch».
- Модели MediaPipe и сцена спальни — в `asset_store/` (вне Git).

- Конфиг: `.xms/config.json` (в Git не попадает). Запасной для старого вида: `.xms/config.v3-top-shorts.json`.
- Для Wardrobe v07 правильный профиль **`v7`**: `.xms/config.v7-wardrobe-v07-asset.json`. Режим look `asset`: пайплайн не перекрашивает ассет, GLB с 8 весами, маской кожи и real-time конверсиями ткани. Проверочный прогон: `20261014-213538-4bf23915`. Подробно в [sessions/2026-10-14.md](sessions/2026-10-14.md). Профиль `v6` — это тот же ассет со старой подкраской, все его подмены теперь видны в манифесте и в отчёте.
- `blender/Xandra_Wardrobe_v07.blend` лежит вне Git, sha256 `f63cef81…`. Запасные конфиги: `.xms/config.v6-wardrobe-v07.json`, `.xms/config.v5-wardrobe.json` (v06; ассета на диске уже нет, не запустится), `.xms/config.v4-blacklace.json`.
- Каждый прогон пишет `look_overrides` (правки вида «было → стало»), `drivers` и `driver_inputs` (например `Sport_Set`), а `animation.export.json` — веса, маски и предупреждения.
- Профиль `v4` — Black Lace: `~/Xandra/model/Xandra_BlackLace_Fitting_v01.blend` (sha256 `c0d5de55…`).
- Профиль `v3` — топ и шорты: `blender/Xandra_Animations.blend` (Git LFS, sha256 `b20530fe…`).
- Скелет у v3–v7 одинаковый (341 кость, rest идентичен) → бандлы, пределы руки и карта Viggle подходят к обоим.
- Сцена спальни: `asset_store/scenes/clip02_scene.blend` (с 2026-10-16; раньше `~/codex/clip02_blender_kit`).
- Кэш Viggle: `~/.cache/xms/viggle/<sha256 видео>-fps24/`. Удалённый кэш клиент восстанавливает бесплатно из уже оплаченной задачи на стороне Viggle (`6f93dee`). Сейчас в кэше: 8 базовых клипов, `sing.mp4` и два ролика из uploads (спиной к столу; в розовом платье).
- Окружение: `.venv` на Python 3.13 по lock-файлу `environments/stage0-macos-arm64.txt` + scipy 1.18.1. Blender 4.5.13 LTS, Apple M4 (Metal).

## Рендер

- Движок `--engine eevee|cycles` (Студия: «Движок»). Cycles — направление для финалов: GPU Metal, OpenImageDenoise, motion blur, persistent data (−43 % времени).
- Сэмплы по умолчанию: EEVEE preview 16 / final 32; Cycles preview 32 / final 64. Любое значение — `--samples N` (Студия: «Сэмплы», «Авто» = по умолчанию).
- Preview 540×960, final 1080×1920 (разрешение задаётся отдельно `--resolution`); EEVEE тени 1 луч / 6 шагов. Final сначала рендерит весь фрагмент в preview (TODO 9).
- Замеры M4: Cycles 540×960 32 сэмпла ≈ 2 с/кадр; 1080×1920 64 сэмпла ≈ 15 с/кадр (без persistent data, заново не мерялось).
- PNG-кадры удаляются после успешной сборки; `./xms clean` для старых запусков (в Корзину).

## Проверки

- 99 unit-тестов, 1 пропускается (нужен реальный образец Viggle в `~/.cache/xms/viggle/text-probe/`, кэш удалён).
- Запуск: `PYTHONPATH=src:tests .venv/bin/python -m unittest discover -s tests -p 'test_*.py'`.

## Ключевые документы

- [README.md](../../README.md) — команды, флаги, раздел «What is still broken».
- [CLAUDE.md](../../CLAUDE.md) — правила работы.
- [docs/research/VIGGLE_COMPARISON.md](../research/VIGGLE_COMPARISON.md), [RENDER_SETTINGS.md](../research/RENDER_SETTINGS.md).
