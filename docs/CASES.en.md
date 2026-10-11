# Wondful AI Renderer · Case Studies (CASES)

Render one base image from a single Blender camera, then swap the environment reference and environment prompt to produce multiple scenes in batch. Every result is compared, silhouette against silhouette, with the product Mask rendered by Blender.

## Acceptance method and thresholds

- **Silhouette IoU**: using the Blender Mask as the seed, OpenCV GrabCut segments the product in the generated image, and the intersection-over-union with the Mask is computed. Threshold **>95%**.
- **Outer contour ≤3 px**: the share of points on the Blender Mask's outer contour that lie within 3 px of the segmented contour in the generated image. Threshold **≥80%**.
- **Bottom offset**: the pixel difference between the lowest point of the segmentation and the lowest point of the Blender Mask (whether the ground contact has drifted). Threshold **<5 px** (absolute value).
- Generated images are first resized to the base image resolution (960×540) before comparison. Script: `cases/tools/sim.py` (identical to `simc.py` from the single-chair case; re-testing the single chair gives the same results).
- **Supplementary metric (not part of the thresholds)**: whether there is a generated-image edge (Canny) within 3 px of the Blender outer contour, `cases/tools/edgechk.py`. It does not depend on segmentation and is used to tell "genuinely misaligned" apart from "GrabCut segmentation failed" (transparent glass, thin tripod legs, dark backgrounds, props touching the product). Heavily textured backgrounds make it more lenient.
- The metrics are approximate measurements.

## Overview

| Case | Scene | Silhouette IoU | Outer contour ≤3 px | Bottom offset | Supplementary edge | Result |
|---|---|---|---|---|---|---|
| Xiaomi YU7 | Snowy slope / Grassland / Desert / Mud | 96.8% / 97.2% / 96.7% / 96.2% | 84% / 88% / 86% / 86% | 0 / +2 / +4 / +1 px | — | ✅ ×4 |
| SheenChair single chair | Nordic / Wabi-sabi / Loft / Mediterranean | 99.2% / 97.9% / 99.0% / 98.4% | 100% / 92% / 99% / 99% | 0 / 0 / +1 / 0 px | 99% / 100% / 98% / 100% | ✅ ×4 |
| Running shoe | Rainy city street at night | 95.9% | 79% | -4 px | 100% | ❌ |
| Running shoe | Sports track at dawn | 97.9% | 94% | +0 px | 99% | ✅ |
| Chronograph watch | Black marble macro | 97.8% | 84% | +0 px | 91% | ✅ |
| Chronograph watch | Outdoor rock climbing | 96.4% | 70% | +0 px | 91% | ❌ |
| Vintage camera | Vintage study | 93.7% | 91% | +3 px | 97% | ❌ |
| Vintage camera | Street film look | 90.4% | 84% | +2 px | 100% | ❌ |
| Iridescent table lamp | Minimalist bedroom at night | 66.8% | 47% | -199 px | 100% | ❌ |
| Iridescent table lamp | Café | 97.5% | 93% | -1 px | 94% | ✅ |

The reason for each of the 5 images that missed the thresholds is written out under its case: on the overlays the product position has essentially not moved; the main cause is GrabCut failing to segment transparent glass, thin tripod legs, dark backgrounds and props touching the product. The vintage camera was redone with its real materials (silver metal lens, black leather body); both images now miss only on IoU (outer contour and bottom offset pass).

---

## Case 1: Xiaomi YU7 · one base image, four outdoor scenes

| Blender clay model | Snowy slope | Grassland |
|---|---|---|
| ![](showcase/yu7_clay.jpg) | ![](showcase/yu7_snow.jpg) | ![](showcase/yu7_grass.jpg) |
| **Desert** | **Mud** | |
| ![](showcase/yu7_sand.jpg) | ![](showcase/yu7_mud.jpg) | |

- Model: free Sketchfab model by Ddiaz Design (about 580k faces), used for showcase purposes only; the repository does not distribute the model file.
- Camera: low 3/4 front-side angle looking slightly upward, with the car parked at the top of a slope / on the ground; the horizon must stay consistent across all four scenes.
- References: the product shape and paint references were official Xiaomi YU7 / YU7 GT promotional images (copyright Xiaomi, used as reference only, not distributed with the repository); no person reference was used.
- Prompt essentials (the original prompts were not archived separately; below are the requirements settled on at the time):
  - Product appearance: bright emerald-green paint (not darkened), thick clear coat, large soft elongated gradient highlights (after the official YU7 GT images); must keep the full-width light bar, five-spoke wheels, side mirrors, window lines and the "Xiaomi YU7" front license plate (in the first mud render the plate was altered; adding this requirement fixed it).
  - Environment/style: cold and bright for the snowy slope, natural sunlight for the grassland, warm light with blowing sand for the desert, overcast for the mud; ground color reflected onto the lower body, solid contact shadows, restrained color grading, fine grain.
- Add-on settings: not archived separately at the time.

| Scene | Silhouette IoU | Outer contour ≤3 px | Bottom offset |
|---|---|---|---|
| Snowy slope | 96.8% | 84% | 0 px |
| Grassland | 97.2% | 88% | +2 px |
| Desert | 96.7% | 86% | +4 px |
| Mud | 96.2% | 86% | +1 px |

## Case 2: Designer chair · four interior styles

| Blender base image | Nordic living room | Wabi-sabi tea room |
|---|---|---|
| ![](showcase/chair_base.jpg) | ![](showcase/chair_nordic.jpg) | ![](showcase/chair_wabi.jpg) |
| **Industrial Loft** | **Mediterranean terrace** | |
| ![](showcase/chair_loft.jpg) | ![](showcase/chair_terrace.jpg) | |

- Model: Khronos glTF Sample Assets · SheenChair, © 2020 Wayfair LLC, CC0 1.0 (https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/SheenChair).
- Base image: Blender 5.1.2, Cycles 48 spp, 960×540, low 3/4 front-side angle, script `chair_scene.py`; base image `chair_base.png`, Mask `chair_mask.png`.
- References and prompts: one environment/style description per scene (Nordic living room / wabi-sabi tea room / industrial Loft / Mediterranean terrace). The original prompts, environment references and add-on settings were not archived separately at the time; newer cases now store a prompt.md for each scene.

| Scene | Silhouette IoU | Outer contour ≤3 px | Bottom offset |
|---|---|---|---|
| Nordic living room | 99.2% | 100% | 0 px |
| Wabi-sabi tea room | 97.9% | 92% | 0 px |
| Industrial Loft | 99.0% | 99% | +1 px |
| Mediterranean terrace | 98.4% | 99% | 0 px |

## Case 3: Running shoe · rainy city night / sports track at dawn

| Blender base image | Clay model | Rainy city street at night | Sports track at dawn |
|---|---|---|---|
| ![](showcase/shoe_base.jpg) | ![](showcase/shoe_clay.jpg) | ![](showcase/shoe_rain.jpg) | ![](showcase/shoe_track.jpg) |

- Model: Khronos glTF Sample Assets · MaterialsVariantsShoe (https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/MaterialsVariantsShoe)
- License and attribution: © 2021 Shopify, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
- Base image: Blender 4.3.2, Cycles 64 spp, 960×540, AZ=45 EL=1.6 TZ=0.4 FILL=0.75 (camera on the toe side, 3/4 front-side, looking slightly down; 50 mm). Uses the glTF default material variant (light-blue knit).
- Settings: 1 main reference per scene in the Environment/Style group, nothing in the Product Shape and Person groups; render mode Standard; Strict Composition Lock / Structure Control Maps / Local Structure Repair on; Identity Preserve: can be left off (no logo on the upper; the embossed lettering on the midsole is kept from the base image).
- Full prompts, reference images, all attempts and overlays: [`cases/shoe/prompt.md`](cases/shoe/prompt.md)

**Product appearance (fixed)**

```text
[Product appearance] Light-blue knit mesh running shoe with a white foam midsole, dark-gray laces and collar lining, two translucent gray-blue TPU overlays on the side, and a heel pull tab. Keep the shoe's silhouette, proportions, position in the frame and viewing angle from image 1 exactly unchanged; do not change the shoe shape, the lacing path or the overlay shapes; do not add any logo or text.
```

**Rainy city street at night** · environment reference [`cases/shoe/refs/ref_rain.png`](cases/shoe/prompt.md) (see image below) (main reference; take: cold blue ambient light + magenta/cyan neon rim light from the back left, wet-ground reflections and a rainy atmosphere; do not take: the signs and object positions in the image.)

<img src="showcase/refs/shoe_rain.jpg" width="360" alt="Reference image">

```text
[Environment/style] Rainy city street at night: the shoe stands on wet black asphalt; puddles in the distance reflect magenta and cyan neon and warm streetlights; large blurred bokeh in the background; fine rain in the air; cold blue ambient light, with neon rim light from the back left tracing the edges of the upper. The road directly beneath the sole is darker, with only one clear dark contact shadow; the reflection is faint and does not hug the midsole; the edge of the white midsole contrasts clearly with the road. Cinematic sneaker ad campaign.
English: light-blue knit running sneaker on a wet neon-lit city street at night in the rain; keep the exact silhouette, scale, position and camera angle of image 1.
```

**Sports track at dawn** · environment reference [`cases/shoe/refs/ref_track.png`](cases/shoe/prompt.md) (see image below) (main reference; take: low-angle golden morning light from the back right, long shadows, thin mist and the red track texture; do not take: object positions on the track.)

<img src="showcase/refs/shoe_track.jpg" width="360" alt="Reference image">

```text
[Environment/style] Sports track at dawn: the shoe stands on a red rubber running track with white lane lines running diagonally; low-angle golden morning light from the back right creates long shadows and a warm rim light; light morning mist; blurred green turf and stands in the background, pale blue sky; natural contact shadow where the sole meets the track. Fresh, energetic sports-brand ad.
English: light-blue knit running sneaker on a red running track at sunrise with golden rim light; keep the exact silhouette, scale, position and camera angle of image 1.
```

| Scene | Silhouette IoU | Outer contour ≤3 px | Bottom offset | Supplementary edge | Result |
|---|---|---|---|---|---|
| Rainy city street at night | 95.9% | 79% | -4 px | 100% | ❌ |
| Sports track at dawn | 97.9% | 94% | +0 px | 99% | ✅ |

> Rainy city street at night: why it missed: outer contour ≤3 px is 79%, 1 percentage point short. On the overlay the shoe shape essentially coincides with the Blender silhouette; the deviation comes from the dark wet ground being close in color to the dark-gray laces/heel, so GrabCut segments poorly near the collar. Also: a "%"-like symbol (not readable text) appears in a neon sign in the background.

> Sports track at dawn: the embossed lettering on the midsole comes from the model's texture and is preserved after generation.

## Case 4: Chronograph watch · black marble macro / outdoor rock climbing

| Blender base image | Clay model | Black marble macro | Outdoor rock climbing |
|---|---|---|---|
| ![](showcase/watch_base.jpg) | ![](showcase/watch_clay.jpg) | ![](showcase/watch_marble.jpg) | ![](showcase/watch_rock.jpg) |

- Model: Khronos glTF Sample Assets · ChronographWatch (https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/ChronographWatch)
- License and attribution: © 2025 Darmstadt Graphics Group GmbH, CC BY 4.0; model and textures by Eric Chadwick, derived from the Sketchfab model "Chronograph Watch Mudmaster" (graphiccompressor, CC BY 4.0). The Khronos / 3D Commerce / DGG marks on the dial are trademarks of their respective owners and are not covered by the CC BY license.
- Base image: Blender 4.3.2, Cycles 64 spp, 960×540, AZ=-25 EL=0.7 FILL=0.8 (dial facing forward, 3/4 front left, looking slightly down; 50 mm). Blender 4.3.2's built-in glTF importer throws an error (TypeError) on this model's KHR_materials_variants, so the "Midnight Gold" variant was first baked into the default material and the variants extension removed before import (geometry unchanged).
- Settings: 1 main reference per scene in the Environment/Style group, nothing in the Product Shape and Person groups; render mode Standard; Strict Composition Lock / Structure Control Maps / Local Structure Repair on; Identity Preserve: on (the dial has wordmarks/logos), edge expansion 2 px; Hard Restore off.
- Full prompts, reference images, all attempts and overlays: [`cases/watch/prompt.md`](cases/watch/prompt.md)

**Product appearance (fixed)**

```text
[Product appearance] Multifunction chronograph watch with a champagne-gold metal case, an octagonal bezel with silver screws and pushers, a black dial, large gold numerals 12/3/6/9 and hands, a small LCD window below, a gold diamond-pattern strap and gray plastic parts. Keep the watch's silhouette, proportions, position in the frame and viewing angle from image 1 exactly unchanged; keep the hand positions and large numerals on the dial unchanged; do not add any new text or logo.
```

**Black marble macro** · environment reference [`cases/watch/refs/ref_marble.png`](cases/watch/prompt.md) (see image below) (main reference; take: low-key studio lighting — long strip highlights from a narrow softbox at the top left, a warm gold accent light on the right, a deep black background, reflections on polished stone; do not take: the specific direction of the stone veining.)

<img src="showcase/refs/watch_marble.jpg" width="360" alt="Reference image">

```text
[Environment/style] Black marble macro: the watch stands on a polished black marble countertop with fine white and pale-gold veining; a narrow softbox at the top left creates long strip highlights on the metal case and the countertop; a faint warm-gold accent light on the right; the background falls off into deep black; the countertop shows a clear reflection of the watch. Low-key luxury watch ad campaign, macro texture.
English: champagne-gold chronograph watch standing on polished black marble, low-key luxury macro lighting; keep the exact silhouette, scale, position and camera angle of image 1.
```

**Outdoor rock climbing** · environment reference [`cases/watch/refs/ref_rock.png`](cases/watch/prompt.md) (see image below) (main reference; take: golden evening sunlight from the left, crisp hard shadows, granite texture and blurred distant mountains; do not take: the position of the climbing rope (it is required to be at the far right of the frame).)

<img src="showcase/refs/watch_rock.jpg" width="360" alt="Reference image">

```text
[Environment/style] Outdoor rock climbing: the watch stands on a high-mountain granite ledge with rough, lichen-covered rock; blue ridgelines and valleys blurred in the distance; golden evening sunlight from the left, crisp shadows, the metal case reflecting warm light and sky blue. An orange climbing rope and steel carabiner are placed only at the far right edge of the frame, clearly apart from the watch, not touching or covering it; there is a clear dark contact shadow between the bottom of the watch and the rock. Rugged outdoor tool-watch ad.
English: champagne-gold chronograph watch standing on a granite mountain ledge at golden hour; keep the exact silhouette, scale, position and camera angle of image 1.
```

| Scene | Silhouette IoU | Outer contour ≤3 px | Bottom offset | Supplementary edge | Result |
|---|---|---|---|---|---|
| Black marble macro | 97.8% | 84% | +0 px | 91% | ✅ |
| Outdoor rock climbing | 96.4% | 70% | +0 px | 91% | ❌ |

> Black marble macro: tiny text on the dial, such as the city abbreviations and the 3D Commerce wordmark, is slightly distorted in the 1376 px output (it is also very small in the base image); the large numerals and hands remain correct.

> Outdoor rock climbing: why it missed: IoU and bottom offset pass, but outer contour ≤3 px is only 70%. The overlay shows the case itself coinciding with the Blender silhouette, but both generations placed the carabiner/climbing rope pressed right against the right side of the case (bbox right edge +18 px), and GrabCut merged it into the watch. Tiny text on the dial is slightly distorted.

## Case 5: Vintage camera · vintage study / street film look

| Blender base image | Clay model | Vintage study | Street film look |
|---|---|---|---|
| ![](showcase/camera_base.jpg) | ![](showcase/camera_clay.jpg) | ![](showcase/camera_study.jpg) | ![](showcase/camera_street.jpg) |

- Model: Khronos glTF Sample Assets · AntiqueCamera (https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/AntiqueCamera)
- License and attribution: © 2018 UX3D, CC0 1.0 (https://creativecommons.org/publicdomain/zero/1.0/); author Maximilian Kamps. The UX3D logo on the model is a UX3D trademark.
- Base image: Blender 4.3.2, Cycles 64 spp, 960×540, AZ=-40 EL=0.65 TZ=0.5 FILL=0.86 (3/4 front left, standing eye height looking slightly down; 50 mm). The whole camera together with its tripod is treated as the product.
- Settings: 1 main reference per scene in the Environment/Style group, nothing in the Product Shape and Person groups; render mode Standard; Strict Composition Lock / Structure Control Maps / Local Structure Repair on; Identity Preserve: on (the body carries the UX3D logo), edge expansion 2 px; Hard Restore off.
- **Redo note (2026-10-07)**: the previous product appearance was mistakenly written as "wooden body, brass lens", but the model actually has a black leather body + silver chrome lens (only the tripod is wooden). This version corrects the product appearance (brass → silver), regenerates both reference images (the street reference no longer contains the word "Caf") and both scenes; the scenes now use light-colored walls/floors to set off the dark body and wooden tripod legs, which makes measurement easier. Each scene was tried 2 times in this round; all attempts (attempts 1–4) are in `cases/camera/attempts/`.
- Full prompts, reference images, all attempts and overlays: [`cases/camera/prompt.md`](cases/camera/prompt.md)

**Product appearance (fixed)**

```text
[Product appearance] Vintage folding bellows camera: the body is a box covered in dark gray-black leather, with black pleated leather bellows; the lens, lens board, front standard and the rail plate extending to the right below the lens are glossy silver metal (chrome finish), with dark glass at the center of the lens; below the body is a black metal tripod head (dotted with small silver studs, with a black handle extending to the left) and a silver metal mounting plate; it is mounted on a distressed walnut tripod, each leg a wide, flat twin-bar structure made of two wooden slats side by side (with light wood grain on the sides); the leg width must match image 1 exactly — do not draw them thinner; the feet end in light-colored metal caps. Keep the silhouette, proportions, position in the frame and viewing angle of the camera and tripod from image 1 exactly unchanged; keep the angle, width and ground-contact points of all three tripod legs unchanged; keep the lens silver metal, do not turn it brass or gold; do not add any text or logo.
```

**Vintage study** · environment reference [`cases/camera/refs/ref_study.png`](cases/camera/prompt.md) (see image below) (main reference; take: pale sage-green wall paneling, a large off-white rug, window light from the left + warm light from a brass desk lamp on the right, film grain; do not take: the specific positions of the bookshelves and desk.)

<img src="showcase/refs/camera_study.jpg" width="360" alt="Reference image">

```text
[Environment/style] Vintage study: all three legs of the camera stand firmly in the middle of a large light off-white wool rug (the rug's edges are far from the feet), with light oak herringbone flooring showing beyond the rug; directly behind the camera is floor-to-ceiling pale sage-green wall paneling, so the dark body and walnut tripod stand out clearly against the light green background; bookshelves and old leather-bound books (no text on the spines) only at the left and right sides of the frame; on a mahogany desk at the right, a brass desk lamp gives off warm yellow tungsten light, and a window on the left lets in soft golden window light; warm vintage tones, bright and airy, slight film grain, warm reflections on the silver lens, soft contact shadows where the feet meet the rug.
English: antique black folding bellows camera with a silver chrome lens on a walnut wooden tripod, standing in the middle of a large cream rug in a bright 1920s study with pale sage-green panelled walls; keep the exact silhouette, scale, position and camera angle of image 1.
```

**Street film look** · environment reference [`cases/camera/refs/ref_street.png`](cases/camera/prompt.md) (see image below) (main reference; take: Portra 400 film tones and grain, a pastel mint-green stucco wall, light-gray cobblestones, soft warm light from the left; do not take: the positions of the wooden door and the bicycle. The reference image has no signs or text at all.)

<img src="showcase/refs/camera_street.jpg" width="360" alt="Reference image">

```text
[Environment/style] Street film look: the camera stands on pale gray-white limestone cobblestones in a lane of an old European town; directly behind the camera is a blurred pastel pale mint-green stucco wall, so the dark body and walnut tripod stand out clearly against the light green background; on both sides, blurred green shutters, wooden doors, an off-white awning (with no text at all) and a bicycle leaning against the wall; soft warm evening light from the left, short faint shadows, soft contact shadows where the feet meet the cobblestones; Kodak Portra 400 film tones, fine grain, slightly faded highlights.
English: antique black folding bellows camera with a silver chrome lens on a walnut wooden tripod on pale limestone cobblestones in an old European lane, pastel mint-green stucco wall behind, Portra 400 film look; keep the exact silhouette, scale, position and camera angle of image 1.
```

| Scene | Silhouette IoU | Outer contour ≤3 px | Bottom offset | Supplementary edge | Result |
|---|---|---|---|---|---|
| Vintage study | 93.7% | 91% | +3 px | 97% | ❌ |
| Street film look | 90.4% | 84% | +2 px | 100% | ❌ |

> Vintage study: why it missed: IoU 93.7% (1.3 points short); outer contour ≤3 px and bottom offset pass. What the segmentation missed are fine details — the handle on the left of the tripod head, the rail plate extending to the right below the lens, and the light wooden slat edge on the outside of the front-right tripod leg; the body, bellows, all three tripod legs and their ground-contact points essentially coincide with the Blender silhouette. Appearance: the lens and standard are silver metal, but under the warm desk lamp the outer ring of the lens picks up a noticeable warm-gold reflection and looks somewhat champagne up close, not fully matching the cool white chrome of the base image. No readable text.

> Street film look: why it missed: IoU 90.4%; outer contour ≤3 px and bottom offset pass, and all four bbox edges are within 1 px. The light wooden slat on the outside of the front-right tripod leg was drawn narrower / blends into the light green wall, and the tripod-head handle, which falls in front of the green shutters, was missed by the segmentation. Appearance: the lens, standard and rail are silver chrome, matching the model; no readable text.

## Case 6: Iridescent glass table lamp · minimalist bedroom at night / café

| Blender base image | Clay model | Minimalist bedroom at night | Café |
|---|---|---|---|
| ![](showcase/lamp_base.jpg) | ![](showcase/lamp_clay.jpg) | ![](showcase/lamp_bedroom.jpg) | ![](showcase/lamp_cafe.jpg) |

- Model: Khronos glTF Sample Assets · IridescenceLamp (https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/IridescenceLamp)
- License and attribution: © 2022 Wayfair LLC, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); model and textures by Eric Chadwick.
- Base image: Blender 4.3.2, Cycles 64 spp, 960×540, AZ=-30 EL=0.55 FILL=0.8 (3/4 front left, seated eye height looking slightly down; 50 mm). Imported as is.
- Settings: 1 main reference per scene in the Environment/Style group, nothing in the Product Shape and Person groups; render mode Standard; Strict Composition Lock / Structure Control Maps / Local Structure Repair on; Identity Preserve: off (no logo).
- Full prompts, reference images, all attempts and overlays: [`cases/lamp/prompt.md`](cases/lamp/prompt.md)

**Product appearance (fixed)**

```text
[Product appearance] Table lamp: a grayish-brown (taupe) linen drum shade; the base is a clear spherical glass globe with an iridescent (rainbow thin-film) sheen, with a thin metal tube visible inside the globe; a bright silver metal neck and round base. Materials and colors follow the Blender materials in image 1. Keep the lamp's silhouette, proportions, position in the frame and viewing angle from image 1 exactly unchanged; keep the shade shape and the spherical base unchanged; do not add any text or logo.
```

**Minimalist bedroom at night** · environment reference [`cases/lamp/refs/ref_bedroom.png`](cases/lamp/prompt.md) (see image below) (main reference; take: a late-night interior, deep blue night outside the window with faint cool moonlight, the warm glow of the table lamp, Japanese minimalist materials; do not take: the positions of objects in the reference image.)

<img src="showcase/refs/lamp_bedroom.jpg" width="360" alt="Reference image">

```text
[Environment/style] Minimalist bedroom at night: late at night, the room is dim, and the table lamp is switched on as the main light source; the taupe shade glows with 2700K warm yellow light, casting a soft halo on the warm-gray microcement wall and the light oak nightstand top; in the background a low bed with off-white linen bedding, and beyond sheer curtains a deep blue night sky, with faint cool moonlight by the window; the spherical glass base refracts the warm light with iridescent highlights, and the outline of the glass globe has a clear bright edge. Quiet, cozy Japanese minimalist home-furnishing ad.
English: table lamp with taupe linen drum shade and iridescent glass globe base, switched on as the main light, on an oak nightstand in a dark minimalist Japandi bedroom late at night; keep the exact silhouette, scale, position and camera angle of image 1.
```

**Café** · environment reference [`cases/lamp/refs/ref_cafe.png`](cases/lamp/prompt.md) (see image below) (main reference; take: slanting evening window light from the left, amber-honey tones, shallow depth of field, a walnut tabletop; do not take: the specific positions of the latte and plant (they are required not to block the lamp).)

<img src="showcase/refs/lamp_cafe.jpg" width="360" alt="Reference image">

```text
[Environment/style] Café: the table lamp sits on a walnut café table, with a latte with latte art and a small potted plant on one side of the tabletop (not blocking the lamp); in the blurred background, a red brick wall, ceramic cups on wooden shelves, a large window and hanging plants; evening sunlight slants in from the left, amber-honey tones, shallow depth of field; the spherical glass base refracts the window light with iridescent highlights, and the base casts a natural contact shadow on the tabletop. Warm lifestyle ad.
English: table lamp with taupe linen drum shade and iridescent glass globe base on a walnut café table, warm afternoon window light, latte and plant nearby; keep the exact silhouette, scale, position and camera angle of image 1.
```

| Scene | Silhouette IoU | Outer contour ≤3 px | Bottom offset | Supplementary edge | Result |
|---|---|---|---|---|---|
| Minimalist bedroom at night | 66.8% | 47% | -199 px | 100% | ❌ |
| Café | 97.5% | 93% | -1 px | 94% | ✅ |

> Minimalist bedroom at night: why it missed: a measurement failure, not a misalignment. The transparent glass globe base caused GrabCut to keep only the shade (bottom -199 px); on the overlay the whole lamp coincides with the Blender silhouette pixel for pixel, and the supplementary edge metric is 100%. Attempt 1 (with a white-shade prompt) passed under the same method (IoU 96.4%), but the shade color was not faithful, so it was not used.

---

## How the images were generated

The scene images for Cases 3 to 6 (and the single-chair case) were generated by Hark's image generation tool following the add-on's prompt structure: image 1 = Blender Camera Base, image 2 = the main Environment/Style reference, prompt = product appearance + environment/style + the hard composition rules that the add-on's "Strict Composition Lock" appends. They are meant to demonstrate the add-on's workflow and expected results, and are not direct output of the add-on's Codex / Antigravity Provider. The environment reference images were generated by the same tool; no third-party copyrighted material is used.

## Acknowledgements / model licenses

- Xiaomi YU7: Sketchfab, Ddiaz Design (showcase use, not distributed)
- SheenChair: © 2020 Wayfair LLC, CC0 1.0
- MaterialsVariantsShoe: © 2021 Shopify, CC BY 4.0
- ChronographWatch: © 2025 Darmstadt Graphics Group GmbH, CC BY 4.0; Eric Chadwick (model and textures); derived from "Chronograph Watch Mudmaster" by graphiccompressor (Sketchfab, CC BY 4.0). The Khronos / 3D Commerce / DGG marks on the dial are trademarks
- AntiqueCamera: © 2018 UX3D, CC0 1.0; Maximilian Kamps. The UX3D logo is a trademark
- IridescenceLamp: © 2022 Wayfair LLC, CC BY 4.0; Eric Chadwick
- All of the Khronos models above come from https://github.com/KhronosGroup/glTF-Sample-Assets
