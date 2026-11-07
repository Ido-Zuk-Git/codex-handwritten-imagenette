# Hand-written Imagenette classifier

No fitting, pretrained components, exemplar matching, or automated parameter search is used.
All constants are manually selected, interpretable visual design choices.
Validation images have not been viewed.

## Iteration 1: color, texture, silhouette, and line evidence

Implemented normalized HSV/RGB masks; spatial color fractions; edge orientation,
texture and smoothness; connected colored silhouettes; and straight-line evidence.
Rules cover sky/parachute, white disk/grass, brass, spaniel coats, olive fish/skin,
orange saws, and the rectangular geometry of machines/buildings.
Train: **22.20% (111/500)**. Validation: **24.80% (124/500)**.
No errors or timeouts after correcting a difference in OpenCV line-array layout.
The first smoke check had 500 runtime errors from that layout assumption (0%);
it was immediately corrected. Broad line/color clues alone are insufficient,
and their initial evidence scales over-favor rectangular machinery.
Comment coverage is incomplete in this draft and will be completed before delivery.

## Iteration 2: circle and geometric region sketches

Train: **26.00% (130/500)**. Validation: **29.00% (145/500)**.
No errors or timeouts. Added circle/ring tests, tire pairs, pale-disk silhouettes,
an analytic spaniel face sketch, an elongated olive-region sketch, and neutral
saw-bar contours. The shape sketches yield many accidental matches in machinery;
horn circle evidence also fires on yellow backgrounds. Next: require supporting
scene/color evidence and improve the balance between natural and rigid objects.

## Iteration 3: surface context and vehicle/building geometry

Train: **36.60% (183/500)**. Validation: **36.00% (180/500)**.
No errors or timeouts. Added color-conditional surface texture/solidity, lower road
context, dark tire silhouettes/pairs, and pitched-roof line pairs. Rebalanced
evidence with penalties for conflicting scene types. Tire pairs improve trucks,
but can confuse architectural windows. Orange hue still confuses warm interiors
with saw motors, and white-background cutouts can create false ball shapes.

## Iteration 4: orientation, polygon panels, pits, and reflective ridges

Train: **36.20% (181/500)**. Validation: **37.00% (185/500)**.
No errors or timeouts. Added a rotation-invariant perpendicular-edge test,
quadrilateral panels, skin-oval context, small dark-pit counts, and analytic
Hessian bright-ridge filters. Horn recall improves, but fish and warm interiors
also contain ridges. Most proposed polygon/skin-oval tests rarely fire; these
features are weak, not a semantic recognition substitute.

## Iteration 5: stricter object support

Train: **38.40% (192/500)**. **Validation: 39.80% (199/500)**.
Narrowed saw orange to vivid red-orange; rejected border-connected pale ball
components; required radial contrast around Hough ball outlines; added a dark
nose to the dog sketch; and downweighted brass ridges on solid yellow regions.
The stricter nose requirement loses too many genuine dogs, so the next version
will use it as supporting evidence rather than a veto.

## Iteration 6: part layouts and colored balls

Train: **39.20% (196/500)**. **Validation: 41.60% (208/500)**.
Added arched dark windows, filled colored spheres, a dark-display-over-cabinet
pump sketch, a low horizontal cassette-slot sketch, and modest contextual
weights for architecture and open grass/sky. Changed nose darkness from a veto
to supporting evidence, but even that still reduces dog recall too much.
Only the original training images have been viewed; both splits remain 500 images.

## Iteration 7: paired blade edges and circular texture

Train: **39.60% (198/500)**. Validation: **39.00% (195/500)**.
Added parallel neutral strips adjoining orange motor regions and low-threshold
circle candidates scored by radial gradient support plus internal pale pits.
Restored the original pale-blaze/dark-ear test because the nose requirement was
too brittle. The ball test helps training recall but also fires on printed
circular graphics. Neutral parallel strips occur in many manufactured objects.

## Iteration 8: steeples, crosses, and perspective

Train: **38.40% (192/500)**. Validation: **38.40% (192/500)**.
Added analytic pointed-tower and cross silhouettes, four-quadrant perspective
rays, and overcast sky context. These are too weak at the available resolution
and introduce architectural false positives on people and branches.

A proposed fur-structure-tensor feature was computed on training images only.
Its responses overlap heavily across classes, so it was discarded before
adding it to a scored iteration. No constants were selected automatically.

## Iteration 9: exposure and contrast normalization

Train: **35.40% (177/500)**. Validation: **35.00% (175/500)**.
No errors or timeouts. Tried per-image robust highlight scaling and percentile
contrast stretching with a noise-protection floor. This weakened the absolute
olive/dark/smooth-sky cues and changed edge densities. Reverted in iteration 10.

## Iteration 10: topological parts and contextual conjunctions

Train: **37.00% (185/500)**. Validation: **38.40% (192/500)**.
Added curved holes enclosed by brass, paired elevated dark speaker grilles,
a thick-to-thin saw silhouette, stronger pale-disk/pit combinations, and
contextual audio/road evidence. These additions remain brittle and do not
justify the accumulated evidence-scale complexity. Next: rewrite the evidence
combination around a single natural/rigid split and clearer physical groups.

## Iteration 11: coherent scene/part evidence

Train: **42.60% (213/500)**. Validation: **40.40% (202/500)**.
Replaced the accumulated score adjustments with one coherent manual rule set:
a shared rigidity clue, vertical-versus-horizontal layout ratios, natural-object
parts, and supporting scene evidence. This improves training accuracy over all
previous versions. No coefficient was fitted or selected by a search script.

## Iteration 12: shadow-tolerant tire arcs

Train: **41.80% (209/500)**. Validation: pending.
Added analytic dark-core/upper-surround disk kernels and tire-pair geometry,
plus smooth painted-panel evidence. Allowed a less elongated overall saw
silhouette because the motor and handle widen the object's bounding box.
Truck recall improves, but some lower windows and dark animal patches also
match the tire kernels.
