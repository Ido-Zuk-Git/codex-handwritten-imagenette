"""Hand-written visual evidence rules; no fitted parameters or stored examples.

All thresholds below are design choices in normalized image/color coordinates.
The scores express independent visual clues, not learned probabilities.
"""
import math
import numpy as np
from PIL import Image, ImageOps
import cv2

CLASSES = ('tench', 'english_springer', 'cassette_player', 'chain_saw', 'church',
           'french_horn', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute')


def ramp(value, low, high):
    # A clipped linear ramp lets a visual clue grow gradually between its stated limits.
    return float(np.clip((value - low) / (high - low), 0, 1))


def fraction(mask):
    # Pixel fractions make the same rules apply to every image size and aspect ratio.
    return float(np.mean(mask)) if mask.size else 0.0


def regions(mask):
    # Thirds separate sky overhead, objects in the center, and road/grass underneath.
    h, w = mask.shape
    # The central half avoids borders; top/bottom thirds capture outdoor scene layout.
    return (fraction(mask), fraction(mask[:h//3]), fraction(mask[2*h//3:]),
            fraction(mask[h//4:3*h//4, w//4:3*w//4]),
            fraction(np.concatenate((mask[:, :w//6].ravel(), mask[:, 5*w//6:].ravel()))))


def blobs(mask):
    # Opening removes isolated JPEG/color noise, while retaining object-sized colored regions.
    clean = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # External contours describe connected silhouettes rather than internal texture holes.
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    out = []
    for contour in contours:
        # Components below 0.3% of the frame are unreliable at the working resolution.
        area = cv2.contourArea(contour)
        if area < h*w*0.003:
            continue
        # Normalized bounding boxes support sphere, fish, and machine aspect-ratio tests.
        x, y, bw, bh = cv2.boundingRect(contour)
        # Circularity is one for an ideal disk; perimeter includes natural boundary roughness.
        perimeter = cv2.arcLength(contour, True)
        circularity = 4*math.pi*area / max(perimeter*perimeter, 1)
        # Fill and area distinguish disks, thin tubes, solid boxes, and irregular foliage.
        out.append((area/(h*w), (x+bw/2)/w, (y+bh/2)/h,
                    bw/w, bh/h, bw/bh, area/(bw*bh), circularity))
    # Largest components carry the main object's color; limit work on highly textured scenes.
    return sorted(out, reverse=True)[:20]


def features(path):
    # PIL is the sole file reader; EXIF orientation preserves the physical up direction.
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        # A 224-pixel maximum side preserves wheels/windows but bounds contour/Hough costs.
        image.thumbnail((224, 224), Image.Resampling.LANCZOS)
        rgb = np.array(image)
    # One OpenCV thread avoids oversubscription when the scorer runs multiple processes.
    cv2.setNumThreads(1)
    h, w = rgb.shape[:2]
    # HSV separates physical hue from illumination, useful for grass, sky, and brass.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    # OpenCV hue spans 180 units and saturation/value span 255; normalize to unit intervals.
    hue, sat, val = hsv[..., 0]/180, hsv[..., 1]/255, hsv[..., 2]/255
    # Unit RGB channels distinguish warm skin, olive fish, and blue sky with simple contrasts.
    red, green, blue = [rgb[..., channel].astype(np.float32)/255 for channel in range(3)]
    masks = {}
    # Blue/cyan medium-bright pixels commonly form parachute or church sky backgrounds.
    masks['sky'] = (blue > red*1.08) & (blue > green*0.98) & (val > 0.40) & (sat > 0.12)
    # Green vegetation is a contextual clue for balls, dogs, saws, fish, and churches.
    masks['green'] = (green > red*1.04) & (green > blue*1.10) & (sat > 0.18)
    # Bright, low-chroma surfaces include balls, spaniel coats, pump cabinets, and clouds.
    masks['white'] = (val > 0.68) & (sat < 0.23)
    # Neutral dark components include tires, cassette cases, hoses, and spaniel patches.
    masks['black'] = (val < 0.25)
    # Gray machinery and stone have modest chroma without the extreme brightness of balls.
    masks['gray'] = (sat < 0.20) & (val > 0.22) & (val < 0.80)
    # Brass is warm yellow with more red than green and substantially less blue.
    masks['gold'] = (hue > 0.075) & (hue < 0.18) & (sat > 0.35) & (val > 0.32)
    # Orange saw housings are vivid red-orange (about 5-27 degrees), redder than brass/stone.
    masks['orange'] = (hue > 0.015) & (hue < 0.075) & (sat > 0.60) & (val > 0.45)
    # Red pump panels and truck bodies straddle the wraparound of the hue circle.
    masks['red'] = ((hue < 0.035) | (hue > 0.96)) & (sat > 0.50) & (val > 0.35)
    # Brown dog fur is warm, fairly saturated, and darker than most human skin.
    masks['brown'] = (hue < 0.12) & (sat > 0.28) & (val > 0.15) & (val < 0.65) & (red > green*1.06)
    # Skin-colored arms frequently border a horizontally held tench; this clue is contextual.
    masks['skin'] = (red > green*1.08) & (green > blue*1.10) & (val > 0.45) & (sat > 0.16) & (sat < 0.60)
    # Olive/dull bronze is typical of tench scales, distinct from bright gold instrument tubing.
    masks['olive'] = (hue > 0.08) & (hue < 0.25) & (sat > 0.16) & (val > 0.15) & (val < 0.65)
    f = {}
    for name, mask in masks.items():
        # Region suffixes expose whole, top, bottom, central, and side color proportions.
        for suffix, value in zip(('', '_top', '_bottom', '_center', '_side'), regions(mask)):
            f[name+suffix] = value
    # Aspect ratio is useful for fish/saws (landscape) versus pump/tower views (portrait).
    f['aspect'] = w/h
    # Grayscale derivatives separate structural outlines from color alone.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # A mild sigma-one blur suppresses JPEG ringing before edge orientation measurement.
    smooth = cv2.GaussianBlur(gray, (0, 0), 1)
    # Sobel's 3-pixel derivatives estimate vertical and horizontal image changes.
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    # Threshold 70 selects visible edges on an eight-bit image, not tiny surface noise.
    strength = np.hypot(gx, gy)
    strong = strength > 70
    # A derivative ratio of 2 selects edges within about 27 degrees of an axis.
    vertical = strong & (np.abs(gx) > 2*np.abs(gy))
    horizontal = strong & (np.abs(gy) > 2*np.abs(gx))
    f['edge'] = fraction(strong)
    # Axis alignment is high for buildings, cassette decks, pumps, and trucks, low for fur.
    f['axis'] = (fraction(vertical)+fraction(horizontal))/max(f['edge'], 0.001)
    f['vertical'] = fraction(vertical)
    f['horizontal'] = fraction(horizontal)
    # Upper/lower edge layouts separate building towers from vehicle wheels/chassis.
    f['edge_top'] = fraction(strong[:h//3])
    f['edge_bottom'] = fraction(strong[2*h//3:])
    # High-frequency residual at sigma one describes fur/foliage, scales, and ball dimples.
    residual = np.abs(gray.astype(np.float32)-smooth.astype(np.float32))
    # Four gray levels select meaningful texture while excluding nearly uniform paint/sky.
    f['texture'] = fraction(residual > 4)
    f['texture_center'] = fraction(residual[h//4:3*h//4, w//4:3*w//4] > 4)
    f['smooth'] = fraction(strength < 20)
    f['sky_smooth'] = fraction(masks['sky'] & (strength < 20))
    # Canny hysteresis thresholds 60/140 produce object contours for geometric line analysis.
    edges = cv2.Canny(gray, 60, 140)
    # Segments longer than one eighth of the frame describe manufactured edges and architecture.
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, threshold=18,
                            minLineLength=min(h, w)//8, maxLineGap=4)
    totals = dict(line_h=0.0, line_v=0.0, line_d=0.0, line_top=0.0, line_bottom=0.0)
    if lines is not None:
        # Flatten either OpenCV return layout into four-coordinate line segments.
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            # Normalize segment length by the frame diagonal to preserve scale invariance.
            dx, dy = abs(float(x2-x1)), abs(float(y2-y1))
            length = math.hypot(dx, dy)/math.hypot(h, w)
            # The 3:1 slope ratio selects approximately horizontal/vertical manufactured lines.
            key = 'line_h' if dx > 3*dy else ('line_v' if dy > 3*dx else 'line_d')
            totals[key] += length
            # Top and bottom halves distinguish spires/canopies from low vehicle chassis.
            totals['line_top' if y1+y2 < h else 'line_bottom'] += length
    f.update(totals)
    # Bounding shape/color evidence is computed separately so rules remain interpretable.
    for name in ('white', 'black', 'gold', 'orange', 'red', 'brown', 'olive'):
        f[name+'_blobs'] = blobs(masks[name])
    # Return masks/gray only for further geometric analysis of this one image.
    return f, gray, masks


def geometry(gray, masks):
    # All shape tests below operate only on the current image, without stored exemplars.
    h, w = gray.shape
    f = dict(ball_shape=0.0, horn_coils=0.0, wheels=0.0, wheel_pair=0.0,
             dog_face=0.0, fish_body=0.0, saw_bar=0.0)
    # Unit luminance and chroma masks supply physically meaningful shape colors.
    light = gray.astype(np.float32)/255
    white = masks['white'].astype(np.float32)
    dark = (gray < 95).astype(np.float32)  # Below 95/255 includes dark fur, tires, and horn gaps.
    # A ball may be shaded gray; neutral highlights need not exceed the strict white threshold.
    pale = (masks['gray'] | masks['white']).astype(np.uint8)
    # Five-pixel closing bridges small ball dimples and printed logos, not separate objects.
    pale = cv2.morphologyEx(pale, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # Analyze bright components at three hand-selected illumination levels for shaded balls.
    for level in (110, 160, 205):
        # Neutral components above each level isolate white balls against darker surroundings.
        mask = pale & (gray > level).astype(np.uint8)
        # A second small closing fills fine texture before measuring circular boundaries.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Tiny dots and huge backgrounds are implausible standalone balls.
            area = cv2.contourArea(contour)
            if area < h*w*0.003 or area > h*w*0.65:
                continue
            # Reject pale regions touching the frame: these are usually product backdrops.
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bx <= 1 or by <= 1 or bx+bw >= w-1 or by+bh >= h-1:
                continue
            # Compare area to a minimum enclosing disk: balls should nearly fill it.
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            fill = area/max(math.pi*radius*radius, 1)
            # Circular outline and compact disk jointly reject dog legs and white cabinets.
            perimeter = cv2.arcLength(contour, True)
            circularity = 4*math.pi*area/max(perimeter*perimeter, 1)
            shape = ramp(fill, 0.60, 0.90)*ramp(circularity, 0.45, 0.80)
            # A sphere-sized component should span at least 6% of the shorter image side.
            shape *= ramp(radius/min(h, w), 0.025, 0.08)
            f['ball_shape'] = max(f['ball_shape'], shape)
    # Gradient-voting circles detect ball outlines, horn coils, and vehicle tires.
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=10,
                               param1=100, param2=24, minRadius=6,
                               maxRadius=int(min(h, w)*0.46))
    candidates = []
    if circles is not None:
        # Sixty-four perimeter samples resolve smooth circles at this image resolution.
        angles = np.arange(64)*2*math.pi/64
        for cx, cy, radius in circles.reshape(-1, 3)[:40]:
            # Coordinates of concentric rings measure hollow brass loops versus filled balls.
            rings = []
            for scale in (0.55, 0.85, 1.0, 1.15):
                # Clip samples to the image; edge-touching circles remain usable.
                xx = np.clip(np.rint(cx+radius*scale*np.cos(angles)).astype(int), 0, w-1)
                yy = np.clip(np.rint(cy+radius*scale*np.sin(angles)).astype(int), 0, h-1)
                rings.append((light[yy, xx], masks['gold'][yy, xx], dark[yy, xx], masks['white'][yy, xx]))
            # A golf ball has pale interior and a substantial contrast to its surroundings.
            inner = float(np.mean(rings[0][0]))
            outer = float(np.mean(rings[3][0]))
            neutral = float(np.mean(rings[0][3]))
            # A 15%-40% luminance rise is strong sphere evidence when the interior is white.
            ball = ramp(inner-outer, 0.08, 0.30)*ramp(neutral, 0.30, 0.80)
            # Require a brightness drop around most of the outline, not one dark wedge in a white backdrop.
            radial = fraction(rings[0][0]-rings[3][0] > 0.08)
            ball *= ramp(radial, 0.50, 0.80)
            # A disk should fit mostly within the frame; cropped macro balls use the dimple rule instead.
            if cx-radius < 0 or cy-radius < 0 or cx+radius >= w or cy+radius >= h:
                ball *= 0.25  # Keep only weak evidence from circles whose support is mostly outside the image.
            f['ball_shape'] = max(f['ball_shape'], ball)
            # Brass coils contain both yellow arcs and dark gaps, unlike a solid yellow wall.
            gold_ring = max(float(np.mean(ring[1])) for ring in rings[:3])
            dark_ring = max(float(np.mean(ring[2])) for ring in rings[:3])
            # Larger than 8% of the shorter dimension favors the horn's main coiled tubing.
            coil = ramp(gold_ring, 0.12, 0.50)*ramp(dark_ring, 0.10, 0.50)
            coil *= ramp(radius/min(h, w), 0.04, 0.12)
            f['horn_coils'] += coil
            # Tires have dark rims in the lower half and a lighter surrounding road/body.
            tire = ramp(float(np.mean(rings[1][2])), 0.35, 0.80)
            tire *= ramp(outer-float(np.mean(rings[1][0])), 0.03, 0.22)
            tire *= ramp(cy/h, 0.40, 0.65)*(1-ramp(radius/min(h, w), 0.20, 0.35))
            f['wheels'] = max(f['wheels'], tire)
            candidates.append((cx/w, cy/h, radius/min(h, w), tire))
    for i, (x, y, r, strength) in enumerate(candidates):
        for xx, yy, rr, other in candidates[i+1:]:
            # Truck tires are separated horizontally, nearly level, and of comparable size.
            pair = strength*other*(1-ramp(abs(y-yy), 0.04, 0.15))
            pair *= ramp(abs(x-xx), 0.12, 0.35)*(1-ramp(abs(math.log(r/rr)), 0.35, 0.90))
            f['wheel_pair'] = max(f['wheel_pair'], pair)
    # Hand-drawn region templates measure a spaniel blaze and an elongated fish flank.
    # A 21-square grid is a geometric sketch, generated by formula rather than an example.
    yy, xx = np.mgrid[-1:1:21j, -1:1:21j]
    # The narrow central blaze and broad pale muzzle contrast with dark lateral ears.
    blaze = (np.abs(xx) < 0.13) & (yy > -0.65) & (yy < 0.15)
    muzzle = ((xx/0.38)**2+((yy-0.30)/0.28)**2 < 1)
    ears = (((np.abs(xx)-0.62)/0.22)**2+((yy-0.05)/0.65)**2 < 1)
    # White center and dark sides are necessary; one homogeneous patch must not match.
    # A small dark nose at the upper muzzle helps reject shirt stripes and display panels.
    nose = ((xx/0.12)**2+((yy-0.12)/0.10)**2 < 1)
    pale_template = ((blaze | muzzle) & ~nose).astype(np.float32)
    nose_template = nose.astype(np.float32)
    ear_template = ears.astype(np.float32)
    # A long elliptical core with an outer ring models a displayed horizontal fish.
    fish_core = ((xx/0.90)**2+(yy/0.25)**2 < 1).astype(np.float32)
    fish_ring = (((xx/0.98)**2+(yy/0.65)**2 < 1) & (np.abs(yy) > 0.32)).astype(np.float32)
    # Olive/gold pixels are the flank; warm skin alone does not describe a complete fish.
    fish_color = (masks['olive'] | masks['gold']).astype(np.float32)
    for size in (28, 40, 56, 80, 112, 152):
        # Several scales cover close portraits, whole dogs, and smaller held fish.
        if size > min(h, w):
            continue
        # Linear resizing keeps the sketch's smooth geometric proportions at every scale.
        pt = cv2.resize(pale_template, (size, size), interpolation=cv2.INTER_LINEAR)
        et = cv2.resize(ear_template, (size, size), interpolation=cv2.INTER_LINEAR)
        # Resize the dark nose patch with the same anatomical proportions.
        nt = cv2.resize(nose_template, (size, size), interpolation=cv2.INTER_LINEAR)
        # Normalized kernels return region means, so the response has a size-independent meaning.
        pw = cv2.matchTemplate(white, pt/max(float(pt.sum()), 1), cv2.TM_CCORR)
        ed = cv2.matchTemplate(dark, et/max(float(et.sum()), 1), cv2.TM_CCORR)
        # Require both pale blaze/muzzle and dark ears, rather than their summed average.
        # The nose must be dark as well as the two lateral ears.
        nd = cv2.matchTemplate(dark, nt/max(float(nt.sum()), 1), cv2.TM_CCORR)
        response = np.minimum(pw, ed)  # Pale blaze/muzzle and dark ears remain usable when the nose is occluded.
        f['dog_face'] = max(f['dog_face'], ramp(float(response.max()), 0.35, 0.80))
        # Fish templates are twice as wide as high and located near the displayed object scale.
        fw = min(w, size*2)
        fh = max(8, size//2)  # An eight-pixel minimum preserves a meaningful flank region.
        ft = cv2.resize(fish_core, (fw, fh), interpolation=cv2.INTER_LINEAR)
        rt = cv2.resize(fish_ring, (fw, fh), interpolation=cv2.INTER_LINEAR)
        core = cv2.matchTemplate(fish_color, ft/max(float(ft.sum()), 1), cv2.TM_CCORR)
        ring = cv2.matchTemplate(fish_color, rt/max(float(rt.sum()), 1), cv2.TM_CCORR)
        # Color contrast to the surrounding ring rejects broad fields of olive vegetation.
        response = core-ring
        f['fish_body'] = max(f['fish_body'], ramp(float(response.max()), 0.15, 0.65))
    # A chainsaw bar is a long narrow gray/black region adjoining a compact orange motor.
    bar_mask = ((gray > 50) & (gray < 210) & ~masks['green']).astype(np.uint8)
    # Three-pixel closing repairs breaks from chain teeth without filling large gaps.
    bar_mask = cv2.morphologyEx(bar_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(bar_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Minimum-area rectangles permit the saw bar to point in any direction.
        (cx, cy), (a, b), angle = cv2.minAreaRect(contour)
        length, width = max(a, b), min(a, b)
        # Bars span at least a quarter of the shorter image side and are over four times as long as wide.
        if width < 3 or length < min(h, w)*0.25:
            continue
        aspect = length/width
        # Rectangular occupancy prevents tiny line fragments from masquerading as a blade.
        fill = cv2.contourArea(contour)/max(a*b, 1)
        bar = ramp(aspect, 3, 7)*ramp(fill, 0.40, 0.75)
        f['saw_bar'] = max(f['saw_bar'], bar)
    return f


def scene_features(gray, masks):
    # Scene-layout tests distinguish outdoor machines, buildings, and organic subjects.
    h, w = gray.shape
    f = {}
    # Sigma-two smoothness describes painted truck panels, unlike green leaf/grass texture.
    blur = cv2.GaussianBlur(gray, (0, 0), 2)
    # Seven luminance levels separate visible fine detail from smooth body paint.
    texture = np.abs(gray.astype(np.float32)-blur.astype(np.float32)) > 7
    for name in ('green', 'gold', 'white', 'brown', 'skin'):
        # Conditional texture reports surface character independent of that color's area.
        f[name+'_texture'] = fraction(texture & masks[name])/max(fraction(masks[name]), 0.001)
        # A 3x3 erosion separates colored surface interiors from fine edges and tubing.
        eroded = cv2.erode(masks[name].astype(np.uint8), np.ones((3, 3), np.uint8))
        f[name+'_solid'] = fraction(eroded)/max(fraction(masks[name]), 0.001)
    # Grassy golf settings often cover the lower half continuously, rather than a small patch.
    f['green_lower'] = fraction(masks['green'][h//2:])
    # Gray road beneath colorful truck panels is more diagnostic than arbitrary gray pixels.
    f['road'] = fraction((masks['gray'] | masks['black'])[3*h//4:])
    # Bright, low-texture borders describe studio product shots, including balls and cassette decks.
    f['white_border'] = (fraction(masks['white'][:h//8])+fraction(masks['white'][-h//8:])+
                         fraction(masks['white'][:, :w//8])+fraction(masks['white'][:, -w//8:]))/4
    # Dark elliptical silhouettes below mid-frame offer a perspective-tolerant tire detector.
    tires = []
    for threshold in (40, 65, 90):
        # Several fixed darkness levels accommodate sunlight, deep shadow, and rubber highlights.
        dark = (gray < threshold).astype(np.uint8)
        # Small closing repairs grooves and wheel hubs before silhouette measurement.
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            # Tires should be image-resolved yet substantially smaller than the truck body.
            a = cv2.contourArea(c)/(h*w)
            if not 0.001 < a < 0.08:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            # Perspective makes tires elliptical; height need not equal width exactly.
            shape = 1-ramp(abs(math.log(bw/bh)), 0.30, 0.90)
            # Filled silhouettes and lower-frame placement reject most lettering and windows.
            fill = ramp(cv2.contourArea(c)/(bw*bh), 0.35, 0.65)
            strength = shape*fill*ramp((y+bh/2)/h, 0.45, 0.65)
            tires.append(((x+bw/2)/w, (y+bh/2)/h, bw/w, bh/h, strength))
    f['tires'] = max((t[4] for t in tires), default=0.0)
    f['tire_pair'] = 0.0
    for i, (x, y, bw, bh, strength) in enumerate(tires):
        for xx, yy, ww, hh, other in tires[i+1:]:
            # Pairs are separated by a body length fraction and lie near a common axle baseline.
            pair = strength*other*ramp(abs(x-xx), 0.12, 0.30)
            # Allow perspective differences, but reject a large panel paired with a tiny dot.
            pair *= (1-ramp(abs(y-yy), 0.04, 0.13))*(1-ramp(abs(math.log(bh/hh)), 0.40, 1))
            f['tire_pair'] = max(f['tire_pair'], pair)
    # Roof apexes are pairs of long sloping straight lines meeting above their eaves.
    edges = cv2.Canny(gray, 60, 140)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, threshold=22,
                           minLineLength=min(h, w)//5, maxLineGap=5)
    slopes = []
    f['roof'] = 0.0
    f['long_vertical'] = 0.0
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            # Order endpoints left-to-right so rising and falling roof sides are explicit.
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            # Tall straight edges support pump cabinets or church towers.
            if abs(x2-x1) < 4:
                f['long_vertical'] = max(f['long_vertical'], abs(y2-y1)/h)
            # Nonvertical slopes between 1:4 and 2:1 cover common pitched-roof angles.
            if x2-x1 < w*0.12 or not 0.25 < abs((y2-y1)/(x2-x1)) < 2:
                continue
            # Store endpoint coordinates in unit frame space for the apex-pair comparison.
            slopes.append((x1/w, y1/h, x2/w, y2/h))
        for a in slopes:
            for b in slopes:
                # A rising left segment should meet a falling right segment at a shared peak.
                if a[3] >= a[1] or b[3] <= b[1]:
                    continue
                # One-tenth-frame tolerance allows modest gaps at a spire or roof ridge.
                meet = 1-ramp(math.hypot(a[2]-b[0], a[3]-b[1]), 0.02, 0.12)
                # Roofs must be substantial and in the upper two thirds, not ground clutter.
                length = ramp(min(a[2]-a[0], b[2]-b[0]), 0.10, 0.30)
                upper = 1-ramp((a[3]+b[1])/2, 0.55, 0.80)
                f['roof'] = max(f['roof'], meet*length*upper)
    return f


def detail_features(gray, masks):
    # Fine structure distinguishes rounded/dimpled balls, metal tubing, and tilted cabinets.
    h, w = gray.shape
    f = dict(rect_wide=0.0, rect_tall=0.0, rectangles=0.0, dimples=0.0,
             tube=0.0, gold_tube=0.0, skin_face=0.0)
    # A two-pixel blur makes orientation depend on object structure rather than JPEG texture.
    blur = cv2.GaussianBlur(gray, (0, 0), 2)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    # Gradient direction modulo pi treats both sides of a straight edge equally.
    angle = np.mod(np.arctan2(gy, gx), math.pi)
    strength = np.hypot(gx, gy)
    # Eighteen ten-degree bins resolve orthogonal panel edges even when the image is tilted.
    hist, _ = np.histogram(angle[strength > 40], bins=18, range=(0, math.pi), weights=strength[strength > 40])
    # Three-bin neighborhoods around perpendicular directions allow ten degrees of edge slop.
    pairs = [sum(hist[(i+offset)%18]+hist[(i+9+offset)%18] for offset in (-1, 0, 1)) for i in range(9)]
    f['oriented'] = float(max(pairs, default=0)/max(float(hist.sum()), 1))
    # Thresholded intensity regions expose cabinet faces and control windows at different lighting.
    for level in (55, 110, 165, 220):
        # Both the dark regions and light holes inside them have useful polygon contours.
        binary = (gray < level).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            # Discard letters and the image border; panels have at least 0.3% frame area.
            if area < h*w*0.003 or area > h*w*0.85:
                continue
            perimeter = cv2.arcLength(c, True)
            # Two-percent polygon simplification tolerates mild panel perspective and JPEG noise.
            poly = cv2.approxPolyDP(c, perimeter*0.02, True)
            # Convex quadrilaterals are the minimal physical description of cabinet panels.
            if len(poly) != 4 or not cv2.isContourConvex(poly):
                continue
            points = poly.reshape(-1, 2).astype(np.float32)
            # Measure four edge vectors to reject very acute arbitrary leaf/branch quadrilaterals.
            vectors = np.roll(points, -1, axis=0)-points
            lengths = np.linalg.norm(vectors, axis=1)
            # A panel cannot contain a degenerate edge of less than three working pixels.
            if float(lengths.min()) < 3:
                continue
            cosine = np.abs(np.sum(vectors*np.roll(vectors, -1, axis=0), axis=1)/(lengths*np.roll(lengths, -1)))
            # Perspective permits angles roughly 60-120 degrees, not sharp triangles.
            rectangular = 1-ramp(float(cosine.max()), 0.25, 0.60)
            if rectangular <= 0:
                continue
            x, y, bw, bh = cv2.boundingRect(poly)
            # Relative size favors resolved panels, not small chance rectangular textures.
            evidence = rectangular*ramp(area/(h*w), 0.003, 0.04)
            f['rectangles'] += evidence
            # Axis projections still provide wide/tall evidence for modestly tilted panels.
            f['rect_wide'] = max(f['rect_wide'], evidence*ramp(bw/bh, 1.2, 2.5))
            f['rect_tall'] = max(f['rect_tall'], evidence*ramp(bh/bw, 1.2, 2.5))
    # A shaded face-sized skin oval over clothing supports a person holding a fish/saw/horn.
    for area, x, y, bw, bh, aspect, fill, circ in blobs(masks['skin']):
        # Faces occupy the upper half and are taller than wide; arms are usually more elongated.
        face = (1-ramp(abs(math.log(aspect/0.75)), 0.25, 0.80))*ramp(fill, 0.35, 0.65)
        # Small hand patches and huge brown backgrounds should not be called human faces.
        face *= ramp(area, 0.003, 0.02)*(1-ramp(area, 0.12, 0.30))
        face *= 1-ramp(y, 0.35, 0.60)
        f['skin_face'] = max(f['skin_face'], face)
    # Dark small depressions in pale surfaces are the dimples of a golf ball.
    # Difference-of-Gaussian scales 0.7/2 isolate marks a few pixels across.
    fine = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 0.7)
    coarse = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2)
    # A three-level darkness difference is sufficient for faint shaded ball dimples.
    pits = ((coarse-fine > 3) & (gray > 100) & (masks['white'] | masks['gray'])).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(pits, 8)
    good = []
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        # Dimple marks are small and compact, unlike long cabinet edges or fur streaks.
        if 2 <= area <= 45 and 0.45 < bw/bh < 2.2 and area/(bw*bh) > 0.35:
            good.append((centers[i][0], centers[i][1]))
    # Local counts in overlapping quarter-frame regions require a collection of repeated pits.
    for cx in (w/4, w/2, 3*w/4):
        for cy in (h/4, h/2, 3*h/4):
            # Radius one quarter of the shorter side approximates a medium/large ball interior.
            radius = min(h, w)/4
            nearby = sum((x-cx)**2+(y-cy)**2 < radius*radius for x, y in good)
            # Fifteen to fifty compact pits constitute progressively stronger dimple evidence.
            f['dimples'] = max(f['dimples'], ramp(nearby, 15, 50))
    # Second-derivative ridges describe reflective tubing without any learned filter bank.
    for sigma in (1.5, 3.0):
        # Scale-normalized Hessians respond comparably to narrow and broader metal tubes.
        smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
        dxx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3)*sigma*sigma
        dyy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3)*sigma*sigma
        dxy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3)*sigma*sigma
        # Hessian eigenvalues distinguish bright elongated ridges from corners and flat paint.
        root = np.sqrt((dxx-dyy)**2+4*dxy*dxy)
        small, large = (dxx+dyy+root)/2, (dxx+dyy-root)/2
        # Negative transverse curvature and weak longitudinal curvature describe a bright tube.
        ridge = (large < -25) & (np.abs(small) < np.abs(large)*0.35)
        f['tube'] = max(f['tube'], fraction(ridge))
        # A bright ridge coinciding with yellow brass is stronger French-horn evidence.
        f['gold_tube'] = max(f['gold_tube'], fraction(ridge & masks['gold']))
    return f


def part_features(gray, masks):
    # Relational parts provide evidence beyond whole-image color and texture.
    h, w = gray.shape
    f = dict(arch_windows=0.0, bright_sphere=0.0, colored_sphere=0.0,
             pump_layout=0.0, deck_layout=0.0, canopy_layout=0.0)
    # Smooth threshold contours describe window openings rather than printed letters.
    smooth = cv2.GaussianBlur(gray, (5, 5), 1)
    for level in (65, 115, 165):
        # Church windows are dark tall openings in a lighter facade.
        binary = (smooth < level).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            # Resolve windows at least six pixels wide and not larger than half a facade.
            if bw < 6 or bh < bw*1.4 or bh > h*0.65:
                continue
            area = cv2.contourArea(c)
            # A useful opening occupies 0.2%-12% of the image, excluding tiny text strokes.
            if not h*w*0.002 < area < h*w*0.12:
                continue
            # The upper corners of an arched opening are light; its midsection is dark.
            crop = binary[y:y+bh, x:x+bw]
            corners = (fraction(crop[:bh//4, :bw//4])+fraction(crop[:bh//4, 3*bw//4:]))/2
            middle = fraction(crop[bh//3:2*bh//3, bw//4:3*bw//4])
            # Count arches only when a solid tall body is topped by an incomplete cap.
            arch = ramp(middle, 0.65, 0.95)*(1-ramp(corners, 0.15, 0.60))
            f['arch_windows'] += arch
    # A strongly saturated ball can be yellow/orange, not just white.
    for name in ('gold', 'orange', 'red', 'sky'):
        # Small closing bridges logos and dimples without merging distant colored objects.
        binary = cv2.morphologyEx(masks[name].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if not h*w*0.01 < area < h*w*0.70:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            # Colored backgrounds touch borders; spherical foreground objects generally do not.
            if x < 2 or y < 2 or x+bw > w-2 or y+bh > h-2:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            # A round filled component is stronger than a yellow curving horn tube.
            fill = area/max(math.pi*r*r, 1)
            circularity = 4*math.pi*area/max(cv2.arcLength(c, True)**2, 1)
            sphere = ramp(fill, 0.70, 0.93)*ramp(circularity, 0.60, 0.85)
            f['colored_sphere'] = max(f['colored_sphere'], sphere)
    # Analytic part maps: a pump has dark upper display over a pale or colored lower cabinet.
    # Twenty-one grid points on each axis sketch spatial proportions, not an example image.
    yy, xx = np.mgrid[0:1:21j, 0:1:21j]
    # The display lies in the upper third, within the cabinet's central width.
    display = ((xx > 0.22) & (xx < 0.78) & (yy > 0.15) & (yy < 0.40)).astype(np.float32)
    # The lower panel occupies most of the lower cabinet and often has a smooth surface.
    panel = ((xx > 0.20) & (xx < 0.80) & (yy > 0.50) & (yy < 0.90)).astype(np.float32)
    # Two long side strips support a rectangular upright silhouette.
    sides = (((xx < 0.12) | (xx > 0.88)) & (yy > 0.10) & (yy < 0.95)).astype(np.float32)
    # A horizontal cassette slot with pale surround describes a deck's front face.
    slot = ((xx > 0.10) & (xx < 0.90) & (yy > 0.35) & (yy < 0.58)).astype(np.float32)
    bands = (((yy > 0.10) & (yy < 0.28)) | ((yy > 0.65) & (yy < 0.85))).astype(np.float32)
    # Canopies form a wide colored cap suspended above narrow dark rigging/body.
    cap = (((xx-0.5)/0.45)**2+((yy-0.25)/0.18)**2 < 1).astype(np.float32)
    rig = ((np.abs(xx-0.5) < (0.45-0.45*yy)) & (yy > 0.45) & (yy < 0.90)).astype(np.float32)
    # Use grayscale contrast instead of a specific brand or learned display pattern.
    light = gray.astype(np.float32)/255
    dark = 1-light
    # Smooth edge energy is the outline signal for vertical cabinet sides.
    gx = np.abs(cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3))/255
    # Blue or neutral smooth light regions can be the clear/overcast sky beneath a canopy.
    sky = (masks['sky'] | (masks['white'] & (np.abs(gray.astype(float)-smooth) < 4))).astype(np.float32)
    # Several fixed heights cover full cabinet shots and close-ups within the working image.
    for height in (48, 72, 104, 144, 192):
        width = int(height*0.55)  # Typical pump cabinets are about half as wide as tall.
        if height > h or width > w:
            continue
        # Region convolutions are means, invariant to the size of the candidate cabinet.
        responses = []
        for source, template in ((dark, display), (light, panel), (gx, sides)):
            kernel = cv2.resize(template, (width, height), interpolation=cv2.INTER_LINEAR)
            response = cv2.matchTemplate(source, kernel/max(float(kernel.sum()), 1), cv2.TM_CCORR)
            responses.append(response)
        # A dark display needs both a lighter lower body and visible side edges.
        contrast = responses[1]+responses[0]-1
        evidence = np.minimum(np.maximum(contrast, 0), responses[2])
        f['pump_layout'] = max(f['pump_layout'], ramp(float(evidence.max()), 0.08, 0.30))
    for width in (48, 72, 104, 144, 192):
        height = max(12, int(width*0.45))  # Audio decks have broad low front panels.
        if height > h or width > w:
            continue
        # Horizontal dark slot versus brighter top/bottom bands is a signed contrast template.
        sk = cv2.resize(slot, (width, height), interpolation=cv2.INTER_LINEAR)
        bk = cv2.resize(bands, (width, height), interpolation=cv2.INTER_LINEAR)
        sd = cv2.matchTemplate(dark, sk/max(float(sk.sum()), 1), cv2.TM_CCORR)
        bl = cv2.matchTemplate(light, bk/max(float(bk.sum()), 1), cv2.TM_CCORR)
        # A 15%-45% dark slot contrast is stronger than tiny shadows on a smooth surface.
        f['deck_layout'] = max(f['deck_layout'], ramp(float((sd+bl-1).max()), 0.15, 0.45))
    return f


def assembly_features(gray, masks):
    # Parallel edge relationships express the physical blade-and-engine assembly of a chainsaw.
    h, w = gray.shape
    f = dict(saw_assembly=0.0, blade=0.0, dimple_circle=0.0)
    # A moderate edge threshold retains weak silver bars without following every wood-grain mark.
    edges = cv2.Canny(gray, 45, 110)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, threshold=20,
                           minLineLength=int(min(h, w)*0.18), maxLineGap=6)
    if lines is not None:
        # Longest 45 lines bound pair comparisons while keeping object-scale straight edges.
        segments = lines.reshape(-1, 4).astype(float)
        lengths = np.hypot(segments[:, 2]-segments[:, 0], segments[:, 3]-segments[:, 1])
        segments = segments[np.argsort(-lengths)[:45]]
        # Intensity and neutrality distinguish a metal bar from a green blade of grass.
        neutral = masks['gray'] | masks['black'] | masks['white']
        light = gray.astype(float)/255
        for i, first in enumerate(segments):
            p, q = first[:2], first[2:]
            direction = q-p
            length = float(np.linalg.norm(direction))
            axis = direction/max(length, 1)
            normal = np.array((-axis[1], axis[0]))  # A 90-degree rotation points across the blade.
            for second in segments[i+1:]:
                a, b = second[:2], second[2:]
                other = b-a
                other_length = float(np.linalg.norm(other))
                # Parallelism within about eight degrees allows perspective and toothed edges.
                if abs(float(np.dot(axis, other/max(other_length, 1)))) < 0.99:
                    continue
                # A blade's sides are separated by a few pixels, not a whole building facade.
                separation = abs(float(np.dot((a+b-p-q)/2, normal)))
                if not 3 < separation < min(h, w)*0.12:
                    continue
                # Overlap projected onto the long axis defines the common paired-edge interval.
                pa, pb = float(np.dot(a-p, axis)), float(np.dot(b-p, axis))
                start, end = max(0, min(pa, pb)), min(length, max(pa, pb))
                overlap = end-start
                # The common strip must be slender and span at least 18% of the shorter side.
                if overlap < min(h, w)*0.18 or overlap < separation*4:
                    continue
                midpoint = (p+axis*((start+end)/2)+(a+b)/2)/2
                # Twenty samples along each of three parallel strips measure a coherent bar boundary.
                t = np.linspace(-overlap*0.40, overlap*0.40, 20)
                samples = []
                for offset in (-1.3, 0, 1.3):
                    coordinates = midpoint[:, None]+axis[:, None]*t+normal[:, None]*separation*offset
                    xx = np.clip(np.rint(coordinates[0]).astype(int), 0, w-1)
                    yy = np.clip(np.rint(coordinates[1]).astype(int), 0, h-1)
                    samples.append((light[yy, xx], neutral[yy, xx]))
                # A blade must differ from its immediate surroundings on both sides.
                left = float(np.mean(np.abs(samples[1][0]-samples[0][0])))
                right = float(np.mean(np.abs(samples[1][0]-samples[2][0])))
                contrast = ramp(min(left, right), 0.07, 0.25)
                # Most of the bar surface should be achromatic metal or dark paint.
                metal = ramp(fraction(samples[1][1]), 0.50, 0.90)
                shape = ramp(overlap/separation, 4, 9)*contrast*metal
                f['blade'] = max(f['blade'], shape)
                # One endpoint should adjoin an orange motor-sized region, not just distant orange scenery.
                engine = 0.0
                for sign in (-1, 1):
                    center = midpoint+sign*axis*overlap*0.60
                    radius = max(10, separation*2.5)  # Motor housings are several bar widths across.
                    x0, x1 = max(0, int(center[0]-radius)), min(w, int(center[0]+radius))
                    y0, y1 = max(0, int(center[1]-radius)), min(h, int(center[1]+radius))
                    orange = fraction(masks['orange'][y0:y1, x0:x1])
                    engine = max(engine, ramp(orange, 0.02, 0.18))
                f['saw_assembly'] = max(f['saw_assembly'], shape*engine)
    # Low-threshold circles offer candidates for faint white-on-white golf-ball outlines.
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=12,
                               param1=65, param2=18, minRadius=8,
                               maxRadius=int(min(h, w)*0.47))
    if circles is not None:
        # Sobel radial alignment measures circular boundaries independent of brightness polarity.
        blur = cv2.GaussianBlur(gray, (0, 0), 1)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        # Dark residuals at one versus two pixels isolate small ball dimples.
        coarse = cv2.GaussianBlur(gray.astype(float), (0, 0), 2)
        fine = cv2.GaussianBlur(gray.astype(float), (0, 0), 0.7)
        pits = (coarse-fine > 3) & (gray > 100) & (masks['white'] | masks['gray'])
        # Ninety-six angular samples resolve partial circular outlines and interior texture rings.
        angles = np.arange(96)*2*math.pi/96
        cs, sn = np.cos(angles), np.sin(angles)
        for cx, cy, radius in circles.reshape(-1, 3)[:30]:
            # Mostly visible disks are needed to compare their internal and external textures.
            if cx-radius < 0 or cy-radius < 0 or cx+radius >= w or cy+radius >= h:
                continue
            # Outline samples measure gradient direction along a candidate circle.
            xx = np.clip(np.rint(cx+radius*cs).astype(int), 0, w-1)
            yy = np.clip(np.rint(cy+radius*sn).astype(int), 0, h-1)
            radial = np.abs(gx[yy, xx]*cs+gy[yy, xx]*sn)
            magnitude = np.hypot(gx[yy, xx], gy[yy, xx])
            # Boundary gradients should point radially at a substantial fraction of angles.
            support = fraction((radial > 30) & (radial > magnitude*0.80))
            # Compare repeated dark depressions inside the sphere with the surrounding surface.
            pin, pout, pale = [], [], []
            for scale in (0.40, 0.65, 0.80, 1.15):
                xx = np.clip(np.rint(cx+radius*scale*cs).astype(int), 0, w-1)
                yy = np.clip(np.rint(cy+radius*scale*sn).astype(int), 0, h-1)
                if scale < 1:
                    pin.append(fraction(pits[yy, xx]))
                    pale.append(fraction(masks['white'][yy, xx] | masks['gray'][yy, xx]))
                else:
                    pout.append(fraction(pits[yy, xx]))
            # Golf-ball interiors are mostly pale and more dimpled than a studio backdrop.
            texture = ramp(float(np.mean(pin)-np.mean(pout)), 0.02, 0.12)
            texture *= ramp(float(np.mean(pale)), 0.60, 0.90)
            f['dimple_circle'] = max(f['dimple_circle'], texture*ramp(support, 0.25, 0.55))
    return f


def architecture_features(gray, masks):
    # Architectural parts include pointed towers, crosses, and a nave's central vanishing point.
    h, w = gray.shape
    f = dict(spire=0.0, cross=0.0, perspective=0.0, smooth_top=0.0)
    # Smooth overhead regions are compatible with sky; textured ones are masonry/foliage.
    smooth = cv2.GaussianBlur(gray, (0, 0), 2)
    residual = np.abs(gray.astype(float)-smooth)
    # Six gray levels tolerate sky JPEG noise while excluding most fine stone/leaf detail.
    f['smooth_top'] = fraction(residual[:h//3] < 6)
    # Analytic silhouettes encode a narrow pointed roof and the small cross above a steeple.
    yy, xx = np.mgrid[-1:1:25j, -1:1:25j]
    # A spire widens linearly downward, with empty sky on both sides.
    spire = ((yy > -0.85) & (yy < 0.80) & (np.abs(xx) < (yy+0.90)*0.28)).astype(np.float32)
    outside = ((yy > -0.75) & (yy < 0.65) & (np.abs(xx) > (yy+1)*0.35)).astype(np.float32)
    # A cross has a long vertical stroke and shorter horizontal arms near its upper third.
    cross = (((np.abs(xx) < 0.10) & (np.abs(yy) < 0.85)) |
             ((np.abs(xx) < 0.65) & (np.abs(yy+0.25) < 0.10))).astype(np.float32)
    # The surrounding square excludes the stroke, measuring clear sky around the cross.
    cross_out = ((np.abs(xx) < 0.85) & (np.abs(yy) < 0.95) & (cross == 0)).astype(np.float32)
    light = gray.astype(np.float32)/255
    # White/gray sky is admitted as well as blue; its smoothness avoids most pale masonry.
    sky = (masks['sky'] | (((gray > 140) & (residual < 6)))).astype(np.float32)
    for height in (24, 40, 64, 96, 144):
        # Narrow tower profiles are about half as wide as high.
        width = max(12, height//2)
        if height > h or width > w:
            continue
        # Contrast between a dark pointed tower and bright sky is the steeple clue.
        sk = cv2.resize(spire, (width, height), interpolation=cv2.INTER_LINEAR)
        ok = cv2.resize(outside, (width, height), interpolation=cv2.INTER_LINEAR)
        body = cv2.matchTemplate(light, sk/max(float(sk.sum()), 1), cv2.TM_CCORR)
        surround = cv2.matchTemplate(light, ok/max(float(ok.sum()), 1), cv2.TM_CCORR)
        clear = cv2.matchTemplate(sky, ok/max(float(ok.sum()), 1), cv2.TM_CCORR)
        # Towers must begin above mid-frame; lower pointed objects are usually foreground clutter.
        limit = min(body.shape[0], max(1, h//2-height//3))
        response = (surround-body)*clear
        f['spire'] = max(f['spire'], ramp(float(response[:limit].max()), 0.15, 0.45))
    for size in (9, 13, 19, 27):
        # Small scales cover actual steeple crosses rather than whole letter-sized building facades.
        if size > min(h, w):
            continue
        ck = cv2.resize(cross, (size, size), interpolation=cv2.INTER_LINEAR)
        ok = cv2.resize(cross_out, (size, size), interpolation=cv2.INTER_LINEAR)
        body = cv2.matchTemplate(light, ck/max(float(ck.sum()), 1), cv2.TM_CCORR)
        surround = cv2.matchTemplate(light, ok/max(float(ok.sum()), 1), cv2.TM_CCORR)
        clear = cv2.matchTemplate(sky, ok/max(float(ok.sum()), 1), cv2.TM_CCORR)
        # The top half restriction avoids display text and standing figures near the ground.
        response = (surround-body)*clear
        f['cross'] = max(f['cross'], ramp(float(response[:max(1, h//2)].max()), 0.15, 0.40))
    # Perspective rays from all quadrants distinguish a cathedral aisle from a flat machine panel.
    edges = cv2.Canny(gray, 60, 140)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, threshold=22,
                           minLineLength=min(h, w)//5, maxLineGap=5)
    if lines is not None:
        lines = lines.reshape(-1, 4).astype(float)
        # A nine-point central grid tolerates off-center architectural vanishing points.
        for fx in (0.35, 0.50, 0.65):
            for fy in (0.35, 0.50, 0.65):
                center = np.array((fx*w, fy*h))
                support = np.zeros(4)  # Four quadrants must supply converging structural lines.
                for x1, y1, x2, y2 in lines:
                    a, b = np.array((x1, y1)), np.array((x2, y2))
                    vector, ray = b-a, (a+b)/2-center
                    length, distance = float(np.linalg.norm(vector)), float(np.linalg.norm(ray))
                    # Skip central tiny rays and horizontal/vertical edges: neither establishes perspective depth.
                    if distance < min(h, w)*0.15 or min(abs(vector[0]), abs(vector[1])) < length*0.15:
                        continue
                    # A line pointing toward the center should agree with its radial direction within about 15 degrees.
                    alignment = abs(float(np.dot(vector, ray)))/max(length*distance, 1)
                    value = ramp(alignment, 0.94, 0.99)*length/math.hypot(h, w)
                    # Boolean quadrant bits encode left/right and above/below relative to the vanishing point.
                    quadrant = int(ray[0] > 0)+2*int(ray[1] > 0)
                    support[quadrant] += value
                # Require balanced rays, rather than one accidentally aligned branch or bar.
                f['perspective'] = max(f['perspective'], ramp(float(np.min(support)), 0.15, 0.80))
    return f


def scores(path):
    # Feature extraction uses no filenames or class labels, only this image's pixels.
    f, gray, masks = features(path)
    # Add current-image circle, region-template, and bar evidence.
    f.update(geometry(gray, masks))
    # Add surface texture, road layout, tire silhouettes, and pitched-roof evidence.
    f.update(scene_features(gray, masks))
    # Add rotation-tolerant panels, face context, dimples, and analytic tube ridges.
    f.update(detail_features(gray, masks))
    # Add arched openings, colored spheres, and cabinet/slot part arrangements.
    f.update(part_features(gray, masks))
    # Add blade/engine relationships and texture inside faint circular outlines.
    f.update(assembly_features(gray, masks))
    # Add pointed steeples, small sky-backed crosses, and balanced nave perspective.
    f.update(architecture_features(gray, masks))
    s = dict.fromkeys(CLASSES, 0.0)
    # Evidence units are manually balanced: 1 weak context, 2-3 moderate, 4-6 specific.
    sky, green, white = f['sky'], f['green'], f['white']
    gold, brown, olive = f['gold'], f['brown'], f['olive']
    # Smooth rounded natural outlines have fewer axis-aligned edges than machine cabinets.
    natural = (1-ramp(f['axis'], 0.68, 0.78))*(1-ramp(f['oriented'], 0.55, 0.75))
    # Rigidity is the complementary clue; church roofs remain less rigid than pump cabinets.
    rigid = max(ramp(f['axis'], 0.68, 0.80), ramp(f['oriented'], 0.55, 0.75))
    # Sky and cloud smoothness should dominate only when the whole scene looks aerial.
    s['parachute'] = 7*ramp(sky, 0.15, 0.65)+4*ramp(f['sky_smooth'], 0.10, 0.50)
    # Overcast aerial scenes still have large featureless areas and few vertical structures.
    s['parachute'] += 4*ramp(f['smooth'], 0.55, 0.85)-3*ramp(f['line_v'], 1, 4)
    # Pale disks plus low-density outlines are stronger than pale disks in busy machinery.
    s['golf_ball'] = 6*f['ball_shape']+3*(1-ramp(f['edge'], 0.10, 0.25))
    # Open grass supports balls; many straight lines or warm skin favor other objects.
    s['golf_ball'] += 3*ramp(green, 0.20, 0.70)-2*rigid-2*ramp(f['skin'], 0.04, 0.20)
    # Brass and repeated circular arcs must be accompanied by a nonvegetated setting.
    s['french_horn'] = 3*ramp(gold, 0.02, 0.15)+3*ramp(f['horn_coils'], 0.50, 5)
    # Thin colored regions support tubing; solid yellow backgrounds are less persuasive.
    s['french_horn'] += 2*natural+ramp(1-f['gold_solid'], 0.30, 0.75)
    # Leafy and blue outdoor settings are weak evidence for an indoor instrument.
    s['french_horn'] -= 4*ramp(green, 0.08, 0.35)+3*ramp(sky, 0.10, 0.45)
    # A white muzzle/blaze with dark ears, appreciable white fur, and curved outlines supports spaniels.
    s['english_springer'] = 5*f['dog_face']+2*ramp(white, 0.05, 0.25)+3*natural
    # Brown fur helps, but extensive architectural lines argue against a close dog portrait.
    s['english_springer'] += ramp(brown, 0.05, 0.25)-2*ramp(f['line_v'], 2, 5)
    # A fish interpretation needs vegetation plus a long olive body and nonrigid outlines.
    s['tench'] = 3*f['fish_body']+3*ramp(green, 0.05, 0.35)+3*natural
    # Skin from hands/face helps; a dog blaze or extensive bright fur opposes this reading.
    s['tench'] += 2*ramp(f['skin'], 0.02, 0.12)-2*f['dog_face']-ramp(f['white_center'], 0.20, 0.50)
    # Orange saw motors may occupy only a small fraction of a woodcutting photograph.
    s['chain_saw'] = 6*ramp(f['orange'], 0.002, 0.03)+2*f['saw_bar']+2*natural
    # Busy wood/fabric scenes support saws, with sky and pale round silhouettes adverse.
    s['chain_saw'] += 2*ramp(f['texture'], 0.35, 0.60)-2*ramp(sky, 0.15, 0.60)-f['ball_shape']
    # Low-chroma axis-aligned boxes with long horizontal controls are typical cassette players.
    s['cassette_player'] = 3*ramp(f['gray']+f['black'], 0.30, 0.80)+3*rigid
    # Decks are wider than tall and can be shown on a plain white product backdrop.
    s['cassette_player'] += 2*ramp(f['line_h']/(f['line_v']+1), 1, 4)+ramp(f['white_border'], 0.65, 0.95)
    # Outdoor grass/sky and many tall vertical lines weaken an indoor audio interpretation.
    s['cassette_player'] -= 4*ramp(sky+green, 0.10, 0.40)+ramp(f['line_v'], 2, 5)
    # Tall building edges, dense masonry, and a pitched roof describe churches.
    s['church'] = 3*ramp(f['line_v'], 1, 5)+2*ramp(f['sky_top'], 0.10, 0.60)+3*f['roof']
    # Dense upper structure is common in architecture; compact smooth product shots lack it.
    s['church'] += 2*ramp(f['edge'], 0.20, 0.40)+ramp(f['gray'], 0.15, 0.50)
    # Extreme boxlike alignment and red advertising more often describe fuel cabinets.
    s['church'] -= ramp(f['axis'], 0.78, 0.88)+ramp(f['red'], 0.03, 0.15)
    # A road, broad horizontal body, and tires identify refuse trucks.
    s['garbage_truck'] = 2*f['tires']+2*f['tire_pair']+3*ramp(f['line_h'], 2, 7)+2*rigid
    # Outdoor sky and neutral lower road provide context; indoor audio components lack both.
    s['garbage_truck'] += 2*ramp(f['sky_top'], 0.05, 0.40)+ramp(f['road'], 0.40, 0.85)
    # More horizontal than vertical structures support truck bodies over upright fuel dispensers.
    s['garbage_truck'] += ramp(f['line_h']/(f['line_v']+1), 1, 4)-ramp(f['line_v'], 2, 5)
    # Pumps have the strongest upright rectangular structure of these classes.
    s['gas_pump'] = 3*ramp(f['line_v'], 1, 5)+3*ramp(f['axis'], 0.72, 0.85)
    # Red branding, portrait framing, and long cabinet edges jointly support pumps.
    s['gas_pump'] += 2*ramp(f['red'], 0.005, 0.08)+ramp(1/f['aspect'], 0.90, 1.50)+ramp(f['long_vertical'], 0.30, 0.70)
    # White-and-dark accidents in rectangular machinery are less credible than curved dog faces.
    s['english_springer'] -= 3*rigid+ramp(f['rectangles'], 2, 8)
    # A small warm oval high in the frame supports an angler displaying the fish below.
    s['tench'] += 2*f['skin_face']
    # Actual yellow highlight ridges support brass, and high skin saturation alone does not.
    s['french_horn'] += 3*ramp(f['gold_tube'], 0.003, 0.025)*(1-ramp(f['gold_solid'], 0.40, 0.75))
    # Thin orange ridges in brass are less like the saw's solid plastic engine housing.
    s['chain_saw'] -= ramp(f['gold_tube'], 0.008, 0.030)*(1-ramp(f['gold_solid'], 0.40, 0.75))
    # Repeated pits become diagnostic only with pale circular shape or low overall edge density.
    s['golf_ball'] += 3*f['dimples']*max(f['ball_shape'], 1-ramp(f['edge'], 0.10, 0.25))
    # A wide isolated cabinet panel is strong audio evidence when there is little outdoor color.
    s['cassette_player'] += 2*f['rect_wide']*(1-ramp(sky+green, 0.05, 0.20))
    # Several crisp upright panels support fuel-dispenser displays rather than rough masonry.
    s['gas_pump'] += f['rect_tall']+ramp(f['rectangles'], 2, 8)
    # Vegetated buildings with a gabled roof are more plausible churches than orange tools.
    s['church'] += 2*f['roof']*ramp(f['line_v'], 0.5, 3)
    # Architectural arches and resolved roof/tower structure deserve additional building evidence.
    s['church'] += 2*ramp(f['arch_windows'], 1, 5)+1.5*ramp(f['line_v'], 0.5, 2)
    # Natural fine woodcutting texture supports otherwise neutral-colored chainsaws.
    s['chain_saw'] += 1.5*natural*ramp(f['texture'], 0.35, 0.60)
    # Slotted low panels and upright display-over-body structure support audio and pump classes.
    s['cassette_player'] += 1.5*f['deck_layout']*rigid
    # The pump layout is useful in conjunction with an upright cabinet, not arbitrary dark patches.
    s['gas_pump'] += f['pump_layout']*ramp(f['line_v'], 1, 3)
    # Golf balls may be colored; clean circular shapes and sparse outlines deserve extra evidence.
    s['golf_ball'] += 6*f['colored_sphere']+1.5*(1-ramp(f['edge'], 0.10, 0.20))
    # Give smooth blue aerial contexts a small margin over weak object-template coincidences.
    s['parachute'] += 1.5*ramp(sky, 0.10, 0.40)*(1-ramp(f['line_v'], 0.5, 2))
    # A pale muzzle and substantial pale coat support a dog when brass-colored regions are scarce.
    s['english_springer'] += ramp(white, 0.06, 0.20)*(1-ramp(gold, 0.02, 0.12))*natural
    # A neutral slender bar next to an orange motor is much more specific than orange hue alone.
    s['chain_saw'] += 4*f['saw_assembly']+f['blade']
    # Repeated pale-surface pits constrained by a circular boundary identify white-on-white balls.
    s['golf_ball'] += 6*f['dimple_circle']
    # A pointed tower or sky-backed cross supports church exteriors; nave perspective supports interiors.
    s['church'] += 2*f['spire']+f['cross']+2*f['perspective']
    # Overcast aerial photographs can have gray smooth sky rather than blue sky.
    s['parachute'] += 2*ramp(f['smooth_top'], 0.85, 0.98)*(1-ramp(f['edge_top'], 0.03, 0.15))
    return s, f


def predict(path: str) -> str:
    # Select the class supported by the strongest sum of hand-written visual clues.
    evidence, _ = scores(path)
    return max(evidence, key=evidence.get)
